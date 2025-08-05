import numpy as np
import math
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle


from tudatpy.numerical_simulation import environment_setup



###############################################################################
# Sensors and Measurements
###############################################################################

def define_radar_sensor(latitude_rad, longitude_rad, height_m):
    '''
    This function will generate the sensor parameters dictionary for a radar
    sensor provided the location in latitude, longitude, height.
    
    It is pre-filled with constraint and noise parameters per assignment
    description.

    Parameters
    ----------
    latitude_rad : float
        geodetic latitude of sensor [rad]
    longitude_rad : float
        geodetic longitude of sensor [rad]
    height_m : float
        geodetic height of sensor [m]

    Returns
    -------
    sensor_params : dictionary
        location, constraint, noise parameters of sensor

    '''
            
    # Compute sensor location in ECEF/ITRF
    sensor_ecef = latlonht2ecef(latitude_rad, longitude_rad, height_m)
        
    # FOV dimensions
    LAM_deg = 10.   # deg
    PHI_deg = 10.   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_deg*np.pi/180
    PHI_half = 0.5*PHI_deg*np.pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
    
    # Constraints/Limits
    az_lim = [0., 2.*np.pi]  # rad
    el_lim = [5.*np.pi/180., np.pi/2.]  # rad
    rg_lim = [0., 5000.*1000.]   # m
    sun_el_mask = -np.pi  # rad
    
    # Measurement types and noise
    meas_types = ['rg', 'ra', 'dec']
    sigma_dict = {}
    sigma_dict['rg'] = 10.               # m
    sigma_dict['ra'] = 0.1*np.pi/180.    # rad
    sigma_dict['dec'] = 0.1*np.pi/180.   # rad
        
    # Location and constraints
    sensor_params = {}
    sensor_params['sensor_ecef'] = sensor_ecef
    sensor_params['el_lim'] = el_lim
    sensor_params['az_lim'] = az_lim
    sensor_params['rg_lim'] = rg_lim
    sensor_params['FOV_hlim'] = FOV_hlim
    sensor_params['FOV_vlim'] = FOV_vlim
    sensor_params['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_params['meas_types'] = meas_types
    sensor_params['sigma_dict'] = sigma_dict

    return sensor_params


def define_optical_sensor(latitude_rad, longitude_rad, height_m):
    '''
    This function will generate the sensor parameters dictionary for an optical
    sensor provided the location in latitude, longitude, height.
    
    It is pre-filled with constraint and noise parameters per assignment
    description.

    Parameters
    ----------
    latitude_rad : float
        geodetic latitude of sensor [rad]
    longitude_rad : float
        geodetic longitude of sensor [rad]
    height_m : float
        geodetic height of sensor [m]

    Returns
    -------
    sensor_params : dictionary
        location, constraint, noise parameters of sensor

    '''
    
    arcsec2rad = (1./3600.)*np.pi/180.
            
    # Compute sensor location in ECEF/ITRF
    sensor_ecef = latlonht2ecef(latitude_rad, longitude_rad, height_m)
        
    # FOV dimensions
    LAM_deg = 4.   # deg
    PHI_deg = 4.   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_deg*np.pi/180
    PHI_half = 0.5*PHI_deg*np.pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
    
    # Constraints/Limits
    az_lim = [0., 2.*np.pi]  # rad
    el_lim = [15.*np.pi/180., np.pi/2.]  # rad
    rg_lim = [0., np.inf]   # m
    sun_el_mask = -12.*np.pi/180.  # rad (Nautical twilight)
    
    # Measurement types and noise
    # meas_types = ['ra', 'dec']
    # sigma_dict = {}
    # sigma_dict['ra'] = arcsec2rad    # rad
    # sigma_dict['dec'] = arcsec2rad   # rad
    
    meas_types = ['mag']
    sigma_dict = {}
    sigma_dict['mag'] = 0.01
    
        
    # Location and constraints
    sensor_params = {}
    sensor_params['sensor_ecef'] = sensor_ecef
    sensor_params['el_lim'] = el_lim
    sensor_params['az_lim'] = az_lim
    sensor_params['rg_lim'] = rg_lim
    sensor_params['FOV_hlim'] = FOV_hlim
    sensor_params['FOV_vlim'] = FOV_vlim
    sensor_params['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_params['meas_types'] = meas_types
    sensor_params['sigma_dict'] = sigma_dict
    

    
    return sensor_params


def compute_measurement(tk, X, sensor_params, bodies=None):
    '''
    This function be used to compute a measurement given an input state vector
    and time.
    
    Parameters
    ------
    tk : float
        time in seconds since J2000
    X : nx1 numpy array
        Cartesian state vector [m, m/s]
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
        
    Returns
    ------
    Y : px1 numpy array
        computed measurements for given state and sensor
    
    '''
    
    if bodies is None:
        body_settings = environment_setup.get_default_body_settings(
            ["Earth"],
            "Earth",
            "J2000")
        bodies = environment_setup.create_system_of_bodies(body_settings)
        
    # Rotation matrices
    earth_rotation_model = bodies.get("Earth").rotation_model
    eci2ecef = earth_rotation_model.inertial_to_body_fixed_rotation(tk)
    ecef2eci = earth_rotation_model.body_fixed_to_inertial_rotation(tk)
        
    # Retrieve measurement types
    meas_types = sensor_params['meas_types']
    
    # Compute station location in ECI    
    sensor_ecef = sensor_params['sensor_ecef']
    sensor_eci = np.dot(ecef2eci, sensor_ecef)    
    
    # Object location in ECI
    r_eci = X[0:3].reshape(3,1)
    
    # Compute range and line of sight vector
    rg = np.linalg.norm(r_eci - sensor_eci)
    rho_hat_eci = (r_eci - sensor_eci)/rg
    
    # Rotate to ENU frame
    rho_hat_ecef = np.dot(eci2ecef, rho_hat_eci)
    rho_hat_enu = ecef2enu(rho_hat_ecef, sensor_ecef)
    
    # Loop over measurement types
    Y = np.zeros((len(meas_types),1))
    ii = 0
    for mtype in meas_types:
        
        if mtype == 'rg':
            Y[ii] = rg  # m
            
        elif mtype == 'ra':
            Y[ii] = math.atan2(rho_hat_eci[1], rho_hat_eci[0]) # rad
            
        elif mtype == 'dec':
            Y[ii] = math.asin(rho_hat_eci[2])  # rad
    
        elif mtype == 'az':
            Y[ii] = math.atan2(rho_hat_enu[0], rho_hat_enu[1])  # rad  
            # if Y[ii] < 0.:
            #     Y[ii] += 2.*np.pi
            
        elif mtype == 'el':
            Y[ii] = math.asin(rho_hat_enu[2])  # rad
            
            
        ii += 1
            
            
    return Y



###############################################################################
# Visibility 
###############################################################################





###############################################################################
# Coordinate Frames
###############################################################################


def ecef2enu(r_ecef, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ECEF to ENU frame.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [m]
    r_site : 3x1 numpy array
      station position vector in ECEF [m]

    Returns
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [m]
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),  math.sin(lon1), 0.],
                   [-math.sin(lon1), math.cos(lon1), 0.],
                   [0.,              0.,             1.]])

    R = np.dot(R1, R3)

    r_enu = np.dot(R, r_ecef)

    return r_enu


def enu2ecef(r_enu, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ENU to ECEF frame.

    Parameters
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [m]
    r_site : 3x1 numpy array
      station position vector in ECEF [m]

    Returns
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),   math.sin(lon1), 0.],
                   [-math.sin(lon1),  math.cos(lon1), 0.],
                   [0.,                           0., 1.]])

    R = np.dot(R1, R3)

    R2 = R.T

    r_ecef = np.dot(R2, r_enu)

    return r_ecef


def ecef2latlonht(r_ecef):
    '''
    This function converts the coordinates of a position vector from
    the ECEF frame to geodetic latitude, longitude, and height.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [m]

    Returns
    ------
    lat : float
      latitude [rad] [-pi/2,pi/2]
    lon : float
      longitude [rad] [-pi,pi]
    ht : float
      height [m]
    '''

    # WGS84 Data (Pratap and Misra P. 103)
    a = 6378137.0   # m
    rec_f = 298.257223563

    # Get components from position vector
    x = float(r_ecef[0])
    y = float(r_ecef[1])
    z = float(r_ecef[2])

    # Compute longitude
    f = 1./rec_f
    e = np.sqrt(2.*f - f**2.)
    lon = math.atan2(y, x)

    # Iterate to find height and latitude
    p = np.sqrt(x**2. + y**2.)  # m
    lat = 0.
    lat_diff = 1.
    tol = 1e-12

    while abs(lat_diff) > tol:
        lat0 = float(lat)  # rad
        N = a/np.sqrt(1 - e**2*(math.sin(lat0)**2))  # km
        ht = p/math.cos(lat0) - N
        lat = math.atan((z/p)/(1 - e**2*(N/(N + ht))))
        lat_diff = lat - lat0


    return lat, lon, ht


def latlonht2ecef(lat, lon, ht):
    '''
    This function converts geodetic latitude, longitude and height
    to a position vector in ECEF.

    Parameters
    ------
    lat : float
      geodetic latitude [rad]
    lon : float
      geodetic longitude [rad]
    ht : float
      geodetic height [m]

    Returns
    ------
    r_ecef = 3x1 numpy array
      position vector in ECEF [m]
    '''
    
    # WGS84 Data (Pratap and Misra P. 103)
    Re = 6378137.0   # m
    rec_f = 298.257223563

    # Compute flattening and eccentricity
    f = 1/rec_f
    e = np.sqrt(2*f - f**2)

    # Compute ecliptic plane and out of plane components
    C = Re/np.sqrt(1 - e**2*math.sin(lat)**2)
    S = Re*(1 - e**2)/np.sqrt(1 - e**2*math.sin(lat)**2)

    rd = (C + ht)*math.cos(lat)
    rk = (S + ht)*math.sin(lat)

    # Compute ECEF position vector
    r_ecef = np.array([[rd*math.cos(lon)], [rd*math.sin(lon)], [rk]])

    return r_ecef

