import numpy as np
import math
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle
import copy
import bisect


from tudatpy.numerical_simulation import environment_setup

# Import utility functions
import TudatPropagator as prop
import EstimationUtilities as est


###############################################################################
# Reward Functions
###############################################################################

def compute_gaussian_renyi_infogain(P0, P1, tau=1.):
    '''
    This function computes the analytic Renyi divergence for a single Gaussian 
    component. It assumes no noise, therefore m0 = m1.

    Parameters
    ------
    P0 : nxn numpy array
        prior covariance matrix
    P1 : nxn numpy array
        posterior covariance matrix
    tau : float (optional)
        tactical importance function value (priority) (default = 1.0)

    Returns
    ------
    R : float
        value of Renyi divergence
    '''
    
    #TODO: Verify this formula!!!

    # Compute integral
    P3 = np.linalg.inv(np.linalg.inv(P0) + np.linalg.inv(P1))
    integral = (np.linalg.det(4.*P3)/np.linalg.det(P0+P1))**0.25

    # Compute information gain
    R = tau*(2. - 2.*integral)

    return R


def reward_renyi_infogain(P0, P1, params):
    
    # Set uniform priority
    tau = 1.
    
    reward = compute_gaussian_renyi_infogain(P0, P1, tau)    
    
    return reward


###############################################################################
# Greedy Sensor Tasking
###############################################################################

def greedy_sensor_tasking(rso_file, sensor_file, visibility_file, truth_file, 
                          reward_fcn):
    
    # Load rso data
    pklFile = open(rso_file, 'rb')
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    # Load sensor and visibility data
    pklFile = open(sensor_file, 'rb')
    data = pickle.load( pklFile )
    sensor_dict = data[0]
    pklFile.close()
    
    pklFile = open(visibility_file, 'rb')
    data = pickle.load( pklFile )
    visibility_dict = data[0]
    pklFile.close()    
    
    pklFile = open(truth_file, 'rb')
    data = pickle.load( pklFile )
    truth_dict = data[0]
    pklFile.close()
    
    
    # Basic setup for propagation
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)      
    
    int_params = {}
    int_params['tudat_integrator'] = 'dp87'
    int_params['step'] = 10.
    int_params['max_step'] = 1000.
    int_params['min_step'] = 1e-3
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12  
    
    state_params = {}    
    state_params['sph_deg'] = 20
    state_params['sph_ord'] = 20   
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create    
    
    # Parse visibility dict to generate time based visibility dict
    time_based_visibility = {}
    for sensor_id in visibility_dict:
        for obj_id in visibility_dict[sensor_id]:
            tk_list = visibility_dict[sensor_id][obj_id]['tk_list']
            
            for tk in tk_list:
                if tk not in time_based_visibility:
                    time_based_visibility[tk] = {}                    
                    
                if sensor_id not in time_based_visibility[tk]:
                    time_based_visibility[tk][sensor_id] = []
                
                # Append object IDs visible to this sensor
                time_based_visibility[tk][sensor_id].append(obj_id)
                
                
    # Unscented Transform parameters
    n = 6
    alpha = 1e-4
    
    # Prior information about the distribution
    beta = 2.
    kappa = 3. - float(n)
    
    # Compute sigma point weights    
    lam = alpha**2.*(n + kappa) - n
    gam = np.sqrt(n + lam)
    Wm = 1./(2.*(n + lam)) * np.ones(2*n,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(n + lam))
    Wc = np.insert(Wc, 0, lam/(n + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)
    
    # Initialize output
    meas_dict = {}
    
    # Loop over times
    tk_list = sorted(list(time_based_visibility.keys()))
    for tk in tk_list:
        
        # Loop over sensors
        for sensor_id in time_based_visibility[tk]:
            
            sensor_params = sensor_dict[sensor_id]
            meas_types = sensor_params['meas_types']
            sigma_dict = sensor_params['sigma_dict']
        
            # Loop over objects visible to this sensor
            obj_id_list = time_based_visibility[tk][sensor_id]
            reward_list = []
            Pk_list = []
            for obj_id in obj_id_list:
                
                # Propagate state and covar
                t0 = rso_dict[obj_id]['epoch_tdb']
                Xo = rso_dict[obj_id]['state']
                Po = rso_dict[obj_id]['covar']
                state_params['mass'] = rso_dict[obj_id]['mass']
                state_params['area'] = rso_dict[obj_id]['area']
                state_params['Cd'] = rso_dict[obj_id]['Cd']
                state_params['Cr'] = rso_dict[obj_id]['Cr']                
                
                tvec = np.array([t0, tk])
                tbar, Xbar, Pbar = prop.propagate_state_and_covar(Xo, Po, tvec, state_params, int_params, bodies=bodies, alpha=alpha)
                
                # Update RSO dict with predicted state and covar
                rso_dict[obj_id]['epoch_tdb'] = tk
                rso_dict[obj_id]['state'] = Xbar
                rso_dict[obj_id]['covar'] = Pbar               
                
                # Compute updated covar
                sqP = np.linalg.cholesky(Pbar)
                Xrep = np.tile(Xbar, (1, n))
                chi_bar = np.concatenate((Xbar, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
                chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
                
                # Computed measurements and covariance
                gamma_til_k, Rk = est.unscented_meas(tk, chi_bar, sensor_params, bodies)
                ybar = np.dot(gamma_til_k, Wm.T)
                ybar = np.reshape(ybar, (len(ybar), 1))
                Y_diff = gamma_til_k - np.dot(ybar, np.ones((1, (2*n+1))))
                Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk
                Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))
                
                # Kalman gain and measurement update
                Kk = np.dot(Pxy, np.linalg.inv(Pyy))
                
                # Joseph form of covariance update
                cholPbar = np.linalg.inv(np.linalg.cholesky(Pbar))
                invPbar = np.dot(cholPbar.T, cholPbar)
                P1 = (np.eye(n) - np.dot(np.dot(Kk, np.dot(Pyy, Kk.T)), invPbar))
                P2 = np.dot(Kk, np.dot(Rk, Kk.T))
                Pk = np.dot(P1, np.dot(Pbar, P1.T)) + P2                
                
                # Compute reward and store with posterior covar
                params = {}
                reward = reward_fcn(Pbar, Pk, params)
                
                reward_list.append(reward)
                Pk_list.append(Pk)
                
            # Find max ind of reward
            max_ind = reward.index(max(reward))
            
            # Update RSO dict with updated covar of this object
            max_obj_id = obj_id_list[max_ind]
            max_Pk = Pk_list[max_ind]
            rso_dict[max_obj_id]['covar'] = max_Pk
            
            # Retrieve truth data and simulate measurement for this time
            tk_truth = truth_dict[max_obj_id]['t_truth']
            Xk_truth = truth_dict[max_obj_id]['X_truth']
            truth_ind = list(tk_truth).index(tk)
            Xk_t = Xk_truth[truth_ind,:].reshape(6,1)
            rso_dict[max_obj_id]['state'] = Xk_t
            
            # Compute measurement and add noise
            Yk = compute_measurement(tk, Xk_t, sensor_params, bodies)
            
            # Add noise
            for ii in range(len(meas_types)):
                meas = meas_types[ii]
                Yk[ii] += np.random.randn()*sigma_dict[meas]
            
            # Store in correct time order
            if len(meas_dict[obj_id]['tk_list']) == 0:
                meas_dict[obj_id]['tk_list'].append(tk)
                meas_dict[obj_id]['Yk_list'].append(Yk)
                meas_dict[obj_id]['sensor_id_list'].append(sensor_id)
                
            else:
                ind = bisect.bisect_right(meas_dict[obj_id]['tk_list'], tk)
                meas_dict[obj_id]['tk_list'].insert(ind, tk)
                meas_dict[obj_id]['Yk_list'].insert(ind, Yk)
                meas_dict[obj_id]['sensor_id_list'].insert(ind, sensor_id)

    
    return meas_dict







###############################################################################
# Sensors and Measurements
###############################################################################

def define_radar_sensor(latitude_rad, longitude_rad, height_m, beamwidth_rad,
                        az_lim, el_lim, rg_lim, sun_el_mask, meas_types,
                        sigma_dict):
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
        
    # Location and constraints
    sensor_params = {}
    sensor_params['sensor_type'] = 'radar'
    sensor_params['sensor_ecef'] = sensor_ecef
    sensor_params['el_lim'] = el_lim
    sensor_params['az_lim'] = az_lim
    sensor_params['rg_lim'] = rg_lim
    sensor_params['beamwidth'] = beamwidth_rad
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
    sensor_params['sensor_type'] = 'optical'
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
            Y[ii] = rg      # m
            
        elif mtype == 'ra':
            Y[ii] = math.atan2(rho_hat_eci[1], rho_hat_eci[0]) # rad
            
        elif mtype == 'dec':
            Y[ii] = math.asin(rho_hat_eci[2])  # rad
    
        elif mtype == 'az':
            Y[ii] = math.atan2(rho_hat_enu[0], rho_hat_enu[1])  # rad  
            if Y[ii] < 0.:
                Y[ii] += 2.*np.pi
            
        elif mtype == 'el':
            Y[ii] = math.asin(rho_hat_enu[2])  # rad
            
            
        ii += 1
            
            
    return Y



###############################################################################
# Visibility 
###############################################################################

def check_visibility(tk, Xk, sensor_params, bodies=None):
    
    # Initialize output
    vis_flag = True
    
    # Geometric constraints
    # Compute range, az, el
    sensor_params_vis = {}
    sensor_params_vis['meas_types'] = ['rg', 'az', 'el']
    sensor_params_vis['sensor_ecef'] = sensor_params['sensor_ecef']
    
    meas = compute_measurement(tk, Xk, sensor_params_vis, bodies)
    rg = meas[0,0]
    az = meas[1,0]
    el = meas[2,0]
    
    # Check constraints
    if az < sensor_params['az_lim'][0] or az > sensor_params['az_lim'][1]:
        vis_flag = False
        
    if el < sensor_params['el_lim'][0] or el > sensor_params['el_lim'][1]:
        vis_flag = False
        
    if rg < sensor_params['rg_lim'][0] or rg > sensor_params['rg_lim'][1]:
        vis_flag = False
    
    # TODO: Lighting constraints for optical sensors
    
    
    return vis_flag, rg, az, el


def compute_visible_passes(tvec, rso_dict, sensor_dict, int_params, bodies=None):
    '''
    This function computes visible passes for a given object catalog and 
    sensors.
    
    Parameters
    ------
    tvec : 1D numpy array
        initial and final time of visibility window seconds since J2000 TDB
    rso_dict : dictionary
        object state parameters including pos/vel in ECI [m] and physical 
        attributes
    sensor_dict : dictionary
        sensor parameters including location in ECEF [m] and constraints
        
    Returns
    ------    
    
    
    '''
    
    # Retrieve input data
    obj_id_list = sorted(list(rso_dict.keys()))
    sensor_id_list = sorted(list(sensor_dict.keys()))
    
    # Make copy of rso_dict to update as objects are propagated
    rso_dict2 = copy.deepcopy(rso_dict)
    
    # Standard orbit propagation parameters
    state_params = {}
    state_params['sph_deg'] = 20
    state_params['sph_ord'] = 20   
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = ['Sun', 'Earth', 'Moon']
    
    
    # Propagate all objects to same starting epoch for visibility check
    # Note: Do not use this code, it is not needed or verified
    # keeping it for reference if needed later
    # t0_all = tvec[0]
    # for obj_id in obj_id_list:
        
    #     # Check if propagation is needed
    #     t0_obj = rso_dict[obj_id]['epoch_tdb']
    #     if abs(t0_obj - t0_all) < 1e-12:
    #         continue
        
    #     # Setup forward or backpropagation interval
    #     tprop = np.array([t0_obj, t0_all])
    #     if t0_obj < t0_all:
    #         int_params['step'] = 4.  
    #         backprop = False
    #     else:
    #         int_params['step'] = -4.  
    #         backprop = True
        
    #     # Setup initial state and object parameters
    #     Xo_obj = rso_dict[obj_id]['state']
    #     state_params['mass'] = rso_dict[obj_id]['mass']
    #     state_params['area'] = rso_dict[obj_id]['area']
    #     state_params['Cd'] = rso_dict[obj_id]['Cd']
    #     state_params['Cr'] = rso_dict[obj_id]['Cr']
        
    #     # Propagate orbit
    #     tout, Xout = prop.propagate_orbit(Xo_obj, tprop, state_params, int_params, bodies)
    
    #     # Retrieve and store updated state
    #     if backprop:
    #         Xf = Xout[0,:].reshape(6,1)
    #     else:
    #         Xf = Xout[-1,:].reshape(6,1)
            
    #     rso_dict[obj_id]['epoch_tdb'] = t0_all
    #     rso_dict[obj_id]['state'] = Xf
    
    
    # Loop over objects
    t0_all = float(tvec[0])
    tf_all = float(tvec[-1])
    visibility_dict = {}
    for obj_id in obj_id_list:
        
        print('')
        print('obj_id', obj_id)
                
        # Retrieve initial state and epoch
        t0_obj = rso_dict[obj_id]['epoch_tdb']
        Xo = rso_dict[obj_id]['state']
        state_params['mass'] = rso_dict[obj_id]['mass']
        state_params['area'] = rso_dict[obj_id]['area']
        state_params['Cd'] = rso_dict[obj_id]['Cd']
        state_params['Cr'] = rso_dict[obj_id]['Cr']
        
        # Conduct backpropagation if needed
        if t0_obj > t0_all:
            backprop_params = {}
            backprop_params['tudat_integrator'] = 'dp7'
            backprop_params['step'] = -4.
            
            tprop = np.array([t0_obj, t0_all])
            tout, Xout = prop.propagate_orbit(Xo, tprop, state_params, backprop_params, bodies)
            
            # Set initial state and propagation time            
            Xo = Xout[0,:].reshape(6,1)
            
        # Propagate orbit
        tvec = np.array([t0_obj, tf_all])
        tout, Xout = prop.propagate_orbit(Xo, tvec, state_params, int_params, bodies)
        
        # Loop over times and check visibility
        for kk in range(len(tout)):
            
            # Retrieve current time and state
            tk = tout[kk]
            Xk = Xout[kk,:].reshape(6,1)
            
            # Check if in desired visibility window
            if tk < t0_all or tk > tf_all:
                continue            
            
            # Loop over sensors
            for sensor_id in sensor_id_list:
                
                # Retrieve sensor parameters
                sensor_params = sensor_dict[sensor_id]
                
                # Check visibility
                vis_flag, rg, az, el = check_visibility(tk, Xk, sensor_params,
                                                        bodies)
                
                # Store output
                if vis_flag:
                    
                    if sensor_id not in visibility_dict:
                        visibility_dict[sensor_id] = {}
                        
                    if obj_id not in visibility_dict[sensor_id]:
                        visibility_dict[sensor_id][obj_id] = {}
                        visibility_dict[sensor_id][obj_id]['tk_list'] = []
                        visibility_dict[sensor_id][obj_id]['rg_list'] = []
                        visibility_dict[sensor_id][obj_id]['el_list'] = []
                        visibility_dict[sensor_id][obj_id]['az_list'] = []
                        
                    visibility_dict[sensor_id][obj_id]['tk_list'].append(tk)
                    visibility_dict[sensor_id][obj_id]['rg_list'].append(rg)
                    visibility_dict[sensor_id][obj_id]['az_list'].append(az)
                    visibility_dict[sensor_id][obj_id]['el_list'].append(el)
                    
    
    return visibility_dict


def compute_visible_passes2(truth_dict, sensor_dict, bodies):
    '''
    This function computes visible passes for a given object catalog and 
    sensors.
    
    Parameters
    ------
    tvec : 1D numpy array
        initial and final time of visibility window seconds since J2000 TDB
    rso_dict : dictionary
        object state parameters including pos/vel in ECI [m] and physical 
        attributes
    sensor_dict : dictionary
        sensor parameters including location in ECEF [m] and constraints
        
    Returns
    ------    
    
    
    '''
    
    # Retrieve input data
    obj_id_list = sorted(list(truth_dict.keys()))
    sensor_id_list = sorted(list(sensor_dict.keys()))
    
    
    # Loop over objects
    visibility_dict = {}
    for obj_id in obj_id_list:
        
        print(obj_id)
        
        # Retrieve object true states and times
        tout = truth_dict[obj_id]['t_truth']
        Xout = truth_dict[obj_id]['X_truth']
        
        # Loop over times and check visibility
        for kk in range(len(tout)):
            
            # Retrieve current time and state
            tk = tout[kk]
            Xk = Xout[kk,:].reshape(6,1)     
            
            # Loop over sensors
            for sensor_id in sensor_id_list:
                
                # Retrieve sensor parameters
                sensor_params = sensor_dict[sensor_id]
                
                # Check visibility
                vis_flag, rg, az, el = check_visibility(tk, Xk, sensor_params,
                                                        bodies)
                
                # Store output
                if vis_flag:
                    
                    if sensor_id not in visibility_dict:
                        visibility_dict[sensor_id] = {}
                        
                    if obj_id not in visibility_dict[sensor_id]:
                        visibility_dict[sensor_id][obj_id] = {}
                        visibility_dict[sensor_id][obj_id]['tk_list'] = []
                        visibility_dict[sensor_id][obj_id]['rg_list'] = []
                        visibility_dict[sensor_id][obj_id]['el_list'] = []
                        visibility_dict[sensor_id][obj_id]['az_list'] = []
                        
                    visibility_dict[sensor_id][obj_id]['tk_list'].append(tk)
                    visibility_dict[sensor_id][obj_id]['rg_list'].append(rg)
                    visibility_dict[sensor_id][obj_id]['az_list'].append(az)
                    visibility_dict[sensor_id][obj_id]['el_list'].append(el)
                    
    
    return visibility_dict


def compute_pass(tk_list, rg_list, az_list, el_list):
    
    # Allowable gap to still be considered the same pass
    max_gap = 600.
    
    # Initialze output
    start_list = []
    stop_list = []
    TCA_list = []
    TME_list = []
    rg_min_list = []
    el_max_list = []
    
    # Loop over times
    rg_min = 1e12
    el_max = -1.
    tk_prior = tk_list[0]
    start = tk_list[0]
    stop = tk_list[0]
    TCA = tk_list[0]
    TME = tk_list[0]
    for kk in range(len(tk_list)):
        
        tk = tk_list[kk]
        rg = rg_list[kk]
        el = el_list[kk]
        
        # If current time is close to previous, pass continues
        if (tk - tk_prior) < (max_gap+1.):

            # Update pass stop time and tk_prior for next iteration
            stop = tk
            tk_prior = tk
            
            # Check if this is pass time of closest approach (TCA)
            if rg < rg_min:
                TCA = tk
                rg_min = float(rg)
            
            # Check if this is pass time of maximum elevation (TME)
            if el > el_max:
                TME = tk
                el_max = float(el)

        # If current time is far from previous or if we reached
        # the end of tk_list, pass has ended
        if ((tk - tk_prior) >= (max_gap+1.) or tk == tk_list[-1]):
            
            # Store stop time if at end of tk_list
            if tk == tk_list[-1]:
                stop = tk
            
            # Store output
            start_list.append(start)
            stop_list.append(stop)
            TCA_list.append(TCA)
            TME_list.append(TME)
            rg_min_list.append(rg_min)
            el_max_list.append(el_max)
            
            # Reset for new pass next round
            start = tk
            TCA = tk
            TME = tk
            stop = tk
            tk_prior = tk
            
            # TODO - LOGIC ERROR
            # Test this code to reset these params
            rg_min = 1e12
            el_max = -1
    
    
    # Store output
    pass_dict = {}
    pass_dict['start_list'] = start_list
    pass_dict['stop_list'] = stop_list
    pass_dict['TCA_list'] = TCA_list
    pass_dict['TME_list'] = TME_list
    pass_dict['rg_min_list'] = rg_min_list
    pass_dict['el_max_list'] = el_max_list
    
    
    return pass_dict


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

