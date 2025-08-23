import numpy as np
import math
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle


from tudatpy.numerical_simulation import environment_setup


import TudatPropagator as prop
import ConjunctionUtilities as conj
import SensorTasking as sensor


###############################################################################
# Basic I/O
###############################################################################

def read_truth_file(truth_file):
    '''
    This function reads a pickle file containing truth data for state 
    estimation.
    
    Parameters
    ------
    truth_file : string
        path and filename of pickle file containing truth data
    
    Returns
    ------
    t_truth : N element numpy array
        time in seconds since J2000
    X_truth : Nxn numpy array
        each row X_truth[k,:] corresponds to Cartesian state at time t_truth[k]
    state_params : dictionary
        propagator params
        
        fields:
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    '''
    
    # Load truth data
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    t_truth = data[0]
    X_truth = data[1]
    state_params = data[2]
    pklFile.close()
    
    return t_truth, X_truth, state_params


def read_measurement_file(meas_file):
    '''
    This function reads a pickle file containing measurement data for state 
    estimation.
    
    Parameters
    ------
    meas_file : string
        path and filename of pickle file containing measurement data
    
    Returns
    ------
    state_params : dictionary
        initial state and covariance for filter execution and propagator params
        
        fields:
            UTC: datetime object for epoch of state/covar
            state: nx1 numpy array contaiing position/velocity state in ECI [m, m/s]
            covar: nxn numpy array containing Gaussian covariance matrix [m^2, m^2/s^2]
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    meas_dict : dictionary
        measurement data over time for the filter 
        
        fields:
            tk_list: list of times in seconds since J2000
            Yk_list: list of px1 numpy arrays containing measurement data
            
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
            
    '''

    # Load measurement data
    pklFile = open(meas_file, 'rb' )
    data = pickle.load( pklFile )
    state_params = data[0]
    sensor_params = data[1]
    meas_dict = data[2]
    pklFile.close()
    
    return state_params, meas_dict, sensor_params



###############################################################################
# Unscented Kalman Filter
###############################################################################


def ukf(state_params, meas_dict, sensor_dict, int_params, filter_params, bodies):    
    '''
    This function implements the Unscented Kalman Filter for the least
    squares cost function.

    Parameters
    ------
    state_params : dictionary
        initial state and covariance for filter execution and propagator params
        
        fields:
            epoch_tdb: epoch of state/covar [seconds since J2000]
            state: nx1 numpy array contaiing position/velocity state in ECI [m, m/s]
            covar: nxn numpy array containing Gaussian covariance matrix [m^2, m^2/s^2]
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    meas_dict : dictionary
        measurement data over time for the filter 
        
        fields:
            tk_list: list of times in seconds since J2000
            Yk_list: list of px1 numpy arrays containing measurement data
            sensor_id_list: list of sensor id's corresponding to each measurement
            
    sensor_dict : dictionary
        location, constraint, noise parameters of sensor, indexed by sensor id
        
    int_params : dictionary
        numerical integration parameters
        
    filter_params : dictionary
        fields:
            Qeci: 3x3 numpy array of SNC accelerations in ECI [m/s^2]
            Qric: 3x3 numpy array of SNC accelerations in RIC [m/s^2]
            alpha: float, UKF sigma point spread parameter, should be in range [1e-4, 1]
            gap_seconds: float, time in seconds between measurements for which SNC should be zeroed out, i.e., if tk-tk_prior > gap_seconds, set Q=0
            
    bodies : tudat object
        contains parameters for the environment bodies used in propagation

    Returns
    ------
    filter_output : dictionary
        output state, covariance, and post-fit residuals at measurement times
        
        indexed first by tk, then contains fields:
            state: nx1 numpy array, estimated Cartesian state vector at tk [m, m/s]
            covar: nxn numpy array, estimated covariance at tk [m^2, m^2/s^2]
            resids: px1 numpy array, measurement residuals at tk [meters and/or radians]
        
    '''
        
    # Retrieve data from input parameters
    t0 = state_params['epoch_tdb']
    Xo = state_params['state']
    Po = state_params['covar']    
    Qeci = filter_params['Qeci']
    Qric = filter_params['Qric']
    alpha = filter_params['alpha']
    gap_seconds = filter_params['gap_seconds']

    n = len(Xo)
    q = int(Qeci.shape[0])
    
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
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    sensor_id_list = meas_dict['sensor_id_list']
    
    # Number of epochs
    N = len(tk_list)
  
    # Loop over times
    Xk = Xo.copy()
    Pk = Po.copy()
    for kk in range(N):
    
        # Current and previous time
        if kk == 0:
            tk_prior = t0
        else:
            tk_prior = tk_list[kk-1]

        tk = tk_list[kk]
        
        # Propagate state and covariance
        # No prediction needed if measurement time is same as current state
        if tk_prior == tk:
            Xbar = Xk.copy()
            Pbar = Pk.copy()
        else:
            tvec = np.array([tk_prior, tk])
            dum, Xbar, Pbar = prop.propagate_state_and_covar(Xk, Pk, tvec, state_params, int_params, bodies, alpha)
       
        # State Noise Compensation
        # Zero out SNC for long time gaps
        delta_t = tk - tk_prior
        if delta_t > gap_seconds:    
            Gamma = np.zeros((n,q))
        else:
            Gamma = np.zeros((n,q))
            Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
            Gamma[q:2*q,:] = delta_t * np.eye(q)

        # Combined Q matrix (ECI and RIC components)
        # Rotate RIC to ECI and add
        rc_vect = Xbar[0:3].reshape(3,1)
        vc_vect = Xbar[3:6].reshape(3,1)
        Q = Qeci + conj.ric2eci(rc_vect, vc_vect, Qric)
                
        # Add Process Noise to Pbar
        Pbar += np.dot(Gamma, np.dot(Q, Gamma.T))
        
        # Remediate covariance if needed
        Pbar = conj.remediate_covariance(Pbar, 1e-12)[0]

        # Recompute sigma points to incorporate process noise        
        sqP = np.linalg.cholesky(Pbar)
        Xrep = np.tile(Xbar, (1, n))
        chi_bar = np.concatenate((Xbar, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
        chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
        
        # Measurement Update: posterior state and covar at tk       
        # Retrieve measurement data
        Yk = Yk_list[kk]
        sensor_id = sensor_id_list[kk]
        sensor_params = sensor_dict[sensor_id]
        
        # Computed measurements and covariance
        gamma_til_k, Rk = unscented_meas(tk, chi_bar, sensor_params, bodies)
        ybar = np.dot(gamma_til_k, Wm.T)
        ybar = np.reshape(ybar, (len(ybar), 1))
        Y_diff = gamma_til_k - np.dot(ybar, np.ones((1, (2*n+1))))
        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk
        Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))
        
        # Kalman gain and measurement update
        Kk = np.dot(Pxy, np.linalg.inv(Pyy))
        Xk = Xbar + np.dot(Kk, Yk-ybar)
        
        # Joseph form of covariance update
        cholPbar = np.linalg.inv(np.linalg.cholesky(Pbar))
        invPbar = np.dot(cholPbar.T, cholPbar)
        P1 = (np.eye(n) - np.dot(np.dot(Kk, np.dot(Pyy, Kk.T)), invPbar))
        P2 = np.dot(Kk, np.dot(Rk, Kk.T))
        P = np.dot(P1, np.dot(Pbar, P1.T)) + P2

        # Recompute measurments using final state to get resids
        P = conj.remediate_covariance(P, 1e-12)[0]
        try:
            sqP = np.linalg.cholesky(P)
        except:
            print(np.linalg.det(P))
            print(np.linalg.eig(P))
            
        Xrep = np.tile(Xk, (1, n))
        chi_k = np.concatenate((Xk, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)        
        gamma_til_post, dum = unscented_meas(tk, chi_k, sensor_params, bodies)
        ybar_post = np.dot(gamma_til_post, Wm.T)
        ybar_post = np.reshape(ybar_post, (len(ybar), 1))
        
        # Post-fit residuals and updated state
        resids = Yk - ybar_post
        
        print('')
        print('kk', kk)
        print('tk', tk)
        print('delta_t', delta_t)
        print('Xbar', Xbar)
        print('Yk', Yk)
        print('ybar', ybar)     
        print('resids', resids)
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['state'] = Xk
        filter_output[tk]['covar'] = P
        filter_output[tk]['resids'] = resids

    
    return filter_output


###############################################################################
# Sensors and Measurements
###############################################################################


def unscented_meas(tk, chi, sensor_params, bodies):
    '''
    This function computes the measurement sigma point matrix.
    
    Parameters
    ------
    tk : float
        time in seconds since J2000
    chi : nx(2n+1) numpy array
        state sigma point matrix
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
        
    Returns
    ------
    gamma_til : px(2n+1) numpy array
        measurement sigma point matrix
    Rk : pxp numpy array
        measurement noise covariance
        
    '''
    
    # Number of states
    n = int(chi.shape[0])
    
    # Rotation matrices
    earth_rotation_model = bodies.get("Earth").rotation_model
    eci2ecef = earth_rotation_model.inertial_to_body_fixed_rotation(tk)
    ecef2eci = earth_rotation_model.body_fixed_to_inertial_rotation(tk)
    
    # Compute sensor position in ECI
    sensor_ecef = sensor_params['sensor_ecef']
    sensor_eci = np.dot(ecef2eci, sensor_ecef)
    
    # Measurement information    
    meas_types = sensor_params['meas_types']
    sigma_dict = sensor_params['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.
    
    # Compute transformed sigma points
    gamma_til = np.zeros((p, (2*n+1)))
    for jj in range(2*n+1):
        
        x = chi[0,jj]
        y = chi[1,jj]
        z = chi[2,jj]
        
        # Object location in ECI
        r_eci = np.reshape([x,y,z], (3,1))
        
        # Compute range and line of sight vector
        rho_eci = r_eci - sensor_eci
        rg = np.linalg.norm(rho_eci)
        rho_hat_eci = rho_eci/rg
        
        # Rotate to ENU frame
        rho_hat_ecef = np.dot(eci2ecef, rho_hat_eci)
        rho_hat_enu = sensor.ecef2enu(rho_hat_ecef, sensor_ecef)
        
        if 'rg' in meas_types:
            rg_ind = meas_types.index('rg')
            gamma_til[rg_ind,jj] = rg
            
        if 'ra' in meas_types:
        
            ra = math.atan2(rho_hat_eci[1], rho_hat_eci[0]) # rad        
        
            # Store quadrant info of mean sigma point        
            if jj == 0:
                quad = 0
                if ra > np.pi/2. and ra < np.pi:
                    quad = 2
                if ra < -np.pi/2. and ra > -np.pi:
                    quad = 3
                    
            # Check and update quadrant of subsequent sigma points
            else:
                if quad == 2 and ra < 0.:
                    ra += 2.*np.pi
                if quad == 3 and ra > 0.:
                    ra -= 2.*np.pi
                    
            ra_ind = meas_types.index('ra')
            gamma_til[ra_ind,jj] = ra
                
        if 'dec' in meas_types:        
            dec = math.asin(rho_hat_eci[2])  # rad
            dec_ind = meas_types.index('dec')
            gamma_til[dec_ind,jj] = dec
            
        if 'az' in meas_types:
            az = math.atan2(rho_hat_enu[0], rho_hat_enu[1])  # rad 
            if az < 0.:
                az += 2*np.pi
            
            # Store quadrant info of mean sigma point        
            if jj == 0:
                quad = 0
                # if az > np.pi/2. and az < np.pi:
                #     quad = 2
                # if az < -np.pi/2. and az > -np.pi:
                #     quad = 3
                
                if az > 0. and az < np.pi/2.:
                    quad = 1
                if az > 3.*np.pi/2.:
                    quad = 4
                    
            # Check and update quadrant of subsequent sigma points
            else:
                if quad == 1 and az > 3.*np.pi/2.:
                    az -= 2.*np.pi
                if quad == 4 and az <np.pi/2.:
                    az += 2.*np.pi
                    
            az_ind = meas_types.index('az')
            gamma_til[az_ind,jj] = az
            
        if 'el' in meas_types:
            el = math.asin(rho_hat_enu[2])  # rad
            el_ind = meas_types.index('el')
            gamma_til[el_ind,jj] = el


    return gamma_til, Rk

