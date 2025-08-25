import numpy as np
import math
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle
import scipy


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
        # Pbar = conj.remediate_covariance(Pbar, 1e-12)[0]

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
        # P = conj.remediate_covariance(P, 1e-12)[0]
        sqP = np.linalg.cholesky(P)
        # except:
        #     print(np.linalg.det(P))
        #     print(np.linalg.eig(P))
            
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
# Unscented Batch
###############################################################################


def unscented_batch(state_params, meas_dict, sensor_dict, int_params, filter_params, bodies):    
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
    tk_output = filter_params['tk_output']

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
    nmeas = sum([len(Yk) for Yk in Yk_list])
    
    # Number of epochs
    N = len(tk_list)
    
    # Initialize 
    X = Xo.copy()
    P = Po.copy()
    maxiters = 10
    xdiff = 1
    rms_prior = 1e6
    xdiff_crit = 1e-5
    rms_crit = 1e-4
    conv_flag = False 
    
    # Begin loop
    iters = 0
    while not conv_flag:

        # Increment loop counter and exit if necessary
        iters += 1
        if iters > maxiters:
            iters -= 1
            print('Solution did not converge in ', iters, ' iterations')
            print('Last xdiff magnitude: ', xdiff)
            break

        # Initialze values for this iteration
        # Reset P every iteration???
        P = Po.copy()
    
        # Compute Sigma Points
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(X, (1, n))
        chi0 = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_v = np.reshape(chi0, (n*(2*n+1), 1), order='F')  
        chi_diff0 = chi0 - np.dot(X, np.ones((1, 2*n+1)))
        
        # Loop over times
        meas_ind = 0
        Y_bar = np.zeros((nmeas, 1))
        Y_til = np.zeros((nmeas, 1))
        gamma_til_mat = np.zeros((nmeas, 2*n+1))
        Rk_list = []
        resids_list = []
        resids_sum = 0.
        for kk in range(N):
            
#            print('\nkk = ', kk)
            
            # Current and previous time
            if kk == 0:
                tk_prior = t0
            else:
                tk_prior = tk_list[kk-1]

            tk = tk_list[kk]
            
            # Initial Conditions for Integration Routine
            int0 = chi_v.copy()
            
            # Integrate Xref and STM
            if tk_prior == tk:
                intout = int0.T
            else:
                int0 = int0.flatten()
                tin = [tk_prior, tk]
                
                tout, intout = prop.propagate_orbit(int0, tin, state_params, int_params, bodies)
    
            # Extract values for later calculations
            chi_v = intout[-1,:]
            chi = np.reshape(chi_v, (n, 2*n+1), order='F')
    
    

            # Propagate state and covariance
            # No prediction needed if measurement time is same as current state
            # if tk_prior == tk:
            #     Xbar = X.copy()
            #     Pbar = P.copy()
            # else:
            #     tvec = np.array([tk_prior, tk])
            #     dum, Xbar, Pbar = prop.propagate_state_and_covar(X, P, tvec, state_params, int_params, bodies, alpha)

            # # Recompute sigma points     
            # sqP = np.linalg.cholesky(Pbar)
            # Xrep = np.tile(Xbar, (1, n))
            # chi_bar = np.concatenate((Xbar, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
            # chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
        
            # Measurement Update      
            # Retrieve measurement data
            Yk = Yk_list[kk]
            sensor_id = sensor_id_list[kk]
            sensor_params = sensor_dict[sensor_id]
            p = len(Yk)
        
            # Computed measurements and covariance
            gamma_til_k, Rk = unscented_meas(tk, chi, sensor_params, bodies)
            
            # Standard implementation computes ybar as the mean of the sigma
            # point, but using Po_bar each iteration can cause these to have
            # large spread and produce poor ybar calculation
            # ybar = np.dot(gamma_til_k, Wm.T)
            
            # Instead, use only the first column of gamma_til_k, corresponding
            # to the mean state calculated with the best updated value of X(t0)
            ybar = gamma_til_k[:,0]
            
            # Reshape and continue
            ybar = np.reshape(ybar, (p, 1))
            resids = Yk - ybar
            cholRk = np.linalg.inv(np.linalg.cholesky(Rk))
            invRk = np.dot(cholRk.T, cholRk)
            
            # Accumulate measurements and computed measurements
            Y_til[meas_ind:meas_ind+p] = Yk
            Y_bar[meas_ind:meas_ind+p] = ybar
            gamma_til_mat[meas_ind:meas_ind+p, :] = gamma_til_k  
            Rk_list.append(Rk)
            
            # Store output
            resids_list.append(resids)
            resids_sum += float(np.dot(resids.T, np.dot(invRk, resids)))
            
            # Increment measurement index
            meas_ind += p
                        
        # Compute covariances
        Rk_full = scipy.linalg.block_diag(*Rk_list)
        Y_diff = gamma_til_mat - np.dot(Y_bar, np.ones((1, 2*n+1)))
            
        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk_full
        Pxy = np.dot(chi_diff0, np.dot(diagWc, Y_diff.T)) 
        
        # Compute Kalman Gain
        cholPyy_inv = np.linalg.inv(np.linalg.cholesky(Pyy))
        Pyy_inv = np.dot(cholPyy_inv.T, cholPyy_inv) 
        
        K = np.dot(Pxy, Pyy_inv)
        
        # Compute updated state and covariance    
        X += np.dot(K, Y_til-Y_bar)
        
        # Joseph Form        
        cholPbar = np.linalg.inv(np.linalg.cholesky(P))
        invPbar = np.dot(cholPbar.T, cholPbar)
        P1 = (np.eye(n) - np.dot(np.dot(K, np.dot(Pyy, K.T)), invPbar))
        P2 = np.dot(K, np.dot(Rk_full, K.T))
        P = np.dot(P1, np.dot(P, P1.T)) + P2
        
        xdiff = np.linalg.norm(np.dot(K, Y_til-Y_bar))
        
        resids_rms = np.sqrt(resids_sum/nmeas)
        resids_diff = abs(resids_rms - rms_prior)/rms_prior
        if resids_diff < rms_crit:
            conv_flag = True
            
        rms_prior = float(resids_rms)
        

        print('Iteration Number: ', iters)
        print('xdiff = ', xdiff)
        print('delta-X = ', np.dot(K, Y_til-Y_bar))
        print('Xo', Xo)
        print('X', X)
        print('resids_rms = ', resids_rms)
        print('resids_diff = ', resids_diff)
        
        
        
    # Setup for full_state_output
    Xo = X.copy()
    Po = P.copy()
    
    # Compute Sigma Points
    sqP = np.linalg.cholesky(Po)
    Xrep = np.tile(Xo, (1, n))
    chi0 = np.concatenate((Xo, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    chi_v = np.reshape(chi0, (n*(2*n+1), 1), order='F')
    
    # Integrate over full time
    full_state_output = {}
    for kk in range(len(tk_output)):
        
        # Current and previous time
        if kk == 0:
            tk_prior = t0
        else:
            tk_prior = tk_output[kk-1]
            
        tk = tk_output[kk]

        # Initial Conditions for Integration Routine
        int0 = chi_v.copy()

        # Integrate Xref and STM
        if tk_prior == tk:
            intout = int0.T
        else:
            int0 = int0.flatten()
            tin = [tk_prior, tk]
            
            tout, intout = prop.propagate_orbit(int0, tin, state_params, int_params, bodies)

        # Extract values for later calculations
        chi_v = intout[-1,:]
        chi = np.reshape(chi_v, (n, 2*n+1), order='F')
    
        # Store output
        Xk = np.dot(chi, Wm.T)
        Xk = np.reshape(Xk, (n, 1))
        chi_diff = chi - np.dot(Xk, np.ones((1, (2*n+1))))
        Pk = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
        
        full_state_output[tk] = {}
        full_state_output[tk]['state'] = Xk
        full_state_output[tk]['covar'] = Pk
        
        if tk in tk_list:
            filter_output[tk] = {}
            filter_output[tk]['state'] = Xk
            filter_output[tk]['covar'] = Pk
            filter_output[tk]['resids'] = resids_list[tk_list.index(tk)]    
        
    # Setup for full_state_output
    # Xo = X.copy()
    # Po = P.copy()
    
    # # Compute Sigma Points
    # sqP = np.linalg.cholesky(Po)
    # Xrep = np.tile(Xo, (1, n))
    # chi0 = np.concatenate((Xo, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    # chi_v = np.reshape(chi0, (n*(2*n+1), 1), order='F')
    
    # # Integrate over full time
    # full_state_output = {}
    # for kk in range(len(tk_output)):
        
    #     # Current and previous time
    #     if kk == 0:
    #         tk_prior = t0
    #     else:
    #         tk_prior = tk_output[kk-1]
            
    #     tk = tk_output[kk]
    
    #     # Propagate state and covariance
    #     # No prediction needed if measurement time is same as current state
    #     if tk_prior == tk:
    #         Xbar = X.copy()
    #         Pbar = P.copy()
    #     else:
    #         tvec = np.array([tk_prior, tk])
    #         dum, X, P = prop.propagate_state_and_covar(X, P, tvec, state_params, int_params, bodies, alpha)
        
    #     # Store output
    #     full_state_output[tk] = {}
    #     full_state_output[tk]['state'] = X
    #     full_state_output[tk]['covar'] = P
        
    #     if tk in tk_list:
    #         filter_output[tk] = {}
    #         filter_output[tk]['state'] = X
    #         filter_output[tk]['covar'] = P
    #         filter_output[tk]['resids'] = resids_list[tk_list.index(tk)]
        
        
        
    
    return filter_output, full_state_output


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

