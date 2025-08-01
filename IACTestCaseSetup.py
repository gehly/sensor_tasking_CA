import numpy as np
import math
from datetime import datetime, timedelta
import os
import pickle
import matplotlib.pyplot as plt
import time

# Load tudatpy modules
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.astro import time_conversion
from tudatpy.astro.time_conversion import DateTime

# Import utility functions
import ConjunctionUtilities as conj
import TudatPropagator as prop


###############################################################################
# Setup Primary Object
###############################################################################

def compute_UTC_from_TDB():
    
    # Desired times in TDB
    TDB_list = [datetime(2025, 7, 29, 12, 0, 0)]
    
    # Convert to UTC for TLE retrieval
    UTC_list = []
    epoch_tdb_list = []
    time_scale_converter = time_conversion.default_time_scale_converter()
    for TDB_dt in TDB_list:
        tudat_datetime_tdb = time_conversion.datetime_to_tudat(TDB_dt)
        epoch_tdb_ii = tudat_datetime_tdb.epoch()
        epoch_utc = time_scale_converter.convert_time(
                    input_scale = time_conversion.tdb_scale,
                    output_scale = time_conversion.utc_scale,
                    input_value = epoch_tdb_ii)
        
        tudat_datetime_utc = time_conversion.date_time_from_epoch(epoch_utc)
        python_datetime_utc = time_conversion.datetime_to_python(tudat_datetime_utc)
        UTC_list.append(python_datetime_utc)
        epoch_tdb_list.append(epoch_tdb_ii)
        
    
    print(UTC_list)    
    
    return UTC_list, epoch_tdb_list


def define_primary_object(epoch_tdb):
    
    # TLE retrieved and propagated to epoch using SGP4 in separate function
    
    # Starlink 3350 (NORAD 52373) 
    # 2025-07-29 12:00:00 TDB = 2025-07-29 11:58:50.816644 UTC
    obj_id = 52373
    Xo = np.reshape([ 4.48960010e+06, -9.70051996e+05, -5.18042169e+06,
                     -9.74421049e+02,  7.19247706e+03, -2.19294121e+03 ], (6,1))
    
    
    # Data from DISCOSweb
    mass = 260.         # kg
    area = 11.087       # m^2
    
    # Nominal Drag and SRP
    Cd = 2.2
    Cr = 1.3
    
    # Initialize RSO dictionary
    rso_dict = {}    
    rso_dict[obj_id] = {}
    rso_dict[obj_id]['epoch_tdb'] = epoch_tdb
    rso_dict[obj_id]['state'] = Xo
    rso_dict[obj_id]['mass'] = mass
    rso_dict[obj_id]['area'] = area
    rso_dict[obj_id]['Cd'] = Cd
    rso_dict[obj_id]['Cr'] = Cr
    
    
    primary_file = os.path.join('data', 'primary_catalog_truth.pkl')
    pklFile = open( primary_file, 'wb' )
    pickle.dump([rso_dict], pklFile, -1)
    pklFile.close()
    
    
    return 


def interp_lagrange(X, Y, xx, p):
    '''
    This function interpolates data using Lagrange method of order P
    
    Parameters
    ------
    X : 1D numpy array
        x-values of data to interpolate
    Y : 2D numpy array
        y-values of data to interpolate
    xx : float
        single x value to interpolate at
    p : int
        order of interpolation
    
    Returns
    ------
    yy : 1D numpy array
        interpolated y-value(s)
        
    References
    ------
    [1] Kharab, A., An Introduction to Numerical Methods: A MATLAB 
        Approach, 2nd ed., 2005.
            
    '''
    
    # Number of data points to use for interpolation (e.g. 8,9,10...)
    N = p + 1

    if (len(X) < N):
        print('Not enough data points for desired Lagrange interpolation!')
        
    # Compute number of elements on either side of middle element to grab
    No2 = 0.5*N
    nn  = int(math.floor(No2))
    
    # Find index such that X[row0] < xx < X[row0+1]
    row0 = list(np.where(X <= xx)[0])[-1]
    
    # Trim data set
    # N is even (p is odd)    
    if (No2-nn == 0): 
        
        # adjust row0 in case near data set endpoints
        if (N == len(X)) or (row0 < nn-1):
            row0 = nn-1
        elif (row0 >= (len(X)-nn)):            
            row0 = len(X) - nn - 1        
    
        # Trim to relevant data points
        X = X[row0-nn+1 : row0+nn+1]
        Y = Y[row0-nn+1 : row0+nn+1, :]


    # N is odd (p is even)
    else:
    
        # adjust row0 in case near data set endpoints
        if (N == len(X)) or (row0 < nn):
            row0 = nn
        elif (row0 > len(X)-nn):
            row0 = len(X) - nn - 1
        else:
            if (xx-X[row0] > 0.5) and (row0+1+nn < len(X)):
                row0 = row0 + 1
    
        # Trim to relevant data points
        X = X[row0-nn:row0+nn+1]
        Y = Y[row0-nn:row0+nn+1, :]
        
    # Compute coefficients
    Pj = np.ones((1,N))
    
    for jj in range(N):
        for ii in range(N):
            
            if jj != ii:
                Pj[0, jj] = Pj[0, jj] * (-xx+X[ii])/(-X[jj]+X[ii])
    
    
    yy = np.dot(Pj, Y)
    
    return yy


def verify_numerical_error():
    
    # Per Baak Appx B (2025) and Ravago (2021) apply 20x20 gravity field
    # Note this will yield ~1000m error after 7 day propagation, but 
    # to achieve meter level requires 100x100 which is undesirably slow
    
    # Also from Baak, DP7 with dt = 4 sec should yield ~1cm numerical error
    # for 7 day forward/backprop, as verified in this function.
    
    # Load primary object data
    primary_file = os.path.join('data', 'primary_catalog_truth.pkl')
    pklFile = open(primary_file, 'rb' )
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    obj_id = int(list(rso_dict.keys())[0])
    
    # Setup dynamics parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)    
    
    state_params = {}
    state_params['mass'] = rso_dict[obj_id]['mass']
    state_params['area'] = rso_dict[obj_id]['area']
    state_params['Cd'] = rso_dict[obj_id]['Cd']
    state_params['Cr'] = rso_dict[obj_id]['Cr']
    state_params['sph_deg'] = 20
    state_params['sph_ord'] = 20   
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create
    
    int_params = {}
    int_params['tudat_integrator'] = 'dp7'
    int_params['step'] = 4.
    
    # Initial state and propagation times
    Xo = rso_dict[obj_id]['state']
    t0 = rso_dict[obj_id]['epoch_tdb']
    tf = t0 + 7.*86400.
    tvec = np.array([t0, tf])
    
    print('prop')
    start = time.time()
    tout, Xout = prop.propagate_orbit(Xo, tvec, state_params, int_params, bodies)
    dp7_prop_time = time.time() - start
    Xf = Xout[-1,:].reshape(6,1)
    
    # Setup backpropagation
    int_params['step'] = -4.
    tvec = np.array([tf, t0])
    
    print('backprop impactor')
    start = time.time()
    tout2, Xout2 = prop.propagate_orbit(Xf, tvec, state_params, int_params, bodies)
    dp7_backprop_time = time.time() - start
    
    # Compute and plot error
    error = Xout2 - Xout
    pos_error = np.linalg.norm(error[:,0:3], axis=1)
    
    tdays = [(ti - tout[0])/86400. for ti in tout]
    
    plt.figure()
    plt.semilogy(tdays, pos_error, 'k')
    # plt.semilogy(tdays, error[:,0], 'r', label='x')
    # plt.semilogy(tdays, error[:,1], 'b', label='y')
    # plt.semilogy(tdays, error[:,2], 'g', label='z')
    plt.title('Forward/Backprop Analysis DP7 step = 4sec')
    plt.ylabel('Error [m]')
    plt.xlabel('Time [days]')
    # plt.legend()
    
    
    # Test variable step integrator
    int_params['tudat_integrator'] = 'dp87'
    int_params['step'] = 10.
    int_params['max_step'] = 1000.
    int_params['min_step'] = 1e-3
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    
    tvec = np.array([t0, tf])
    
    print('variable prop')
    start = time.time()
    tout3, Xout3 = prop.propagate_orbit(Xo, tvec, state_params, int_params, bodies)
    dp87_prop_time = time.time() - start
        
    # Compute and plot error
    dp87_err = []
    for ii in range(len(tout3)):
        ti = tout3[ii]
        Xi_var = Xout3[ii,:]
        Xi_fixed = interp_lagrange(tout, Xout, ti, 8).flatten()
        
        print('')
        print(ii, (ti-t0)/86400.)
        print(Xi_var)
        print(Xi_fixed)
        
        err = np.linalg.norm(Xi_var[0:3] - Xi_fixed[0:3])
        print(err)
        dp87_err.append(err)
        
        
    
    
    
    
    
    tdays = [(ti - tout[0])/86400. for ti in tout3]
    
    plt.figure()
    plt.semilogy(tdays, dp87_err, 'k', label='3D pos')
    plt.title('Variable Step Integrator Analysis DP87 tol=1e-12')
    plt.ylabel('Error [m]')
    plt.xlabel('Time [days]')
    plt.legend()
    
    
    
    plt.show()
    
    print('')
    print('dp7 prop time', dp7_prop_time)
    print('dp7 backprop time', dp7_backprop_time)
    print('dp87 prop time', dp87_prop_time)
    
    
    
    return



###############################################################################
# Setup Secondary Objects
###############################################################################



def create_conjunction(rso_dict, asset_id, impactor_id, impactor_case, TCA_hrs, rho_ric, drho_ric, halt_flag=False):
    
    # Asset data
    Xo_true = rso_dict[asset_id]['state']
    
    # Basic setup parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)    
    
    state_params = {}
    state_params['mass'] = rso_dict[asset_id]['mass']
    state_params['area'] = rso_dict[asset_id]['area']
    state_params['Cd'] = rso_dict[asset_id]['Cd']
    state_params['Cr'] = rso_dict[asset_id]['Cr']
    state_params['sph_deg'] = 8
    state_params['sph_ord'] = 8    
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create

    int_params = {}
    int_params['tudat_integrator'] = 'rk4'
    int_params['step'] = 1.
    
    # int_params = {}
    # int_params['tudat_integrator'] = 'rkf78'
    # int_params['step'] = 10.
    # int_params['max_step'] = 1000.
    # int_params['min_step'] = 1e-3
    # int_params['rtol'] = 1e-12
    # int_params['atol'] = 1e-12
    
    # Integration times
    t0 = rso_dict[asset_id]['epoch_tdb']
    tf = t0 + TCA_hrs*3600.
    tvec = np.array([t0, tf])
    
    print('prop')
    tout, Xout = prop.propagate_orbit(Xo_true, tvec, state_params, int_params, bodies)
    Xf_true = Xout[-1,:].reshape(6,1)
    
    
    # Compute impactor truth state
    rc_vect = Xf_true[0:3].reshape(3,1)
    vc_vect = Xf_true[3:6].reshape(3,1)
    
    # rho_ric = np.array([0., 0., 0.]).reshape(3,1)
    # drho_ric = np.array([100., -15300., 100.]).reshape(3,1)
    
    rho_eci = coord.ric2eci(rc_vect, vc_vect, rho_ric)
    #drho_eci = coord.ric2eci(rc_vect, vc_vect, drho_ric)
    drho_eci = coord.ric2eci_vel(rc_vect, vc_vect, rho_ric, drho_ric)
    r_eci = rc_vect + rho_eci
    v_eci = vc_vect + drho_eci
    
    Xf_imp_true = np.concatenate((r_eci, v_eci), axis=0)    
    kep_imp = astro.cart2kep(Xf_imp_true, GME*1e9)
    
    rp = float(kep_imp[0,0]*(1-kep_imp[1,0]))
    ra = float(kep_imp[0,0]*(1+kep_imp[1,0]))
    
    if rp < 6578000:
        mistake
    
    print('')
    print('Xf_true', Xf_true)
    print('Xf_imp_true', Xf_imp_true)
    print('kep_imp', kep_imp)
    print('rp', rp)
    print('ra', ra)
    print('miss distance', np.linalg.norm(Xf_true[0:3] - Xf_imp_true[0:3]))
    print('impact velocity', np.linalg.norm(Xf_true[3:6] - Xf_imp_true[3:6]))
    print('')
    
    if halt_flag:
        mistake
        
        
    # Backpropagate impactor
    dum, mass, area, Cd, Cr = object_params(impactor_case)
    
    # Basic setup parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)    
    
    state_params = {}
    state_params['mass'] = mass
    state_params['area'] = area
    state_params['Cd'] = Cd
    state_params['Cr'] = Cr
    state_params['sph_deg'] = 8
    state_params['sph_ord'] = 8    
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create

    int_params = {}
    int_params['tudat_integrator'] = 'rk4'
    int_params['step'] = -1.
    
    # int_params['tudat_integrator'] = 'rkf78'
    # int_params['step'] = 10.
    # int_params['max_step'] = 1000.
    # int_params['min_step'] = 1e-3
    # int_params['rtol'] = 1e-12
    # int_params['atol'] = 1e-12
    
    # Integration times
    tvec = np.array([tf, t0])
    
    print('backprop impactor')
    tout2, Xout2 = prop.propagate_orbit(Xf_imp_true, tvec, state_params, int_params, bodies)
    
    Xo_imp_true = Xout2[0,:].reshape(6,1)
    Xf_imp_check = Xout2[-1,:].reshape(6,1)
    
    print('')
    print('Xo true', Xo_true)
    print('Xf true', Xf_true)
    print('xf_imp_true', Xf_imp_check)
    print('Xo_imp_true', Xo_imp_true)
    
    
    # Add to output
    rso_dict[impactor_id] = {}
    rso_dict[impactor_id]['epoch_tdb'] = t0
    rso_dict[impactor_id]['state'] = Xo_imp_true
    rso_dict[impactor_id]['mass'] = mass
    rso_dict[impactor_id]['area'] = area
    rso_dict[impactor_id]['Cd'] = Cd
    rso_dict[impactor_id]['Cr'] = Cr
    
    
    
    return rso_dict


if __name__ == '__main__':
    
    plt.close('all')
    
    # UTC_list, epoch_tdb_list = compute_UTC_from_TDB()
    
    # define_primary_object(epoch_tdb_list[0])

    verify_numerical_error()

