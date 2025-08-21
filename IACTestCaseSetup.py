import numpy as np
import math
from datetime import datetime, timedelta
import os
import pickle
import matplotlib.pyplot as plt
import time
import sys
import csv

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
import SensorTasking as sensor

metis_dir = r'C:\Users\sgehly\Documents\code\metis'
sys.path.append(metis_dir)

import estimation.analysis_functions as analysis
import estimation.estimation_functions as metis_est
import dynamics.dynamics_functions as metis_dyn
import sensors.measurement_functions as metis_mfunc
import sensors.sensors as metis_sensors


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


###############################################################################
# Setup Secondary Objects
###############################################################################


def secondary_params(case_id):
    '''
    This function contains all the basic setup parameters for the secondary
    objects that will have close approaches with the primary.
    
    '''
        
    # Global Parameters
    Cd = 2.2
    Cr = 1.3
    
    if case_id == 0:
        
        # 1U Cubesat
        # Impact
        obj_id = 90000
        mass = 1.
        area = 0.01
        rho_ric = np.array([0., 0., 0.]).reshape(3,1)
        drho_ric = np.array([10., -15190., 4.]).reshape(3,1)
        TCA_hrs = 30.
        
    elif case_id == 1:
        
        # Rocket Body (SpaceX Merlin-V/Falcon Upper Stage) - DISCOSweb
        # Miss 100m radial
        obj_id = 91000
        mass = 4300.
        area = 40.
        rho_ric = np.array([100., 0., 0.]).reshape(3,1)
        drho_ric = np.array([10., -1., 800.]).reshape(3,1)
        TCA_hrs = 42.
        
    elif case_id == 2:
        
        # Defunct Starlink
        # Impact
        obj_id = 92000
        mass = 260.         # kg
        area = 11.0       # m^2
        rho_ric = np.array([0., 0., 0.]).reshape(3,1)
        # drho_ric = np.array([-4., -5600., 7400.]).reshape(3,1)
        drho_ric = np.array([0., -7., -300.]).reshape(3,1)
        TCA_hrs = 60.
        
    elif case_id == 3:
        
        # 3U Cubesat
        # Miss 500m Along-Track
        obj_id = 93000
        mass = 4.           # kg
        area = 0.03         # m^2
        rho_ric = np.array([0., 500., 0.]).reshape(3,1)
        drho_ric = np.array([10., 10., 600.]).reshape(3,1)
        TCA_hrs = 80.
        
    elif case_id == 4:
        
        # Rocket Body (SpaceX Merlin-V/Falcon Upper Stage) - DISCOSweb
        # Impact
        obj_id = 94000
        mass = 4300.           # kg
        area = 40.             # m^2
        rho_ric = np.array([0., 0., 0.]).reshape(3,1)
        drho_ric = np.array([10., 10., 600.]).reshape(3,1)
        TCA_hrs = 97.
        
    elif case_id == 5:
        
        # Defunct Starlink
        #  Miss 1000m Along Track
        obj_id = 95000
        mass = 260.         # kg
        area = 11.0       # m^2        
        rho_ric = np.array([0., 1000., 0.]).reshape(3,1)
        # drho_ric = np.array([-4., -5600., 7400.]).reshape(3,1)
        drho_ric = np.array([(0., 10., 400.)]).reshape(3,1)
        # drho_ric = np.array([0., -7., -300.]).reshape(3,1)
        TCA_hrs = 98.
        
    elif case_id == 6:
        
        # 1U Cubesat
        # Miss 100m Cross Track
        obj_id = 96000
        mass = 1.
        area = 0.01
        rho_ric = np.array([0., 0., 100.]).reshape(3,1)
        drho_ric = np.array([10., -15190., 4.]).reshape(3,1)
        TCA_hrs = 99.
        
    elif case_id == 7:
        
        # 3U Cubesat Impact
        obj_id = 97000
        mass = 4.           # kg
        area = 0.03         # m^2
        rho_ric = np.array([0., 0., 0.]).reshape(3,1)
        drho_ric = np.array([10., -15190., 4.]).reshape(3,1)
        TCA_hrs = 125.
        
    
    elif case_id == 8:
        
        # Rocket Body (SpaceX Merlin-V/Falcon Upper Stage) - DISCOSweb
        # Impact
        obj_id = 98000
        mass = 4300.           # kg
        area = 40.             # m^2
        rho_ric = np.array([0., 0., 0.]).reshape(3,1)
        drho_ric = np.array([10., -15190., 4.]).reshape(3,1)
        TCA_hrs = 145.
        
        
    elif case_id == 9:
        
        # 1U Cubesat
        # Miss 50m Radial
        obj_id = 99000
        mass = 1.
        area = 0.01
        rho_ric = np.array([50., 0., 0.]).reshape(3,1)
        drho_ric = np.array([10., -1., 800.]).reshape(3,1)
        TCA_hrs = 162.
        
 
    
    return obj_id, mass, area, Cd, Cr, TCA_hrs, rho_ric, drho_ric



def build_truth_catalog(rso_file, case_id):
    
    # Load RSO dict
    pklFile = open(rso_file, 'rb' )
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    primary_id = 52373
    
    # for case_id in range(10):

    rso_dict = create_conjunction(rso_dict, primary_id, case_id)
    
    print('')
    print('number rso', len(rso_dict))
    
    
    pklFile = open( rso_file, 'wb' )
    pickle.dump([rso_dict], pklFile, -1)
    pklFile.close()
    
    return


def create_conjunction(rso_dict, primary_id, case_id, halt_flag=False):
    
    # Primary object data
    Xo_true = rso_dict[primary_id]['state']
    
    # Secondary object data
    secondary_id, mass, area, Cd, Cr, TCA_hrs, rho_ric, drho_ric = \
        secondary_params(case_id)
    
    # Basic setup parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)    
    
    state_params = {}
    state_params['mass'] = rso_dict[primary_id]['mass']
    state_params['area'] = rso_dict[primary_id]['area']
    state_params['Cd'] = rso_dict[primary_id]['Cd']
    state_params['Cr'] = rso_dict[primary_id]['Cr']
    state_params['sph_deg'] = 20
    state_params['sph_ord'] = 20   
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create

    int_params = {}
    int_params['tudat_integrator'] = 'dp7'
    int_params['step'] = 4.
    
    # Integration times
    t0 = rso_dict[primary_id]['epoch_tdb']
    tf = t0 + TCA_hrs*3600.
    tvec = np.array([t0, tf])
    
    print('prop')
    tout, Xout = prop.propagate_orbit(Xo_true, tvec, state_params, int_params, bodies)
    Xf_true = Xout[-1,:].reshape(6,1)
    
    
    # Compute impactor truth state
    rc_vect = Xf_true[0:3].reshape(3,1)
    vc_vect = Xf_true[3:6].reshape(3,1)
    
    
    rho_eci = conj.ric2eci(rc_vect, vc_vect, rho_ric)
    #drho_eci = conj.ric2eci(rc_vect, vc_vect, drho_ric)
    drho_eci = conj.ric2eci_vel(rc_vect, vc_vect, rho_ric, drho_ric)
    r_eci = rc_vect + rho_eci
    v_eci = vc_vect + drho_eci
    
    Xf_imp_true = np.concatenate((r_eci, v_eci), axis=0)    
    kep_imp = cart2kep(Xf_imp_true, 3.986e14)
    
    rp = float(kep_imp[0,0]*(1-kep_imp[1,0]))
    ra = float(kep_imp[0,0]*(1+kep_imp[1,0]))
    
    if rp < 6578000:
        mistake
    
    print('')
    print('Xf_true', Xf_true)
    print('Xf_imp_true', Xf_imp_true)
    print('kep_imp', kep_imp)
    print('hp', (rp-6378000.)/1000.)
    print('ha', (ra-6378000.)/1000.)
    print('inc', float(kep_imp[2,0]))
    print('miss distance', np.linalg.norm(Xf_true[0:3] - Xf_imp_true[0:3]))
    print('impact velocity', np.linalg.norm(Xf_true[3:6] - Xf_imp_true[3:6]))
    print('')
    
    if halt_flag:
        mistake
        
        
    # Backpropagate impactor
    
    # Basic setup parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)    
    
    state_params = {}
    state_params['mass'] = mass
    state_params['area'] = area
    state_params['Cd'] = Cd
    state_params['Cr'] = Cr
    state_params['sph_deg'] = 20
    state_params['sph_ord'] = 20   
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create

    int_params = {}
    int_params['tudat_integrator'] = 'dp7'
    int_params['step'] = -4.

    
    # Integration times
    tvec = np.array([tf, t0])
    
    print('backprop impactor')
    tout2, Xout2 = prop.propagate_orbit(Xf_imp_true, tvec, state_params, int_params, bodies)
    
    Xo_imp_true = Xout2[0,:].reshape(6,1)
    Xf_imp_check = Xout2[-1,:].reshape(6,1)
    
    # print('')
    # print('Xo true', Xo_true)
    # print('Xf true', Xf_true)
    # print('xf_imp_true', Xf_imp_check)
    # print('Xo_imp_true', Xo_imp_true)
    
    
    # Add to output
    rso_dict[secondary_id] = {}
    rso_dict[secondary_id]['epoch_tdb'] = t0
    rso_dict[secondary_id]['state'] = Xo_imp_true
    rso_dict[secondary_id]['mass'] = mass
    rso_dict[secondary_id]['area'] = area
    rso_dict[secondary_id]['Cd'] = Cd
    rso_dict[secondary_id]['Cr'] = Cr
    
    
    return rso_dict


###############################################################################
# Create Tertiaries
###############################################################################

def create_tertiary_object(Xo_secondary):
    
    # Uncertainties for perturbation
    sig_sma = 0.
    sig_ecc = 1e-4
    sig_inc = 3.
    sig_RAAN = 3.
    sig_AOP = 3.
    sig_TA = 5.
    
    # Convert secondary state to orbit elements
    elem = cart2kep(Xo_secondary, 3.986e14)
    print('secondary elem', elem)
        
    # Perturb orbit elements
    elem[0] += sig_sma*np.random.randn()    
    pert_ecc = sig_ecc*np.random.randn()
    while elem[1] + pert_ecc < 0:
        pert_ecc = sig_ecc*np.random.randn()
    elem[1] += pert_ecc
    elem[2] += sig_inc*np.random.randn()
    elem[3] += sig_RAAN*np.random.randn()
    elem[4] += sig_AOP*np.random.randn()
    elem[5] += sig_TA*np.random.randn()
    
    # Convert to Cartesian state
    Xo_tertiary = kep2cart(elem, 3.986e14)
    
    print('tertiary elem', elem)
    print('cart diff', Xo_tertiary - Xo_secondary)
    
    return Xo_tertiary


def create_tertiary_catalog(rso_file):
    
    # Load true RSO dict
    pklFile = open(rso_file, 'rb' )
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    # secondary_id_list = [90000, 91000, 92000, 93000, 94000, 95000, 96000,
    #                      97000, 98000, 99000]
    
    # for secondary_id in secondary_id_list:
        
    #     t0_secondary = rso_dict[secondary_id]['epoch_tdb']
    #     Xo_secondary = rso_dict[secondary_id]['state']
        
    #     for ii in range(1,11):
            
    #         tertiary_id = secondary_id + ii
    #         Xo_tertiary = create_tertiary_object(Xo_secondary)
    #         mass = np.random.rand()*999.+1.
    #         A_m = np.random.rand()*0.09 + 0.01
    #         area = A_m*mass
    #         Cd = 2.2
    #         Cr = 1.3
            
    #         # Add to output
    #         rso_dict[tertiary_id] = {}
    #         rso_dict[tertiary_id]['epoch_tdb'] = t0_secondary
    #         rso_dict[tertiary_id]['state'] = Xo_tertiary
    #         rso_dict[tertiary_id]['mass'] = mass
    #         rso_dict[tertiary_id]['area'] = area
    #         rso_dict[tertiary_id]['Cd'] = Cd
    #         rso_dict[tertiary_id]['Cr'] = Cr
    
    
    tertiary_id_list = [92009]
    
    for tertiary_id in tertiary_id_list:
        
        secondary_id = int(np.floor(tertiary_id/10.)*10)
        print(secondary_id)
        
        t0_secondary = rso_dict[secondary_id]['epoch_tdb']
        Xo_secondary = rso_dict[secondary_id]['state']
        
        Xo_tertiary = create_tertiary_object(Xo_secondary)
        mass = np.random.rand()*999.+1.
        A_m = np.random.rand()*0.09 + 0.01
        area = A_m*mass
        Cd = 2.2
        Cr = 1.3
        
        # Add to output
        rso_dict[tertiary_id] = {}
        rso_dict[tertiary_id]['epoch_tdb'] = t0_secondary
        rso_dict[tertiary_id]['state'] = Xo_tertiary
        rso_dict[tertiary_id]['mass'] = mass
        rso_dict[tertiary_id]['area'] = area
        rso_dict[tertiary_id]['Cd'] = Cd
        rso_dict[tertiary_id]['Cr'] = Cr
        
    
    
    print(list(rso_dict.keys()))
    print(len(rso_dict))
    
    pklFile = open(rso_file, 'wb')
    pickle.dump([rso_dict], pklFile, -1)
    pklFile.close()
    
    return


###############################################################################
# Create Estimated Catalog
###############################################################################

def perturb_state_vector(Xo, P):
    
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6,))
    Xf = Xo + pert_vect.reshape(Xo.shape)
    
    return Xf


def propagate_twobody_orbit(Xo, tk_list, state_params, int_params):
    
    # Run integrator
    tout, Xout = metis_dyn.general_dynamics(Xo, tk_list, state_params, int_params)
    
    return tout, Xout


def filter_setup(Xo, tk_list, meas_types, sigma_dict):
    
    # # Retrieve latest EOP data from celestrak.com
    # eop_alldata = eop.get_celestrak_eop_alldata()
        
    # # Retrieve polar motion data from file
    # XYs_df = eop.get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = 3.986e14
    
    # Define integrator parameters
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = metis_dyn.ode_twobody
    
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'seconds'
    
    # Filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 2.
    
    # Sensor and measurement parameters
    sensor_id_list = ['UNSW Falcon']
    sensor_params = metis_sensors.define_sensors(sensor_id_list)
    # sensor_params['eop_alldata'] = eop_alldata
    # sensor_params['XYs_df'] = XYs_df
    
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = meas_types
        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
        
    # Propagate orbit
    tout, X_truth = propagate_twobody_orbit(Xo, tk_list, state_params, int_params)
        
    truth_dict = {}
    meas_dict = {}
    meas_dict['tk_list'] = []
    meas_dict['Yk_list'] = []
    meas_dict['sensor_id_list'] = []
    
    for kk in range(len(tk_list)):
        
        # UTC = tk_list[kk]
        epoch_tdb = tk_list[kk]
        # EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = X_truth[kk,:].reshape(6,1)
        truth_dict[tk_list[kk]] = Xk
        
        for sensor_id in sensor_id_list:
            # Yk = mfunc.compute_measurement(Xk, state_params, sensor_params,
            #                                sensor_id, UTC, EOP_data, XYs_df)
            
            # Use positions as measurements
            Yk = Xk[0:3].reshape(3,1)
            
            sigma_dict = sensor_params[sensor_id]['sigma_dict']
            for mtype in meas_types:
                ind = meas_types.index(mtype)
                Yk[ind] += np.random.randn()*sigma_dict[mtype]
            
            meas_dict['tk_list'].append(epoch_tdb)
            meas_dict['Yk_list'].append(Yk)
            meas_dict['sensor_id_list'].append(sensor_id)
            
    
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
    
    # Initial state for filter
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['X'] = perturb_state_vector(Xo, P)
    state_dict[tk_list[0]]['P'] = P
    
    
    return state_dict, meas_dict, params_dict, truth_dict


def run_filter(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):
    
    params_dict['int_params']['intfcn'] = metis_dyn.ode_twobody_stm
    filter_output, full_state_output = metis_est.ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)    
    # analysis.compute_orbit_errors(filter_output, full_state_output, truth_dict)
    
    t0 = sorted(list(state_dict.keys()))[0]
    Xo = filter_output[t0]['X']
    Po = filter_output[t0]['P']
    
    return Xo, Po


def create_estimated_catalog(rso_file, output_file):
    
    
    # Load RSO dict
    pklFile = open(rso_file, 'rb' )
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    # Initialize output
    estimated_rso_dict = {}
    
    # # Miss list
    # random_obj_ids = random_object_ids()
    # miss_list = []
    # for ii in range(10):
    #     miss_list.append(random_obj_ids[ii*6+3])
    #     miss_list.append(random_obj_ids[ii*6+4])
    #     miss_list.append(random_obj_ids[ii*6+5])
        

    # Measurement functions
    meas_fcn = metis_mfunc.H_inertial_xyz
    meas_types = ['x', 'y', 'z']
    
    # Loop over objects
    obj_id_list = list(rso_dict.keys())
    for obj_id in obj_id_list:
        
        # Retrieve object data
        epoch_tdb0 = rso_dict[obj_id]['epoch_tdb']    
        Xo_true = rso_dict[obj_id]['state']        
        Cd = float(rso_dict[obj_id]['Cd'])
        Cr = float(rso_dict[obj_id]['Cr'])
        mass = float(rso_dict[obj_id]['mass'])
        area = float(rso_dict[obj_id]['area'])
           
        # This setup yields about meter level position errors
        sigma_dict = {}
        sigma_dict['x'] = 100.
        sigma_dict['y'] = 100.
        sigma_dict['z'] = 100.    
        
        kep = cart2kep(Xo_true, 3.986e14)
        period = 2.*np.pi*np.sqrt(float(kep[0,0])**3/(3.986e14))    
        tsec = list(np.linspace(0., period, 20))
        
        print(tsec)
        
        tk_list = [epoch_tdb0 + sec for sec in tsec]
        
        state_dict, meas_dict, params_dict, truth_dict = \
            filter_setup(Xo_true, tk_list, meas_types, sigma_dict)
        
        Xo, Po = run_filter(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
        
        print('')
        print('obj_id', obj_id)
        print('Xo', Xo)
        print('Po', np.sqrt(np.diag(Po)))
        
        print('Xo - Xo_true', Xo - Xo_true)
        
        mistake
        
        
        if obj_id > 89000:
            Po *= 1000.
            # Xo = perturb_state_vector(Xo_true, Po)
            
        else:
            Po *= 100.
        
        
        
        
        
        
        
        # Store output
        estimated_rso_dict[obj_id] = {}
        estimated_rso_dict[obj_id]['epoch_tdb'] = epoch_tdb0
        estimated_rso_dict[obj_id]['state'] = Xo
        estimated_rso_dict[obj_id]['covar'] = Po
        estimated_rso_dict[obj_id]['mass'] = mass
        estimated_rso_dict[obj_id]['area'] = area
        estimated_rso_dict[obj_id]['Cd'] = Cd
        estimated_rso_dict[obj_id]['Cr'] = Cr
    
    
    print(estimated_rso_dict)
    
    pklFile = open( output_file, 'wb' )
    pickle.dump([estimated_rso_dict], pklFile, -1)
    pklFile.close()
    
    
    return


def generate_truth_data(rso_file, truth_file, tf_days, dt):
    
    # Load rso data
    pklFile = open(rso_file, 'rb')
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    int_params = {}
    int_params['tudat_integrator'] = 'dp7'
    int_params['step'] = dt
    
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create) 
    state_params = {}
    state_params['sph_deg'] = 20
    state_params['sph_ord'] = 20   
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create
    
    truth_dict = {}
    for obj_id in rso_dict:
        
        print('obj_id', obj_id)
        
        t0 = rso_dict[obj_id]['epoch_tdb']
        Xo = rso_dict[obj_id]['state']
        state_params['mass'] = rso_dict[obj_id]['mass']
        state_params['area'] = rso_dict[obj_id]['area']
        state_params['Cd'] = rso_dict[obj_id]['Cd']
        state_params['Cr'] = rso_dict[obj_id]['Cr']
        
        tvec = np.array([t0, t0+tf_days*86400.])
        tout, Xout = prop.propagate_orbit(Xo, tvec, state_params, int_params, bodies)
    
        truth_dict[obj_id] = {}
        truth_dict[obj_id]['t_truth'] = tout
        truth_dict[obj_id]['X_truth'] = Xout
        
    
    pklFile = open( truth_file, 'wb' )
    pickle.dump([truth_dict], pklFile, -1)
    pklFile.close()    
    
    return


###############################################################################
# Sensor Setup
###############################################################################

def define_sensors():
    
    sensor_dict = {}
    
    # TIRA tracking radar
    # Dieter Mehrholz "A Tracking and Imaging Radar System for Space Object 
    # Reconnaissance" (1996)
    latitude_rad = np.radians(50.6174)
    longitude_rad = np.radians(7.1308)
    height_m = 294.6
    beamwidth_rad = np.radians(0.45)
        
    # Constraints/Limits
    # Cerutti-Maori (SDC8) for range limit of 5000 km
    az_lim = [0., 2.*np.pi]  # rad
    el_lim = [5.*np.pi/180., np.pi/2.]  # rad
    rg_lim = [0., 5000.*1000.]   # m
    sun_el_mask = -np.pi  # rad
    
    # Measurement types and noise
    # Approximated from Cerruti-Maori (SDC8)
    meas_types = ['rg', 'az', 'el']
    sigma_dict = {}
    sigma_dict['rg'] = 10.                  # m
    # sigma_dict['rr'] = 0.01                   # m/s
    sigma_dict['az'] = np.radians(0.01)     # rad
    sigma_dict['el'] = np.radians(0.01)    # rad
    
    
    sensor_dict['TIRA'] = \
        sensor.define_radar_sensor(latitude_rad, longitude_rad, height_m,
                                   beamwidth_rad, az_lim, el_lim, rg_lim, 
                                   sun_el_mask, meas_types, sigma_dict)
    
    
    
    
    # ALTAIR radar
    # Reagan Test Site (Kwajalein Atoll)
    # Location from Vallado
    # Beamwidth from Abouzahra and Avent (1994) Table 1
    latitude_rad = np.radians(9.39)
    longitude_rad = np.radians(167.48)
    height_m = np.radians(62.7)
    beamwidth_rad = np.radians(1.1)  # 1.1 deg (UHF) or 2.8 deg (VHF)
    
    # Constraints/Limits
    # Vallado Table 4-3 for range limit of 4500 km
    az_lim = [0., 2.*np.pi]  # rad
    el_lim = [5.*np.pi/180., np.pi/2.]  # rad
    rg_lim = [0., 5000.*1000.]               # [0., 4500.*1000.]   # m
    sun_el_mask = -np.pi  # rad
    
    # Measurement types and noise
    # Range from Abouzahra, angles approximated from Vallado
    meas_types = ['rg', 'az', 'el']
    sigma_dict = {}
    sigma_dict['rg'] = 10.0   # 13.5 (Abouzahra)                  # m
    sigma_dict['az'] = np.radians(0.01)     # 0.03 (Vallado) # rad
    sigma_dict['el'] = np.radians(0.01)    # rad
    
    
    sensor_dict['ALTAIR'] = \
        sensor.define_radar_sensor(latitude_rad, longitude_rad, height_m,
                                   beamwidth_rad, az_lim, el_lim, rg_lim, 
                                   sun_el_mask, meas_types, sigma_dict)
        
        
    print(sensor_dict)
    
    sensor_file = os.path.join('data', 'sensor_data.pkl')
    pklFile = open( sensor_file, 'wb' )
    pickle.dump([sensor_dict], pklFile, -1)
    pklFile.close()
    
    
    return


###############################################################################
# Verification
###############################################################################

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


def test_estimated_catalog_metrics(rso_file, primary_id, secondary_id, tf_days,
                                   all_metrics=False):
    
    
    # Load RSO dict
    pklFile = open(rso_file, 'rb' )
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    
    
    # Compute TCA
    # Basic setup parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)    
        
    
    int_params = {}
    int_params['tudat_integrator'] = 'dp87'
    int_params['step'] = 10.
    int_params['max_step'] = 60.
    int_params['min_step'] = 1e-3
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12    
    
    rso1_params = {}    
    rso1_params['sph_deg'] = 20
    rso1_params['sph_ord'] = 20   
    rso1_params['central_bodies'] = ['Earth']
    rso1_params['bodies_to_create'] = bodies_to_create
    rso1_params['mass'] = rso_dict[primary_id]['mass']
    rso1_params['area'] = rso_dict[primary_id]['area']
    rso1_params['Cd'] = rso_dict[primary_id]['Cd']
    rso1_params['Cr'] = rso_dict[primary_id]['Cr']
       
    
    rso2_params = {}    
    rso2_params['sph_deg'] = 20
    rso2_params['sph_ord'] = 20   
    rso2_params['central_bodies'] = ['Earth']
    rso2_params['bodies_to_create'] = bodies_to_create
    rso2_params['mass'] = rso_dict[secondary_id]['mass']
    rso2_params['area'] = rso_dict[secondary_id]['area']
    rso2_params['Cd'] = rso_dict[secondary_id]['Cd']
    rso2_params['Cr'] = rso_dict[secondary_id]['Cr']
    
    t0 = rso_dict[primary_id]['epoch_tdb']
    tf = t0 + tf_days*86400.
    trange = np.array([t0, tf])
    
    X1_0 = rso_dict[primary_id]['state']    
    X2_0 = rso_dict[secondary_id]['state']
    
    
    
    T_list, rho_list = conj.compute_TCA(X1_0, X2_0, trange, rso1_params,
                                        rso2_params, int_params, bodies=bodies)
    
    print('')
    print('TCA_hrs', [(ti - t0)/3600. for ti in T_list])
    print(rho_list)
    
    TCA_hrs = (T_list[0]-t0)/3600.
    
    
    if all_metrics:
        P1_0 = rso_dict[primary_id]['covar']
        P2_0 = rso_dict[secondary_id]['covar']
    
        # Propagate state and covariances to TCA
        t_tca = T_list[0]
        tvec = np.array([t0, t_tca])
        tf, X1_f, P1_f = prop.propagate_state_and_covar(X1_0, P1_0, tvec, rso1_params, int_params, bodies=bodies, alpha=1e-4)
        tf, X2_f, P2_f = prop.propagate_state_and_covar(X2_0, P2_0, tvec, rso2_params, int_params, bodies=bodies, alpha=1e-4)
        
            
        # Compute miss distance, mahalanobis distance, Pc, Uc
        r_A = X1_f[0:3].reshape(3,1)
        r_B = X2_f[0:3].reshape(3,1)
        v_A = X1_f[3:6].reshape(3,1)
        v_B = X2_f[3:6].reshape(3,1)
        P_A = P1_f[0:3,0:3]
        P_B = P2_f[0:3,0:3]
        
        print(r_A)
        print(r_B)
        print(P_A)
        print(P_B)
        
        d2 = conj.compute_euclidean_distance(r_A, r_B)
        dM = conj.compute_mahalanobis_distance(r_A, r_B, P_A, P_B)
        rho_eci = r_B - r_A
        rho_ric = conj.eci2ric(r_A, v_A, rho_eci)
        vrel = np.linalg.norm(v_A - v_B)
        
        radius1 = np.sqrt(rso1_params['area']/np.pi)
        radius2 = np.sqrt(rso2_params['area']/np.pi)
        HBR = radius1+radius2
        
        Pc = conj.Pc2D_Foster(X1_f, P1_f, X2_f, P2_f, HBR, rtol=1e-8, HBR_type='circle')
        Uc = conj.Uc2D(X1_f, P1_f, X2_f, P2_f, HBR)
    
        
        # Print results
        print('')
        print('obj1', primary_id)
        print('obj2', secondary_id)
        print('X1', X1_f)
        print('X2', X2_f)
        print('P1', np.sqrt(np.diag(P1_f)))
        print('P2', np.sqrt(np.diag(P2_f)))
        
        print('')
        print('TCA [hrs]', TCA_hrs)
        print('miss distance', d2)
        print('mahalanobis distance', dM)
        print('relative velocity', vrel)
        print('Pc', Pc)
        print('Uc', Uc)
        
        
        return TCA_hrs, d2, dM, rho_ric, vrel, Pc, Uc
    
    # Propagate state to TCA
    t_tca = T_list[0]
    tvec = np.array([t0, t_tca])
    tout1, Xout1 = prop.propagate_orbit(X1_0, tvec, rso1_params, int_params, bodies=bodies)
    tout2, Xout2 = prop.propagate_orbit(X2_0, tvec, rso2_params, int_params, bodies=bodies)
    
    X1_f = Xout1[-1,:].reshape(6,1)
    X2_f = Xout2[-1,:].reshape(6,1)
        
    
    # Compute miss distance, mahalanobis distance, Pc, Uc
    r_A = X1_f[0:3].reshape(3,1)
    r_B = X2_f[0:3].reshape(3,1)
    v_A = X1_f[3:6].reshape(3,1)
    v_B = X2_f[3:6].reshape(3,1)

    
    d2 = conj.compute_euclidean_distance(r_A, r_B)
    rho_eci = r_B - r_A
    rho_ric = conj.eci2ric(r_A, v_A, rho_eci)
    vrel = np.linalg.norm(v_A - v_B)
    
    
    return TCA_hrs, d2, rho_ric, vrel


def generate_true_risk_metrics(rso_file, metrics_file, tf_days):
    
    
    pklFile = open(rso_file, 'rb' )
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    # obj_id_list = sorted(list(rso_dict.keys()))
    obj_id_list = [52373, 92009]
    
    with open(metrics_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Primary ID','Secondary ID', 'TCA [hours]', 
                         'Miss Distance [m]', 'Radial [m]',
                         'Tangential [m]', 'Normal [m]',
                         'Relative Velocity [m/s]'])
    
    
        primary_id = obj_id_list[0]
        
        for secondary_id in obj_id_list[1:]: 
            
            print('secondary id', secondary_id)
        
            TCA_hrs, d2, rho_ric, vrel = \
                test_estimated_catalog_metrics(rso_file, primary_id, secondary_id, tf_days,
                                               all_metrics=False)
                
            rdist = float(rho_ric[0,0])
            tdist = float(rho_ric[1,0])
            ndist = float(rho_ric[2,0])
            
            writer.writerow([primary_id, secondary_id, TCA_hrs, d2, rdist, tdist,
                             ndist, vrel])
            
    csvfile.close()
    
    
    
    return


def generate_visibility_dict(truth_file, sensor_file, visibility_file):
    
    # Load objects and sensors
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    truth_dict = data[0]
    pklFile.close()
    
    pklFile = open(sensor_file, 'rb' )
    data = pickle.load( pklFile )
    sensor_dict = data[0]
    pklFile.close()
    
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)  
    
    
    visibility_dict = sensor.compute_visible_passes2(truth_dict, sensor_dict,
                                                     bodies)
    
    
    print(visibility_dict)
    
    # Save visibility file
    pklFile = open(visibility_file, 'wb')
    pickle.dump([visibility_dict], pklFile, -1)
    pklFile.close()
    
    
    return


def compute_visibility_stats(rso_file, visibility_file, obj_id_list):
    
    
    pklFile = open(rso_file, 'rb' )
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    pklFile = open(visibility_file, 'rb' )
    data = pickle.load( pklFile )
    visibility_dict = data[0]
    pklFile.close()
    
    t0 = rso_dict[obj_id_list[0]]['epoch_tdb']
        
    for sensor_id in visibility_dict:
        
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(obj_id_list)))
        yval = 1
        fig, ax = plt.subplots()
        
        for obj_id in obj_id_list:
            
            # Check if any visible passes
            if obj_id not in visibility_dict[sensor_id]:
                print('\nSensor', sensor_id, 'has no visibile passes for', obj_id)
                continue
            
            tk_list = visibility_dict[sensor_id][obj_id]['tk_list']
            rg_list = visibility_dict[sensor_id][obj_id]['rg_list']
            az_list = visibility_dict[sensor_id][obj_id]['az_list']
            el_list = visibility_dict[sensor_id][obj_id]['el_list']
            
            pass_dict = sensor.compute_pass(tk_list, rg_list, az_list, el_list)
            start_list = pass_dict['start_list']
            stop_list = pass_dict['stop_list']
            TCA_list = pass_dict['TCA_list']
            TME_list = pass_dict['TME_list']
            rg_min_list = pass_dict['rg_min_list']
            el_max_list = pass_dict['el_max_list']
            
            npass = len(start_list)
            duration_list = [(stop - start) for stop, start in zip(stop_list, start_list)]
            
            print('\nSensor', sensor_id, 'observing object', obj_id)
            print('Number of passes:', npass)
            print('Pass Durations [sec]', duration_list)
            
            # Setup for plots
            thrs = [(tk - t0)/3600. for tk in tk_list]
            rg_km = [rg/1000. for rg in rg_list]
            az_deg = [np.degrees(az) for az in az_list]
            el_deg = [np.degrees(el) for el in el_list]
            
            TCA_inds = [tk_list.index(TCA) for TCA in TCA_list]
            
            
            # plt.figure()
            # plt.subplot(3,1,1)
            # plt.plot(thrs, rg_km, 'k.')
            # plt.ylabel('Range [km]')
            # plt.title('Sensor ' + sensor_id + ' observing object ' + str(obj_id))
            # plt.subplot(3,1,2)
            # plt.plot(thrs, az_deg, 'k.')
            # plt.ylabel('Az [deg]')
            # plt.subplot(3,1,3)
            # plt.plot(thrs, el_deg, 'k.')
            # plt.ylabel('El [deg]')
            # plt.xlabel('Time [hours]')
            
            
            
            # plot all together using different rows for each object and square markers
            plt.plot(thrs, [yval]*len(tk_list), 's', color=colors[yval-1])
            
            yval += 1
        
        ax.set_yticks(range(1,len(obj_id_list)+1), labels=[str(obj_id) for obj_id in obj_id_list])
        ax.set_xlabel('Time [hours]')
        ax.set_title('Visibility from ' + sensor_id)
        
            
            
    plt.show()
    
    return




###############################################################################
# Utilities
###############################################################################


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


def cart2kep(cart, GM):
    '''
    This function converts a Cartesian state vector in inertial frame to
    Keplerian orbital elements.
    
    Parameters
    ------
    cart : 6x1 numpy array
    
    Cartesian Coordinates (Inertial Frame)
    ------
    cart[0] : x
      Position in x               [km]
    cart[1] : y
      Position in y               [km]
    cart[2] : z
      Position in z               [km]
    cart[3] : dx
      Velocity in x               [km/s]
    cart[4] : dy
      Velocity in y               [km/s]
    cart[5] : dz
      Velocity in z               [km/s]
      
    Returns
    ------
    elem : 6x1 numpy array
    
    Keplerian Orbital Elements
    ------
    elem[0] : a
      Semi-Major Axis             [km]
    elem[1] : e
      Eccentricity                [unitless]
    elem[2] : i
      Inclination                 [deg]
    elem[3] : RAAN
      Right Asc Ascending Node    [deg]
    elem[4] : w
      Argument of Periapsis       [deg]
    elem[5] : theta
      True Anomaly                [deg]    
      
    '''
    
    # Retrieve input cartesian coordinates
    r_vect = cart[0:3].reshape(3,1)
    v_vect = cart[3:6].reshape(3,1)

    # Calculate orbit parameters
    r = np.linalg.norm(r_vect)
    ir_vect = r_vect/r
    v2 = np.linalg.norm(v_vect)**2
    h_vect = np.cross(r_vect, v_vect, axis=0)
    h = np.linalg.norm(h_vect)

    # Calculate semi-major axis
    a = 1./(2./r - v2/GM)     # km
    
    # Calculate eccentricity
    e_vect = np.cross(v_vect, h_vect, axis=0)/GM - ir_vect
    e = np.linalg.norm(e_vect)

    # Calculate RAAN and inclination
    ih_vect = h_vect/h

    RAAN = math.atan2(ih_vect[0,0], -ih_vect[1,0])   # rad
    i = math.acos(ih_vect[2,0])   # rad
    if RAAN < 0.:
        RAAN += 2.*math.pi

    # Apply correction for circular orbit, choose e_vect to point
    # to ascending node
    if e != 0:
        ie_vect = e_vect/e
    else:
        ie_vect = np.array([[math.cos(RAAN)], [math.sin(RAAN)], [0.]])

    # Find orthogonal unit vector to complete perifocal frame
    ip_vect = np.cross(ih_vect, ie_vect, axis=0)

    # Form rotation matrix PN
    PN = np.concatenate((ie_vect, ip_vect, ih_vect), axis=1).T

    # Calculate argument of periapsis
    w = math.atan2(PN[0,2], PN[1,2])  # rad
    if w < 0.:
        w += 2.*math.pi

    # Calculate true anomaly
    cross1 = np.cross(ie_vect, ir_vect, axis=0)
    tan1 = np.dot(cross1.T, ih_vect).flatten()[0]
    tan2 = np.dot(ie_vect.T, ir_vect).flatten()[0]
    theta = math.atan2(tan1, tan2)    # rad
    
    # Update range of true anomaly for elliptical orbits
    if a > 0. and theta < 0.:
        theta += 2.*math.pi
    
    # Convert angles to deg
    i *= 180./math.pi
    RAAN *= 180./math.pi
    w *= 180./math.pi
    theta *= 180./math.pi
    
    # Form output
    elem = np.array([[a], [e], [i], [RAAN], [w], [theta]])
      
    return elem


def kep2cart(elem, GM):
    '''
    This function converts a vector of Keplerian orbital elements to a
    Cartesian state vector in inertial frame.
    
    Parameters
    ------
    elem : 6x1 numpy array
    
    Keplerian Orbital Elements
    ------
    elem[0] : a
      Semi-Major Axis             [km]
    elem[1] : e
      Eccentricity                [unitless]
    elem[2] : i
      Inclination                 [deg]
    elem[3] : RAAN
      Right Asc Ascending Node    [deg]
    elem[4] : w
      Argument of Periapsis       [deg]
    elem[5] : theta
      True Anomaly                [deg]
      
      
    Returns
    ------
    cart : 6x1 numpy array
    
    Cartesian Coordinates (Inertial Frame)
    ------
    cart[0] : x
      Position in x               [km]
    cart[1] : y
      Position in y               [km]
    cart[2] : z
      Position in z               [km]
    cart[3] : dx
      Velocity in x               [km/s]
    cart[4] : dy
      Velocity in y               [km/s]
    cart[5] : dz
      Velocity in z               [km/s]  
      
    '''
    
    # Retrieve input elements, convert to radians
    a = float(elem[0,0])
    e = float(elem[1,0])
    i = float(elem[2,0]) * math.pi/180
    RAAN = float(elem[3,0]) * math.pi/180
    w = float(elem[4,0]) * math.pi/180
    theta = float(elem[5,0]) * math.pi/180

    # Calculate h and r
    p = a*(1 - e**2)
    h = np.sqrt(GM*p)
    r = p/(1. + e*math.cos(theta))

    # Calculate r_vect and v_vect
    r_vect = r * \
        np.array([[math.cos(RAAN)*math.cos(theta+w) - math.sin(RAAN)*math.sin(theta+w)*math.cos(i)],
                  [math.sin(RAAN)*math.cos(theta+w) + math.cos(RAAN)*math.sin(theta+w)*math.cos(i)],
                  [math.sin(theta+w)*math.sin(i)]])

    vv1 = math.cos(RAAN)*(math.sin(theta+w) + e*math.sin(w)) + \
          math.sin(RAAN)*(math.cos(theta+w) + e*math.cos(w))*math.cos(i)

    vv2 = math.sin(RAAN)*(math.sin(theta+w) + e*math.sin(w)) - \
          math.cos(RAAN)*(math.cos(theta+w) + e*math.cos(w))*math.cos(i)

    vv3 = -(math.cos(theta+w) + e*math.cos(w))*math.sin(i)
    
    v_vect = -GM/h * np.array([[vv1], [vv2], [vv3]])

    cart = np.concatenate((r_vect, v_vect), axis=0)
    
    return cart


if __name__ == '__main__':
    
    plt.close('all')
    
    # UTC_list, epoch_tdb_list = compute_UTC_from_TDB()
    
    # define_primary_object(epoch_tdb_list[0])

    # verify_numerical_error()
    
    rso_file = os.path.join('data', 'rso_catalog_truth.pkl')
    estimated_rso_file = os.path.join('data', 'estimated_rso_catalog.pkl')
    sensor_file = os.path.join('data', 'sensor_data.pkl')
    visibility_file = os.path.join('data', 'visibility_data.pkl')
    metrics_file = os.path.join('data', 'risk_metrics_truth5.csv')
    
    
    # build_truth_catalog(rso_file, 6)
    
    # create_tertiary_catalog(rso_file)
    
    tf_days = 7.
    # generate_true_risk_metrics(rso_file, metrics_file, tf_days)
    
    
    truth_file = os.path.join('data', 'propagated_truth_10sec.pkl')
    tf_days = 7.
    dt = 10.
    # generate_truth_data(rso_file, truth_file, tf_days, dt)
    
    # define_sensors()
    
    generate_visibility_dict(truth_file, sensor_file, visibility_file)

    # obj_id_list = [52373, 90000, 91000, 92000, 93000, 94000, 95000, 96000,
    #                97000, 98000, 99000]
    # compute_visibility_stats(rso_file, visibility_file, obj_id_list)


    # create_estimated_catalog(rso_file, estimated_rso_file)

    # 
    # test_estimated_catalog_metrics(estimated_rso_file, 52373, 91000, all_metrics=True)

    







