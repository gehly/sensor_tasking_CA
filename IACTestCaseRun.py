import numpy as np
import math
from datetime import datetime, timedelta
import os
import pickle
import matplotlib.pyplot as plt
import time
import bisect

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
import EstimationUtilities as est
import AnalysisFunctions as analysis


###############################################################################
# Baseline Scenario
#
# Assume all sensors can provide measurements on all objects whenever they are
# in view. This provides the most accurate state estimates possible given the
# sensor constraints, observation geometry, etc. to function as a baseline for
# comparison of more realistic sensor tasking that can only observe one object
# at a time.
#
###############################################################################

def generate_baseline_measurements(rso_file, sensor_file, visibility_file,
                                   truth_file, meas_file):
    
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
    
    body_settings = environment_setup.get_default_body_settings(
        ["Earth"],
        "Earth",
        "J2000")
    bodies = environment_setup.create_system_of_bodies(body_settings)
    
    
    t0 = rso_dict[52373]['epoch_tdb']
    TCA_dict = {}
    TCA_dict[52373] = np.inf
    TCA_dict[90000] = t0 + 30.*3600.
    TCA_dict[91000] = t0 + 42.*3600.
    TCA_dict[92000] = t0 + 60.*3600.
    TCA_dict[93000] = t0 + 80.*3600.
    TCA_dict[94000] = t0 + 97.*3600.
    TCA_dict[95000] = t0 + 98.*3600.
    TCA_dict[96000] = t0 + 99.*3600.
    TCA_dict[97000] = t0 + 125.*3600.
    TCA_dict[98000] = t0 + 145.*3600.
    TCA_dict[99000] = t0 + 162.*3600.
    
    obj_id_list = [52373, 90000, 91000, 92000, 93000, 94000, 95000, 96000,
                   97000, 98000, 99000]
    
    # obj_id_list = [52373]
    
    
    # Initialize output
    meas_dict = {}
    
    # Parse visibility dict, add noise to measurements and store
    for sensor_id in visibility_dict:
        
        sensor_params = sensor_dict[sensor_id]
        meas_types = sensor_dict[sensor_id]['meas_types']
        sigma_dict = sensor_dict[sensor_id]['sigma_dict']
        
        # for obj_id in visibility_dict[sensor_id]:
        for obj_id in obj_id_list:
            tk_list = visibility_dict[sensor_id][obj_id]['tk_list']
            rg_list = visibility_dict[sensor_id][obj_id]['rg_list']
            az_list = visibility_dict[sensor_id][obj_id]['az_list']
            el_list = visibility_dict[sensor_id][obj_id]['el_list']
            
            if obj_id not in meas_dict:
                meas_dict[obj_id] = {}
                meas_dict[obj_id]['tk_list'] = []
                meas_dict[obj_id]['Yk_list'] = []
                meas_dict[obj_id]['sensor_id_list'] = []
                
            for kk in range(len(tk_list)):
                tk = tk_list[kk]
                
                # Check if tk is past TCA
                if tk > (TCA_dict[obj_id]-3600.):
                    break   
                
                # Retrieve object state at this time
                tk_truth = truth_dict[obj_id]['t_truth']
                Xk_truth = truth_dict[obj_id]['X_truth']
                ind = list(tk_truth).index(tk)
                Xk = Xk_truth[ind,:].reshape(6,1)
                
                # Compute measurement and add noise
                Yk = sensor.compute_measurement(tk, Xk, sensor_params, bodies)
                
                # Add noise
                for ii in range(len(meas_types)):
                    meas = meas_types[ii]
                    Yk[ii] += np.random.randn()*sigma_dict[meas]
                
                
                
                # Yk = np.array([[rg_list[kk] + np.random.randn()*sigma_dict['rg']],
                #                [az_list[kk] + np.random.randn()*sigma_dict['az']],
                #                [el_list[kk] + np.random.randn()*sigma_dict['el']]])
                
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
                    
                    
    
    # Plot measurements for verification
    for obj_id in meas_dict:
        t0 = rso_dict[obj_id]['epoch_tdb']
        thrs = [(tk - t0)/3600. for tk in meas_dict[obj_id]['tk_list']]
        rg_km = [Yk[0]/1000. for Yk in meas_dict[obj_id]['Yk_list']]
        az_deg = [np.degrees(Yk[1]) for Yk in meas_dict[obj_id]['Yk_list']]
        el_deg = [np.degrees(Yk[2]) for Yk in meas_dict[obj_id]['Yk_list']]
        sensor_id_list = [1]*len(meas_dict[obj_id]['sensor_id_list'])
        for ii in range(len(meas_dict[obj_id]['sensor_id_list'])):
            if meas_dict[obj_id]['sensor_id_list'][ii] == 'ALTAIR':
                sensor_id_list[ii] = 2
                
        plt.figure()
        plt.subplot(4,1,1)
        plt.plot(thrs, rg_km, 'k.')
        plt.ylabel('Range [km]')
        plt.title('Measurements for Object ' + str(obj_id))
        plt.subplot(4,1,2)
        plt.plot(thrs, az_deg, 'k.')
        plt.ylabel('Az [deg]')
        plt.subplot(4,1,3)
        plt.plot(thrs, el_deg, 'k.')
        plt.ylabel('El [deg]')
        plt.subplot(4,1,4)
        plt.plot(thrs, sensor_id_list, 'ks')
        plt.ylabel('Sensor')
        plt.xlabel('Time [hours]')
        
        plt.figure()
        plt.plot(list(range(len(thrs))), thrs, 'k.')
        plt.ylabel('thrs')       
        
    plt.show()
    
    # Save measurement data
    pklFile = open( meas_file, 'wb' )
    pickle.dump([meas_dict], pklFile, -1)
    pklFile.close()
    
    return


# def generate_greedy_measurements(rso_file, sensor_file, visibility_file,
#                                  truth_file, meas_file, reward_fcn):
    
    
#     # Load rso data
#     pklFile = open(rso_file, 'rb')
#     data = pickle.load( pklFile )
#     rso_dict = data[0]
#     pklFile.close()
    
#     t0_all = rso_dict[52373]['epoch_tdb']
    
#     # Load sensor and visibility data
#     pklFile = open(sensor_file, 'rb')
#     data = pickle.load( pklFile )
#     sensor_dict = data[0]
#     pklFile.close()
    
#     pklFile = open(visibility_file, 'rb')
#     data = pickle.load( pklFile )
#     visibility_dict = data[0]
#     pklFile.close()    
    
#     pklFile = open(truth_file, 'rb')
#     data = pickle.load( pklFile )
#     truth_dict = data[0]
#     pklFile.close()
    
    
    
#     # Parse visibility dict to generate time based visibility dict
#     time_based_visibility = {}
#     for sensor_id in visibility_dict:
#         for obj_id in visibility_dict[sensor_id]:
#             tk_list = visibility_dict[sensor_id][obj_id]['tk_list']
            
#             for tk in tk_list:
#                 if tk not in time_based_visibility:
#                     time_based_visibility[tk] = {}                    
                    
#                 if sensor_id not in time_based_visibility[tk]:
#                     time_based_visibility[tk][sensor_id] = []
                
#                 # Append object IDs visible to this sensor
#                 time_based_visibility[tk][sensor_id].append(obj_id)
    
    
#     # Process data in 1 day increments
#     meas_dict = {}
#     for day in range(6,7):      
        
#         # Load data if needed
#         if day > 0:
#             pklFile = open(meas_file, 'rb')
#             data = pickle.load( pklFile )
#             meas_dict = data[0]
#             rso_dict = data[1]
#             pklFile.close()
        
#         # Reduce visibility dict to time window of interest
#         t0_interval = t0_all + day*86400.
#         # tf_interval = t0_interval + 86400.
        
#         tk_vis = np.array([])
#         tk_coarse = np.array([])
#         for hr in range(0,24):
#             tk_hr = np.arange(t0_interval+hr*3600., t0_interval+hr*3600. + 600., 10.)
#             tk_vis = np.append(tk_vis, tk_hr)
            
#             tk_hr2 = np.arange(t0_interval+hr*3600., t0_interval+hr*3600. + 600., 60.)
#             tk_coarse = np.append(tk_coarse, tk_hr2)
            
#         print(tk_vis)
#         print(len(tk_vis))
        
#         tk_list_coarse = []
#         visibility_dict_interval = {}
#         for tk in sorted(list(time_based_visibility.keys())):
#             # if tk >= t0_interval and tk < tf_interval and math.fmod((tk-t0_interval),60)==0:
            
#             # print(tk)
#             # print(tk_vis[0])
#             # mistake
                
#             if tk in tk_vis:
#                 visibility_dict_interval[tk] = time_based_visibility[tk]
                
#                 if tk in tk_coarse:
#                     tk_list_coarse.append(tk)
                
                
#         tk_check = sorted(list(visibility_dict_interval.keys()))
#         print((tk_check[-1] - tk_check[0]))
        
#         print((tk_check[0] - t0_all))
#         print((tk_check[-1] - t0_all))
        
#         print('tk vis', tk_check[0:20])
#         print('tk coarse', tk_list_coarse[0:20])
#         print('tk vis', len(tk_check))
#         print('tk coarse', len(tk_list_coarse))
#         print('obj id list', sorted(list(meas_dict.keys())))
#         print('nobj', len(meas_dict))
        
#         # mistake
        
                
#         # Process data to generate measurements and updated state catalog
#         meas_dict, rso_dict = \
#             sensor.greedy_sensor_tasking_multistep(rso_dict, sensor_dict,
#                                                    visibility_dict_interval, 
#                                                    visibility_dict, truth_dict, 
#                                                    meas_dict, reward_fcn, tk_list_coarse)
    

    
#         # Save measurement data
#         pklFile = open( meas_file, 'wb' )
#         pickle.dump([meas_dict, rso_dict], pklFile, -1)
#         pklFile.close()
    
    
#     return


def generate_greedy_measurements_tif(rso_file, sensor_file, visibility_file,
                                     truth_file, meas_file, reward_fcn):
    
    
    # Load rso data
    pklFile = open(rso_file, 'rb')
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    t0_all = rso_dict[52373]['epoch_tdb']
    
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
    
    # Form initial target priorities
    fixed_tif = True
    
    tif_base = 1.0
    tif_high = 1.0
    
    primary_id = 52373
    secondary_id_list = [90000, 91000, 92000, 93000, 94000, 95000, 96000, 97000,
                         98000, 99000]
    
    TCA_dict = {}
    TCA_dict[90000] = t0_all + 30.*3600.
    TCA_dict[91000] = t0_all + 42.*3600.
    TCA_dict[92000] = t0_all + 60.*3600.
    TCA_dict[93000] = t0_all + 80.*3600.
    TCA_dict[94000] = t0_all + 97.*3600.
    TCA_dict[95000] = t0_all + 98.*3600.
    TCA_dict[96000] = t0_all + 99.*3600.
    TCA_dict[97000] = t0_all + 125.*3600.
    TCA_dict[98000] = t0_all + 145.*3600.
    TCA_dict[99000] = t0_all + 162.*3600.
    
    for obj_id in rso_dict:
        if obj_id == primary_id or obj_id in secondary_id_list:
            rso_dict[obj_id]['tif'] = tif_high
        else:
            rso_dict[obj_id]['tif'] = tif_base    
    
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
    
    
    # Process data in 1 day increments
    meas_dict = {}
    for day in range(5,7):      
        
        # Load data if needed
        if day > 0:
            pklFile = open(meas_file, 'rb')
            data = pickle.load(pklFile)
            meas_dict = data[0]
            rso_dict = data[1]
            pklFile.close()
                           
        # print(len(rso_dict))
        # mistake
        
        # Reduce visibility dict to time window of interest
        t0_interval = t0_all + day*86400.
        # tf_interval = t0_interval + 86400.
        
        tk_vis = np.array([])
        tk_coarse = np.array([])
        for hr in range(0,24):
            tk_hr = np.arange(t0_interval+hr*3600., t0_interval+hr*3600. + 600., 10.)
            tk_vis = np.append(tk_vis, tk_hr)
            
            tk_hr2 = np.arange(t0_interval+hr*3600., t0_interval+hr*3600. + 600., 60.)
            tk_coarse = np.append(tk_coarse, tk_hr2)
            
        print(tk_vis)
        print(len(tk_vis))
        
        tk_list_coarse = []
        visibility_dict_interval = {}
        for tk in sorted(list(time_based_visibility.keys())):
            # if tk >= t0_interval and tk < tf_interval and math.fmod((tk-t0_interval),60)==0:
            
            # print(tk)
            # print(tk_vis[0])
            # mistake
                
            if tk in tk_vis:
                visibility_dict_interval[tk] = time_based_visibility[tk]
                
                if tk in tk_coarse:
                    tk_list_coarse.append(tk)
                
                
        tk_check = sorted(list(visibility_dict_interval.keys()))
        print((tk_check[-1] - tk_check[0]))
        
        print((tk_check[0] - t0_all))
        print((tk_check[-1] - t0_all))
        
        print('tk vis', tk_check[0:20])
        print('tk coarse', tk_list_coarse[0:20])
        print('tk vis', len(tk_check))
        print('tk coarse', len(tk_list_coarse))
        print('obj id list', sorted(list(meas_dict.keys())))
        print('nobj', len(meas_dict))
        
        # mistake
                
        # Process data to generate measurements and updated state catalog
        meas_dict, rso_dict = \
            sensor.greedy_sensor_tasking_multistep_tif(rso_dict, sensor_dict,
                                                       visibility_dict_interval, 
                                                       visibility_dict, truth_dict, 
                                                       meas_dict, reward_fcn,
                                                       tk_list_coarse, TCA_dict,
                                                       tif_base, fixed_tif)
    

    
        # Save measurement data
        pklFile = open( meas_file, 'wb' )
        pickle.dump([meas_dict, rso_dict], pklFile, -1)
        pklFile.close()
    
    
    return


def filter_process_measurements(rso_file, sensor_file, meas_file, output_file,
                                obj_id_list=[]):
    
    
    # Load rso data
    pklFile = open(rso_file, 'rb')
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    # Load sensor data
    pklFile = open(sensor_file, 'rb')
    data = pickle.load( pklFile )
    sensor_dict = data[0]
    pklFile.close()    
    
    # Load measurement data
    pklFile = open(meas_file, 'rb')
    data = pickle.load( pklFile )
    meas_dict = data[0]
    # rso_dict2 = data[1]
    pklFile.close() 
    
    # for obj_id in rso_dict2:
    #     print('')
    #     print(obj_id)
    #     print(rso_dict2[obj_id]['state'])
        
    # mistake
    
    # Standard data
    filter_params = {}
    filter_params['Qeci'] = 1e-13*np.diag([1., 1., 1.])
    filter_params['Qric'] = 0*np.diag([1., 1., 1.])
    filter_params['alpha'] = 1e-2
    
    
    int_params = {}
    int_params['tudat_integrator'] = 'dp87'
    int_params['step'] = 10.
    int_params['max_step'] = 100.
    int_params['min_step'] = 1e-3
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12 
    
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)   
    state_params = {}    
    state_params['sph_deg'] = 20
    state_params['sph_ord'] = 20   
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create
    
    # Loop over objects
    output_dict = {}
    if len(obj_id_list) == 0:
        obj_id_list = sorted(list(meas_dict.keys()))

    for obj_id in obj_id_list:
        
        if obj_id not in meas_dict:
            continue
        
        print('')
        print('obj_id', obj_id)
        t0 = rso_dict[obj_id]['epoch_tdb']
        
        if obj_id == 95000:
            filter_params['gap_seconds'] = 1e6
        else:
            filter_params['gap_seconds'] = 900.  
        
        # Retrieve state parameters
        state_params['epoch_tdb'] = rso_dict[obj_id]['epoch_tdb']
        state_params['state'] = rso_dict[obj_id]['state']
        state_params['covar'] = rso_dict[obj_id]['covar']
        state_params['mass'] = rso_dict[obj_id]['mass']
        state_params['area'] = rso_dict[obj_id]['area']
        state_params['Cd'] = rso_dict[obj_id]['Cd']
        state_params['Cr'] = rso_dict[obj_id]['Cr']
        
        # if obj_id == 52373:
        #     state_params['covar'] *= 100.
        # else:
        #     state_params['covar'] *= 0.01
        
        # Retrieve measurement data
        # filter_meas_dict = meas_dict[obj_id]
        
        # Reduce time window
        tk_max = t0 + 7.*24.*3600.
        tk_list = meas_dict[obj_id]['tk_list']
        Yk_list = meas_dict[obj_id]['Yk_list']
        sensor_id_list = meas_dict[obj_id]['sensor_id_list']
        ind = bisect.bisect_right(tk_list, tk_max)
        
        filter_meas_dict = {}
        filter_meas_dict['tk_list'] = tk_list[0:ind]
        filter_meas_dict['Yk_list'] = Yk_list[0:ind]
        filter_meas_dict['sensor_id_list'] = sensor_id_list[0:ind]
        
        
        # Run filter
        filter_output = est.ukf2(state_params, filter_meas_dict, sensor_dict,
                                 int_params, filter_params, bodies)
        
        output_dict[obj_id] = filter_output
        
        
    # Save output
    pklFile = open( output_file, 'wb' )
    pickle.dump([output_dict], pklFile, -1)
    pklFile.close()
    
    
    return


def filter_process_meas_and_save(rso_file, sensor_file, meas_file, output_file,
                                 truth_file):
    
    # risk_object_list = [52373, 90000, 91000, 92000, 93000, 94000, 95000,
    #                     96000, 97000, 98000, 99000]
    
    risk_object_list = []
    
    # Load rso data
    pklFile = open(rso_file, 'rb')
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    # Load sensor data
    pklFile = open(sensor_file, 'rb')
    data = pickle.load( pklFile )
    sensor_dict = data[0]
    pklFile.close()    
    
    # Load measurement data
    pklFile = open(meas_file, 'rb')
    data = pickle.load( pklFile )
    meas_dict = data[0]
    rso_dict2 = data[1]
    pklFile.close() 
    
    pklFile = open(truth_file, 'rb')
    data = pickle.load( pklFile )
    truth_dict = data[0]
    pklFile.close()   
    
    # Standard data
    filter_params = {}
    filter_params['Qeci'] = 1e-13*np.diag([1., 1., 1.])
    filter_params['Qric'] = 0*np.diag([1., 1., 1.])
    filter_params['alpha'] = 1e-2
    filter_params['gap_seconds'] = 900.
    
    int_params = {}
    int_params['tudat_integrator'] = 'dp87'
    int_params['step'] = 10.
    int_params['max_step'] = 100.
    int_params['min_step'] = 1e-3
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12 
    
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)   
    state_params = {}    
    state_params['sph_deg'] = 20
    state_params['sph_ord'] = 20   
    state_params['central_bodies'] = ['Earth']
    state_params['bodies_to_create'] = bodies_to_create
    
    # Loop over objects
    output_dict = {}
    obj_id_list = sorted(list(meas_dict.keys()))
    for obj_id in obj_id_list:
        
        if obj_id not in meas_dict:
            continue
        
        print('')
        print('obj_id', obj_id)
        t0 = rso_dict[obj_id]['epoch_tdb']
        
        if obj_id == 95000:
            filter_params['gap_seconds'] = 1e6
        else:
            filter_params['gap_seconds'] = 900. 
        
        # Retrieve state parameters
        state_params['epoch_tdb'] = rso_dict[obj_id]['epoch_tdb']
        state_params['state'] = rso_dict[obj_id]['state']
        state_params['covar'] = rso_dict[obj_id]['covar']
        state_params['mass'] = rso_dict[obj_id]['mass']
        state_params['area'] = rso_dict[obj_id]['area']
        state_params['Cd'] = rso_dict[obj_id]['Cd']
        state_params['Cr'] = rso_dict[obj_id]['Cr']
        
        # if obj_id == 52373:
        #     state_params['covar'] *= 100.
        # else:
        #     state_params['covar'] *= 0.01
        
        # Retrieve measurement data
        # filter_meas_dict = meas_dict[obj_id]
        
        # Reduce time window
        tk_max = t0 + 7.*24.*3600.
        tk_list = meas_dict[obj_id]['tk_list']
        Yk_list = meas_dict[obj_id]['Yk_list']
        sensor_id_list = meas_dict[obj_id]['sensor_id_list']
        ind = bisect.bisect_right(tk_list, tk_max)
        
        filter_meas_dict = {}
        filter_meas_dict['tk_list'] = tk_list[0:ind]
        filter_meas_dict['Yk_list'] = Yk_list[0:ind]
        filter_meas_dict['sensor_id_list'] = sensor_id_list[0:ind]
        
        
        # Run filter
        filter_output = est.ukf2(state_params, filter_meas_dict, sensor_dict,
                                 int_params, filter_params, bodies)
        
        output_dict[obj_id] = filter_output
        
        tk_filter = sorted(list(filter_output.keys()))
        tf_filter = tk_filter[-1]
        Xf_filter = filter_output[tf_filter]['state']
        Pf_filter = filter_output[tf_filter]['covar']
        
        if obj_id in risk_object_list:        
        
            rso_dict2[obj_id]['epoch_tdb'] = tf_filter
            rso_dict2[obj_id]['state'] = Xf_filter
            rso_dict2[obj_id]['covar'] = Pf_filter
            
        else:
            
            print(obj_id)
            t_truth = truth_dict[obj_id]['t_truth']
            X_truth = truth_dict[obj_id]['X_truth']
            truth_ind = list(t_truth).index(tf_filter)
                
            Xf_true = X_truth[truth_ind,:].reshape(6,1)
            
            rso_dict2[obj_id]['epoch_tdb'] = tf_filter
            rso_dict2[obj_id]['state'] = Xf_true
            rso_dict2[obj_id]['covar'] = Pf_filter
        
        
    # Save output
    pklFile = open( output_file, 'wb' )
    pickle.dump([output_dict], pklFile, -1)
    pklFile.close()
    
    pklFile = open( meas_file, 'wb' )
    pickle.dump([meas_dict, rso_dict2], pklFile, -1)
    pklFile.close()
    
    
    return


# def batch_process_baseline_measurements(rso_file, sensor_file, meas_file,
#                                         output_file, window_hrs):
    
    
#     # Load rso data
#     pklFile = open(rso_file, 'rb')
#     data = pickle.load( pklFile )
#     rso_dict = data[0]
#     pklFile.close()
    
#     # Load sensor data
#     pklFile = open(sensor_file, 'rb')
#     data = pickle.load( pklFile )
#     sensor_dict = data[0]
#     pklFile.close()    
    
#     # Load measurement data
#     pklFile = open(meas_file, 'rb')
#     data = pickle.load( pklFile )
#     meas_dict = data[0]
#     pklFile.close() 
    
#     # Standard data
#     filter_params = {}
#     filter_params['Qeci'] = 1e-13*np.diag([1., 1., 1.])
#     filter_params['Qric'] = 0*np.diag([1., 1., 1.])
#     filter_params['alpha'] = 1e-2
#     filter_params['gap_seconds'] = 600.
    
    
#     int_params = {}
#     int_params['tudat_integrator'] = 'dp87'
#     int_params['step'] = 10.
#     int_params['max_step'] = 100.
#     int_params['min_step'] = 1e-3
#     int_params['rtol'] = 1e-12
#     int_params['atol'] = 1e-12 
    
#     bodies_to_create = ['Sun', 'Earth', 'Moon']
#     bodies = prop.tudat_initialize_bodies(bodies_to_create)   
#     state_params = {}    
#     state_params['sph_deg'] = 20
#     state_params['sph_ord'] = 20   
#     state_params['central_bodies'] = ['Earth']
#     state_params['bodies_to_create'] = bodies_to_create
    
#     # Loop over objects
#     output_dict = {}
#     full_output_dict = {}
#     obj_id_list = list(meas_dict.keys())
#     for obj_id in obj_id_list:
        
        
#         t0 = rso_dict[obj_id]['epoch_tdb']
#         output_dict[obj_id] = {}
#         full_output_dict[obj_id] = {}
        
#         # Retrieve state parameters
#         state_params['epoch_tdb'] = rso_dict[obj_id]['epoch_tdb']
#         state_params['state'] = rso_dict[obj_id]['state']
#         state_params['covar'] = rso_dict[obj_id]['covar']
#         state_params['mass'] = rso_dict[obj_id]['mass']
#         state_params['area'] = rso_dict[obj_id]['area']
#         state_params['Cd'] = rso_dict[obj_id]['Cd']
#         state_params['Cr'] = rso_dict[obj_id]['Cr']
        
#         # if obj_id == 52373:
#         #     state_params['covar'] *= 100.
#         # else:
#         #     state_params['covar'] *= 0.01
        
#         # Retrieve measurement data
#         # filter_meas_dict = meas_dict[obj_id]
        
        
#         tk_list = meas_dict[obj_id]['tk_list']
#         Yk_list = meas_dict[obj_id]['Yk_list']
#         sensor_id_list = meas_dict[obj_id]['sensor_id_list']
        
#         # Loop over time in blocks
#         tk_max = 0
#         while tk_max < tk_list[-1]:
        
#             # Reduce time window
#             tk_max = t0 + window_hrs*3600.
        
#             ind_0 = bisect.bisect_left(tk_list, t0)
#             ind_f = bisect.bisect_right(tk_list, tk_max)
            
#             print('')
#             print('obj_id', obj_id)
#             print('t0', t0)
#             print('tk_max', tk_max)
#             print('dt_hrs', (tk_max-t0)/3600.)
#             print('ind_0', ind_0)
#             print('ind_f', ind_f)
            
#             # if tk_max > rso_dict[obj_id]['epoch_tdb'] + window_hrs*3600.:
#             #     mistake
            
        
#             filter_meas_dict = {}
#             filter_meas_dict['tk_list'] = tk_list[ind_0:ind_f]
#             filter_meas_dict['Yk_list'] = Yk_list[ind_0:ind_f]
#             filter_meas_dict['sensor_id_list'] = sensor_id_list[ind_0:ind_f]
        
#             # Set tk_output in filter_params
#             filter_params['tk_output'] = list(np.arange(t0, tk_max+1., 10.))
            
        
#             # Run filter
#             filter_output, full_output = \
#                 est.unscented_batch(state_params, filter_meas_dict,
#                                     sensor_dict, int_params,
#                                     filter_params, bodies)
        
#             output_dict[obj_id].update(filter_output)
#             full_output_dict[obj_id].update(full_output)
            
#             # Update for next iteration
#             t0 += window_hrs*3600.
#             state_params['epoch_tdb'] = t0
#             state_params['state'] = full_output[t0]['state'] + np.array([[10.], [10.], [10.], [1e-2], [1e-2], [1e-2]])
#             state_params['covar'] = np.diag([1e6, 1e6, 1e6, 1, 1, 1])
#             # state_params['covar'] *= 100.
            
#             # if tk_max > rso_dict[obj_id]['epoch_tdb'] + 20*3600.:
#             #     break
        
        
#         # Save output
#         pklFile = open( output_file, 'wb' )
#         pickle.dump([output_dict, full_output_dict], pklFile, -1)
#         pklFile.close()
    
    
#     return


def process_filter_output(output_file, truth_file):
    
    pklFile = open(output_file, 'rb')
    data = pickle.load( pklFile )
    output_dict = data[0]
    pklFile.close()
    
    pklFile = open(truth_file, 'rb')
    data = pickle.load( pklFile )
    truth_dict = data[0]
    pklFile.close()
    
    for obj_id in output_dict:
        
        analysis.compute_errors(truth_dict, output_dict, obj_id)
    
    
    return


# def process_baseline_batch_output(output_file, truth_file):
    
#     pklFile = open(output_file, 'rb')
#     data = pickle.load( pklFile )
#     output_dict = data[0]
#     full_output_dict = data[1]
#     pklFile.close()
    
#     pklFile = open(truth_file, 'rb')
#     data = pickle.load( pklFile )
#     truth_dict = data[0]
#     pklFile.close()
    
#     for obj_id in output_dict:
        
#         analysis.compute_batch_errors(truth_dict, output_dict, full_output_dict, obj_id)
    
    
#     return


def process_cdm_output(rso_file, est_output_file, cdm_file):
    
    pklFile = open(est_output_file, 'rb')
    data = pickle.load( pklFile )
    output_dict = data[0]
    pklFile.close()
    
    pklFile = open(rso_file, 'rb')
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
        
    primary_id = 52373
    t0 = rso_dict[primary_id]['epoch_tdb']
    tf = t0 + 7.*86400.
    
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create)
    
    cdm_dict = analysis.risk_metric_evolution(cdm_file, output_dict, rso_dict,
                                              primary_id, tf, bodies)
    
    
    # Save output
    pklFile = open( cdm_file, 'wb' )
    pickle.dump([cdm_dict], pklFile, -1)
    pklFile.close()
    
    return


def generate_case_summary(meas_file, output_file, truth_file):
    
    pklFile = open(meas_file, 'rb')
    data = pickle.load( pklFile )
    meas_dict = data[0]
    pklFile.close()
    
    pklFile = open(output_file, 'rb')
    data = pickle.load( pklFile )
    output_dict = data[0]
    pklFile.close()
    
    pklFile = open(truth_file, 'rb')
    data = pickle.load( pklFile )
    truth_dict = data[0]
    pklFile.close()
    
    
    # Number of measurements
    primary_id = 52373
    secondary_id_list = [90000, 91000, 92000, 93000, 94000, 95000, 96000,
                         97000, 98000, 99000]
    nobj_detected = len(meas_dict)    
    nmeas_secondary = 0
    nmeas_tertiary = 0
    for obj_id in meas_dict:
        if obj_id == primary_id:
            nmeas_primary = len(meas_dict[obj_id]['tk_list'])
        elif obj_id in secondary_id_list:
            nmeas_secondary += len(meas_dict[obj_id]['tk_list'])
        else:
            nmeas_tertiary += len(meas_dict[obj_id]['tk_list'])
            
    
    
    
    
    # Plot all 3D position errors
    plt.figure()
    obj_id_list = sorted(list(output_dict))
    print(obj_id_list)

    diverge_list = []
    for obj_id in obj_id_list:
        if obj_id == primary_id or obj_id in secondary_id_list:
            continue
        
        thrs, pos3D = analysis.compute_errors(truth_dict, output_dict, obj_id, False)
        if max(pos3D) > 1e4:
            diverge_list.append(obj_id)
    
        plt.semilogy(thrs, pos3D, color='0.8')
        
    for obj_id in secondary_id_list:
        if obj_id not in obj_id_list:
            continue
        
        thrs, pos3D = analysis.compute_errors(truth_dict, output_dict, obj_id, False)
        if max(pos3D) > 1e4:
            diverge_list.append(obj_id)
    
        plt.semilogy(thrs, pos3D, color='b')
        
        
        
    
    thrs, pos3D = analysis.compute_errors(truth_dict, output_dict, primary_id, False)

    plt.semilogy(thrs, pos3D, color='r')
    
    plt.xlabel('Time [hours]')
    plt.ylabel('3D Pos Error [m]')
    plt.show()
    
    nmeas_total = nmeas_primary + nmeas_secondary + nmeas_tertiary
    
    
    print('')
    print(meas_file)
    print('Objects and Measurements')
    print('Num Objects Detected', nobj_detected)
    print('Total Num Meas', nmeas_total)
    print('Num Meas Primary (Starlink)', nmeas_primary, nmeas_primary/nmeas_total*100.)
    print('Num Meas Secondary', nmeas_secondary, nmeas_secondary/nmeas_total*100.)
    print('Num Meas Tertiary', nmeas_tertiary, nmeas_tertiary/nmeas_total*100.)
    print('Bad object list', diverge_list)

    return


def plot_risk_metrics(baseline_cdm_file, greedy_cdm_file, priority_cdm_file, truth_file):
    
    pklFile = open(baseline_cdm_file, 'rb')
    data = pickle.load( pklFile )
    baseline_cdm_dict = data[0]
    pklFile.close()
    
    pklFile = open(greedy_cdm_file, 'rb')
    data = pickle.load( pklFile )
    greedy_cdm_dict = data[0]
    pklFile.close()
    
    pklFile = open(priority_cdm_file, 'rb')
    data = pickle.load( pklFile )
    priority_cdm_dict = data[0]
    pklFile.close()
    
    pklFile = open(truth_file, 'rb')
    data = pickle.load( pklFile )
    truth_dict = data[0]
    pklFile.close()
    
    # Set t0 for all objects
    primary_id = 52373
    t_truth = truth_dict[primary_id]['t_truth']
    t0 = t_truth[0]
    
    TCA_dict = {}
    TCA_dict[90000] = t0 + 30.*3600.
    TCA_dict[91000] = t0 + 42.*3600.
    TCA_dict[92000] = t0 + 60.*3600.
    TCA_dict[93000] = t0 + 80.*3600.
    TCA_dict[94000] = t0 + 97.*3600.
    TCA_dict[95000] = t0 + 98.*3600.
    TCA_dict[96000] = t0 + 99.*3600.
    TCA_dict[97000] = t0 + 125.*3600.
    TCA_dict[98000] = t0 + 145.*3600.
    TCA_dict[99000] = t0 + 162.*3600.
    
    baseline_plot_data = {}
    for cdm_id in baseline_cdm_dict:
        
        CDM_epoch = baseline_cdm_dict[cdm_id]['CDM_epoch']
        secondary_id = baseline_cdm_dict[cdm_id]['secondary_id']
        miss_distance = baseline_cdm_dict[cdm_id]['miss_distance']
        Pc = baseline_cdm_dict[cdm_id]['Pc2D_Foster']
        Uc = baseline_cdm_dict[cdm_id]['Uc2D']
        
        thrs = (CDM_epoch - t0)/3600.
        
        if secondary_id not in baseline_plot_data:
            baseline_plot_data[secondary_id] = {}
            baseline_plot_data[secondary_id]['thrs'] = []
            baseline_plot_data[secondary_id]['miss_distance'] = []
            baseline_plot_data[secondary_id]['Pc'] = []
            baseline_plot_data[secondary_id]['Uc'] = []
            
        baseline_plot_data[secondary_id]['thrs'].append(thrs)
        baseline_plot_data[secondary_id]['miss_distance'].append(miss_distance)
        baseline_plot_data[secondary_id]['Pc'].append(Pc)
        baseline_plot_data[secondary_id]['Uc'].append(Uc)
        
    
    greedy_plot_data = {}
    for cdm_id in greedy_cdm_dict:
        
        CDM_epoch = greedy_cdm_dict[cdm_id]['CDM_epoch']
        secondary_id = greedy_cdm_dict[cdm_id]['secondary_id']
        miss_distance = greedy_cdm_dict[cdm_id]['miss_distance']
        Pc = greedy_cdm_dict[cdm_id]['Pc2D_Foster']
        Uc = greedy_cdm_dict[cdm_id]['Uc2D']
        
        if CDM_epoch > (TCA_dict[secondary_id] - 3600.):
            continue
        
        thrs = (CDM_epoch - t0)/3600.
        
        if secondary_id not in greedy_plot_data:
            greedy_plot_data[secondary_id] = {}
            greedy_plot_data[secondary_id]['thrs'] = []
            greedy_plot_data[secondary_id]['miss_distance'] = []
            greedy_plot_data[secondary_id]['Pc'] = []
            greedy_plot_data[secondary_id]['Uc'] = []
            
        greedy_plot_data[secondary_id]['thrs'].append(thrs)
        greedy_plot_data[secondary_id]['miss_distance'].append(miss_distance)
        greedy_plot_data[secondary_id]['Pc'].append(Pc)
        greedy_plot_data[secondary_id]['Uc'].append(Uc)  
        
        
    priority_plot_data = {}
    for cdm_id in priority_cdm_dict:
        
        CDM_epoch = priority_cdm_dict[cdm_id]['CDM_epoch']
        secondary_id = priority_cdm_dict[cdm_id]['secondary_id']
        miss_distance = priority_cdm_dict[cdm_id]['miss_distance']
        Pc = priority_cdm_dict[cdm_id]['Pc2D_Foster']
        Uc = priority_cdm_dict[cdm_id]['Uc2D']
        
        if CDM_epoch > (TCA_dict[secondary_id] - 3600.):
            continue
        
        thrs = (CDM_epoch - t0)/3600.
        
        if secondary_id not in priority_plot_data:
            priority_plot_data[secondary_id] = {}
            priority_plot_data[secondary_id]['thrs'] = []
            priority_plot_data[secondary_id]['miss_distance'] = []
            priority_plot_data[secondary_id]['Pc'] = []
            priority_plot_data[secondary_id]['Uc'] = []
            
        priority_plot_data[secondary_id]['thrs'].append(thrs)
        priority_plot_data[secondary_id]['miss_distance'].append(miss_distance)
        priority_plot_data[secondary_id]['Pc'].append(Pc)
        priority_plot_data[secondary_id]['Uc'].append(Uc)  
    
    
    for obj_id in baseline_plot_data:
        
        plt.figure()
        
        plt.subplot(3,1,1)
        plt.semilogy(baseline_plot_data[obj_id]['thrs'], baseline_plot_data[obj_id]['miss_distance'], 'ko-', label='Baseline')
        plt.semilogy(priority_plot_data[obj_id]['thrs'], priority_plot_data[obj_id]['miss_distance'], 'bo-', label='Priority')
        plt.semilogy(greedy_plot_data[obj_id]['thrs'], greedy_plot_data[obj_id]['miss_distance'], 'ro-', label='Maintenance')
        plt.ylabel('Miss Dist [m]')
        plt.title('Object ' + str(obj_id))
        plt.ylim([0.1, 20000])
        plt.legend()
        
        plt.subplot(3,1,2)
        plt.semilogy(baseline_plot_data[obj_id]['thrs'], baseline_plot_data[obj_id]['Pc'], 'ko-')
        plt.semilogy(priority_plot_data[obj_id]['thrs'], priority_plot_data[obj_id]['Pc'], 'bo-')
        plt.semilogy(greedy_plot_data[obj_id]['thrs'], greedy_plot_data[obj_id]['Pc'], 'ro-')
        plt.ylabel('Pc')
        plt.ylim([1e-10, 2])
        
        plt.subplot(3,1,3)
        plt.semilogy(baseline_plot_data[obj_id]['thrs'], baseline_plot_data[obj_id]['Uc'], 'ko-')
        plt.semilogy(priority_plot_data[obj_id]['thrs'], priority_plot_data[obj_id]['Uc'], 'bo-')
        plt.semilogy(greedy_plot_data[obj_id]['thrs'], greedy_plot_data[obj_id]['Uc'], 'ro-')
        plt.ylabel('Uc')
        plt.xlabel('Time [hours]')
        plt.ylim([1e-10, 2])
        
        
    
    plt.show()
    
    return


# have a function that generates a CDM following each measurement update and
# maintain a dictionary of CDMs that can be easily looked up to see if an 
# object is included - to be used for prioritization (dynamically updated CDM
# dictionary)




if __name__ == '__main__':
    
    plt.close('all')

    rso_file = os.path.join('data', 'rso_catalog_truth.pkl')
    estimated_rso_file = os.path.join('data', 'estimated_rso_catalog_batchPo.pkl')
    sensor_file = os.path.join('data', 'sensor_data_rgazel.pkl')
    truth_file = os.path.join('data', 'propagated_truth_10sec.pkl')
    visibility_file = os.path.join('data', 'visibility_data.pkl')  
    
    
    # meas_file = os.path.join('data', 'baseline_measurement_data_rgazel.pkl')
    # output_file = os.path.join('data', 'baseline_output_batchPo_rgazel.pkl')
    baseline_cdm_file = os.path.join('data', 'baseline_cdm_batchPo_rgazel.pkl')
    
    
    # meas_file = os.path.join('data', 'greedy_renyi_measurement_data_rgazel.pkl')
    # output_file = os.path.join('data', 'greedy_renyi_output_batchPo_rgazel_all.pkl')
    # greedy_cdm_file = os.path.join('data', 'greedy_renyi_cdm_batchPo_rgazel.pkl')
    
    # meas_file = os.path.join('data', 'greedy_renyi_measurement_data_rgazel_10sec_limitvis_multistep.pkl')
    # output_file = os.path.join('data', 'greedy_renyi_output_batchPo_rgazel_10sec_limitvis_multistep_all.pkl')
    greedy_cdm_file = os.path.join('data', 'greedy_renyi_cdm_batchPo_rgazel_10sec_limitvis_multistep.pkl')
    
    
    # meas_file = os.path.join('data', 'priority_basic_measurement_data_rgazel_10sec_limitvis_multistep.pkl')
    # output_file = os.path.join('data', 'priority_basic_output_batchPo_rgazel_10sec_limitvis_multistep_all.pkl')
    # priority_cdm_file = os.path.join('data', 'priority_basic_cdm_batchPo_rgazel_10sec_limitvis_multistep.pkl')
    
    # meas_file = os.path.join('data', 'priority_risk_measurement_data_rgazel_10sec_limitvis_multistep_tif01_full.pkl')
    # output_file = os.path.join('data', 'priority_risk_output_batchPo_rgazel_10sec_limitvis_multistep_tif01_all_Q0.pkl')
    priority_cdm_file = os.path.join('data', 'priority_risk_cdm_batchPo_rgazel_10sec_limitvis_multistep_tif01_Q0.pkl')
    
    
    meas_file = os.path.join('data', 'catalog_maint_measurement_data_rgazel_10sec_limitvis_multistep_day5.pkl')
    output_file = os.path.join('data', 'catalog_maint_output_batchPo_rgazel_10sec_limitvis_multistep_secondaries.pkl')
    catalog_maint_cdm_file = os.path.join('data', 'catalog_maint_cdm_batchPo_rgazel_10sec_limitvis_multistep.pkl')
    
    
    # generate_baseline_measurements(rso_file, sensor_file, visibility_file,
    #                                truth_file, meas_file)   
    
    
    
    reward_fcn = sensor.reward_renyi_infogain
    generate_greedy_measurements_tif(estimated_rso_file, sensor_file, visibility_file,
                                     truth_file, meas_file, reward_fcn)
    
    
    # obj_id_list = [52373, 90000, 91000, 92000, 93000, 94000, 95000, 96000, 97000, 98000, 99000]
    # obj_id_list = [91005, 95001, 95002, 97006]
    # obj_id_list = [95000]
    # obj_id_list = []
    # filter_process_measurements(estimated_rso_file, sensor_file, meas_file,
    #                             output_file, obj_id_list)

    # process_filter_output(output_file, truth_file)
    
    
    # filter_process_meas_and_save(estimated_rso_file, sensor_file, meas_file, output_file, truth_file)


    # window_hrs = 8.
    # batch_process_baseline_measurements(estimated_rso_file, sensor_file, meas_file, output_file, window_hrs)

    
    
    # process_baseline_batch_output(output_file, truth_file)
    
    # process_cdm_output(estimated_rso_file, output_file, priority_cdm_file)

    # generate_case_summary(meas_file, output_file, truth_file)
    
    
    # plot_risk_metrics(baseline_cdm_file, greedy_cdm_file, priority_cdm_file, truth_file)









