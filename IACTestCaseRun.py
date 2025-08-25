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
    
    # obj_id_list = [52373, 90000, 91000, 92000, 93000, 94000, 95000, 96000,
    #                97000, 98000, 99000]
    
    obj_id_list = [52373]
    
    
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


def filter_process_baseline_measurements(rso_file, sensor_file, meas_file, output_file):
    
    
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
    pklFile.close() 
    
    # Standard data
    filter_params = {}
    filter_params['Qeci'] = 1e-13*np.diag([1., 1., 1.])
    filter_params['Qric'] = 0*np.diag([1., 1., 1.])
    filter_params['alpha'] = 1e-2
    filter_params['gap_seconds'] = 600.
    
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
    obj_id_list = list(meas_dict.keys())
    for obj_id in obj_id_list:
        
        print('')
        print('obj_id', obj_id)
        t0 = rso_dict[obj_id]['epoch_tdb']
        
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
        tk_max = t0 + 6.*3600.
        tk_list = meas_dict[obj_id]['tk_list']
        Yk_list = meas_dict[obj_id]['Yk_list']
        sensor_id_list = meas_dict[obj_id]['sensor_id_list']
        ind = bisect.bisect_right(tk_list, tk_max)
        
        filter_meas_dict = {}
        filter_meas_dict['tk_list'] = tk_list[0:ind]
        filter_meas_dict['Yk_list'] = Yk_list[0:ind]
        filter_meas_dict['sensor_id_list'] = sensor_id_list[0:ind]
        
        
        # Run filter
        filter_output = est.ukf(state_params, filter_meas_dict, sensor_dict,
                                int_params, filter_params, bodies)
        
        output_dict[obj_id] = filter_output
        
        
    # Save output
    pklFile = open( output_file, 'wb' )
    pickle.dump([output_dict], pklFile, -1)
    pklFile.close()
    
    
    return


def batch_process_baseline_measurements(rso_file, sensor_file, meas_file, output_file):
    
    
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
    pklFile.close() 
    
    # Standard data
    filter_params = {}
    filter_params['Qeci'] = 1e-13*np.diag([1., 1., 1.])
    filter_params['Qric'] = 0*np.diag([1., 1., 1.])
    filter_params['alpha'] = 1e-2
    filter_params['gap_seconds'] = 600.
    
    
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
    obj_id_list = list(meas_dict.keys())
    for obj_id in obj_id_list:
        
        print('')
        print('obj_id', obj_id)
        t0 = rso_dict[obj_id]['epoch_tdb']
        
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
        tk_max = t0 + 6.*3600.
        tk_list = meas_dict[obj_id]['tk_list']
        Yk_list = meas_dict[obj_id]['Yk_list']
        sensor_id_list = meas_dict[obj_id]['sensor_id_list']
        ind = bisect.bisect_right(tk_list, tk_max)
        
        filter_meas_dict = {}
        filter_meas_dict['tk_list'] = tk_list[0:ind]
        filter_meas_dict['Yk_list'] = Yk_list[0:ind]
        filter_meas_dict['sensor_id_list'] = sensor_id_list[0:ind]
        
        # Set tk_output in filter_params
        filter_params['tk_output'] = list(np.arange(t0, tk_max+1., 10.))
        
        
        # Run filter
        filter_output = est.unscented_batch(state_params, filter_meas_dict,
                                            sensor_dict, int_params,
                                            filter_params, bodies)
        
        output_dict[obj_id] = filter_output
        
        
    # Save output
    pklFile = open( output_file, 'wb' )
    pickle.dump([output_dict], pklFile, -1)
    pklFile.close()
    
    
    return


def process_baseline_filter_output(output_file, truth_file):
    
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


def process_baseline_cdm_output(rso_file, est_output_file, cdm_file):
    
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


# have a function that generates a CDM following each measurement update and
# maintain a dictionary of CDMs that can be easily looked up to see if an 
# object is included - to be used for prioritization (dynamically updated CDM
# dictionary)




if __name__ == '__main__':
    
    plt.close('all')

    rso_file = os.path.join('data', 'rso_catalog_truth.pkl')
    sensor_file = os.path.join('data', 'sensor_data_rgradec_lownoise.pkl')
    visibility_file = os.path.join('data', 'visibility_data.pkl')
    meas_file = os.path.join('data', 'baseline_measurement_data_rgradec_lownoise_52373.pkl')
    truth_file = os.path.join('data', 'propagated_truth_10sec.pkl')
    estimated_rso_file = os.path.join('data', 'estimated_rso_catalog_diagPo.pkl')
    output_file = os.path.join('data', 'baseline_output_diagPo_rgradec_lownoise_52373_6hr_batch.pkl')
    # cdm_file = os.path.join('data', 'baseline_cdm_batchPo_rgradec_lownoise.pkl')
    
    
    # generate_baseline_measurements(rso_file, sensor_file, visibility_file,
    #                                truth_file, meas_file)    
    
    
    
    # filter_process_baseline_measurements(estimated_rso_file, sensor_file, meas_file, output_file)

    # process_baseline_filter_output(output_file, truth_file)
    
    
    # process_baseline_cdm_output(estimated_rso_file, output_file, cdm_file)













