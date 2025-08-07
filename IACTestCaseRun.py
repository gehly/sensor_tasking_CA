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

def generate_baseline_measurements(rso_file, sensor_file, visibility_file):
    
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
    
    # Initialize output
    meas_dict = {}
    
    # Parse visibility dict, add noise to measurements and store
    for sensor_id in visibility_dict:
        
        meas_types = sensor_dict[sensor_id]['meas_types']
        sigma_dict = sensor_dict[sensor_id]['sigma_dict']
        
        for obj_id in visibility_dict[sensor_id]:
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
                Yk = np.array([[rg_list[kk] + np.random.randn()*sigma_dict['rg']],
                               [az_list[kk] + np.random.randn()*sigma_dict['az']],
                               [el_list[kk] + np.random.randn()*sigma_dict['el']]])
                
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
    meas_file = os.path.join('data', 'baseline_measurement_data.pkl')
    pklFile = open( meas_file, 'wb' )
    pickle.dump([meas_dict], pklFile, -1)
    pklFile.close()
    
    return


def process_baseline_measurements():
    
    
    return


# have a function that generates a CDM following each measurement update and
# maintain a dictionary of CDMs that can be easily looked up to see if an 
# object is included - to be used for prioritization (dynamically updated CDM
# dictionary)




if __name__ == '__main__':
    
    plt.close('all')

    rso_file = os.path.join('data', 'rso_catalog_truth.pkl')
    sensor_file = os.path.join('data', 'sensor_data.pkl')
    visibility_file = os.path.join('data', 'visibility_data.pkl')
    generate_baseline_measurements(rso_file, sensor_file, visibility_file)
















