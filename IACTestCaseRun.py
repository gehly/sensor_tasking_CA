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

def generate_baseline_measurements(sensor_file, visibility_file):
    
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





















