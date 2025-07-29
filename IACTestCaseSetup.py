import numpy as np
from datetime import datetime, timedelta
import os
import pickle

# Load tudatpy modules
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.astro import time_conversion
from tudatpy.astro.time_conversion import DateTime




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


if __name__ == '__main__':
    
    UTC_list, epoch_tdb_list = compute_UTC_from_TDB()
    
    define_primary_object(epoch_tdb_list[0])



