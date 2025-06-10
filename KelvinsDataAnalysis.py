###############################################################################
# This code contains functions to analyze the ESA Kelvins dataset
#
#
# References:
#
#  [1] https://kelvins.esa.int/collision-avoidance-challenge/
#
#  [2] Uriot, T., et al., "Spacecraft collision avoidance challenge: Design and
#      results of a machine learning competition," Astrodynamics, Vol. 6, 
#      No. 2, 2022.
#
#
###############################################################################



import numpy as np
import pandas as pd
import os



###############################################################################
# Basic I/O
###############################################################################

def read_kelvins_data(csv_file):
    '''
    This function reads an input CSV file containing conjunction data from the
    ESA Kelvins Machine Learning challenge and stores it in a pandas dataframe
    for analysis.
    
    Parameters
    ------
    csv_file : string
        path and filename of CSV file containing conjunction data    
    
    Returns
    ------
    kelvins_df : pandas dataframe
        columns contain conjunction data including time to TCA, miss distance,
        and collision probability
    
    '''
    
    kelvins_df = pd.read_csv(csv_file)
    
    return kelvins_df


def kelvins_df2dict(kelvins_df):
    
    event_id_list = conjunction_df['event_id'].tolist()
    
    kelvins_dict = {}
    for ii in range(len(event_id_list)):
        
        # Retrieve event id and initialize dictionary entry if needed
        event_id = event_id_list[ii]
        if event_id not in kelvins_dict:
            kelvins_dict[event_id] = {}
            kelvins_dict[event_id]['time_to_tca'] = []
            kelvins_dict[event_id]['risk'] = []
            
        # Append data to lists
        kelvins_dict[event_id]['time_to_tca'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('time_to_tca')])
        kelvins_dict[event_id]['risk'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('time_to_tca')])
        
    
    
    return kelvins_dict


def kelvins_data_stats(kelvins_df):
    '''
    
    '''
    
    num_cdm = len(conjunction_df)
    
    
    
    num_event = event_id[-1]+1
    
    print(num_cdm)
    print(num_event)
    
    
    
    
    return


def clean_kelvins_data(original_df):
    
    
    return reduced_df


###############################################################################
# Data Analysis
###############################################################################





###############################################################################
# Unit Test
###############################################################################

def check_kelvins_Pc():
    
    
    
    
    return




if __name__ == '__main__':
    
    kelvins_data = os.path.join('data', 'test_data.csv')
    
    conjunction_df = read_kelvins_data(kelvins_data)
    
    kelvins_data_stats(conjunction_df)
