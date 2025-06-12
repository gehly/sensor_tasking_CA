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


def kelvins_df_to_dict(kelvins_df):
    
    event_id_list = kelvins_df['event_id'].tolist()
    
    kelvins_dict = {}
    for ii in range(len(event_id_list)):
        
        # Retrieve event id and initialize dictionary entry if needed
        event_id = event_id_list[ii]
        if event_id not in kelvins_dict:
            kelvins_dict[event_id] = {}
            kelvins_dict[event_id]['time_to_tca'] = []
            kelvins_dict[event_id]['risk'] = []
            
            
        # Retrieve state and covariance data
        reduced_df = kelvins_df[ii:ii+1]
        r_RTN, v_RTN, P1_RTN, P2_RTN = get_state_and_covar(reduced_df)
        
        mistake
        
        # Append data to lists
        kelvins_dict[event_id]['time_to_tca'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('time_to_tca')])
        kelvins_dict[event_id]['risk'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('risk')])
        
    
    print(len(kelvins_dict))
    print(kelvins_dict[0])
    
    return kelvins_dict


def get_state_and_covar(df):
    
    r_RTN = df[['relative_position_r', 'relative_position_t', 'relative_position_n']].to_numpy()
    v_RTN = df[['relative_velocity_r', 'relative_velocity_t', 'relative_velocity_n']].to_numpy()
    
    std1 = df[['t_sigma_r', 't_sigma_t', 't_sigma_n',
               't_sigma_rdot', 't_sigma_tdot', 't_sigma_ndot']].to_numpy().flatten()
    
    corr1 = df[['t_ct_r',
                't_cn_r', 't_cn_t',
                't_crdot_r', 't_crdot_t', 't_crdot_n',
                't_ctdot_r', 't_ctdot_t', 't_ctdot_n', 't_ctdot_rdot',
                't_cndot_r', 't_cndot_t', 't_cndot_n', 't_cndot_rdot', 't_cndot_tdot']].to_numpy().flatten()
    
    std2 = df[['c_sigma_r', 'c_sigma_t', 'c_sigma_n',
               'c_sigma_rdot', 'c_sigma_tdot', 'c_sigma_ndot']].to_numpy().flatten()
    
    corr2 = df[['c_ct_r',
                'c_cn_r', 'c_cn_t',
                'c_crdot_r', 'c_crdot_t', 'c_crdot_n',
                'c_ctdot_r', 'c_ctdot_t', 'c_ctdot_n', 'c_ctdot_rdot',
                'c_cndot_r', 'c_cndot_t', 'c_cndot_n', 'c_cndot_rdot', 'c_cndot_tdot']].to_numpy().flatten()
    
    
    r_RTN = np.reshape(r_RTN, (3,1))
    v_RTN = np.reshape(v_RTN, (3,1))
    
    # P1_RTN = np.zeros((6,6))
    # P1_RTN[0,0] = std1[0]**2.  # r
    # P1_RTN[1,1] = std1[1]**2.  # t
    # P1_RTN[2,2] = std1[2]**2.  # n
    # P1_RTN[3,3] = std1[3]**2.  # rdot
    # P1_RTN[4,4] = std1[4]**2.  # tdot
    # P1_RTN[5,5] = std1[5]**2.  # ndot
    
    # P1_RTN[0,1] = P1_RTN[1,0] = corr1[0]*std1[0]*std1[1]
    # P1_RTN[0,2] = P1_RTN[2,0] = corr1[1]*std1[0]*std1[2]
    # P1_RTN[1,2] = P1_RTN[2,1] = corr1[2]*std1[1]*std1[2]
    # P1_RTN[0,3] = P1_RTN[3,0] = corr1[3]*std1[0]*std1[3]
    # P1_RTN[1,3] = P1_RTN[3,1] = corr1[4]*std1[1]*std1[3]
    # P1_RTN[2,3] = P1_RTN[3,2] = corr1[5]*std1[2]*std1[3]
    # P1_RTN[0,4] = P1_RTN[4,0] = corr1[6]*std1[0]*std1[4]
    # P1_RTN[1,4] = P1_RTN[4,1] = corr1[7]*std1[1]*std1[4]
    # P1_RTN[2,4] = P1_RTN[4,2] = corr1[8]*std1[2]*std1[4]
    # P1_RTN[3,4] = P1_RTN[4,3] = corr1[9]*std1[3]*std1[4]
    # P1_RTN[0,5] = P1_RTN[5,0] = corr1[10]*std1[0]*std1[5]
    # P1_RTN[1,5] = P1_RTN[5,1] = corr1[11]*std1[1]*std1[5]
    # P1_RTN[2,5] = P1_RTN[5,2] = corr1[12]*std1[2]*std1[5]
    # P1_RTN[3,5] = P1_RTN[5,3] = corr1[13]*std1[3]*std1[5]
    # P1_RTN[4,5] = P1_RTN[5,4] = corr1[14]*std1[4]*std1[5]
    
    
    P1_RTN = compute_covar_matrix(std1, corr1)
    P2_RTN = compute_covar_matrix(std2, corr2)
    
    
    print(P1_RTN)
    print(P1_RTN - P1_RTN.T)
    print(np.linalg.det(P1_RTN[0:3,0:3]))
    
    print(P2_RTN)
    print(P2_RTN - P2_RTN.T)
    print(np.linalg.det(P2_RTN[0:3,0:3]))
    
    mistake
    
    
    return r_RTN, v_RTN, P1_RTN, P2_RTN


def compute_covar_matrix(std, corr):
    
    P = np.zeros((6,6))
    P[0,0] = std[0]**2.  # r
    P[1,1] = std[1]**2.  # t
    P[2,2] = std[2]**2.  # n
    P[3,3] = std[3]**2.  # rdot
    P[4,4] = std[4]**2.  # tdot
    P[5,5] = std[5]**2.  # ndot
    
    P[0,1] = P[1,0] = corr[0]*std[0]*std[1]
    P[0,2] = P[2,0] = corr[1]*std[0]*std[2]
    P[1,2] = P[2,1] = corr[2]*std[1]*std[2]
    P[0,3] = P[3,0] = corr[3]*std[0]*std[3]
    P[1,3] = P[3,1] = corr[4]*std[1]*std[3]
    P[2,3] = P[3,2] = corr[5]*std[2]*std[3]
    P[0,4] = P[4,0] = corr[6]*std[0]*std[4]
    P[1,4] = P[4,1] = corr[7]*std[1]*std[4]
    P[2,4] = P[4,2] = corr[8]*std[2]*std[4]
    P[3,4] = P[4,3] = corr[9]*std[3]*std[4]
    P[0,5] = P[5,0] = corr[10]*std[0]*std[5]
    P[1,5] = P[5,1] = corr[11]*std[1]*std[5]
    P[2,5] = P[5,2] = corr[12]*std[2]*std[5]
    P[3,5] = P[5,3] = corr[13]*std[3]*std[5]
    P[4,5] = P[5,4] = corr[14]*std[4]*std[5]
    
    
    return P


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
    
    # provide X1 = [r_RTN, v_RTN] and X2 = [0] to get correct formulation of
    # encounter frame using existing code
    
    
    return




if __name__ == '__main__':
    
    kelvins_data = os.path.join('data', 'test_data.csv')
    
    kelvins_df = read_kelvins_data(kelvins_data)    
    
    kelvins_df_to_dict(kelvins_df)
