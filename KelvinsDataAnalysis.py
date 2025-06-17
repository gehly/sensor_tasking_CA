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


import ConjunctionUtilities as ConjUtil

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


def clean_kelvins_data(original_df):
    
    
    return reduced_df


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
            kelvins_dict[event_id]['max_risk_estimate'] = []
            kelvins_dict[event_id]['max_risk_scaling'] = []
            kelvins_dict[event_id]['miss_distance'] = []
            kelvins_dict[event_id]['relative_speed'] = []
            kelvins_dict[event_id]['r_RTN'] = []
            kelvins_dict[event_id]['v_RTN'] = []
            kelvins_dict[event_id]['P1_RTN'] = []
            kelvins_dict[event_id]['P2_RTN'] = []
            kelvins_dict[event_id]['t_j2k_sma'] = []
            kelvins_dict[event_id]['t_j2k_ecc'] = []
            kelvins_dict[event_id]['t_j2k_inc'] = []
            kelvins_dict[event_id]['c_j2k_sma'] = []
            kelvins_dict[event_id]['c_j2k_ecc'] = []
            kelvins_dict[event_id]['c_j2k_inc'] = []
            kelvins_dict[event_id]['mahalanobis_distance'] = []
            kelvins_dict[event_id]['t_position_covariance_det'] = []
            kelvins_dict[event_id]['c_position_covariance_det'] = []            
            
            
        # Retrieve state and covariance data
        reduced_df = kelvins_df[ii:ii+1]
        r_RTN, v_RTN, P1_RTN, P2_RTN = get_state_and_covar(reduced_df)
        
        # Append data to lists
        kelvins_dict[event_id]['time_to_tca'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('time_to_tca')])
        kelvins_dict[event_id]['risk'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('risk')])
        kelvins_dict[event_id]['max_risk_estimate'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('max_risk_estimate')])
        kelvins_dict[event_id]['max_risk_scaling'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('max_risk_scaling')])
        kelvins_dict[event_id]['miss_distance'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('miss_distance')])
        kelvins_dict[event_id]['relative_speed'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('relative_speed')])
        kelvins_dict[event_id]['t_j2k_sma'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('t_j2k_sma')])
        kelvins_dict[event_id]['t_j2k_ecc'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('t_j2k_ecc')])
        kelvins_dict[event_id]['t_j2k_inc'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('t_j2k_inc')])
        kelvins_dict[event_id]['c_j2k_sma'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('c_j2k_sma')])
        kelvins_dict[event_id]['c_j2k_ecc'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('c_j2k_ecc')])
        kelvins_dict[event_id]['c_j2k_inc'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('c_j2k_inc')])
        kelvins_dict[event_id]['mahalanobis_distance'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('mahalanobis_distance')])
        kelvins_dict[event_id]['t_position_covariance_det'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('t_position_covariance_det')])
        kelvins_dict[event_id]['c_position_covariance_det'].append(kelvins_df.iloc[ii, kelvins_df.columns.get_loc('c_position_covariance_det')])

        kelvins_dict[event_id]['r_RTN'].append(r_RTN)
        kelvins_dict[event_id]['v_RTN'].append(v_RTN)
        kelvins_dict[event_id]['P1_RTN'].append(P1_RTN)
        kelvins_dict[event_id]['P2_RTN'].append(P2_RTN)
        
        
    
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
   
    P1_RTN = compute_covar_matrix(std1, corr1)
    P2_RTN = compute_covar_matrix(std2, corr2)
    
    
    # print(P1_RTN)
    # print(P1_RTN - P1_RTN.T)
    # print(np.linalg.det(P1_RTN[0:3,0:3]))
    
    # print(P2_RTN)
    # print(P2_RTN - P2_RTN.T)
    # print(np.linalg.det(P2_RTN[0:3,0:3]))
    
    # mistake
    
    
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





###############################################################################
# Data Analysis
###############################################################################





###############################################################################
# Unit Test
###############################################################################


def verify_kelvins_dict(kelvins_dict):
    
    event_id_list = sorted(list(kelvins_dict.keys()))
    
    # Loop over events and CDMs, retrieve data and recompute determinants and
    # collision risk metrics
    for event_id in event_id_list:
        
        num_cdm = len(kelvins_dict[event_id]['time_to_tca'])
        for ii in range(num_cdm):
            
            risk = kelvins_dict[event_id]['risk'][ii]
            miss_distance = kelvins_dict[event_id]['miss_distance'][ii]
            relative_speed = kelvins_dict[event_id]['relative_speed'][ii]
            mahalanobis_distance = kelvins_dict[event_id]['mahalanobis_distance'][ii]
            t_position_covariance_det = kelvins_dict[event_id]['t_position_covariance_det'][ii]
            c_position_covariance_det = kelvins_dict[event_id]['c_position_covariance_det'][ii]
            
            r_RTN = kelvins_dict[event_id]['r_RTN'][ii]
            v_RTN = kelvins_dict[event_id]['v_RTN'][ii]
            P1_RTN = kelvins_dict[event_id]['P1_RTN'][ii]
            P2_RTN = kelvins_dict[event_id]['P2_RTN'][ii]
            
            
            
            # Recompute metrics to check
            check_miss_distance = np.linalg.norm(r_RTN) - miss_distance
            check_relative_speed = np.linalg.norm(v_RTN) - relative_speed
            dM = ConjUtil.compute_mahalanobis_distance(r_RTN, np.zeros((3,1)), P1_RTN[0:3,0:3], P2_RTN[0:3,0:3])
            check_mahalanobis_distance = dM - mahalanobis_distance
            check_t_poscov_det = np.linalg.det(P1_RTN[0:3,0:3]) - t_position_covariance_det
            check_c_poscov_det = np.linalg.det(P2_RTN[0:3,0:3]) - c_position_covariance_det
            
            print(check_miss_distance)
            print(check_relative_speed)
            print(check_mahalanobis_distance)
            print(dM, mahalanobis_distance)
            print(check_t_poscov_det)
            print(check_c_poscov_det)
            
            mistake
            
    
    
    return


def check_kelvins_Pc():
    
    # provide X1 = [r_RTN, v_RTN] and X2 = [0] to get correct formulation of
    # encounter frame using existing code
    
    
    return




if __name__ == '__main__':
    
    kelvins_data = os.path.join('data', 'test_data.csv')
    
    kelvins_df = read_kelvins_data(kelvins_data)    
    
    kelvins_dict = kelvins_df_to_dict(kelvins_df)
    
    verify_kelvins_dict(kelvins_dict)
