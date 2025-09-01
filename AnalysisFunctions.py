import numpy as np
import matplotlib.pyplot as plt
import bisect
import os
import pickle
import math

import ConjunctionUtilities as conj
import TudatPropagator as prop


def compute_errors(truth_dict, output_dict, obj_id, plot_flag=True):
    
    t_truth = truth_dict[obj_id]['t_truth']
    X_truth = truth_dict[obj_id]['X_truth']
    filter_output = output_dict[obj_id]
        
    # Times
    t0 = t_truth[0]
    tk_list = sorted(list(filter_output.keys()))
    thrs = [(tk - t0)/3600. for tk in tk_list]
    
    # Number of states and measurements
    Xo = filter_output[tk_list[0]]['state']
    resids0 = filter_output[tk_list[0]]['resids']
    n = len(Xo)
    p = len(resids0)
    

    # Compute state errors
    X_err = np.zeros((n, len(filter_output)))
    X_err_ric = np.zeros((3, len(filter_output)))

    resids = np.zeros((p, len(filter_output)))
    sig_x = np.zeros(len(filter_output),)
    sig_y = np.zeros(len(filter_output),)
    sig_z = np.zeros(len(filter_output),)
    sig_dx = np.zeros(len(filter_output),)
    sig_dy = np.zeros(len(filter_output),)
    sig_dz = np.zeros(len(filter_output),)
    sig_r = np.zeros(len(filter_output),)
    sig_i = np.zeros(len(filter_output),)
    sig_c = np.zeros(len(filter_output),)
    pos3D = np.zeros(len(filter_output),)
    
    for kk in range(len(tk_list)):
        tk = tk_list[kk]
        X = filter_output[tk]['state']
        P = filter_output[tk]['covar']
        
        truth_ind = list(t_truth).index(tk)
                
        X_true = X_truth[truth_ind,:].reshape(6,1)
        X_err[:,kk] = (X - X_true).flatten()
        pos3D[kk] = np.linalg.norm(X_err[0:3,kk])
        sig_x[kk] = np.sqrt(P[0,0])
        sig_y[kk] = np.sqrt(P[1,1])
        sig_z[kk] = np.sqrt(P[2,2])
        sig_dx[kk] = np.sqrt(P[3,3])
        sig_dy[kk] = np.sqrt(P[4,4])
        sig_dz[kk] = np.sqrt(P[5,5])
        
        # RIC Errors and Covariance
        rc_vect = X_true[0:3].reshape(3,1)
        vc_vect = X_true[3:6].reshape(3,1)
        err_eci = X_err[0:3,kk].reshape(3,1)
        P_eci = P[0:3,0:3]
        
        err_ric = conj.eci2ric(rc_vect, vc_vect, err_eci)
        P_ric = conj.eci2ric(rc_vect, vc_vect, P_eci)
        X_err_ric[:,kk] = err_ric.flatten()
        sig_r[kk] = np.sqrt(P_ric[0,0])
        sig_i[kk] = np.sqrt(P_ric[1,1])
        sig_c[kk] = np.sqrt(P_ric[2,2])
        
        resids[:,kk] = filter_output[tk]['resids'].flatten()
    

    # Compute and print statistics
    print('\n\nState Error and Residuals Analysis')
    print('\n\t\t\t\t  Mean\t\tSTD')
    print('----------------------------------------')
    print('X ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[0,35:])), '\t{0:0.2E}'.format(np.std(X_err[0,35:])))
    print('Y ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[1,35:])), '\t{0:0.2E}'.format(np.std(X_err[1,35:])))
    print('Z ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[2,35:])), '\t{0:0.2E}'.format(np.std(X_err[2,35:])))
    print('dX ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[3,35:])), '\t{0:0.2E}'.format(np.std(X_err[3,35:])))
    print('dY ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[4,35:])), '\t{0:0.2E}'.format(np.std(X_err[4,35:])))
    print('dZ ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[5,35:])), '\t{0:0.2E}'.format(np.std(X_err[5,35:])))
    print('')
    print('Radial [m]\t\t', '{0:0.2E}'.format(np.mean(X_err_ric[0,35:])), '\t{0:0.2E}'.format(np.std(X_err_ric[0,35:])))
    print('In-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[1,35:])), '\t{0:0.2E}'.format(np.std(X_err_ric[1,35:])))
    print('Cross-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[2,35:])), '\t{0:0.2E}'.format(np.std(X_err_ric[2,35:])))
    print('')
    
        
    resids[1,:] *= 180./np.pi
    resids[2,:] *= 180./np.pi
    
    print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
    print('Az [deg]\t\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
    print('El [deg]\t\t', '{0:0.2E}'.format(np.mean(resids[2,:])), '\t{0:0.2E}'.format(np.std(resids[2,:])))
        


    if plot_flag:
        
        # State Error Plots   
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(thrs, X_err[0,:], 'k.')
        plt.plot(thrs, 3*sig_x, 'k--')
        plt.plot(thrs, -3*sig_x, 'k--')
        plt.title('ECI Position Errors for Object ' + str(obj_id))
        plt.ylabel('X Err [m]')
        
        plt.subplot(3,1,2)
        plt.plot(thrs, X_err[1,:], 'k.')
        plt.plot(thrs, 3*sig_y, 'k--')
        plt.plot(thrs, -3*sig_y, 'k--')
        plt.ylabel('Y Err [m]')
        
        plt.subplot(3,1,3)
        plt.plot(thrs, X_err[2,:], 'k.')
        plt.plot(thrs, 3*sig_z, 'k--')
        plt.plot(thrs, -3*sig_z, 'k--')
        plt.ylabel('Z Err [m]')
    
        plt.xlabel('Time [hours]')
        
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(thrs, X_err[3,:], 'k.')
        plt.plot(thrs, 3*sig_dx, 'k--')
        plt.plot(thrs, -3*sig_dx, 'k--')
        plt.title('ECI Velocity Errors for Object ' + str(obj_id))
        plt.ylabel('dX Err [m/s]')
        
        plt.subplot(3,1,2)
        plt.plot(thrs, X_err[4,:], 'k.')
        plt.plot(thrs, 3*sig_dy, 'k--')
        plt.plot(thrs, -3*sig_dy, 'k--')
        plt.ylabel('dY Err [m/s]')
        
        plt.subplot(3,1,3)
        plt.plot(thrs, X_err[5,:], 'k.')
        plt.plot(thrs, 3*sig_dz, 'k--')
        plt.plot(thrs, -3*sig_dz, 'k--')
        plt.ylabel('dZ Err [m/s]')
    
        plt.xlabel('Time [hours]')
        
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(thrs, X_err_ric[0,:], 'k.')
        plt.plot(thrs, 3*sig_r, 'k--')
        plt.plot(thrs, -3*sig_r, 'k--')
        plt.title('RIC Position Errors for Object ' + str(obj_id))
        plt.ylabel('Radial [m]')
        
        plt.subplot(3,1,2)
        plt.plot(thrs, X_err_ric[1,:], 'k.')
        plt.plot(thrs, 3*sig_i, 'k--')
        plt.plot(thrs, -3*sig_i, 'k--')
        plt.ylabel('In-Track [m]')
        
        plt.subplot(3,1,3)
        plt.plot(thrs, X_err_ric[2,:], 'k.')
        plt.plot(thrs, 3*sig_c, 'k--')
        plt.plot(thrs, -3*sig_c, 'k--')
        plt.ylabel('Cross-Track [m]')
    
        plt.xlabel('Time [hours]')
        
    
        
        # Residuals
        plt.figure()
            
        plt.subplot(3,1,1)
        plt.plot(thrs, resids[0,:], 'k.')
        plt.title('Measurement Residuals for Object ' + str(obj_id))
        plt.ylabel('Range [m]')
        
        plt.subplot(3,1,2)
        plt.plot(thrs, resids[1,:], 'k.')
        plt.ylabel('Az [deg]')
        
        plt.subplot(3,1,3)
        plt.plot(thrs, resids[2,:], 'k.')
        plt.ylabel('El [deg]')
        
        plt.xlabel('Time [hours]')
            
        
            
        
        plt.show()
    
    
    
    
    return thrs, pos3D


def compute_batch_errors(truth_dict, output_dict, full_output_dict, obj_id):
    
    t_truth = truth_dict[obj_id]['t_truth']
    X_truth = truth_dict[obj_id]['X_truth']
    filter_output = output_dict[obj_id]
    full_state_output = full_output_dict[obj_id]
    
    print(X_truth[0])
    
    # print(filter_output.keys())
    
    
    # Times
    tk_list = list(full_state_output.keys())
    t0 = t_truth[0]
    thrs = [(tk - t0)/3600. for tk in tk_list]
    
    meas_tk_list = list(filter_output.keys())
    meas_t0 = sorted(meas_tk_list)[0]
    thrs_meas = [(tk - t0)/3600. for tk in meas_tk_list]
    
    
    # Number of states and measurements
    Xo = filter_output[meas_t0]['state']
    resids0 = filter_output[meas_t0]['resids']
    n = len(Xo)
    p = len(resids0)
    

    # Compute state errors
    X_err = np.zeros((n, len(full_state_output)))
    X_err_ric = np.zeros((3, len(full_state_output)))
    X_err_meas = np.zeros((n, len(filter_output)))
    X_err_ric_meas = np.zeros((3, len(filter_output)))
    resids = np.zeros((p, len(filter_output)))
    sig_x = np.zeros(len(full_state_output),)
    sig_y = np.zeros(len(full_state_output),)
    sig_z = np.zeros(len(full_state_output),)
    sig_dx = np.zeros(len(full_state_output),)
    sig_dy = np.zeros(len(full_state_output),)
    sig_dz = np.zeros(len(full_state_output),)
    sig_r = np.zeros(len(full_state_output),)
    sig_i = np.zeros(len(full_state_output),)
    sig_c = np.zeros(len(full_state_output),)
    
    meas_ind = 0 
    for kk in range(len(full_state_output)):
        tk = tk_list[kk]
        X = full_state_output[tk]['state']
        P = full_state_output[tk]['covar']
                
        truth_ind = list(t_truth).index(tk)
                
        X_true = X_truth[truth_ind,:].reshape(6,1)
        X_err[:,kk] = (X - X_true).flatten()
        sig_x[kk] = np.sqrt(P[0,0])
        sig_y[kk] = np.sqrt(P[1,1])
        sig_z[kk] = np.sqrt(P[2,2])
        sig_dx[kk] = np.sqrt(P[3,3])
        sig_dy[kk] = np.sqrt(P[4,4])
        sig_dz[kk] = np.sqrt(P[5,5])
        
        # RIC Errors and Covariance
        rc_vect = X_true[0:3].reshape(3,1)
        vc_vect = X_true[3:6].reshape(3,1)
        err_eci = X_err[0:3,kk].reshape(3,1)
        P_eci = P[0:3,0:3]
        
        err_ric = conj.eci2ric(rc_vect, vc_vect, err_eci)
        P_ric = conj.eci2ric(rc_vect, vc_vect, P_eci)
        X_err_ric[:,kk] = err_ric.flatten()
        sig_r[kk] = np.sqrt(P_ric[0,0])
        sig_i[kk] = np.sqrt(P_ric[1,1])
        sig_c[kk] = np.sqrt(P_ric[2,2])
        
        # Store data at meas times
        if tk in meas_tk_list:
            X_err_meas[:,meas_ind] = (X - X_true).flatten()
            X_err_ric_meas[:,meas_ind] = err_ric.flatten()
            resids[:,meas_ind] = filter_output[tk]['resids'].flatten()
            meas_ind += 1
        
    

    # Compute and print statistics
    print('\n\nState Error and Residuals Analysis')
    print('\n\t\t\t\t  Mean\t\tSTD')
    print('----------------------------------------')
    print('X ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[0,35:])), '\t{0:0.2E}'.format(np.std(X_err[0,35:])))
    print('Y ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[1,35:])), '\t{0:0.2E}'.format(np.std(X_err[1,35:])))
    print('Z ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[2,35:])), '\t{0:0.2E}'.format(np.std(X_err[2,35:])))
    print('dX ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[3,35:])), '\t{0:0.2E}'.format(np.std(X_err[3,35:])))
    print('dY ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[4,35:])), '\t{0:0.2E}'.format(np.std(X_err[4,35:])))
    print('dZ ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[5,35:])), '\t{0:0.2E}'.format(np.std(X_err[5,35:])))
    print('')
    print('Radial [m]\t\t', '{0:0.2E}'.format(np.mean(X_err_ric[0,35:])), '\t{0:0.2E}'.format(np.std(X_err_ric[0,35:])))
    print('In-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[1,35:])), '\t{0:0.2E}'.format(np.std(X_err_ric[1,35:])))
    print('Cross-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[2,35:])), '\t{0:0.2E}'.format(np.std(X_err_ric[2,35:])))
    print('')
    
        
    resids[1,:] *= 180./np.pi
    resids[2,:] *= 180./np.pi
    
    print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
    print('Az [deg]\t\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
    print('El [deg]\t\t', '{0:0.2E}'.format(np.mean(resids[2,:])), '\t{0:0.2E}'.format(np.std(resids[2,:])))
        


    
    # State Error Plots   
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err[0,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[0,:], 'b.')
    plt.plot(thrs, 3*sig_x, 'k--')
    plt.plot(thrs, -3*sig_x, 'k--')
    plt.ylabel('X Err [m]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err[1,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[1,:], 'b.')
    plt.plot(thrs, 3*sig_y, 'k--')
    plt.plot(thrs, -3*sig_y, 'k--')
    plt.ylabel('Y Err [m]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err[2,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[2,:], 'b.')
    plt.plot(thrs, 3*sig_z, 'k--')
    plt.plot(thrs, -3*sig_z, 'k--')
    plt.ylabel('Z Err [m]')

    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err[3,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[3,:], 'b.')
    plt.plot(thrs, 3*sig_dx, 'k--')
    plt.plot(thrs, -3*sig_dx, 'k--')
    plt.ylabel('dX Err [m/s]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err[4,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[4,:], 'b.')
    plt.plot(thrs, 3*sig_dy, 'k--')
    plt.plot(thrs, -3*sig_dy, 'k--')
    plt.ylabel('dY Err [m/s]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err[5,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[5,:], 'b.')
    plt.plot(thrs, 3*sig_dz, 'k--')
    plt.plot(thrs, -3*sig_dz, 'k--')
    plt.ylabel('dZ Err [m/s]')

    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err_ric[0,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[0,:], 'b.')
    plt.plot(thrs, 3*sig_r, 'k--')
    plt.plot(thrs, -3*sig_r, 'k--')
    plt.ylabel('Radial [m]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err_ric[1,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[1,:], 'b.')
    plt.plot(thrs, 3*sig_i, 'k--')
    plt.plot(thrs, -3*sig_i, 'k--')
    plt.ylabel('In-Track [m]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err_ric[2,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[2,:], 'b.')
    plt.plot(thrs, 3*sig_c, 'k--')
    plt.plot(thrs, -3*sig_c, 'k--')
    plt.ylabel('Cross-Track [m]')

    plt.xlabel('Time [hours]')
    

    
    # Residuals
    plt.figure()
        
    plt.subplot(3,1,1)
    plt.plot(thrs_meas, resids[0,:], 'k.')
    plt.title('Measurement Residuals for Object ' + str(obj_id))
    plt.ylabel('Range [m]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs_meas, resids[1,:], 'k.')
    plt.ylabel('Az [deg]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs_meas, resids[2,:], 'k.')
    plt.ylabel('El [deg]')
    
    plt.xlabel('Time [hours]')
        
    
        
    
    plt.show()
    
    
    
    
    return


def risk_metric_evolution(cdm_file, output_dict, rso_dict, primary_id, tf, bodies=None):
    
    if bodies is None:
        bodies_to_create = ['Sun', 'Earth', 'Moon']
        bodies = prop.tudat_initialize_bodies(bodies_to_create)
        
    # Secondary object id's
    # secondary_id_list = sorted(list(rso_dict.keys()))
    # del_ind = secondary_id_list.index(primary_id)
    # del secondary_id_list[del_ind]
    secondary_id_list = [90000, 91000, 92000, 93000, 94000, 95000, 96000, 97000,
                         98000, 99000]
    
    
    t0 = rso_dict[52373]['epoch_tdb']
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
    
    # Allowable gap to still be considered the same pass
    max_gap = 600.
    
    # Find times when object estimates are updated
    obs_dict = {}
    stop_list_all = []
    obj_id_list_all = []
    for obj_id in output_dict:
        filter_output = output_dict[obj_id]
        tk_list = sorted(list(filter_output.keys()))
        
        tk_prior = tk_list[0]
        start = tk_list[0]
        stop = tk_list[0]
        start_list = []
        stop_list = []
        for kk in range(len(tk_list)):
            
            tk = tk_list[kk]
            
            # If current time is close to previous, pass continues
            if (tk - tk_prior) < (max_gap+1.):

                # Update pass stop time and tk_prior for next iteration
                stop = tk
                tk_prior = tk
                
            # If current time is far from previous or if we reached
            # the end of tk_list, pass has ended
            if ((tk - tk_prior) >= (max_gap+1.) or tk == tk_list[-1]):
                
                # Store stop time if at end of tk_list
                if tk == tk_list[-1]:
                    stop = tk
                
                # Store output
                start_list.append(start)
                stop_list.append(stop)
                
                if len(stop_list_all) == 0:
                    stop_list_all.append(stop)
                    obj_id_list_all.append(obj_id)
                    
                else:                
                    ind = bisect.bisect_right(stop_list_all, stop)
                    stop_list_all.insert(ind, stop)
                    obj_id_list_all.insert(ind, obj_id)
                
                # Reset for new pass next round
                start = tk
                stop = tk
                tk_prior = tk
                
                
    print(stop_list_all)
    print(obj_id_list_all)
    print(len(obj_id_list_all))

              
    # Create initial baseline CDM dictionary
    # Process all risk metrics for the given RSO data, before any measurements
    # are collected
    cdm_dict = {}
    cdm_id = 0    
    for secondary_id in secondary_id_list:
        TCA = TCA_dict[secondary_id]
        cdm_dict[cdm_id] = conj.compute_risk_metrics(rso_dict, primary_id,
                                                     secondary_id, TCA,
                                                     bodies)                
        cdm_id += 1  
        

        pklFile = open(cdm_file, 'wb')
        pickle.dump([cdm_dict], pklFile, -1)
        pklFile.close()
                
    # Create CDM dictionary over time
    # Loop over stop times and compute CDMs using latest RSO estimates    
    for kk in range(len(stop_list_all)):
        
        # Retrieve object data
        tk = stop_list_all[kk]
        obj_id = obj_id_list_all[kk]
        Xk = output_dict[obj_id][tk]['state']
        Pk = output_dict[obj_id][tk]['covar']
        
        print('')
        print(obj_id)
        print('prior Pk', np.sqrt(np.diag(rso_dict[obj_id]['covar'])))
        print('post Pk', np.sqrt(np.diag(Pk)))
        
        # Update RSO dict
        rso_dict[obj_id]['epoch_tdb'] = tk
        rso_dict[obj_id]['state'] = Xk
        rso_dict[obj_id]['covar'] = Pk
        
                
        # Compute CDM data
        if obj_id == primary_id:
            
            # Loop over all secondaries and recompute
            for secondary_id in secondary_id_list:
                
                # Skip this object if past TCA
                TCA = TCA_dict[secondary_id]
                if tk > (TCA-3600.):
                    continue
                
                cdm_dict[cdm_id] = conj.compute_risk_metrics(rso_dict, primary_id,
                                                             secondary_id, 
                                                             TCA, bodies)                
                cdm_id += 1            
            
        else:
            TCA = TCA_dict[obj_id]
            
            # Skip this object if past TCA
            if tk > (TCA-3600.):
                continue
            
            cdm_dict[cdm_id] = conj.compute_risk_metrics(rso_dict, primary_id,
                                                         obj_id, TCA, bodies)
            
            cdm_id += 1
            
        # if math.fmod(cdm_id, 10) == 0:
            
        pklFile = open(cdm_file, 'wb')
        pickle.dump([cdm_dict], pklFile, -1)
        pklFile.close()
        
    
    
    # print(cdm_dict)
    
    return cdm_dict


def plot_cdm_data(cdm_file, rso_file):
    
    pklFile = open(cdm_file, 'rb')
    data = pickle.load( pklFile )
    cdm_dict = data[0]
    pklFile.close()
    
    pklFile = open(rso_file, 'rb')
    data = pickle.load( pklFile )
    rso_dict = data[0]
    pklFile.close()
    
    t0 = rso_dict[52373]['epoch_tdb']
    
    plot_dict = {}
    secondary_id_list = []
    for cdm_id in cdm_dict:
        
        cdm_data = cdm_dict[cdm_id]
        
        CDM_epoch = cdm_data['CDM_epoch']
        TCA_epoch = cdm_data['TCA_epoch']
        primary_id = cdm_data['primary_id']
        primary_data = cdm_data['primary_data']
        secondary_id = cdm_data['secondary_id']
        secondary_data = cdm_data['secondary_data']
        
        miss_distance = cdm_data['miss_distance']
        mahalanobis_distance = cdm_data['mahalanobis_distance']
        RTN_miss_distance = cdm_data['RTN_miss_distances']
        relative_velocity = cdm_data['relative_velocity']
        HBR = cdm_data['HBR']
        Pc = cdm_data['Pc2D_Foster'] 
        Uc = cdm_data['Uc2D'] 
        
        if secondary_id not in plot_dict:
            plot_dict[secondary_id] = {}
            plot_dict[secondary_id]['thrs'] = []
            plot_dict[secondary_id]['miss_distance'] = []
            plot_dict[secondary_id]['Pc'] = []
            plot_dict[secondary_id]['Uc'] = []
            
        plot_dict[secondary_id]['thrs'].append((CDM_epoch-t0)/3600.)
        plot_dict[secondary_id]['miss_distance'].append(miss_distance)
        plot_dict[secondary_id]['Pc'].append(Pc)
        plot_dict[secondary_id]['Uc'].append(Uc)
            
         
            
    secondary_id_list = sorted(list(plot_dict.keys()))
    
    # Plot miss distance    
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(secondary_id_list)+1))
    plt.figure()
    ii = 0
    for secondary_id in plot_dict:
        thrs = plot_dict[secondary_id]['thrs']
        miss_distance = plot_dict[secondary_id]['miss_distance']
        plt.semilogy(thrs, miss_distance, 'o--', color=colors[ii], label=str(secondary_id))
        ii += 1
        
    plt.ylabel('Miss Distance [m]')
    plt.xlabel('Time [hours]')
    plt.legend()
    
    # Plot Pc
    plt.figure()
    ii = 0
    for secondary_id in plot_dict:
        thrs = plot_dict[secondary_id]['thrs']
        Pc = plot_dict[secondary_id]['Pc']
        plt.semilogy(thrs, Pc, 'o--', color=colors[ii], label=str(secondary_id))
        ii += 1
        
    plt.ylabel('Pc')
    plt.xlabel('Time [hours]')
    plt.legend()
    
    # Plot Uc
    plt.figure()
    ii = 0
    for secondary_id in plot_dict:
        thrs = plot_dict[secondary_id]['thrs']
        Uc = plot_dict[secondary_id]['Uc']
        plt.semilogy(thrs, Uc, 'o--', color=colors[ii], label=str(secondary_id))
        ii += 1
        
    plt.ylabel('Uc')
    plt.xlabel('Time [hours]')
    plt.legend()
    
    
    plt.show()
    
    
    return





if __name__ == '__main__':
    
    plt.close('all')
    
    cdm_file = os.path.join('data', 'greedy_renyi_cdm_batchPo_rgazel_60sec.pkl')
    rso_file = os.path.join('data', 'rso_catalog_truth.pkl')
    plot_cdm_data(cdm_file, rso_file)


