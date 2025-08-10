import numpy as np
import matplotlib.pyplot as plt
import bisect
import os
import pickle
import math

import ConjunctionUtilities as conj
import TudatPropagator as prop


def compute_errors(truth_dict, output_dict, obj_id):
    
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
    
    for kk in range(len(tk_list)):
        tk = tk_list[kk]
        X = filter_output[tk]['state']
        P = filter_output[tk]['covar']
        
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
    
    
    
    
    return


def risk_metric_evolution(output_dict, rso_dict, primary_id, tf, bodies=None):
    
    if bodies is None:
        bodies_to_create = ['Sun', 'Earth', 'Moon']
        bodies = prop.tudat_initialize_bodies(bodies_to_create)
        
    # Secondary object id's
    secondary_id_list = sorted(list(rso_dict.keys()))
    del_ind = secondary_id_list.index(primary_id)
    del secondary_id_list[del_ind]
    
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
        cdm_dict[cdm_id] = conj.compute_risk_metrics(rso_dict, primary_id,
                                                     secondary_id, tf,
                                                     bodies)                
        cdm_id += 1  
    

                
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
                cdm_dict[cdm_id] = conj.compute_risk_metrics(rso_dict, primary_id,
                                                             secondary_id, tf,
                                                             bodies)                
                cdm_id += 1            
            
        else:
            cdm_dict[cdm_id] = conj.compute_risk_metrics(rso_dict, primary_id,
                                                         obj_id, tf, bodies)
            
            cdm_id += 1
            
        if math.fmod(cdm_id, 10) == 0:
            
            print('cdm_id', cdm_id)
            cdm_file = os.path.join('data', 'baseline_cdm_data.pkl')
            pklFile = open( cdm_file, 'wb' )
            pickle.dump([cdm_dict], pklFile, -1)
            pklFile.close()
        
    
    
    print(cdm_dict)
    
    return cdm_dict








