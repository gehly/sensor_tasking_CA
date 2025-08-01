###############################################################################
# This file contains code developed from the open source NASA CARA Analysis 
# Tools, provided under the NASA Open Source Software Agreement.
#
# Copyright © 2020 United States Government as represented by the Administrator 
# of the National Aeronautics and Space Administration. All Rights Reserved.
#
# Modified (port to python) by Steve Gehly Feb 27, 2024
#
# References:
#
#  [1] Denenberg, E., "Satellite Closest Approach Calculation Through 
#      Chebyshev Proxy Polynomials," Acta Astronautica, 2020.
#
#  [2] Hall, D.T., Hejduk, M.D., and Johnson, L.C., "Remediating Non-Positive
#      Definite State Covariances for Collision Probability Estimation," 2017.
#
#  [3] https://github.com/nasa/CARA_Analysis_Tools
#
#  [4] Foster, J., Estes, H., "A parametric analysis of orbital debris collision 
#      probability and maneuver rate for space vehicles," Tech Report, 1992.
#
#  [5] Alfano, S., "Review of Conjunction Probability Methods for Short-term 
#      Encounters," Advances in the Astronautical Sciences, Vol. 127, Jan 2007, 
#      pp 719-746.
#
#  [6] Alfano, S., "Satellite Conjuction Monte Carlo Analysis," AAS Spaceflight
#      Mechanics Meeting (AAS-09-233), 2009.
#
#  [7] Delande, E.D., Jones, B.A., and Jah, M.K., "Exploring an Alternative
#      Approach to the Assessment of Collision Risk," JGCD, 2023.
#  
#
#
###############################################################################

import numpy as np
import math
from datetime import datetime
from scipy.integrate import dblquad
from scipy.special import erfcinv
from scipy.optimize import fsolve
import pickle
import time
import os
import matplotlib.pyplot as plt

import TudatPropagator as prop

###############################################################################
# Basic I/O
###############################################################################

def read_cdm_file(cdm_file):
    '''
    This function reads a text file containing for a Conjunction Data Message
    (CDM) and returns a dictionary containing the same data, indexed by 
    5 digit NORAD ID.
    
    Parameters
    ------
    cdm_file : string
        path and filename of pickle file containing CDM data
    
    Returns
    ------
    cdm_data : dict
        extracted CDM data including TCA, miss distance, Pc [meters, seconds] 
    
    '''
    
    cdm_data = {}
    
    with open(cdm_file) as file:
        lines = [line.rstrip() for line in file]        
    
        
    for ii in range(len(lines)):
        line = lines[ii]
                
        try:
            key, value = [x.strip() for x in line.split('=')]
            

            
            if key == 'TCA':
                cdm_data['TCA_UTC'] = datetime.fromisoformat(value)
                
            if key == 'MISS_DISTANCE':
                md, unit = value.split()
                cdm_data['MISS_DISTANCE'] = float(md)
                
            if key == 'RELATIVE_SPEED':
                speed, unit = value.split()
                cdm_data['RELATIVE_SPEED'] = float(speed)
                
            if key == 'RELATIVE_POSITION_R':
                r_pos, unit = value.split()
                r_pos = float(r_pos)
                
            if key == 'RELATIVE_POSITION_T':
                t_pos, unit = value.split()
                t_pos = float(t_pos)
            
            if key == 'RELATIVE_POSITION_N':
                n_pos, unit = value.split()
                n_pos = float(n_pos)
                
            if key == 'RELATIVE_VELOCITY_R':
                r_vel, unit = value.split()
                r_vel = float(r_vel)
                
            if key == 'RELATIVE_VELOCITY_T':
                t_vel, unit = value.split()
                t_vel = float(t_vel)
                
            if key == 'RELATIVE_VELOCITY_N':
                n_vel, unit = value.split()
                n_vel = float(n_vel)
                
            if key == 'COLLISION_PROBABILITY':
                Pc = float(value)
                cdm_data['Pc'] = Pc
                
            if key == 'OBJECT' and value == 'OBJECT1':
                obj1_ind = ii

                
            if key == 'OBJECT' and value == 'OBJECT2':             
                obj2_ind = ii
                
        except:
            continue
        
    # Store output data 
    rtn_pos = np.reshape([r_pos, t_pos, n_pos], (3,1))
    rtn_vel = np.reshape([r_vel, t_vel, n_vel], (3,1))
    cdm_data['rtn_pos'] = rtn_pos
    cdm_data['rtn_vel'] = rtn_vel
    
    # Retrieve data for object 1
    lines_obj1 = lines[obj1_ind:obj2_ind] 
    obj1_id, X1, P1 = read_cdm_state_covar(lines_obj1)
    
    cdm_data['obj1'] = {}
    cdm_data['obj1']['obj_id'] = obj1_id
    cdm_data['obj1']['X'] = X1
    cdm_data['obj1']['P'] = P1
    
    # Retrieve data for object 2
    lines_obj2 = lines[obj2_ind:] 
    obj2_id, X2, P2 = read_cdm_state_covar(lines_obj2)
    
    cdm_data['obj2'] = {}
    cdm_data['obj2']['obj_id'] = obj2_id
    cdm_data['obj2']['X'] = X2
    cdm_data['obj2']['P'] = P2

    
    return cdm_data


def read_cdm_state_covar(lines):
    
    for ii in range(len(lines)):
        line = lines[ii]
                
        try:
            key, value = [x.strip() for x in line.split('=')]
            
            if key == 'OBJECT_DESIGNATOR':
                obj_id = int(value)
                        
            if key == 'X':
                x_pos, unit = value.split()
                x_pos = float(x_pos)
                if unit == '[km]':
                    x_pos *= 1000.
            
            if key == 'Y':
                y_pos, unit = value.split()
                y_pos = float(y_pos)
                if unit == '[km]':
                    y_pos *= 1000.
                    
            if key == 'Z':
                z_pos, unit = value.split()
                z_pos = float(z_pos)
                if unit == '[km]':
                    z_pos *= 1000.
                    
            if key == 'X_DOT':
                x_vel, unit = value.split()
                x_vel = float(x_vel)
                if unit == '[km/s]':
                    x_vel *= 1000.
                    
            if key == 'Y_DOT':
                y_vel, unit = value.split()
                y_vel = float(y_vel)
                if unit == '[km/s]':
                    y_vel *= 1000.
                    
            if key == 'Z_DOT':
                z_vel, unit = value.split()
                z_vel = float(z_vel)
                if unit == '[km/s]':
                    z_vel *= 1000.
            
            if key == 'CR_R':
                CR_R, unit = value.split()
                CR_R = float(CR_R)
                
            if key == 'CT_R':
                CT_R, unit = value.split()
                CT_R = float(CT_R)
                
            if key == 'CT_T':
                CT_T, unit = value.split()
                CT_T = float(CT_T)
                
            if key == 'CN_R':
                CN_R, unit = value.split()
                CN_R = float(CN_R)
                
            if key == 'CN_T':
                CN_T, unit = value.split()
                CN_T = float(CN_T)
                
            if key == 'CN_N':
                CN_N, unit = value.split()
                CN_N = float(CN_N)
                
            if key == 'CRDOT_R':
                CRDOT_R, unit = value.split()
                CRDOT_R = float(CRDOT_R)
                
            if key == 'CRDOT_T':
                CRDOT_T, unit = value.split()
                CRDOT_T = float(CRDOT_T)
                
            if key == 'CRDOT_N':
                CRDOT_N, unit = value.split()
                CRDOT_N = float(CRDOT_N)
                
            if key == 'CRDOT_RDOT':
                CRDOT_RDOT, unit = value.split()
                CRDOT_RDOT = float(CRDOT_RDOT)
                
            if key == 'CTDOT_R':
                CTDOT_R, unit = value.split()
                CTDOT_R = float(CTDOT_R)
                
            if key == 'CTDOT_T':
                CTDOT_T, unit = value.split()
                CTDOT_T = float(CTDOT_T)
            
            if key == 'CTDOT_N':
                CTDOT_N, unit = value.split()
                CTDOT_N = float(CTDOT_N)
                
            if key == 'CTDOT_RDOT':
                CTDOT_RDOT, unit = value.split()
                CTDOT_RDOT = float(CTDOT_RDOT)
                
            if key == 'CTDOT_TDOT':
                CTDOT_TDOT, unit = value.split()
                CTDOT_TDOT = float(CTDOT_TDOT)
                
            if key == 'CNDOT_R':
                CNDOT_R, unit = value.split()
                CNDOT_R = float(CNDOT_R)
                
            if key == 'CNDOT_T':
                CNDOT_T, unit = value.split()
                CNDOT_T = float(CNDOT_T)
                
            if key == 'CNDOT_N':
                CNDOT_N, unit = value.split()
                CNDOT_N = float(CNDOT_N)
                
            if key == 'CNDOT_RDOT':
                CNDOT_RDOT, unit = value.split()
                CNDOT_RDOT = float(CNDOT_RDOT)
                
            if key == 'CNDOT_TDOT':
                CNDOT_TDOT, unit = value.split()
                CNDOT_TDOT = float(CNDOT_TDOT)
                
            if key == 'CNDOT_NDOT':
                CNDOT_NDOT, unit = value.split()
                CNDOT_NDOT = float(CNDOT_NDOT)
            
        except:
            continue
        
    X = np.reshape([x_pos, y_pos, z_pos, x_vel, y_vel, z_vel], (6,1))
    P = np.zeros((6,6))
    P[0,0] = CR_R
    P[1,1] = CT_T
    P[2,2] = CN_N
    P[3,3] = CRDOT_RDOT
    P[4,4] = CTDOT_TDOT
    P[5,5] = CNDOT_NDOT
    P[0,1] = P[1,0] = CT_R
    P[0,2] = P[2,0] = CN_R
    P[0,3] = P[3,0] = CRDOT_R
    P[0,4] = P[4,0] = CTDOT_R
    P[0,5] = P[5,0] = CNDOT_R
    P[1,2] = P[2,1] = CN_T
    P[1,3] = P[3,1] = CRDOT_T
    P[1,4] = P[4,1] = CTDOT_T
    P[1,5] = P[5,1] = CNDOT_T
    P[2,3] = P[3,2] = CRDOT_N
    P[2,4] = P[4,2] = CTDOT_N
    P[2,5] = P[5,2] = CNDOT_N
    P[3,4] = P[4,3] = CTDOT_RDOT
    P[3,5] = P[5,3] = CNDOT_RDOT
    P[4,5] = P[5,4] = CNDOT_TDOT
    
    return obj_id, X, P


###############################################################################
# Basic Risk Metrics
###############################################################################


def compute_euclidean_distance(r_A, r_B):
    
    d = np.linalg.norm(r_A - r_B)
    
    return d


def compute_mahalanobis_distance(r_A, r_B, P_A, P_B):    
    
    r_A = np.reshape(r_A, (3,1))
    r_B = np.reshape(r_B, (3,1))
    Psum = P_A + P_B
    invP = np.linalg.inv(Psum)
    diff = r_A - r_B
    M = float(np.sqrt(np.dot(diff.T, np.dot(invP, diff)))[0,0])
    
    # print(r_A)
    # print(r_B)
    # print(P_A)
    # print(P_B)
    
    # print(Psum)
    # print(invP)
    # print(diff)
    # print(np.dot(diff.T, np.dot(invP, diff)))
    # print(M)
    # print('')
    
    return M


###############################################################################
# 2D Probability of Collision (Pc) Functions
###############################################################################

def Pc2D_Foster(X1, P1, X2, P2, HBR, rtol=1e-8, HBR_type='circle'):
    '''
    This function computes the probability of collision (Pc) in the 2D 
    encounter plane following the method of Foster. The code has been ported
    from the MATLAB library developed by the NASA CARA team, listed in Ref 3.
    The function supports 3 types of hard body regions: circle, square, and 
    square equivalent to the area of the circle. The input covariance may be
    either 3x3 or 6x6, but only the 3x3 position covariance will be used in
    the calculation of Pc.
    
    
    Parameters
    ------
    X1 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 1 in ECI [m, m/s]
    P1 : 6x6 numpy array
        Estimated covariance of Object 1 in ECI [m^2, m^2/s^2]
    X2 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 2 in ECI [m, m/s]
    P2 : 6x6 numpy array
        Estimated covariance of Object 2 in ECI [m^2, m^2/s^2]
    HBR : float
        hard-body region (e.g. radius for spherical object) [m]
    rtol : float, optional
        relative tolerance for numerical quadrature (default=1e-8)
    HBR_type : string, optional
        type of hard body region ('circle', 'square', or 'squareEqArea')
        (default='circle')
    
    Returns
    ------
    Pc : float
        probability of collision
    
    '''
    
    # Retrieve and combine the position covariance
    Peci = P1[0:3,0:3] + P2[0:3,0:3]
    
    # Construct the relative encounter frame
    r1 = np.reshape(X1[0:3], (3,1))
    v1 = np.reshape(X1[3:6], (3,1))
    r2 = np.reshape(X2[0:3], (3,1))
    v2 = np.reshape(X2[3:6], (3,1))
    r = r1 - r2
    v = v1 - v2
    h = np.cross(r, v, axis=0)
    
    # Unit vectors of relative encounter frame
    yhat = v/np.linalg.norm(v)
    zhat = h/np.linalg.norm(h)
    xhat = np.cross(yhat, zhat, axis=0)
    
    # Transformation matrix
    eci2xyz = np.concatenate((xhat.T, yhat.T, zhat.T))
    
    # Transform combined covariance to relative encounter frame (xyz)
    Pxyz = np.dot(eci2xyz, np.dot(Peci, eci2xyz.T))
    
    # 2D Projected covariance on the x-z plane of the relative encounter frame
    red = np.array([[1., 0., 0.], [0., 0., 1.]])
    Pxz = np.dot(red, np.dot(Pxyz, red.T))

    # Exception Handling
    # Remediate non-positive definite covariances
    Lclip = (1e-4*HBR)**2.
    Pxz_rem, Pxz_det, Pxz_inv, posdef_status, clip_status = remediate_covariance(Pxz, Lclip)
    
    
    # Calculate Double Integral
    x0 = np.linalg.norm(r)
    z0 = 0.
    
    # Set up quadrature
    atol = 1e-13
    Integrand = lambda z, x: math.exp(-0.5*(Pxz_inv[0,0]*x**2. + Pxz_inv[0,1]*x*z + Pxz_inv[1,0]*x*z + Pxz_inv[1,1]*z**2.))

    if HBR_type == 'circle':
        lower_semicircle = lambda x: -np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
        upper_semicircle = lambda x:  np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR, x0+HBR, lower_semicircle, upper_semicircle, epsabs=atol, epsrel=rtol)[0])
        
    elif HBR_type == 'square':
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR, x0+HBR, z0-HBR, z0+HBR, epsabs=atol, epsrel=rtol)[0])
        
    elif HBR_type == 'squareEqArea':
        HBR_eq = HBR*np.sqrt(math.pi)/2.
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR_eq, x0+HBR_eq, z0-HBR_eq, z0+HBR_eq, epsabs=atol, epsrel=rtol)[0])
    
    else:
        print('Error: HBR type is not supported! Must be circle, square, or squareEqArea')
        print(HBR_type)
    
    return Pc


def remediate_covariance(Praw, Lclip, Lraw=[], Vraw=[]):
    '''
    This function provides a level of exception handling by detecting and 
    remediating non-positive definite covariances in the collision probability
    calculation, following the procedure in Hall et al. (Ref 2). This code has
    been ported from the MATLAB library developed by the NASA CARA team, 
    listed in Ref 3.
    
    The function employs an eigenvalue clipping method, such that eigenvalues
    below the specified Lclip value are reset to Lclip. The covariance matrix,
    determinant, and inverse are then recomputed using the original 
    eigenvectors and reset eigenvalues to ensure the output is positive (semi)
    definite. An input of Lclip = 0 will result in the output being positive
    semi-definite.
    
    Parameters
    ------
    Praw : nxn numpy array
        unremediated covariance matrix    
    
    Returns
    ------
    
    
    '''
    
    # Ensure the covariance has all real elements
    if not np.all(np.isreal(Praw)):
        print('Error: input Praw is not real!')
        print(Praw)
        return
    
    # Calculate eigenvectors and eigenvalues if not input
    if len(Lraw) == 0 and len(Vraw) == 0:
        Lraw, Vraw = np.linalg.eig(Praw)
        
    # Define the positive definite status of Praw
    posdef_status = np.sign(min(Lraw))
    
    # Clip eigenvalues if needed, and record clipping status
    Lrem = Lraw.copy()
    if min(Lraw) < Lclip:
        clip_status = True
        Lrem[Lraw < Lclip] = Lclip
    else:
        clip_status = False
        
    # Determinant of remediated covariance
    Pdet = np.prod(Lrem)
    
    # Inverse of remediated covariance
    Pinv = np.dot(Vraw, np.dot(np.diag(1./Lrem), Vraw.T))
    
    # Remediated covariance
    if clip_status:
        Prem = np.dot(Vraw, np.dot(np.diag(Lrem), Vraw.T))
    else:
        Prem = Praw.copy()
    
    
    return Prem, Pdet, Pinv, posdef_status, clip_status



###############################################################################
# 2D Outer Probability Measure (OPM) Functions
###############################################################################

def Uc2D(X1, P1, X2, P2, HBR):
    '''
    This function computes the outer probability measure (Uc) in the 2D 
    encounter plane. The input covariance may be either 3x3 or 6x6, but only 
    the 3x3 position covariance will be used in the calculation of Uc.
    
    
    Parameters
    ------
    X1 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 1 in ECI [m, m/s]
    P1 : 6x6 numpy array
        Estimated covariance of Object 1 in ECI [m^2, m^2/s^2]
    X2 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 2 in ECI [m, m/s]
    P2 : 6x6 numpy array
        Estimated covariance of Object 2 in ECI [m^2, m^2/s^2]
    HBR : float
        hard-body region (e.g. radius for spherical object) [m]
    
    Returns
    ------
    Pc : float
        probability of collision
    
    '''
    
    # Retrieve and combine the position covariance
    Peci = P1[0:3,0:3] + P2[0:3,0:3]
    
    # Construct the relative encounter frame
    r1 = np.reshape(X1[0:3], (3,1))
    v1 = np.reshape(X1[3:6], (3,1))
    r2 = np.reshape(X2[0:3], (3,1))
    v2 = np.reshape(X2[3:6], (3,1))
    r = r1 - r2
    v = v1 - v2
    h = np.cross(r, v, axis=0)
    
    # Degenerate case: if norm(r) < HBR the possibility Uc = 1.0
    MD = np.linalg.norm(r)
    if MD < HBR:
        return 1.0
    
    # Unit vectors of relative encounter frame
    yhat = v/np.linalg.norm(v)
    zhat = h/np.linalg.norm(h)
    xhat = np.cross(yhat, zhat, axis=0)
    
    # Transformation matrix
    eci2xyz = np.concatenate((xhat.T, yhat.T, zhat.T))
    
    # Transform combined covariance to relative encounter frame (xyz)
    Pxyz = np.dot(eci2xyz, np.dot(Peci, eci2xyz.T))
    
    # 2D Projected covariance on the x-z plane of the relative encounter frame
    red = np.array([[1., 0., 0.], [0., 0., 1.]])
    Pxz = np.dot(red, np.dot(Pxyz, red.T))

    # Exception Handling
    # Remediate non-positive definite covariances
    Lclip = (1e-4*HBR)**2.
    Pxz_rem, Pxz_det, Pxz_inv, posdef_status, clip_status = remediate_covariance(Pxz, Lclip)
    
    # # Setup and solve set of equations
    # params = {}
    # params['HBR'] = HBR
    # params['MD'] = MD
    # params['Pinv'] = Pxz_inv

    # root = fsolve(mahalanobis_opm, [MD, 0., 0.], args=(params))

    # # Compute Uc (Delande Eq 6)
    # x = float(root[0])
    # z = float(root[1])
    # diff = np.array([[x - MD], [z]])
    # dM = float(np.dot(diff.T, np.dot(Pxz_inv, diff))[0,0])
    # Uc = np.exp(-0.5*dM)    
    
    
    # Compute Uc (Delande Eq 6) - use particles to compute Uc
    Uc = 0.
    N = 1000
    for ii in range(N):
        theta = float(ii)/N
        xi = HBR*np.cos(theta*2*np.pi)
        zi = HBR*np.sin(theta*2*np.pi)
        diff = np.array([[xi - MD], [zi]])
        dM = float(np.dot(diff.T, np.dot(Pxz_inv, diff))[0,0])
        Uci = np.exp(-0.5*dM)   
        if Uci > Uc:
            Uc = float(Uci)
            
    
    return Uc


def mahalanobis_opm(x, params):
    '''
    This function is used in fsolve to find the point on the HBR circle that
    yields the minimum Mahalanobis distance. This corresponds to the maximum
    of the Gaussian possibility function.
    
    The minimum Mahalanobis distance is found by incorporating a Lagrange
    multiplier lambda into the objective function, yielding a set of 3 
    equations and 3 unknowns.
    
    F = p11*(x-MD)^2 + 2*p12*(x-MD)*z + p22*z^2 + lambda*(x^2 + z^2 - HBR^2)
    
    x^2 + z^2 - HBR^2 = 0
    2*p11*(x-MD) + 2*p12*z + 2*lambda*x = 0
    2*p22*z + 2*p12*(x-MD) + 2*lambda*z = 0

    Parameters
    ------
    x : list
        coordinates in the encounter plane and Lagrange multiplier used for
        minimization [x, z, lambda]
    params : dictionary
        contains inverse of covariance matrix in encounter plane, miss 
        distance, and hardbody radius
        
    Returns
    ------
    list of equations equal to zero, to be solved by fsolve
    
    '''
    
    Pinv = params['Pinv']
    HBR = params['HBR']
    MD = params['MD']
    
    p11 = float(Pinv[0,0])
    p12 = float(Pinv[0,1])
    p22 = float(Pinv[1,1])
    
    return [x[0]**2. + x[1]**2. - HBR**2.,
            2.*p11*(x[0] - MD) + 2.*p12*x[1] + 2.*x[0]*x[2],
            2.*p22*x[1] + 2.*p12*(x[0] - MD) + 2.*x[1]*x[2]]



###############################################################################
# Time of Closest Approach (TCA) Calculation
###############################################################################


def compute_TCA(X1, X2, trange, rso1_params, rso2_params, int_params, 
                bodies=None, rho_min_crit=0., N=16, subinterval_factor=0.5):
    '''
    This function computes the Time of Closest Approach using Chebyshev Proxy
    Polynomials. Per Section 3 of Denenberg (Ref 1), the function subdivides
    the full time interval specified by trange into units no more than half the 
    orbit period of the smaller orbit, which should contain at most one local
    minimum of the relative distance between orbits. The funtion will either
    output the time and Euclidean distance at closest approach over the full
    time interval, or a list of all close approaches under a user-specified
    input rho_min_crit.
    
    Parameters
    ------
    X1 : 6x1 numpy array
        cartesian state vector of object 1 in ECI [m, m/s]
    X2 : 6x1 numpy array
        cartesian state vector of object 2 in ECI [m, m/s]
    trange : 2 element list or array [t0, tf]
        initial and final time for the full interval [sec since J2000]
    rso1_params : dictionary
        state_params of object 1
    rso2_params : dictionary
        state_params of object 2
    int_params : dictionary
        integration parameters
    bodies : tudat object, optional
        contains parameters for the environment bodies used in propagation        
    rho_min_crit : float, optional
        critical value of minimum distance (default=0.)
        if > 0, output will contain all close approaches under this distance
    N : int, optional
        order of the Chebyshev Proxy Polynomial approximation (default=16)
        default corresponds to recommended value from Denenberg Section 3
    subinterval_factor : float, optional
        factor to multiply smaller orbit period by (default=0.5)
        default corresponds to recommended value from Denenberg Section 3
        
    Returns
    ------
    T_list : list
        time in seconds since J2000 at which relative distance between objects
        is minimum or under rho_min_crit
    rho_list : list
        list of ranges between the objects
    
    '''
    
    # Setup Tudat propagation if needed
    if bodies is None:
        bodies = prop.tudat_initialize_bodies()
        
    # Setup first time interval
    subinterval = compute_subinterval(X1, X2, subinterval_factor)
    t0 = trange[0]
    a = trange[0]
    b = min(trange[-1], a + subinterval)
        
    # Compute interpolation matrix for Chebyshev Proxy Polynomials of order N
    # Note that this can be reused for all polynomial approximations as it
    # only depends on the order
    interp_mat = compute_interpolation_matrix(N)
    
    # Loop over times in increments of subinterval until end of trange
    T_list = []
    rho_list = []
    rho_min = np.inf
    tmin = 0.
    while b <= trange[1]:
        
        print('')
        print('current interval [t0, tf]')
        print(a, b)
        print('dt [sec]', b-a)
        print('dt total [hours]', (b-trange[0])/3600.)
    
        # Determine Chebyshev-Gauss-Lobato node locations
        tvec = compute_CGL_nodes(a, b, N)
        
        # Evaluate function at node locations
        gvec, dum1, dum2, dum3, X1out, X2out, ti_out =  gvec_tudat(t0, tvec, X1, X2, rso1_params, rso2_params, int_params, bodies)
        # print('gvec', gvec)
        
        
        # Find the roots of the relative range rate g(t)
        troots = compute_gt_roots(gvec, interp_mat, a, b)
        # print('troots', troots)
        
        # If this is first pass, include the interval endpoints for evaluation
        if np.isinf(rho_min):
            troots = np.concatenate((troots, np.array([trange[0], trange[-1]])))
            
                
        # Check if roots constitute a global minimum and/or are below the
        # critical threshold
        if len(troots) > 0:
            
            # dum, rvec, ivec, cvec = gvec_fcn(t0, troots, X1, X2, params)
            # dum, rvec, ivec, cvec =  gvec_tudat(t0, troots, X1, X2, rso1_params, rso2_params, int_params, bodies)
            dum1, rvec, ivec, cvec, dum2, dum3, dum4 = gvec_tudat(t0, troots, X1, X2, rso1_params, rso2_params, int_params, bodies)
            for ii in range(len(troots)):
                rho = np.sqrt(rvec[ii]**2 + ivec[ii]**2 + cvec[ii]**2)
                
                # print('ti', troots[ii])
                # print('rho', rho)
                
                # Store if below critical threshold
                if rho < rho_min_crit:
                    rho_list.append(rho)
                    T_list.append(troots[ii])
                
                # Update global minimum
                if rho < rho_min:
                    rho_min = rho
                    tmin = troots[ii]
            
        # Increment time interval
        if b == trange[-1]:
            break
        
        a = float(b)
        if b + subinterval <= trange[-1]:
            b += subinterval
        else:
            b = trange[-1]  
            
        # Update state vectors for next iteration
        X1 = X1out
        X2 = X2out
        t0 = ti_out
            
    # # Evaluate the relative range at the endpoints of the interval to ensure
    # # these are not overlooked
    # dum, rvec, ivec, cvec = gvec_fcn(trange, X1, X2, params)
    # rho0 = np.sqrt(rvec[0]**2 + ivec[0]**2 + cvec[0]**2)
    # rhof = np.sqrt(rvec[-1]**2 + ivec[-1]**2 + cvec[-1]**2)
    
    # # Store the global minimum and append to lists if others are below 
    # # critical threshold
    # rho_candidates = [rho_min, rho0, rhof]
    # global_min = min(rho_candidates)
    # global_tmin = [tmin, trange[0], trange[-1]][rho_candidates.index(global_min)]
    
    
    
    # if ((rho0 < rho_min_crit) and (rhof < rho_min_crit)) or (rho0 == rhof):
    #     T_list = [trange[0], trange[-1]]
    #     rho_list = [rho0, rhof]
    # elif rho0 < rhof:
    #     T_list = [trange[0]]
    #     rho_list = [rho0]
    # elif rhof < rho0:
    #     T_list = [trange[-1]]
    #     rho_list = [rhof]   
        
    # If a global minimum has been found, store output
    if rho_min < np.inf and tmin not in T_list:
        T_list.append(tmin)
        rho_list.append(rho_min)
    
    # # Otherwise, compute and store the minimum range and TCA using the 
    # # endpoints of the interval
    # else:
           
        
    # Sort output
    if len(T_list) > 1:
        sorted_inds = np.argsort(T_list)
        T_list = [T_list[ii] for ii in sorted_inds]
        rho_list = [rho_list[ii] for ii in sorted_inds]
    
    return T_list, rho_list




def gvec_tudat(t0, tvec, X1, X2, rso1_params, rso2_params, int_params, bodies):
    '''
    This function computes terms for the Denenberg TCA algorithm.
    
    '''
    
    # Compute function values to find roots of
    # In order to minimize rho, we seek zeros of first derivative
    # f(t) = dot(rho_vect, rho_vect)
    # g(t) = df/dt = 2*dot(drho_vect, rho_vect)
    gvec = np.zeros(len(tvec),)
    rvec = np.zeros(len(tvec),)
    ivec = np.zeros(len(tvec),)
    cvec = np.zeros(len(tvec),)
    jj = 0
    for ti in tvec:
        
        if ti == t0:
        # if ti == tvec[0]:
            X1_t = X1
            X2_t = X2
            
        else:
            tin = [t0, ti]
            # tin = [tvec[0], ti]
            tout1, Xout1 = prop.propagate_orbit(X1, tin, rso1_params, int_params, bodies)
            tout2, Xout2 = prop.propagate_orbit(X2, tin, rso2_params, int_params, bodies)
            
            X1_t = Xout1[-1,:]
            X2_t = Xout2[-1,:]        
        
        rc_vect = X1_t[0:3].reshape(3,1)
        vc_vect = X1_t[3:6].reshape(3,1)
        rd_vect = X2_t[0:3].reshape(3,1)
        vd_vect = X2_t[3:6].reshape(3,1)
        
        rho_eci = rd_vect - rc_vect
        drho_eci = vd_vect - vc_vect
        rho_ric = eci2ric(rc_vect, vc_vect, rho_eci)
        drho_ric = eci2ric_vel(rc_vect, vc_vect, rho_ric, drho_eci)
        
        rho_ric = rho_ric.flatten()
        drho_ric = drho_ric.flatten()
        
        # print('')
        # print('ti', ti)
        # print('X1_t', X1_t)
        # print('X2_t', X2_t)
        
        gvec[jj] = float(2*np.dot(rho_ric.T, drho_ric))
        rvec[jj] = float(rho_ric[0])
        ivec[jj] = float(rho_ric[1])
        cvec[jj] = float(rho_ric[2])
        jj += 1    
    
    
    return gvec, rvec, ivec, cvec, X1_t, X2_t, ti


def compute_CGL_nodes(a, b, N):
    '''
    This function computes the location of the Chebyshev-Gauss-Lobatto nodes
    over the interval [a,b] given the order of the Chebyshev Proxy Polynomial 
    N. Per the algorithm in Denenberg, these nodes can be computed once and 
    used to approximate the derivative of the distance function, as well as the 
    relative distance components in RIC coordinates, for the same interval.
    
    Parameters
    ------
    a : float
        lower bound of interval
    b : float
        upper bound of interval
    N : int
        order of the Chebyshev Proxy Polynomial approximation
        
    Returns
    ------
    xvec : 1D (N+1) numpy array
        CGL node locations
    
    '''
    
    # Compute CGL nodes (Denenberg Eq 11)
    jvec = np.arange(0,N+1)
    xvec = ((b-a)/2.)*(np.cos(np.pi*jvec/N)) + ((b+a)/2.)
    
    return xvec


def compute_interpolation_matrix(N):
    '''
    This function computes the (N+1)x(N+1) interpolation matrix given the order
    of the Chebyshev Proxy Polynomial N. Per the algorithm in Denenberg, this 
    matrix can be computed once and reused to approximate the derivative of the
    distance function over multiple intervals, as well as to compute the 
    relative distance components in RIC coordinates.
    
    Parameters
    ------
    N : int
        order of the Chebyshev Proxy Polynomial approximation
    
    Returns
    ------
    interp_mat : (N+1)x(N+1) numpy array
        interpolation matrix
        
    '''
    
    # Compute values of pj (Denenberg Eq 13)
    pvec = np.ones(N+1,)
    pvec[0] = 2.
    pvec[N] = 2.
    
    # Compute terms of interpolation matrix (Denenberg Eq 12)
    # Set up arrays of j,k values and compute outer product matrix
    jvec = np.arange(0,N+1)
    kvec = jvec.copy()
    jk_mat = np.dot(jvec.reshape(N+1,1),kvec.reshape(1,N+1))
    
    # Compute cosine term and pj,pk matrix, then multiply component-wise
    Cmat = np.cos(np.pi/N*jk_mat)
    pjk_mat = (2./N)*(1./np.dot(pvec.reshape(N+1,1), pvec.reshape(1,N+1)))
    interp_mat = np.multiply(pjk_mat, Cmat)
    
    return interp_mat


def compute_gt_roots(gvec, interp_mat, a, b):
    
    # Order of approximation
    N = len(gvec) - 1
    
    # Compute aj coefficients (Denenberg Eq 14)
    aj_vec = np.dot(interp_mat, gvec.reshape(N+1,1))
    
    # Compute the companion matrix (Denenberg Eq 18)
    Amat = np.zeros((N,N))
    Amat[0,1] = 1.
    Amat[-1,:] = -aj_vec[0:N].flatten()/(2*aj_vec[N])
    Amat[-1,-2] += 0.5
    for jj in range(1,N-1):
        Amat[jj,jj-1] = 0.5
        Amat[jj,jj+1] = 0.5
        
    # Compute eigenvalues
    # TODO paper indicates some eigenvalues may have small imaginary component
    # but testing seems to show this is much more significant issue, needs
    # further analysis
    eig, dum = np.linalg.eig(Amat)
    eig_real = np.asarray([np.real(ee) for ee in eig if (np.isreal(ee) and ee >= -1. and ee <= 1.)])
    roots = (b+a)/2. + eig_real*(b-a)/2.

    return roots


def compute_subinterval(X1, X2, subinterval_factor=0.5, GM=3.986004415e14):
    '''
    This function computes an appropriate length subinterval of the specified
    (finite) total interval on which to find the closest approach. Per the
    discussion in Denenberg Section 3, for 2 closed orbits, there will be at
    most 4 extrema (2 minima) during one revolution of the smaller orbit. Use
    of a subinterval equal to half this time yields a unique (local) minimum
    over the subinterval and has shown to work well in testing.
    
    Parameters
    ------
    X1 : 6x1 numpy array
        cartesian state vector of object 1 in ECI [m, m/s]
    X2 : 6x1 numpy array
        cartesian state vector of object 2 in ECI [m, m/s]
    subinterval_factor : float, optional
        factor to multiply smaller orbit period by (default=0.5)
    GM : float, optional
        gravitational parameter (default=GME) [m^3/s^2]
        
    Returns
    ------
    subinterval : float
        duration of appropriate subinterval [sec]
        
    '''
    
    # Compute semi-major axis
    a1 = compute_SMA(X1, GM)
    a2 = compute_SMA(X2, GM)
    
    # print('a1', a1)
    # print('a2', a2)
    
    # If both orbits are closed, choose the smaller to compute orbit period
    if (a1 > 0.) and (a2 > 0.):
        amin = min(a1, a2)
        period = 2.*np.pi*np.sqrt(amin**3./GM)
        
    # If one orbit is closed and the other is an escape trajectory, choose the
    # closed orbit to compute orbit period
    elif a1 > 0.:
        period = 2.*np.pi*np.sqrt(a1**3./GM)
    
    elif a2 > 0.:
        period = 2.*np.pi*np.sqrt(a2**3./GM)
        
    # If both orbits are escape trajectories, choose an arbitrary period 
    # corresponding to small orbit
    else:
        period = 3600.
        
    # print(period)

        
    # Scale the smaller orbit period by user input
    subinterval = period*subinterval_factor
    
    # print('subinterval', subinterval)

    
    return subinterval


def compute_SMA(cart, GM=3.986004415e14):
    '''
    This function computes semi-major axis given a Cartesian state vector in
    inertial coordinates.
    
    Parameters
    ------
    cart : 6x1 numpy array
        cartesian state vector in ECI [m, m/s]
    GM : float, optional
        gravitational parameter (default=GME) [m^3/s^2]
        
    Returns
    ------
    a : float
        semi-major axis [m]
    
    '''
    
    # Retrieve position and velocity vectors
    r_vect = cart[0:3].flatten()
    v_vect = cart[3:6].flatten()

    # Calculate orbit parameters
    r = np.linalg.norm(r_vect)
    v2 = np.dot(v_vect, v_vect)

    # Calculate semi-major axis
    a = 1./(2./r - v2/GM)        
    
    return a


def eci2ric(rc_vect, vc_vect, Q_eci=[]):
    '''
    This function computes the rotation from ECI to RIC and rotates input
    Q_eci (vector or matrix) to RIC.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Q_eci : 3x1 or 3x3 numpy array
      vector or matrix in ECI

    Returns
    ------
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in RIC
    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))

    # Rotate Q_eci as appropriate for vector or matrix
    if len(Q_eci) == 0:
        Q_ric = ON
    elif np.size(Q_eci) == 3:
        Q_eci = Q_eci.reshape(3,1)
        Q_ric = np.dot(ON, Q_eci)
    else:
        Q_ric = np.dot(np.dot(ON, Q_eci), ON.T)

    return Q_ric


def ric2eci(rc_vect, vc_vect, Q_ric=[]):
    '''
    This function computes the rotation from RIC to ECI and rotates input
    Q_ric (vector or matrix) to ECI.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in RIC

    Returns
    ------
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in ECI
    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))
    NO = ON.T

    # Rotate Qin as appropriate for vector or matrix
    if len(Q_ric) == 0:
        Q_eci = NO
    elif np.size(Q_ric) == 3:
        Q_eci = np.dot(NO, Q_ric)
    else:
        Q_eci = np.dot(np.dot(NO, Q_ric), NO.T)

    return Q_eci


def eci2ric_vel(rc_vect, vc_vect, rho_ric, drho_eci):
    '''
    This function computes the rotation from ECI to RIC and rotates input
    relative velocity drho_eci to RIC.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    rho_ric : 3x1 numpy array
      relative position vector in RIC
    drho_eci : 3x1 numpy array
      relative velocity vector in ECI

    Returns
    ------
    drho_ric : 3x1 numpy array
      relative velocity in RIC

    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)
    rho_ric = rho_ric.reshape(3,1)
    drho_eci = drho_eci.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))

    # Compute angular velocity vector
    dtheta = h/rc**2.
    w = np.reshape([0., 0., dtheta], (3,1))
    
    # Compute relative velocity vector using kinematic identity
    drho_ric = np.dot(ON, drho_eci) - np.cross(w, rho_ric, axis=0)

    return drho_ric


def ric2eci_vel(rc_vect, vc_vect, rho_ric, drho_ric):
    '''
    This function computes the rotation from RIC to ECI and rotates input
    relative velocity drho_ric to ECI.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    rho_ric : 3x1 numpy array
      relative position vector in RIC
    drho_ric : 3x1 numpy array
      relative velocity vector in RIC

    Returns
    ------
    drho_eci : 3x1 numpy array
      relative velocity in ECI

    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)
    rho_ric = rho_ric.reshape(3,1)
    drho_ric = drho_ric.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))
    NO = ON.T
    
    # Compute angular velocity vector
    dtheta = h/rc**2.
    w = np.reshape([0., 0., dtheta], (3,1))
    
    # Compute relative velocity vector using kinematic identity
    drho_eci = np.dot(NO, (drho_ric + np.cross(w, rho_ric, axis=0))) 
    
    return drho_eci


def unit_test_tca():
    '''
    This function performs a unit test of the compute_TCA function. The object
    parameters are such that a collision is expected 30 minutes after the
    initial epoch (zero miss distance).
    
    The TCA function is run twice, using a fixed step RK4 and variable step
    RKF78 to compare the results.
    
    '''
    
    # Initial time and state vectors
    t0 = (datetime(2024, 3, 23, 5, 30, 0) - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    
    X1 = np.array([[ 3.75944379e+05],
                   [ 6.08137408e+06],
                   [ 3.28340214e+06],
                   [-5.32161464e+03],
                   [-2.32172417e+03],
                   [ 4.89152047e+03]])
    
    X2 = np.array([[ 3.30312011e+06],
                   [-2.69542170e+06],
                   [-5.71365135e+06],
                   [-4.06029364e+03],
                   [-6.22037456e+03],
                   [ 9.09217382e+02]])
    
    # Basic setup parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create) 
    
    rso1_params = {}
    rso1_params['mass'] = 260.
    rso1_params['area'] = 17.5
    rso1_params['Cd'] = 2.2
    rso1_params['Cr'] = 1.3
    rso1_params['sph_deg'] = 8
    rso1_params['sph_ord'] = 8    
    rso1_params['central_bodies'] = ['Earth']
    rso1_params['bodies_to_create'] = bodies_to_create
    
    rso2_params = {}
    rso2_params['mass'] = 100.
    rso2_params['area'] = 1.
    rso2_params['Cd'] = 2.2
    rso2_params['Cr'] = 1.3
    rso2_params['sph_deg'] = 8
    rso2_params['sph_ord'] = 8    
    rso2_params['central_bodies'] = ['Earth']
    rso2_params['bodies_to_create'] = bodies_to_create
    
    int_params = {}
    int_params['tudat_integrator'] = 'rk4'
    int_params['step'] = 1.
    
    # Expected result
    TCA_true = 764445600.0  
    rho_true = 0.
    
    # Interval times
    tf = t0 + 3600.
    trange = np.array([t0, tf])
    
    # RK4 test
    start = time.time()
    # T_list, rho_list = compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
    #                                int_params, bodies)
    

    
    # print('')
    # print('RK4 TCA unit test runtime [seconds]:', time.time() - start)
    # print('RK4 TCA error [seconds]:', T_list[0]-TCA_true)
    # print('RK4 miss distance error [m]:', rho_list[0]-rho_true)
    
    
    # RK78 test
    int_params['tudat_integrator'] = 'rkf78'
    int_params['step'] = 10.
    int_params['max_step'] = 1000.
    int_params['min_step'] = 1e-3
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    
    
    start = time.time()
    T_list, rho_list = compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
                                   int_params, bodies)
    

    print('')
    print('RK78 TCA unit test runtime [seconds]:', time.time() - start)
    print('RK78 TCA error [seconds]:', T_list[0]-TCA_true)
    print('RK78 miss distance error [m]:', rho_list[0]-rho_true)
    
    
    
    return


def unit_test_opm():
    '''
    Unit test from Delande et al. [7] which uses data from Alfano 2009 [6]
    Test Case 1.
    
    '''
    
    cdm_file = os.path.join('unit_test', 'AlfanoTestCase01.cdm')
    cdm_data = read_cdm_file(cdm_file)
    
    # print(cdm_data)
    
    X1 = cdm_data['obj1']['X']
    P1 = cdm_data['obj1']['P']
    X2 = cdm_data['obj2']['X']
    P2 = cdm_data['obj2']['P']
    
    HBR = 15.
    # HBR = 2.
    
    # Expected solution for Pc = 1.46749549E-01 for HBR = 15
    Pc = Pc2D_Foster(X1, P1, X2, P2, HBR, rtol=1e-8, HBR_type='circle')
    
    Uc = Uc2D(X1, P1, X2, P2, HBR)
    
    print('')
    print(Pc)
    print(Uc)
    
    
    # Scale factors for covariances
    scale_factors = np.logspace(-2, 12, 25)
    Pc_list = []
    Uc_list = []
    for scale in scale_factors:
        P1_ii = scale*P1
        P2_ii = scale*P2
        
        Pc_ii = Pc2D_Foster(X1, P1_ii, X2, P2_ii, HBR, rtol=1e-8, HBR_type='circle')
        
        Uc_ii = Uc2D(X1, P1_ii, X2, P2_ii, HBR)
        
        Pc_list.append(Pc_ii)
        Uc_list.append(Uc_ii)
        
    
    plt.figure()
    plt.loglog(scale_factors, Pc_list, 'bo-')
    plt.loglog(scale_factors, Uc_list, 'ro-')
    
    plt.show()
    
    
    return



if __name__ == '__main__':
    
    plt.close('all')
    
    unit_test_tca()
    
    # unit_test_opm()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    