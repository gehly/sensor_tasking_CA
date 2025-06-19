###############################################################################
# This file contains code to implement a conjunction risk classifier based on
# Dempster-Shafer evidence theory.
#
# References:
#
#  [1] Sanchez, L., "Robust Artificial Intelligence for Space Traffic
#      Management, PhD dissertation, 2025.
#
###############################################################################

import numpy as np
import math
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

def compute_Pc(uvec, HBR):
    '''
    This function computes the 2D Pc assuming a Gaussian distribution in the
    encounter plane, using [1] Eq 3.1.
    
    Parameters
    ------
    uvec : numpy array
        miss distance and covariance parameters
    
    Returns
    ------
    Pc : float
        probability of collision
        
    '''
    
    # Retrieve input data
    x_in = float(uvec[0])
    z_in = float(uvec[1])
    var_x = float(uvec[2])
    var_z = float(uvec[3])
    covar_xz = float(uvec[4])
    
    P0 = np.array([[var_x, covar_xz], [covar_xz, var_z]])
    
    # Rotate to new coordinates with miss distance on x-axis
    theta = math.atan2(z_in, x_in)
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    xz = np.dot(R, np.array([[x_in],[z_in]])).flatten()
    
    x0 = float(xz[0])
    z0 = float(xz[1])
    if abs(z0) < 1e-15:
        z0 = 0.
    else:
        print(xz)
        mistake
        
    Pxz = R @ P0 @ R.T
    Pxz_inv = np.linalg.inv(Pxz)
    Pxz_det = np.linalg.det(Pxz)
    
    # Compute integral
    # Set up quadrature
    rtol = 1e-8
    atol = 1e-13
    Integrand = lambda z, x: math.exp(-0.5*(Pxz_inv[0,0]*x**2. + Pxz_inv[0,1]*x*z + Pxz_inv[1,0]*x*z + Pxz_inv[1,1]*z**2.))

    lower_semicircle = lambda x: -np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
    upper_semicircle = lambda x:  np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
    Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR, x0+HBR, lower_semicircle, upper_semicircle, epsabs=atol, epsrel=rtol)[0])

    
    return Pc


def unit_test_dilution():
    '''
    This function tests Pc calculation by comparison to Sanchez Fig 3.1 [1].
    
    '''
    
    # Fixed params
    HBR = 5.
    z = 6.
    var_z = 9.
    covar_xz = 0.
    
    x_list = [0., 2., 5., 20., 50.]
    sig_x_list = list(np.logspace(-1, 3, 50))
    
    plot_Pc = np.zeros((len(x_list), len(sig_x_list)))
    for x in x_list:
        
        ii = x_list.index(x)        
        for sig_x in sig_x_list:
            
            jj = sig_x_list.index(sig_x)
            var_x = sig_x**2.
            uvec = np.array([x, z, var_x, var_z, covar_xz])
            Pc = compute_Pc(uvec, HBR)
            
            plot_Pc[ii,jj] = Pc
    
    
    plt.figure()
    plt.loglog(sig_x_list, plot_Pc[0,:], 'k--', label='x = ' + str(x_list[0]))
    plt.loglog(sig_x_list, plot_Pc[1,:], 'b--', label='x = ' + str(x_list[1]))
    plt.loglog(sig_x_list, plot_Pc[2,:], 'g--', label='x = ' + str(x_list[2]))
    plt.loglog(sig_x_list, plot_Pc[3,:], 'r--', label='x = ' + str(x_list[3]))
    plt.loglog(sig_x_list, plot_Pc[4,:], 'm--', label='x = ' + str(x_list[4]))
    plt.ylim([1e-4, 1.0])
    plt.xlim([1e-1, 1e3])
    plt.legend()
    plt.xlabel('$\sigma_x$ [m]')
    plt.ylabel('$P_c$')
    
    plt.show()
    
    
    
    return


def unit_test_sanchez_bpa():
    '''
    This function follows the example of [1] Section 3.2.1.
    
    '''
    
    # Fixed params
    mu_z = 6.
    var_z = 81.
    covar_xz = 0.
    HBR = 5.
    
    # Interval 1
    mu_x = np.array([4., 7.])
    var_x = np.array([1., 6.25])
    
    # Interval 2
    mu_x = np.array([15., 25.])
    var_x = np.array([4., 36.])
    
    
    
    
    uvec = np.array([4., 6., 1., 81., 0.])
    Pc = compute_Pc(uvec, HBR)
    
    print(Pc)
    
    return



if __name__ == '__main__':
    
    plt.close('all')
    
    unit_test_dilution()
        
    # unit_test_sanchez_bpa()













