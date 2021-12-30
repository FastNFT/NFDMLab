# This file is part of NFDMLab.
#
# NFDMLab is free software; you can redistribute it and/or
# modify it under the terms of the version 2 of the GNU General
# Public License as published by the Free Software Foundation.
#
# NFDMLab is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with NFDMLab; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# 02111-1307 USA
#
# Contributors:
# Sander Wahls (TU Delft) 2018-2019
# Shrinivas Chimmalgi (TU Delft) 2019
# Vinod Bajaj (TU Delft) 2019

import numpy as np
def Get_Beta_Gamma_Profile(alpha, beta20, gamma0, dz, nz):
    ''' Returns the group velocity dispersion parameter and nonlinear parameter axial profile (vector) ie z dependent values such that
    for given attenuation coefficient the fiber model is exactly solvable.
    The function solves a cubic equation to find suitable values of GVM paramter and nonlinear parameter.
    For details: see
    Bajaj, Vinod, et al. "Exact NFDM transmission in the presence of fiber-loss." Journal of Lightwave Technology 38.11 (2020): 3051-3058.

    USAGE

    profile = Get_Beta_Gamma_Profile(alpha, beta20, gamma0, dz, nz)


    INPUT

    alpha - power loss coefficient, ie, P=P0*exp(-alpha*z)
    betap - group velocity dispersion at the starting end of the fiber
    gamma - nonlinearity coefficient at the starting end of the fiber
    dz - propagation stepsize
    nz - number of steps to take, ie, ztotal = dz*nz

    OUTPUT

    profile - a dictionary with fiber profile parameters
              with keys:
              'Rad_z' : fiber radius along the length
              'BETA2' : group velocity dispersion parameter along the length
              'GAMMA' : nonlinear parameter along the length
              'D_z' : normalized dispersion parameter profile along the length (normalized w.r.t start-end dispersion value)
              'R_z' : normalized nonlinear parameter profile along the length (normalized w.r.t start-end nonlinear parameter value)
              'avg_D_z' : average normalized dispersion parameter (used for modified NFT)
              'avg_R_z' : average normalized nonlinear parameter
    '''
    alpha = alpha/2 # convert to field attenuation.
    c = 3e8   # light speed m/s
    lambda0 = 1.55e-6    # center wavelength
    kappa = lambda0**2*1e-6/(2*np.pi*c)
    BETA20 = abs(beta20)
    GAMMA0 = abs(gamma0)
    R0 = (BETA20/kappa + 20)/8      # core radius in micro meter
    n2 = GAMMA0*(lambda0*np.pi*(R0*1e-6)**2)/(2*np.pi)
    #dz = dz
    #z = np.arange(0,self.n_steps_per_span)*dz
    a = BETA20*R0**2
    Rad_z = np.zeros(nz)
    for i in range(0,nz):
        p = [8*kappa,-20*kappa,0,-a*np.exp(-2*alpha*dz*i)]
        Rad_roots = np.roots(p)
        ### select the one with lowest imaginary part , then use it's real part
        #Rad_img = abs(Rad_roots.imag)
        #j = 0
        #if Rad_imag[j]>=Rad_imag[j+1]:
        #    j = j+1
        #if Rad_imag[j]>=Rad_imag[j+2]:
        #    j = j+2
        Rad_z[i] = Rad_roots[0].real

    profile = {}
    profile['Rad_z'] = Rad_z
    profile['BETA2'] = -kappa*(8*np.array(Rad_z)-20)
    profile['GAMMA'] = np.divide((2*np.pi*n2), (lambda0*np.pi*(Rad_z*1e-6)**2))
    profile['D_z'] = np.divide(profile['BETA2'], profile['BETA2'][0])
    profile['R_z'] = np.divide(profile['GAMMA'], profile['GAMMA'][0])
    profile['avg_D_z'] = np.mean(profile['D_z'])
    profile['avg_R_z'] = np.mean(profile['R_z'])

    # self._BETA2 = -kappa*(8*np.array(Rad_z)-20)
    # self._GAMMA = np.divide((2*np.pi*n2),(lambda0*np.pi*(Rad_z*1e-6)**2))
    # self._D_z = np.divide(self._BETA2,self._BETA2[0])
    # self._R_z = np.divide(self._GAMMA,self._GAMMA[0])
    # self._avg_D_z = np.mean(self._D_z)
    # self._avg_R_z = np.mean(self._R_z)
    return profile
