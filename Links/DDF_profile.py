import numpy as np
def Get_Beta_Gamma_Profile(alpha, beta2, gamma, dz, nz):
    ''' Returns the Beta2 and Gamma profile (vector) ie z dependent values
    The function reuqires attenuation parameter and intial value of Beta2(0) and GAMMA(0)
    and span length as input.'''
    alpha = alpha/2 # convert to field attenuation.
    c = 3e8   # light speed m/s
    lambda0 = 1.55e-6    # center wavelength
    kappa = lambda0**2*1e-6/(2*np.pi*c)
    BETA20 = abs(beta2)
    GAMMA0 = abs(gamma)
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
