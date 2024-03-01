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
# Marius Brehler (TU Dortmund) 2018-2019
# Marius Brehler 2019

from Links import BaseLink
from Links._DDFssprop import _DDFssprop
import numpy as np
from scipy.constants import Planck
from Links.DDF_profile import Get_Beta_Gamma_Profile

class DDFSplitStep(BaseLink):
    """Simulates a link with one or several single mode dispersion decreasing fiber spans connected by
    EDF amplifiers using a split step method based on SSPROP."""

    def __init__(self, dt, dz, nz,
                 alpha=0.0, beta2=1.0, gamma=-1.0,
                 verbose=False, n_spans=1,
                 post_boost=False, noise=False, noise_figure=3,
                 center_frequency = 193.1e12):
        """Constructor.

        Parameters
        ----------

        dt : float
            Time step in s.
        dz : float
            Spatial step in m.
        nz : int
            Number of spatial steps for one fiber span.
        alpha : float or numpy.array(float)
            Fiber loss coefficient in 1/m. It is possible to pass a vector of
            length nz in order to specify an individual loss coefficient for
            each of the segments of a span. Noise-free Raman amplification can
            be implemented using this feature.
        beta2 : float
            Fiber dispersion coefficient in s**2/m.
        gamma : float
            Fiber nonlinearity coefficient in (W m)**(-1).
        verbose : bool
            Set to True for diagnostic ouputs.
        n_spans : int
            Number of fiber spans in the link.
        post_boost : bool
            Accumulated fiber loss is compensated with a gain at the end of each
            fiber span if True. Requires alpha to be a scalar. EDFA
            amplification can be implemented using this feature.
        noise : bool
            Add amplified spontaneous emission (ASE) noise at the end of each
            fiber span. EDFA amplification can be implemented using this
            feature.
        noise_figure : float
            Noise figure in dB for determining the ASE noise if noise==True.
        center_frequency : float
            Center frequency in Hz for determining the ASE noise if noise==True.
        """
        BaseLink.__init__(self)
        self._dt = dt
        self._dz = dz
        self._nz = nz
        self._alpha = alpha *np.log(10)*0.1
        self._gain = self._dz * self._nz * alpha
        self._beta2 = beta2
        self._gamma = gamma
        self._verbose = verbose
        self._n_spans = n_spans
        self._post_boost = post_boost
        self._noise = noise
        self._noise_figure = noise_figure
        self._center_frequency = center_frequency
        self._span_length = nz*dz
        self._n_spans = n_spans
        self._BETA2 = None
        self._GAMMA = None
        self._D_z = None

    def _ASE_noise_power(self):
        '''Returns the amplified spontaneous emission (ASE) power using the
        equations 7.2.11 and 7.2.15 in the 4th edition of "Fiber-Optic
        Communication Systems" by G. P. Agrawal (Wiley 2010).'''

        if self._noise == False:
            return 0.0
        G = 10**(self._gain/10.0)
        Fn = 10**(self._noise_figure/10.0)
        n_sp = (G*Fn - 1.0)/2.0/(G - 1.0)
        return np.max([0, n_sp * Planck * self._center_frequency * (G - 1)])

    def _ASE_noise(self, n_samples):
        '''Generates a vector of white Gaussian noise whose variance is the ASE
        noise power times the simulation bandwidth.'''

        wgn = np.sqrt(0.5)*(np.random.randn(n_samples) + 1j*np.random.randn(n_samples))

        # The expected power of c*wgn, where c>0 is t.b.d., is
        #
        #  P1 = E[|c*wgn[0]|**2+...+|c*wgn[n_samples-1]|^2]/n_samples = c^2
        #
        # We want this to be equal to the integral of S_ASE(f) = ASE_noise_power
        # over the simulation bandwidth 1/dt, i.e.,

        P2 = self._ASE_noise_power() / self._dt
                   # Solving P1=c^2=P2 for c leads to

        c = np.sqrt(P2)
        return c*wgn

    def _ASE_noise_sanity_check(self):
        '''Plots an averaged periodogram of the ASE noise.'''
        import matplotlib.pyplot as plt
        fig = plt.figure()
        N = 2**18
        K = 100
        X2 = np.zeros(N)
        for k in range(0, K):
            X2 += np.abs(np.fft.fft(self._ASE_noise(N)))**2 * self._dt / N
        X2 /= (K - 1)
        f = np.fft.fftfreq(N, 1/self._dt)
        plt.semilogy(f, X2)
        plt.title("The PSD should be flat at {}".format(self._ASE_noise_power()))
        fig.show()

    def transmit(self, input):
        # Docstring is inherited from base class.
        profile = Get_Beta_Gamma_Profile(self._alpha, self._beta2, self._gamma, self._dz, self._nz)
        self._BETA2 = profile['BETA2']
        self._GAMMA = profile['GAMMA']

        if self._n_spans == 1:
            uu = _DDFssprop(input, self._dt, self._dz, self._nz, self._alpha, self._BETA2, self._GAMMA)

            if self._post_boost == True:
                if self._alpha.size != 1:
                    raise Exception('alpha array not supported together with boost')

                uu *= np.exp(self._gain*np.log(10)*0.05)
                uu += self._ASE_noise(np.size(uu))

        else:
            uu = input
            for span_i in range(0,self._n_spans):
                uu = _DDFssprop(uu, self._dt, self._dz, self._nz, self._alpha, self._BETA2, self._GAMMA)

                if self._verbose:
                    print("Finished span",span_i+1)

                if self._post_boost == True:
                    if self._alpha.size != 1:
                        raise Exception('alpha array not supported together with boost')
                    uu *= np.exp(self._gain*np.log(10)*0.05)
                    uu += self._ASE_noise(np.size(uu))

        return uu
