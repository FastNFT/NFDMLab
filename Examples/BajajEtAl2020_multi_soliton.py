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
# Vinod Bajaj (TU Delft) 2019, 2021
# Sander Wahls (TU Delft) 2018-2019
# Shrinivas Chimmalgi (TU Delft) 2018
# Sander Wahls (KIT), 2024

import numpy as np
import math

from Examples import BaseExample

class BajajEtAl2020_multi_soliton(BaseExample):
    '''This example recreates the multi-soliton transmission in dispersion-decreasing
    fiber from the paper

    "Exact NFDM Transmission in the Presence of Fiber-Loss" by V. Bajaj,
    S. Chimmalgi, V. Aref, and S. Wahls,

    published in the Journal of Lightwave Technology 38(11), 2020.'''


    def __init__(self):
        # Fiber parameters        

        self.beta2 =-25.491e-27       
        """Dispersion coefficient in s**2/m."""

        self.gamma = 1.36e-3
        """Nonlinearity coefficient in (W m)**(-1)."""

        self.fiber_type = "DDF"
        """Fiber type: "DDF" dispersion for decreasing fiber or "SSMF" for standard single mode fiber."""

        self.Tscale = 4.1022e-11

        """Time scale used during normalization in s."""

        self.alpha = 0.2e-3
        """Loss coefficient in 1/m."""

        self.n_spans = 8
        """Number of fiber spans."""

        self.n_steps_per_span = 500
        """Number of spatial steps per fiber span during simulations. Use more than 2 steps per km length for the transmission over the DDF for better accuracy."""

        self.fiber_span_length = 80e3 #If you change it, then change the average dispersion (D_z) in the demodulator
        """Length of a fiber span in m."""

        self.post_boost = True
        """Boost at end of each span (lumped amplification). True or False."""

        self.path_average = False
        """Use path-average fiber parameters during normalization. True or False. False for the transmission over DDF."""

        self.noise = False
        """Add ASE noise (lumped amplification only). True or False."""

        self.noise_figure = 6
        """Noise figure in dB."""

        # Receiver bandwidth

        self.tx_bandwidth = 33*1e9
        """Bandwidth of the ideal low-pass at the transmitter in Hz."""

        self.rx_bandwidth = 33*1e9 # Hz
        """Bandwidth of the ideal low-pass at the receiver in Hz."""

        # Modulator parameters

        self.constellation_level = 4
        """Level of the QAM constellation (4, 16, 256, ...)."""

        self.eigenvalues = np.array([0.45j-0.6, 0.3j-0.4, 0.45j-0.2, 0.3j, 0.45j+0.2, 0.3j+0.4, 0.45j+0.6])
        """Eigenvalue pattern."""

        self.residues_amplitude = np.exp(np.array([11.85, 7.06, 7.69, 3.81, 1.93, -0.62, -5.43]))
        """Spectral amplitudes for each of the eigenvalues. These values get
        multiplied with symbols drawn from the constellation before pulse
        generation). """


        self.reconfigure()

    def reconfigure(self):
        distance = self.n_spans*self.fiber_span_length #m
        assert(np.size(self.eigenvalues) == np.size(self.residues_amplitude))
        T = np.array([-9*np.pi, 9*np.pi])


        # Normalization

        if self.path_average == True:
            from Normalization import Lumped
            self._normalization = Lumped(self.beta2, self.gamma,
                                         self.Tscale,
                                         alpha=np.mean(self.alpha)*np.log(10)*0.1,
                                         zamp=self.fiber_span_length)
        else:
            from Normalization import Lossless
            self._normalization = Lossless(self.beta2, self.gamma, self.Tscale)


        # Constellation

        from Constellations import QAMConstellation
        m = int(math.sqrt(self.constellation_level))
        assert self.constellation_level == m*m
        self._constellation = QAMConstellation(m, m)

        from Links.DDF_profile import Get_Beta_Gamma_Profile
        dz = self.fiber_span_length/self.n_steps_per_span
        nz = self.n_spans*self.n_steps_per_span
        profile = Get_Beta_Gamma_Profile(self.alpha * np.log(10) * 0.1, self.beta2, self.gamma, dz,
                                         self.n_steps_per_span)

        # Modulator

        from Modulators import DiscSpecModulator
        normalized_distance = self.normalization.norm_dist(distance)
        required_normalized_dt = (T[1] - T[0])/512
        self._modulator = DiscSpecModulator(self.eigenvalues,
                                            self.residues_amplitude,
                                            normalized_distance*profile['avg_D_z'],
                                            T, required_normalized_dt)

        # Link

        from Links import SMFSplitStep                                # Propagation in Dispersion Decreasing Fiber
        dt = self.normalization.denorm_time(self.modulator.normalized_dt)

        self._link = SMFSplitStep(dt, dz, self.n_steps_per_span, self.fiber_type,
                                  self.alpha, self.beta2, self.gamma,
                                  False, self.n_spans, self.post_boost,
                                  self.noise, self.noise_figure)

        # Filters

        from Filters import PassThrough
        from Filters import FFTLowPass
        self._tx_filter = PassThrough()
        self._rx_filter = FFTLowPass(self.rx_bandwidth/2, dt)
