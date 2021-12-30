# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:13:41 2019

@author: vinodbajaj
"""

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
# Shrinivas Chimmalgi (TU Delft) 2018

import numpy as np
import math

from Examples import BaseExample

class GuiEtAl2018_DDF(BaseExample):
    '''This example loosely recreates the experiment presented in the paper

    "Nonlinear frequency division multiplexing with b-modulation: shifting the
    energy barrier " by T. Gui, G. Zhou, C. Lu, A.P.T. Lau, and S. Wahls

    published in Optics Express 26(21), 2018.'''

    def __init__(self):
        # Fiber parameters

        self.beta2 = -25.491e-27
        """Dispersion parameter in s**2/m."""

        self.gamma = 1.3e-3
        """Nonlinearity parameter in (W m)**(-1)."""

        self.fiber_type = "DDF"
        """Fiber type: "DDF" dispersion decreasing fiber or "SSMF" standard single mode fiber."""
        self.Tscale = 1.25e-9 # s
        """Time scale used during normalization in s."""

        self.alpha = np.array([0.2e-3])
        """Loss coefficient in 1/m."""

        self.post_boost = True
        """Boost at the end of each span (lumped amplification). True or
        False."""

        self.path_average = False
        """Use path-averaged fiber parameters during normalization. True or
        False."""

        self.noise = True
        """Add ASE noise during amplification (only if post_boost == True).
        True or False."""

        self.noise_figure = 3
        """Noise figure in dB."""

        self.n_spans = 8
        """Number of fiber spans."""
        
        self.fiber_span_length = 80e3
        """Length of a fiber span in m."""

        self.n_steps_per_span = 500
        """Number of spatial steps per span during the numerical simulation of
        the fiber transmission."""

        # Modulator parameters
        self.constellation_type = 'PSK'
        """We support 'QAM' and 'PSK'."""

        self.constellation_level = 16
        """Level of the QAM constellation used as input for the reshaping
        process (4, 16, 256, ...)."""

        self.n_symbols_per_block = 9
        """Number of carriers."""

        self.Ed = 4
        """Desired average pulse energy in normalized units. Used during the
        reshaping of the constellation. (See the paper for details.)"""

        # Filters

        self.tx_bandwidth = 33e9
        """Bandwidth of the (ideal) low-pass filter at the transmitter in Hz."""

        self.rx_bandwidth = 33e9 # in GHz
        """Bandwidth of the (ideal) low-pass filter at the receiver in Hz."""

        self.reconfigure()

    def reconfigure(self):
        distance = self.n_spans*self.fiber_span_length #m
        from Modulators.CarrierWaveforms import flat_top
        carrier_waveform = lambda xi : flat_top(xi, T0)
        self._carrier_spacing = 15.0
        T0 = 4.5
        T = np.array([-2.25, 2.25])   # changed from -1 1

        # Normalization

        if self.path_average == True:
            from Normalization import Lumped
            self._normalization = Lumped(self.beta2,
                                         self.gamma,
                                         self.Tscale,
                                         alpha=np.mean(self.alpha)*np.log(10)*0.1,
                                         zamp=self.fiber_span_length)
        else:
            from Normalization import Lossless
            self._normalization = Lossless(self.beta2, self.gamma, self.Tscale)

        # Constellation
        if self.constellation_type == 'QAM':
            from Constellations import ReshapedQAMConstellation
            m = int(math.sqrt(self.constellation_level))
            assert self.constellation_level == m*m
            bnds = np.array([0, 4/np.abs(carrier_waveform(0.0))])
            self._constellation = ReshapedQAMConstellation(m, m,carrier_waveform,self.Ed, bnds)
        elif self.constellation_type == 'PSK':
            from Constellations import ReshapedPSKConstellation
            bnds = np.array([0,4/np.abs(carrier_waveform(0.0))])
            self._constellation = ReshapedPSKConstellation(self.constellation_level,carrier_waveform,self.Ed,bnds)
        else:
            raise Exception('Constellation not supported')


        from Links.DDF_profile import Get_Beta_Gamma_Profile
        dz = self.fiber_span_length/self.n_steps_per_span
        profile = Get_Beta_Gamma_Profile(self.alpha * np.log(10) * 0.1, self.beta2, self.gamma, dz,
                                         self.n_steps_per_span)



        # Modulator

        from Modulators import ContSpecModulator
        normalized_distance = self.normalization.norm_dist(distance)
        normalized_duration = T[1] - T[0]
        required_normalized_dt = normalized_duration/self.n_symbols_per_block/128 # originally it was 8 instead of 128 in denominator
        required_dxi = self._carrier_spacing / 8
        self._modulator = ContSpecModulator(carrier_waveform,
                                               self._carrier_spacing,
                                               self.n_symbols_per_block,
                                               normalized_distance*profile['avg_D_z'],
                                               T,
                                               required_normalized_dt,
                                               required_dxi,
                                               "b")

        # Link

        from Links import SMFSplitStep
        dt = self._normalization.denorm_time(self.modulator.normalized_dt)
        self._link = SMFSplitStep(dt, dz, self.n_steps_per_span, self.fiber_type,
                                  self.alpha, self.beta2, self.gamma, False, self.n_spans,
                                  self.post_boost, self.noise, self.noise_figure)

        # Filters

        from Filters import PassThrough
        self._tx_filter = PassThrough()
        from Filters import FFTLowPass
        self._rx_filter = FFTLowPass(self.rx_bandwidth/2, dt)
