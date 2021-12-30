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
# Christoph Mahnke 2018
# Marius Brehler (TU Dortmund) 2019

import numpy as np
import matplotlib.pyplot as plt
import math
from Examples import BaseExample

class LeArefBuelow2017(BaseExample):
    '''This example loosely recreates the first of the experiments
    presented in the papers

    "125 Gbps Pre-Compensated Nonlinear Frequency-Division Multiplexed
    Transmission"

    and

    "High Speed Precompensated Nonlinear Frequency-Division Multiplexed
    Transmissions"

    by Le, Aref and Buelow presented at the 43rd European Conference on
    Optical Communication (ECOC'17) and published in the Journal of
    Lightwave Technology 36(6), 2018, respectively.

    It is NFDM system with 64 sinc carriers, a symbol duration of 6ns and
    a bandwidth of 32GHz that uses 50% precompensation.'''

    def __init__(self):

        # Link parameters

        self.alpha = np.array([0.2e-3])
        """Loss coefficient in 1/m."""

        self.beta2 = -5.75e-27
        """Dispersion coefficient in s**2/m."""

        self.gamma = 1.6e-3
        """Nonlinearity coefficient in (W m)**(-1)."""

        self.fiber_type = "SMF"
        """Fiber type: standard fiber."""

        self.Tscale = 4e-10
        """Time scale used during normalization in s."""

        self.n_spans = 12
        """Number of fiber spans."""

        self.fiber_span_length = 81.3e3
        """Span length in m."""

        self.n_steps_per_span = 40
        """Number of spatial steps per span during the numerical simulation of
        the fiber transmission."""

        self.center_frequency = 193.1e12
        """Center frequency of the optical signal in Hz."""

        self.post_boost = True
        """Boost at the end of each span (lumped amplification). True or
        False."""

        self.path_average = True
        """Use path-averaged fiber parameters during normalization. True or
        False."""

        self.noise = True
        """Add ASE noise during amplification (only if post_boost == True).
        True or False."""

        self.noise_figure = 3
        """Noise figure in dB."""

        # Modulator parameters

        self.n_symbols_per_block = 64
        """Number of carriers per block."""

        self.n_samples = 512
        """Number of time domain samples used to represent one block."""

        self.block_duration = 6e-9
        """Duration of one block in s."""

        self.T0 = 2e-9
        """The parameter T0 in the paper in s."""

        self.time_shift = 1.5e-9
        """This parameter let's us align the generated signals by shifting them
        before they are truncated to block_duration. Hand-tuned. The value is
        specified in s."""

        self.constellation_type = 'QAM'
        """We support 'QAM' and 'PSK'."""

        self.constellation_level = 16
        """For QAM: 4, 16, 64, 256, .... For PSK: 8, 16, 32, ...."""

        self.roll_off_factor = 0.0
        """We use raised cosine carriers in this example. A roll-off factor of
        0.0 corresponds to a sinc, which was used in the paper. Possible values
        are between 0.0 and 1.0."""

        self.percent_precompensation = 50
        """The effect of fiber propagation leads to a phase-shift in the
        nonlinear Fourier domain, which can be compensated at the Tx
        (choose 100), Rx (choose 0) or partially at both (choose 50)."""

        self.power_control_factor = 3
        """The power_control_factor below is called "A" in the paper. No value
        was given in the paper."""

        self.use_power_normalization_map = False
        """Activates a power normalization map as descriped in Yangzhang et. al
        (Proc. OFC'17). Not used in the paper. Does not work well here as it
        seems to broaden the pulses. Since the block_duration is already
        very short, the truncation error becomes even more severe. Should be
        only with a longer block_duration and an adapted power_control_factor.
        Set to True or False."""

        # Filters

        self.tx_bandwidth = 34e9
        """Bandwidth of the (ideal) low-pass filter at the transmitter in Hz."""

        self.rx_bandwidth = 34e9 # in GHz
        """Bandwidth of the (ideal) low-pass filter at the receiver in Hz."""

        self.reconfigure()

    def reconfigure(self):

        # Normalization

        if self.path_average == True:
            from Normalization import Lumped
            self._normalization = Lumped(self.beta2,
                                         self.gamma,
                                         self.Tscale,
                                         alpha=np.mean(self.alpha)*np.log(10)*0.1,
                                         zamp=self.fiber_span_length)
        else:
            from Normalization import Normalization
            self._normalization = Losslss(self.beta2,
                                          self.gamma,
                                          self.Tscale)

        # Constellation

        if self.constellation_type == 'QAM':
            from Constellations.QAMConstellation import QAMConstellation
            m = int(math.sqrt(self.constellation_level))
            assert self.constellation_level == m*m
            self._constellation = QAMConstellation(m, m)
        elif self.constellation_type == 'PSK':
            from Constellations.MPSKConstellation import MPSKConstellation
            self._constellation = MPSKConstellation(self.constellation_level)
        else:
            raise Exception('constellation format not supported')

        # Modulator

        self._carrier_spacing = np.pi / self._normalization.norm_time(self.T0)
        normalized_time_shift = self.normalization.norm_time(self.time_shift)
        if self.roll_off_factor == 0:
            carrier_waveform = lambda xi : (np.sinc(xi/self._carrier_spacing)
                                            *np.exp(-2j*xi*normalized_time_shift))
        else:
            from Modulators.CarrierWaveforms import raised_cosine
            carrier_waveform = lambda xi : (raised_cosine(np.array(xi),
                                                          self.roll_off_factor,
                                                          self._carrier_spacing)
                                            * np.exp(-2j*xi*normalized_time_shift))

        distance = self.n_spans * self.fiber_span_length
        from Modulators import ContSpecModulator
        normalized_distance = self._normalization.norm_dist(distance)
        normalized_duration = self._normalization.norm_time(self.block_duration)

        required_normalized_dt = normalized_duration / self.n_symbols_per_block / 8
        required_dxi = self._carrier_spacing / 8
        self._modulator = ContSpecModulator(carrier_waveform,
                                            self._carrier_spacing,
                                            self.n_symbols_per_block,
                                            normalized_distance,
                                            normalized_duration*np.array([-0.5, 0.5]),
                                            required_normalized_dt,
                                            required_dxi,
                                            "b/a",
                                            self.percent_precompensation,
                                            self.power_control_factor,
                                            self.use_power_normalization_map)

        # Link

        from Links import SMFSplitStep
        dt = self._normalization.denorm_time(self.modulator.normalized_dt)
        dz = self.fiber_span_length/self.n_steps_per_span
        self._link = SMFSplitStep(dt,
                                  dz,
                                  self.n_steps_per_span,
                                  self.fiber_type,
                                  self.alpha,
                                  self.beta2,
                                  self.gamma,
                                  False,
                                  self.n_spans,
                                  self.post_boost,
                                  self.noise,
                                  self.noise_figure,
                                  self.center_frequency)

        # Filters

        from Filters import FFTLowPass
        self._tx_filter = FFTLowPass(self.tx_bandwidth/2, dt)
        self._rx_filter = FFTLowPass(self.rx_bandwidth/2, dt)
