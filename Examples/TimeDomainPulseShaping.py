import numpy as np
import math

from Examples import BaseExample
from Constellations import QAMConstellation
from Links import SMFSplitStep
from Normalization import Lumped
from Filters import FFTLowPass, DispersionCompensation, Concatenate
from Modulators import TimeDomainModulator, CarrierWaveforms

class TimeDomainPulseShaping(BaseExample):
    """Convetional time domain pulse shaping with raised cosines and a linear
    dispersion compensation filter."""

    def __init__(self):

        self.roll_off_factor = 0.5
        """Roll-off factor for the raised cosine pulses."""

        self.T0 = 2.5e-11
        """Symbol duration in s."""

        self.n_symbols_per_block = 128
        """Number of symbols per block."""

        self.n_guard_symbols = 32
        """The length of the guard interval is given by this number times T0."""

        self.beta2 = -5e-27
        """Dispersion parameter in s**2/m."""

        self.gamma = 1.2e-3
        """Nonlinearity parameter in (W m)**(-1)."""

        self.Tscale = 1.25e-9 # s
        """Time scale used during normalization in s."""

        self.alpha = np.array([0.2e-3])
        """Loss coefficient in 1/m."""

        self.post_boost = True
        """Boost at the end of each span (lumped amplification). True or
        False."""

        self.path_average = True
        """Use path-averaged fiber parameters during normalization. True or
        False."""

        self.noise = True
        """Add ASE noise during amplification (only if post_boost == True).
        True or False."""

        self.noise_figure = 6
        """Noise figure in dB."""

        self.n_spans = 8
        """Number of fiber spans."""

        self.fiber_span_length = 80e3
        """Length of a fiber span in m."""

        self.n_steps_per_span = 40
        """Number of spatial steps per span during the numerical simulation of
        the fiber transmission."""

        self.tx_bandwidth = 40e9
        """Bandwidth of the (ideal) low-pass filter at the transmitter in Hz."""

        self.rx_bandwidth = 40e9 # in GHz
        """Bandwidth of the (ideal) low-pass filter at the receiver in Hz."""

        self.constellation_level = 4
        """Level of the QAM constellation (4, 16, 256, ...)."""

        self.dispersion_compensation = True
        """Compensate for chromatic dispersion at the end of the link. True or
        False."""

        self.tx_gain = 0.065
        """Gain (in linear units) that is applied at the transmitter before
        transmission. Use for power control."""

        self.dt_factor = 16
        """Factor that determines the time domain step size, which is
        proportional to 1/dt_factor."""

        self.make_n_samples_pow2 = False
        """If true, it is ensured that the number of samples is a power of two."""

        self.reconfigure()

    def reconfigure(self):

        self._normalization = Lumped(self.beta2,
                                     self.gamma,
                                     self.T0,
                                     self.alpha,
                                     self.fiber_span_length)

        normalized_T0 = self._normalization.norm_time(self.T0)
        self.pulse_fun = lambda t : self.tx_gain*CarrierWaveforms.raised_cosine(t, self.roll_off_factor, normalized_T0)
        self.pulse_spacing = normalized_T0
        requested_normalized_dt = normalized_T0 / self.dt_factor

        m = int(math.sqrt(self.constellation_level))
        assert self.constellation_level == m*m
        self._constellation = QAMConstellation(m, m)

        self._modulator = TimeDomainModulator(self.pulse_fun,
                                              self.pulse_spacing,
                                              requested_normalized_dt,
                                              self.n_symbols_per_block,
                                              self.n_guard_symbols,
                                              self.make_n_samples_pow2)
        self.dt = self._normalization.denorm_time(self._modulator.normalized_dt)

        dz = self.fiber_span_length/self.n_steps_per_span
        self._link = SMFSplitStep(self.dt,
                                  dz,
                                  self.n_steps_per_span,
                                  self.alpha,
                                  self.beta2,
                                  self.gamma,
                                  False,
                                  self.n_spans,
                                  self.post_boost,
                                  self.noise,
                                  self.noise_figure)

        self._tx_filter = FFTLowPass(self.tx_bandwidth/2, self.dt)
        rx_lowpass = FFTLowPass(self.rx_bandwidth/2, self.dt)
        if self.dispersion_compensation:
            dcf = DispersionCompensation(self.beta2, self.fiber_span_length*self.n_spans, self.dt)
            self._rx_filter = Concatenate(rx_lowpass, dcf)
        else:
            self._rx_filter = rx_lowpass
