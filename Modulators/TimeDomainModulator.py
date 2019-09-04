import numpy as np
from Modulators import BaseModulator
from Helpers import NFSpectrum
from Helpers import SumOfShiftedWaveforms

class TimeDomainModulator(BaseModulator):
    """TODO"""

    def __init__(self,
                 pulse_fun,
                 pulse_spacing,
                 dt,
                 n_symbols_per_block,
                 n_guard_symbols):

        self._norm_dt = dt # Here: normalized = unnormalized (= no normalization neccessary)
        self._n_samples = int(pulse_spacing/dt * (n_symbols_per_block + n_guard_symbols))
        self._n_symbols_per_block = n_symbols_per_block
        self._normalized_dt = dt

        t = (np.arange(0, self.n_samples) - (self.n_samples-1)/2)*dt
        self._sum_pulses = SumOfShiftedWaveforms(pulse_fun,
                                                 pulse_spacing,
                                                 n_symbols_per_block,
                                                 t)

    def modulate(self, symbols):
        nc = np.size(symbols)
        assert nc == self.n_symbols_per_block
        q_tx = self._sum_pulses.generate_waveform(symbols)
        return 1e-1*q_tx, NFSpectrum('none', 'none')

    def demodulate(self, q_rx):
        symbols = 10*self._sum_pulses.extract_symbols(q_rx)
        return symbols, NFSpectrum('none', 'none')
