import numpy as np
from Modulators import BaseModulator
from Helpers import NFSpectrum, next_pow2
from Helpers import SumOfShiftedWaveforms

class TimeDomainModulator(BaseModulator):
    """TODO"""

    def __init__(self,
                 pulse_fun,
                 pulse_spacing,
                 requested_normalized_dt,
                 n_symbols_per_block,
                 n_guard_symbols,
                 make_n_samples_pow2=False):

        self._n_samples = int(pulse_spacing/requested_normalized_dt * (n_symbols_per_block + n_guard_symbols))
        T1 = self.n_samples/2*requested_normalized_dt
        T0 = -T1
        if make_n_samples_pow2:
            self._n_samples = next_pow2(self._n_samples)
        t = np.linspace(T0, T1, self.n_samples)
        self._sum_pulses = SumOfShiftedWaveforms(pulse_fun,
                                                 pulse_spacing,
                                                 n_symbols_per_block,
                                                 t)
        self._normalized_dt = (T1 - T0)/self.n_samples
        self._n_symbols_per_block = n_symbols_per_block

    def modulate(self, symbols):
        nc = np.size(symbols)
        assert nc == self.n_symbols_per_block
        q_tx = self._sum_pulses.generate_waveform(symbols)
        return q_tx, NFSpectrum('none', 'none')

    def demodulate(self, q_rx):
        symbols = self._sum_pulses.extract_symbols(q_rx)
        return symbols, NFSpectrum('none', 'none')
