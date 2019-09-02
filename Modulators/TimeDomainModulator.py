import numpy as np
from Modulators import BaseModulator
from Helpers import NFSpectrum

class TimeDomainModulator(BaseModulator):
    """TODO"""

    def __init__(self,
                 pulse_fun,
                 pulse_spacing,
                 dt,
                 n_symbols_per_block,
                 n_guard_symbols):
        self._pulse_fun = pulse_fun
        self._pulse_spacing = pulse_spacing

        self._norm_dt = dt # Here: normalized = unnormalized (= no normalization neccessary)
        self._n_samples = int(pulse_spacing/dt * (n_symbols_per_block + n_guard_symbols))
        self._n_symbols_per_block = n_symbols_per_block
        self._normalized_dt = dt

        self._t = (np.arange(0, self.n_samples) - (self.n_samples-1)/2)*dt
        self._pulse_centers = (np.arange(0, n_symbols_per_block)-(n_symbols_per_block-1)/2.0)*pulse_spacing
        self._pulse_center_idx = np.zeros(n_symbols_per_block, dtype=int)
        for n in range(0, n_symbols_per_block):
            self._pulse_center_idx[n] = np.argmin(abs(self._t - self._pulse_centers[n]))

        vals = self._pulse_fun(self._t)
        tol = 10 * np.finfo(vals[0]).eps * np.max(np.abs(vals))
        idx = np.argwhere(np.abs(vals)>tol)
        i1 = idx[0][0]
        i2 = idx[-1][0]
        self._t_min = self._t[i1]
        self._t_max = self._t[i2]

    def modulate(self, symbols):
        nc = np.size(symbols)
        assert nc == self.n_symbols_per_block
        q_tx = np.zeros(self.n_samples, dtype=complex)
        for n in range(0, nc):
            shifted_t = self._t - self._pulse_centers[n]
            idx = np.logical_and(shifted_t>=self._t_min, shifted_t<=self._t_max)
            q_tx[idx] = q_tx[idx] + symbols[n]*self._pulse_fun(shifted_t[idx])
        return 1e-1*q_tx, NFSpectrum('none', 'none')

    def demodulate(self, q_rx):
        q_tx_hat = q_rx
        symbols = np.zeros(self.n_symbols_per_block, dtype=complex)
        scl = 1e-1*np.abs(self._pulse_fun(0.0))
        for n in range(0, self.n_symbols_per_block):
            symbols[n] = q_tx_hat[self._pulse_center_idx[n]] / scl
        return symbols, NFSpectrum('none', 'none')
