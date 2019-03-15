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
# License along with SSPROP; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# 02111-1307 USA
#
# Contributors:
# Sander Wahls (TU Delft) 2018-2019

import numpy as np

class ModulationEfficiency:
    """Estimates modulation efficiency."""

    def __init__(self, percentage=99):
        """Constructor

        Parameters
        ----------

        percentage : float
            This percentage is used to determine the bandwidth. See compute(...).
        """
        self._percentage = percentage

    def compute(self, t, q_tx, q_rx, nerr, nbits):
        """Estimates the modulation efficiency in bits/s/Hz.

        The estimated modulation efficiency is computed as
        nbits/duration/bandwidth, where

        - nbits is the number of transmitted bits,
        - duration is t[-1]-t[0], and
        - the bandwidth is the width of the frequency interval [-f,f] that contains the percentage of the total signal energy specified during construction.

        Parameters
        ----------
        t : numpy.array(float)
            Vector of time points in s.
        q_tx : numpy.array(complex)
            Vector containing the value of the fiber input at the time points
            specified at the time points in t.
        q_rx : numpy.array(complex)
            Vector containing the value of the fiber output at the time points
            specified at the time points in t.
        nerr : int
            Number of bit errors observed after the transmission.
        nbits : int
            Number of bits transmitted.

        Returns
        -------
        float
            The modulation efficiency in bits/s/Hz.
        float
            The gross bit rate in bits/s.
        float
            The percent bandwidth in Hz.
        """
        bandwidth_at_tx = self._compute_bandwidth(t, q_tx)
        bandwidth_at_rx = self._compute_bandwidth(t, q_rx)
        bandwidth = np.amax([bandwidth_at_tx, bandwidth_at_rx])
        duration = t[-1] - t[0]
        bit_rate = nbits / duration
        return bit_rate / bandwidth, bit_rate, bandwidth

    def _compute_bandwidth(self, t, q):
        Q = np.fft.fft(q)
        dt = t[1] - t[0]
        total_energy = dt*np.linalg.norm(Q)**2
        current_energy = dt*(abs(Q[0])**2)
        i = 1
        max_i = np.floor(np.size(Q)/2.0)
        while current_energy < self._percentage/100.0*total_energy:
            if i>max_i:
                break
            current_energy += dt*( abs(Q[i])**2 + abs(Q[-i])**2 )
            i += 1
        freq_Hz = np.fft.fftfreq(np.size(q), d=t[1]-t[0])
        return 2*freq_Hz[i]
