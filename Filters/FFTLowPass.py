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
# Sander Wahls (TU Delft) 2018

import numpy as np
from Filters import BaseFilter

class FFTLowPass(BaseFilter):
    '''Low-pass fiter implemented using the FFT.'''

    def __init__(self, cutoff_frequency_Hz, h_sec):
        """Constructor.

        Parameters
        ----------
            cutoff_frequency_Hz : float
            h_sec : float
        """
        self._cutoff_frequency_Hz = cutoff_frequency_Hz
        self._h_sec = h_sec

    def filter(self, input):
        """Removes high frequency components from a signal.

        Parameters
        ----------
        input : numpy.array(complex)
            Vector of equi-distant time domain samples q[n]=q(t0+n*h_sec),
            n=0,1,2,...,N-1, where h_sec is the sampling interval that was
            provided to __init__(...).

        Returns
        -------
        numpy.array(complex)
            Vector of the same properties as the input that represent a filtered
            version of the input signal in which frequency components outside
            the interval [-cutoff_frequency_Hz, cutoff_frequency_Hz], where
            cutoff_frequency_Hz is the value provided earlier to __init__(...),
            have been removed.
        """
        ft = np.fft.fft(input)
        N = np.size(ft)
        freq_Hz = np.fft.fftfreq(N, d=self._h_sec)
        idx = np.abs(freq_Hz) > self._cutoff_frequency_Hz
        ft[idx] = 0.0
        return np.fft.ifft(ft)
