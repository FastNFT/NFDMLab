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

import numpy as np
from Filters import BaseFilter

class DispersionCompensation(BaseFilter):
    '''Dispersion compensation fiter implemented using the FFT.'''

    def __init__(self, beta2, link_length, h_sec):
        """Constructor.

        Parameters
        ----------
            beta2 : float
            link_length : float
            h_sec : float
        """
        self._beta2 = beta2
        self._link_length = link_length
        self._h_sec = h_sec

    def filter(self, input):
        """Removes chromatic disperion effects from a signal.

        Parameters
        ----------
        input : numpy.array(complex)
            Vector of equi-distant time domain samples q[n]=q(t0+n*h_sec),
            n=0,1,2,...,N-1, where h_sec is the sampling interval that was
            provided to __init__(...).

        Returns
        -------
        numpy.array(complex)
            Vector of the same properties as the input.
        """
        ft = np.fft.fft(input)
        N = np.size(ft)
        freq_Hz = np.fft.fftfreq(N, d=self._h_sec)
        ft *= np.exp(-0.5j*(2*np.pi*freq_Hz)**2*self._link_length*self._beta2)
        return np.fft.ifft(ft)
