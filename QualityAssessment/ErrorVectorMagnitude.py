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

class ErrorVectorMagnitude:
    '''Computes error vector magnitudes.'''

    def in_dB(self, correct_symbols, recovered_symbols):
        """Error vector magnitude in decibel.

        Parameters
        ----------

        correct_symbols : numpy.array(complex)
            Vector with correct symbols.
        recovered_symbols : numpy.array(complex)
            Vector with received symbols (same length as correct_symbols).

        Returns
        -------
        float
            10*log10(P_error / P_reference)
        """
        num = np.linalg.norm(correct_symbols - recovered_symbols)
        den = np.linalg.norm(correct_symbols)
        return 20*np.log10(num/den) # The *2 corrects the sqrt in linalg.norm
