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
from Helpers import checked_get

class BitErrorRatio:
    """Computes bit error ratios."""

    def __init__(self, constellation):
        """Constructor.

        Parameters
        ----------
        constellation : object
            Constellation object of a class derived from BaseConstellation.
        """
        self._constellation = constellation

    def compute(self, correct_symbols, recovered_symbols):
        """Computers the bit error ratio between two symbol vectors.

        Parameters
        ----------
        correct_symbols : numpy.array(complex)
            Vector of symbols drawn from the constellation.
        received_symbols : numpy.array(complex)
            Vector of symbols drawn from the constellation of the same length as
            correct_symbols.

        Returns
        -------
        ber : float
            Bit error ratio, n_err/n_bits.
        n_err : int
            Number of bit errors.
        n_bits : integer
            Total number of bits.
        """
        correct_bits = self._constellation.idx2bits(
            self._constellation.symbol2idx(correct_symbols))
        recovered_bits = self._constellation.idx2bits(
            self._constellation.symbol2idx(recovered_symbols))
        n_err = np.sum(np.sum(correct_bits != recovered_bits))
        n_bits = np.size(correct_bits)
        return n_err/n_bits, n_err, n_bits
