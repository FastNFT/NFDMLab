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
from Constellations import BaseConstellation

class QAMConstellation(BaseConstellation):
    """Quadrature amplitude modulation (QAM) constellation (implements BaseConstellation)."""

    def __init__(self, m, n):
        """Constructor for a m x n QAM constellation.

        Parameters
        ----------

        m : int
            Number of rows in the constellation.
        n : int
            Number of columns in the constellation.
        """
        self._alphabet = np.zeros(m*n, dtype=complex)
        for i in range(0, m):
            for j in range(0, n):
                self.alphabet[i*n+j] = (i - (m-1)/2.0) + 1.0j*(j - (n-1)/2.0)
        self._alphabet = self.alphabet / np.max(np.abs(self.alphabet))
        self._bit_matrix = self.gray_code(m, n)
        self._name = "%d-QAM Constellation" % self.size()
