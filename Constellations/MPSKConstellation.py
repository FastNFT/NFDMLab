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
# Shrinivas Chimmalgi (TU Delft) 2018

import numpy as np
from Constellations import BaseConstellation
from QualityAssessment import ConstellationDiagram

class MPSKConstellation(BaseConstellation):
    """Phase shift keying (PSK) modulation. Implements BaseConstellation."""

    def __init__(self, m, rotate_by_delta_half=False):
        """Contructor for a PSK constellation.

        Parameters
        ----------

        m : int
            Number of constellation points
        rotate_by_delta_half : Boolean
            Rotate constellation by 2pi/m
        """
        del_theta = 2*np.pi/m;
        if not rotate_by_delta_half:
            self._alphabet = np.exp(np.arange(0,m)*1j*del_theta)
        else:
            self._alphabet = np.exp((np.arange(0,m)*del_theta + del_theta/2)*1j)
        nbits = int(np.ceil(np.log2(m)))
        vals = np.arange(0,m,dtype=int)
        gray_code = np.bitwise_xor(vals,(np.floor(vals/2)).astype(int))
        self._bit_matrix  = np.zeros((m, nbits), dtype=int)
        for i in range(0,m):
           tmp_str = np.binary_repr(gray_code[i],width=nbits)
           self._bit_matrix[i,:] = [int(j) for  element in tmp_str for j in element]
        self._name = "%d-PSK Constellation" % self.size()
