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

from abc import ABC, abstractmethod
import numpy as np
from QualityAssessment import ConstellationDiagram
from Helpers import checked_get

class BaseConstellation(ABC):
    """Base class for constellations such as QAM or PSK.

    Notes
    -----

    * Other constellations should be derived from this class. They need to initialize self._alphabet, which is returned by the alphabet property, self._bit_matrix, which is returned by the bit_matrix property, and self._str, which is returned by the str property.

    * Bits are represented by the numbers zero and one.
    """

    @property
    def alphabet(self):
        """numpy.array (complex) : Vector of constellation points. The number of
        points has to be a power of two."""

        return checked_get(self, "_alphabet", np.ndarray)

    @property
    def bit_matrix(self):
        """numpy.array(int) : Matrix of bit patterns.

        The size of the matrix should be N x M, where N is the size of the
        alphabet and M is the number of bits per symbol. The n-th row of the
        matrix contains the bit pattern of the n-th symbol in the alphabet. The
        allowed values in the matrix are zero and one."""

        return checked_get(self, "_bit_matrix", np.ndarray)

    @property
    def name(self):
        """str : Identifcation string for the constellation, should also specify parameters."""

        return checked_get(self, "_name", str)

    def size(self):
        """int : Number of elements in the alphabet."""

        return np.size(self.alphabet)

    def idx2symbol(self, idx):
        """Translates a vector of indices into a vector of symbols.

        Parameters
        ----------
        idx : numpy.array(integer)
            Vector of indices.

        Returns
        -------
        numpy.array(complex)
            A array with the same length as idx whose n-th element is
            self.alphabet[idx[n]].
        """

        ni = np.size(idx)
        symbols = np.zeros(ni, dtype=complex)
        for i in range(0, ni):
            symbols[i] = self.alphabet[idx[i]]
        return symbols

    def symbol2idx(self, symbols):
        """Translates a vector of symbols into a vector of indices.

        Parameters
        ----------
        symbols : numpy.array(complex)
            Vector of symbols.

        Returns
        -------
        numpy.array(integer)
            A vector with the same length as symbols whose n-th element is the
            integer number i=i(n) in 0,...,numpy.size(symbols)-1 that minimizes
            the Eucledian distance \|symbols[n] - self.alphabet[i]\|.
        """

        ns = np.size(symbols)
        idx = np.zeros(ns, dtype=int)
        for i in range(0, ns):
            dists = np.abs(symbols[i] - self.alphabet)
            idx[i] = np.argmin(dists)
        return idx

    def idx2bits(self, idx):
        """Translates a vector of indices into a bit pattern.

        Parameters
        ----------
        idx : numpy.array(complex)
            Vector of indices

        Returns
        -------
        numpy.array(complex)
            Vector of bits
        """
        if np.size(idx)==1:
            return self.bit_matrix[idx, :]
        bits = np.zeros((np.size(idx), np.size(self.bit_matrix, 1)), dtype=int)
        for i in range(0, np.size(idx)):
            bits[i,:] = self.bit_matrix[idx[i], :]
        return bits

    def show(self, new_fig=False):
        """Shows the constellation alphabet together with the bit patterns.

        Parameters
        ----------

        new_fig : bool
            Always creates a new figure if True, reuses existing figures if False.

        Returns
        -------

        Figure
            matplotlib figure handle (only if new_fig=True)
        """
        constellation_diagram = ConstellationDiagram(self)
        return constellation_diagram.plot(np.array([]),
                                          show_bits=True,
                                          title=self.name,
                                          new_fig=new_fig)

    def gray_code(self, m, n):
        """Gray code bit matrix.

        Parameters
        ----------

        m : int
        n : int

        Returns
        -------

        numpy.array(int)
            Matrix as described in the bit_matrix property. The number of bit patterns (rows) is m*n. The number of bits per pattern (columns) is ceil(log2(m*n)).
        """
        bit_matrix = np.array([[0], [1]], dtype=int)
        for i in range(0, int(np.ceil(np.log2(m*n)))-1):
            r = np.size(bit_matrix, 0)
            tmp1 = np.hstack((np.zeros((r, 1), dtype=int), bit_matrix))
            tmp2 = np.hstack((np.ones((r, 1), dtype=int), np.flipud(bit_matrix)))
            bit_matrix = np.vstack((tmp1, tmp2))
        for i in range(1, m+1, 2):
            bit_matrix[i*n:(i+1)*n, :] = np.flipud(bit_matrix[i*n:(i+1)*n, :])
        return bit_matrix
