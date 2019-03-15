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
# Sander Wahls (TU Delft) 2019

from abc import ABC, abstractmethod
import numpy as np

class BaseLink(ABC):
    '''Base class for fiber-optic links. All link classes should be derived from
    it.

    Link classes simulate the transmission of a fiber input through a series of
    fiber spans, possibly connected by amplifies.

    Any list class should implement the abstract transmit(...) method defined
    below and set the attributes

    - _span_length
    - _n_spans

    during construction. Users can access them via the read-only properties
    span_length and n_spans.
    '''

    @property
    def span_length(self):
        """Length of a fiber span in m."""
        return self._span_length

    @property
    def n_spans(self):
        """Number of spans in the fiber link."""
        return self._n_spans

    @abstractmethod
    def transmit(self, input):
        """Simulates the transmission of a fiber input through the link.

        Parameters
        ----------
        input : numpy.array(complex)
            Vector of equispaced time domain samples representing the fiber
            input.

        Returns
        -------
        numpy.array(complex)
            Vector of equispaced time domain samples representing the fiber
            output.
        """
        pass

    def nonlinear_length(self, input):
        return 1.0 / np.max(np.abs(input)**2) / self._gamma
