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

class BaseFilter(ABC):
    '''Base class for filters. All filter classes should be derived from it.'''

    @abstractmethod
    def filter(self, input):
        """Filters a time domain signal.

        Parameters
        ----------
        input : numpy.array(complex)
            Vector of time domain samples.

        Returns
        -------
        numpy.array(complex)
            Vector of time domain samples that contains the filtered input.
        """
        pass
