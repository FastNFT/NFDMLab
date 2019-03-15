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
from Helpers import checked_get

class BaseModulator(ABC):
    """Base class for modulators. All modulator classes should be derived from
    it.

    Modulator classes need to implement the abstract methods modulate and
    demodulate defined below, in which symbols are embedded in / extracted from
    time-domain signals. The time-domain here is normalized.

    During construction, any modulator class should set the attributes

    - _norm_dt
    - _n_samples
    - _n_symbols_per_block

    Users can access these attributes (read-only) via the properties norm_dt,
    n_samples and n_symbols_per_block that already defined in this class.
    """
    def __init__(self):
        pass

    @property
    def normalized_dt(self):
        """Time difference, in normalized units, between consecutive samples of
        fiber inputs / outputs (read-only)."""
        return checked_get(self, "_normalized_dt", float)

    @property
    def n_samples(self):
        """Number of samples per fiber input / fiber output (read-only)."""
        return checked_get(self, "_n_samples", int)

    @property
    def n_symbols_per_block(self):
        """Number of symbols embedded in each fiber input / output
        (read-only)."""
        return checked_get(self, "_n_symbols_per_block", int)

    @abstractmethod
    def modulate(self, symbols):
        """Embeds a block of symbols in a nonlinear Fourier spectrum and
        generates the corresponding fiber input.

        Parameters
        ----------
        symbols : numpy.array(complex)
            A vector of symbols, drawn from a constellation. The length of the
            vector has to be self.n_blocks_per_symbol.

        Returns
        -------
        q_tx : numpy.array(complex)
            The vector q_tx is of length self.n_samples. It contains the
            time-domain samples q_tx(n*self.norm_dt), where
            n=0,1,...,self.n_samples-1. Note that the time here is normalized.
        nfspec_tx : NFSpectrum
            Nonlinear Fourier spectrum of the generated signal q_tx(t).
        """
        pass

    @abstractmethod
    def demodulate(self, q_rx):
        """Extracts a block of symbols from a given fiber output. This includes
        equalization, i.e., removing the impact of fiber propagation.

        Parameters
        ----------
        q_rx : numpy.array(complex)
            The vector q_rx is of length self.n_samples. It contains the
            time-domain samples q_rx(n*self.norm_dt), where
            n=0,1,...,self.n_samples-1. Note that the time here is normalized.

        Returns
        -------
        symbols_rx : numpy.array(complex)
            A vector of length self.n_symbols_per_block. Note that these values
            do not have to be drawn from a specific constellation. The task of
            assigning these values to constellation points is up to the user.
        nfspec_rx : NFSpectrum
            Nonlinear Fourier spectrum of the provided signal q_rx(t)
        """
        pass
