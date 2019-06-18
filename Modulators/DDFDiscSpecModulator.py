# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:30:13 2019

@author: vinodbajaj
"""

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
import matplotlib.pyplot as plt
from FNFTpy.fnft_nsev_inverse_wrapper import nsev_inverse_wrapper, nsev_inverse_xi_wrapper
from FNFTpy.options_handling import fnft_nsev_inverse_default_options_wrapper
from FNFTpy.fnft_nsev_wrapper import nsev_wrapper
from FNFTpy.options_handling import fnft_nsev_default_options_wrapper
from Modulators import BaseModulator
from Helpers import NFSpectrum
from Helpers import next_pow2

class DDFDiscSpecModulator(BaseModulator):
    """This modulator embeds symbols in residues of a multi-soliton with fixed
    eigenvalue pattern."""

    def __init__(self, eigenvalues, residues_amplitude, normalized_distance,
                 T, required_normalized_dt):
        """Constructor.

        Parameters
        ----------
        eigenvalues : numpy.array(complex)
            Vector of eigenvalues used to generate fiber inputs.
        residues_amplitude : numpy.array(complex)
            Vector of scaling factors. The final residues for the eigenvalue is
            obtained by multiplying these factors with the symbols.
        normalized_distance : float
            Fiber length in normalized units.
        T : numpy.array(float)
            Vector of length two with T[0]<T[1]. Specifies the time interval, in
            normalized units, used for the generation of the fiber inputs.
        required_normalized_dt : float
            The modulator will choose the number of time domain samples such
            that the time step, in normalized units, does not exceed this value.
        """

        # Save some parameters for later use.

        self._eigenvalues = eigenvalues
        self._residues_amplitude = residues_amplitude
        self._normalized_distance = normalized_distance
        self._n_symbols_per_block = np.size(eigenvalues)
        assert np.size(residues_amplitude) == self.n_symbols_per_block
        self._T = T

        # Determine number of samples and time step from time step requirements

        self._n_samples = next_pow2((T[1] - T[0])/required_normalized_dt + 1)
        self._normalized_dt = (T[1] - T[0])/(self.n_samples - 1)

        # Setup options for the forward and inverse NFT

        self._opts_inv = fnft_nsev_inverse_default_options_wrapper()
        self._opts_fwd = fnft_nsev_default_options_wrapper()
        self._opts_inv.discspec_type = 1 # residues
        self._opts_fwd.discspec_type = 1 # residues
        self._opts_fwd.contspec_type = 3 # skip computation of continuous spectrum

    def _new_nfspec(self):
        """Generates an empty NFSpectrum object with the right discspec_type and
        plotting hints."""

        nfspec = NFSpectrum("none", "b/a'")
        nfspec.bound_state_plot_range = np.zeros(4)
        nfspec.bound_state_plot_range[0] = np.max(np.real(self._eigenvalues))
        nfspec.bound_state_plot_range[1] = np.min(np.real(self._eigenvalues))
        nfspec.bound_state_plot_range[3] = np.max(np.imag(self._eigenvalues))
        nfspec.bound_state_plot_range *= 1.5
        return nfspec

    def modulate(self, symbols):
        # Docstring is inherited from base class.

        # Determine residues

        nc = np.size(symbols)
        assert nc == self.n_symbols_per_block
        residues = self._residues_amplitude*symbols

        # Call inverse NFT to generate the multi-soliton

        rdict = nsev_inverse_wrapper(0,
                                     [],
                                     -1,
                                     1,
                                     nc,
                                     self._eigenvalues,
                                     residues,
                                     self.n_samples,
                                     self._T[0],
                                     self._T[1],
                                     +1,
                                     self._opts_inv)
        if rdict['return_value'] != 0:
            raise Exception("FNFT failed")

        # Save generated nonlinear Fourier spectrum for the user

        nfspec = self._new_nfspec()
        nfspec.bound_states = self._eigenvalues
        nfspec.normconsts = residues

        return rdict['q'], nfspec


    def demodulate(self, q):
        # Docstring is inherited from base class.

        # Call NFT to obtain discrete spectrum

        assert np.size(q) == self.n_samples
        rdict = nsev_wrapper(self.n_samples,
                             q,
                             self._T[0],
                             self._T[1],
                             -1,
                             1,
                             0,
                             self.n_symbols_per_block,
                             +1,
                             self._opts_fwd)
        if rdict['return_value'] != 0:
            raise Exception("FNFT failed")

        # Save nonlinear Fourier spectrum for the user, add plotting hints

        nfspec = self._new_nfspec()
        nfspec.normconsts = rdict['disc_res']
        nfspec.bound_states = rdict['bound_states']

        # Recover symbols from residues
        D_z_avg = 0.413                   # Normalized dispersion averaged over a span length of 80 km
        # Need to be updated for variable length of span and different Beta2(0).
        symbols = np.zeros(self.n_symbols_per_block, dtype=complex)
        phase_shifts = np.exp(-2j*nfspec.bound_states**2*D_z_avg*self._normalized_distance)
        if rdict['bound_states_num']>0:
            for i in range(0, self.n_symbols_per_block):
                dists = np.abs(self._eigenvalues[i] - nfspec.bound_states)
                idx = np.argmin(dists)
                symbols[i] = phase_shifts[idx]*nfspec.normconsts[idx]/self._residues_amplitude[i]
        return symbols, nfspec
