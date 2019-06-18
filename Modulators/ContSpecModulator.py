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

class ContSpecModulator(BaseModulator):
    """This modulator embeds symbols in the continuous spectrum using a classic
    multi-carrier approach."""

    def __init__(self,
                 carrier_waveform_fun,
                 carrier_spacing,
                 n_symbols_per_block,
                 normalized_distance,
                 T,
                 required_normalized_dt,
                 required_dxi,
                 contspec_type,
                 percent_precompensation = 0.0,
                 power_control_factor = 1.0,
                 use_power_normalization_map = False):
        """Constructor.

        Parameters
        ----------
        carrier_waveform_fun : function
            A function that maps numpy.array(float) to numpy.array(complex). The
            inputs of this function are vectors with nonlinear frequencies xi.
            The outputs are vectors with the values of the carrier waveforms at
            the corresponding xi. See Helpers.CarrierWaveforms for examples.
        carrier_spacing : float
            Spacing between consequtive carriers in the nonlinear frequency
            domain (xi).
        n_symbols_per_block : int
            Number of carriers.
        normalized_distance : float
            Length of the fiber link in normalized units.
        T : numpy.array(float)
            Vector of length two with T[0]<T[1]. Specifies a time interval, in
            normalized units, used during the generation of the fiber inputs.
        required_normalized_dt : float
            The modulator will choose the number of time domain samples such
            that the time step, in normalized units, does not exceed this value.
        required_dxi : float
            The modulator will choose the number of samples in the nonlinear
            frequency domain such that the nonlinear frequency step is not
            larger than this value.
        contspec_type : str
            Type of continuous spectrum used for encoding data. Choose "b/a"
            for reflection coefficients and "b" for b-coefficients.
        percent_precompensation : float
            Value between 0.0 and 100.0. A value of 50.0 means that transmitter
            and receiver both compensate 50% of the phase change in the
            continuous spectrum induced by the channel. A value of 0.0 means
            that the phase changes is completely reverted at the receiver.
        power_control_factor : float
            A positive real number. The carrier waveform is effectively scaled
            by this number.
        use_power_normalization_map : bool
            If True, the modulator applies a power normalized map as described
            in Yangzhang et. al (Proc. OFC'17) to the nonlinear Fourier spectrum
            it has generated before passing it on to the inverse NFT. Currently
            only implemented for reflection coefficients (contspec_type="b/a").
        """

        # Save some given parameters for later use.

        self._carrier_waveform_fun = carrier_waveform_fun
        self._carrier_spacing = carrier_spacing
        self._n_symbols_per_block = n_symbols_per_block
        self._normalized_distance = normalized_distance
        self._T = T
        self._contspec_type = contspec_type
        self._percent_precompensation = percent_precompensation
        self._power_control_factor = power_control_factor
        self._use_power_normalization_map = use_power_normalization_map

        # Choose number of time domain samples such that the user requirements
        # on the time step are fulfilled.

        self._n_samples = next_pow2((T[1] - T[0])/required_normalized_dt + 1)
        self._normalized_dt = (self._T[1] - self._T[0])/(self.n_samples - 1)
        assert(self.normalized_dt <= required_normalized_dt)

        # The time window T provided by the user is typically too short to fit
        # the complete signal that corresponds to the generated nonlinear
        # Fourier spectra. Normally, this causes numerical errors.
        # To avoid these errors, we internally extend the time domain and
        # increase the number of time domain samples such that the time step
        # does not change. The additional samples generated outside the interval
        # T[0] <= t <= T[1] will be discarded in the end.

        self._O = 4
        self._Tnew = np.array([T[0]-self.normalized_dt*(self._O-1)*self.n_samples/2,
                               T[1]+self.normalized_dt*(self._O-1)*self.n_samples/2])
        self._n_samples_new = self._O*self.n_samples

        # We now determine the number nonlinear Fourier domain samples based on
        # the requirements on the nonlinear frequency step of the user

        self._M = self._n_samples_new
        self._dxi = np.Inf
        while self._dxi > required_dxi:
            self._M *= 2
            rv, self._XI = nsev_inverse_xi_wrapper(self._n_samples_new,
                                                   self._Tnew[0],
                                                   self._Tnew[1],
                                                   self._M)
            self._dxi = (self._XI[1] - self._XI[0])/(self._M - 1)

        # Generate some time and nonlinear frequency grids for later use.

        self._t = self._T[0] + np.arange(0, self.n_samples)*self.normalized_dt
        self._xi = self._XI[0] + np.arange(0, self._M)*self._dxi
        self._carrier_centers = (np.arange(0, n_symbols_per_block)-(n_symbols_per_block-1)/2.0)*carrier_spacing
        self._carrier_center_idx = np.zeros(n_symbols_per_block, dtype=int)
        for n in range(0, n_symbols_per_block):
            self._carrier_center_idx[n] = np.argmin(abs(self._xi - self._carrier_centers[n]))

        # Set the right type of continuos spectrum for the NFT routines.

        self._opts_inv = fnft_nsev_inverse_default_options_wrapper()
        self._opts_fwd = fnft_nsev_default_options_wrapper()
        #self._opts_inv.discretization = self._opts_fwd.discretization
        if self._contspec_type == "b":
            self._opts_fwd.contspec_type = 1 # b_of_xi
            self._opts_inv.contspec_type = 1 # ab
        elif self._contspec_type != "b/a":
            raise Exception("Unknown contspec_type '"+str(self._contspec_type)+"'")
        self._opts_fwd.discspec_type = 3 # skip computation of discrete spectrum

        # Determine interval [xi_min, xi_max] outside of which the carrier
        # waveform is effectively zero. This is essential to speed the
        # modulation process when the number of carriers is getting larger.

        vals = self._carrier_waveform_fun(self._xi)
        tol = 10 * np.finfo(vals[0]).eps * np.max(np.abs(vals))
        idx = np.argwhere(np.abs(vals)>tol)
        i1 = idx[0][0]
        i2 = idx[-1][0]
        self._xi_min = self._xi[i1]
        self._xi_max = self._xi[i2]

    def _new_nfspec(self):
        """Generates an empty NFSpectrum object with the right contspec_type and
        plotting hints."""
        nfspec = NFSpectrum(self._contspec_type, "none")
        nfspec.xi = self._xi
        nfspec.xi_plot_range = np.array([-1.5, 1.5])*self.n_symbols_per_block/2*self._carrier_spacing
        return nfspec

    def modulate(self, symbols):
        # Docstring is inherited from base class.

        # Generate a nonlinear Fourier spectrum in which the given symbols are
        # embedded.

        nc = np.size(symbols)
        assert nc == self.n_symbols_per_block
        nfspec = self._new_nfspec()
        nfspec.cont = np.zeros(self._M, dtype=complex)
        for n in range(0, nc):
            shifted_xi = self._xi - self._carrier_centers[n]
            idx = np.logical_and(shifted_xi>=self._xi_min, shifted_xi<=self._xi_max)
            nfspec.cont[idx] = nfspec.cont[idx] + symbols[n]*self._carrier_waveform_fun(shifted_xi[idx])
        for n in range(0, nc):
            i = self._carrier_center_idx[n]

        # Apply the power control factor and, if requested, the power
        # normalization map

        nfspec.cont *= self._power_control_factor
        if self._use_power_normalization_map:
            if self._contspec_type == "b":
                raise Exception("Power normalization map currently only implemented for continuous spectra of reflection coefficient type (b/a)")
            nfspec.cont = np.sqrt(np.exp(np.abs(nfspec.cont)**2) - 1.0) * np.exp(1j*np.angle(nfspec.cont))

        # Pre-equalize for the phase change induced by the channel

        nfspec.cont *= np.exp(-2.0j*self._xi**2*self._normalized_distance*(self._percent_precompensation/100))

        # Generate corresponding time domain signal on the extended grid
        # specified construction

        rdict = nsev_inverse_wrapper(self._M,
                                     nfspec.cont,
                                     self._XI[0],
                                     self._XI[1],
                                     0,
                                     [],
                                     [],
                                     self._n_samples_new,
                                     self._Tnew[0],
                                     self._Tnew[1],
                                     +1,
                                     self._opts_inv)
        if rdict['return_value'] != 0:
            raise Exception("FNFT failed")

        # Remove all samples that are not in the time range specified by the
        # user.

        q = rdict['q'][ int((self._O-1)*self.n_samples/2)
                       :int((self._O+1)*self.n_samples/2)]

        return q, nfspec

    def demodulate(self, q):
        # Docstring is inherited from base class.

        # Oversample the given time domain signal and compute the continous
        # spectrum.

        assert np.size(q) == self.n_samples
        from scipy import signal
        DO = self._O * self.n_samples
        qO, tO = signal.resample(q, DO, self._t)
        rdict = nsev_wrapper(DO,
                             qO,
                             tO[0],
                             tO[-1],
                             self._XI[0],
                             self._XI[1],
                             self._M,
                             0,
                             +1,
                             self._opts_fwd)
        if rdict['return_value'] != 0:
            raise Exception("FNFT failed")

        nfspec = self._new_nfspec()
        if self._contspec_type == "b/a":
            nfspec.cont = rdict['cont_ref']
        elif self._contspec_type == "b":
            nfspec.cont = rdict['cont_b']
        else:
            raise Exception("Unknown contspec_type '"+str(self._contspec_type)+"'")

        # Remove any remaining phase changes due to the channel that have not
        # already been removed at the transmitter

        nfspec.cont *= np.exp(-2.0j*self._xi**2*self._normalized_distance
                               *(1 - self._percent_precompensation/100))

        # Recover the symbols

        symbols = np.zeros(self.n_symbols_per_block, dtype=complex)
        scl = np.abs(self._carrier_waveform_fun(0.0)) * self._power_control_factor
        matched_fiter = True
        if matched_fiter:
            percentage_averaging = 70      # % of carrier spacing. It specifies range of 'xi' to average for symbol detection
            xi_avg_range = 0.5*self._carrier_spacing*percentage_averaging/100 # xi range for averaging
            idx_diff = int(xi_avg_range/self._dxi)         # get the index difference 
            carrier_shape_filter = self._carrier_waveform_fun(self._xi[int(self._M/2)-idx_diff:int(self._M/2)+idx_diff])# get carrier shape for the filter
            for n in range(0, self.n_symbols_per_block):
                idx_l = self._carrier_center_idx[n] - idx_diff # lower index of the xi average range; 
                idx_u = self._carrier_center_idx[n] + idx_diff # upper index of the xi average range;                 
                matched_filtered_data = np.multiply(nfspec.cont[idx_l:idx_u],carrier_shape_filter)
                symbols[n] = np.mean(matched_filtered_data)/np.mean(carrier_shape_filter)
                if self._use_power_normalization_map:
                    symbols[n] = np.sqrt(np.log(np.abs(symbols[n])**2 + 1.0)) * np.exp(1j*np.angle(symbols[n]))
                symbols[n] /= scl
            
        else:
            for n in range(0, self.n_symbols_per_block):
                symbols[n] = nfspec.cont[self._carrier_center_idx[n]]
                print('carrier centers',self._carrier_center_idx[n])
                if self._use_power_normalization_map:
                    symbols[n] = np.sqrt(np.log(np.abs(symbols[n])**2 + 1.0)) * np.exp(1j*np.angle(symbols[n]))
                symbols[n] /= scl

        return symbols, nfspec
