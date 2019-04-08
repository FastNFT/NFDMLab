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

import numpy as np

from abc import ABC, abstractmethod
from Constellations import BaseConstellation
from Modulators import BaseModulator
from Links import BaseLink
from Normalization import BaseNormalization
from Filters import BaseFilter
from Helpers import NFSpectrum
from QualityAssessment import ErrorVectorMagnitude
from QualityAssessment import BitErrorRatio
from QualityAssessment import ConstellationDiagram
from QualityAssessment import ModulationEfficiency
import Helpers.plot as hplt
from Helpers import checked_get

class BaseExample(ABC):
    """Base class for examples. All example classes should be derived from this
    class.

    An example class should set all parameters that users are encouraged to
    change as public attributes during construction. The constructor should not
    ask additional parameters from the user. After setting the attributes, the
    constructor of the example class should configure itself by calling
    reconfigure(). This method needs to be implemented by the example class
    itself.

    The reconfigure method should set the following private attributes based on
    the public attributes of the example class set during construction:

    - _constellation
    - _modulator
    - _link
    - _normalization
    - _tx_filter
    - _rx_filter

    Users are provided read-only access to these private attributes through the
    properties constellation, modulator, link, normalization, tx_filter and
    rx_filter, respectively, which are already implemented in this base class.

    The private attributes above should be objects of a class derived from the
    corresponding base classes, i.e., BaseConstellation, BaseModulator,
    BaseLink, Normalization and BaseFilter, respectively.

    Users should be able to change any of the public attributes set in the
    constructor, after which they are expected to manually call reconfigure().
    """

    @property
    def constellation(self):
        """Provides read-only access to self._constellation."""
        return checked_get(self, "_constellation", BaseConstellation)

    @property
    def modulator(self):
        """Provides read-only access to self._modulator."""
        return checked_get(self, "_modulator", BaseModulator)

    @property
    def link(self):
        """Provides read-only access to self._link."""
        return checked_get(self, "_link", BaseLink)

    @property
    def normalization(self):
        """Provides read-only access to self._normalization."""
        return checked_get(self, "_normalization", BaseNormalization)

    @property
    def tx_filter(self):
        """Provides read-only access to self._tx_filter."""
        return checked_get(self, "_tx_filter", BaseFilter)

    @property
    def rx_filter(self):
        """Provides read-only access to self._rx_filter."""
        return checked_get(self, "_rx_filter", BaseFilter)

    @abstractmethod
    def reconfigure():
        """Updates self._constellation, self._modulator, self._link,
        self._normalization, self._tx_filter and self._rx_filter based on the
        public attributes set in the constructor.

        Any example needs to implement this."""
        pass

    def prepare_fiber_input(self, n_blocks, seed=None):
        """Generates random fiber inputs that can be passed to
        transceiver_fiber_input().

        The function generates n_blocks pulses. For each pulse, it draws
        self.modulator.n_symbols_per_block random symbols from the
        constellation. It then uses the modulator to generate the corresponding
        fiber inputs both in the time and nonlinear Fourier domain. The time
        domain input is denormalized to real-world units using the
        normalization.

        Parameters
        ----------
        n_blocks : int
            Number of blocks (pulses) to be generated.
        seed : None or int
            If not None, this seed is to initialize the random number generator.

        Returns
        -------
        tx_data : dictionary
            Dictionary with the fields

            "t" : numpy.array(float)
                Vector of time points.
            "q" : array
                Array of length n_blocks containing the generated pulses q_tx(t).
                Each element of the array is a numpy.array(complex) vector which
                contains the values of q_tx(t) at the corresponding time
                specified in the "t" field.
            "nfspecs" : array
                Array containing the nonlinear Fourier spectra of the generated
                pulses. Each element of the array is a NFSpectrum object.
            "symbols" : array
                Array containing the tx symbols. Each entry of the array is a
                numpy.array(complex) vector that the contains the symbols
                modulated into the corresponding pulse q(t).
            "n_blocks" : int
                Number of blocks that was provided to this function.
        """
        modulator = self.modulator
        constellation = self.constellation
        normalization = self.normalization
        n_symbols_per_block = modulator.n_symbols_per_block
        D = modulator.n_samples

        tx_data = {}
        tx_data["t"] = normalization.denorm_time(np.arange(0, n_blocks*D)
                                                 * modulator.normalized_dt)
        tx_data["q"] = np.zeros(n_blocks*D, dtype=complex)
        tx_data["nfspecs"] = []
        tx_data["symbols"] = np.zeros(n_blocks*n_symbols_per_block,
                                      dtype=complex)

        rng = np.random.RandomState(seed)
        for n in range(0, n_blocks):
            symbol_indices_tx = rng.randint(0, constellation.size(),
                                            n_symbols_per_block)
            symbols_tx_block = constellation.idx2symbol(symbol_indices_tx)
            (q_tx_block, nfspec_tx_block) = modulator.modulate(symbols_tx_block)
            tx_data["q"][n*D:(n+1)*D] = q_tx_block
            tx_data["nfspecs"].append(nfspec_tx_block)
            tx_data["symbols"][n*n_symbols_per_block:(n+1)*n_symbols_per_block] = symbols_tx_block

        tx_data["q"] = self.tx_filter.filter(tx_data["q"])
        tx_data["q"] = normalization.denorm_field(tx_data["q"])
        tx_data["n_blocks"] = n_blocks
        return tx_data

    def transceive_fiber_input(self, tx_data):
        """Transmits and demodulates fiber inputs generated by
        prepare_fiber_input().

        The function concatenates the individual time domain pulses in tx_data
        and transmits the results using the link. It then splits the received
        signal again into blocks, denormalizes each block and estimates the
        received symbols using the modulator.

        Parameters
        ----------
        tx_data : dictionary
            Data generated by prepare_fiber_input()

        Returns
        -------
        dictionary
            Dictionary with the fields

            "t" : numpy.array(float)
                Vector of time points (in secs)
            "q" : array
                Array of length n_blocks containing the fiber outputs q_rx(t).
                Each element of the array is a numpy.array(complex) vector which
                contains the values of q_rx(t) at the corresponding time specified
                in the "t" field.
            "nfspecs" : array
                Array containing the nonlinear Fourier spectra of the received
                pulses. Each element of the array is a NFSpectrum object.
            "symbols" : array
                Array containing the rx symbols. Each entry of the array is a
                numpy.array(complex) vector that the contains the symbols
                modulated into the corresponding pulse q_rx(t).
        """
        modulator = self.modulator
        constellation = self.constellation
        normalization = self.normalization
        n_symbols_per_block = modulator.n_symbols_per_block
        n_blocks = tx_data["n_blocks"]
        D = modulator.n_samples

        rx_data = {}
        rx_data["t"] = tx_data["t"]
        rx_data["q"] = self.link.transmit(tx_data["q"])
        rx_data["q"] = self.rx_filter.filter(rx_data["q"])
        rx_data["nfspecs"] = []
        rx_data["symbols"] = np.zeros(n_blocks*n_symbols_per_block,
                                      dtype=complex)
        for n in range(0, n_blocks):
            q_rx_block_norm = normalization.norm_field(rx_data["q"][n*D:(n+1)*D])
            (symbols_rx_block, nfspec_rx_block) = modulator.demodulate(q_rx_block_norm)
            rx_data["nfspecs"].append(nfspec_rx_block)
            rx_data["symbols"][n*n_symbols_per_block:(n+1)*n_symbols_per_block] = symbols_rx_block
        return rx_data

    def run(self, n_blocks, seed=None):
        """This function performs a complete simulation. It first calls
        prepare_fiber_inputs() to generate the tx_data, passes it
        transceive_fiber_input() in order to obtain rx_data, and returns both.

        Parameters
        ----------
        n_blocks : int
            See the documentation of prepare_fiber_input()
        seed : int
            See the documentation of prepare_fiber_input()

        Returns
        -------
        tx_data : dictionary
            See the description of prepare_fiber_input(...).
        rx_data : dictionary
            See the description of transceive_fiber_input(...).
        """
        tx_data = self.prepare_fiber_input(n_blocks, seed)
        rx_data = self.transceive_fiber_input(tx_data)
        return tx_data, rx_data

    def evaluate_results(self, tx_data, rx_data):
        """Evaluates the results of a simulation by showing several plots and
        printing some key measures to the console.

        Parameters
        ----------

        tx_data : dictionary
            Fiber input data generated by run() or prepare_fiber_inputs()
        rx_data : dictionary
            Fiber output data generated by run() or transceive_fiber_inputs()
        """
        hplt.plot_q(tx_data["t"], tx_data["q"], "Fiber input", new_fig=True)
        hplt.plot_q(rx_data["t"], rx_data["q"], "Fiber output", new_fig=True)
        tx_data["nfspecs"][0].show(new_fig=True, title="Nonlinear Fourier spectrum of the first block")
        rx_data["nfspecs"][0].show(legend=["tx", "rx"])

        constellation_diagram = ConstellationDiagram(self.constellation)
        constellation_diagram.plot(rx_data["symbols"], new_fig=True)

        error_vector_magnitude = ErrorVectorMagnitude()
        evm = error_vector_magnitude.in_dB(tx_data["symbols"], rx_data["symbols"])

        bit_error_ratio = BitErrorRatio(self.constellation)
        ber, nerr, nbits = bit_error_ratio.compute(tx_data["symbols"], rx_data["symbols"])

        modulation_efficiency = ModulationEfficiency()
        eff, br, bw_rx = modulation_efficiency.compute(rx_data["t"], rx_data["q"], rx_data["q"], nerr, nbits)
        _, _, bw_tx = modulation_efficiency.compute(tx_data["t"], tx_data["q"], tx_data["q"], 0, 1)

        tx_power_level = np.mean(np.abs(tx_data["q"])**2)
        tx_power_level_in_dBm = 10*np.log10(tx_power_level / 0.001)

        link_length = self.link.span_length * self.link.n_spans
        block_duration = (tx_data["t"][-1] - tx_data["t"][0])/tx_data["n_blocks"]
        bw_sim = 1 / (tx_data["t"][1] - tx_data["t"][0])

        print("Link length =", link_length*1e-3, "km")
        print("Nonlinear length =", self.link.nonlinear_length(tx_data["q"])*1e-3, "km")
        print("Span length =", self.link.span_length*1e-3, "km")
        print("Block duration =", block_duration*1e9, "ns")
        print("Tx Power Level =", tx_power_level_in_dBm, "dBm")
        print("Tx signal bandwidth = ", bw_tx*1e-9, "GHz")
        print("Rx signal bandwidth = ", bw_rx*1e-9, "GHz")
        print("Simulation bandwidth = ", bw_sim*1e-9, "GHz")
        print("Gross bit rate =", br*1e-9, "Gbit/s")
        print("Modulation efficiency =", eff, "bits/s/Hz")
        print("Bit error ratio (uncoded) =", ber)
        print("Error vector magnitude =", evm, "dB")
