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

import numpy as np
import matplotlib.pyplot as plt

class ConstellationDiagram:
    '''Plots a constellation diagram.'''

    def __init__(self, constellation):
        """Constructor.

        Parameters
        ----------

        constellation : object
            The constellation from which the symbols passed to plot(...) are
            drawn. Has to be of a class derived from BaseConstellation.
        """
        self._constellation = constellation

        # Set ranges for plotting
        max_real = 1.5*np.max(np.abs(np.real(self._constellation.alphabet)))
        max_imag = 1.5*np.max(np.abs(np.imag(self._constellation.alphabet)))
        self._axis = 1.5 * np.max([max_real, max_imag]) * np.array([-1., 1.])

    def plot(self, symbols, show_bits=False, title=None, new_fig=False):
        """Plots a constellation diagram with both the received symbols and the
        original modulation alphabet.

        Parameters
        ----------

        symbols : numpy.array(complex)
            Vector of symbols.
        show_bits : bool
            If True, the corresponding bit patterns are shown together with the
            constellation alphabet.
        title : str or None
            Title of the figure
        new_fig : bool
            If true, a new figure is created for the plot.

        Returns
        -------

        object
            The created Matplotlib figure (only if new_fig==True).
        """
        if new_fig:
            fig = plt.figure()
        plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.5)
        plt.scatter(np.real(self._constellation.alphabet),
                    np.imag(self._constellation.alphabet),
                    marker='x')
        if show_bits:
            r = np.size(self._constellation.alphabet)
            for n in range(0, r):
                bit_str = str(self._constellation.idx2bits(n))[1:-1:2]
                symbol = self._constellation.alphabet[n]
                plt.annotate(bit_str,
                             (np.real(symbol), np.imag(symbol)-0.075),
                             ha="center", va="top")
        plt.axis('square')
        plt.xlim(self._axis)
        plt.ylim(self._axis)
        plt.xlabel('Real part')
        plt.ylabel('Imaginary part')
        if title==None:
            plt.title('Constellation diagram')
        else:
            plt.title(title)
        if new_fig:
            fig.show()
            return fig
        else:
            return
