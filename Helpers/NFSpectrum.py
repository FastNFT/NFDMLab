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
import matplotlib.pyplot as plt

class NFSpectrum:
    """Stores and plots nonlinear Fourier spectra.

    The following attributes can be set/read by the user:

    xi : np.array(float)
        Nonlinear frequency grid (a vector).
    cont : np.array(complex)
        Vector specficying the continuous spectrum at the nonlinear frequencies
        in xi. Has to be of the type specified during construction.
    bound_states : np.array(complex)
        Vector containing the bound states (eigenvalues).
    normconsts : np.array(complex)
        Vector specifying the residues (b/a') or norming constants (b) for each
        of the bound states.
    xi_plot_range : np.array(float)
        Vector of length two with xi_plot_range[0]<xi_plot_range[1]. Speficies
        the xi range shown when the continuous spectrum is plotted using show().
        The default is an empty vector [], in which case no range are set.
    bound_state_plot_range : np.array(float)
        Vector of length four of the form [re_min, re_max, im_min, im_max].
        Speficies the range of real/imaginary parts for the bound states when
        the bound states are plotted using show(). The default is an empty
        vector [], in which case no ranges are set.
    """

    @property
    def cont_type(self):
        '''Type of the continuous spectrum stored in this object (read-only, set
        during construction).

        Returns
        -------
        "none" : str
            if no continuous spectrum is stored.
        "b/a" : str
            if the continuous spectrum is a reflection coefficient.
        "b" : str
            if the continuous spectrum is a b-coefficient.'''
        return self._cont_type

    @property
    def disc_type(self):
        '''Type of the discrete spectrum stored in this object (read-only, set
        during construction).

        Returns
        -------
        "none" : str
            if no discrete spectrum is stored.
        "b/a'" : str
            if the discrete spectrum contains eigenvalues and residues.
        "b" : str
            if the discrete spectrum contains eigenvalues and norming constants.'''

        return self._disc_type

    def __init__(self, cont_type, disc_type):
        """Constructor. Initializes all attributes mentioned above except
        cont_type and disc_type to empty values.

        Parameters
        ----------
        cont_type : str
            See the cont_type property.
        disc_type : str
            See the disc_type property.
        """

        if cont_type in ["none", "b/a", "b"]:
            self._cont_type = cont_type
        else:
            raise ValueError()
        if disc_type in ["none", "b/a'", "b"]:
            self._disc_type = disc_type
        else:
            raise ValueError()

        self.xi = np.array([])
        self.cont = np.array([])
        self.bound_states = np.array([])
        self.normconsts = np.array([])

        self.xi_plot_range = np.array([])
        self.bound_state_plot_range = np.array([])

    def _set_xi_range_and_legend(self, ax, legend):
        has_xi_range = np.size(self.xi_plot_range)>0
        if has_xi_range:
            ax.set_xlim(self.xi_plot_range)
        if legend is not None:
            ax.legend(legend)

    def show_contspec_mag(self, ax, legend=None):
        """Plots the magnitude of the continous spectrum.

        Parameters
        ----------
        ax : matplotlib.axes object
            Axes used for plotting.
        legend : array(str)
            Array of legend entries. If None, no legend is added.
        """

        ax.plot(self.xi, np.abs(self.cont))
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$|'+self.cont_type+r'(\xi)|$')
        self._set_xi_range_and_legend(ax, legend)

    def show_contspec_angle(self, ax, legend=None):
        """Plots the angle (phase) of the continous spectrum.

        Parameters
        ----------
        ax : matplotlib.axes object
            Axes used for plotting.
        legend : array(str)
            Array of legend entries. If None, no legend is added.
        """

        ax.plot(self.xi, np.angle(self.cont))
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\angle '+self.cont_type+r'(\xi)$')
        self._set_xi_range_and_legend(ax, legend)

    def show_discspec(self, ax, legend=None):
        """Plots the bound states of the discrete spectrum.

        Parameters
        ----------
        ax : matplotlib.axes object
            Axes used for plotting.
        legend : array(str)
            Array of legend entries. If None, no legend is added.
        """

        ax.plot(np.real(self.bound_states), np.imag(self.bound_states), 'x')
        ax.set_xlabel(r'Real part of eigenvalues')
        ax.set_ylabel(r'Imaginary part of eigenvalues')
        if np.size(self.bound_state_plot_range)>=4:
            plt.xlim(self.bound_state_plot_range[0:2])
            plt.ylim(self.bound_state_plot_range[2:4])
        if legend is not None:
            ax.legend(legend)

    def show(self, new_fig=False, title=None, legend=None):
        """Plots the nonlinear Fourier spectrum stored in this object. This
        routine can be called repeatedly, in which several plots are shown
        in the same figure.

        Parameters
        ----------
        new_fig : bool
            If True, a new figute is created by the routine.
        title : str
            Title of the plot.
        legend : array(str)
            Array of legend entries. If None, no legend is added.

        Returns
        -------
        matplotlib.figure object
            Figure object, only if new_fig==True.
        """

        nsubplots = 3
        if self.cont_type is "none":
            nsubplots -= 2
        if self.disc_type is "none":
            nsubplots -= 1
        if nsubplots == 0:
            return

        if new_fig:
            fig = plt.figure()
            # make figure wider to fit subplot
            sz = fig.get_size_inches()
            sz[0] *= nsubplots
            fig.set_size_inches(sz)
            for i in range(1, nsubplots+1):
                plt.subplot(1, nsubplots, i)
        else:
            fig = plt.gcf()
        axes = fig.get_axes()

        if self.cont_type is not "none":
            self.show_contspec_mag(axes[0], legend)
            self.show_contspec_angle(axes[1])
        if self.disc_type is not "none":
            self.show_discspec(axes[-1], legend)
        if title is not None:
            plt.suptitle(title)
        if new_fig:
            fig.show()
            return fig
