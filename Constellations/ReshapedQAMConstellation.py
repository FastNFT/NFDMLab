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
import scipy.integrate as integrate
import warnings

from Constellations import QAMConstellation

class ReshapedQAMConstellation(QAMConstellation):
    """Reshaped quadrature amplitude modulation (QAM) constellation (implements
    BaseConstellation).

    The goal of the reshaping procedure is to fit the average (normalized)
    energy of the generated pulses to a desired value Ed. For details, see
    Gui et al., Opt. Express 26(21), 2018.
    """

    def __init__(self, m, n, b0_fun, Ed, bnds):
        """Constructor for a reshaped m x n QAM constellation.

        Parameters
        ----------
        m : int
        n : int
        b0_fun : function
            Carrier waveform. The function should maps any input vector of the
            type numpy.array(float), which represents a vector of nonlinear
            frequencies xi, to a output vector of the type numpy.array(complex),
            which represents the values of the carrier waveform at these xi.
        Ed : float
            Desired average energy of the generated pulses (with respect to
            normalized units). Should be positive.
        bnds : numpy.array(float)
            Vector with two entries [a,b], which are used as initial bounds in
            the bisection produdure based on which the constellation is
            reshaped. It should be 0<a<b.
        """
        super().__init__(m, n)
        alphabet0 = self.alphabet / np.max(np.abs(self.alphabet))
        niter = 35
        abs_vals = np.unique(np.abs(alphabet0))
        M = np.size(alphabet0)
        K = np.size(abs_vals)
        for k in range(0, K):
            Ek_target = Ed*M*abs_vals[k]**2/(np.linalg.norm(alphabet0)**2)
            lb = bnds[0]
            ub = bnds[1]
            for iter in range(0, niter):
                Ak = 0.5*(lb + ub)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    I = integrate.quad(lambda xi: -np.log(1 - Ak*(abs_vals[k]**2)*(np.abs(b0_fun(xi))**2)), -np.inf, np.inf)
                Ek = I[0]/np.pi
                if Ek<Ek_target:
                    lb = Ak
                else:
                    ub = Ak
            if Ek<0.95*Ek_target:
                raise ValueError('scale_modulation failed - try to increase upper bound')
            Ak = lb
            idx = np.abs(alphabet0) == abs_vals[k]
            self._alphabet[idx] = np.sqrt(Ak)*alphabet0[idx]
        self._name = "Reshaped %d-QAM Constellation" % self.size()
