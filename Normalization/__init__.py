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
# Christoph Mahnke 2018
# Sander Wahls 2019

import numpy as np

from abc import ABC, abstractmethod

class BaseNormalization(ABC):
    """Base class for normalization. Normalization classes let us switch
    to normalized forms of the transmission equations and back. All
    normalization classes should be derived from this class.
    """

    @abstractmethod
    def norm_field(self, A):
        """Apply normalization to a field A."""
        pass

    @abstractmethod
    def denorm_field(self, u):
        """De-normalize a normalized field u."""
        pass

    @abstractmethod
    def norm_time(self, t):
        """Apply normalization to a time variable t."""
        pass

    @abstractmethod
    def denorm_time(self, tv):
        """De-normalize a normalized time variable tau."""
        pass

    @abstractmethod
    def norm_dist(self, z):
        """Apply normalization to a space variable z."""
        pass

    @abstractmethod
    def denorm_dist(self, xi):
        """De-normalize a normalized space variable xi."""
        pass

    @abstractmethod
    def norm_alpha(self, alpha):
        """Apply normalization to an attentuation parameter alpha."""
        pass

    @abstractmethod
    def denorm_alpha(self, Gamma):
        """De-normalize a normalized attenuation parameter Gamma."""
        pass

class Lossless(BaseNormalization):
    """The Lossless object allows to normalize and de-normalize field, time and spatial variables.

    Here, it is assumed that the fiber is lossless.

    Initialization:

        normobj = Normalization(b2, gamma, Tscal)

        Arguments:

            - b2    : GVD parameter in units of s**2/m
            - gamma : nonlinearity parameter in units of (W m)**(-1)
            - Tscal : scaling time in units of s

    Examples:

        - initialize:

            b2 = -20e-27
            gamma = 0.001
            Ts = 1e-12
            normobj = Normalization(b2, gamma, Ts)

        - normalize a field amplitude A:

            u = normobj.norm_field(A)

        - de-normalize a normalized field amplitude u:

            A = normobj.denorm_field(u)

        - normalize a time t:

            tau = normobj.norm_time(t)

        - normalize a distance L:

            xi = normobj.norm_dist(L)

    """
    def __init__(self, b2, gamma, Tscal):
        self.b2 = b2
        self.gamma = gamma
        self.Tscal = Tscal
        self.time_factor = 1.0 / Tscal
        self.ampl_factor = Tscal / np.sqrt(np.abs(b2/gamma))
        self.dist_factor = np.abs(b2) / Tscal**2

    def __repr__(self):
        rs="fiber parameters:\n    beta_2 = %.2e s**2/m\n    gamma  = %.2e (Wm)**(-1)\nscaling time:\n    Tscal= %.2e s"%(
            self.b2, self.gamma, self.Tscal)
        rs+="\nscaling factors:\n    time_factor  = %.2e\n    ampli_factor = %.2e\n    dist_factor  = %.2e"%(
            self.time_factor, self.ampl_factor, self.dist_factor)
        return rs

    def norm_field(self, A):
        """Apply normalization to a field A."""
        return self.ampl_factor * A

    def denorm_field(self, u):
        """De-normalize a normalized field u."""
        return u / self.ampl_factor

    def norm_time(self, t):
        """Apply normalization to a time variable t."""
        return t * self.time_factor

    def denorm_time(self, tv):
        """De-normalize a normalized time variable tau."""
        return tv / self.time_factor

    def norm_dist(self, z):
        """Apply normalization to a space variable z."""
        return z * self.dist_factor

    def denorm_dist(self, xi):
        """De-normalize a normalized space variable xi."""
        return xi / self.dist_factor

    def norm_alpha(self, alpha):
        """Apply normalization to an attentuation parameter alpha."""
        return alpha / self.dist_factor

    def denorm_alpha(self, Gamma):
        """De-normalize a normalized attenuation parameter Gamma."""
        return Gamma * self.dist_factor


class Lumped(Lossless):
    """The Lumped object allows to normalize and de-normalize for the case of lumped amplfication.

    Here, it is assumed that fiber losses are compensated by lumped amplification.
    When the distance between the amplifiers is small compared to e.g. the dispersion length,
    the lossless case can be simulated using a reduced nonlinearity parameter gamma1.

    See: S.T. Le et al.: 'Nonlinear inverse synthesis technique for optical
         links with lumped amplification', Opt. Expr. 23, 8317 (2015)

    Initialization:

        normobj = Normalization(b2, gamma, Tscal, alpha, zamp)

        Arguments:

            - b2    : GVD parameter in units of s**2/m
            - gamma : nonlinearity parameter in units of (W m)**(-1)
            - Tscal : scaling time in units of s
            - alpha : loss coefficient in units of 1/m
            - zamp  : distance between lumped amplifiers in units of m

    Examples:

        - initialize:

            b2 = -20e-27
            gamma = 0.001
            Ts = 1e-12
            alpha = 4.6e-5   # = 0.2dB/km
            zamp = 10e3
            normobj = NormalizationLumped(b2, gamma, Ts, alpha, zamp)

        - normalize a field amplitude A:

            u = normobj.norm_field(A)

        - de-normalize a normalized field amplitude u:

            A = normobj.denorm_field(u)

        - normalize a time t:

            tau = normobj.norm_time(t)

        - normalize a distance L:

            xi = normobj.norm_dist(L)

    """
    def __init__(self, b2, gamma, Tscal, alpha, zamp):
        self.b2 = b2
        self.gamma0 = gamma
        G = np.exp(alpha * zamp)
        self.gamma1 = gamma * (G-1.0)/(G * np.log(G))
        self.zamp = zamp
        self.alpha = alpha
        self.Tscal = Tscal
        self.time_factor = 1.0 / Tscal
        self.ampl_factor = Tscal / np.sqrt(np.abs(b2/self.gamma1))
        self.dist_factor = np.abs(b2) / Tscal**2

    def __repr__(self):
        rs="fiber parameters:\n    beta_2 = %.2e s**2/m\n    gamma  = %.2e (Wm)**(-1)\nscaling time:\n    Tscal= %.2e s"%(
            self.b2, self.gamma0, self.Tscal)
        rs+="\nloss and lumped ampl.\n   alpha=%.2e (1/m) zamp=%.2e m "%(self.alpha, self.zamp)
        rs+="\neffective nonlinearity parameter\n    gamma1 = %.2e (Wm)**(-1)    (=%.3f gamma)"%(self.gamma1, self.gamma1/self.gamma0)
        rs+="\nscaling factors:\n    time_factor  = %.2e\n    ampli_factor = %.2e\n    dist_factor  = %.2e"%(
            self.time_factor, self.ampl_factor, self.dist_factor)
        return rs
