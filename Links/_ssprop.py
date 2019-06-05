# Copyright 2019, Marius Brehler
# Copyright 2018, Sander Wahls (TU Delft)
# Copyright 2018, Marius Brehler (TU Dortmund)
# Copyright 2006, Thomas E. Murphy
#
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

import numpy as np
from warnings import warn

def _ssprop(u0,dt,dz,nz,alpha,betap,gamma,maxiter=4,tol=1e-5):
    """This function solves the nonlinear Schrodinger equation for
    pulse propagation in an optical fiber using the split-step
    Fourier method.

    The following effects are included in the model: group velocity
    dispersion (GVD), higher order dispersion, loss, and self-phase
    modulation (gamma).

    USAGE

    u1 = ssprop(u0,dt,dz,nz,alpha,betap,gamma)
    u1 = ssprop(u0,dt,dz,nz,alpha,betap,gamma,maxiter)
    u1 = ssprop(u0,dt,dz,nz,alpha,betap,gamma,maxiter,tol)

    INPUT

    u0 - starting field amplitude (vector)
    dt - time step
    dz - propagation stepsize
    nz - number of steps to take, ie, ztotal = dz*nz
    alpha - power loss coefficient, ie, P=P0*exp(-alpha*z)
    betap - dispersion polynomial coefs, [beta_0, ..., beta_m]
    gamma - nonlinearity coefficient
    maxiter - max number of iterations (default = 4)
    tol - convergence tolerance (default = 1e-5)

    OUTPUT

    u1 - field at the output

    NOTES  The dimensions of the input and output quantities can
    be anything, as long as they are self consistent.  E.g., if
    |u|^2 has dimensions of Watts and dz has dimensions of
    meters, then gamma should be specified in W^-1*m^-1.
    Similarly, if dt is given in picoseconds, and dz is given in
    meters, then beta(n) should have dimensions of ps^(n-1)/m.

    AUTHORS

    This function is a Python port of the script ssprop.m by
    Thomas E. Murphy (tem@umd.edu), which comes with the SSPROP
    package (https://www.photonics.umd.edu/software/ssprop/).
    The port has been written by Sander Wahls and has been
    modified by Marius Brehler.
    """

    nt = np.size(u0)
    assert nt%2 == 0
    w = 2*np.pi*np.hstack((np.arange(0, nt/2), np.arange(-nt/2, 0)))/(dt*nt)

    halfstep = 0
    if (np.size(alpha) != 1) & (np.size(alpha) != nz):
        raise Exception('alpha is neither constant nor is the vector of length nz')

    for ii in range(0, np.size(betap)):
        halfstep = halfstep - 1j*betap[ii]*(w)**ii/np.math.factorial(ii)
    #halfstep = np.exp(halfstep*dz/2)

    u1 = u0
    ufft = np.fft.fft(u0)
    for iz in range(0, nz):
        if nz>1 and np.size(alpha) == nz:
            alpha_step = -alpha[iz]/2
        else:
            alpha_step = -alpha/2

        uhalf = np.fft.ifft(np.exp((alpha_step+halfstep)*dz/2)*ufft)
        for ii in range(0, maxiter):
            uv = uhalf * np.exp(-1j*gamma*(np.abs(u1)**2 + np.abs(u0)**2)*dz/2)
            uv = np.fft.fft(uv)
            ufft = np.exp((alpha_step+halfstep)*dz/2)*uv
            uv = np.fft.ifft(ufft)
            if (np.linalg.norm(uv-u1)/np.linalg.norm(u1) < tol):
                u1 = uv
                break
            else:
                u1 = uv
            if (ii == maxiter):
                warn("Failed to converge to %e in %d iterations" % (tol, maxiter))
        u0 = u1

    return u1
