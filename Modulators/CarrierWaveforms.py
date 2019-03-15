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

def flat_top(xivec, T0):
    """Flat-top carrier that is bandlimited with bandwidth 2*T0. The effective
    bandwidth, based on visual inspection in linear scale, is around T0.

    The top of this filter has a width of approximately 20/T0. The visible
    bottom (in linear scale) has a width of approximately 45/T0."""
    tophat_coeffs = np.array([1.007812499990869, 2.015624999967228,
                              2.015624998481229, 2.015624285101227,
                              2.015576901606147, 2.014596710132849,
                              2.005424182936140, 1.958132920846161,
                              1.807566405118839, 1.490558213477831,
                              1.031171573261926, 0.563957100582878,
                              0.229897459751809, 0.064961507923051,
                              0.011287414498426, 0.000905697614070])

    tmp = np.arange(0,np.size(tophat_coeffs));
    ns = np.size(xivec)
    if ns == 1:
        xi = xivec
        vals = np.inner(tophat_coeffs,( np.sinc(xi*T0/np.pi-tmp) + np.sinc(xi*T0/np.pi+tmp)))
    else:
        vals = np.zeros(ns, dtype=complex)
        for n in range(0,ns):
            xi = xivec[n]
            vals[n] = np.inner(tophat_coeffs,( np.sinc(xi*T0/np.pi-tmp) + np.sinc(xi*T0/np.pi+tmp)))
    return vals

def raised_cosine(xivec, roll_off_factor, T0):
    vals = np.sinc(xivec/T0)
    if np.isscalar(vals):
        xivec = np.array([xivec])
        vals = np.array([vals])
    if roll_off_factor == 0.0:
        return vals
    idx = np.abs(np.abs(xivec) - T0/2.0/roll_off_factor) < 10.0*np.sqrt(np.finfo(xivec[0]).eps)
    vals[idx] *= np.pi/4.0
    vals[~idx] *= np.cos(roll_off_factor*np.pi*xivec[~idx]/T0) / (1.0 - (2.0*roll_off_factor*xivec[~idx]/T0)**2)
    return vals
