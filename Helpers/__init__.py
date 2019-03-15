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

from Helpers.NFSpectrum import NFSpectrum

def checked_get(obj, attr_name, base_class):
    """Auxiliary function that checks whether the object obj has set an
    attribute of the name attr_name to a value of a type derived from
    base_class.

    Parameters
    ----------
    obj : object
    attr_name : string
    base_class : class

    Returns
    _______
    getattr(obj, attr_name)

    Raises
    ------
    NamedErrors if the attribute has not been set or is of a type that is
    not derived from base_class
    """
    if not hasattr(obj, attr_name):
        raise NameError(obj.__class__.__name__+" object has not set attribute "+attr_name)
    if not isinstance(getattr(obj, attr_name), base_class):
        raise NameError(obj.__class__.__name__+" object's attribute "+attr_name+" type is not derived from "+base_class.__name__)
    return getattr(obj, attr_name)

def next_pow2(int_num):
    """Next positive power of two greater than or equal to a given number.

    Parameters
    ----------
    int_num : int or float
        A finite number

    Returns
    -------
    int
        The smallest integer of the form 2**n, n=0,1,2,..., such that
        int_num <= 2**n
    """
    pow2 = 1
    while pow2 < int_num:
        pow2 <<= 1
    return pow2
