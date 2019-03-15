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
# Marius Brehler (TU Dortmund) 2018

import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 16})

def plot_q(t, q, title, new_fig=False):
    if new_fig:
        fig = plt.figure()
    plt.plot(t*1e9, abs(q))
    plt.xlabel('t in ns')
    plt.ylabel(r'|q(t)| in $\sqrt{\mathrm{W}}$')
    plt.title(title)
    if new_fig:
        fig.show()
        return fig
    else:
        return

def plot_qout(t, q, title):
    plt.plot(t*1e9, abs(q))
    plt.xlabel('t in ns')
    #plt.ylabel(r'|q(t)| in $\sqrt{\mathrm{W}}$')
    plt.title(title)
    return

def plot_r(xi, r, title, ncarriers, carrier_spacing):
    idx = abs(xi) < ncarriers*carrier_spacing
    plt.plot(xi[idx], abs(r[idx]))
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'|r($\xi$)|')
    plt.title(title)
    return

def plot_contspec(tx_data, rx_data, ylabel, block=1, new_fig=False):
    if new_fig:
        fig = plt.figure()
    plt.plot(tx_data["nfspecs"][block].xi, abs(tx_data["nfspecs"][block].cont))
    plt.plot(rx_data["nfspecs"][block].xi, abs(rx_data["nfspecs"][block].cont))
    plt.xlabel(r'$\xi$')
    plt.ylabel(ylabel)
    plt.title("Continuous spectrum of block {}".format(block))
    plt.legend(["tx", "rx"])
    if new_fig:
        fig.show()
        return fig
    else:
        return
