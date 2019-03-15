#!/bin/bash

echo "This script is part of NFDMLab.

NFDMLab is free software; you can redistribute it and/or
modify it under the terms of the version 2 of the GNU General
Public License as published by the Free Software Foundation.

NFDMLab is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public
License along with NFDMLab; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
02111-1307 USA

===============================================================

This Bash script tries to install NFDMLab and its dependencies
on a fresh installation of Ubuntu 18.04 Desktop. It might also
work for other versions of Ubuntu, but we have not tested this.

The script assumes that the current user may sudo and asks for
the user password. Please note that running this script might
have unintended consequences. Use at own risk.

===============================================================
"

read -p"Press Enter to continue or Ctrl+C to abort"
echo ""

# Run some basic tests to make sure we are on the right platform

if ! hash apt-get 2>/dev/null; then
    echo "Error: apt-get is not available."
    exit 1
fi
if ! hash python3 2>/dev/null; then
    echo "Error: python3 is not available."
    exit 2
fi
if ! python3 -mplatform | grep -q 'Ubuntu'; then
    echo "Error: This does not seem to be a Ubuntu system!"
    exit 3
fi

# Some of our dependencies are in universe

sudo add-apt-repository universe

# Install FNFT

sudo apt-get -y install git gcc gfortran cmake fftw3-dev
mkdir -p ~/git
cd ~/git
git clone https://github.com/FastNFT/FNFT
cd FNFT
mkdir build
cd build
cmake .. -DENABLE_FFTW=ON -DBUILD_TESTS=OFF
make -j4
sudo make install
sudo ldconfig

# Install FNFTpy

cd ~/git
git clone https://github.com/xmhk/FNFTpy
if ! echo $PYTHONPATH | grep -q ":$HOME/git/FNFTpy:"; then
    echo "export PYTHONPATH=\"\${PYTHONPATH}\":$HOME/git/FNFTpy:" >> ~/.bashrc
fi

# Install Jupyter notebook

sudo apt-get -y install python3-pip python3-dev
sudo -H pip3 install --upgrade pip
sudo -H pip3 install jupyter

# Install NFDMLab

sudo apt-get -y install python3-numpy python3-scipy python3-matplotlib
cd ~/git
git clone https://github.com/FastNFT/NFDMLab.git nfdmlab
echo "
NFDMLab has been installed in the directory $HOME/git/nfdmlab."
