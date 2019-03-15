Installation
============

Automatic Installation under Ubuntu 18.04
---------------------------------------------------

Users of `Ubuntu Linux 18.04 <https://www.ubuntu.com/download/desktop>`_ may use an automatic installation script.

Open a terminal and execute the commands

.. code-block:: bash

  wget -nc https://github.com/FastNFT/NFDMLab/raw/master/ubuntu_1804_install.sh
  bash ./ubuntu_1804_install.sh
  source ~/.bashrc

This script will install NFDMLab in the folder ``~/git/nfdmlab``. Proceed to the section :ref:`testing` below after installation.

Manual Installation
-------------------

NFDMLab is developed under Linux and written in `Python <https://www.python.org>`_. Please make sure that a recent version of Python 3 is available on your computer. It should be possible to install NFDMLab under other operating systems, but we have no experience with such setups. Before NFDMLab can be run, the following dependencies have to be installed.

1. `Matplotlib <https://matplotlib.org>`_, `NumPy <http://www.numpy.org>`_ and `SciPy <https://www.scipy.org>`_
2. `Jupyter Notebooks <https://jupyter.org>`_ (optional, only for interactive examples)
3. `FNFT <https://github.com/FastNFT/FNFT>`_
4. `FNFTpy <https://github.com/xmhk/FNFTpy>`_

Please see the documentations of these packages for details on how to install them. After installation of the dependencies, download a recent `release of NFDMLab <https://github.com/FastNFT/NFDMLab/releases>`_, extract the files into some directory and proceed to the section :ref:`testing`.

.. _testing:

Testing the Installation
------------------------

Once all required components are installed, change into the directory that contains NFDMLab and start Python. For example, if you used the installation script:

.. code-block:: bash

  cd ~/git/nfdmlab
  python3

Then, in Python, try the following:

.. code-block:: python

  import Examples
  ex = Examples.BuelowArefIdler2016()
  [tx_data, rx_data] = ex.run(5)
  ex.evaluate_results(tx_data, rx_data)

or load one of the files in Notebook directory in Jupyter for the interactive version:

.. code-block:: bash

  jupyter notebook Notebooks/BuelowArefIdler2016.ipynb
