Getting Started
===============

NFDMLab is organized around `Example` classes. Each example bundles the details of a fiber-optic transmission simulation. We now discuss how examples are loaded and run, how the generated data can be analyzed, and how examples can be modified.

Loading Existing Examples
-------------------------

Users can create a instance of an existing example class in order to perform simulations. For example:

.. code-block:: python

  import Examples
  ex = Examples.BuelowArefIdler2016()

A list of existing examples is provided under :doc:`examples`.

Running an Example
------------------

Any example class is derived from :class:`Examples.BaseExample`. Simulations can be performed using the method :func:`Examples.BaseExample.run`, which every example class inherits from BaseExample. For example:

.. code-block:: python

  tx_data, rx_data = ex.run(5) # 5 is the number of bursts

The dictionaries `tx_data` and `rx_data` contain time- and nonlinear frequency domain descriptions of the fiber inputs and fiber outputs together with the transmitted and received symbols. See the documentation of the run method for more information.

Evaluating Simulation Results
-----------------------------

The method :func:`Examples.BaseExample.evaluate_results` can be used for a quick analysis of the simulation results stored in `tx_data` and `rx_data`. For example:

.. code-block:: python

  ex.evaluate_results(tx_data, rx_data)

This will generate some plots and print some information about the transmission.

The classes in :doc:`qualityassessment` can be used for custom analyses. For example:

.. code-block:: python

  from QualityAssessment import BitErrorRatio
  ber = BitErrorRate(ex.constellation)
  ber_value, n_err, n_bits = ber.compute(tx_data["symbols"], rx_data["symbols"])
  print("The bit error ratio is", ber_value)

Changing Examples
-----------------

Examples can be reconfigured by changing their public attributes and calling the reconfigure method. For example:

.. code-block:: python

  ex.constellation = "PSK"
  ex.constellation_level = 8
  ex.reconfigure()

We could now rerun the simulation as described above. The :doc:`examples` page provides information about the public attributes provided by each example class.

Writing Own Examples
--------------------

You can write own examples by defining a new example class that is derived from :class:`Examples.BaseExample`. Please read the documentation of BaseExample and its methods to learn what your example class should implement.
