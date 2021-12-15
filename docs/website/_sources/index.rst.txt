PVRPM's documentation
=================================
About
--------
In photovoltaics (PV), the ability to perform an accurate techno-economic analysis is essential. Creating economically efficient PV systems is beneficial to both consumers and producers alike. This package: Python PhotoVoltaic Reliability Performance Model (PyPVRPM), fills this need. PyPVRPM is a simulation tool that uses NREL's SAM software to model the performance of a PV plant throughout its lifetime. It simulates failures, repairs, and monitoring across the lifespan based on user-defined distributions and values. This allows a more accurate representation of cost and availability throughout a lifetime than SAM's base simulation done from the GUI. By varying repair rates and monitoring solutions, one can compare different configurations to find the most optimal setup for implementing an actual PV power plant.

A few assumptions are taken in the tool, alongside specific ways calculations are made. Please see the logic diagram to understand how the simulation works. Also, view the example configuration to get an idea of setting up your case.

PVRPM Requires a valid SAM case created in its GUI before use. The SAM case provides the information for PVRPM to operate, alongside simulations using its LCOE calculator defined in the case, weather files, etc. Please see the getting started tutorials to learn how to set up your SAM case and PVRPM YAML configuration.

.. toctree::
  :maxdepth: 2
  :glob:
  :caption: Tutorials:

  tutorial*

.. toctree::
  :maxdepth: 2
  :glob:
  :caption: Examples:

  example*

.. toctree::
  :maxdepth: 2
  :glob:
  :caption: API:

  api/*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
