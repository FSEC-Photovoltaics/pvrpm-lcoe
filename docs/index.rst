PVRPM's documentation
=================================
About
--------
The PhotoVoltaic Reliability Performance Model (PVRPM) is a simulation tool that using NREL's SAM tool to model performance of a PV plant over the course of it's lifetime. It simulates failures, repairs, and monitoring across the lifespan based on user defined distributions and values. This allows a more accurate representation of cost and availability over the course of a lifetime compared to SAM's base simulation you can do from the GUI. There are many setups and configurations possible, allowing one to use this tool to compare different repair and monitoring practices to find what gives the lowest LCOE.

There are a few assumptions taken in the tool, alongside a specific way calculations are made. Please see the logic diagram to understand how the simulation works. Also, view the example configuration to get an idea how to setup your case.

PVRPM Requires a valid SAM case built in the it's GUI before using. The SAM case provides the information for PVRPM to operate, alongside doing simulations using it's LCOE calculator defined in the case, weather files, etc. Please see the getting started tutorials to learn how to setup your SAM case and PVRPM YAML configuration.

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
