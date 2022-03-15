Installation
=================================
.. toctree::
  :hidden:

This document covers installation and setup of the tool. The tool requires you to build a valid case in SAM, so if you haven't already download and install SAM from here: https://sam.nrel.gov/download.html

**Currently, the supported SAM version is 2021.12.02!**

SAM can be installed on Windows, MAC, or Linux.


Installation
--------------
Requires python >= 3.8

Works on Windows and Linux x64 OSes.

**Currently, there are issues running on macOS. Please see this issue for updates:** https://github.com/FSEC-Photovoltaics/pvrpm-lcoe/issues/49


**Recommended** using pip:
(Replace `@master` with the branch release name if you want a release version)

.. code-block:: bash
  :linenos:

  # for latest development branch
  pip install git+https://github.com/FSEC-Photovoltaics/pvrpm-lcoe/@master

  # for specific version
  pip install git+https://github.com/FSEC-Photovoltaics/pvrpm-lcoe/@vx.x.x

Using the wheel file downloaded from https://github.com/FSEC-Photovoltaics/pvrpm-lcoe/releases

.. code-block:: bash
  :linenos:

  pip install wheel
  pip install pvrpm-x.x.x-py3-none-any.whl

Manually:

.. code-block:: bash
  :linenos:

  git clone https://github.com/FSEC-Photovoltaics/pvrpm-lcoe
  cd pvrpm-lcoe
  python setup.py install

If you want to build the documentation:

.. code-block:: bash
  :linenos:

  git clone https://github.com/FSEC-Photovoltaics/pvrpm-lcoe
  cd pvrpm-lcoe
  pip install .[docs]
  cd docs
  make html

If you want to run automated tests (will take a while based on compute power):

.. code-block:: bash
  :linenos:

  git clone https://github.com/FSEC-Photovoltaics/pvrpm-lcoe
  cd pvrpm-lcoe
  pip install .[testing]
  pytest

For setting up the package in edit mode to modify PVRPM for fixing bugs or adding features:

.. code-block:: bash
  :linenos:

  git clone https://github.com/FSEC-Photovoltaics/pvrpm-lcoe
  cd pvrpm-lcoe
  pip install -e .

Edits to the code can be made in the `pvrpm-lcoe` folder and be tested by running PVRPM from the command line or custom wrapper script.
