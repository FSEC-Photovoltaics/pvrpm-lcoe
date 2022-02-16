Getting Started
=================================
.. toctree::
  :hidden:

**Make sure to follow** :doc:`Installation <tutorial_1installation>` **and install SAM before continuing here!**

PVRPM enhances SAM's models for PV systems to obtain a more accurate LCOE than its base simulation. It also allows studying the effects of monitoring and repair techniques on PV systems throughout their lifetime.

To get started, create a new SAM case in the SAM GUI. From there, you **must choose the Detailed Photovoltaic model** for PVRPM to work. This is required because PVRPM need's specific parameters that only exist in this model.

Then, choose your financial model. It must be a financial model that supports lifetime losses. Any financial model under the `Detailed Photovoltaic Model` will work **except** the `LCOE Calculator (FCR Method)` and `No Financial Model`. Please read SAM's documentation for help in setting up the case as it goes into more detail on these models.

Once that is set up, you can download the example configuration and modify it as needed. Below explains from start to finish of how to run a simulation with PVRPM.

Exporting the SAM Case
--------------------------------
PVRPM works by reading the information in your SAM case and the PVRPM YAML configuration to properly run the simulation. For the first step, you need to export your SAM case to JSON files, which PVRPM uses to read in your SAM case.

To do this:
1. Open the SAM GUI, then open your ``.sam`` case file.
2. Once it is open, click the drop-down menu for the case, which is located next to the case name on the top bar.
3. Click ``Generate Code``, then ``PySAM JSON`` (**not** ``JSON for Inputs``).
4. A file explorer window will open, allowing you to select a place to save the JSON files. Select an empty folder.

Once this is done, the selected folder will contain a few JSON files. The names of these files will follow the convention of ``case-name_module.json`` where ``case-name`` is the name of the case, and ``module`` is the module that JSON represents. Pay attention to what modules you have; you'll need to know that for the next part. You can remove the ``.h`` and ``.so`` files.

Configuring PVRPM
------------------------

This will go over every configuration option to set up the case study step by step. The example configuration is also heavily commented on to help with the parameters. Also, please study the logic diagram as it can help when setting up the configuration here. *Also note all of the values listed in examples are entirely arbitrary and do not represent a realistic case.*

You can download the example configuration file :download:`here <../pvrpm/config/example_config.yml>` or view the example configuration file :doc:`here <example_pvrpm_config>`.

Module Order
----------------
These modules correspond to the modules listed in the JSON files you obtained. The modules must be simulated in the correct order for PVRPM to properly run simulations using PySAM (SAM's Python interface). A typical order is in the example, but you may need to modify it depending on your case.

To make the module order easy to set up, go to this website: https://nrel-pysam.readthedocs.io/en/master/Configs.html#sam-simulation-configurations

Then, find the modules you have under ``SSC Compute Modules`` column in the table; the SAM configuration column should be one of the ``Detailed PV Model`` rows. Once you find one containing all your modules, put them in the order as they appear in the ``SSC Compute Modules`` column.

.. code-block:: yaml
  :linenos:

  module_order: # the order the modules of this case should be executed
                # check to see what modules you use by the name of the json output files
                # typically, this should be Pvsamv1 -> grid -> utiltiy -> others
                # see https://nrel-pysam.readthedocs.io/en/master/Configs.html#sam-simulation-configurations
  # Only detailed PV models are supported, which is Pvsamv1, your case must have this module
  # Also, only LCOE calculators that support lifetime are allowed, PVRPM will check this and notify you if your LCOE calculator doesn't support lifetime
    - Pvsamv1
    - Grid
    - Utilityrate5
    - Cashloan


Run Setup
----------------
Here, set the results folder location. On Windows, use only one backslash to separate directories; you do not need to escape them or spaces. Then set the number of realizations you want to run and the confidence interval for calculating results.

The results folder and number of realizations can be overridden from the command line.

.. code-block:: yaml
  :linenos:

  ### Run Setup ###
  # results folder doesn't need to exist, it will be created if it doesnt already
  # For windows use single backslash. Do not need to escape spaces in folder names
  results_folder: /path/to/results/folder
  num_realizations: 2 # number of realizations to run
  conf_interval: 90 #XX % confidence interval around the mean will be calculated for summary results

Case Setup
----------------
Set up the basics of the case. Set the number of trackers, combiners, and transformers. Set ``num_trackers`` to ``0`` if you are not using trackers. The worst-case tracker can be set to true if you are using trackers. This means that failures in tracking components result in them being stuck in the worst way: the module is pointing the opposite of the sun's travel arc.

.. code-block:: yaml
  :linenos:

  ### Case Setup ###
  num_combiners: 2 # total number of DC combiner boxes
  num_transformers: 1 # total number of Transformers
  num_trackers: 2 # total number of trackers

  ### Financial Inputs ###
  present_day_labor_rate: 100 # dollars per hour
  inflation: 2.5 # in percent

  ### Failure Tracker Algorithm ###
  use_worst_case_tracker: false

Component Level Setup
------------------------
Each component level requires a setup of failures, monitoring, and repairs. This is required for module, string, combiner, inverter, disconnect, transformer, and grid. However, if you are not setting a component level to fail (``can_fail: false``), you can remove the rest of the sections below it. For example, if string's ``can_fail: false``, I can remove the ``failure``, ``monitoring``, and ``repair`` sections.

There are many options for types of failures, monitoring, and the component, alongside various combinations to get different behaviors.

Keep in mind the way these operations are as follows:

1. First, using the distributions defined for failures, monitoring, and repairs, a ``time_to_failure``, ``time_to_detection``, and ``time_to_repair`` is generated for each component.

2. ``time_to_failure`` then counts down to 0. Once a failure occurs, ``time_to_detection`` counts down to 0 (if monitoring is defined). Finally, ``time_to_repair`` counts to 0, which repairs the component and resets these values.

Component Behaviors
########################
``can_fail`` can be set to ``true`` or ``false``, which dictates whether the components in the component level (``module``, ``string``, etc.) can fail. If this is ``false``, then nothing will fall for this level. You can remove the ``failures``, ``repairs``, and ``monitoring`` sections.
``can_repair`` dictates if the components can be repaired at this level. Typically, leave this ``true`` if components can fail.
``can_monitor`` turns on or off component-level monitoring. This signifies some type of monitoring in place; if you want to simulate this type of monitoring, set this to ``true``.

Warranty
###############
Components can be set to have a warranty. Components covered under warranty do not incur repair costs when they are repaired. A repaired component resets the warranty to the specified time in this section. For no warranty, remove this section.

Distributions for Failures, Repairs, Monitoring
############################################################
Every failure, repair, and monitoring mode requires a distribution to be defined that dictates how long until the specified action occurs.

PVRPM has some built-in distributions, where only a mean and standard deviation is needed to model the distribution properly. Under the hood, PVRPM uses ``scipy.stats`` distributions to generate samples. However, ``scipy.stats`` documentation for each function is unclear on how to convert the mean and std into usable values for the distribution, which is why PVRPM will do that for you.
However, not every single ``scipy`` distribution is wrapped by PVRPM. These are the distributions wrapped by PVRPM (use these as the distribution option):

  - exponential
  - normal
  - uniform
  - lognormal
  - weibull

Using these distributions as the ``distribution`` parameter for any failure, repair, or monitoring only requires you to provide the mean and standard deviation **in days**. The ``weibull`` distribution also allows you to give the ``shape`` for this distribution instead of the standard deviation. On the `Wikipedia page <https://en.wikipedia.org/wiki/Weibull_distribution>`_ for the weibull distribution is the parameter ``k`` and ``lambda`` is calculated from the mean. Using the ``std`` option, make sure it is large since weibull distributions have large STDs by design.

If these distributions don't properly model your data, you can use any distribution listed in the `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ module. The ``distribution`` parameter in the configuration should be set to the function name of the distribution in the ``scipy.stats`` module. The ``parameters`` key will then be keyword arguments to the function. Make sure to carefully read scipy's documentation, as each function is different in how you need to define it. Remember, the samples from the distribution represent the number of days before that event occurs for a component.

.. code-block:: yaml
  :linenos:

  distribution: normal
  parameters: # parameters for distribution chosen above, either mean and std for a built in distribution or kwargs to the scipy function
    mean: 1460 # years converted to days
    std: 365 # days


Failure Setup
###############
PVRPM currently has two failure modes: total failures and concurrent failures. Total failure modes use the shortest time to failure taken from the defined failures as the time it takes for a component to completely fail. Every component gets a different time to failure, depending on the samples drawn from the distribution.

For a total failure setup, the failure requires the distribution, its parameters, the labor time to fix a component, and the cost to repair the component.

Optionally, there are two fraction modes for a failure: ``fraction`` and ``decay_fraction``. Setting the ``fraction`` will tell PVRPM to fail that fraction (between 0 and 1) of components in the component level consistently throughout the simulation. This means PVRPM will maintain ``fraction`` of the components with this failure mode throughout the simulation. Remember, PVRPM will always pick the failure mode in this section **with the shortest time to failure**, so if you set two failures mode, where one is always shorter than the other, then the longer failure mode will never occur, **even if the fraction is defined on the longer failure mode**.
The ``decay_fraction`` also selects ``decay_fraction`` of the components to fail; however, it decays with each failure. If you set ``decay_fraction`` to 0.5, then at first, 50 percent of the components will fail with this failure mode, then 25  percent, then 12.5 percent, etc., until it approaches 0, which in reality would mean the number of failures from this mode would be 0 when ``decay_fraction`` is small enough.

A typical setup of failures is to have a long "end of life" failure with a large time, and failures with shorter time to failures with a ``fraction`` or ``decay_fraction``, so some will fail with the shorter failures, and most will fail with the end of life failure.

Concurrent failures work the same way as above, except each failure mode is counted **concurrently**. This means that failure modes defined as concurrent failures **do not have the shortest time picked among the modes; instead, each failure mode will fail the component independent of each other and the total failure mode**. You can view this mode as "partial failures", where failures of this nature happen more often than total failures but cost less and are faster to repair. You can use ``fraction`` and ``decay_fraction`` here as needed.

A typical setup for concurrent failure modes is to list routine failures every year or two to a ``fraction`` of the components.

Total failure mode chooses the quickest time to failure from the different modes, and concurrent failure modes all operate independently of each other; they fail each component independent of other failures. Further note, **when a component is repaired from a *total failure*, all *concurrent failures* get reset** since this is a full replacement, and the partial failures that affected the old component won't affect the new one.

.. code-block:: yaml
  :linenos:

  failures:
    eol_failures: # this key name can be anything you want
      distribution: normal
      parameters: # parameters for distribution chosen above, either mean and std for a built in distribution or kwargs to the scipy function
        mean: 3650 # years converted to days
        std: 365 # days
      labor_time: 2 # in hours
      cost: 322 # in USD
    routine_failures: # this key name can be anything you want
      distribution: normal
      parameters:
        mean: 365 # mean in days, or you can do 1 / (num_failures / year * 365)
        std: 365
      labor_time: 2 # in hours
      cost: 322 # in USD
      fraction: 0.1 # > 0 and < 1, fraction of these components that are normal failures, maintained throughout the simulation
    defective_failures: # this key name can be anything you want
      distribution: exponential
      parameters:
        mean: 100 # mean in days, or you can do 1 / (num_failures / year * 365)
      labor_time: 2 # in hours
      cost: 322 # in USD
      decay_fraction: 0.2 # > 0 and < 1, fraction of these components that are defective

  concurrent_failures: # this happens all in parallel, independent of each other
    cell_failure: # this key name can be anything you want
      distribution: normal
      parameters: # parameters for distribution chosen above, either mean and std for a built in distribution or kwargs to the scipy function
        mean: 365 # years converted to days
        std: 365 # days
      labor_time: 2 # in hours
      cost: 322 # in USD
      decay_fraction: 0.2
    wiring_failure: # this key name can be anything you want
      distribution: normal
      parameters:
        mean: 365 # mean in days, or you can do 1 / (num_failures / year * 365)
        std: 365
      labor_time: 2 # in hours
      cost: 322 # in USD
      fraction: 0.1 # > 0 and < 1, fraction of these components that are normal failures, maintained throughout the simulation

Repair Setup
###############
Repairs are much more straightforward. They only need the distribution and its parameters defined for every repair mode. You can either have one repair mode that applies to all failures or a repair mode for each failure mode. You also must list repairs for total failures and concurrent failures separately.

.. code-block:: yaml
  :linenos:

  repairs:
    all_repairs: # this key name can be anything you want
      distribution: lognormal
      parameters:
        mean: 60 # in days
        std: 20 # in days

  concurrent_repairs:
    cell_repair: # this key name can be anything you want
      distribution: lognormal
      parameters:
        mean: 7 # in days
        std: 3 # in days

    wire_repair: # this key name can be anything you want
      distribution: lognormal
      parameters:
        mean: 3 # in days
        std: 3 # in days


Monitoring
###############
Multiple monitoring modes are available for components. You can remove any section you are not using. It is also optional; you can disable all monitoring, in which components that fail are immediately repaired. The modes available are:

  - Component Level: monitoring at the level of the component, which usually offers quick time to detection.
  - Cross Level: monitoring done at a higher level to lower-level components. Meaning inverter monitoring string, combiner, etc.
  - Independent: monitoring done independently of any component level, such as drone IR imaging.

Component level monitoring is defined under each component level's configuration. It simply requires distribution and parameters that signify the time to detection in days to detect a failed component in this level.

.. code-block:: yaml
  :linenos:

  monitoring:
    normal_monitoring: # this key name can be anything you want
      distribution: exponential
      parameters:
        mean: 5

Cross-level monitoring is a bit more complex. Alongside the distribution and parameters, some thresholds control how the monitoring works. A ``global_threshold`` option defines the fraction of components in the monitored level **must fail** before monitoring can detect failed components. This can be seen as enough modules to fail before monitoring at the inverter can start detecting those failures. In PVRPM, this is replicated by the ``global_threshold`` must be met before ``time to detection`` counts down. There is also a ``failure_per_threshold``, the fraction of **lower level** components that must fail **per upper-level component**. For example, if monitoring at the string with a ``failure_per_threshold`` of 0.1, then 10 percent of modules under a single string must fail before the string monitoring can detect module failures. Both thresholds can be defined simultaneously, but one must be defined for this monitoring to work.

.. code-block:: yaml
  :linenos:

  component_level_monitoring:
  # lists what monitoring each component level has for levels BELOW it
  # this is for cross level monitoring only, for defining monitoring at each level use the monitoring distributions above
  string: # component level that has the monitoring for levels below
    module: # componenet level that is below's key. same keys used above: module, string, combiner, inverter, disconnect, grid, transformer
      global_threshold: 0.2 # fraction on [0, 1] that specifies how many of this component type must fail before detection can occur.
                             # this means that until this threshold is met, component failures can never be detected
                             # In the simulation, the time to detection doesn't count down until this threshold is met, which at that point the compouning function will be used along with the distribution as normal
      failure_per_threshold: 0.1 # the fraction of components that must fail per string for failures to be detected at that specified string, or if the total number of failures reach the global_threshold above
      distribution: normal # distribution that defines how long this monitoring takes to detect a failure at this level (independent)
                           # the value calculated from this distribution will be reduced by the compounding factor for every failure in this level
      parameters:
        mean: 1200
        std: 365

  combiner:
    string:
        failure_per_threshold: 0.2 # this fraction of strings must fail on a specific combiner for detection for those to start
        # failures are not compounded globally in this case, only per each combiner
        distribution: normal # distribution that defines how long this monitoring takes to detect a failure at this level (independent)
        parameters:
          mean: 1825
          std: 365

Independent monitoring works outside of component levels. It represents monitoring that **detects all failures in any component level** instantly. It can happen statically every set number of days, defined by the ``interval``, or at a threshold of failed components. There are a few ways to define this threshold. First, the threshold can be defined as ``global_threshold``, which works differently than cross-level monitoring. This value is based on the **DC availability**, meaning the power reaching the inverter. This is calculated using the operating strings and combiners to determine how many modules reach the inverter. With this, combiners and strings are weighted higher than module failures.

The other way to define a threshold is more similar to cross-level monitoring. Using the ``failure_per_threshold`` sets a threshold of failed components for **each level** that must be reached before monitoring occurs. This uses OR logic, meaning only one level has to drop below this threshold for the independent monitoring for **all levels**.

Finally, you can combine all these arguments together; ``interval``, ``global_threshold``, and ``failure_per_threshold``.

You must specify the labor time for each independent monitoring defined, which is in hours for other parameters. There is also **an optional distribution and parameters** that can be defined as the ``time_to_detection`` for components under the levels when the independent monitoring occurs. Think of it as the time it takes to complete the independent monitoring. Not setting this means that the ``time_to_detection`` gets set to zero when independent monitoring occurs.

.. code-block:: yaml
  :linenos:

  indep_monitoring:
    drone_ir:  # this name can be anything you want!
      interval: 1095 # interval in days when this static monitoring occurs
      cost: 50000 # cost of this monitoring in USD
      labor_time: 1 # in hours
      distribution: normal
      parameters:
        mean: 14
        std: 5
      levels: # the component levels this detects on
        - module
        - string
        - combiner

    drone_ir2: # list as many static monitoring methods as you want
      interval: 365 # this monitoring will happen every 365 days, alongside the threshold.
      # a indep monitoring triggered by a threshold RESETs the countdown to the interval
      global_threshold: 0.1 # if DC availability drops by this threshold amount, then this indep monitoring will occur
      # DC availability is the DC power reaching the inverter(s), which is affected by combiners, strings, and module failures
      failure_per_threshold: 0.2 # this threshold is PER LEVEL, if the availability of ANY of the defined levels drops by this threshold amount, this indep monitoring will occur
      cost: 100000
      labor_time: 1 # in hours
      levels:
        - module
        - combiner
        - string

    drone_ir3: # list as many static monitoring methods as you want
      failure_per_threshold: 0.2 # this threshold is PER LEVEL, people if the availability of ANY of the defined levels drops by this threshold amount, this indep monitoring will occur
      cost: 1500
      labor_time: 1 # in hours
      levels:
        - module

Running the simulation
------------------------
The example configuration provided shows how all these options are defined; please consult it as necessary.

Now that you have your SAM case JSONs, and your PVRPM configuration, you can run the simulation:

.. code-block:: bash
  :linenos:

  pvrpm run --case /path/to/directory/with/jsons /path/to/pvrpm/config.yaml


You can also parallelize realizations to decrease the overall run time. To use all your CPU cores to run PVRPM:

.. code-block:: bash
  :linenos:

  pvrpm run --case /path/to/directory/with/jsons --threads 0 /path/to/pvrpm/config.yaml

  PVRPM will alert you to unknown keys in your configuration if you misspelled something and tell you any incorrect or missing parameters you may have.

Once the simulation is completed, result graphs and CSV files will be saved to the defined results folder.
