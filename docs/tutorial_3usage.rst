Usage
=================================
.. toctree::
  :hidden:

Command line usage for PVRPM:

.. code-block:: bash
  :linenos:

  $ pvrpm --help
    Usage: pvrpm [OPTIONS] COMMAND [ARGS]...

      Perform cost modeling for PV systems using SAM and PVRPM

    Options:
      --help  Show this message and exit.

    Commands:
      run     Run the PVRPM LCOE cost model for the case
      sim     Load the SAM case and test the basic SAM simulation
      verify  Verify the case and pvrpm configuration files


Verify configuration:
-----------------------------

.. code-block:: bash
  :linenos:

  $ pvrpm verify --help
    Usage: pvrpm verify [OPTIONS] CONFIG

      Verify the case and pvrpm configuration files

    Options:
      --case <path>  Path to directory containing json export from SAM for the
                     case
      --help         Show this message and exit.

  $ pvrpm verify --case /path/to/case/jsons/ /path/to/config.yml
    2022-02-23 13:30:06,142--INFO: Configuration verified successfully!

Run SAM simulation:
-----------------------------
Mainly for debugging purposes.

.. code-block:: bash
  :linenos:

  $ pvrpm sim --help
    Usage: pvrpm sim [OPTIONS] CONFIG

      Load the SAM case and test the basic SAM simulation

      The config YAML file should specify module order of simulation

    Options:
      --case <path>  Path to directory containing json export from SAM for the
                     case
      --verbose      Enable verbosity in SAM simulation
      --help         Show this message and exit.

  $ pvrpm verify --case /path/to/case/jsons/ --verbose /path/to/config.yml
    0.08 %  @ -1
    0.16 %  @ -1
    0.24 %  @ -1
    0.32 %  @ -1
    0.40 %  @ -1
    0.48 %  @ -1
    0.56 %  @ -1
    0.64 %  @ -1
    ...
    99.76 %  @ -1
    99.84 %  @ -1
    99.92 %  @ -1

Run PVRPM simulation:
-----------------------------

.. code-block:: bash
  :linenos:

  $ pvrpm run --help
    Usage: pvrpm run [OPTIONS] CONFIG

    Run the PVRPM LCOE cost model for the case

    The config YAML file should specify module order of simulation

    Options:
    --case <path>                   Path to directory containing json export
                                    from SAM for the case
    --threads <num_threads>         Number of threads to use for paralized
                                    simulations, set to -1 to use all CPU
                                    threads
    --realization, --realizations <num_realizations>
                                    Set the number of realizations for this run,
                                    overrides configuration file value
    --results <path/to/results/folder>
                                    Folder to use for saving the results,
                                    overrides configuration file value
    --trace                         Enable debug stack traces
    --debug INTEGER                 Save simulation state every specified number
                                    of days for debugging. Saved to results
                                    folder specified
    --noprogress                    Disable progress bars for realizations
    --help

  $ pvrpm run --case /path/to/case/jsons/ --threads -1 --realizations 10 --trace /path/to/config.yml
    2022-02-23 13:51:36,250--WARNING: Lifetime daily DC and AC losses will be overridden for this run.
    2022-02-23 13:51:36,250--WARNING: There is a non-zero value in the fixed annual O&M costs input. These will be overwritten with the new values.
    2022-02-23 13:51:36,250--WARNING: Degradation is set by the PVRPM script, you have entered a non-zero degradation to the degradation input. This script will set the degradation input to zero.
    2022-02-23 13:51:36,252--INFO: Running base case simulation...
    2022-02-23 13:52:24,146--INFO: Base case simulation took: 47.89 seconds

    2022-02-23 13:56:35,089--INFO: Generating results...
    2022-02-23 13:56:39,469--INFO: Graphs saved to /path/to/results/
    2022-02-23 13:56:43,264--INFO: Results saved to /path/to/results/
