---
title: 'PyPVRPM: Photovoltaic Reliability and Performance Model in Python'
tags:
  - Python
  - LCOE
  - Photovoltaic
  - Solar
  - Energy
  - Cost model
authors:
  - name: Brandon Silva^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: 1
  - name: Paul Lunis^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: 1
  - name: Hubert Seigneur^[corresponding author]
    affiliation: 1
  - name: Marios Theristis^[corresponding author]
    affiliation: 1
affiliations:
 - name: University of Central Florida
   index: 1

date: 08 December 2021
bibliography: paper.bib

---

# Summary

In photovoltaics (PV), the ability to perform an accurate techno-economic analysis is essential. Creating economically efficient PV systems is beneficial to both consumers and producers alike. This package: Python PhotoVoltaic Reliability Performance Model (PyPVRPM), fills this need. PyPVRPM is a simulation tool that uses NREL's SAM [@SAM:2020] software to model the performance of a PV plant throughout its lifetime. It simulates failures, repairs, and monitoring across the lifespan based on user-defined distributions and values. This allows a more accurate representation of cost and availability throughout a lifetime than SAM's base simulation done from the GUI. By varying repair rates and monitoring solutions, one can compare different configurations to find the most optimal setup for implementing an actual PV power plant.

# Statement of need

As photovoltaic technology becomes cheaper and more widely used, solutions are needed to monitor these systems, reducing the downtime of PV power plants. However, it is difficult to determine the cost and benefits of different monitoring techniques and repairing failures when detected in the field. PyPVRPM provides a solution to do a detailed analysis on a system over significant periods that can be finely tuned to each use case for comparison.

The original PVRPM was written in SAM's LK scripting language [@PVRPM]. The LK scripting language lacked many needed features to expand on this original work, such as efficient matrix operations, command-line interfaces, and a lack of external libraries to perform computations. The original PVRPM also did not include simulation of monitoring techniques, which PyPVRPM implements. PyPVRPM takes the original logic from PVRPM, refines it, and adds new features to make it faster, easily configurable, and run multiple simulations at once. PyPVRPM can simulate many of the levels in a PV plant, from the module level up to the grid. This tool will be crucial in helping the Department of Energy (DOE) reach the goal of three cents per kilowatt-hour Levelized cost of energy (LCOE) for PV plants.

Researchers will be able to use this package to help determine if new and improved monitoring techniques provide enough benefits to PV systems. Repairs and failure rates can also be compared to perform risk assessments depending on how many failures are expected in a given PV system's lifetime. It is easy to see what solution would be the best by running these simulations. Companies looking to create new PV power plants can utilize this package to show investors or banks that their proposed PV system is viable to secure investments or loans.

# Acknowledgements

We acknowledge the financial support of the DOE in developing this package.

# References
