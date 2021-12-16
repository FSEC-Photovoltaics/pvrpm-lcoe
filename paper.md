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
  - name: Marios Theristis^[corresponding author]
    affiliation: 2
  - name: Hubert Seigneur^[corresponding author]
    affiliation: 1

affiliations:
 - name: University of Central Florida
   index: 1
 - name: Sandia National Laboratories
   index: 2

date: 08 December 2021
bibliography: paper.bib

---

# Summary

The ability to perform accurate techno-economic analysis of photovoltaic (PV) systems is essential for bankability and investment purposes. Most energy yield models assume an almost flawless operation (i.e., no failures); however, realistically, components fail and get repaired stochastically. This package, PyPVRPM, is a Python translation and improvement of the LK-based PhotoVoltaic Reliability Performance Model (PVRPM). PyPVRPM is a simulation tool that uses NREL's SAM [@SAM:2020] software to model the performance of a PV plant throughout its lifetime by considering component reliability metrics. It does so by NREL's Python interface for SAM [@PYSAM]. Besides the numerous benefits from migrating to python (e.g., speed, libraries, batch analyses, etc.), it also expands from the failure and repair processes from the LK version by including the ability to vary monitoring strategies. These are based on user-defined distributions and values, enabling a more accurate and realistic representation of cost and availability throughout a PV system's lifetime. In addition to the benefits in energy yield and financial estimates, one can vary repair rates and monitoring solutions to compare different configurations. These configurations can then optimize operation and maintenance (O&M) scheduling in actual PV power plants and investigate various end-of-life scenarios.

# Statement of need

As photovoltaic technology becomes cheaper and more widely used, solutions are needed to monitor these systems to safeguard performance and reduce downtime. However, it is not easy to quantify the benefits of different levels of monitoring (e.g., inverter, combiner, string, or module-level monitoring) since it depends on the quality of components, financial metrics, and climatic conditions. PyPVRPM provides the ability to perform detailed stochastic analysis on a system over significant periods that can be finely tuned to each use case for comparison.

The original PVRPM was written in SAM's LK scripting language [@PVRPM]. The LK scripting language lacked many needed features to expand on this original work, such as efficient matrix operations, command-line interfaces, and a lack of external libraries to perform computations. Furthermore, the original PVRPM did not include simulation of monitoring techniques, which PyPVRPM implements. PyPVRPM takes the original logic from PVRPM, refines it, and adds new features to make it faster, easily configurable, more powerful. This is done using established Python libraries such as numpy and pandas. PyPVRPM also can batch simulate in an automated manner on a computer cluster. PyPVRPM can simulate any component of a PV plant from the module level up to the grid.

Researchers and relevant stakeholders will be able (but not limited) to use this package to a) determine realistic energy yields and revenues, b) examine whether new and improved monitoring solutions provide benefits to PV investments, and c) optimize O&M and end-of-life strategies. Repairs and failure rates can also be compared to perform risk assessments depending on how many failures are expected in a given PV system's lifetime. This library enables bankability, low risk, and optimization studies. As an open-source software in python, PyPVRPM opens up new opportunities to the PV community to accelerate the development of new capabilities within the PhotoVoltaic Reliability Performance Model.

# Acknowledgements

This work was supported by the U.S. Department of Energy's Office of Energy Efficiency and Renewable Energy (EERE) under the Solar Energy Technologies Office Award Number DE-EE0008157.
Sandia National Laboratories is a multimission laboratory managed and operated by National Technology & Engineering Solutions of Sandia, LLC, a wholly-owned subsidiary of Honeywell International Inc., for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA0003525. This paper describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the paper do not necessarily represent the views of the U.S. Department of Energy or the United States Government.

# References
