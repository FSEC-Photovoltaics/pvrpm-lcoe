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

The ability to perform accurate techno-economic analysis of solar photovoltaic (PV) systems is essential for bankability and investment purposes. Most energy yield models assume an almost flawless operation (i.e., no failures); however, realistically, components fail and get repaired stochastically. This package, PyPVRPM, is a Python translation and improvement of the Language Kit (LK) based PhotoVoltaic Reliability Performance Model (PVRPM), which was first developed at Sandia National Laboratories in Goldsim software [@PVRPM:2011] [@PVRPM:2012]. PyPVRPM allows the user to define a PV system at a specific location and incorporate failure, repair, and detection rates and distributions to calculate energy yield and other financial metrics such as the levelized cost of energy and net present value [@PVRPM:2017]. Our package is a simulation tool that uses NREL's Python interface for System Advisor Model (SAM) [@SAM:2020] [@PYSAM] to evaluate the performance of a PV plant throughout its lifetime by considering component reliability metrics. Besides the numerous benefits from migrating to Python (e.g., speed, libraries, batch analyses), it also expands on the failure and repair processes from the LK version by including the ability to vary monitoring strategies. These failures, repairs, and monitoring processes are based on user-defined distributions and values, enabling a more accurate and realistic representation of cost and availability throughout a PV system's lifetime.  

# Statement of need

As photovoltaic technology becomes cheaper and more widely used, solutions are needed to accurately assess its performance and monitor it to safeguard performance and reduce downtime. Existing PV performance models assume ideal operation without considering any failures or repairs. In contrast, the value of monitoring at different levels (e.g., inverter, combiner, string, module-level monitoring) is unclear since these depend on the quality of components, financial metrics, and climatic conditions. PVRPM was initially developed in Goldsim [@PVRPM:2011] [@PVRPM:2012] and then adapted in LK script [@osti_1761998] to improve the accuracy of energy yield simulations by inclusion of realistic energy losses, including operations and maintenance (O&M) costs [@PVRPM:EVAL]. However, these models lack many needed features such as efficient matrix operations, command-line interfaces, and external libraries to perform computations.

Furthermore, the LK-based PVRPM does not include simulation of monitoring configurations (e.g., inverter-level vs. string-level monitoring), which PyPVRPM implements. PyPVRPM takes the original logic from PVRPM, refines it, and adds new features to make it faster, easily configurable, and more powerful. This is done using established Python libraries such as NumPy and Pandas. PyPVRPM can also batch simulate in an automated manner on a computer cluster. PyPVRPM can simulate any component of a PV plant from the module level up to the grid while also considering different monitoring scenarios.

Researchers and relevant stakeholders will be able (but not limited) to use this package to a) determine realistic energy yields and revenues, b) examine whether new and improved monitoring solutions provide benefits to PV investments, and c) optimize O&M and end-of-life replacement strategies. Repair and failure rates can also be compared to perform risk assessments depending on how many failures are expected in a given PV system's lifetime. This library enables bankability and optimization studies. As open-source software in Python, PyPVRPM opens up new opportunities to the PV community to accelerate the development of new capabilities within the PhotoVoltaic Reliability Performance Model.

# Acknowledgements

This work was supported by the U.S. Department of Energy's Office of Energy Efficiency and Renewable Energy (EERE) under the Solar Energy Technologies Office Award Number DE-EE0008157.
Sandia National Laboratories is a multimission laboratory managed and operated by National Technology & Engineering Solutions of Sandia, LLC, a wholly-owned subsidiary of Honeywell International Inc., for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA0003525. This paper describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the paper do not necessarily represent the views of the U.S. Department of Energy or the United States Government.

# References
