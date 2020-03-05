---
title: 'RdTools: a python package for reproducible timeseries analysis of photovoltaic systems'
tags:
  - Python
  - solar energy
  - photovoltaics
  - renewable energy
authors:
  - name: 
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: 
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
 - name: 
   index: 1
 - name: 
   index: 2
date: 13 August 2017 
bibliography: paper.bib
---

# Summary

Accurate technical performance modeling is crucial to correctly assess the
financial viability of commercial photovoltaic (PV) projects.  Analysis of
timeseries datasets from operational PV systems can inform modeling of future
projects, but differences in analysis techniques and analyst preferences can
lead to inconsistent results [@Jordan2020]. RdTools is an open-source library
to support reproducible technical analysis of PV time series data. The library
aims to provide best practice analysis routines along with the building blocks
for users to tailor their own analyses.

The RdTools API offers a high-level `SystemAnalysis` class to simplify and
standardize the process of running an end-to-end data analysis. The high-level
interface does allow some customization of the analysis but defaults to best
practice methods to encourage standardized and reproducible analysis. However,
RdTools also provides access to the low-level analysis functions used by
`SystemAnalysis` so users can construct fully customized analyses as needed.

Although some of its functions are widely applicable to PV data analysis,
RdTools currently focuses on PV system degradation analysis [@Jordan2018] and
PV module soiling analysis [@Deceglie2018].  Future releases will include
functionality for assessing other sources of PV system underperformance.

RdTools has been used in several PV degradation studies.  Meyers et al. used
RdTools to benchmark a novel degradation rate assessment method [Meyers2020]
and Deceglie et al. applied RdTools to timeseries data from over 500 PV systems
to analyze how PV system configuration affects degradation rate [Deceglie2019].
RdTools is in active use as the core analysis package for the National
Renewable Energy Laboratory's PV Fleet Performance Data Initiative
[PVFleets2019].

RdTools is intended for use by academic researchers, industry engineers, and
PV system owners and operators.  It is part of a broader ecosystem of 
open-source python packages for PV modeling and analysis [@Holmgren2018].
RdTools is developed on GitHub by contributors from national laboratories and
industry. API documentation, usage examples, and other package documentation
is available at its online documentation hosted by readthedocs. 

# Acknowledgements

To-Do

# References