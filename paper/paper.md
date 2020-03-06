---
title: 'RdTools: a python package for reproducible timeseries analysis of photovoltaic systems'
tags:
  - Python
  - solar energy
  - photovoltaics
  - degradation
  - soiling
  - system performance
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

# Introduction

Assessing the financial viability of commercial photovoltaic (PV) projects is
based on technical system performance modeling.  The expected energy
production for a given PV system configuration and location is modeled using
a weather dataset for the location and various assumptions about performance
losses over time.  Among these loss sources are longterm PV system degradation
(gradual production loss as the PV array ages) and array soiling (temporary
production loss due to particulate accumulation on the array).  Because these
effects are strongly influenced by local climate, operational data from
existing PV systems can offer great insight into performance losses.

Although analysis of PV system datasets helps inform modeling of future
projects, differences in analysis techniques and analyst preferences can lead
to inconsistent conclusions [@Jordan2020]. RdTools is an open-source library
to support reproducible technical analysis of PV time series data. The library
aims to provide best practice analysis routines along with the building blocks
for users to tailor their own analyses. In particular, PV production data 
across several years is evaluated to obtain rates of performance degradation
and soiling loss.  

# RdTools

A typical RdTools analysis follows this process:

1) Read in timeseries production and weather data
2) Normalize the measured production using a naive expected energy model
3) Filter out data that might bias the analysis results
4) Aggregate data to reduce noise and scatter
5) Analyze aggregated production to extract degradation and soiling rates

The RdTools API offers a high-level `SystemAnalysis` class to simplify and
standardize the process of running an end-to-end data analysis. The high-level
interface does allow some customization of the analysis but defaults to best
practice methods to encourage standardized and reproducible analysis. However,
RdTools also provides access to the low-level analysis functions used by
`SystemAnalysis` so users can construct fully customized analyses as needed.
Although some of its functions are widely applicable to PV data analysis,
RdTools currently focuses on PV system degradation analysis [@Jordan2018] and
PV array soiling analysis [@Deceglie2018].  Future releases will include
functionality for assessing other sources of PV system underperformance.

RdTools is developed on GitHub by contributors from national laboratories and
industry.  It relies on the pvlib [@pvlib] Python package as well as the
broader scientific python ecosystem (e.g. pandas [@pandas], numpy [@numpy],
and matplotlib [@matplotlib]).

API documentation, usage examples, and other package documentation
is available at its online documentation hosted by readthedocs. 

# Applications

RdTools has been used in several PV degradation studies.  Meyers et al. used
RdTools to benchmark a novel degradation rate assessment method [@Meyers2020]
and Deceglie et al. applied RdTools to timeseries data from over 500 PV systems
to analyze how PV system configuration affects degradation rate [@Deceglie2019].
RdTools is in active use as the core analysis package for the National
Renewable Energy Laboratory's PV Fleet Performance Data Initiative
[@PVFleets2019].

RdTools is intended for use by academic researchers, industry engineers, and
PV system owners and operators.  It is part of a broader ecosystem of 
open-source python packages for PV modeling and analysis [@Holmgren2018].

# Acknowledgements

The authors acknowledge support from the U.S. Department of Energy’s Solar
Energy Technologies Office. This work was authored, in part, by Alliance for
SustainableEnergy, LLC, the manager and operator of the National Renewable 
Energy Laboratory for the U.S. Department of Energy (DOE) under Contract
No. DE-AC36-08GO28308. Funding provided by U.S. Department of Energy Office
of Energy Efficiency and Renewable Energy Solar Energy Technologies Office.
The views expressed in the article do not necessarily represent the views of
the DOE or the U.S. Government. The U.S. Government retains and the publisher,
by accepting the article for publication, acknowledges that the U.S. Government
retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or
reproduce the published form of this work, or allow others to do so, for U.S.
Government purposes. 

# References