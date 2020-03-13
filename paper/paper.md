---
title: 'DRAFT: RdTools: a Python package for reproducible timeseries analysis of photovoltaic systems'
tags:
  - Python
  - solar energy
  - photovoltaics
  - degradation
  - soiling
  - system performance
  - renewable energy
authors:
  - name: Kevin Anderson
    orcid: 0000-0002-1166-7957
    affiliation: 1
  - name: Michael Deceglie
    orcid: 0000-0001-7063-9676
    affiliation: 1
  - name: Chris Deline
    orcid: 0000-0002-9867-8930
    affiliation: 1
  - name: Adam Shinn
    orcid: 0000-0002-5473-3299
    affiliation: 2
affiliations:
 - name: National Renewable Energy Laboratory, Golden, CO
   index: 1
 - name: kWh Analytics
   index: 2
date: 5 March 2020
bibliography: paper.bib
---

# Introduction

Financial analysts assess the financial viability of commercial photovoltaic
(PV) projects with technical system performance models.  The expected energy
production for a given PV system configuration and location is modeled using
a weather dataset for the location and various assumptions about performance
losses over time.  Among these loss sources are longterm PV system degradation
(gradual PV efficiency loss as the system ages) and array soiling
(sunlight blocking by particulate accumulation on the array).  Because these
losses vary in magnitude across system designs and local climates, it
is useful to characterize them using operational measurements from
existing PV systems to improve modeling of future systems.

However, extracting these loss signals from operational time series data can
be difficult due to the complications seen in real-world datasets -- data loss,
seasonality, sensor calibration drift, and other confounding factors can bias
extracted loss rates if not handled appropriately. Additionally,
differences in analysis techniques and analyst preferences can lead
to inconsistent conclusions [@Jordan2020].  RdTools is an open-source library
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

RdTools offers a high-level `SystemAnalysis` class to simplify and
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
industry.  It relies on the pvlib Python package [@pvlib] and the
broader scientific Python ecosystem (e.g. pandas [@pandas], numpy [@numpy],
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
[@PVFleets2019].  It is also used for internal analyses by several PV industry
groups. 

RdTools is intended for use by academic researchers, industry engineers, and
PV system owners and operators.  It is part of a growing ecosystem of 
open-source Python packages for PV modeling and analysis [@Holmgren2018].

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