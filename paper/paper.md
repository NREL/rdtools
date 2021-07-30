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
  - name: Author 1
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Author 2
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Author 3
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
 - name: National Renewable Energy Laboratory, Golden, CO
   index: 1
 - name: Affiliation 2
   index: 2
date: Day Month Year
bibliography: paper.bib
---

# Introduction

Financial analysts assess the financial viability of commercial photovoltaic
(PV) projects with technical system performance models.  The expected energy
production for a given PV system configuration and location is modeled using
a weather dataset for the location and various assumptions about performance
losses over time.  Among these loss sources are longterm PV system degradation
(gradual system efficiency loss as the system ages), array soiling
(sunlight blocking by particulate accumulation on the array), and availability
losses (system component downtime due to electrical faults or other reasons).
Reducing the uncertainty in the loss assumptions included in PV
system modeling carries significant financial value, but these losses can vary
from system to system because of differences in system design, component technology
and manufacturer, climate conditions, and other influences, making it difficult
to make generalizations about these loss factors.

Analysis of long-term datasets from fielded systems is one route to
understanding how these loss factors vary across system and climate parameters. 
PV modules often experience slow degradation rates below one percent per year
[@Jordan2016] and the multi-year datasets from operational PV systems enable
analysis of how degradation is influenced by climactic conditions and PV cell
technology.  Soiling rates depend strongly on local conditions, so extracting
soiling signals from historical field data gives insight to regional trends.
Estimated production loss from poor system availability can be paired with
component metadata to reveal trends across manufacturers and component types.

However, extracting these loss signals from operational time series data can
be difficult due to the complications seen in real-world datasets -- if not
appropriately accounted for, data loss, seasonality, sensor calibration drift,
and other confounding factors can bias extracted loss rates. Additionally,
differences in analysis techniques and analyst preferences can lead
to inconsistent conclusions [@Jordan2020].  RdTools is an open-source library
to support reproducible technical analysis of PV time series data. The library
aims to provide best practice analysis routines along with the building blocks
for users to tailor their own analyses. In particular, PV production data 
across several years is evaluated to obtain rates of performance degradation,
soiling loss, and system availability.  Documentation of the analysis algorithms
implemented in RdTools have been separately published.

# RdTools

A typical RdTools degradation and soiling analysis follows this process:

1) Read in timeseries production and weather data
2) Normalize the measured production using a naive expected energy model
3) Filter out data that might bias the analysis results
4) Aggregate data to reduce noise and scatter
5) Analyze aggregated performance to extract degradation and soiling rates

RdTools offers a high-level `TrendAnalysis` class to simplify and
standardize the process of running an end-to-end data analysis. The high-level
interface does allow some customization of the analysis but defaults to best
practice methods to encourage standardized and reproducible analysis. However,
RdTools also provides access to the low-level analysis functions used by
`TrendAnalysis` so users can construct fully customized analyses as needed.
Although some of its functions are widely applicable to PV data analysis,
RdTools currently focuses on PV system degradation analysis [@Jordan2018] and
PV array soiling analysis [@Deceglie2018].  Future releases will include
functionality for assessing other sources of PV system underperformance as well
as improvements to the degradation and soiling workflows to keep pace with
state-of-the-art practices. 

RdTools is available under an MIT license and is developed on GitHub by
contributors from national laboratories and
industry.  It relies on the pvlib Python package [@pvlib] and packages from the
broader scientific Python ecosystem: pandas [@pandas], numpy [@numpy],
and matplotlib [@matplotlib].  The first major release (1.0.0, November 2016)
focused on system degradation rate analysis.  Subsequent releases have brought
improvements to the degradation analysis as well as new methods for soiling and
availability analysis.

API documentation, usage examples, and other package documentation
is available at its online documentation hosted by readthedocs. 

# Applications

RdTools is intended for use by academic researchers, industry engineers, and
PV system owners and operators.  It is part of a growing ecosystem of 
open-source Python packages for PV modeling and analysis [@Holmgren2018].

To date, RdTools has been used in over 40 journal articles conference
papers.  For example, Meyers et al. used
RdTools to benchmark a novel degradation rate assessment method [@Meyers2020]
and Deceglie et al. applied RdTools to timeseries data from over 500 PV systems
to analyze how PV system configuration affects degradation rate [@Deceglie2019].
RdTools is in active use as the core analysis package for the National
Renewable Energy Laboratory's PV Fleet Performance Data Initiative
[@PVFleets2019].  It is also used for analyses by other national laboratories
and several PV industry groups.  

# Acknowledgements

The authors acknowledge support from the U.S. Department of Energy’s Solar
Energy Technologies Office. This work was authored, in part, by Alliance for
Sustainable Energy, LLC, the manager and operator of the National Renewable 
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