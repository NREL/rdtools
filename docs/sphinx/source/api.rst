.. currentmodule:: rdtools

#############
API reference
#############


Submodules
==========

RdTools is organized into submodules focused on different parts of the data
analysis workflow.  

.. autosummary::
   :toctree: generated/

   analysis
   degradation
   soiling   
   availability
   filtering
   normalization
   aggregation
   clearsky_temperature
   plotting

Analysis
===========

Conduct end-to-end Degradation and Soiling analysis with sensor_analysis or clearsky_analysis

.. autosummary::
   :toctree: generated/

   analysis.RdAnalysis
   analysis.RdAnalysis.sensor_analysis
   analysis.RdAnalysis.clearsky_analysis
   analysis.RdAnalysis.plot_degradation_summary
   analysis.RdAnalysis.plot_soiling_rate_histogram
   analysis.RdAnalysis.plot_soiling_interval
   analysis.RdAnalysis.plot_soiling_monte_carlo
   analysis.RdAnalysis.plot_pv_vs_irradiance
   
   
Degradation
===========

.. automodule:: rdtools.degradation
   :noindex:

.. autosummary::
   :toctree: generated/

   degradation_classical_decomposition
   degradation_ols
   degradation_year_on_year


Soiling
=======

.. automodule:: rdtools.soiling
   :noindex:

.. autosummary::
   :toctree: generated/

   soiling_srr
   monthly_soiling_rates
   annual_soiling_ratios
   SRRAnalysis
   SRRAnalysis.run


System Availability
===================

.. automodule:: rdtools.availability
   :noindex:

.. autosummary::
   :toctree: generated/
   
   AvailabilityAnalysis
   AvailabilityAnalysis.run
   AvailabilityAnalysis.plot


Filtering
=========

.. automodule:: rdtools.filtering
   :noindex:

.. autosummary::
   :toctree: generated/
    
   clip_filter
   csi_filter
   poa_filter
   tcell_filter
   normalized_filter


Normalization
=============

.. automodule:: rdtools.normalization
   :noindex:

.. autosummary::
   :toctree: generated/

   energy_from_power
   interpolate
   irradiance_rescale
   normalize_with_expected_power
   normalize_with_pvwatts
   normalize_with_sapm
   pvwatts_dc_power
   sapm_dc_power
   delta_index
   check_series_frequency


Aggregation
===========

.. automodule:: rdtools.aggregation
   :noindex:

.. autosummary::
   :toctree: generated/

   aggregation_insol


Clear-Sky Temperature
=====================

.. automodule:: rdtools.clearsky_temperature
   :noindex:

.. autosummary::
   :toctree: generated/

   get_clearsky_tamb


Plotting
========

.. automodule:: rdtools.plotting
   :noindex:

.. autosummary::
   :toctree: generated/

   degradation_summary_plots
   soiling_monte_carlo_plot
   soiling_interval_plot
   soiling_rate_histogram
   availability_summary_plots
