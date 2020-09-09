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

   degradation
   soiling   
   filtering
   normalization
   aggregation
   clearsky_temperature
   plotting


Degradation
===========

Functions for estimating degradation rates from PV system data.

.. autosummary::
   :toctree: generated/

   degradation.degradation_classical_decomposition
   degradation.degradation_ols
   degradation.degradation_year_on_year
   degradation.bootstrap_YOY


Soiling
=======

Functions for estimating soiling rates from PV system data. 

.. autosummary::
   :toctree: generated/

   soiling.soiling_srr
   soiling.SRRAnalysis
   soiling.SRRAnalysis.run
   soiling.soiling_cods
   soiling.CODSAnalysis
   soiling.CODSAnalysis.iterative_signal_decomposition
   soiling.CODSAnalysis.run_bootstrap


Filtering
=========

Functions to perform filtering on PV system data. 

.. autosummary::
   :toctree: generated/
    
   filtering.clip_filter
   filtering.csi_filter
   filtering.poa_filter
   filtering.tcell_filter
   filtering.normalized_filter


Normalization
=============

Functions for normalizing power measurements for further analysis. 

.. autosummary::
   :toctree: generated/

   normalization.check_series_frequency
   normalization.delta_index
   normalization.energy_from_power
   normalization.interpolate
   normalization.interpolate_series
   normalization.irradiance_rescale
   normalization.normalize_with_expected_power
   normalization.normalize_with_pvwatts
   normalization.normalize_with_sapm
   normalization.pvwatts_dc_power
   normalization.sapm_dc_power
   normalization.t_step_nanoseconds
   normalization.trapz_aggregate


Aggregation
===========

Functions to aggregate PV system data. 

.. autosummary::
   :toctree: generated/

   aggregation.aggregation_insol


Clear-Sky Temperature
=====================

Functions for modeling ambient temperature. 

.. autosummary::
   :toctree: generated/

   clearsky_temperature.get_clearsky_tamb


Plotting
========

Functions to visualize degradation and soiling analysis results. 

.. autosummary::
   :toctree: generated/

   plotting.degradation_summary_plots
   plotting.soiling_monte_carlo_plot
   plotting.soiling_interval_plot
   plotting.soiling_rate_histogram
