**************************
v3.0.0 (December XX, 2024)
**************************

Enhancements
------------
* Add `CITATION.cff` file for citation information (:pull:`434`)
* Added checks to TrendAnalysis for `filter_params` and `filter_params_aggregated`. Raises an error if unkown filter is supplied. (:pull:`436`)


Bug fixes
---------
* Set marker linewidth to zero in `rdtools.plotting.degradation_summary_plots` (:pull:`433`)
* Fix `energy_from_power`` returns incorrect index for shifted hourly data (:issue:`370`, :pull:`437`)


Requirements
------------
* Updated tornado==6.4.2 in ``notebook_requirements.txt`` (:pull:`438`)

