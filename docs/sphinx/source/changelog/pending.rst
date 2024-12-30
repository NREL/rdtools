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
* Fix :py:func:`~rdtools.normalization.energy_from_power` returns incorrect index for shifted hourly data (:issue:`370`, :pull:`437`)
* Add warning to clearsky workflow when ``power_expected`` is passed by user (:pull:`439`)
* Fix different results with Nan's and Zeros in power series (:issue:`313`, :pull:`442`)
* Fix pandas deprecation warnings in tests (:pull:`444`)


Requirements
------------
* Updated tornado==6.4.2 in ``notebook_requirements.txt`` (:pull:`438`)
* Updated Jinja2==3.1.5 in ``notebook_requirements.txt`` (:pull:`447`)


Tests
-----
* Add tests for pvlib clearsky fiter in analysis chain (:pull:`441`)
