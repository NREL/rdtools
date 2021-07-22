************************
Pending
************************

API Changes
-----------
* The calculations internal to the soiling SRR algorithm have changed such that
  consecutive cleaning events are no longer removed. (:pull:`199`, :issue:`189`)

* The calculations internal to the soiling SRR algorithm have changed such that
  "invalid" intervals are retained at the beginning and end of the dataset for the
  purposes of the SRR Monte Carlo.  Invalid intervals are those that do not qualify
  to be fit as soiling intervals based on  ``min_interval_length``,
  ``max_relative_slope_error``, and ``max_negative_step``. (:pull:`199`, :issue:`272`)

* The default ``day_scale`` parameter in soiling functions and methods was changed
  from 14 to 13. A recommendation to use an odd value along with a warning for even
  values was also added. (:pull:`199`, :issue:`189`)

* The default ``min_interval_length`` in soiling functions and methods was changed
  from 2 to 7. (:pull:`199`)

Enhancements
------------

* A new parameter ``outlier_factor`` was added to soiling functions and methods to
  enable better control of cleaning event detection. (:pull:`199`)

* Add ``sensor_filter_components`` and ``clearsky_filter_components`` to
  :py:class:`~rdtools.analysis_chains.TrendAnalysis` (:issue:`236`, :pull:`263`)

* Boolean input kwarg ``detailed`` has been added to
  :py:class:`~rdtools.plotting.degradation_summary_plots` to color-code 
  degradation plots by the number of times data points are used in the
  degradation distribution. (:issue:`269`, :pull:`282`)
  
* :py:class:`~rdtools.degradation.degradation_year_on_year` adds a new 
  ``usage_of_points`` entry in the ``calc_info`` return dictionary.
  (:issue:`269`, :pull:`282`)


Bug fixes
---------
* Unexpected recoveries when using ``method=random_clean`` in the soiling module
  have been fixed. (:pull:`199`, :issue:`234`)

* Improved NaN pixel handling in
  :py:func:`~rdtools.clearsky_temperature.get_clearsky_tamb` (:pull:`274`).

Testing
-------



Documentation
-------------
* Corrected a typo in the :py:class:`~rdtools.analysis_chains.TrendAnalysis`
  docstring (:pull:`264`)
* Enabled intersphinx so that function parameter types are linked to external
  documentation (:pull:`258`)


Enhancements
------------
* Add new function :py:func:`~rdtools.filtering.logic_clip_filter` (:pull:`200`).
* Add new function :py:func:`~rdtools.filtering.xgboost_clip_filter` (:pull:`200`).
* Add new function :py:func:`~rdtools.plotting.tune_clip_filter_plot` (:pull:`200`).


Requirements
------------
* Update pinned versions of several dependencies (:pull:`261`, :pull:`275`):

    * ``requirements.txt``: cached-property, certifi, chardet, idna, matplotlib, numpy, Pillow,
      requests, urllib
    * ``docs/notebook_requirements.txt``: argon2-cffi, bleach, cffi, colorama, Jinja2,
      numexpr, packaging, pycparser, pygments


Example Updates
---------------
  

Contributors
------------
* Mark Mikofski (:ghuser:`mikofski`)
* Kevin Anderson (:ghuser:`kanderso-nrel`)
* Michael Deceglie (:ghuser:`mdeceglie`)
* Matthew Muller (:ghuser:`matt14muller`)
* Kirsten Perry (:ghuser:`kperrynrel`)
* Chris Deline (:ghuser:`cdeline`)
