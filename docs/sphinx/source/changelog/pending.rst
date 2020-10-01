************************
Pending
************************

API Changes
-----------
* Change the column names 'slope', 'slope_low', and 'slope_high' to 'soiling_rate', 'soiling_rate_low', and 'soiling_rate_high' in ``calc_info['soiling_interval_summary']`` returned from :py:meth:`~rdtools.soiling.SRRAnalysis.run()` and :py:func:`rdtools.soiling.soiling_srr()` (:pull:`193`).


Enhancements
------------
* Add new function :py:func:`~rdtools.soiling.monthly_soiling_rates` (:pull:`193`).
* Add new function :py:func:`~rdtools.annual_soiling_ratios` (:pull:`193`).


Bug fixes
---------


Testing
-------


Documentation
-------------

Requirements
------------


Example Updates
---------------
* :py:func:`~rdtools.soiling.monthly_soiling_rates` added to degradation_and_soiling_example_pvdaq_4.ipynb
  

Contributors
------------
* Mike Deceglie (:ghuser:`mdeceglie`)
