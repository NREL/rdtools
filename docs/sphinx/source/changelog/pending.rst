************************
Pending
************************
These changes are not yet part of a tagged release, but are available in the `development branch on github <https://github.com/NREL/rdtools/tree/development>`_.

API Changes
-----------

* Change the column names 'slope', 'slope_low', and 'slope_high' to 'soiling_rate', 'soiling_rate_low', and 'soiling_rate_high' in ``calc_info['soiling_interval_summary']`` returned from :py:meth:`~rdtools.soiling.SRRAnalysis.run()` and :py:func:`rdtools.soiling.soiling_srr()` (:pull:`193`).


Enhancements
------------

* Add new function :py:func:`~rdtools.soiling.monthly_soiling_rates` (:pull:`193`).
* Add new function :py:func:`~rdtools.annual_soiling_ratios` (:pull:`193`).
* Create new module :py:mod:`~rdtools.availability` and class
  :py:class:`~rdtools.availability.AvailabilityAanlysis` for estimating
  timeseries system availability (:pull:`131`)
* Create new plotting function :py:func:`~rdtools.plotting.availability_summary_plots`
  (:pull:`131`)

Bug fixes
---------


Testing
-------


Documentation
-------------
* Update landing page and add new "Inverter Downtime" documentation page
  based on the availability notebook (:pull:`131`)

Requirements
------------
* notebook_requirements.txt updated (:pull:`209`) 


Example Updates
---------------
* :py:func:`~rdtools.soiling.monthly_soiling_rates` added to degradation_and_soiling_example_pvdaq_4.ipynb
* Add new ``system_availability_example.ipynb`` notebook (:pull:`131`)
  

Contributors
------------
* Kevin Anderson (:ghuser:`kanderso-nrel`)
* Mike Deceglie (:ghuser:`mdeceglie`)
