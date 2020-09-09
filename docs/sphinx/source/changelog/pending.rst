************************
Pending
************************

API Changes
-----------
* The calculations internal to the SRR algorithm have changed such that consecutive
  cleaning events are no longer removed. (:pull:`199`, :issue:`189`)

* The default ``day_scale`` parameter in soiling funtions and methods was changed
  from 14 to 13. (:pull:`199`, :issue:`189`)

Enhancements
------------
* Add new function :py:func:`~rdtools.soiling.monthly_soiling_rates` (:pull:`193`).
* Add new function :py:func:`~rdtools.annual_soilng_ratios` (:pull:`193`).

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

