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


Bug fixes
---------
* Unexpected recoveries when using ``method=random_clean`` in the soiling module
  have been fixed. (:pull:`199`, :issue:`234`)


Testing
-------


Documentation
-------------


Documentation
-------------
* Corrected a typo in the :py:class:`~rdtools.analysis_chains.TrendAnalysis`
  docstring (:pull:`264`)

Requirements
------------
* Update specified versions of bleach in
  ``docs/notebook_requirements.txt`` and matplotlib
  in ``requirements.txt`` (:pull:`261`)


Example Updates
---------------
  

Contributors
------------
* Kevin Anderson (:ghuser:`kanderso-nrel`)
* Michael Deceglie (:ghuser:`mdeceglie`)
* Matthew Muller (:ghuser:`matt14muller`)

