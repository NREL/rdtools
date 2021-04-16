************************
Pending
************************

API Changes
-----------
* The calculations internal to the SRR algorithm have changed such that consecutive
  cleaning events are no longer removed. (:pull:`199`, :issue:`189`)

* The default ``day_scale`` parameter in soiling functions and methods was changed
  from 14 to 13. (:pull:`199`, :issue:`189`)

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

