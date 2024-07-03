*******
pending
*******

Breaking changes
------------
These changes have the potential to change answers in existing scripts
when compared with older versions of RdTools

* Use the pvlib method for clear sky detection by default in :py:func:`~rdtools.analysis_chains.TrendAnalysis` (:pull:`412`)

Enhancements
------------
* Added a new wrapper function for clearsky filters (:pull:`412`)
* Improve test coverage, especially for the newly added filter capabilities (:pull:`413`)

Bug fixes
---------
* tbd

Requirements
------------
* Specified versions in ``requirements.txt`` and ``docs/notebook_requirements.txt`` have been updated (:pull:`412`)

Deprecations
------------
* Removed  :py:func:`~rdtools.normalization.sapm_dc_power` (:pull:`419`)
* Removed  :py:func:`~rdtools.normalization.normalize_with_sapm` (:pull:`419`)

Contributors
------------
* Martin Springer (:ghuser:`martin-springer`)
* Michael Deceglie (:ghuser:`mdeceglie`)