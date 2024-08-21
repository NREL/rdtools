*******
pending
*******

Breaking changes
------------
These changes have the potential to change answers in existing scripts
when compared with older versions of RdTools

* Use the pvlib method for clear sky detection by default in :py:func:`~rdtools.analysis_chains.TrendAnalysis` (:pull:`412`)

* Change default clipping filter `model` to `logic` (:pull:`425`)

* Turn on the `two_way_window_filter` by default in `TrendAnalysis`
(:pull:`425`)

Enhancements
------------
* Added a new wrapper function for clearsky filters (:pull:`412`)
* Improve test coverage, especially for the newly added filter capabilities (:pull:`413`)
* Added codecov.yml configuration file (:pull:`420`)

Bug fixes
---------
* Fix typos in citation section of the readme file (:issue:`414`, :pull:`421`)
* Fix deploy workflow to pypi (:issue:`416`, :pull:`427`)

Requirements
------------
* Specified versions in ``requirements.txt`` and ``docs/notebook_requirements.txt`` have been updated (:pull:`412`)
* Increase maximum version of pvlib to <0.12 (:pull:`423`)
* Update fonttools version to 4.43.0 in ``requirements.txt`` (:pull:`404`)
* Update jinja2 from 3.0.0 to 3.1.3 in ``notebook_requirements.txt`` (:pull:`405`)
* Update pillow version to 10.3.0 in ``requirements.txt`` (:pull:`410`)
* Update certifi version to 2024.7.4 in ``requirements.txt`` (:pull:`424`)

Deprecations
------------
* Removed  :py:func:`~rdtools.normalization.sapm_dc_power` (:pull:`419`)
* Removed  :py:func:`~rdtools.normalization.normalize_with_sapm` (:pull:`419`)

Contributors
------------
* Martin Springer (:ghuser:`martin-springer`)
* Michael Deceglie (:ghuser:`mdeceglie`)
