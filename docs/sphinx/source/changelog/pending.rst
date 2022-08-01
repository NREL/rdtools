************************
Pending
************************

Enhancements
------------
* Specifying ``detailed=True`` in :py:func:`rdtools.plotting.degradation_summary_plots`
  now shows the number of year-on-year slopes in addition to color coding points
  (:issue:`298`, :pull:`324`)
* The Combined estimation Of Degradation and Soiling (CODS) algorithm is implemented
  in the soiling module and illustrated in an example notebook (:pull:`150`, :pull:`333`)
* Circular block bootstrapping added as a method for calculating uncertainty in
  ``degradation_year_on_year()`` via the ``Uncertainty_method`` argument (:pull:`150`)

Testing
-------
* Added a CI notebook check (:pull:`270`)

Requirements
------------
* Upgrade the notebook environment from python 3.7 to python 3.10.
  Several dependency versions in ``docs/notebook_requirements.txt`` are
  updated as well. (:issue:`319`, :pull:`326`)
* Bump ``ipython==7.16.3``, ``jupyter-console==6.4.0``,
  and ``prompt-toolkit==3.0.27`` in ``docs/notebook_requirements.txt``
  and bump ``Pillow==9.0.0`` in ``requirements.txt`` (:pull:`314`)
* Bump ``sphinx`` version from 3.2 to 4.5 and ``nbsphinx`` version
  from 0.8.5 to 0.8.8 in the optional ``[doc]`` requirements (:pull:`317`, :pull:`325`)
* ``arch`` and ``filterpy`` added as dependencies (:pull:`150`)
