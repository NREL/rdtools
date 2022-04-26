************************
Pending
************************

Enhancements
------------
* Specifying ``detailed=True`` in :py:func:`rdtools.plotting.degradation_summary_plots`
  now shows the number of year-on-year slopes in addition to color coding points
  (:issue:`298`, :pull:`324`)

Testing
-------
* Added a CI notebook check (:pull:`270`)

Requirements
------------
* Bump ``ipython==7.16.3``, ``jupyter-console==6.4.0``,
  and ``prompt-toolkit==3.0.27`` in ``docs/notebook_requirements.txt``
  and bump ``Pillow==9.0.0`` in ``requirements.txt`` (:pull:`314`)
* Bump ``nbsphinx`` version from 0.8.5 to 0.8.8 in the optional
  ``[doc]`` requirements (:pull:`317`)
