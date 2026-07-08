***************
Pending Changes
***************

This document tracks changes pending release for the soiling module refactor.
Changes from :pull:`426`, :pull:`429`, :pull:`432`, and :pull:`435` are integrated via :pull:`479`.


Breaking Changes
----------------
* Upgrade soiling algorithms SRR and CODS. Remove experimental warning label. (:pull:`426`, :pull:`479`)


API Changes
-----------
* Renamed SRR soiling parameters for clarity and consistency (:pull:`479`):

  - ``min_interval_length`` -> ``min_interval_days`` in :py:func:`~rdtools.soiling.soiling_srr`
    and :py:func:`~rdtools.soiling.monthly_soiling_rates`
  - ``max_negative_step`` -> ``max_neg_step`` in :py:func:`~rdtools.soiling.SRRAnalysis._calc_result_df`

* Consolidated soiling method names (:pull:`479`):

  - Removed ``method="perfect_clean_complex"`` - use ``method="perfect_clean"``
    with ``detect_neg_shifts=True`` and ``piecewise_fit=True`` instead
  - Removed ``method="inferred_clean_complex"`` - use ``method="inferred_clean"``
    with ``detect_neg_shifts=True`` and ``piecewise_fit=True`` instead

* Renamed SRR parameters for new features (:pull:`479`):

  - ``neg_shift`` -> ``detect_neg_shifts``
  - ``piecewise`` -> ``piecewise_fit``


Enhancements
------------
* Added negative shift detection capability to :py:func:`~rdtools.soiling.soiling_srr`
  via the ``detect_neg_shifts`` parameter. When enabled, detects negative shifts
  in soiling profiles that may indicate partial cleaning events or sensor issues.
  (:pull:`426`, :pull:`435`, :pull:`479`)

* Added piecewise linear fitting capability to :py:func:`~rdtools.soiling.soiling_srr`
  via the ``piecewise_fit`` parameter. Detects changes in soiling rate slope within
  intervals using segmented regression. (:pull:`426`, :pull:`435`, :pull:`479`)

* Added ``collapse_window_days`` parameter to :py:func:`~rdtools.soiling.SRRAnalysis._calc_daily_df`
  to control collapsing of consecutive cleaning events (default: 5 days). (:pull:`479`)

* Added ``forward_median_window`` parameter to :py:func:`~rdtools.soiling.SRRAnalysis._calc_result_df`
  for forward median calculation in shift validation (default: 10). (:pull:`479`)

* Added ``neg_shift_factor`` parameter to control sensitivity of negative shift
  detection (default: 2.5). (:pull:`479`)

* Added ``min_piecewise_days`` parameter to set minimum interval length for
  piecewise fitting attempts (default: 27 days). (:pull:`479`)

* Added ``inferred_recovery`` and ``inferred_begin_shift`` columns to soiling interval
  summary output for tracking inferred cleaning recovery and beginning shift values.
  (:pull:`435`, :pull:`479`)

* Consecutive cleaning events are now always collapsed to single events
  (previously only when ``piecewise=True``). This improves soiling interval
  detection consistency. (:pull:`479`)

* Add capability to seed the CircularBlockBootstrap for reproducible uncertainty
  calculations. (:pull:`429`, :pull:`435`)

* Removed experimental warning label from soiling module. The SRR and CODS
  algorithms are now considered stable. (:pull:`426`, :pull:`435`, :pull:`479`)


Bug Fixes
---------
* Fixed pylint bare except error for :py:func:`~rdtools.soiling.segmented_soiling_period`
  in ``soiling.py`` (:pull:`432`, :pull:`435`)

* Fixed variable shadowing bug in ``CODSAnalysis.iterative_signal_decomposition``
  where a local variable ``degradation`` shadowed the imported ``degradation``
  module, causing ``UnboundLocalError``. Renamed to ``degradation_value``. (:pull:`479`)

* Fixed pandas Copy-on-Write (CoW) compatibility issue in
  :py:func:`~rdtools.soiling.SRRAnalysis._calc_result_df` where
  ``bfill(inplace=True)`` on a chained assignment failed silently with pandas 2.x+.
  Changed to assignment-based approach. This was causing incorrect soiling ratio
  calculations when ``detect_neg_shifts=True``. (:pull:`479`)

* Fixed ``segmented_soiling_period`` regression error where R² calculation was
  attempted on invalid piecewise fits. When a change point was rejected for being
  too close to interval boundaries, the code incorrectly tried to call
  ``linregress`` with all-NaN values, causing "Cannot calculate a linear regression
  if all x values are identical" errors. (:pull:`479`)


Documentation
-------------
* Removed experimental warning docstrings and runtime warnings from soiling
  plotting functions: :py:func:`~rdtools.plotting.soiling_monte_carlo_plot`,
  :py:func:`~rdtools.plotting.soiling_interval_plot`, and
  :py:func:`~rdtools.plotting.soiling_rate_histogram`. (:pull:`435`, :pull:`479`)

* Added new example notebook ``soiling_options_comparison_v2.ipynb`` demonstrating
  the various soiling analysis options including ``detect_neg_shifts`` and
  ``piecewise_fit``. (:pull:`479`)


Testing
-------
* Added new test fixtures for soiling data with negative shifts
  (``soiling_normalized_daily_with_neg_shifts``) and piecewise slope changes
  (``soiling_normalized_daily_with_piecewise_slope``). (:pull:`435`, :pull:`479`)

* Added comprehensive tests for negative shift detection (``test_negative_shifts``),
  piecewise fitting (``test_piecewise``), and combined functionality
  (``test_piecewise_and_neg_shifts``). (:pull:`435`, :pull:`479`)

* Added pytests to cover invalid segmentations for
  :py:func:`~rdtools.soiling.segmented_soiling_period` including tests for
  non-datetime index, no change point found, short periods, and long periods.
  (:pull:`432`, :pull:`435`, :pull:`479`)

* Updated test parameter names to match API changes (``neg_shift`` ->
  ``detect_neg_shifts``, ``piecewise`` -> ``piecewise_fit``, etc.). (:pull:`479`)

* Updated test function names (``test_soiling_srr_min_interval_length`` ->
  ``test_soiling_srr_min_interval_days``, etc.). (:pull:`479`)


Code Quality
------------
* Reformatted ``soiling.py`` with consistent double-quote string formatting
  and improved function signature readability. (:pull:`479`)

* Removed unused ``warnings`` import from ``plotting.py``. (:pull:`479`)


Contributors
------------
* Martin Springer (:ghuser:`martin-springer`)
* Matthew Muller (:ghuser:`mmuller`)
* Noah Moyer (:ghuser:`noromo01`)
* Quyen Nguyen (:ghuser:`qnguyen345`)
