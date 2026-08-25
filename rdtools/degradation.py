'''Functions for calculating the degradation rate of photovoltaic systems.'''

from collections import namedtuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from rdtools.bootstrap import _make_time_series_bootstrap_samples, \
    _construct_confidence_intervals
from rdtools import utilities


def degradation_ols(energy_normalized, confidence_level=68.2):
    '''
    Estimate the trend of a timeseries using ordinary least-squares regression
    and calculate various statistics including a Monte Carlo-derived confidence
    interval of slope.

    Parameters
    ----------
    energy_normalized: pandas.Series
        Daily or lower frequency time series of normalized system ouput.
    confidence_level: float, default 68.2
        The size of the confidence interval to return, in percent.

    Returns
    -------
    Rd_pct : float
        Estimated degradation relative to the year 0 system capacity [%/year]
    Rd_CI : numpy.array
        The calculated confidence interval bounds.
    calc_info : dict
        A dict that contains slope, intercept,
        root mean square error of regression ('rmse'), standard error
        of the slope ('slope_stderr'), intercept ('intercept_stderr'),
        and least squares RegressionResults object ('ols_results')
    '''

    energy_normalized.name = 'energy_normalized'
    df = energy_normalized.to_frame()

    # calculate a years column as x value for regression, ignoring leap years
    day_diffs = (df.index - df.index[0])
    df['days'] = day_diffs / pd.Timedelta('1d')
    df['years'] = df.days / 365.0

    # add intercept-constant to the exogeneous variable
    df = sm.add_constant(df)

    # perform regression
    ols_model = sm.OLS(endog=df.energy_normalized,
                       exog=df.loc[:, ['const', 'years']],
                       hasconst=True, missing='drop')

    results = ols_model.fit()

    # collect intercept and slope
    b, m = results.params

    # rate of degradation in terms of percent/year
    Rd_pct = 100.0 * m / b

    # Calculate RMSE
    rmse = np.sqrt(results.mse_resid)

    # Collect standrd errors
    stderr_b, stderr_m = results.bse

    # Monte Carlo for error in degradation rate
    Rd_CI = _degradation_CI(results, confidence_level=confidence_level)

    calc_info = {
        'slope': m,
        'intercept': b,
        'rmse': rmse,
        'slope_stderr': stderr_m,
        'intercept_stderr': stderr_b,
        'ols_result': results,
    }

    return (Rd_pct, Rd_CI, calc_info)


def degradation_classical_decomposition(energy_normalized,
                                        confidence_level=68.2):
    '''
    Estimate the trend of a timeseries using a classical decomposition approach
    (moving average) and calculate various statistics, including the result of
    a Mann-Kendall test and a Monte Carlo-derived confidence interval of slope.

    Parameters
    ----------
    energy_normalized: pandas.Series
        Daily or lower frequency time series of normalized system ouput.
        Must be regular time series.
    confidence_level: float, default 68.2
        The size of the confidence interval to return, in percent.

    Returns
    -------
    Rd_pct : float
        Estimated degradation relative to the year 0 system capacity [%/year]
    Rd_CI : numpy.array
        The calculated confidence interval bounds.
    calc_info : dict
        A dict that contains slope, intercept,
        root mean square error of regression ('rmse'), standard error
        of the slope ('slope_stderr'), intercept ('intercept_stderr'),
        and least squares RegressionResults object ('ols_results'),
        pandas series for the annual rolling mean ('series'), and
        Mann-Kendall test trend ('mk_test_trend')
    '''

    energy_normalized.name = 'energy_normalized'
    df = energy_normalized.to_frame()

    df_check_freq = df.copy()

    # The frequency attribute will be set to None if rows are dropped.
    # We can use this to check for missing data and raise a ValueError.
    df_check_freq = df_check_freq.dropna()

    if df_check_freq.index.freq is None:
        raise ValueError('Classical decomposition requires a regular time '
                         'series with defined frequency and no missing data.')

    # calculate a years column as x value for regression, ignoring leap years
    day_diffs = (df.index - df.index[0])
    df['days'] = day_diffs / pd.Timedelta('1d')
    df['years'] = df.days / 365.0

    # Compute yearly rolling mean to isolate trend component using
    # moving average
    energy_ma = df['energy_normalized'].rolling('365d', center=True).mean()
    has_full_year = (df["years"] >= df["years"].iloc[0] + 0.5) & (
        df["years"] <= df["years"].iloc[-1] - 0.5
    )
    energy_ma[~has_full_year] = np.nan
    df['energy_ma'] = energy_ma

    # add intercept-constant to the exogeneous variable
    df = sm.add_constant(df)

    # perform regression
    ols_model = sm.OLS(endog=df.energy_ma, exog=df.loc[:, ['const', 'years']],
                       hasconst=True, missing='drop')

    results = ols_model.fit()

    # collect intercept and slope
    b, m = results.params

    # rate of degradation in terms of percent/year
    Rd_pct = 100.0 * m / b

    # Calculate RMSE
    rmse = np.sqrt(results.mse_resid)

    # Collect standrd errors
    stderr_b, stderr_m = results.bse

    # Perform Mann-Kendall
    test_trend, h, p, z = _mk_test(df.energy_ma.dropna(), alpha=0.05)

    # Monte Carlo for error in degradation rate
    Rd_CI = _degradation_CI(results, confidence_level=confidence_level)

    calc_info = {
        'slope': m,
        'intercept': b,
        'rmse': rmse,
        'slope_stderr': stderr_m,
        'intercept_stderr': stderr_b,
        'ols_result': results,
        'series': df.energy_ma,
        'mk_test_trend': test_trend
    }

    return (Rd_pct, Rd_CI, calc_info)


def degradation_theil_sen(energy_normalized, confidence_level=68.2):
    '''
    Estimate the trend of a timeseries using the Theil-Sen estimator -- a
    robust non-parametric regression that takes the median of the slopes
    between all pairs of observations.

    Compared to :py:func:`degradation_ols`, Theil-Sen is less sensitive to
    outliers and makes no distributional assumption about residuals.  Useful
    when the input contains a handful of bad days that would dominate an
    OLS fit. The pairwise-slope calculation is roughly :math:`O(n^2)`, so
    very-high-frequency inputs (e.g. minute-level) are not recommended;
    aggregate to daily or lower first.

    The confidence interval is derived from the rank statistics of the
    pairwise slopes (no Monte Carlo or bootstrap), and is converted to
    %/year of the year-0 capacity by dividing by the Theil-Sen intercept.

    Parameters
    ----------
    energy_normalized : pandas.Series
        Daily or lower frequency time series of normalized system output.
    confidence_level : float, default 68.2
        The size of the confidence interval to return, in percent.

    Returns
    -------
    Rd_pct : float
        Estimated degradation relative to the year-0 system capacity [%/year]
    Rd_CI : numpy.array
        The calculated confidence interval bounds (length 2).
    calc_info : dict
        A dict that contains:

        * ``slope`` - estimated slope (median of pairwise slopes), in units
          of normalized energy per year.
        * ``intercept`` - intercept of the median-slope line, in normalized
          energy units (the implied year-0 capacity).
        * ``slope_low`` / ``slope_high`` - lower / upper bound of the
          rank-based confidence interval on ``slope``.
        * ``theilslopes_result`` - the raw
          :py:func:`scipy.stats.theilslopes` return.
    '''
    from scipy.stats import theilslopes

    series = energy_normalized.dropna()
    if series.shape[0] < 2:
        raise ValueError('Theil-Sen estimator requires at least 2 non-NaN '
                         'observations')

    years = (series.index - series.index[0]) / pd.Timedelta('365D')
    alpha = confidence_level / 100.0
    result = theilslopes(series.values, years.values, alpha=alpha)

    # scipy returns either a plain (slope, intercept, lo_slope, up_slope)
    # tuple or a TheilslopesResult namedtuple; both unpack positionally.
    slope, intercept, lo_slope, up_slope = result

    Rd_pct = 100.0 * slope / intercept
    Rd_CI = 100.0 * np.array([lo_slope, up_slope]) / intercept

    calc_info = {
        'slope': slope,
        'intercept': intercept,
        'slope_low': lo_slope,
        'slope_high': up_slope,
        'theilslopes_result': result,
    }

    return (Rd_pct, Rd_CI, calc_info)


def degradation_fourier(energy_normalized,
                        skip_year1_in_seasonal_fit=True,
                        seasonal_period_days=365.25, harmonics=1,
                        seasonal_trend_method='yoy',
                        slope_method='theil_sen',
                        confidence_level=68.2):
    '''
    Estimate the year-1 degradation rate of a seasonal time series using a
    two-stage seasonal-Fourier regression.

    Stage 1 estimates the seasonal Fourier coefficients from
    ``energy_normalized`` (optionally excluding the first 365 days) using
    either a joint OLS fit of trend + Fourier regressors
    (``seasonal_trend_method='ols'``) or a robust two-step
    YoY-detrend-then-Fourier fit (``seasonal_trend_method='yoy'``, the
    default). Stage 2 subtracts the resulting seasonal pattern from the
    year-1 window (the first 365 days of ``energy_normalized``) and fits a
    simple linear regression (Theil-Sen by default, or OLS, selectable via
    ``slope_method``) to the deseasonalized residual to extract the slope.

    Estimating seasonality from the full multi-year series, then fitting
    the slope only on year 1, lets the year-1 trend be inferred without
    contamination from a poorly-constrained seasonal estimate -- which is
    the dominant source of bias when fitting a one-year window directly,
    since the linear trend and seasonal Fourier regressors are not
    orthogonal on a single annual cycle.

    Parameters
    ----------
    energy_normalized : pandas.Series
        Daily or lower frequency time series of normalized system output.
        Must span at least 365 days. When
        ``skip_year1_in_seasonal_fit=True`` (the default), must also extend
        past the first 365 days so the seasonal fit has data to work with;
        the exact requirement depends on ``seasonal_trend_method`` (roughly
        one additional year for ``'ols'`` and two additional years for
        ``'yoy'``, since the latter needs two full cycles to form YoY pairs).
    skip_year1_in_seasonal_fit : bool, default True
        If ``True`` (default), exclude the first 365 days of
        ``energy_normalized`` from the stage-1 seasonal fit. This is the
        safer default because the reason to reach for a two-stage
        estimator in the first place is typically that year 1 is
        suspected of an unusual transient (LID, light-soaking, initial
        stabilization); allowing year 1 into the seasonal fit risks
        absorbing that anomaly into the harmonic coefficients and biasing
        the recovered year-1 slope. Set ``False`` when year 1 is expected
        to behave like steady state and you cannot afford to lose a year
        of data from the stage-1 fit.
    seasonal_period_days : float, default 365.25
        Fundamental period of the seasonal harmonics, in days.
    harmonics : int, default 1
        Number of harmonic pairs to include. ``harmonics=1`` adds the
        annual sine and cosine; ``harmonics=2`` also adds the semi-annual
        pair, etc.
    seasonal_trend_method : {'yoy', 'ols'}, default 'yoy'
        How stage 1 handles the linear trend that must be removed from
        the multi-year data before (or while) fitting the Fourier
        coefficients:

        * ``'yoy'`` (default) -- first estimate the multi-year trend
          robustly with :py:func:`degradation_year_on_year` (the median
          of pairwise annual differences, which is immune to a small
          fraction of bad days and by construction unaffected by any
          stationary 365-day-periodic seasonality). Subtract that trend
          from the series, then fit only intercept + Fourier regressors
          via OLS to the detrended residual. Requires the stage-1 window
          to span at least two full years so YoY can form pairs;
          combined with ``skip_year1_in_seasonal_fit=True`` this means
          the input must span at least three full years.
        * ``'ols'`` -- jointly fit intercept, linear trend, and the
          ``2*harmonics`` sine/cosine regressors in a single OLS call.
          Faster and works on shorter records (only ``2 * harmonics + 2``
          samples needed at stage 1), but sensitive to outliers in the
          multi-year record; recommended when you already trust the
          upstream filtering.

        Stage-1 fitting of the Fourier coefficients themselves is always
        OLS: the multi-regressor sine/cosine design has no clean robust
        univariate analog, and stage 1 sees enough data (thousands of
        points across many cycles) that individual outliers on the
        harmonic coefficients wash out.
    slope_method : {'theil_sen', 'ols'}, default 'theil_sen'
        Regression used in stage 2 to fit the slope of the deseasonalized
        year-1 window. ``'theil_sen'`` (default) uses
        :py:func:`degradation_theil_sen`, a robust median-of-pairwise-
        slopes estimator that is resistant to a handful of bad days in
        the year-1 window (rank-based CI, no bootstrap). ``'ols'`` uses
        :py:func:`degradation_ols`-style OLS with parametric Monte-Carlo
        CIs; use it when the year-1 window is known to be clean and you
        want a slightly narrower interval under Gaussian residuals.
    confidence_level : float, default 68.2
        The size of the confidence interval to return, in percent.

    Returns
    -------
    Rd_pct : float
        Estimated year-1 degradation relative to the year-0 system
        capacity [%/year].
    Rd_CI : numpy.array
        The calculated confidence interval bounds (length 2). For
        ``slope_method='ols'`` this is derived from the stage-2 OLS
        slope's standard error via Monte Carlo, mirroring
        :py:func:`degradation_ols`. For ``slope_method='theil_sen'``
        it comes from the rank-based CI on the Theil-Sen pairwise
        slopes. Neither propagates the stage-1 seasonal-coefficient
        uncertainty -- a typically small contribution when
        ``energy_normalized`` spans several seasonal cycles.
    calc_info : dict
        A dict that always contains:

        * ``slope`` / ``intercept`` - stage-2 slope (normalized energy
          per year) and intercept (year-0 capacity).
        * ``seasonal_coeffs`` - the fitted sine/cosine coefficients as a
          length-``2*harmonics`` numpy array, ordered
          ``(sin_1, cos_1, sin_2, cos_2, ...)``.
        * ``seasonal_period_days`` / ``harmonics`` - echo of the inputs.
        * ``seasonal_ols_result`` - the stage-1 ``statsmodels``
          ``RegressionResults`` (for ``seasonal_trend_method='ols'`` this
          holds the joint trend+Fourier fit; for ``'yoy'`` it holds the
          intercept+Fourier fit on the YoY-detrended residual).
        * ``skip_year1_in_seasonal_fit`` - echo of the input flag.
        * ``year1_end`` - the ``pandas.Timestamp`` marking the end of the
          year-1 window (start + 365 days).
        * ``slope_method`` - echo of the ``slope_method`` argument.
        * ``seasonal_trend_method`` - echo of the
          ``seasonal_trend_method`` argument.
        * ``seasonal_trend_slope_per_day`` - the linear trend slope (in
          input units per day) used by stage 1 for detrending. For
          ``'ols'`` this is the coefficient of the trend column in the
          joint fit; for ``'yoy'`` it is derived from the YoY point
          estimate.
        * ``yoy_stage1_rd_pct`` - the raw ``Rd_pct`` returned by
          :py:func:`degradation_year_on_year` at stage 1 (``None`` when
          ``seasonal_trend_method='ols'``).

        When ``slope_method='ols'`` it additionally contains
        ``rmse``, ``slope_stderr``, ``intercept_stderr``, and
        ``ols_result`` (the stage-2 ``statsmodels`` results).
        When ``slope_method='theil_sen'`` it additionally contains
        ``slope_low``, ``slope_high`` (rank-based CI on the slope) and
        ``theilslopes_result`` (the raw scipy return).
    '''
    if slope_method not in ('ols', 'theil_sen'):
        raise ValueError(
            f"unknown slope_method '{slope_method}'; "
            "expected 'ols' or 'theil_sen'"
        )
    if seasonal_trend_method not in ('ols', 'yoy'):
        raise ValueError(
            f"unknown seasonal_trend_method '{seasonal_trend_method}'; "
            "expected 'ols' or 'yoy'"
        )

    energy_normalized = energy_normalized.dropna().sort_index()
    if energy_normalized.shape[0] < 2:
        raise ValueError(
            'degradation_fourier requires at least 2 non-NaN '
            'observations in energy_normalized'
        )

    if harmonics < 1:
        raise ValueError('harmonics must be >= 1')

    start = energy_normalized.index.min()
    year1_end = start + pd.Timedelta(days=365.0)
    if energy_normalized.index.max() < year1_end:
        raise ValueError(
            'energy_normalized must span at least 365 days to define a '
            'year-1 window'
        )

    year1 = energy_normalized.loc[start:year1_end]

    # Stage 1 seasonal-fit input.
    if skip_year1_in_seasonal_fit:
        seasonal_input = energy_normalized.loc[year1_end:]
        # Exclude the boundary point so it isn't shared with the year-1 slice
        # (purely cosmetic for the stage-1 fit, but keeps the partition clean).
        seasonal_input = seasonal_input.iloc[1:] if (
            len(seasonal_input) > 0
            and seasonal_input.index[0] == year1_end
        ) else seasonal_input
        if seasonal_input.shape[0] < 2 * harmonics + 2:
            raise ValueError(
                'after skipping the first 365 days, energy_normalized has '
                'too few samples to constrain the seasonal fit '
                f'(need at least {2 * harmonics + 2}, '
                f'got {seasonal_input.shape[0]}); '
                'pass skip_year1_in_seasonal_fit=False or supply a longer series'
            )
    else:
        seasonal_input = energy_normalized

    # Stage 1: fit the seasonal Fourier coefficients. The way the linear
    # trend is handled depends on seasonal_trend_method (see docstring).
    epoch = seasonal_input.index.min()
    days_s = (seasonal_input.index - epoch) / pd.Timedelta('1d')
    omega = 2.0 * np.pi / seasonal_period_days

    fourier_cols = []
    for k in range(1, harmonics + 1):
        fourier_cols.append(np.sin(k * omega * days_s))
        fourier_cols.append(np.cos(k * omega * days_s))

    if seasonal_trend_method == 'ols':
        # Joint OLS fit of [intercept, days_trend, sin, cos, ...].
        seasonal_cols = (
            [np.ones_like(days_s, dtype=float), days_s.to_numpy()]
            + fourier_cols
        )
        X_seasonal = np.column_stack(seasonal_cols)
        seasonal_model = sm.OLS(seasonal_input.to_numpy(), X_seasonal).fit()
        # Fourier coefficients follow the intercept and trend columns.
        seasonal_coeffs = np.asarray(seasonal_model.params[2:])
        seasonal_trend_slope_per_day = float(seasonal_model.params[1])
        yoy_stage1_rd_pct = None
    else:  # 'yoy'
        # Estimate the trend robustly with YoY (median of pairwise annual
        # slopes), then fit only intercept + Fourier to the detrended
        # residual. Calling degradation_year_on_year with recenter=False
        # keeps the input scale intact so the returned Rd_pct equals
        # 100 * (slope in input units per year); uncertainty_method=None
        # skips the bootstrap since we only need the point estimate.
        #
        # Re-infer the freq attribute on the sliced index if we can, so
        # YoY's ``at least two years'' pre-check uses a proper DateOffset
        # step instead of falling back to Timedelta(diff().median()),
        # which mis-fires by ~1 day on regular monthly/weekly inputs.
        yoy_input = seasonal_input.copy()
        try:
            inferred_freq = pd.infer_freq(yoy_input.index)
        except (TypeError, ValueError):
            inferred_freq = None
        if inferred_freq is not None:
            try:
                yoy_input.index = pd.DatetimeIndex(
                    yoy_input.index, freq=inferred_freq
                )
            except (TypeError, ValueError):
                pass
        yoy_stage1_rd_pct = float(degradation_year_on_year(
            yoy_input, recenter=False, uncertainty_method=None,
        ))
        seasonal_trend_slope_per_day = (
            yoy_stage1_rd_pct / 100.0 / 365.25
        )
        detrended = (
            seasonal_input.to_numpy()
            - seasonal_trend_slope_per_day * days_s.to_numpy()
        )
        seasonal_cols = [np.ones_like(days_s, dtype=float)] + fourier_cols
        X_seasonal = np.column_stack(seasonal_cols)
        seasonal_model = sm.OLS(detrended, X_seasonal).fit()
        # Fourier coefficients follow the intercept only (no trend column).
        seasonal_coeffs = np.asarray(seasonal_model.params[1:])

    # Subtract the stage-1 seasonal pattern from the year-1 slice.
    days_y1_seasonal = (year1.index - epoch) / pd.Timedelta('1d')
    seasonal_y1 = np.zeros(len(year1), dtype=float)
    for k in range(1, harmonics + 1):
        c_k = seasonal_coeffs[2 * (k - 1)]
        d_k = seasonal_coeffs[2 * (k - 1) + 1]
        seasonal_y1 += (c_k * np.sin(k * omega * days_y1_seasonal)
                        + d_k * np.cos(k * omega * days_y1_seasonal))
    deseasonalized = pd.Series(year1.to_numpy() - seasonal_y1,
                               index=year1.index)

    # Stage 2: fit the slope of the deseasonalized year-1 slice using the
    # selected method. The slice is anchored at the first year-1 timestamp
    # so the intercept (and Rd_pct) match the convention used by
    # degradation_ols / degradation_theil_sen.
    if slope_method == 'ols':
        years_y1 = (year1.index - start) / pd.Timedelta('365D')
        X_trend = sm.add_constant(years_y1.to_numpy())
        trend_model = sm.OLS(deseasonalized.to_numpy(), X_trend).fit()
        intercept, slope = trend_model.params
        rmse = np.sqrt(trend_model.mse_resid)
        stderr_b, stderr_m = trend_model.bse
        Rd_pct = 100.0 * slope / intercept
        Rd_CI = _degradation_CI(trend_model, confidence_level=confidence_level)
        slope_info = {
            'slope': slope,
            'intercept': intercept,
            'rmse': rmse,
            'slope_stderr': stderr_m,
            'intercept_stderr': stderr_b,
            'ols_result': trend_model,
        }
    else:  # 'theil_sen'
        Rd_pct, Rd_CI, slope_info = degradation_theil_sen(
            deseasonalized, confidence_level=confidence_level)

    calc_info = {
        **slope_info,
        'seasonal_coeffs': seasonal_coeffs,
        'seasonal_period_days': seasonal_period_days,
        'harmonics': harmonics,
        'seasonal_ols_result': seasonal_model,
        'skip_year1_in_seasonal_fit': skip_year1_in_seasonal_fit,
        'year1_end': year1_end,
        'slope_method': slope_method,
        'seasonal_trend_method': seasonal_trend_method,
        'seasonal_trend_slope_per_day': seasonal_trend_slope_per_day,
        'yoy_stage1_rd_pct': yoy_stage1_rd_pct,
    }

    return (Rd_pct, Rd_CI, calc_info)


def degradation_year_on_year(energy_normalized, recenter=True,
                             exceedance_prob=95, confidence_level=68.2,
                             uncertainty_method='simple', block_length=30,
                             multi_yoy=False):
    '''
    Estimate the trend of a timeseries using the year-on-year decomposition
    approach and calculate a Monte Carlo-derived confidence interval of slope.

    Parameters
    ----------
    energy_normalized: pandas.Series
        Daily or lower frequency time series of normalized system ouput.
    recenter : bool, default True
        Specify whether data is internally recentered to normalized yield
        of 1 based on first year median. If False, ``Rd_pct`` is calculated
        assuming ``energy_normalized`` is passed already normalized to the
        year 0 system capacity.
    exceedance_prob : float, default 95
        The probability level to use for exceedance value calculation,
        in percent.
    confidence_level : float, default 68.2
        The size of the confidence interval to return, in percent.
    uncertainty_method : string, default 'simple'
        Either 'simple', 'circular_block', or None
        Determines what bootstrapping method to use to construct confidence
        intervals and exceedance levels. If None (or anything other than the three
        alternatives), the algorithm does not construct confidence intervals,
        is considerably faster, and only returns the `Rd_pct`.
    block_length : int, default 30
        If `uncertainty_method` is 'circular_block', `block_length`
        determines the length of the blocks used in the circular block bootstrapping
        in number of days. Must be shorter than a third of the time series.
    multi_yoy : bool, default False
        Whether to return the standard Year-on-Year slopes where each slope
        is calculated over points separated by 365 days (default) or
        multi_year-on-year where points can be separated by N * 365 days
        where N is an integer from 1 to the length of the dataset in years.

    Returns
    -------
    Rd_pct : float
        Estimated degradation relative to the year 0 median system capacity [%/year]
    confidence_interval : numpy.array
        confidence interval (size specified by ``confidence_level``) of
        degradation rate estimate
    calc_info : dict

        * `YoY_values` - pandas series of year on year slopes with integer index.
          When ``multi_yoy=True`` the index is non-monotonic because multiple
          overlapping annual slopes can share the same right-endpoint position.
        * `renormalizing_factor` - float of value used to recenter data
        * `exceedance_level` - the degradation rate that was outperformed with
          probability of `exceedance_prob`
        * `usage_of_points` - number of times each point in energy_normalized
          is used to calculate a degradation slope. 0: point is never used. 1:
          point is either used as a start or endpoint. 2: point is used as both
          start and endpoint for an Rd calculation. With ``multi_yoy=True``,
          values can be larger than 2 because each point participates in
          multiple slopes.
        * `YoY_times` - pandas DataFrame with columns ``dt_right``, ``dt_center``,
          and ``dt_left`` giving, for each entry in ``YoY_values``, the
          timestamps of the right endpoint, the midpoint, and the left endpoint
          of the slope. This can be used to recover the original timestamp-
          indexed behavior of ``YoY_values`` (for example,
          ``calc_info['YoY_values'].set_axis(calc_info['YoY_times']['dt_right'])``).
    '''

    # Ensure the data is in order
    energy_normalized = energy_normalized.sort_index()
    energy_normalized.name = 'energy'
    energy_normalized.index.name = 'dt'

    # Detect less than 2 years of data. This is complicated by two things:
    #   - leap days muddle the precise meaning of "two years of data".
    #   - can't just check the number of days between the first and last
    #     index values, since non-daily (e.g. weekly) inputs span
    #     a longer period than their index values directly indicate.
    # See the unit tests for several motivating cases.
    if energy_normalized.index.inferred_freq is not None:
        step = pd.tseries.frequencies.to_offset(energy_normalized.index.inferred_freq)
    else:
        step = energy_normalized.index.to_series().diff().median()

    if energy_normalized.index[-1] < energy_normalized.index[0] + pd.DateOffset(years=2) - step:
        raise ValueError('must provide at least two years of normalized energy')

    # If circular block bootstrapping...
    if uncertainty_method == 'circular_block':
        # ... require regular logging frequency
        freq = pd.infer_freq(energy_normalized.index)
        if isinstance(freq, type(None)):
            raise ValueError('energy_normalized must have a fixed frequency')
        # ... require a block length shorter than a third of the time series
        if block_length > (len(energy_normalized) / 3):
            raise ValueError(
                'block_length must must be shorter than a third of the time series')

    # Auto center
    if recenter:
        start = energy_normalized.index[0]
        oneyear = start + pd.Timedelta('364D')
        renorm = utilities.robust_median(energy_normalized[start:oneyear])
    else:
        renorm = 1.0

    energy_normalized = energy_normalized.reset_index()
    energy_normalized['energy'] = energy_normalized['energy'] / renorm

    # dataframe container for combined year-over-year changes
    df = pd.DataFrame()
    if multi_yoy:
        year_range = range(1, int((energy_normalized.iloc[-1]['dt'] -
                                   energy_normalized.iloc[0]['dt']).days/365)+1)
    else:
        year_range = [1]
    for y in year_range:
        energy_normalized['dt_shifted'] = energy_normalized.dt + pd.DateOffset(years=y)
        # Merge with what happened one year ago, use tolerance of 8 days to allow
        # for weekly aggregated data
        df_temp = pd.merge_asof(energy_normalized[['dt', 'energy']],
                                energy_normalized.sort_values('dt_shifted'),
                                left_on='dt', right_on='dt_shifted',
                                suffixes=['', '_left'],
                                tolerance=pd.Timedelta('8D')
                                )
        df = pd.concat([df, df_temp], ignore_index=True)

    df['time_diff_years'] = (df.dt - df.dt_left) / pd.Timedelta('365D')
    df['yoy'] = 100.0 * (df.energy - df.energy_left) / (df.time_diff_years)

    yoy_result = df.yoy.dropna()

    if not len(yoy_result):
        raise ValueError('no year-over-year aggregated data pairs found')

    Rd_pct = yoy_result.median()

    YoY_times = df.dropna(subset=['yoy'], inplace=False).copy()

    # calculate usage of points.
    df_left = YoY_times.set_index(YoY_times.dt_left)  # .drop_duplicates('dt_left')
    df_right = YoY_times.set_index(YoY_times.dt)  # .drop_duplicates('dt')
    usage_of_points = df_right.yoy.notnull().astype(int).add(
                df_left.yoy.notnull().astype(int),
                fill_value=0).groupby(level=0).sum()
    usage_of_points.name = 'usage_of_points'

    pandas_version = pd.__version__.split(".")
    if int(pandas_version[0]) < 2:
        # For old Pandas versions < 2.0.0, time columns cannot be averaged
        # with each other, so we use a custom function to calculate center label
        YoY_times['dt_center'] = _avg_timestamp_old_Pandas(YoY_times['dt'], YoY_times['dt_left'])
    else:
        YoY_times['dt_center'] = pd.to_datetime(YoY_times[['dt', 'dt_left']].mean(axis=1))

    YoY_times = YoY_times[['dt', 'dt_center', 'dt_left']]
    YoY_times = YoY_times.rename(columns={'dt': 'dt_right'})

    # apply integer index to the yoy_result; multi-YoY has duplicate timestamps.
    yoy_result.index = YoY_times.index
    yoy_result.index.name = 'dt'

    # the following is throwing a futurewarning if infer_objects() isn't included here.
    # see https://github.com/pandas-dev/pandas/issues/57734
    energy_normalized = energy_normalized.merge(usage_of_points, how='left', left_on='dt',
                                                right_index=True, left_index=False
                                                ).infer_objects().fillna(0.0)

    if uncertainty_method == 'simple':  # If we need the full results
        calc_info = {
            'YoY_values': yoy_result,
            'renormalizing_factor': renorm,
            'usage_of_points': energy_normalized.set_index('dt')['usage_of_points'],
            'YoY_times': YoY_times[['dt_right', 'dt_center', 'dt_left']]
        }

        # bootstrap to determine 68% CI and exceedance probability
        n1 = len(yoy_result)
        reps = 10000
        xb1 = np.random.choice(yoy_result, (n1, reps), replace=True)
        mb1 = np.median(xb1, axis=0)

        half_ci = confidence_level / 2.0
        Rd_CI = np.percentile(mb1, [50.0 - half_ci, 50.0 + half_ci])

        P_level = np.percentile(mb1, 100.0 - exceedance_prob)

        calc_info['exceedance_level'] = P_level

        return (Rd_pct, Rd_CI, calc_info)

    elif uncertainty_method == 'circular_block':
        # Number of bootstrap repetitions
        reps = 1000

        # Construct degradation trend time series
        N = len(energy_normalized)
        numeric_index = np.arange(N)
        days_per_index = \
            (energy_normalized.dt.iloc[-1] - energy_normalized.dt.iloc[0]).days / N
        degradation_trend = 1 + (Rd_pct / 100 / 365.0 * numeric_index
                                 * days_per_index)
        degradation_trend = pd.Series(
            index=energy_normalized.dt, data=degradation_trend)

        # Generate bootstrap_samples
        bootstrap_samples = _make_time_series_bootstrap_samples(
            energy_normalized.set_index('dt')['energy'], degradation_trend,
            sample_nr=reps, block_length=block_length)

        # Construct confidence interval
        Rd_CI, exceedance_level, bootstrap_rates = \
            _construct_confidence_intervals(
                bootstrap_samples, degradation_year_on_year,
                exceedance_prob=exceedance_prob, confidence_level=confidence_level,
                recenter=False, uncertainty_method='none')

        # Save calculation information
        calc_info = {
            'YoY_values': yoy_result,
            'renormalizing_factor': renorm,
            'exceedance_level': exceedance_level,
            'usage_of_points': energy_normalized.set_index('dt')['usage_of_points'],
            'YoY_times': YoY_times[['dt_right', 'dt_center', 'dt_left']],
            'bootstrap_rates': bootstrap_rates}

        return (Rd_pct, Rd_CI, calc_info)

    else:  # If we do not need confidence intervals and exceedance level
        # TODO: Consider returning a tuple for consistency with other branches, e.g.:
        # return (Rd_pct, None, {
        #     'YoY_values': yoy_result,
        #     'usage_of_points': energy_normalized.set_index('dt')['usage_of_points'],
        #     'YoY_times': YoY_times[['dt_right', 'dt_center', 'dt_left']]}
        # )
        # Note: Current behavior intentionally returns only the scalar Rd_pct
        # to maintain compatibility (see test_bootstrap_module).
        return Rd_pct


# Registry of built-in year-1 regression methods recognised by
# :py:func:`degradation_hybrid` when ``year1_method`` is a string.
# Each entry pairs the underlying function with:
#
# * ``input_kind`` -- ``'year1'`` if the function operates on the year-1
#   slice (first 365 days) carved out by :py:func:`degradation_hybrid`;
#   ``'full'`` if it consumes the whole ``energy_normalized`` series
#   internally (e.g. :py:func:`degradation_fourier` needs the years past
#   year 1 for the seasonal-Fourier fit).
# * ``min_years`` -- integer minimum input span, in years, that this
#   year-1 method needs to succeed on its own.
#   :py:func:`degradation_hybrid` combines it via ``max`` with its own
#   years-2+ requirement (3 years total: 1 for year 1 + 2 for the YoY
#   call on the post-year-1 window) to produce a single up-front
#   data-length check.
#
# User-supplied callables passed via ``year1_method`` are treated as
# ``'year1'``-kind with ``min_years=1``; wrap them in a closure that
# captures the full series if you need ``'full'``-kind semantics.
_Year1Method = namedtuple('_Year1Method', ['func', 'input_kind', 'min_years'])

_YEAR1_METHODS = {
    'ols':       _Year1Method(degradation_ols,       'year1', 1),
    'theil_sen': _Year1Method(degradation_theil_sen, 'year1', 1),
    'fourier':   _Year1Method(degradation_fourier,   'full',  3),
}


def degradation_hybrid(energy_normalized,
                       year1_method='ols', year1_kwargs=None,
                       recenter_year2=True, confidence_level=68.2,
                       yoy_kwargs=None):
    '''
    Estimate a two-piece (nonlinear) degradation profile by fitting a
    user-selected regression method on the first 365 days and the
    year-on-year method on the remainder. This is useful when early-life
    behavior (e.g. light-induced degradation, light-soaking, initial
    stabilization) differs qualitatively from steady-state degradation and
    a single rate would mask that nonlinearity.

    The year-1 rate is reported as %/year of the year-0 system capacity
    (from the year-1 fit). The years-2+ rate is reported as %/year of the
    capacity at the start of year 2: with ``recenter_year2=True`` the year-on-
    year call recenters its input to the median of its first 365 days, which
    for the post-split window is the start-of-year-2 baseline.

    Despite the name (kept for backward compatibility with earlier internal
    APIs), the year-1 piece is no longer restricted to OLS -- see
    ``year1_method`` below.

    Parameters
    ----------
    energy_normalized : pandas.Series
        Daily or lower frequency time series of normalized system output.
        Must span at least 3 years to populate both pieces (1 year for the
        year-1 fit + 2 years for the years-2+ YoY call).
    year1_method : str or callable, default 'ols'
        The regression method used on the first 365 days.
        Built-in string choices:

        * ``'ols'`` (default) - :py:func:`degradation_ols`.
        * ``'theil_sen'`` - :py:func:`degradation_theil_sen`. Robust to
          outliers in the year-1 window via a non-parametric
          median-of-pairwise-slopes fit; cost grows as :math:`O(n^2)`, so
          daily-or-lower aggregation is recommended.
        * ``'fourier'`` - :py:func:`degradation_fourier`. Two-stage
          seasonal-Fourier estimator that fits a linear trend + Fourier
          seasonal regressors on the multi-year window (defaulting to a
          robust YoY-detrend at stage 1 and Theil-Sen at stage 2). This
          entry consumes the full ``energy_normalized`` series internally
          (so the seasonal coefficients are constrained on every cycle the
          input contains, not just the year-1 window), and reports the
          year-1 slope from the deseasonalized first 365 days.

        Alternatively, pass any callable with the signature
        ``f(energy_normalized, confidence_level=..., **year1_kwargs)
        -> (Rd_pct, Rd_CI, calc_info)``. The returned ``calc_info`` dict
        should contain ``'slope'`` and ``'intercept'`` to remain compatible
        with :py:func:`rdtools.plotting.hybrid_degradation_summary_plots`.
        (:py:func:`degradation_classical_decomposition` can be used this
        way -- see the example notebook for an interpolating wrapper.)
    year1_kwargs : dict, optional
        Extra keyword arguments forwarded to the year-1 method.
        ``confidence_level`` is set by this function and must not be supplied
        here.
    recenter_year2 : bool, default True
        Whether the year-on-year call should recenter the post-split window
        to its first-year median (recommended). If False, ``energy_normalized``
        is assumed to be already normalized to the year 0 capacity and the
        years-2+ rate is reported on that baseline.
    confidence_level : float, default 68.2
        The size of the confidence interval to return, in percent. Passed to
        both underlying methods.
    yoy_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :py:func:`degradation_year_on_year` (e.g. ``uncertainty_method``,
        ``multi_yoy``). ``recenter`` and ``confidence_level`` are set by this
        function and should not be supplied here.

    Returns
    -------
    Rd_pct_year1 : float
        Estimated year-1 degradation, %/year of year-0 capacity.
    Rd_pct_years2plus : float
        Estimated steady-state degradation, %/year of capacity at the start
        of year 2 (when ``recenter_year2=True``).
    calc_info : dict
        Detailed results with keys:

        * ``year1`` - full ``(Rd_pct, Rd_CI, calc_info)`` tuple returned by
          the selected ``year1_method`` on the year-1 window.
        * ``years2plus`` - full ``(Rd_pct, Rd_CI, calc_info)`` tuple returned
          by :py:func:`degradation_year_on_year` on the years-2+ window.
        * ``year1_method`` - the resolved year-1 callable.
        * ``split_date`` - ``pandas.Timestamp`` at which the two windows meet.
        * ``renormalizing_factor_year2`` - the median used to recenter the
          year-2+ window (``1.0`` when ``recenter_year2=False``).
    '''

    # Resolve the year-1 method (string registry or user-supplied callable).
    if isinstance(year1_method, str):
        try:
            entry = _YEAR1_METHODS[year1_method]
        except KeyError:
            raise ValueError(
                f"unknown year1_method '{year1_method}'; expected one of "
                f"{sorted(_YEAR1_METHODS)} or a callable"
            )
        year1_func = entry.func
        year1_input_kind = entry.input_kind
        method_min_years = entry.min_years
    elif callable(year1_method):
        year1_func = year1_method
        year1_input_kind = 'year1'
        method_min_years = 1
    else:
        raise TypeError(
            "year1_method must be a string or callable, "
            f"got {type(year1_method).__name__}"
        )

    if year1_kwargs is None:
        year1_kwargs = {}
    if 'confidence_level' in year1_kwargs:
        raise ValueError(
            "'confidence_level' is controlled by degradation_hybrid "
            "and cannot be passed via year1_kwargs"
        )

    if yoy_kwargs is None:
        yoy_kwargs = {}
    for reserved in ('recenter', 'confidence_level'):
        if reserved in yoy_kwargs:
            raise ValueError(
                f"'{reserved}' is controlled by degradation_hybrid "
                "and cannot be passed via yoy_kwargs"
            )

    energy_normalized = energy_normalized.sort_index()

    # Up-front data-length check. The hybrid framework itself needs
    # 3 years (1 year for the year-1 fit + 2 years for the years-2+ YoY
    # call); some methods declare a stricter requirement via
    # _Year1Method.min_years. A small ~7-day tolerance (0.02 years)
    # absorbs the fractional-day slack that weekly/monthly grids naturally
    # have at the trailing edge.
    if energy_normalized.shape[0] > 0:
        required_years = max(3.0, float(method_min_years))
        input_span_years = (
            (energy_normalized.index[-1] - energy_normalized.index[0])
            .total_seconds() / (365.25 * 86400.0)
        )
        if input_span_years < required_years - 0.02:
            method_label = (
                year1_method if isinstance(year1_method, str)
                else getattr(year1_method, '__name__', 'callable')
            )
            raise ValueError(
                f"hybrid analysis with year1_method={method_label!r} "
                f"requires at least {required_years:.2f} years of data "
                f"(got {input_span_years:.2f}); supply more data or "
                "choose a less data-hungry year-1 method"
            )

    start = energy_normalized.index[0]
    split = start + pd.Timedelta(days=365.0)

    s1 = energy_normalized.loc[start:split]
    s2 = energy_normalized.loc[split:]

    if s1.dropna().shape[0] < 2:
        raise ValueError(
            'hybrid analysis requires at least 2 samples in the first '
            'year window'
        )

    # Dispatch based on the year-1 method's declared input contract.
    if year1_input_kind == 'full':
        year1_input = energy_normalized
    else:  # 'year1'
        year1_input = s1.copy()

    year1_result = year1_func(year1_input, confidence_level=confidence_level,
                              **year1_kwargs)

    years2plus_result = degradation_year_on_year(
        s2.copy(), recenter=recenter_year2,
        confidence_level=confidence_level, **yoy_kwargs)

    Rd_pct_year1 = year1_result[0]
    Rd_pct_years2plus = years2plus_result[0]

    if isinstance(years2plus_result, tuple):
        renorm_year2 = years2plus_result[2].get('renormalizing_factor', 1.0)
    else:
        renorm_year2 = 1.0

    calc_info = {
        'year1': year1_result,
        'years2plus': years2plus_result,
        'year1_method': year1_func,
        'split_date': split,
        'renormalizing_factor_year2': renorm_year2,
    }

    return (Rd_pct_year1, Rd_pct_years2plus, calc_info)


def _avg_timestamp_old_Pandas(dt, dt_left):
    '''
    For old Pandas versions < 2.0.0, time columns cannot be averaged
    together.  From https://stackoverflow.com/questions/57812300/
    python-pandas-to-calculate-mean-of-datetime-of-multiple-columns

    Parameters
    ----------
    dt : pandas.Series
        First series with datetime values
    dt_left : pandas.Series
        Second series with datetime values.

    Returns
    -------
    pandas.Series
        Series with the average timestamp of df1 and df2.
    '''
    import calendar

    # Remove timezone from datetime values for averaging
    temp_df = pd.DataFrame(
        {"dt": dt.dt.tz_localize(None), "dt_left": dt_left.dt.tz_localize(None)}
    )

    # conversion from dates to seconds since epoch (unix time)
    def to_unix(s):
        if isinstance(s, pd.Timestamp):
            return calendar.timegm(s.timetuple())
        else:
            return pd.NaT

    # sum the seconds since epoch, calculate average, and convert back to readable date
    averages = []
    for index, row in temp_df.iterrows():
        unix = [to_unix(i) for i in row]
        # unix = [pd.Timestamp(i).timestamp() for i in row]
        try:
            average = sum(unix) / len(unix)
            # averages.append(datetime.datetime.utcfromtimestamp(average).strftime('%Y-%m-%d'))
            averages.append(pd.to_datetime(average, unit='s'))
        except TypeError:
            averages.append(pd.NaT)
    temp_df['averages'] = averages

    dt_center = temp_df["averages"].dt.tz_localize(dt.dt.tz)
    dt_center.index = dt.index
    dt_center.name = "averages"

    return dt_center


def _mk_test(x, alpha=0.05):
    '''
    Mann-Kendall test of significance for trend (used in classical
    decomposition function)

    Parameters
    ----------
    x : numeric
        A data vector to test for trend.
    alpha: float, default 0.05
        The test significance level.

    Returns
    -------
    trend : str
        Tells the trend ('increasing', 'decreasing', or 'no trend')
    h : bool
        True (if trend is present) or False (if trend is absent)
    p : float
        p value of the significance test
    z : float
        normalized test statistic
    '''

    from scipy.stats import norm

    n = len(x)

    # calculate S
    x = np.array(x)
    s = np.sum(np.triu(np.sign(-np.subtract.outer(x, x)), 1))

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:
        # there is no tie
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:
        # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n * (n - 1) * (2 * n + 5) +
                 np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)

    # calculate the p_value for two tail test
    p = 2 * (1 - norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1 - alpha / 2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, z


def _degradation_CI(results, confidence_level):
    '''
    Monte Carlo estimation of uncertainty in degradation rate from OLS results

    Parameters
    ----------
    results: OLSResults object from fitting a model of the form:
        results = sm.OLS(endog = df.energy_ma,
                         exog = df.loc[:,['const','years']]).fit()
    confidence_level: the size of the confidence interval to return, in percent

    Returns
    -------
    Confidence interval for degradation rate

    '''

    sampled_normal = np.random.multivariate_normal(results.params,
                                                   results.cov_params(),
                                                   10000)
    dist = sampled_normal[:, 1] / sampled_normal[:, 0]
    half_ci = confidence_level / 2.0
    Rd_CI = np.percentile(dist, [50.0 - half_ci, 50.0 + half_ci]) * 100.0
    return Rd_CI
