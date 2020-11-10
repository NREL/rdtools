'''Functions for calculating the degradation rate of photovoltaic systems.'''

import pandas as pd
import numpy as np
import statsmodels.api as sm


def degradation_ols(energy_normalized, confidence_level=68.2):
    '''
    Estimate the trend of a timeseries using ordinary least-squares regression
    and calculate various statistics including a Monte Carlo-derived confidence
    interval of slope.

    Parameters
    ----------
    energy_normalized: pd.Series
        Daily or lower frequency time series of normalized system ouput.
    confidence_level: float, default 68.2
        The size of the confidence interval to return, in percent.

    Returns
    -------
    Rd_pct : float
        Estimated degradation relative to the year 0 system capacity [%/year]
    Rd_CI : np.array
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
    df['days'] = day_diffs.astype('timedelta64[s]') / (60 * 60 * 24)
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
    energy_normalized: pd.Series
        Daily or lower frequency time series of normalized system ouput.
        Must be regular time series.
    confidence_level: float, default 68.2
        The size of the confidence interval to return, in percent.

    Returns
    -------
    Rd_pct : float
        Estimated degradation relative to the year 0 system capacity [%/year]
    Rd_CI : np.array
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
    df['days'] = day_diffs.astype('timedelta64[s]') / (60 * 60 * 24)
    df['years'] = df.days / 365.0

    # Compute yearly rolling mean to isolate trend component using
    # moving average
    it = df.iterrows()
    energy_ma = []
    for i, row in it:
        if row.years - 0.5 >= min(df.years) and \
           row.years + 0.5 <= max(df.years):
            roll = df[(df.years <= row.years + 0.5) &
                      (df.years >= row.years - 0.5)]
            energy_ma.append(roll.energy_normalized.mean())
        else:
            energy_ma.append(np.nan)

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


def degradation_year_on_year(energy_normalized, recenter=True,
                             exceedance_prob=95, confidence_level=68.2):
    '''
    Estimate the trend of a timeseries using the year-on-year decomposition
    approach and calculate a Monte Carlo-derived confidence interval of slope.

    Parameters
    ----------
    energy_normalized: pd.Series
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

    Returns
    -------
    Rd_pct : float
        Estimated degradation relative to the year 0 median system capacity [%/year]
    confidence_interval : np.array
        confidence interval (size specified by ``confidence_level``) of
        degradation rate estimate
    calc_info : dict

        * `YoY_values` - pandas series of right-labeled year on year slopes
        * `renormalizing_factor` - float of value used to recenter data
        * `exceedance_level` - the degradation rate that was outperformed with
          probability of `exceedance_prob`
    '''

    # Ensure the data is in order
    energy_normalized = energy_normalized.sort_index()
    energy_normalized.name = 'energy'
    energy_normalized.index.name = 'dt'

    # Detect sub-daily data:
    if min(np.diff(energy_normalized.index.values, n=1)) < \
            np.timedelta64(23, 'h'):
        raise ValueError('energy_normalized must not be '
                         'more frequent than daily')

    # Detect less than 2 years of data
    if energy_normalized.index[-1] - energy_normalized.index[0] < \
            pd.Timedelta('730d'):
        raise ValueError('must provide at least two years of '
                         'normalized energy')

    # Auto center
    if recenter:
        start = energy_normalized.index[0]
        oneyear = start + pd.Timedelta('364d')
        renorm = energy_normalized[start:oneyear].median()
    else:
        renorm = 1.0

    energy_normalized = energy_normalized.reset_index()
    energy_normalized['energy'] = energy_normalized['energy'] / renorm

    energy_normalized['dt_shifted'] = energy_normalized.dt + \
                                      pd.DateOffset(years=1)

    # Merge with what happened one year ago, use tolerance of 8 days to allow
    # for weekly aggregated data
    df = pd.merge_asof(energy_normalized[['dt', 'energy']], energy_normalized,
                       left_on='dt', right_on='dt_shifted',
                       suffixes=['', '_right'],
                       tolerance=pd.Timedelta('8D')
                       )

    df['time_diff_years'] = (df.dt - df.dt_right).astype('timedelta64[h]') / \
                            8760.0
    df['yoy'] = 100.0 * (df.energy - df.energy_right) / (df.time_diff_years)
    df.index = df.dt

    yoy_result = df.yoy.dropna()

    calc_info = {
        'YoY_values': yoy_result,
        'renormalizing_factor': renorm
    }

    if not len(yoy_result):
        raise ValueError('no year-over-year aggregated data pairs found')

    Rd_pct = yoy_result.median()

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
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])

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
