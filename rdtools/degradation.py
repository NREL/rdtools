''' Degradation Module

This module contains functions to calculate the degradation rate of
photovoltaic systems.
'''

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.api as sm


def degradation_ols(normalized_energy):
    '''
    Description
    -----------
    OLS routine

    Parameters
    ----------
    normalized_energy: Pandas Time Series (numeric)
        Daily or lower frequency time series of normalized system ouput.

    Returns
    -------
    (degradation rate, confidence interval, calc_info)
        calc_info is a dict that contains slope, intercept,
        root mean square error of regression ('rmse'), standard error
        of the slope ('slope_stderr'), intercept ('intercept_stderr'),
        and least squares RegressionResults object ('ols_results')
    '''

    normalized_energy.name = 'normalized_energy'
    df = normalized_energy.to_frame()

    # calculate a years column as x value for regression, ignoreing leap years
    day_diffs = (df.index - df.index[0])
    df['days'] = day_diffs.astype('timedelta64[s]') / (60 * 60 * 24)
    df['years'] = df.days / 365.0

    # add intercept-constant to the exogeneous variable
    df = sm.add_constant(df)

    # perform regression
    ols_model = sm.OLS(endog=df.normalized_energy, exog=df.loc[:, ['const', 'years']],
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
    Rd_CI = _degradation_CI(results)

    calc_info = {
        'slope': m,
        'intercept': b,
        'rmse': rmse,
        'slope_stderr': stderr_m,
        'intercept_stderr': stderr_b,
        'ols_result': results,
    }

    return (Rd_pct, Rd_CI, calc_info)


def degradation_classical_decomposition(normalized_energy):
    '''
    Description
    -----------
    Classical decomposition routine

    Parameters
    ----------
    normalized_energy: Pandas Time Series (numeric)
        Daily or lower frequency time series of normalized system ouput.
        Must be regular time series.

    Returns
    -------
    (degradation rate, confidence interval, calc_info)
    calc_info is a dict that contains values for
        slope, intercept, root mean square error of regression ('rmse'),
        standard error of the slope ('slope_stderr') and intercept ('intercept_stderr'),
        least squares RegressionResults object ('ols_results'),
        pandas series for the annual rolling mean ('series'), and
        Mann-Kendall test trend ('mk_test_trend')
    '''

    normalized_energy.name = 'normalized_energy'
    df = normalized_energy.to_frame()

    df_check_freq = df.copy()

    # The frequency attribute will be set to None if rows are dropped.
    # We can use this to check for missing data and raise a ValueError.
    df_check_freq = df_check_freq.dropna()

    if df_check_freq.index.freq is None:
        raise ValueError('Classical decomposition requires a regular time series with'
                         ' defined frequency and no missing data.')

    # calculate a years column as x value for regression, ignoreing leap years
    day_diffs = (df.index - df.index[0])
    df['days'] = day_diffs.astype('timedelta64[s]') / (60 * 60 * 24)
    df['years'] = df.days / 365.0

    # Compute yearly rolling mean to isolate trend component using moving average
    it = df.iterrows()
    energy_ma = []
    for i, row in it:
        if row.years - 0.5 >= min(df.years) and row.years + 0.5 <= max(df.years):
            roll = df[(df.years <= row.years + 0.5) & (df.years >= row.years - 0.5)]
            energy_ma.append(roll.normalized_energy.mean())
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
    Rd_CI = _degradation_CI(results)

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


def degradation_year_on_year(normalized_energy, recenter=True, exceedance_prob=95):
    '''
    Description
    -----------
    Year-on-year decomposition method

    Parameters
    ----------
    normalized_energy:  Pandas data series (numeric)
        corrected performance ratio timeseries index in monthly format
    recenter:  bool, default value True
        specify whether data is centered to normalized yield of 1 based on first year
    exceedance_prob (float): the probability level to use for exceedance value calculation

    Returns
    -------
    tuple of (degradation_rate, confidence interval, calc_info)
        degradation_rate:  float
            rate of relative performance change in %/yr
        confidence_interval:  float
            one-sigma confidence interval of degradation rate estimate
        calc_info:  dict
            ('YoY_values') pandas series of right-labeled year on year slopes
            ('renormalizing_factor') float of value used to recenter data
            ('exceedance_level') the degradation rate that was ouperformed with
            probability of exceedance_prob
    '''

    # Ensure the data is in order
    normalized_energy = normalized_energy.sort_index()
    normalized_energy.name = 'energy'
    normalized_energy.index.name = 'dt'

    # Detect sub-daily data:
    if min(np.diff(normalized_energy.index.values, n=1)) < np.timedelta64(23, 'h'):
        raise ValueError('normalized_energy must not be more frequent than daily')

    # Detect less than 2 years of data
    if normalized_energy.index[-1] - normalized_energy.index[1] < pd.Timedelta('730h'):
        raise ValueError('must provide at least two years of normalized energy')

    # Auto center
    if recenter:
        start = normalized_energy.index[0]
        oneyear = start + pd.Timedelta('364d')
        renorm = normalized_energy[start:oneyear].median()
    else:
        renorm = 1.0

    normalized_energy = normalized_energy.reset_index()
    normalized_energy['energy'] = normalized_energy['energy'] / renorm

    normalized_energy['dt_shifted'] = normalized_energy.dt + pd.DateOffset(years=1)

    # Merge with what happened one year ago, use tolerance of 8 days to allow for
    # weekly aggregated data
    df = pd.merge_asof(normalized_energy[['dt', 'energy']], normalized_energy,
                       left_on='dt', right_on='dt_shifted',
                       suffixes=['', '_right'],
                       tolerance=pd.Timedelta('8D')
                       )

    df['time_diff_years'] = (df.dt - df.dt_right).astype('timedelta64[h]') / 8760.0
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
    Rd_CI = np.percentile(mb1, [15.9, 84.1])

    P_level = np.percentile(mb1, 100 - exceedance_prob)

    calc_info['exceedance_level'] = P_level

    return (Rd_pct, Rd_CI, calc_info)


def _mk_test(x, alpha=0.05):
    '''
    Description
    -----------
    Mann-Kendall test of significance for trend (used in classical decomposition function)

    Parameters
    ----------
    x: a vector of data type float
    alpha: float, significance level (0.05 default)

    Returns
    -------
    trend: string, tells the trend (increasing, decreasing or no trend)
    h: boolean, True (if trend is present) or False (if trend is absence)
    p: float, p value of the significance test
    z: float, normalized test statistics
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


def _degradation_CI(results):
    '''
    Description
    -----------
    Monte Carlo estimation of uncertainty in degradation rate from OLS results

    Parameters
    ----------
    results: OLSResults object from fitting a model of the form:
    results = sm.OLS(endog = df.energy_ma, exog = df.loc[:,['const','years']]).fit()
    Returns
    -------
    68.2% confidence interval for degradation rate

    '''

    sampled_normal = np.random.multivariate_normal(results.params, results.cov_params(), 10000)
    dist = sampled_normal[:, 1] / sampled_normal[:, 0]
    Rd_CI = np.percentile(dist, [50 - 34.1, 50 + 34.1]) * 100.0
    return Rd_CI
