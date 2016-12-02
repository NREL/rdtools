''' Degradation Module

This module contains functions to calculate the degradation rate of
photovoltaic systems.
'''

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.api as sm


def degradation_with_ols(normalized_energy):
    '''
    Description

    Parameters
    ----------
    normalized_energy: Pandas Series (numeric)
        Monthly time series of normalized system ouput.

    Returns
    -------
    degradation: dictionary
        Contains degradation rate and standard errors of regression
    '''

    y = normalized_energy

    if pd.infer_freq(normalized_energy.index) == 'MS' and len(y) > 12:
        # apply 12-month rolling mean
        y = y.rolling(12, center=True).mean()

    # remove NaN values
    y = y.dropna()

    # number of examples
    N = len(y)

    # integer-months, the exogeneous variable
    months = np.arange(0, len(y))

    # add intercept-constant to the exogeneous variable
    X = sm.add_constant(months)
    columns = ['constant', 'months']
    exog = pd.DataFrame(X, index=y.index, columns=columns)

    # fit linear model
    ols_model = sm.OLS(endog=y, exog=exog, hasconst=True)
    results = ols_model.fit()

    # collect intercept and slope
    b, m = results.params

    # rate of degradation in terms of percent/year
    Rd_pct = 100 * (m * 12) / b

    # root mean square error
    rmse = np.sqrt(np.power(y - results.predict(exog=exog), 2).sum() / (N - 2))

    # total sum of squares of the time variable
    tss_months = np.power(months - months.mean(), 2).sum()

    # standard error of the slope and intercept
    stderr_b = rmse * np.sqrt((1/N) + np.power(months.mean(), 2) / tss_months)
    stderr_m = rmse * np.sqrt(1 / tss_months)

    # standard error of the regression
    stderr_Rd = np.sqrt((stderr_m * 12/b)**2 + ((-12*m / b**2) * stderr_b)**2)
    stderr_Rd_pct = 100 * stderr_Rd

    degradation = {
        'Rd_pct': Rd_pct,
        'slope': m,
        'intercept': b,
        'rmse': rmse,
        'slope_stderr': stderr_m,
        'intercept_stderr': stderr_b,
        'Rd_stderr_pct': stderr_Rd_pct,
    }

    return degradation
