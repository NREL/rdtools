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
        Energy time series to be normalized.

    Returns
    -------
    degradation: dictionary
        Contains degradation rate and standard errors of regression
    '''

    y = normalized_energy

    if pd.infer_freq(normalized_energy.index) == 'MS' and len(y) > 12:
        y = y.rolling(12, center=True).mean()

    # remove NaN values
    y = y.dropna()

    # integer-months
    months = np.arange(0, len(y))
    X = sm.add_constant(months)
    columns = ['constant', 'months']
    exog = pd.DataFrame(X, index=y.index, columns=columns)

    ols_model = sm.OLS(endog=y, exog=exog, hasconst=True)
    results = ols_model.fit()

    # collect intercept and slope
    b, m = results.params

    Rd = (m * 12) / b

    N = len(y)
    rmse = np.sqrt(np.power(y - results.predict(exog=exog), 2).sum() / N)
    SE_m = rmse * np.sqrt((1/N) + np.power(months.mean(), 2)
                          / np.power(months - months.mean(), 2).sum())
    SE_b = rmse * np.sqrt(1 / np.power(months - months.mean(), 2).sum())
    SE_Rd = np.power(SE_m * 12/b, 2) + np.power((-12*m / b**2) * SE_b, 2)

    degradation = {
        'Rd': Rd,
        'slope': m,
        'intercept': b,
        'rmse': rmse,
        'slope_stderr': SE_m,
        'intercept_stderr': SE_b,
        'Rd_stderr': SE_Rd,
    }

    return degradation
