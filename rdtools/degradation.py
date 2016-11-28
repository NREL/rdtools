''' Degradation Module

This module contains functions to calculate the degradation rate of
photovoltaic systems.
'''

from __future__ import division
import pandas as pd
import numpy as np


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

    if pd.infer_freq(normalized_energy.index) == 'MS':
        y = y.rolling(12, center=True).mean()

    y = y.dropna()  # remove NaN values
    x = pd.Series(np.arange(0, len(y)), index=y.index)  # integer-months

    results = pd.ols(y=y, x=x)

    m, b = results.sm_ols.params

    Rd = (m * 12) / b

    N = len(y)
    rmse = np.sqrt(np.power(y - results.predict(x=x), 2).sum() / N)
    SE_m = rmse * np.sqrt((1/N) + np.power(x.mean(), 2)
                          / np.power(x - x.mean(), 2).sum())
    SE_b = rmse * np.sqrt(1 / np.power(x - x.mean(), 2).sum())
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
