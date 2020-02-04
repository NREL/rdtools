'''Functions for filtering and subsetting PV system data.'''

import pandas as pd
import numpy as np


def poa_filter(poa, low_irradiance_cutoff=200, high_irradiance_cutoff=1200):
    '''
    Filter POA irradiance readings outside acceptable measurement bounds.

    Parameters
    ----------
    poa : pd.Series
        POA irradiance measurements.
    low_irradiance_cutoff : float, default 200
        The lower bound of acceptable values.
    high_irradiance_cutoff : float, default 1200
        The upper bound of acceptable values.

    Returns
    -------
    pd.Series
        Boolean Series of whether the given measurement is within acceptable
        bounds.
    '''
    return (poa > low_irradiance_cutoff) & (poa < high_irradiance_cutoff)


def tcell_filter(tcell, low_tcell_cutoff=-50, high_tcell_cutoff=110):
    '''
    Filter temperature readings outside acceptable measurement bounds.

    Parameters
    ----------
    tcell : pd.Series
        Cell temperature measurements.
    low_tcell_cutoff : float, default -50
        The lower bound of acceptable values.
    high_tcell_cutoff : float, default 110
        The upper bound of acceptable values.

    Returns
    -------
    pd.Series
        Boolean Series of whether the given measurement is within acceptable
        bounds.
    '''
    return (tcell > low_tcell_cutoff) & (tcell < high_tcell_cutoff)


def clip_filter(power, quant=0.98, low_power_cutoff=0.01):
    '''
    Filter data points likely to be affected by clipping
    with power greater than or equal to 99% of the `quant`
    quantile and less than `low_power_cutoff`

    Parameters
    ----------
    power : pd.Series
        AC power in Watts
    quant : float, default 0.98
        Value for upper threshold quantile
    low_power_cutoff : float, default 0.01
        Value for low-power cutoff (in Watts)

    Returns
    -------
    pd.Series
        Boolean Series of whether the given measurement is below 99% of the
        quantile filter and above the low-power cutoff.
    '''
    v = power.quantile(quant)
    return (power < v * 0.99) & (power > low_power_cutoff)


def csi_filter(measured_poa, clearsky_poa, threshold=0.15):
    '''
    Filtering based on clear-sky index (csi)

    Parameters
    ----------
    measured_poa : pd.Series
        Plane of array irradiance based on measurments
    clearsky_poa : pd.Series
        Plane of array irradiance based on a clear sky model
    threshold : float, default 0.15
        threshold for filter

    Returns
    -------
    pd.Series
        Boolean Series of whether the clear-sky index is within the threshold
        around 1.
    '''

    csi = measured_poa / clearsky_poa
    return (csi >= 1.0 - threshold) & (csi <= 1.0 + threshold)


def _all_close_to_first(data, rtol=1e-5, atol=1e-8):
    '''
    Returns True if all values in x are close to data[0].

    Parameters
    ----------
    x : array
    rtol : float, default 1e-5
        relative tolerance for detecting a change in data values
    atol : float, default 1e-8
        absolute tolerance for detecting a change in data values

    Returns
    -------
    Boolean
    '''
    return np.allclose(a=data, b=data[0], rtol=rtol, atol=atol)


def stale_values_filter(data, window=3, rtol=1e-5, atol=1e-8):
    '''
    Detects stale data.

    For a window of length N, the last value (index N-1) is considered stale
    if all values in the window are close to the first value (index 0).

    Parameters
    ----------
    data : pd.Series
        data to be processed
    window : int, default 3
        number of consecutive values which, if unchanged, indicates stale data
    rtol : float, default 1e-5
        relative tolerance for detecting a change in data values
    atol : float, default 1e-8
        absolute tolerance for detecting a change in data values

    Parameters rtol and atol have the same meaning as in numpy.allclose

    Returns
    -------
    pd.Series
        Boolean Series of whether the value is part of a stale sequence of data
    Raises
    ------
        ValueError if window < 2
    '''
    if window < 2:
        raise ValueError('window set to {}, must be at least 2'.format(window))

    flags = data.rolling(window=window).apply(
        _all_close_to_first, raw=True, kwargs={'rtol': rtol, 'atol': atol}
    ).fillna(False).astype(bool)
    return flags


def interpolation_filter(data, window=3, rtol=1e-5, atol=1e-8):
    '''
    Detects sequences of data which appear linear.

    Sequences are linear if the first difference appears to be constant.
    For a window of length N, the last value (index N-1) is flagged
    if all values in the window appear to be a line segment.

    Parameters
    ----------
    data : pd.Series
        data to be processed
    window : int, default 3
        number of sequential values that, if the first difference is constant,
        are classified as a linear sequence
    rtol : float, default 1e-5
        tolerance relative to max(abs(x.diff()) for detecting a change
    atol : float, default 1e-8
        absolute tolerance for detecting a change in first difference

    Returns
    -------
    pd.Series
        True if the value is part of a linear sequence

    Raises
    ------
        ValueError if window < 3
    '''
    if window < 3:
        raise ValueError('window set to {}, must be at least 3'.format(window))

    # reduce window by 1 because we're passing the first difference
    flags = stale_values_filter(data.diff(periods=1), window=window-1, rtol=rtol,
                                atol=atol)
    return flags

