'''Functions for filtering and subsetting PV system data.'''

import numpy as np


def normalized_filter(normalized, low_cutoff=0.01, high_cutoff=None):
    '''
    Select normalized yield between ``low_cutoff`` and ``high_cutoff``

    Parameters
    ----------
    normalized : pd.Series
        Normalized power measurements.
    low_cutoff : float, default 0.01
        The lower bound of acceptable values.
    high_cutoff : float, optional
        The upper bound of acceptable values.

    Returns
    -------
    pd.Series
        Boolean Series of whether the given measurement is within acceptable
        bounds.
    '''

    if low_cutoff is None:
        low_cutoff = -np.inf
    if high_cutoff is None:
        high_cutoff = np.inf

    return (normalized > low_cutoff) & (normalized < high_cutoff)


def poa_filter(poa_global, low_irradiance_cutoff=200,
               high_irradiance_cutoff=1200):
    '''
    Filter POA irradiance readings outside acceptable measurement bounds.

    Parameters
    ----------
    poa_global : pd.Series
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
    return ((poa_global > low_irradiance_cutoff) &
            (poa_global < high_irradiance_cutoff))


def tcell_filter(temperature_cell, low_tcell_cutoff=-50, high_tcell_cutoff=110):
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
    return ((temperature_cell > low_tcell_cutoff) &
            (temperature_cell < high_tcell_cutoff))


def clip_filter(power_ac, quant=0.98):
    '''
    Filter data points likely to be affected by clipping
    with power greater than or equal to 99% of the `quant`
    quantile.

    Parameters
    ----------
    power_ac : pd.Series
        AC power in Watts
    quant : float, default 0.98
        Value for upper threshold quantile

    Returns
    -------
    pd.Series
        Boolean Series of whether the given measurement is below 99% of the
        quantile filter.
    '''
    v = power_ac.quantile(quant)
    return (power_ac < v * 0.99)


def csi_filter(poa_global_meas, poa_global_sim, threshold=0.15):
    '''
    Filtering based on clear-sky index (csi)

    Parameters
    ----------
    poa_global_meas : pd.Series
        Plane of array irradiance based on measurments
    poa_global_sim : pd.Series
        Plane of array irradiance based on a clear sky model
    threshold : float, default 0.15
        threshold for filter

    Returns
    -------
    pd.Series
        Boolean Series of whether the clear-sky index is within the threshold
        around 1.
    '''

    csi = poa_global_meas / poa_global_sim
    return (csi >= 1.0 - threshold) & (csi <= 1.0 + threshold)
