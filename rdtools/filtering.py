import pandas as pd
import numpy as np


def normalized_filter(normalized, low_cutoff=0, high_cutoff=None):
    '''Return a boolean pandas filter that selects normalized yield between low_cutoff and high_cutoff'''

    if low_cutoff is None:
        low_cutoff = -np.inf
    if high_cutoff is None:
        high_cutoff = np.inf

    return (normalized > low_cutoff) & (normalized < high_cutoff)


def poa_filter(poa, low_irradiance_cutoff=200, high_irradiance_cutoff=1200):
    # simple filter based on irradiance sensors
    return (poa > low_irradiance_cutoff) & (poa < high_irradiance_cutoff)


def tcell_filter(tcell, low_tcell_cutoff=-50, high_tcell_cutoff=110):
    # simple filter based on temperature sensors
    return (tcell > low_tcell_cutoff) & (tcell < high_tcell_cutoff)


def clip_filter(power, quant=0.98, low_power_cutoff=0.01):
    '''
    Filter data points likely to be affected by clipping
    with power greater than or equal to 99% of the 'quant'
    quantile and less than 'low_power_cutoff'

    Parameters
    ----------
    power: Pandas series (numeric)
        AC power
    quant: float
        threshold for quantile
    low_power_cutoff

    Returns
    -------
    Pandas Series (boolean)
        mask to exclude points equal to and
        above 99% of the percentile threshold
    '''
    v = power.quantile(quant)
    return (power < v * 0.99) & (power > low_power_cutoff)


def csi_filter(measured_poa, clearsky_poa, threshold=0.15):
    '''
    Filtering based on clear sky index (csi)

    Parameters
    ----------
    measured_poa: Pandas series (numeric)
        Plane of array irradiance based on measurments
    clearsky_poa: Pandas series (numeric)
        Plane of array irradiance based on a clear sky model
    threshold: float
        threshold for filter

    Returns
    -------
    Pandas Series (boolean)
        mask to exclude points below the threshold
    '''

    csi = measured_poa / clearsky_poa
    return (csi >= 1.0 - threshold) & (csi <= 1.0 + threshold)
