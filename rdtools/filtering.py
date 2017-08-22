import pandas as pd

def get_nominal_value(series, window):
    v1 = series[:window].median()
    v2 = series[len(series) - window:].median()
    v3 = series[int(len(series) * 0.5 - window * 0.5):int(len(series) * 0.5 + window * 0.5)].median()
    v = max(v1, v2, v3)
    return v


def poa_filter(poa, low_irradiance_cutoff=200, high_irradiance_cutoff=1200):
    # simple filter based on irradiance sensors
    return (poa > low_irradiance_cutoff) & (poa < high_irradiance_cutoff)

def tcell_filter(tcell, low_tcell_cutoff=-50, high_tcell_cutoff=110):
    # simple filter based on temperature sensors
    return (tcell > low_tcell_cutoff) & (tcell < high_tcell_cutoff)
    
def clip_filter(power, quant=0.95, low_power_cutoff=0.01):
    '''
    Clipping data points with power greater
    than or equal to 99% of the 95th percentile
    and less than 0.01 W

    Parameters
    ----------
    power: Pandas series (numeric)
        AC power
    quant: float
        threshold for quantile
    low_power_cutoff    

    Returns
    -------
    aggregated: Pandas Series (boolean)
        mask to exclude points equal to and 
        above 99% of the percentile threshold
    '''    
    v = power.quantile(quant)
    return (power < v * 0.99) & (power > low_power_cutoff)

def outage_filter(prt, ndays=30, nperiods=96):
    '''
    Clipping data points corresponding to outage

    Parameters
    ----------
    prt: Pandas series (numeric)
        normalized energy
    ndays: int
        number of days in the window
    nperiods: int
        number of periods in each day

    Returns
    -------
    aggregated: Pandas Series (boolean)
        mask to exclude points equal to and 
        above 99% of the percentile threshold
    '''
    v = prt.rolling(window=ndays * nperiods, min_periods=3).median()
    b = get_nominal_value(prt, 1000) * 0.3
    return (prt > v - b) & (prt < v + b)

def csi_filter(measured_poa, clearsky_poa, threshold=0.1):
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
    aggregated: Pandas Series (boolean)
        mask to exclude points below the threshold
    '''


    csi = measured_poa / clearsky_poa
    return (csi >= 1.0 - threshold) & (csi <= 1.0 + threshold)
