'''Functions for filtering and subsetting PV system data.'''


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
