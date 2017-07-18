import pandas as pd


def csi_filter(measured_poa, clearsky_poa, threshold=0.9):
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
    return csi >= threshold
