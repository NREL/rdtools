'''Functions for calculating weighted aggregates of PV system data.'''


def aggregation_insol(energy_normalized, insolation, frequency='D'):
    '''
    Insolation weighted aggregation

    Parameters
    ----------
    energy_normalized : pandas.Series
        Normalized energy time series
    insolation : pandas.Series
        Time series of insolation associated with each `energy_normalized`
        point
    frequency : str or pandas.tseries.offsets.DateOffset
        Pandas frequency specification at which to aggregate. Default is daily aggregation.
        For more information see
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    Returns
    -------
    aggregated : pandas.Series
        Insolation weighted average, aggregated at frequency
    '''
    aggregated = (insolation * energy_normalized).resample(frequency).sum() / \
        insolation.resample(frequency).sum()

    return aggregated
