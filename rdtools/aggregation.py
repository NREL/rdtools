'''Functions for calculating weighted aggregates of PV system data.'''


def aggregation_insol(energy_normalized, insolation, frequency='D'):
    '''
    Insolation weighted aggregation

    Parameters
    ----------
    energy_normalized : pd.Series
        Normalized energy time series
    insolation : pd.Series
        Time series of insolation associated with each `energy_normalized`
        point
    frequency : Pandas offset string, default 'D'
        Target frequency at which to aggregate

    Returns
    -------
    aggregated : pd.Series
        Insolation weighted average, aggregated at frequency
    '''
    aggregated = (insolation * energy_normalized).resample(frequency).sum() / \
        insolation.resample(frequency).sum()

    return aggregated
