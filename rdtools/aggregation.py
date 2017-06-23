import pandas as pd


def aggregation_insol(normalized_energy, insolation, frequency='D'):
    '''
    Insolation weighted aggregation

    Parameters
    ----------
    normalized_energy: Pandas series (numeric)
        Normalized energy time series
    insolation: Pandas series (numeric)
        Time series of insolation associated with each normalize_energy point
    frequency: Pandas offset string
        Target frequency at which to aggregate

    Returns
    -------
    aggregated: Pandas Series (numeric)
        Insolation weighted average, aggregated at frequency
    '''
    aggregated = (insolation * normalized_energy).resample(frequency).sum() / insolation.resample(frequency).sum()

    return aggregated
