'''
Aggregation Helper Functions
'''

def aggregation_insol(normalized_energy, insolation, frequency='D'):
    '''
    Insolation weighted aggregation

    Parameters
    ----------
    normalized_energy : pd.Series
        Normalized energy time series
    insolation : pd.Series
        Time series of insolation associated with each `normalized_energy`
        point
    frequency : Pandas offset string, default 'D'
        Target frequency at which to aggregate

    Returns
    -------
    aggregated : pd.Series
        Insolation weighted average, aggregated at frequency
    '''
    aggregated = (insolation * normalized_energy).resample(frequency).sum() / \
        insolation.resample(frequency).sum()

    return aggregated
