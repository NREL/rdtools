import pandas as pd
import numpy as np
from typing import Callable, Tuple
from arch.bootstrap import CircularBlockBootstrap

def make_time_series_bootstrap_samples(
    signal: pd.Series, model_fit: pd.Series, sample_nr: int = 1000,
    block_length: int = 90, decomposition_type: str = 'multiplicative'
    ) -> pd.DataFrame:
    ''' Generate bootstrap samples based a time series signal and its model fit 
    
    Parameters
    ----------
    signal : pd.Series
        The time series signal that you want to make bootstrap samples of
    model_fit : pd.Series
        A model fit to the signal
    sample_nr : int, default 10
        The number of samples that you want to generate
    block_length : int, default 90
        Length of blocks to shuffle in block bootstrapping
    decomposition_type : string, default 'multiplicative'
        The type of decomposition to use with the model,
        either 'multiplicative' or 'additive'
    
    Returns
    -------
    bootstrap_samples : pd.DataFrame
        A dataframe contianing the bootstrap samples in the columns
    '''
    if decomposition_type == 'multiplicative':
        residuals = signal / model_fit
    elif decomposition_type == 'additive':
        residuals = signal - model_fit
    else:
        raise ValueError(
            'decomposition_type needs to be either \'multiplicative\' or'
            + ' \'additive\'')

    # Initialize return dataframe
    bootstrap_samples = pd.DataFrame(
        index=signal.index, columns=range(sample_nr))

    # Create circular blocks of boostrap samples
    bs = CircularBlockBootstrap(block_length, residuals)
    for b, bootstrapped_residuals in enumerate(bs.bootstrap(sample_nr)):
        if decomposition_type == 'multiplicative':
            bootstrap_samples.loc[:, b] = \
                model_fit * bootstrapped_residuals[0][0].values
        elif decomposition_type == 'additive':
            bootstrap_samples.loc[:, b] = \
                model_fit + bootstrapped_residuals[0][0].values
    
    return bootstrap_samples


def construct_confidence_intervals(
    bootstrap_samples: pd.DataFrame, fitting_function: 
    Callable[[pd.Series], float], exceedance_prob: float = 95,
    confidence_level: float = 68.2, **kwargs
    ) -> Tuple[np.array, float, pd.Series]:
    ''' Construct confidence intervals based on a set of bootstrap samples and
        a fitting function that takes a pandas series as input and returns a
        float
    
    Parameters
    ----------
    bootstrap_samples : pd.DataFrame
        A dataframe contianing the bootstrap samples in the columns
    fitting_function : function
        A function that fits a model to the bootstrap samples. Should take a
        series as input and returns a float
    exceedance_prob : float, default 95
        The probability level to use for exceedance value calculation,
        in percent.
    confidence_level : float, default 68.2
        The size of the confidence interval to return, in percent.
    **kwargs
        Keyword arguments to pass on to the `fitting_function`

    Returns
    -------
    confidence_interval : tuple(float, float)
        The confidence interval of the metric that is estimated in the
        `fitting_function`
    exceedance_level : float
        the degradation rate that was outperformed with probability of 
        `exceedance_prob`
    metrics : pd.Series
        Series of result metrics of the `fitting_function`
    '''
    # Estimate the set of metrics using the fitting function
    metrics = bootstrap_samples.apply(fitting_function, **kwargs)

    # Construct the confidence interval
    half_ci = confidence_level / 2.0
    confidence_interval = np.percentile(metrics, [50.0 - half_ci, 50.0 + half_ci])

    # Estimate exceedance level
    exceedance_level = np.percentile(metrics, 100.0 - exceedance_prob)
    
    return confidence_interval, exceedance_level, metrics
