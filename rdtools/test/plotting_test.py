import pandas as pd
import numpy as np
from rdtools.normalization import normalize_with_pvwatts
from rdtools.degradation import degradation_year_on_year
from rdtools.soiling import soiling_srr
from rdtools.plotting import (
    degradation_summary_plots,
    soiling_monte_carlo_plot,
    soiling_interval_plot,
    soiling_rate_histogram
)
import matplotlib.pyplot as plt
import pytest

# bring in soiling pytest fixtures
from soiling_test import (
    times, # can't rename this or else the others can't find it
    normalized_daily as soiling_normalized_daily,
    insolation as soiling_insolation,
)

# can't import degradation fixtures because it's a unittest file. 
# roll our own here instead:
@pytest.fixture()
def degradation_power_signal():
    ''' Returns a clean offset sinusoidal with exponential degradation '''
    idx = pd.date_range('2017-01-01', '2020-01-01', freq='d', tz='UTC')
    annual_rd = -0.005
    daily_rd = 1 - (1 - annual_rd)**(1/365)
    day_count = np.arange(0, len(idx))
    degradation_derate = (1 + daily_rd) ** day_count

    power = 1 - 0.1*np.cos(day_count/365 * 2*np.pi)
    power *= degradation_derate
    power = pd.Series(power, index=idx)
    return power


@pytest.fixture()
def degradation_info(degradation_power_signal):
    '''
    Return results of running YoY degradation on raw power.

    Note: no normalization needed since power is ~(1.0 + seasonality + deg)

    Returns
    -------
    power_signal : pd.Series
    degradation_rate : float
    confidence_interval : np.array of length 2
    calc_info : dict with keys:
        ['YoY_values', 'renormalizing_factor', 'exceedance_level']
    '''
    rd, rd_ci, calc_info = degradation_year_on_year(degradation_power_signal)
    return degradation_power_signal, rd, rd_ci, calc_info


def test_degradation_summary_plots(degradation_info):
    power, yoy_rd, yoy_ci, yoy_info = degradation_info
    
    # test defaults
    result = degradation_summary_plots(yoy_rd, yoy_ci, yoy_info, power)
    assert isinstance(result, plt.Figure)

    # test kwargs
    kwargs = dict(
        hist_xmin=-1,
        hist_xmax=1,
        bins=100,
        scatter_ymin=0,
        scatter_ymax=1,
        plot_color='g',
        summary_title='test',
        scatter_alpha=1.0,
    )
    result = degradation_summary_plots(yoy_rd, yoy_ci, yoy_info, power,
                                       **kwargs)
    assert isinstance(result, plt.Figure)



@pytest.fixture()
def soiling_info(soiling_normalized_daily, soiling_insolation):
    '''
    Return results of running soiling_srr.

    Returns
    -------
    calc_info : dict with keys:
        ['renormalizing_factor', 'exceedance_level',
        'stochastic_soiling_profiles', 'soiling_interval_summary',
        'soiling_ratio_perfect_clean']
    '''
    reps = 10
    sr, sr_ci, calc_info = soiling_srr(soiling_normalized_daily,
                                       soiling_insolation,
                                       reps=reps,
                                       random_seed=1977)
    return calc_info


def test_soiling_monte_carlo_plot(soiling_normalized_daily, soiling_info):
    # test defaults
    result = soiling_monte_carlo_plot(soiling_info, soiling_normalized_daily)
    assert isinstance(result, plt.Figure)


def test_soiling_monte_carlo_plot_kwargs(soiling_normalized_daily, soiling_info):
    # test kwargs
    kwargs = dict(
        point_alpha=0.1,
        profile_alpha=0.4,
        ymin=0,
        ymax=1,
        profiles=5,
        point_color='k',
        profile_color='b',
    )
    result = soiling_monte_carlo_plot(soiling_info, soiling_normalized_daily,
                                      **kwargs)
    assert isinstance(result, plt.Figure)


def test_soiling_interval_plot(soiling_normalized_daily, soiling_info):
    # test defaults
    result = soiling_interval_plot(soiling_info, soiling_normalized_daily)
    assert isinstance(result, plt.Figure)


def test_soiling_interval_plot_kwargs(soiling_normalized_daily, soiling_info):
    # test kwargs
    kwargs = dict(
        point_alpha=0.1,
        profile_alpha=0.5,
        ymin=0,
        ymax=1,
        point_color='k',
        profile_color='g',
    )
    result = soiling_interval_plot(soiling_info, soiling_normalized_daily,
                                   **kwargs)
    assert isinstance(result, plt.Figure)



def test_soiling_rate_histogram(soiling_info):
    # test defaults
    result = soiling_rate_histogram(soiling_info)
    assert isinstance(result, plt.Figure)


def test_soiling_rate_histogram_kwargs(soiling_info):
    # test kwargs
    kwargs = dict(
        bins=10,
    )
    result = soiling_rate_histogram(soiling_info, **kwargs)
    assert isinstance(result, plt.Figure)
