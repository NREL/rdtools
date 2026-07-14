import pandas as pd
import numpy as np
from rdtools.degradation import degradation_year_on_year, degradation_hybrid
from rdtools.filtering import logic_clip_filter
from rdtools.soiling import soiling_srr
from rdtools.plotting import (
    degradation_summary_plots,
    hybrid_degradation_summary_plots,
    soiling_monte_carlo_plot,
    soiling_interval_plot,
    soiling_rate_histogram,
    tune_filter_plot,
    availability_summary_plots,
    degradation_timeseries_plot
)
import matplotlib.pyplot as plt
import matplotlib
import plotly
import pytest
import re
import copy

from conftest import assert_isinstance


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
    assert_isinstance(result, plt.Figure)
    plt.close('all')


def test_degradation_summary_plots_kwargs(degradation_info):
    power, yoy_rd, yoy_ci, yoy_info = degradation_info

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
        detailed=True,
    )
    result = degradation_summary_plots(yoy_rd, yoy_ci, yoy_info, power,
                                       **kwargs)
    assert_isinstance(result, plt.Figure)

    # ensure the number of points is included when detailed=True
    ax = result.axes[1]
    labels = [c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)]
    text = labels[0].get_text()
    assert re.search(r'n = \d', text)
    plt.close('all')


@pytest.fixture()
def hybrid_degradation_info(degradation_power_signal):
    '''
    Return results of running degradation_hybrid on the synthetic
    power signal.

    Returns
    -------
    power_signal : pd.Series
    rd_pct_year1 : float
    rd_pct_years2plus : float
    calc_info : dict with keys ``year1``, ``years2plus``, ``split_date``,
        ``renormalizing_factor_year2``
    '''
    rd1, rd2, info = degradation_hybrid(degradation_power_signal)
    return degradation_power_signal, rd1, rd2, info


def test_hybrid_degradation_summary_plots(hybrid_degradation_info):
    power, rd1, rd2, info = hybrid_degradation_info

    # test defaults
    result = hybrid_degradation_summary_plots(rd1, rd2, info, power)
    assert_isinstance(result, plt.Figure)
    # two axes: scatter on the left, histogram on the right
    assert len(result.axes) == 2
    plt.close('all')


def test_hybrid_degradation_summary_plots_kwargs(hybrid_degradation_info):
    power, rd1, rd2, info = hybrid_degradation_info

    kwargs = dict(
        hist_xmin=-2,
        hist_xmax=2,
        bins=50,
        scatter_ymin=0.5,
        scatter_ymax=1.2,
        year1_color='g',
        years2plus_color='m',
        summary_title='hybrid test',
        scatter_alpha=1.0,
    )
    result = hybrid_degradation_summary_plots(rd1, rd2, info, power, **kwargs)
    assert_isinstance(result, plt.Figure)
    assert result._suptitle.get_text() == 'hybrid test'
    plt.close('all')


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
    np.random.seed(1977)
    sr, sr_ci, calc_info = soiling_srr(soiling_normalized_daily,
                                       soiling_insolation,
                                       reps=reps)
    return calc_info


def test_soiling_monte_carlo_plot(soiling_normalized_daily, soiling_info):
    # test defaults
    result = soiling_monte_carlo_plot(soiling_info, soiling_normalized_daily)
    assert_isinstance(result, plt.Figure)
    plt.close('all')


def test_soiling_monte_carlo_plot_kwargs(soiling_normalized_daily,
                                         soiling_info):
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
    assert_isinstance(result, plt.Figure)
    plt.close('all')


def test_soiling_interval_plot(soiling_normalized_daily, soiling_info):
    # test defaults
    result = soiling_interval_plot(soiling_info, soiling_normalized_daily)
    assert_isinstance(result, plt.Figure)
    plt.close('all')


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
    assert_isinstance(result, plt.Figure)
    plt.close('all')


def test_soiling_rate_histogram(soiling_info):
    # test defaults
    result = soiling_rate_histogram(soiling_info)
    assert_isinstance(result, plt.Figure)
    plt.close('all')


def test_soiling_rate_histogram_kwargs(soiling_info):
    # test kwargs
    kwargs = dict(
        bins=10,
    )
    result = soiling_rate_histogram(soiling_info, **kwargs)
    assert_isinstance(result, plt.Figure)
    plt.close('all')


@pytest.fixture()
def clipping_power_degradation_signal():
    clipping_power_series = pd.Series(np.arange(1, 101))
    # Add datetime index to second series
    time_range = pd.date_range("2016-12-02T11:00:00.000Z", "2017-06-06T07:00:00.000Z", freq="h")
    clipping_power_series.index = pd.to_datetime(time_range[:100])
    return clipping_power_series


@pytest.fixture()
def clipping_info(clipping_power_degradation_signal):
    '''
    Return results of clipping filter applied to a degradation signal.

    Returns
    -------
    signal_filtered: Pandas series, filtered degradation power signal
    clipping_mask_series: Pandas series, boolean mask time series for
    clipping, with True indicating a non-clipping period and False
    representing a clipping period
    '''
    clipping_mask_series = logic_clip_filter(clipping_power_degradation_signal)
    return clipping_mask_series


def test_clipping_filter_plots(clipping_info,
                               clipping_power_degradation_signal):
    clipping_mask_series = clipping_info
    # test defaults
    result = tune_filter_plot(clipping_power_degradation_signal,
                              clipping_mask_series,
                              display_web_browser=False)
    assert_isinstance(result, plotly.graph_objs._figure.Figure)


def test_filter_plots_kwargs(clipping_info,
                             clipping_power_degradation_signal):
    clipping_mask_series = clipping_info

    # test kwargs
    kwargs = dict(
        display_web_browser=False
    )
    result = tune_filter_plot(clipping_power_degradation_signal,
                              clipping_mask_series,
                              **kwargs)
    assert_isinstance(result, plotly.graph_objs._figure.Figure)


def test_availability_summary_plots(availability_analysis_object):
    aa = availability_analysis_object
    result = availability_summary_plots(
        aa.power_system, aa.power_subsystem, aa.loss_total,
        aa.energy_cumulative, aa.energy_expected_rescaled,
        aa.outage_info)
    assert_isinstance(result, plt.Figure)
    plt.close('all')


def test_availability_summary_plots_empty(availability_analysis_object):
    # empty outage_info
    aa = availability_analysis_object
    empty = aa.outage_info.iloc[:0, :]
    result = availability_summary_plots(
        aa.power_system, aa.power_subsystem, aa.loss_total,
        aa.energy_cumulative, aa.energy_expected_rescaled,
        empty)
    assert_isinstance(result, plt.Figure)
    plt.close('all')


def test_degradation_timeseries_plot(degradation_info):
    import warnings
    power, yoy_rd, yoy_ci, yoy_info = degradation_info

    # test defaults; expect both the UserWarning (center) and the
    # FutureWarning (min_periods_divisor) to be raised when neither is passed.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result_right = degradation_timeseries_plot(yoy_info)
        assert_isinstance(result_right, plt.Figure)
        rightax = result_right.get_axes()[0].get_xlim()[0]
        assert len(w) > 0, "Expected at least one warning to be raised"
        assert any(issubclass(warn.category, UserWarning) for warn in w), \
            "Expected a UserWarning to be raised for multi-YoY values"
        assert any(issubclass(warn.category, FutureWarning)
                   and "min_periods_divisor" in str(warn.message) for warn in w), \
            "Expected a FutureWarning about min_periods_divisor default"

    with pytest.raises(KeyError):
        degradation_timeseries_plot({'a': 1}, include_ci=False, center=False,
                                    min_periods_divisor=2)

    # Add multi-YoY test by duplication idx=100.
    yoy_multi = copy.deepcopy(yoy_info)
    new_idx = yoy_multi['YoY_values'].index[100]
    new_val = yoy_multi['YoY_values'].iloc[100]
    yoy_values_multi = pd.concat([yoy_multi['YoY_values'], pd.Series([new_val], index=[new_idx])])
    yoy_multi['YoY_values'] = yoy_values_multi
    result = degradation_timeseries_plot(yoy_info=yoy_multi, include_ci=False,
                                         center=True, min_periods_divisor=2)
    centerax = result.get_axes()[0].get_xlim()[0]
    assert_isinstance(result, plt.Figure)
    assert centerax < rightax, "Expected center-labeled plot to be left of right-labeled plot"

    # explicit min_periods_divisor=4 (the planned future default) also works
    result4 = degradation_timeseries_plot(yoy_info=yoy_multi, include_ci=False,
                                          center=True, min_periods_divisor=4)
    assert_isinstance(result4, plt.Figure)

    plt.close('all')
