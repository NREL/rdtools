""" Filtering Module Tests. """

import pytest
import pandas as pd
import numpy as np
from rdtools import (csi_filter,
                     poa_filter,
                     tcell_filter,
                     clip_filter,
                     quantile_clip_filter,
                     normalized_filter,
                     logic_clip_filter,
                     xgboost_clip_filter)


def test_csi_filter():
    ''' Unit tests for clear sky index filter.'''

    measured_poa = np.array([1, 1, 0, 1.15, 0.85])
    clearsky_poa = np.array([1, 2, 1, 1.00, 1.00])
    filtered = csi_filter(measured_poa,
                          clearsky_poa,
                          threshold=0.15)
    # Expect clearsky index is filtered with threshold of +/- 0.15.
    expected_result = np.array([True, False, False, True, True])
    assert filtered.tolist() == expected_result.tolist()


def test_poa_filter():
    ''' Unit tests for plane of array insolation filter.'''

    measured_poa = np.array([201, 1199, 500, 200, 1200])
    filtered = poa_filter(measured_poa,
                          poa_global_low=200,
                          poa_global_high=1200)
    # Expect high and low POA cutoffs to be non-inclusive.
    expected_result = np.array([True, True, True, False, False])
    assert filtered.tolist() == expected_result.tolist()


def test_tcell_filter():
    ''' Unit tests for cell temperature filter.'''

    tcell = np.array([-50, -49, 0, 109, 110])
    filtered = tcell_filter(tcell,
                            temperature_cell_low=-50,
                            temperature_cell_high=110)

    # Expected high and low tcell cutoffs to be non-inclusive.
    expected_result = np.array([False, True, True, True, False])
    assert filtered.tolist() == expected_result.tolist()


@pytest.fixture
def generate_power_time_series_no_clipping():
    power_no_datetime_index = pd.Series(np.arange(1, 101))
    power_datetime_index = pd.Series(np.arange(1, 101))
    # Add datetime index to second series
    time_range = pd.date_range('2016-12-02T11:00:00.000Z',
                               '2017-06-06T07:00:00.000Z', freq='H')
    power_datetime_index.index = pd.to_datetime(time_range[:100])
    # Note: Power is expected to be Series object with a datetime index.
    return power_no_datetime_index, power_datetime_index


@pytest.fixture
def generate_power_time_series_clipping():
    power_no_datetime_index = pd.Series(np.arange(1, 51))
    power_no_datetime_index = pd.concat([power_no_datetime_index,
                                         power_no_datetime_index[::-1]])
    power_no_datetime_index[48:52] = 50
    power_no_datetime_index = power_no_datetime_index.reset_index(drop=True)
    power_datetime_index = power_no_datetime_index.copy()
    # Add datetime index to second series
    time_range = pd.date_range('2016-12-02T11:00:00.000Z',
                               '2017-06-06T07:00:00.000Z', freq='H')
    power_datetime_index.index = pd.to_datetime(time_range[:100])
    # Note: Power is expected to be Series object with a datetime index.
    return power_no_datetime_index, power_datetime_index


def test_quantile_clip_filter():
    ''' Unit tests for inverter clipping filter.'''
    power = pd.Series(np.arange(1, 101))
    # Note: Power is expected to be Series object because clip_filter makes
    #       use of the Series.quantile() method.
    filtered = quantile_clip_filter(power, quantile=0.98)
    # Expect 99% of the 98th quantile to be filtered
    expected_result = power < (98 * 0.99)
    assert ((expected_result == filtered).all())


def test_logic_clip_filter(generate_power_time_series_no_clipping,
                           generate_power_time_series_clipping):
    ''' Unit tests for logic clipping filter.'''
    power_no_datetime_index_nc, power_datetime_index_nc = \
        generate_power_time_series_no_clipping
    # Test that a Type Error is raised when a pandas series
    # without a datetime index is used.
    pytest.raises(TypeError,  logic_clip_filter,
                  power_no_datetime_index_nc)
    # Expect none of the sequence to be clipped (as it's
    # constantly increasing)
    mask_nc = logic_clip_filter(power_datetime_index_nc)
    # Test the time series where the data is clipped
    power_no_datetime_index_c, power_datetime_index_c = \
        generate_power_time_series_clipping
    # Expect 4 values in middle of sequence to be clipped (when x=50)
    mask_c = logic_clip_filter(power_datetime_index_c)
    filtered_c = power_datetime_index_c[mask_c]
    assert (mask_nc.all()) & (len(filtered_c) == 96)


def test_xgboost_clip_filter(generate_power_time_series_no_clipping,
                             generate_power_time_series_clipping):
    ''' Unit tests for geometric clipping filter.'''
    # Test the time series where the data isn't clipped
    power_no_datetime_index_nc, power_datetime_index_nc = \
        generate_power_time_series_no_clipping
    # Test that a Type Error is raised when a pandas series
    # without a datetime index is used.
    pytest.raises(TypeError,  xgboost_clip_filter,
                  power_no_datetime_index_nc)
    # Expect none of the sequence to be clipped (as it's
    # constantly increasing)
    mask_nc = xgboost_clip_filter(power_datetime_index_nc)
    # Test the time series where the data is clipped
    power_no_datetime_index_c, power_datetime_index_c = \
        generate_power_time_series_clipping
    # Expect 4 values in middle of sequence to be clipped (when x=50)
    mask_c = xgboost_clip_filter(power_datetime_index_c)
    filtered_c = power_datetime_index_c[mask_c]
    assert (mask_nc.all()) & (len(filtered_c) == 96)


def test_clip_filter(generate_power_time_series_no_clipping):
    ''' Unit tests for inverter clipping filter.'''
    # Create a time series to test
    power_no_datetime_index_nc, power_datetime_index_nc = \
        generate_power_time_series_no_clipping
    # Check that the master wrapper defaults to the
    # quantile_clip_filter_function.
    # Note: Power is expected to be Series object because clip_filter makes
    #       use of the Series.quantile() method.
    filtered_quantile = clip_filter(power_no_datetime_index_nc, quantile=0.98)
    # Expect 99% of the 98th quantile to be filtered
    expected_result_quantile = power_no_datetime_index_nc < (98 * 0.99)
    # Check that the wrapper handles the xgboost clipping
    # function with kwargs.
    filtered_xgboost = clip_filter(power_datetime_index_nc,
                                   'xgboost_clip_filter',
                                   mounting_type="Fixed")
    # Check that the wrapper handles the logic clipping
    # function with kwargs.
    filtered_logic = clip_filter(power_datetime_index_nc,
                                 'logic_clip_filter',
                                 mounting_type="Fixed",
                                 max_rolling_derivative_cutoff=0.3)
    # Check that the function returns a Typr Error if a wrong keyword
    # arg is passed in the kwarg arguments.
    pytest.raises(TypeError, clip_filter, power_datetime_index_nc,
                  'xgboost_clip_filter',
                  derivative_cutoff=0.3)
    assert ((expected_result_quantile == filtered_quantile).all()) &\
        (filtered_xgboost.all()) & (filtered_logic.all())


def test_normalized_filter_default():
    pd.testing.assert_series_equal(normalized_filter(pd.Series([-5, 5])),
                                   pd.Series([False, True]))
    pd.testing.assert_series_equal(normalized_filter(
                        pd.Series([-1e6, 1e6]),
                        energy_normalized_low=None,
                        energy_normalized_high=None),
                        pd.Series([True, True]))

    pd.testing.assert_series_equal(normalized_filter(
                                pd.Series([-2, 2]),
                                energy_normalized_low=-1,
                                energy_normalized_high=1),
                                pd.Series([False, False]))

    eps = 1e-16
    pd.testing.assert_series_equal(normalized_filter(
                        pd.Series([0.01 - eps, 0.01 + eps, 1e308])),
                        pd.Series([False, True, True]))
