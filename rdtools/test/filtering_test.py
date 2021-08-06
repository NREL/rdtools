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
import warnings


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
def generate_power_time_series_irregular_intervals():
    power_datetime_index = pd.Series(np.arange(1, 62))
    # Add datetime index to second series
    time_range_1 = pd.date_range('2016-12-02T11:00:00.000Z',
                                 '2017-06-06T07:00:00.000Z', freq='1T')
    power_datetime_index.index = pd.to_datetime(time_range_1[:61])
    power_datetime_index_2 = pd.Series(np.arange(100, 200))
    time_range_2 = pd.date_range(power_datetime_index.index.max(),
                                 '2017-06-06T07:00:00.000Z', freq='15T')
    power_datetime_index_2.index = pd.to_datetime(time_range_2[:100])
    power_datetime_index_2 = power_datetime_index_2.iloc[1:]
    power_datetime_index = pd.concat([power_datetime_index,
                                      power_datetime_index_2])
    power_datetime_index_3 = pd.Series(list(reversed(np.arange(100, 200))))
    time_range_3 = pd.date_range(power_datetime_index.index.max(),
                                 '2017-06-06T07:00:00.000Z', freq='5T')
    power_datetime_index_3.index = pd.to_datetime(time_range_3[:100])
    power_datetime_index_3 = power_datetime_index_3.iloc[1:]
    power_datetime_index = pd.concat([power_datetime_index,
                                      power_datetime_index_3])
    power_datetime_index.sort_index()
    # Note: Power is expected to be Series object with a datetime index.
    return power_datetime_index


@pytest.fixture
def generate_power_time_series_one_min_intervals():
    power_datetime_index = pd.Series(np.arange(1, 51))
    power_datetime_index = pd.concat([power_datetime_index,
                                      power_datetime_index[::-1]])
    # Add datetime index to second series
    time_range = pd.date_range('2016-12-02T11:00:00.000Z',
                               '2017-06-06T07:00:00.000Z', freq='1T')
    power_datetime_index.index = pd.to_datetime(time_range[:100])
    # Note: Power is expected to be Series object with a datetime index.
    return power_datetime_index


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
                           generate_power_time_series_clipping,
                           generate_power_time_series_one_min_intervals,
                           generate_power_time_series_irregular_intervals):
    ''' Unit tests for logic clipping filter.'''
    power_no_datetime_index_nc, power_datetime_index_nc = \
        generate_power_time_series_no_clipping
    # Test that a Type Error is raised when a pandas series
    # without a datetime index is used.
    pytest.raises(TypeError,  logic_clip_filter,
                  power_no_datetime_index_nc)
    # Test that an error is thrown when we don't include the correct
    # mounting configuration input
    pytest.raises(ValueError,  logic_clip_filter,
                  power_datetime_index_nc, 'not_fixed')
    # Test that an error is thrown when there are 10 or fewer readings
    # in the time series
    pytest.raises(Exception,  logic_clip_filter,
                  power_datetime_index_nc[:9])
    # Scramble the index and run through the filter. This should throw
    # an IndexError.
    power_datetime_index_nc_shuffled = power_datetime_index_nc.sample(frac=1)
    pytest.raises(IndexError,  logic_clip_filter,
                  power_datetime_index_nc_shuffled, 'fixed')
    # Generate 1-minute interval data, run it through the function, and
    # check that the associated data returned is 1-minute
    power_datetime_index_one_min_intervals = \
        generate_power_time_series_one_min_intervals
    mask_one_min = logic_clip_filter(power_datetime_index_one_min_intervals)
    # Generate irregular interval data, and run it through the XGBoost model
    power_datetime_index_irregular = \
        generate_power_time_series_irregular_intervals()
    # Make sure that the routine throws a warning when the data sampling
    # frequency is less than 95% consistent
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        logic_clip_filter(power_datetime_index_irregular)
        # Warning thrown for it being an experimental filter + irregular
        # sampling frequency.
        assert len(w) == 3
    # Check that the returned time series index for the logic filter is
    # the same as the passed time series index
    mask_irregular = logic_clip_filter(power_datetime_index_irregular)
    # Expect none of the sequence to be clipped (as it's
    # constantly increasing)
    mask_nc = logic_clip_filter(power_datetime_index_nc)
    # Test the time series where the data is clipped
    power_no_datetime_index_c, power_datetime_index_c = \
        generate_power_time_series_clipping
    # Expect 4 values in middle of sequence to be clipped (when x=50)
    mask_c = logic_clip_filter(power_datetime_index_c)
    filtered_c = power_datetime_index_c[mask_c]
    assert bool(mask_nc.all(axis=None))
    assert (len(filtered_c) == 96)
    assert bool((mask_one_min.index.to_series().diff()[1:] ==
                 np.timedelta64(60, 's')).all(axis=None))
    assert bool((mask_irregular.index == power_datetime_index_irregular.index)
                .all(axis=None))


def test_xgboost_clip_filter(generate_power_time_series_no_clipping,
                             generate_power_time_series_clipping,
                             generate_power_time_series_one_min_intervals,
                             generate_power_time_series_irregular_intervals):
    ''' Unit tests for XGBoost clipping filter.'''
    # Test the time series where the data isn't clipped
    power_no_datetime_index_nc, power_datetime_index_nc = \
        generate_power_time_series_no_clipping
    # Test that a Type Error is raised when a pandas series
    # without a datetime index is used.
    pytest.raises(TypeError,  xgboost_clip_filter,
                  power_no_datetime_index_nc)
    # Test that an error is thrown when we don't include the correct
    # mounting configuration input
    pytest.raises(ValueError,  xgboost_clip_filter,
                  power_datetime_index_nc, 'not_fixed')
    # Test that an error is thrown when there are 10 or fewer readings
    # in the time series
    pytest.raises(Exception,  xgboost_clip_filter,
                  power_datetime_index_nc[:9])
    # Scramble the index and run through the filter. This should throw
    # an IndexError.
    power_datetime_index_nc_shuffled = power_datetime_index_nc.sample(frac=1)
    pytest.raises(IndexError,  xgboost_clip_filter,
                  power_datetime_index_nc_shuffled, 'fixed')
    # Generate 1-minute interval data, run it through the function, and
    # check that the associated data returned is 1-minute
    power_datetime_index_one_min_intervals = \
        generate_power_time_series_one_min_intervals
    mask_one_min = xgboost_clip_filter(power_datetime_index_one_min_intervals)
    # Generate irregular interval data, and run it through the XGBoost model
    power_datetime_index_irregular = \
        generate_power_time_series_irregular_intervals
    # Check that the returned time series index for XGBoost is the same
    # as the passed time series index
    mask_irregular = xgboost_clip_filter(power_datetime_index_irregular)
    # Expect none of the sequence to be clipped (as it's
    # constantly increasing)
    mask_nc = xgboost_clip_filter(power_datetime_index_nc)
    # Test the time series where the data is clipped
    power_no_datetime_index_c, power_datetime_index_c = \
        generate_power_time_series_clipping
    # Expect 4 values in middle of sequence to be clipped (when x=50)
    mask_c = xgboost_clip_filter(power_datetime_index_c)
    filtered_c = power_datetime_index_c[mask_c]
    assert bool(mask_nc.all(axis=None))
    assert (len(filtered_c) == 96)
    assert bool((mask_one_min.index.to_series().diff()[1:] ==
                 np.timedelta64(60, 's')).all(axis=None))
    assert bool((mask_irregular.index == power_datetime_index_irregular.index)
                .all(axis=None))


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
    # Check that the clip filter defaults to quantile clip filter when
    # deprecated params are passed
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        clip_filter(power_datetime_index_nc, 0.98)
        assert len(w) == 1
    # Check that a ValueError is thrown when a model is passed that
    # is not in the acceptable list.
    pytest.raises(ValueError, clip_filter,
                  power_datetime_index_nc,
                  'random_forest')
    # Check that the wrapper handles the xgboost clipping
    # function with kwargs.
    filtered_xgboost = clip_filter(power_datetime_index_nc,
                                   'xgboost',
                                   mounting_type="fixed")
    # Check that the wrapper handles the logic clipping
    # function with kwargs.
    filtered_logic = clip_filter(power_datetime_index_nc,
                                 'logic',
                                 mounting_type="fixed",
                                 rolling_range_max_cutoff=0.3)
    # Check that the function returns a Typr Error if a wrong keyword
    # arg is passed in the kwarg arguments.
    pytest.raises(TypeError, clip_filter, power_datetime_index_nc,
                  'xgboost',
                  rolling_range_max_cutoff=0.3)
    assert bool((expected_result_quantile == filtered_quantile)
                .all(axis=None))
    assert bool(filtered_xgboost.all(axis=None))
    assert bool(filtered_logic.all(axis=None))


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
