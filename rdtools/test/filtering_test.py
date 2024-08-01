""" Filtering Module Tests. """

import pytest
import pandas as pd
import numpy as np
from rdtools import (clearsky_filter,
                     csi_filter,
                     pvlib_clearsky_filter,
                     poa_filter,
                     tcell_filter,
                     clip_filter,
                     quantile_clip_filter,
                     normalized_filter,
                     logic_clip_filter,
                     xgboost_clip_filter,
                     two_way_window_filter,
                     insolation_filter,
                     hampel_filter,
                     directional_tukey_filter,
                     hour_angle_filter)
import warnings
from conftest import assert_warnings
from pandas import testing as tm


def test_clearsky_filter(mocker):
    ''' Unit tests for clearsky filter wrapper function.'''
    measured_poa = pd.Series([1, 1, 0, 1.15, 0.85])
    clearsky_poa = pd.Series([1, 2, 1, 1.00, 1.00])

    # Check that a ValueError is thrown when a model is passed that
    # is not in the acceptable list.
    with pytest.raises(ValueError):
        clearsky_filter(measured_poa,
                        clearsky_poa,
                        model='invalid')

    # Check that the csi_filter function is called
    mock_csi_filter = mocker.patch('rdtools.filtering.csi_filter')
    clearsky_filter(measured_poa,
                    clearsky_poa,
                    model='csi')
    mock_csi_filter.assert_called_once()

    # Check that the pvlib_clearsky_filter function is called
    mock_pvlib_filter = mocker.patch('rdtools.filtering.pvlib_clearsky_filter')
    clearsky_filter(measured_poa,
                    clearsky_poa,
                    model='pvlib')
    mock_pvlib_filter.assert_called_once()


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


@pytest.mark.parametrize("lookup_parameters", [True, False])
def test_pvlib_clearsky_filter(lookup_parameters):
    ''' Unit tests for pvlib clear sky filter.'''

    index = pd.date_range(start='01/05/2024 15:00', periods=120, freq='min')
    poa_global_clearsky = pd.Series(np.linspace(800, 919, 120), index=index)

    # Add cloud event
    poa_global_measured = poa_global_clearsky.copy()
    poa_global_measured.iloc[60:70] = [500, 400, 300, 200, 100, 0, 100, 200, 300, 400]

    filtered = pvlib_clearsky_filter(poa_global_measured,
                                     poa_global_clearsky,
                                     window_length=10,
                                     lookup_parameters=lookup_parameters)

    # Expect clearsky index is filtered.
    expected_result = poa_global_measured > 500
    pd.testing.assert_series_equal(filtered, expected_result)


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
    # Create a series that is tz-naive to test on
    power_datetime_index_tz_naive = power_datetime_index.copy()
    power_datetime_index_tz_naive.index =  \
        power_datetime_index_tz_naive.index.tz_localize(None)
    # Note: Power is expected to be Series object with a datetime index.
    return power_no_datetime_index, power_datetime_index, \
        power_datetime_index_tz_naive


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
    power_no_datetime_index = pd.Series(np.arange(2, 101, 2))
    power_no_datetime_index = pd.concat([power_no_datetime_index,
                                         power_no_datetime_index[::-1]])
    power_no_datetime_index[48:52] = 110
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
    power_no_datetime_index_nc, power_datetime_index_nc, power_nc_tz_naive = \
        generate_power_time_series_no_clipping
    # Test that a Type Error is raised when a pandas series
    # without a datetime index is used.
    with pytest.raises(TypeError):
        logic_clip_filter(power_no_datetime_index_nc)
    # Test that an error is thrown when we don't include the correct
    # mounting configuration input
    with pytest.raises(ValueError):
        logic_clip_filter(power_datetime_index_nc, 'not_fixed')
    # Test that an error is thrown when there are 10 or fewer readings
    # in the time series
    with pytest.raises(Exception):
        logic_clip_filter(power_datetime_index_nc[:9])
    # Test that a warning is thrown when the time series is tz-naive
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as record:
        logic_clip_filter(power_nc_tz_naive)
        # Warning thrown for it being an experimental filter + tz-naive
        assert_warnings(['Function expects timestamps in local time'],
                        record)
    # Scramble the index and run through the filter. This should throw
    # an IndexError.
    power_datetime_index_nc_shuffled = power_datetime_index_nc.sample(frac=1)
    with pytest.raises(IndexError):
        logic_clip_filter(power_datetime_index_nc_shuffled, 'fixed')
    # Generate 1-minute interval data, run it through the function, and
    # check that the associated data returned is 1-minute
    power_datetime_index_one_min_intervals = \
        generate_power_time_series_one_min_intervals
    mask_one_min = logic_clip_filter(power_datetime_index_one_min_intervals)
    # Generate irregular interval data, and run it through the XGBoost model
    power_datetime_index_irregular = \
        generate_power_time_series_irregular_intervals
    # Make sure that the routine throws a warning when the data sampling
    # frequency is less than 95% consistent
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as record:
        logic_clip_filter(power_datetime_index_irregular)
        # Warning thrown for it being an experimental filter + irregular
        # sampling frequency.
        assert_warnings(['Variable sampling frequency across time series'],
                        record)

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
    power_no_datetime_index_nc, power_datetime_index_nc, power_nc_tz_naive = \
        generate_power_time_series_no_clipping
    # Test that a Type Error is raised when a pandas series
    # without a datetime index is used.
    with pytest.raises(TypeError):
        xgboost_clip_filter(power_no_datetime_index_nc)
    # Test that an error is thrown when we don't include the correct
    # mounting configuration input
    with pytest.raises(ValueError):
        xgboost_clip_filter(power_datetime_index_nc, 'not_fixed')
    # Test that an error is thrown when there are 10 or fewer readings
    # in the time series
    with pytest.raises(Exception):
        xgboost_clip_filter(power_datetime_index_nc[:9])
    # Test that a warning is thrown when the time series is tz-naive
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as record:
        xgboost_clip_filter(power_nc_tz_naive)
        # Warning thrown for it being an experimental filter + tz-naive
        assert_warnings(['The XGBoost filter is an experimental',
                         'Function expects timestamps in local time'],
                        record)
    # Scramble the index and run through the filter. This should throw
    # an IndexError.
    power_datetime_index_nc_shuffled = power_datetime_index_nc.sample(frac=1)
    with pytest.raises(IndexError):
        xgboost_clip_filter(power_datetime_index_nc_shuffled, 'fixed')
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


def test_clip_filter(generate_power_time_series_clipping, mocker):
    ''' Unit tests for inverter clipping filter.'''
    # Create a time series to test
    _, power = generate_power_time_series_clipping

    # Check the default behavior
    expected = logic_clip_filter(power)
    mock_logic_clip_filter = mocker.patch('rdtools.filtering.logic_clip_filter', return_value=expected)
    filtered = clip_filter(power)
    mock_logic_clip_filter.assert_called_once()
    tm.assert_series_equal(filtered, expected)

    # Check each of the models
    expected_kwargs = {
    'mounting_type':'single_axis_tracking',
    'rolling_range_max_cutoff':0.3,
    'roll_periods':3
    }
    expected = logic_clip_filter(power, **expected_kwargs)
    mock_logic_clip_filter = mocker.patch('rdtools.filtering.logic_clip_filter', return_value=expected)
    filtered = clip_filter(power, model='logic', **expected_kwargs)
    mock_logic_clip_filter.assert_called_once()
    actual_kwargs = mock_logic_clip_filter.call_args.kwargs
    assert actual_kwargs == expected_kwargs
    tm.assert_series_equal(filtered, expected)

    expected_kwargs = {
    'quantile':0.95
    }
    expected = quantile_clip_filter(power, **expected_kwargs)
    mock_quantile_clip_filter = mocker.patch('rdtools.filtering.quantile_clip_filter', return_value=expected)
    filtered = clip_filter(power, model='quantile', **expected_kwargs)
    mock_quantile_clip_filter.assert_called_once()
    actual_kwargs = mock_quantile_clip_filter.call_args.kwargs
    assert actual_kwargs == expected_kwargs
    tm.assert_series_equal(filtered, expected)

    expected_kwargs = {
    'mounting_type':'single_axis_tracking'
    }
    expected = xgboost_clip_filter(power, **expected_kwargs)
    mock_xgboost_clip_filter = mocker.patch('rdtools.filtering.xgboost_clip_filter', return_value=expected)
    filtered = clip_filter(power, model='xgboost', **expected_kwargs)
    mock_xgboost_clip_filter.assert_called_once()
    actual_kwargs = mock_xgboost_clip_filter.call_args.kwargs
    assert actual_kwargs == expected_kwargs
    tm.assert_series_equal(filtered, expected)

    mocker.stopall()

    # Check that a ValueError is thrown when a model is passed that
    # is not in the acceptable list.
    with pytest.raises(ValueError):
        clip_filter(power, 'random_forest')

    # Check that the function returns a Type Error if a wrong keyword
    # arg is passed in the kwarg arguments.
    with pytest.raises(TypeError):
        clip_filter(power,
                    'xgboost',
                    rolling_range_max_cutoff=0.3)


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


def test_two_way_window_filter():
    # Create a pandas Series with 10 entries and daily index
    index = pd.date_range(start='1/1/2022', periods=10, freq='D')
    series = pd.Series([1, 2, 3, 4, 20, 6, 7, 8, 9, 10], index=index)

    # Call the function with the test data
    result = two_way_window_filter(series)

    # Check that the result is a pandas Series of the same length as the input
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)

    # Check that the result only contains boolean values
    assert set(result.unique()).issubset({True, False})

    # Check that the result is as expected
    # Here we're checking that the outlier is marked as False
    expected_result = pd.Series([True]*4 + [False]*2 + [True]*4, index=index)
    pd.testing.assert_series_equal(result, expected_result)


def test_insolation_filter():
    # Create a pandas Series with 10 entries
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Call the function with the test data
    result = insolation_filter(series)

    # Check that the result is a pandas Series of the same length as the input
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)

    # Check that the result only contains boolean values
    assert set(result.unique()).issubset({True, False})

    # Check that the result is as expected
    # Here we're checking that the bottom 10% of values are marked as False
    expected_result = pd.Series([False] + [True]*9)
    pd.testing.assert_series_equal(result, expected_result)


def test_hampel_filter():
    # Create a pandas Series with 10 entries and daily index
    index = pd.date_range(start='1/1/2022', periods=10, freq='D')
    series = pd.Series([1, 2, 3, 4, 100, 6, 7, 8, 9, 10], index=index)

    # Call the function with the test data
    result = hampel_filter(series)

    # Check that the result is a pandas Series of the same length as the input
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)

    # Check that the result only contains boolean values
    assert set(result.unique()).issubset({True, False})

    # Check that the result is as expected
    expected_result = pd.Series([True]*3 + [True] + [False] + [True]*5, index=index)
    pd.testing.assert_series_equal(result, expected_result)


def test_directional_tukey_filter():
    # Create a pandas Series with 10 entries and daily index
    index = pd.date_range(start='1/1/2022', periods=7, freq='D')
    series = pd.Series([1, 2, 3, 25, 4, 5, 6], index=index)

    # Call the function with the test data
    result = directional_tukey_filter(series)

    # Check that the result is a pandas Series of the same length as the input
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)

    # Check that the result is as expected
    expected_result = pd.Series([True, True, True, False, True, True, True], index=index)
    pd.testing.assert_series_equal(result, expected_result)


def test_hour_angle_filter():
    # Create a pandas Series with 5 entries and 15 min index
    index = pd.date_range(start='29/04/2022 15:00', periods=5, freq='H')
    series = pd.Series([1, 2, 3, 4, 5], index=index)

    # Define latitude and longitude
    lat, lon = 39.7413, -105.1684  # NREL, Golden, CO

    # Call the function with the test data
    result = hour_angle_filter(series, lat, lon)

    # Check that the result is a pandas Series of the same length as the input
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)

    # Check that the result is the correct boolean Series
    expected_result = pd.Series([False, False, True, True, True], index=index)
    pd.testing.assert_series_equal(result, expected_result)
