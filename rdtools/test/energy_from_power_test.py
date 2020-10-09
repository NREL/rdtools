import pandas as pd
import numpy as np
from rdtools import energy_from_power
import pytest


# Tests for resampling at same frequency
def test_energy_from_power_calculation():
    power_times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:00', freq='15T')
    result_times = power_times[1:]
    power_series = pd.Series(data=4.0, index=power_times)
    expected_energy_series = pd.Series(data=1.0, index=result_times)
    expected_energy_series.name = 'energy_Wh'

    result = energy_from_power(power_series, max_timedelta=pd.to_timedelta('15 minutes'))

    pd.testing.assert_series_equal(result, expected_energy_series)


def test_energy_from_power_max_interval():
    power_times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:00', freq='15T')
    result_times = power_times[1:]
    power_series = pd.Series(data=4.0, index=power_times)
    expected_energy_series = pd.Series(data=np.nan, index=result_times)
    expected_energy_series.name = 'energy_Wh'

    result = energy_from_power(power_series, max_timedelta=pd.to_timedelta('5 minutes'))

    # We expect series of NaNs, because max_interval_hours is smaller than the
    # time step of the power time series
    pd.testing.assert_series_equal(result, expected_energy_series)


def test_energy_from_power_validation():
    power_series = pd.Series(data=[4.0] * 4)
    with pytest.raises(ValueError):
        energy_from_power(power_series, max_timedelta=pd.to_timedelta('15 minutes'))


def test_energy_from_power_single_argument():
    power_times = pd.date_range('2018-04-01 12:00', '2018-04-01 15:00', freq='15T')
    result_times = power_times[1:]
    power_series = pd.Series(data=4.0, index=power_times)
    missing = pd.to_datetime('2018-04-01 13:00:00')
    power_series = power_series.drop(missing)

    expected_energy_series = pd.Series(data=1.0, index=result_times)
    expected_nan = [missing]
    expected_nan.append(pd.to_datetime('2018-04-01 13:15:00'))
    expected_energy_series.loc[expected_nan] = np.nan
    expected_energy_series.name = 'energy_Wh'

    # Test that the result has the expected missing timestamp based on median timestep
    result = energy_from_power(power_series)
    pd.testing.assert_series_equal(result, expected_energy_series)


# Tests for downsampling
def test_energy_from_power_downsample():
    times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:00', freq='15T')
    time_series = pd.Series(data=[1.0, 2.0, 3.0, 4.0, 5.0], index=times)

    expected_energy_series = pd.Series(index=[pd.to_datetime('2018-04-01 13:00:00')],
                                       data=3.0, name='energy_Wh')
    expected_energy_series.index.freq = '60T'
    result = energy_from_power(time_series, '60T')
    pd.testing.assert_series_equal(result, expected_energy_series)


def test_energy_from_power_downsample_max_timedelta_exceeded():
    times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:00', freq='15T')
    time_series = pd.Series(data=[1.0, 2.0, 3.0, 4.0, 5.0], index=times)

    expected_energy_series = pd.Series(index=[pd.to_datetime('2018-04-01 13:00:00')],
                                       data=1.5, name='energy_Wh')
    expected_energy_series.index.freq = '60T'
    result = energy_from_power(time_series.drop(time_series.index[2]), '60T', pd.to_timedelta('15 minutes'))
    pd.testing.assert_series_equal(result, expected_energy_series)


def test_energy_from_power_downsample_max_timedelta_not_exceeded():
    times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:00', freq='15T')
    time_series = pd.Series(data=[1.0, 2.0, 3.0, 4.0, 5.0], index=times)

    expected_energy_series = pd.Series(index=[pd.to_datetime('2018-04-01 13:00:00')],
                                       data=3.0, name='energy_Wh')
    expected_energy_series.index.freq = '60T'
    result = energy_from_power(time_series.drop(time_series.index[2]), '60T', pd.to_timedelta('60 minutes'))
    pd.testing.assert_series_equal(result, expected_energy_series)


def test_energy_from_power_for_issue_107():
    times = pd.date_range('2018-04-01 12:00', '2018-04-01 16:00', freq='15T')
    dc_power = pd.Series(index=times, data=1.0)
    dc_power = dc_power.drop(dc_power.index[5:12])

    expected_times = pd.date_range('2018-04-01 13:00', '2018-04-01 16:00', freq='60T')
    expected_energy_series = pd.Series(index=expected_times,
                                       data=[1.0, np.nan, np.nan, 1.0],
                                       name='energy_Wh')
    result = energy_from_power(dc_power, '60T')
    pd.testing.assert_series_equal(result, expected_energy_series)


# Tests for upsampling
def test_energy_from_power_upsample():
    times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:30', freq='30T')
    time_series = pd.Series(data=[1.0, 3.0, 5.0, 6.0], index=times)

    expected_result_times = pd.date_range('2018-04-01 12:15', '2018-04-01 13:30', freq='15T')
    expected_energy_series = pd.Series(index=expected_result_times,
                                       data=[0.375, 0.625, 0.875, 1.125, 1.3125, 1.4375],
                                       name='energy_Wh')

    result = energy_from_power(time_series, '15T', pd.to_timedelta('30 minutes'))
    pd.testing.assert_series_equal(result, expected_energy_series)


def test_energy_from_power_upsample_maxtimedelta_not_exceeded():
    times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:30', freq='30T')
    time_series = pd.Series(data=[1.0, 3.0, 5.0, 6.0], index=times)

    expected_result_times = pd.date_range('2018-04-01 12:15', '2018-04-01 13:30', freq='15T')
    expected_energy_series = pd.Series(index=expected_result_times,
                                       data=[0.375, 0.625, 0.875, 1.125, 1.3125, 1.4375],
                                       name='energy_Wh')

    result = energy_from_power(time_series.drop(time_series.index[1]), '15T', pd.to_timedelta('60 minutes'))
    pd.testing.assert_series_equal(result, expected_energy_series)


def test_energy_from_power_upsample_maxtimedelta_exceeded():
    times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:30', freq='30T')
    time_series = pd.Series(data=[1.0, 3.0, 5.0, 6.0], index=times)

    expected_result_times = pd.date_range('2018-04-01 12:15', '2018-04-01 13:30', freq='15T')
    expected_energy_series = pd.Series(index=expected_result_times,
                                       data=[np.nan, np.nan, np.nan, np.nan, 1.3125, 1.4375],
                                       name='energy_Wh')

    result = energy_from_power(time_series.drop(time_series.index[1]), '15T', pd.to_timedelta('30 minutes'))
    pd.testing.assert_series_equal(result, expected_energy_series)


def test_energy_from_power_single_value_input():
    times = pd.date_range('2019-01-01', freq='15T', periods=1)
    power = pd.Series([100.], index=times)
    expected_result = pd.Series([25.], index=times, name='energy_Wh')
    result = energy_from_power(power)
    pd.testing.assert_series_equal(result, expected_result)


def test_energy_from_power_single_value_input_no_freq():
    power = pd.Series([1], pd.date_range('2019-01-01', periods=1, freq='15T'))
    power.index.freq = None
    match = "Could not determine period of input power"
    with pytest.raises(ValueError, match=match):
        energy_from_power(power)


def test_energy_from_power_single_value_with_target():
    times = pd.date_range('2019-01-01', freq='15T', periods=1)
    power = pd.Series([100.], index=times)
    expected_result = pd.Series([100.], index=times, name='energy_Wh')
    result = energy_from_power(power, target_frequency='H')
    pd.testing.assert_series_equal(result, expected_result)
