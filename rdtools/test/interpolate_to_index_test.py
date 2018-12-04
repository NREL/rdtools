import pandas as pd
import numpy as np
from rdtools import interpolate_to_index
import pytest


@pytest.fixture
def time_series():
    times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:15', freq='15T')
    time_series = pd.Series(data=[9, 6, 3, 3, 6, 9], index=times, name='foo')
    time_series = time_series.drop(times[4])
    return time_series


@pytest.fixture
def target_index():
    return pd.date_range('2018-04-01 12:00', '2018-04-01 13:15', freq='20T')


@pytest.fixture
def expected_result(target_index, time_series):
    return pd.Series(data=[9.0, 5.0, 3.0, np.nan], index=target_index, name=time_series.name)


def test_interpolate_to_index_calculation(time_series, target_index, expected_result):

    interpolated = interpolate_to_index(time_series, target_index, pd.to_timedelta('15 minutes'))
    pd.testing.assert_series_equal(interpolated, expected_result)


def test_interpolate_to_index_two_argument(time_series, target_index, expected_result):

    # Test that a warning is raised when max_timedelta is omitted
    with pytest.warns(UserWarning):
        interpolated = interpolate_to_index(time_series, target_index)
    pd.testing.assert_series_equal(interpolated, expected_result)


def test_interpolate_to_index_tz_validation(time_series, target_index, expected_result):
    with pytest.raises(ValueError):
        interpolate_to_index(time_series, target_index.tz_localize('UTC'), pd.to_timedelta('15 minutes'))

    time_series = time_series.copy()
    time_series.index = time_series.index.tz_localize('UTC')

    with pytest.raises(ValueError):
        interpolate_to_index(time_series, target_index, pd.to_timedelta('15 minutes'))


def test_interpolate_to_index_same_tz(time_series, target_index, expected_result):
    time_series = time_series.copy()
    expected_result = expected_result.copy()

    time_series.index = time_series.index.tz_localize('America/Denver')
    target_index = target_index.tz_localize('America/Denver')
    expected_result.index = expected_result.index.tz_localize('America/Denver')

    interpolated = interpolate_to_index(time_series, target_index, pd.to_timedelta('15 minutes'))
    pd.testing.assert_series_equal(interpolated, expected_result)


def test_interpolate_to_index_different_tz(time_series, target_index, expected_result):
    time_series = time_series.copy()
    expected_result = expected_result.copy()

    time_series.index = time_series.index.tz_localize('America/Denver').tz_convert('UTC')
    target_index = target_index.tz_localize('America/Denver')
    expected_result.index = expected_result.index.tz_localize('America/Denver')

    interpolated = interpolate_to_index(time_series, target_index, pd.to_timedelta('15 minutes'))
    pd.testing.assert_series_equal(interpolated, expected_result)
