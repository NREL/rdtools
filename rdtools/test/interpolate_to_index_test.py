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
def expected_series(target_index, time_series):
    return pd.Series(data=[9.0, 5.0, 3.0, np.nan], index=target_index, name=time_series.name)


@pytest.fixture
def test_df(time_series):
    time_series1 = time_series.copy()
    time_series2 = time_series.copy()

    time_series2.index = time_series2.index + pd.to_timedelta('30 minutes')
    time_series2.name = 'bar'

    test_df = pd.concat([time_series1, time_series2], axis=1)

    return test_df


@pytest.fixture
def df_target_index(target_index):
    return target_index + pd.to_timedelta('15 minutes')


@pytest.fixture
def df_expected_result(df_target_index, test_df):
    expected_df_result = pd.DataFrame({
        test_df.columns[0]: [6.0, 3.0, np.nan, 9.0],
        test_df.columns[1]: [np.nan, 8.0, 4.0, 3.0]
    }, index=df_target_index)

    return expected_df_result


def test_interpolate_to_index_calculation(time_series, target_index, expected_series):

    interpolated = interpolate_to_index(time_series, target_index, pd.to_timedelta('15 minutes'))
    pd.testing.assert_series_equal(interpolated, expected_series)


def test_interpolate_to_index_two_argument(time_series, target_index, expected_series):

    # Test that a warning is raised when max_timedelta is omitted
    with pytest.warns(UserWarning):
        interpolated = interpolate_to_index(time_series, target_index)
    pd.testing.assert_series_equal(interpolated, expected_series)


def test_interpolate_to_index_tz_validation(time_series, target_index, expected_series):
    with pytest.raises(ValueError):
        interpolate_to_index(time_series, target_index.tz_localize('UTC'), pd.to_timedelta('15 minutes'))

    time_series = time_series.copy()
    time_series.index = time_series.index.tz_localize('UTC')

    with pytest.raises(ValueError):
        interpolate_to_index(time_series, target_index, pd.to_timedelta('15 minutes'))


def test_interpolate_to_index_same_tz(time_series, target_index, expected_series):
    time_series = time_series.copy()
    expected_series = expected_series.copy()

    time_series.index = time_series.index.tz_localize('America/Denver')
    target_index = target_index.tz_localize('America/Denver')
    expected_series.index = expected_series.index.tz_localize('America/Denver')

    interpolated = interpolate_to_index(time_series, target_index, pd.to_timedelta('15 minutes'))
    pd.testing.assert_series_equal(interpolated, expected_series)


def test_interpolate_to_index_different_tz(time_series, target_index, expected_series):
    time_series = time_series.copy()
    expected_series = expected_series.copy()

    time_series.index = time_series.index.tz_localize('America/Denver').tz_convert('UTC')
    target_index = target_index.tz_localize('America/Denver')
    expected_series.index = expected_series.index.tz_localize('America/Denver')

    interpolated = interpolate_to_index(time_series, target_index, pd.to_timedelta('15 minutes'))
    pd.testing.assert_series_equal(interpolated, expected_series)


def test_interpolate_to_index_dataframe(test_df, df_target_index, df_expected_result):
    interpolated = interpolate_to_index(test_df, df_target_index, pd.to_timedelta('15 minutes'))
    pd.testing.assert_frame_equal(interpolated, df_expected_result)
