import pandas as pd
import numpy as np
from rdtools import interpolate
import pytest


@pytest.fixture
def time_series():
    times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:15', freq='15T')
    time_series = pd.Series(data=[9, 6, 3, 3, 6, 9], index=times, name='foo')
    time_series = time_series.drop(times[4])
    return time_series


@pytest.fixture
def target_index(time_series):
    return pd.date_range(time_series.index.min(), time_series.index.max(), freq='20T')


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
    col0 = test_df.columns[0]
    col1 = test_df.columns[1]
    expected_df_result = pd.DataFrame({
        col0: [6.0, 3.0, np.nan, 9.0],
        col1: [np.nan, 8.0, 4.0, 3.0]
    }, index=df_target_index)

    expected_df_result = expected_df_result[test_df.columns]
    return expected_df_result


def test_interpolate_freq_specification(time_series, target_index, expected_series):
    # test the string specification
    interpolated = interpolate(time_series, target_index.freq.freqstr, pd.to_timedelta('15 minutes'),
                               warning_threshold=0.21)
    pd.testing.assert_series_equal(interpolated, expected_series)

    # test the DateOffset specification
    interpolated = interpolate(time_series, target_index.freq, pd.to_timedelta('15 minutes'),
                               warning_threshold=0.21)
    pd.testing.assert_series_equal(interpolated, expected_series)


def test_interpolate_calculation(time_series, target_index, expected_series):

    interpolated = interpolate(time_series, target_index, pd.to_timedelta('15 minutes'),
                               warning_threshold=0.21)
    pd.testing.assert_series_equal(interpolated, expected_series)


def test_interpolate_two_argument(time_series, target_index, expected_series):

    expected_series.iloc[-1] = 6.0
    interpolated = interpolate(time_series, target_index)
    pd.testing.assert_series_equal(interpolated, expected_series)


def test_interpolate_tz_validation(time_series, target_index, expected_series):
    with pytest.raises(ValueError):
        interpolate(time_series, target_index.tz_localize('UTC'), pd.to_timedelta('15 minutes'))

    time_series = time_series.copy()
    time_series.index = time_series.index.tz_localize('UTC')

    with pytest.raises(ValueError):
        interpolate(time_series, target_index, pd.to_timedelta('15 minutes'))


def test_interpolate_same_tz(time_series, target_index, expected_series):
    time_series = time_series.copy()
    expected_series = expected_series.copy()

    time_series.index = time_series.index.tz_localize('America/Denver')
    target_index = target_index.tz_localize('America/Denver')
    expected_series.index = expected_series.index.tz_localize('America/Denver')

    interpolated = interpolate(time_series, target_index, pd.to_timedelta('15 minutes'),
                               warning_threshold=0.21)
    pd.testing.assert_series_equal(interpolated, expected_series)


def test_interpolate_different_tz(time_series, target_index, expected_series):
    time_series = time_series.copy()
    expected_series = expected_series.copy()

    time_series.index = time_series.index.tz_localize('America/Denver').tz_convert('UTC')
    target_index = target_index.tz_localize('America/Denver')
    expected_series.index = expected_series.index.tz_localize('America/Denver')

    interpolated = interpolate(time_series, target_index, pd.to_timedelta('15 minutes'),
                               warning_threshold=0.21)
    pd.testing.assert_series_equal(interpolated, expected_series)


def test_interpolate_dataframe(test_df, df_target_index, df_expected_result):
    interpolated = interpolate(test_df, df_target_index, pd.to_timedelta('15 minutes'),
                               warning_threshold=0.21)
    pd.testing.assert_frame_equal(interpolated, df_expected_result)


def test_interpolate_warning(test_df, df_target_index, df_expected_result):
    N = len(test_df)
    all_idx = list(range(N))
    # drop every other value in the first third of the dataset
    index_with_gaps = all_idx[:N//3][::2] + all_idx[N//3:]
    test_df = test_df.iloc[index_with_gaps, :]
    with pytest.warns(UserWarning):
        interpolate(test_df, df_target_index, pd.to_timedelta('15 minutes'),
                    warning_threshold=0.1)

    with pytest.warns(None) as record:
        interpolate(test_df, df_target_index, pd.to_timedelta('15 minutes'),
                    warning_threshold=0.5)
        if record:
            pytest.fail("normalize.interpolate raised a warning about "
                        "excluded data even though the threshold was high")
