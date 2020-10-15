import pandas as pd
import numpy as np
from rdtools import energy_from_power
import pytest


@pytest.fixture
def times():
    return pd.date_range(start='20200101 12:00', end='20200101 13:00', freq='15T')


@pytest.fixture
def power(times):
    return pd.Series([1.0, 2.0, 3.0, 2.0, 1.0], index=times)


def test_energy_from_power_single_arg(power):
    expected = power.iloc[1:]*0.25
    expected.name = 'energy_Wh'
    result = energy_from_power(power)
    pd.testing.assert_series_equal(result, expected)


def test_energy_from_power_instantaneous(power):
    expected = (0.25*(power + power.shift())/2).dropna()
    expected.name = 'energy_Wh'
    result = energy_from_power(power, power_type='instantaneous')
    pd.testing.assert_series_equal(result, expected)


def test_energy_from_power_max_timedelta_inference(power):
    expected = power.iloc[1:]*0.25
    expected.name = 'energy_Wh'
    expected.iloc[:2] = np.nan
    result = energy_from_power(power.drop(power.index[1]))
    pd.testing.assert_series_equal(result, expected)


def test_energy_from_power_max_timedelta(power):
    expected = power.iloc[1:]*0.25
    expected.name = 'energy_Wh'
    result = energy_from_power(power.drop(power.index[1]),
                               max_timedelta=pd.to_timedelta('30 minutes'))
    pd.testing.assert_series_equal(result, expected)


def test_energy_from_power_upsample(power):
    expected = power.resample('10T').asfreq().interpolate()/6
    expected = expected.iloc[1:]
    expected.name = 'energy_Wh'
    result = energy_from_power(power, target_frequency='10T')
    pd.testing.assert_series_equal(result, expected)


def test_energy_from_power_downsample(power):
    expected = power.resample('20T').asfreq()
    expected = expected.iloc[1:]
    expected = pd.Series([0.75, 0.833333333, 0.416666667], index=expected.index)
    expected.name = 'energy_Wh'
    result = energy_from_power(power, target_frequency='20T')
    pd.testing.assert_series_equal(result, expected)


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
