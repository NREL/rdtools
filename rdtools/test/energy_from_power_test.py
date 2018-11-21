import pandas as pd
from rdtools import energy_from_power
import pytest


def test_energy_from_power_calculation():
    power_times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:00', freq='15T')
    result_times = power_times[1:]
    power_series = pd.Series(data=4.0, index=power_times)
    expected_energy_series = pd.Series(data=1.0, index=result_times)

    result = energy_from_power(power_series, 0.25)

    pd.testing.assert_series_equal(result, expected_energy_series)


def test_energy_from_power_max_interval():
    power_times = pd.date_range('2018-04-01 12:00', '2018-04-01 13:00', freq='15T')
    result_times = power_times[1:]
    power_series = pd.Series(data=4.0, index=power_times)
    expected_energy_series = pd.Series(data=1.0, index=result_times)

    result = energy_from_power(power_series, 0.1)

    # We expect an empty series, because max_interval_hours is smaller than the
    # time step of the power time series
    pd.testing.assert_series_equal(result, expected_energy_series[[False] * 4])


def test_energy_from_power_validation():
    power_series = pd.Series(data=[4.0] * 4)
    with pytest.raises(ValueError):
        energy_from_power(power_series, 0.25)
