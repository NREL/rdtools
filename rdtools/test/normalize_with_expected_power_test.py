import pandas as pd
import pytest
from rdtools.normalization import normalize_with_expected_power


@pytest.fixture()
def times_15():
    return pd.date_range(start="20200101 12:00", end="20200101 13:00", freq="15min")


@pytest.fixture()
def times_30():
    return pd.date_range(start="20200101 12:00", end="20200101 13:00", freq="30min")


@pytest.fixture()
def pv_15(times_15):
    return pd.Series([1.0, 2.5, 3.0, 2.2, 2.1], index=times_15)


@pytest.fixture()
def expected_15(times_15):
    return pd.Series([1.2, 2.3, 2.8, 2.1, 2.0], index=times_15)


@pytest.fixture()
def irradiance_15(times_15):
    return pd.Series([1000.0, 850.0, 950.0, 975.0, 890.0], index=times_15)


@pytest.fixture()
def pv_30(times_30):
    return pd.Series([1.0, 3.0, 2.1], index=times_30)


@pytest.fixture()
def expected_30(times_30):
    return pd.Series([1.2, 2.8, 2.0], index=times_30)


@pytest.fixture()
def irradiance_30(times_30):
    return pd.Series([1000.0, 950.0, 890.0], index=times_30)


def test_normalize_with_expected_power_uniform_frequency(pv_15, expected_15, irradiance_15):
    norm, insol = normalize_with_expected_power(
        pv_15, expected_15, irradiance_15)

    expected_norm = pv_15.iloc[1:]/expected_15.iloc[1:]
    expected_norm.name = 'energy_Wh'

    expected_insol = irradiance_15.iloc[1:]*0.25
    expected_insol.name = 'energy_Wh'

    pd.testing.assert_series_equal(norm, expected_norm)
    pd.testing.assert_series_equal(insol, expected_insol)


def test_normalize_with_expected_power_energy_option(pv_15, expected_15, irradiance_15):
    norm, insol = normalize_with_expected_power(
        pv_15, expected_15, irradiance_15, pv_input='energy')

    expected_norm = pv_15/(0.25*expected_15.iloc[1:])
    expected_norm.name = 'energy_Wh'

    expected_insol = irradiance_15.iloc[1:]*0.25
    expected_insol = expected_insol.reindex(expected_norm.index)
    expected_insol.name = 'energy_Wh'

    pd.testing.assert_series_equal(norm, expected_norm)
    pd.testing.assert_series_equal(insol, expected_insol)


def test_normalize_with_expected_power_low_freq_pv(pv_30, expected_15, irradiance_15):
    norm, insol = normalize_with_expected_power(
        pv_30, expected_15, irradiance_15)

    pv_energy = pv_30.iloc[1:]*0.5
    expected_energy = expected_15.iloc[1:]*0.25
    # aggregate to 30 min level
    expected_energy = expected_energy.rolling(2).sum()
    expected_energy = expected_energy.reindex(pv_energy.index)
    expected_norm = pv_energy/expected_energy
    expected_norm.name = 'energy_Wh'

    expected_insol = irradiance_15.iloc[1:]*0.25
    # aggregate to 30 min level
    expected_insol = expected_insol.rolling(2).sum()
    expected_insol = expected_insol.reindex(pv_energy.index)
    expected_insol.name = 'energy_Wh'

    pd.testing.assert_series_equal(norm, expected_norm)
    pd.testing.assert_series_equal(insol, expected_insol)


def test_normalized_with_expected_power_low_freq_expected(pv_15, expected_30, irradiance_30):
    norm, insol = normalize_with_expected_power(
        pv_15, expected_30, irradiance_30)

    expected_15 = expected_30.reindex(pv_15.index).interpolate()
    expected_energy = expected_15.iloc[1:]*0.25
    expected_norm = pv_15.iloc[1:]*0.25/expected_energy
    expected_norm.name = 'energy_Wh'

    irradiance_15 = irradiance_30.reindex(pv_15.index).interpolate()
    expected_insol = irradiance_15.iloc[1:]*0.25
    expected_insol.name = 'energy_Wh'

    pd.testing.assert_series_equal(norm, expected_norm)
    pd.testing.assert_series_equal(insol, expected_insol)
