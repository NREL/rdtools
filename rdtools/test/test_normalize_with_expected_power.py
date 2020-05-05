import pandas as pd
import pytest
from rdtools.normalization import normalize_with_expected_power
from pandas import Timestamp
import numpy as np

@pytest.fixture()
def times_15():
	return pd.date_range(start='20200101 12:00', end= '20200101 13:00', freq='15T')

@pytest.fixture()
def times_30():
	return pd.date_range(start='20200101 12:00', end= '20200101 13:00', freq='30T')

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
	norm, insol = normalize_with_expected_power(pv_15, expected_15, irradiance_15)
	expected_norm = pd.Series({Timestamp('2020-01-01 12:15:00', freq='15T'): 1.0,
							   Timestamp('2020-01-01 12:30:00', freq='15T'): 1.0784313725490198,
                               Timestamp('2020-01-01 12:45:00', freq='15T'): 1.0612244897959184,
                               Timestamp('2020-01-01 13:00:00', freq='15T'): 1.0487804878048783})
	expected_norm.name = 'energy_Wh'

	expected_insol = pd.Series({Timestamp('2020-01-01 12:15:00', freq='15T'): 231.25,
                                Timestamp('2020-01-01 12:30:00', freq='15T'): 225.0,
                                Timestamp('2020-01-01 12:45:00', freq='15T'): 240.625,
                                Timestamp('2020-01-01 13:00:00', freq='15T'): 233.125})
	expected_insol.name = 'energy_Wh'

	pd.testing.assert_series_equal(norm, expected_norm)
	pd.testing.assert_series_equal(insol, expected_insol)

def test_normalize_with_expected_power_energy_option(pv_15, expected_15, irradiance_15):
	norm, insol = normalize_with_expected_power(pv_15, expected_15, irradiance_15, pv_input='energy')
	expected_norm = pd.Series({Timestamp('2020-01-01 12:00:00', freq='15T'): np.nan,
                               Timestamp('2020-01-01 12:15:00', freq='15T'): 5.714285714285714,
                               Timestamp('2020-01-01 12:30:00', freq='15T'): 4.705882352941177,
                               Timestamp('2020-01-01 12:45:00', freq='15T'): 3.5918367346938775,
                               Timestamp('2020-01-01 13:00:00', freq='15T'): 4.097560975609756})
	expected_norm.name = 'energy_Wh'

	expected_insol = pd.Series({Timestamp('2020-01-01 12:15:00', freq='15T'): 231.25,
 								Timestamp('2020-01-01 12:30:00', freq='15T'): 225.0,
 								Timestamp('2020-01-01 12:45:00', freq='15T'): 240.625,
 								Timestamp('2020-01-01 13:00:00', freq='15T'): 233.125})
	expected_insol.name = 'energy_Wh'

	pd.testing.assert_series_equal(norm, expected_norm)
	pd.testing.assert_series_equal(insol, expected_insol)

def test_normalize_with_expected_power_low_freq_pv(pv_30, expected_15, irradiance_15):
	norm, insol = normalize_with_expected_power(pv_30, expected_15, irradiance_15)

	expected_norm = pd.Series({Timestamp('2020-01-01 12:30:00', freq='30T'): 0.9302325581395349,
                               Timestamp('2020-01-01 13:00:00', freq='30T'): 1.1333333333333333})
	expected_norm.name = 'energy_Wh'
	expected_insol = pd.Series({Timestamp('2020-01-01 12:30:00', freq='30T'): 456.25,
 	                            Timestamp('2020-01-01 13:00:00', freq='30T'): 473.75})
	expected_insol.name = 'energy_Wh'

	pd.testing.assert_series_equal(norm, expected_norm)
	pd.testing.assert_series_equal(insol, expected_insol)

def test_normalized_with_expected_power_low_freq_expected(pv_15, expected_30, irradiance_30):
	norm, insol = normalize_with_expected_power(pv_15, expected_30, irradiance_30)

	expected_norm = pd.Series({Timestamp('2020-01-01 12:15:00', freq='15T'): 1.09375,
                               Timestamp('2020-01-01 12:30:00', freq='15T'): 1.1458333333333335,
                               Timestamp('2020-01-01 12:45:00', freq='15T'): 1.0000000000000002,
                               Timestamp('2020-01-01 13:00:00', freq='15T'): 0.9772727272727274})
	expected_norm.name = 'energy_Wh'

	expected_insol = pd.Series({Timestamp('2020-01-01 12:15:00', freq='15T'): 246.875,
			                    Timestamp('2020-01-01 12:30:00', freq='15T'): 240.625,
                                Timestamp('2020-01-01 12:45:00', freq='15T'): 233.75,
                                Timestamp('2020-01-01 13:00:00', freq='15T'): 226.25})
	expected_insol.name = 'energy_Wh'
	
	pd.testing.assert_series_equal(norm, expected_norm)
	pd.testing.assert_series_equal(insol, expected_insol)






