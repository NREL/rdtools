""" Energy Normalization with PVWatts Unit Tests. """

import unittest

import pandas as pd
import numpy as np

from rdtools.normalization import normalize_with_pvwatts
from rdtools.normalization import pvwatts_dc_power

class PVWattsNormalizationTestCase(unittest.TestCase):
    ''' Unit tests for energy normalization module. '''

    def setUp(self):
        # define module constants and parameters

        # define dummy power data
        power_index = pd.date_range(start='2012-01-01',
                                    periods=12,
                                    freq='MS')
        dummy_power = np.repeat(a=50, repeats=12)
        self.power = pd.Series(dummy_power, index=power_index)

        # define dummy irradiance
        irrad_index = pd.date_range(start='2012-01-01',
                                    periods=12,
                                    freq='MS')
        dummy_irrad = np.repeat(a=400, repeats=12)
        self.poa_global = pd.Series(dummy_irrad, index=irrad_index)

        # define dummy temperature data
        temp_index = pd.date_range(start='2012-01-01',
                                   periods=12,
                                   freq='MS')
        dummy_temp = np.repeat(a=30, repeats=12)
        self.temp = pd.Series(dummy_temp, index=temp_index)

        # define dummy energy data
        energy_index = pd.date_range(start='2012-01-01',
                                     periods=12,
                                     freq='MS')
        power_meas = 19.75  # power in dummy conditions
        hours = (energy_index - energy_index.shift(-1)).astype('int64') / (10.0**9 * 3600.0)
        dummy_energy = hours * power_meas
        self.energy = pd.Series(dummy_energy, index=energy_index)

        # define gamma_pdc for pvw temperature factor
        self.gamma_pdc = -0.0025

        # define an irregular pandas series
        times = pd.DatetimeIndex(['2012-01-01 12:00', '2012-01-01 12:05', '2012-01-01 12:06',
                                 '2012-01-01 12:09'])
        data = [1, 2, 3, 4]
        self.irregular_timeseries = pd.Series(data=data, index=times)

    def tearDown(self):
        pass

    def test_pvwatts_dc_power(self):
        ''' Test PVWatts DC power caculation. '''

        dc_power = pvwatts_dc_power(self.poa_global, self.power,
                                    temperature_cell=self.temp,
                                    gamma_pmp=self.gamma_pdc)

        # Assert output has same frequency and length as input
        self.assertEqual(self.poa_global.index.freq, dc_power.index.freq)
        self.assertEqual(len(self.poa_global), len(dc_power))

        # Assert value of output Series is equal to value expected
        self.assertTrue((dc_power == 19.75).all())

    def test_normalization_with_pvw(self):
        ''' Test PVWatts normalization. '''

        pvw_kws = {
            'poa_global': self.poa_global,
            'power_dc_rated': self.power,
            'temperature_cell': self.temp,
            'gamma_pmp': self.gamma_pdc,
        }

        corr_energy, insolation = normalize_with_pvwatts(self.energy, pvw_kws)

        # Test output is same frequency and length as energy
        self.assertEqual(corr_energy.index.freq, self.energy.index.freq)
        self.assertEqual(len(corr_energy), 12)

        # Test corrected energy is equal to 1.0
        # first value should be nan because we have no irradiance
        # data prior to the first energy point
        self.assertTrue(np.isnan(corr_energy.iloc[0]))
        self.assertTrue((corr_energy.iloc[1:] == 1.0).all())  # rest should be 1

        # Test expected behavior when energy has no explicit frequency
        self.energy.index.freq = None
        corr_energy, insolation = normalize_with_pvwatts(self.energy, pvw_kws)
        self.assertTrue(np.isnan(corr_energy.iloc[0]))  # first value should be nan
        self.assertTrue((corr_energy.iloc[1:] == 1.0).all())  # rest should be 1

        # Test for valueError when energy frequency can't be inferred
        with self.assertRaises(ValueError):
            corr_energy, insolation = normalize_with_pvwatts(self.irregular_timeseries, pvw_kws)


if __name__ == '__main__':
    unittest.main()
