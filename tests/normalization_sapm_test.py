""" Energy Normalization with SAPM Unit Tests. """

import unittest

import pandas as pd
import numpy as np
import pvlib

from rdtools.normalization import normalize_with_sapm
from rdtools.normalization import sapm_dc_power


class SapmNormalizationTestCase(unittest.TestCase):
    ''' Unit tests for energy normalization module. '''

    def setUp(self):
        # define module constants and parameters
        module = {}
        module['A0'] = 0.0315
        module['A1'] = 0.05975
        module['A2'] = -0.01067
        module['A3'] = 0.0008
        module['A4'] = -2.24e-5
        module['B0'] = 1
        module['B1'] = -0.002438
        module['B2'] = 0.00031
        module['B3'] = -1.246e-5
        module['B4'] = 2.11e-7
        module['B5'] = -1.36e-9
        module['FD'] = 1
        module_parameters = {
            'pdc0': 2.1,
            'gamma_pdc': -0.0045
            }

        # define location
        test_location = pvlib.location\
            .Location(latitude=37.88447702, longitude=-122.2652549)

        self.pvsystem = pvlib.pvsystem\
            .LocalizedPVSystem(location=test_location,
                               surface_tilt=20,
                               surface_azimuth=180,
                               module=module,
                               module_parameters=module_parameters,
                               racking_model='insulated_back_polymerback',
                               modules_per_string=6)

        # define dummy energy data
        energy_freq = 'MS'
        energy_periods = 12
        energy_index = pd.date_range(start='2012-01-01',
                                     periods=energy_periods,
                                     freq=energy_freq)
        dummy_energy = np.repeat(a=100, repeats=energy_periods)
        self.energy = pd.Series(dummy_energy, index=energy_index)
        self.energy_periods = 12

        # define dummy meteorological data
        irrad_columns = ['DNI', 'GHI', 'DHI', 'Temperature', 'Wind Speed']
        irrad_freq = 'D'
        irrad_periods = 31 * energy_periods
        irrad_index = pd.date_range(start='2012-01-01',
                                    periods=irrad_periods,
                                    freq=irrad_freq)
        irrad_index = pd.date_range(start='2012-01-01',
                                    periods=irrad_periods,
                                    freq=irrad_freq)
        self.irrad = pd.DataFrame([[100, 45, 30, 25, 10]],
                                  index=irrad_index,
                                  columns=irrad_columns)

    def tearDown(self):
        pass

    def test_sapm_dc_power(self):
        ''' Test SAPM DC power. '''

        dc_power = sapm_dc_power(self.pvsystem, self.irrad)
        self.assertEqual(self.irrad.index.freq, dc_power.index.freq)
        self.assertEqual(len(self.irrad), len(dc_power))

    def test_normalization_with_sapm(self):
        ''' Test SAPM normalization. '''

        sapm_kws = {
            'pvlib_pvsystem': self.pvsystem,
            'irradiance': self.irrad,
        }

        corr_energy = normalize_with_sapm(self.energy, sapm_kws)

        # Test output is same frequency and length as energy
        self.assertEqual(corr_energy.index.freq, self.energy.index.freq)
        self.assertEqual(len(corr_energy), 13)

        # TODO, test for:
        #     incorrect data format
        #     incomplete data
        #     missing pvsystem metadata
        #     missing measured irradiance data
        #     irradiance freq > energy freq, issue/warining?

if __name__ == '__main__':
    unittest.main()
