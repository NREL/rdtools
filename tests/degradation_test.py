""" Degradation Module Tests. """

import unittest
import sys

import pandas as pd
import numpy as np

from rdtools import degradation_ols, degradation_classical_decomposition, degradation_year_on_year


class DegradationTestCase(unittest.TestCase):
    ''' Unit tests for degradation module.'''

    @classmethod
    def get_corr_energy(cls, rd, input_freq):

        #lock seed to make test deterministic
        np.random.seed(0)

        daily_rd = rd / 365.0

        start = '2012-01-01'
        if input_freq == 'S':
            end = '2012-03-01'
        else:
            end = '2015-01-01'

        if input_freq == 'Irregular_D':
            freq = 'D'
        else:
            freq = input_freq

        x = pd.date_range(start=start, end=end, freq=freq)
        day_deltas = (x - x[0]).astype('timedelta64[s]') / (60.0 * 60.0 * 24)
        noise = (np.random.rand(len(day_deltas)) - 0.5) / 1e3

        y = 1 + daily_rd * day_deltas + noise

        corr_energy = pd.Series(data=y, index=x)

        if input_freq == 'Irregular_D':
            corr_energy = corr_energy.sample(frac=0.8, replace=False)
            corr_energy = corr_energy.sort_index()

        return corr_energy

    @classmethod
    def setUpClass(cls):
        super(DegradationTestCase, cls).setUpClass()
        # define module constants and parameters

        # All frequencies
        cls.list_all_input_freq = ['MS', 'M', 'W',
                                   'D', 'H', 'T', 'S', 'Irregular_D']

        # Allowed frequencies for degradation_ols
        cls.list_ols_input_freq = ['MS', 'M', 'W',
                                   'D', 'H', 'T', 'S', 'Irregular_D']

        '''
        Allowed frequencies for degradation_classical_decomposition
        in principle CD works on higher frequency data but that makes the
        tests painfully slow
        '''
        cls.list_CD_input_freq = ['MS', 'M', 'W', 'D']

        # Allowed frequencies for degradation_year_on_year
        cls.list_YOY_input_freq = ['MS', 'M', 'W', 'D', 'Irregular_D']

        cls.rd = -0.005

        test_corr_energy = {}

        for input_freq in cls.list_all_input_freq:
            corr_energy = cls.get_corr_energy(cls.rd, input_freq)
            test_corr_energy[input_freq] = corr_energy

        cls.test_corr_energy = test_corr_energy

    def test_degradation_with_ols(cls):
        ''' Test degradation with ols. '''

        funcName = sys._getframe().f_code.co_name
        print('\r', 'Running ', funcName)

        # test ols degradation calc
        for input_freq in cls.list_ols_input_freq:
            print('Frequency: ', input_freq)
            rd_result = degradation_ols(cls.test_corr_energy[input_freq])
            cls.assertAlmostEqual(rd_result[0], 100 * cls.rd, places=1)
            print('Actual: ', 100 * cls.rd)
            print('Estimated: ', rd_result[0])

    def test_degradation_classical_decomposition(cls):
        ''' Test degradation with classical decomposition. '''

        funcName = sys._getframe().f_code.co_name
        print('\r', 'Running ', funcName)

        # test classical decomposition degradation calc
        for input_freq in cls.list_CD_input_freq:
            print('Frequency: ', input_freq)
            rd_result = degradation_classical_decomposition(
                cls.test_corr_energy[input_freq])
            cls.assertAlmostEqual(rd_result[0], 100 * cls.rd, places=1)
            print('Actual: ', 100 * cls.rd)
            print('Estimated: ', rd_result[0])

    def test_degradation_year_on_year(cls):
        ''' Test degradation with year on year approach. '''

        funcName = sys._getframe().f_code.co_name
        print('\r', 'Running ', funcName)

        # test YOY degradation calc
        for input_freq in cls.list_YOY_input_freq:
            print('Frequency: ', input_freq)
            rd_result = degradation_year_on_year(
                cls.test_corr_energy[input_freq])
            cls.assertAlmostEqual(rd_result[0], 100 * cls.rd, places=1)
            print('Actual: ', 100 * cls.rd)
            print('Estimated: ', rd_result[0])

    def test_confidence_intervals(cls):

        funcName = sys._getframe().f_code.co_name
        print('\r', 'Running ', funcName)

        input_freq = "W"

        for func in [degradation_ols, degradation_year_on_year]:

            ci1 = 68.2
            ci2 = 95
            r1 = func(cls.test_corr_energy[input_freq], confidence_level=ci1)
            r2 = func(cls.test_corr_energy[input_freq], confidence_level=ci2)

            print("func: {}, ci: {}, ({}) {} ({})".format(str(func).split(' ')[1], ci1, r1[1][0], r1[0], r1[1][1]))
            print("func: {}, ci: {}, ({}) {} ({})".format(str(func).split(' ')[1], ci2, r2[1][0], r2[0], r2[1][1]))

            # lower limit is lower than median and upper limit is higher than median
            assert r1[0] > r1[1][0] and r1[0] < r1[1][1]
            assert r2[0] > r2[1][0] and r2[0] < r2[1][1]

            # 95% interval is bigger than 68% interval
            assert abs(r1[0] - r1[1][1]) < abs(r2[0] - r2[1][1])
            assert abs(r1[0] - r1[1][0]) < abs(r2[0] - r2[1][0])

            # actual rd is within confidence interval
            assert 100.0 * cls.rd > r2[1][0] and 100.0 * cls.rd < r2[1][1]


if __name__ == '__main__':
    unittest.main()
