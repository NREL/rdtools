""" Degradation Module Tests. """

import unittest

import pandas as pd
import numpy as np

from rdtools import degradation_ols


class DegradationTestCase(unittest.TestCase):
    ''' Unit tests for degradation module. '''

    def setUp(self):
        # define module constants and parameters

        x = pd.date_range(start='2012-01-01', end='2012-03-01', freq='MS')
        N = len(x)
        months = np.arange(N)
        self.rd = -0.005
        y = np.ones(N) * np.power(1 + self.rd/12, months)
        self.test_corr_energy = pd.Series(data=y, index=x)

    def tearDown(self):
        pass

    def test_degradation_ols(self):
        ''' Test degradation with ols. '''

        # test ols degradation calc
        rd_result = degradation_ols(self.test_corr_energy)
        self.assertAlmostEqual(rd_result['Rd_pct'], 100*self.rd, places=3)

        # TODO
        # - support for different time series frequencies
        # - inputs

if __name__ == '__main__':
    unittest.main()
