"""losses module tests"""

import unittest
import pandas as pd
import numpy as np

from rdtools.losses import calculate_pr, performance_ratio


class PRTestCase(unittest.TestCase):
    """
    Unit tests for performance ratio functions
    """

    def setUp(self):
        st = '2019-01-01 10:00'
        ed = '2019-01-01 15:00'
        kWdc = 100.0
        gamma = -0.004

        idx = pd.date_range(st, ed, freq='15min')
        poa = pd.Series(index=idx, data=500)
        tcell = pd.Series(index=idx, data=30)
        power = kWdc * poa/1000 * (1 + gamma*(tcell - 25))

        self.kWdc = kWdc
        self.gamma = gamma
        self.poa = poa
        self.tcell = tcell
        self.power = power
        self.args = {
            'system_size': kWdc,
            'gamma_pdc': gamma,
            'power': power,
            'poa': poa,
            'tcell': tcell
        }

    def test_calculate_pr_simple(self):
        """test base case"""
        pr = calculate_pr(self.power, self.power)
        self.assertAlmostEqual(pr, 1.0)
        pr = calculate_pr(0.5 * self.power, self.power)
        self.assertAlmostEqual(pr, 0.5)

    def test_calculate_pr_freq(self):
        """test `freq` parameter to correctly roll-up results"""
        power = self.power.copy()
        power[:4] *= 0.25
        pr = calculate_pr(power, self.power, freq='h')
        self.assertTrue(hasattr(pr, 'index'))
        self.assertEqual(pr.index.freqstr, 'H')
        self.assertAlmostEqual(pr[0], 0.25)
        self.assertAlmostEqual(pr[1:].mean(), 1.0)

    def test_calculate_pr_filt(self):
        """test `filt` parameter to correctly filter out data"""
        # derate performance for some times, but filter them out so that
        # returned PR is still 100%
        power = self.power.copy()
        power[:5] *= 0.5
        filt = power == power.max()
        pr = calculate_pr(power, self.power, filt=filt)
        self.assertAlmostEqual(pr, 1.0)

    def test_calculate_pr_filter_na(self):
        """test `filter_na` to remove nan (True) or set to zero (False)"""
        # null out first hour, plus first 15min of second hour
        power = self.power.copy()
        power[:5] = np.nan
        pr = calculate_pr(power, self.power, freq='h', filter_na=True)
        self.assertTrue(np.isnan(pr[0]))
        self.assertAlmostEqual(pr[1], 1.0)

        pr = calculate_pr(power, self.power, freq='h', filter_na=False)
        self.assertTrue(pr[0] == 0)
        self.assertAlmostEqual(pr[1], 0.75)

    def test_performance_ratio_simple(self):
        """test base case"""
        pr = performance_ratio(**self.args)
        self.assertEqual(pr, 1.0)

        args = self.args.copy()
        args['power'] *= 0.5
        pr = performance_ratio(**args)
        self.assertEqual(pr, 0.5)

    def test_performance_ratio_gamma(self):
        """test qualitative behavior of gamma"""
        # make gamma more negative.  since Tcell = 30, this will bias P_exp low
        args = self.args.copy()
        args['gamma_pdc'] = -0.005
        pr = performance_ratio(**args)
        self.assertTrue(pr > 1.0)

    def test_performance_ratio_freq(self):
        """test `freq' parameter to correctly roll-up results"""
        # derate first hour to have poor performance
        args = self.args.copy()
        args['power'][:4] *= 0.25
        pr = performance_ratio(**args, freq='h')
        self.assertTrue(hasattr(pr, 'index'))
        self.assertEqual(pr.index.freqstr, 'H')
        self.assertAlmostEqual(pr[0], 0.25)
        self.assertAlmostEqual(pr[1:].mean(), 1.0)

    def test_performance_ratio_clip_limit(self):
        """test `clip_limit` to filter when system should be clipping"""
        args = self.args.copy()
        # set first timestamp to high-irradiance (should clip) and no power.
        # if PR is still 100% then it filtered the point for clipping
        args['poa'][0] = 1000
        args['power'][0] = 0
        pr = performance_ratio(**args, clip_limit=75)
        self.assertAlmostEqual(pr, 1.0)
        pr = performance_ratio(**args)
        self.assertTrue(pr < 1.0)

    def test_performance_ratio_low_light_limit(self):
        """test `low_light_limit` parameter to remove"""
        args = self.args.copy()
        # set first timestamp to low-irradiance (should filter) and no power.
        # if PR is still 100% then it filtered the point for low-light
        args['poa'][0] = 50
        args['power'][0] = 0
        pr = performance_ratio(**args, low_light_limit=100)
        self.assertAlmostEqual(pr, 1.0)
        pr = performance_ratio(**args)
        self.assertTrue(pr < 1.0)

    def test_performance_ratio_tcell_ref(self):
        """test `tcell_ref` parameter"""
        # make tcell_ref = 0 to bias P_exp low
        pr = performance_ratio(**self.args, tcell_ref=0)
        self.assertTrue(pr > 1.0)
        pr = performance_ratio(**self.args, tcell_ref=50)
        self.assertTrue(pr < 1.0)


if __name__ == "__main__":
    unittest.main()
