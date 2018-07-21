""" Filtering Module Tests. """

import unittest

import pandas as pd
import numpy as np

from rdtools import csi_filter, poa_filter, tcell_filter, clip_filter


class CSIFilterTestCase(unittest.TestCase):
    ''' Unit tests for clear sky index filter.'''

    def setUp(self):
        self.measured_poa = np.array([1, 1, 0, 1.15, 0.85])
        self.clearsky_poa = np.array([1, 2, 1, 1.00, 1.00])

    def test_csi_filter(self):
        filtered = csi_filter(self.measured_poa,
                              self.clearsky_poa,
                              threshold=0.15)

        # Expect clearsky index is filtered with threshold of +/- 0.15.
        expected_result = np.array([True, False, False, True, True])
        self.assertListEqual(filtered.tolist(), expected_result.tolist())


class POAFilterTestCase(unittest.TestCase):
    ''' Unit tests for plane of array insolation filter.'''

    def setUp(self):
        self.measured_poa = np.array([201, 1199, 500, 200, 1200])

    def test_poa_filter(self):
        filtered = poa_filter(self.measured_poa,
                              low_irradiance_cutoff=200,
                              high_irradiance_cutoff=1200)

        # Expect high and low POA cutoffs to be non-inclusive.
        expected_result = np.array([True, True, True, False, False])
        self.assertListEqual(filtered.tolist(), expected_result.tolist())


class TcellFilterTestCase(unittest.TestCase):
    ''' Unit tests for cell temperature filter.'''

    def setUp(self):
        self.tcell = np.array([-50, -49, 0, 109, 110])

    def test_tcell_filter(self):
        filtered = tcell_filter(self.tcell,
                                low_tcell_cutoff=-50,
                                high_tcell_cutoff=110)

        # Expected high and low tcell cutoffs to be non-inclusive.
        expected_result = np.array([False, True, True, True, False])
        self.assertListEqual(filtered.tolist(), expected_result.tolist())


class ClipFilterTestCase(unittest.TestCase):
    ''' Unit tests for inverter clipping filter.'''

    def setUp(self):
        self.power = pd.Series(np.arange(1, 101))
        # Note: Power is expected to be Series object because clip_filter makes
        #       use of the Series.quantile() method.

    def test_clip_filter_upper(self):
        filtered = clip_filter(self.power, quant=0.98,
                               low_power_cutoff=0)

        # Expect 99% of the 98th quantile to be filtered
        expected_result = self.power < (98 * 0.99)
        self.assertTrue((expected_result == filtered).all())

    def test_tcell_filter_low_cutoff(self):
        filtered = clip_filter(self.power, quant=0.98,
                               low_power_cutoff=2)

        # Expect power <=2 to be filtered
        expected_result = (self.power < (98 * 0.99)) & (self.power > 2)
        self.assertTrue((expected_result == filtered).all())


if __name__ == '__main__':
    unittest.main()
