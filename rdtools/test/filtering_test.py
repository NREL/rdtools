""" Filtering Module Tests. """

import unittest

import pandas as pd
import numpy as np

from rdtools import csi_filter, poa_filter, tcell_filter, clip_filter, normalized_filter


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
                              poa_global_low=200,
                              poa_global_high=1200)

        # Expect high and low POA cutoffs to be non-inclusive.
        expected_result = np.array([True, True, True, False, False])
        self.assertListEqual(filtered.tolist(), expected_result.tolist())


class TcellFilterTestCase(unittest.TestCase):
    ''' Unit tests for cell temperature filter.'''

    def setUp(self):
        self.tcell = np.array([-50, -49, 0, 109, 110])

    def test_tcell_filter(self):
        filtered = tcell_filter(self.tcell,
                                temperature_cell_low=-50,
                                temperature_cell_high=110)

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
        filtered = clip_filter(self.power, quantile=0.98)

        # Expect 99% of the 98th quantile to be filtered
        expected_result = self.power < (98 * 0.99)
        self.assertTrue((expected_result == filtered).all())
        

class GeometricClipFilterTestCase(unittest.TestCase):
    ''' Unit tests for geometric clipping filter.'''

    def setUp(self):
        self.power_no_datetime_index = pd.Series(np.arange(1, 101))
        self.power_datetime_index = pd.Series(np.arange(1, 101))
        #Add datetime index to second series
        time_range = pd.date_range('2016-12-02T11:00:00.000Z', '2017-06-06T07:00:00.000Z', freq='H')
        self.power_datetime_index.index = pd.to_datetime(time_range[:100])
        # Note: Power is expected to be Series object with a datetime index.

    def test_clip_filter(self):
        #Test that a Type Error is raised when a pandas series without a datetime index is used.
        self.assertRaises(TypeError,  geometric_clip_filter, self.power_no_datetime_index)
        # Expect none of the sequence to be clipped (as it's constantly increasing)
        filtered, mask = geometric_clip_filter(self.power_datetime_index)
        self.assertTrue(mask.all() == False)


def test_normalized_filter_default():
    pd.testing.assert_series_equal(normalized_filter(pd.Series([-5, 5])),
                                   pd.Series([False, True]))

    pd.testing.assert_series_equal(normalized_filter(pd.Series([-1e6, 1e6]),
                                                     energy_normalized_low=None,
                                                     energy_normalized_high=None),
                                   pd.Series([True, True]))

    pd.testing.assert_series_equal(normalized_filter(pd.Series([-2, 2]),
                                                     energy_normalized_low=-1,
                                                     energy_normalized_high=1),
                                   pd.Series([False, False]))

    pd.testing.assert_series_equal(normalized_filter(pd.Series([0.01 - 1e-16, 0.01 + 1e-16, 1e308])),
                                   pd.Series([False, True, True]))

if __name__ == '__main__':
    unittest.main()
