import unittest
import pandas as pd

from rdtools.aggregation import aggregation_insol


class AggregationTestCase(unittest.TestCase):
    '''Unit tests for aggregation module'''

    def setUp(self):
        ind = pd.date_range('2015-01-01', '2015-01-02 23:59', freq='12h')

        self.insol = pd.Series(data=[500, 1000, 500, 1000], index=ind)
        self.energy = pd.Series(data=[1.0, 4, 1.0, 4], index=ind)

        self.aggregated = aggregation_insol(self.energy, self.insol, frequency='D')

    # Test for the expected energy waited result
    def test_aggregation_insol(self):
        self.assertTrue((self.aggregated == 3.0).all())

    # Test for expected aggregation frequency
    def test_aggregation_freq(self):
        self.assertEqual(str(self.aggregated.index.freq), '<Day>')


if __name__ == '__main__':
    unittest.main()
