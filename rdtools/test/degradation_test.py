""" Degradation Module Tests. """

import unittest
import pytest
import sys

import pandas as pd
import numpy as np
import logging

from rdtools import degradation_ols, degradation_classical_decomposition, degradation_year_on_year


class DegradationTestCase(unittest.TestCase):
    ''' Unit tests for degradation module.'''

    @classmethod
    def get_corr_energy(cls, rd, input_freq):

        # lock seed to make test deterministic
        np.random.seed(0)

        daily_rd = rd / 365.0

        start = '2012-01-01'
        if input_freq == "s":
            end = '2012-03-01'
        else:
            end = '2015-01-01'

        if input_freq == 'Irregular_D':
            freq = 'D'
        else:
            freq = input_freq

        x = pd.date_range(start=start, end=end, freq=freq)
        day_deltas = (x - x[0]) / pd.Timedelta('1d')
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
        cls.list_all_input_freq = ["MS", "ME", "W", "D", "h", "min", "s", "Irregular_D"]

        # Allowed frequencies for degradation_ols
        cls.list_ols_input_freq = ["MS", "ME", "W", "D", "h", "min", "s", "Irregular_D"]

        '''
        Allowed frequencies for degradation_classical_decomposition
        in principle CD works on higher frequency data but that makes the
        tests painfully slow
        '''
        cls.list_CD_input_freq = ["MS", "ME", "W", "D"]

        # Allowed frequencies for degradation_year_on_year
        cls.list_YOY_input_freq = ["MS", "ME", "W", "D", "Irregular_D"]

        # ------------------------------------------------------------------------------------------------
        # Allow pandas < 2.2.0 to use 'M' as an alias for MonthEnd
        # https://pandas.pydata.org/docs/whatsnew/v2.2.0.html#deprecate-aliases-m-q-y-etc-in-favour-of-me-qe-ye-etc-for-offsets
        # Check pandas version and set frequency alias
        pandas_version = pd.__version__.split(".")
        if int(pandas_version[0]) < 2 or (
            int(pandas_version[0]) == 2 and int(pandas_version[1]) < 2
        ):
            for list in [
                cls.list_all_input_freq,
                cls.list_ols_input_freq,
                cls.list_CD_input_freq,
                cls.list_YOY_input_freq,
            ]:
                if "ME" in list:
                    list.remove("ME")
                    list.append(pd.tseries.offsets.MonthEnd())
        # ------------------------------------------------------------------------------------------------

        cls.rd = -0.005

        test_corr_energy = {}

        for input_freq in cls.list_all_input_freq:
            corr_energy = cls.get_corr_energy(cls.rd, input_freq)
            test_corr_energy[input_freq] = corr_energy

        cls.test_corr_energy = test_corr_energy

    def test_degradation_with_ols(self):
        ''' Test degradation with ols. '''

        funcName = sys._getframe().f_code.co_name
        logging.debug('Running {}'.format(funcName))

        # test ols degradation calc
        for input_freq in self.list_ols_input_freq:
            logging.debug('Frequency: {}'.format(input_freq))
            rd_result = degradation_ols(self.test_corr_energy[input_freq])
            self.assertAlmostEqual(rd_result[0], 100 * self.rd, places=1)
            logging.debug('Actual: {}'.format(100 * self.rd))
            logging.debug('Estimated: {}'.format(rd_result[0]))

    def test_degradation_classical_decomposition(self):
        ''' Test degradation with classical decomposition. '''

        funcName = sys._getframe().f_code.co_name
        logging.debug('Running {}'.format(funcName))

        # test classical decomposition degradation calc
        for input_freq in self.list_CD_input_freq:
            logging.debug('Frequency: {}'.format(input_freq))
            rd_result = degradation_classical_decomposition(
                self.test_corr_energy[input_freq])
            self.assertAlmostEqual(rd_result[0], 100 * self.rd, places=1)
            logging.debug('Actual: {}'.format(100 * self.rd))
            logging.debug('Estimated: {}'.format(rd_result[0]))

    def test_degradation_year_on_year(self):
        ''' Test degradation with year on year approach. '''

        funcName = sys._getframe().f_code.co_name
        logging.debug('Running {}'.format(funcName))

        # test YOY degradation calc
        for input_freq in self.list_YOY_input_freq:
            logging.debug('Frequency: {}'.format(input_freq))
            print(self.test_corr_energy[input_freq])
            rd_result = degradation_year_on_year(
                self.test_corr_energy[input_freq])
            self.assertAlmostEqual(rd_result[0], 100 * self.rd, places=1)
            logging.debug('Actual: {}'.format(100 * self.rd))
            logging.debug('Estimated: {}'.format(rd_result[0]))

    def test_degradation_year_on_year_circular_block_bootstrap(self):
        ''' Test degradation with year on year approach with circular block bootstrapping. '''

        funcName = sys._getframe().f_code.co_name
        logging.debug('Running {}'.format(funcName))

        # test YOY degradation calc
        for input_freq in self.list_YOY_input_freq:
            if input_freq != 'Irregular_D':
                logging.debug('Frequency: {}'.format(input_freq))
                length_of_series = len(self.test_corr_energy[input_freq])
                block_length = 30 if length_of_series > 100 else int(length_of_series / 5)
                rd_result = degradation_year_on_year(
                    self.test_corr_energy[input_freq],
                    uncertainty_method='circular_block',
                    block_length=block_length)
                self.assertAlmostEqual(rd_result[0], 100 * self.rd, places=1)
                logging.debug('Actual: {}'.format(100 * self.rd))
                logging.debug('Estimated: {}'.format(rd_result[0]))

    def test_confidence_intervals(self):

        funcName = sys._getframe().f_code.co_name
        logging.debug('Running {}'.format(funcName))

        input_freq = "W"

        for func in [degradation_ols, degradation_year_on_year]:

            ci1 = 68.2
            ci2 = 95
            r1 = func(self.test_corr_energy[input_freq], confidence_level=ci1)
            r2 = func(self.test_corr_energy[input_freq], confidence_level=ci2)

            logging.debug("func: {}, ci: {}, ({}) {} ({})"
                          .format(str(func).split(' ')[1], ci1, r1[1][0], r1[0], r1[1][1]))
            logging.debug("func: {}, ci: {}, ({}) {} ({})"
                          .format(str(func).split(' ')[1], ci2, r2[1][0], r2[0], r2[1][1]))

            # lower limit is lower than median and upper limit is higher than median
            self.assertTrue(r1[0] > r1[1][0] and r1[0] < r1[1][1])
            self.assertTrue(r2[0] > r2[1][0] and r2[0] < r2[1][1])

            # 95% interval is bigger than 68% interval
            self.assertTrue(abs(r1[0] - r1[1][1]) < abs(r2[0] - r2[1][1]))
            self.assertTrue(abs(r1[0] - r1[1][0]) < abs(r2[0] - r2[1][0]))

            # actual rd is within confidence interval
            self.assertTrue(100.0 * self.rd > r2[1][0] and 100.0 * self.rd < r2[1][1])

    def test_usage_of_points(self):

        funcName = sys._getframe().f_code.co_name
        logging.debug('Running {}'.format(funcName))

        input_freq = "D"
        rd_result = degradation_year_on_year(
            self.test_corr_energy[input_freq])
        self.assertTrue((np.sum(rd_result[2]['usage_of_points'])) == 1462)


@pytest.mark.parametrize(
    "start,end,freq",
    [
        ("2014-01-01", "2015-12-31", "D"),  # no leap day
        ("2015-01-01", "2016-12-31", "D"),  # leap day included in index
        ("2015-01-01", "2016-12-29", "7D"),  # leap day in period but not in index
        ("2016-06-01", "2018-05-31", "D"),  # leap year, but no leap day in period
        #  ('2016-02-29', '2018-02-28', 'd'),   # starts on leap day (doesn't work)
        ("2014-03-01", "2016-02-29", "D"),  # ends on leap day
        ("2015-01-01", "2016-12-31", "ME"),  # month end
        ("2015-01-01", "2016-12-31", "MS"),  # month start
    ],
)
def test_yoy_two_years_error(start, end, freq):
    # ----------------------------------------------------------------
    # Allow pandas < 2.2.0 to use 'M' as an alias for MonthEnd
    # https://pandas.pydata.org/docs/whatsnew/v2.2.0.html#deprecate-aliases-m-q-y-etc-in-favour-of-me-qe-ye-etc-for-offsets
    if freq == "ME":
        freq = pd.tseries.offsets.MonthEnd()
    # ----------------------------------------------------------------

    times = pd.date_range(start, end, freq=freq)
    series = pd.Series(1, index=times)
    # introduce NaN at the end to ensure that the 2 year requirement applies to
    # timestamps, not non-nan values:
    series.iloc[-5:] = np.nan
    # should not raise an error
    _ = degradation_year_on_year(series)
    # but if we shorten it by one element, then it should:
    with pytest.raises(ValueError, match='must provide at least two years'):
        _ = degradation_year_on_year(series.iloc[:-1])
    with pytest.raises(ValueError, match='must provide at least two years'):
        _ = degradation_year_on_year(series.iloc[1:])


if __name__ == '__main__':
    # Initialize logger when run as a module:
    #     python -m tests.degradation_test
    logging.root.handlers = []
    logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s| %(message)s',
                        level=logging.DEBUG,
                        stream=sys.stdout)
    unittest.main()
