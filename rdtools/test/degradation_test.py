""" Degradation Module Tests. """

import unittest
import pytest
import sys

import pandas as pd
import numpy as np
import logging

from rdtools import (degradation_ols, degradation_classical_decomposition,
                     degradation_year_on_year, degradation_hybrid_ols_yoy)


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
        day_deltas = (x - x[0]) / pd.Timedelta('1D')
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

    def test_avg_timestamp_old_Pandas(self):
        """Test the _avg_timestamp_old_Pandas function for correct averaging."""
        from rdtools.degradation import _avg_timestamp_old_Pandas
        funcName = sys._getframe().f_code.co_name
        logging.debug('Running {}'.format(funcName))
        dt = pd.Series(self.get_corr_energy(0, 'D').index[-4:].tz_localize('UTC'),
                       index=self.get_corr_energy(0, 'D').index[-4:].tz_localize('UTC'))
        dt_right = pd.Series(self.get_corr_energy(0, 'D').index[-3:].tz_localize('UTC') +
                             pd.Timedelta(days=365),
                             index=self.get_corr_energy(0, 'D').index[-3:].tz_localize('UTC'))
        # Expected dtype depends on pandas version (ns for <3.0, s for >=3.0)
        pandas_version = pd.__version__.split(".")
        if int(pandas_version[0]) < 3:
            expected_dtype = "datetime64[ns, UTC]"
        else:
            expected_dtype = "datetime64[s, UTC]"
        # Expected result is the midpoint between each pair
        expected = pd.Series(
            [
                pd.NaT,
                pd.Timestamp("2015-06-30 12:00:00"),
                pd.Timestamp("2015-07-01 12:00:00"),
                pd.Timestamp("2015-07-02 12:00:00"),
            ],
            index=self.get_corr_energy(0, "D").index[-4:],
            name="averages",
            dtype=expected_dtype,
        ).tz_localize("UTC")

        result = _avg_timestamp_old_Pandas(dt, dt_right).asfreq(freq='D')

        pd.testing.assert_series_equal(result, expected)


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


def test_degradation_year_on_year_multi():
    """Test degradation_year_on_year with multi_yoy=True. Thanks GPT!"""
    rd = -0.005
    # Generate a daily time series with 3 years of data
    idx = pd.date_range('2017-01-01', '2020-01-01', freq='D', tz='UTC')
    daily_rd = (1 + rd)**(1/365) - 1
    day_count = np.arange(len(idx))
    degradation_derate = (1 + daily_rd) ** day_count
    power = 1 - 0.1 * np.cos(day_count / 365 * 2 * np.pi)
    power *= degradation_derate
    power = pd.Series(power, index=idx)
    # Standard yoy baseline
    (rd0, rd_ci0, calc_info0) = degradation_year_on_year(power, multi_yoy=False)
    # Run multi_yoy test
    rd_result = degradation_year_on_year(power, multi_yoy=True)
    # Should return a tuple (Rd_pct, Rd_CI, calc_info)
    assert isinstance(rd_result, tuple)
    assert len(rd_result) == 3
    Rd_pct, Rd_CI, calc_info = rd_result
    # Check that the result is close to expected degradation
    assert np.isclose(Rd_pct, 100 * rd, atol=0.5)
    # Check that YoY_values exists and is a Series
    assert isinstance(calc_info['YoY_values'], pd.Series)
    # Should have more YoY value for multi_yoy than standard
    assert len(calc_info['YoY_values']) > len(calc_info0['YoY_values'])


def test_classical_decomposition_missing_data():
    """Test that classical decomposition raises error for missing data."""
    # Create a regular time series with missing values (NaN)
    idx = pd.date_range("2012-01-01", "2015-01-01", freq="D")
    series = pd.Series(1.0, index=idx)
    series.iloc[100:105] = np.nan  # introduce missing data

    with pytest.raises(ValueError, match="regular time series"):
        degradation_classical_decomposition(series)


def test_classical_decomposition_irregular_frequency():
    """Test that classical decomposition raises error for irregular frequency."""
    # Create an irregular time series by sampling randomly
    idx = pd.date_range("2012-01-01", "2015-01-01", freq="D")
    series = pd.Series(1.0, index=idx)
    series = series.sample(frac=0.8, replace=False).sort_index()

    with pytest.raises(ValueError, match="regular time series"):
        degradation_classical_decomposition(series)


def test_yoy_circular_block_no_frequency():
    """Test circular_block raises error when frequency cannot be inferred."""
    # Create an irregular time series
    idx = pd.date_range("2012-01-01", "2015-01-01", freq="D")
    series = pd.Series(1.0, index=idx)
    series = series.sample(frac=0.8, replace=False).sort_index()

    with pytest.raises(ValueError, match="fixed frequency"):
        degradation_year_on_year(series, uncertainty_method="circular_block")


def test_yoy_circular_block_too_long():
    """Test circular_block raises error when block_length is too long."""
    idx = pd.date_range("2012-01-01", "2015-01-01", freq="D")
    series = pd.Series(1.0, index=idx)

    # block_length must be less than 1/3 of the series length
    too_long = len(series) // 2

    with pytest.raises(ValueError, match="shorter than a third"):
        degradation_year_on_year(
            series, uncertainty_method="circular_block", block_length=too_long
        )


def test_yoy_no_pairs_found():
    """Test year_on_year raises error when no valid pairs can be formed."""
    # Create a series that's just over 1 year but with NaN in positions
    # that prevent any valid year-over-year pairs
    idx = pd.date_range("2012-01-01", "2014-06-01", freq="D")
    series = pd.Series(1.0, index=idx)
    # Make all values NaN except first few and last few (too far apart for pairs)
    series.iloc[10:-10] = np.nan

    with pytest.raises(ValueError, match="no year-over-year"):
        degradation_year_on_year(series)


def _build_two_rate_series(rd1_pct, rd2_pct, start='2018-01-01',
                           end='2022-01-01', noise=1e-3, seed=0):
    """Synthetic daily series with a piecewise-linear degradation profile."""
    idx = pd.date_range(start, end, freq='D')
    years = (idx - idx[0]) / pd.Timedelta('365D')
    rate1 = rd1_pct / 100.0
    rate2 = rd2_pct / 100.0
    y0 = 1.0
    y_after_year1 = y0 + rate1 * 1.0
    y = np.where(years < 1.0,
                 y0 + rate1 * years,
                 y_after_year1 + rate2 * (years - 1.0))
    rng = np.random.default_rng(seed)
    y = y + rng.normal(0, noise, len(y))
    return pd.Series(y, index=idx)


def test_degradation_hybrid_ols_yoy_basic():
    """Recover known year-1 and post-year-1 rates from a synthetic series."""
    rd1, rd2, info = degradation_hybrid_ols_yoy(
        _build_two_rate_series(rd1_pct=-2.0, rd2_pct=-0.5)
    )
    assert np.isclose(rd1, -2.0, atol=0.2)
    # Year-2+ rate is reported relative to start-of-year-2 capacity (~0.98),
    # so the expected value is rd2 / 0.98 (still close to rd2 for small rd1).
    assert np.isclose(rd2, -0.5 / 0.98, atol=0.2)
    # Should also return the full tuples from each underlying call
    assert len(info['year1']) == 3
    assert len(info['years2plus']) == 3


def test_degradation_hybrid_ols_yoy_too_short():
    """Series shorter than year1_split + 2 years raises from YoY."""
    # only 2 years of data; year-1 window has 1 year, year-2+ window has 1 year
    series = _build_two_rate_series(rd1_pct=-1.0, rd2_pct=-0.5,
                                    start='2018-01-01', end='2020-01-01')
    with pytest.raises(ValueError, match='must provide at least two years'):
        degradation_hybrid_ols_yoy(series)


def test_degradation_hybrid_ols_yoy_calc_info_structure():
    """calc_info exposes the documented keys and consistent renorm factor."""
    series = _build_two_rate_series(rd1_pct=-1.0, rd2_pct=-0.5)
    _, _, info = degradation_hybrid_ols_yoy(series)
    for key in ('year1', 'years2plus', 'split_date',
                'renormalizing_factor_year2'):
        assert key in info
    assert info['split_date'] == series.index[0] + pd.Timedelta(days=365.0)
    yoy_calc_info = info['years2plus'][2]
    assert info['renormalizing_factor_year2'] == \
        yoy_calc_info['renormalizing_factor']


def test_degradation_hybrid_ols_yoy_reserved_kwargs_rejected():
    """recenter / confidence_level cannot be smuggled in via yoy_kwargs."""
    series = _build_two_rate_series(rd1_pct=-1.0, rd2_pct=-0.5)
    with pytest.raises(ValueError, match="'recenter'"):
        degradation_hybrid_ols_yoy(series, yoy_kwargs={'recenter': False})
    with pytest.raises(ValueError, match="'confidence_level'"):
        degradation_hybrid_ols_yoy(
            series, yoy_kwargs={'confidence_level': 90})


def test_degradation_hybrid_ols_yoy_fractional_split():
    """Non-integer year1_split (e.g. 0.5) is supported via Timedelta."""
    series = _build_two_rate_series(rd1_pct=-1.0, rd2_pct=-0.5,
                                    start='2018-01-01', end='2023-01-01')
    _, _, info = degradation_hybrid_ols_yoy(series, year1_split=0.5)
    assert info['split_date'] == series.index[0] + pd.Timedelta(days=0.5*365.0)


def test_degradation_hybrid_ols_yoy_recenter_false():
    """With recenter_year2=False, the renorm factor is 1.0."""
    series = _build_two_rate_series(rd1_pct=-1.0, rd2_pct=-0.5)
    _, _, info = degradation_hybrid_ols_yoy(series, recenter_year2=False)
    assert info['renormalizing_factor_year2'] == 1.0


def test_mk_test_no_trend():
    """Test Mann-Kendall test with no trend (z == 0 case)."""
    from rdtools.degradation import _mk_test

    # Constant series should have no trend
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    trend, h, p, z = _mk_test(x)

    assert trend == "no trend"
    assert z == 0
    assert not h


def test_mk_test_with_ties():
    """Test Mann-Kendall test with tied values."""
    from rdtools.degradation import _mk_test

    # Series with ties (repeated values)
    x = np.array([1, 2, 2, 3, 3, 3, 4, 5])
    trend, h, p, z = _mk_test(x)

    # Should still detect increasing trend
    assert trend == "increasing"


def test_mk_test_decreasing():
    """Test Mann-Kendall test with clear decreasing trend."""
    from rdtools.degradation import _mk_test

    x = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    trend, h, p, z = _mk_test(x)

    assert trend == "decreasing"
    assert z < 0


if __name__ == '__main__':
    # Initialize logger when run as a module:
    #     python -m tests.degradation_test
    logging.root.handlers = []
    logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s| %(message)s',
                        level=logging.DEBUG,
                        stream=sys.stdout)
    unittest.main()
