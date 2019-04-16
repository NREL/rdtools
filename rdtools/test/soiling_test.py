import pandas as pd
import numpy as np
from rdtools import soiling_srr
from rdtools.soiling import NoValidIntervalError
import pytest


@pytest.fixture()
def times():
    tz = 'Etc/GMT+7'
    times = pd.date_range('2019/01/01', '2019/03/16', freq='D', tz=tz)
    return times


@pytest.fixture()
def normalized_daily(times):
    interval_1 = 1 - 0.005 * np.arange(0, 25, 1)
    interval_2 = 1 - 0.002 * np.arange(0, 25, 1)
    interval_3 = 1 - 0.001 * np.arange(0, 25, 1)
    profile = np.concatenate((interval_1, interval_2, interval_3))
    np.random.seed(1977)
    noise = 0.01 * np.random.rand(75)
    normalized_daily = pd.Series(data=profile, index=times)
    normalized_daily = normalized_daily + noise

    return normalized_daily


@pytest.fixture()
def insolation(times):
    insolation = np.empty((75,))
    insolation[:30] = 8000
    insolation[30:45] = 6000
    insolation[45:] = 7000

    insolation = pd.Series(data=insolation, index=times)

    return insolation


def test_soiling_srr(normalized_daily, insolation, times):

    reps = 10
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=reps,
                                          random_seed=1977)
    assert 0.963133 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio different from expected value'
    assert np.array([0.961054, 0.964019]) == pytest.approx(sr_ci, abs=1e-6),\
        'Confidence interval different from expected value'
    assert 0.958292 == pytest.approx(soiling_info['exceedance_level'], abs=1e-6),\
        'Exceedance level different from expected value'
    assert 0.984079 == pytest.approx(soiling_info['renormalizing_factor'], abs=1e-6),\
        'Renormalizing factor different from expected value'
    assert len(soiling_info['stochastic_soiling_profiles']) == reps,\
        'Length of soiling_info["stochastic_soiling_profiles"] different than expected'
    assert isinstance(soiling_info['stochastic_soiling_profiles'], list),\
        'soiling_info["stochastic_soiling_profiles"] is not a list'

    # Check soiling_info['soiling_interval_summary']
    expected_summary_columns = ['start', 'end', 'slope', 'slope_low', 'slope_high',
                                'inferred_start_loss', 'inferred_end_loss', 'length', 'valid']
    actual_summary_columns = soiling_info['soiling_interval_summary'].columns.values

    for x in actual_summary_columns:
        assert x in expected_summary_columns,\
            "'{}' not an expected column in soiling_info['soiling_interval_summary']".format(x)
    for x in expected_summary_columns:
        assert x in actual_summary_columns,\
            "'{}' was expected as a column, but not in soiling_info['soiling_interval_summary']".format(x)
    assert isinstance(soiling_info['soiling_interval_summary'], pd.DataFrame),\
        'soiling_info["soiling_interval_summary"] not a dataframe'
    expected_means = pd.Series({'slope': -0.002617290,
                                'slope_low': -0.002828525,
                                'slope_high': -0.002396639,
                                'inferred_start_loss': 1.021514,
                                'inferred_end_loss': 0.9572880,
                                'length': 24.0,
                                'valid': 1.0})
    expected_means = expected_means[['slope', 'slope_low', 'slope_high',
                                     'inferred_start_loss', 'inferred_end_loss',
                                     'length', 'valid']]
    pd.testing.assert_series_equal(expected_means, soiling_info['soiling_interval_summary'].mean(),
                                   check_exact=False, check_less_precise=6)

    # Check soiling_info['soiling_ratio_perfect_clean']
    pd.testing.assert_index_equal(soiling_info['soiling_ratio_perfect_clean'].index, times, check_names=False)
    assert 0.967170 == pytest.approx(soiling_info['soiling_ratio_perfect_clean'].mean(), abs=1e-6),\
        "The mean of soiling_info['soiling_ratio_perfect_clean'] differs from expected"
    assert isinstance(soiling_info['soiling_ratio_perfect_clean'], pd.Series),\
        'soiling_info["soiling_ratio_perfect_clean"] not a pandas series'


def test_soiling_srr_with_precip(normalized_daily, insolation, times):
    precip = pd.Series(index=times, data=0)
    precip['2019-02-24 00:00:00-07:00'] = 1

    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          random_seed=1977, precip=precip, precip_clean_only=True)
    assert 0.948867 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio with precip_clean_only=True different from expected'


def test_soiling_srr_confidence_levels(normalized_daily, insolation):
    'Tests SRR with different confidence level settingsf from above'
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, confidence_level=95, reps=10,
                                          random_seed=1977, exceedance_prob=80.0)
    assert np.array([0.957272, 0.964763]) == pytest.approx(sr_ci, abs=1e-6),\
        'Confidence interval with confidence_level=95 different than expected'
    assert 0.961285 == pytest.approx(soiling_info['exceedance_level'], abs=1e-6),\
        'soiling_info["exceedance_level"] different than expected when exceedance_prob=80'


def test_soiling_srr_dayscale(normalized_daily, insolation):
    'Test that a long dayscale can prevent valid intervals from being found'
    with pytest.raises(NoValidIntervalError):
        sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, confidence_level=68.2,
                                              reps=10, random_seed=1977, day_scale=90)


def test_soiling_srr_clean_threshold(normalized_daily, insolation):
    '''Test that clean test_soiling_srr_clean_threshold works with a float and
    can cause no soiling intervals to be found'''
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          random_seed=1977, clean_threshold=0.01)
    assert 0.963133 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio with specified clean_threshold different from expected value'

    with pytest.raises(NoValidIntervalError):
        sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                              random_seed=1977, clean_threshold=0.1)


def test_soiling_srr_trim(normalized_daily, insolation):
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          random_seed=1977, trim=True)

    assert 0.978369 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio with trim=True different from expected value'
    assert len(soiling_info['soiling_interval_summary']) == 1,\
        'Wrong number of soiling intervals found with trim=True'


def test_soiling_srr_method(normalized_daily, insolation):
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          random_seed=1977, method='random_clean')
    assert 0.918767 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio with method="random_clean" different from expected value'

    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          random_seed=1977, method='perfect_clean')
    assert 0.965653 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio with method="perfect_clean" different from expected value'


def test_soiling_srr_recenter_false(normalized_daily, insolation):
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          random_seed=1977, recenter=False)
    assert 1 == soiling_info['renormalizing_factor'],\
        'Renormalizing factor != 1 with recenter=False'
    assert 0.965158 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio different than expected when recenter=False'


def test_soiling_srr_negative_step(normalized_daily, insolation):
    stepped_daily = normalized_daily.copy()
    stepped_daily.iloc[37:] = stepped_daily.iloc[25:] - 0.1

    sr, sr_ci, soiling_info = soiling_srr(stepped_daily, insolation, reps=10,
                                          random_seed=1977)

    assert list(soiling_info['soiling_interval_summary']['valid'].values) == [True, False, True],\
        'Soiling interval validity differs from expected when a large negative step\
        is incorporated into the data'

    assert 0.934927 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio different from expected when a large negative step is incorporated into the data'


def test_soiling_srr_max_negative_slope_error(normalized_daily, insolation):
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          random_seed=1977, max_relative_slope_error=50.0)

    assert list(soiling_info['soiling_interval_summary']['valid'].values) == [True, True, False],\
        'Soiling interval validity differs from expected when max_relative_slope_error=50.0'

    assert 0.952995 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio different from expected when max_relative_slope_error=50.0'
