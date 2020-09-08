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
    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=reps)
    assert 0.964369 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio different from expected value'
    assert np.array([0.962540, 0.965295]) == pytest.approx(sr_ci, abs=1e-6),\
        'Confidence interval different from expected value'
    assert 0.960205 == pytest.approx(soiling_info['exceedance_level'], abs=1e-6),\
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
    expected_means = pd.Series({'slope': -0.002644544,
                                'slope_low': -0.002847504,
                                'slope_high': -0.002455915,
                                'inferred_start_loss': 1.020124,
                                'inferred_end_loss': 0.9566552,
                                'length': 24.0,
                                'valid': 1.0})
    expected_means = expected_means[['slope', 'slope_low', 'slope_high',
                                     'inferred_start_loss', 'inferred_end_loss',
                                     'length', 'valid']]
    pd.testing.assert_series_equal(expected_means, soiling_info['soiling_interval_summary'].mean(),
                                   check_exact=False, check_less_precise=6)

    # Check soiling_info['soiling_ratio_perfect_clean']
    pd.testing.assert_index_equal(soiling_info['soiling_ratio_perfect_clean'].index, times, check_names=False)
    assert 0.968265 == pytest.approx(soiling_info['soiling_ratio_perfect_clean'].mean(), abs=1e-6),\
        "The mean of soiling_info['soiling_ratio_perfect_clean'] differs from expected"
    assert isinstance(soiling_info['soiling_ratio_perfect_clean'], pd.Series),\
        'soiling_info["soiling_ratio_perfect_clean"] not a pandas series'


def test_soiling_srr_with_precip(normalized_daily, insolation, times):
    precip = pd.Series(index=times, data=0)
    precip['2019-01-18 00:00:00-07:00'] = 1
    precip['2019-02-20 00:00:00-07:00'] = 1

    kwargs = {
        'reps': 10,
        'precipitation_daily': precip
    }
    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, clean_criterion='precip_and_shift', **kwargs)
    assert 0.982546 == pytest.approx(sr, abs=1e-6),\
        "Soiling ratio with clean_criterion='precip_and_shift' different from expected"
    np.random.seed(1977)    
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, clean_criterion='precip_or_shift', **kwargs)
    assert 0.973433 == pytest.approx(sr, abs=1e-6),\
        "Soiling ratio with clean_criterion='precip_or_shift' different from expected"
    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, clean_criterion='precip', **kwargs)
    assert 0.976196 == pytest.approx(sr, abs=1e-6),\
        "Soiling ratio with clean_criterion='precip' different from expected"
    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, clean_criterion='shift', **kwargs)
    assert 0.964369 == pytest.approx(sr, abs=1e-6),\
        "Soiling ratio with clean_criterion='shift' different from expected"


def test_soiling_srr_confidence_levels(normalized_daily, insolation):
    'Tests SRR with different confidence level settingsf from above'
    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, confidence_level=95, reps=10,
                                          exceedance_prob=80.0)
    assert np.array([0.959322, 0.966066]) == pytest.approx(sr_ci, abs=1e-6),\
        'Confidence interval with confidence_level=95 different than expected'
    assert 0.962691 == pytest.approx(soiling_info['exceedance_level'], abs=1e-6),\
        'soiling_info["exceedance_level"] different than expected when exceedance_prob=80'


def test_soiling_srr_dayscale(normalized_daily, insolation):
    'Test that a long dayscale can prevent valid intervals from being found'
    with pytest.raises(NoValidIntervalError):
        np.random.seed(1977)
        sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, confidence_level=68.2,
                                              reps=10, day_scale=90)


def test_soiling_srr_clean_threshold(normalized_daily, insolation):
    '''Test that clean test_soiling_srr_clean_threshold works with a float and
    can cause no soiling intervals to be found'''
    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          clean_threshold=0.01)
    assert 0.964369 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio with specified clean_threshold different from expected value'

    with pytest.raises(NoValidIntervalError):
        np.random.seed(1977)
        sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                              clean_threshold=0.1)


def test_soiling_srr_trim(normalized_daily, insolation):
    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          trim=True)

    assert 0.978093 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio with trim=True different from expected value'
    assert len(soiling_info['soiling_interval_summary']) == 1,\
        'Wrong number of soiling intervals found with trim=True'


def test_soiling_srr_method(normalized_daily, insolation):
    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          method='random_clean')
    assert 0.920444 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio with method="random_clean" different from expected value'

    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          method='perfect_clean')
    assert 0.966912 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio with method="perfect_clean" different from expected value'


def test_soiling_srr_recenter_false(normalized_daily, insolation):
    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          recenter=False)
    assert 1 == soiling_info['renormalizing_factor'],\
        'Renormalizing factor != 1 with recenter=False'
    assert 0.966387 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio different than expected when recenter=False'


def test_soiling_srr_negative_step(normalized_daily, insolation):
    stepped_daily = normalized_daily.copy()
    stepped_daily.iloc[37:] = stepped_daily.iloc[25:] - 0.1

    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(stepped_daily, insolation, reps=10)

    assert list(soiling_info['soiling_interval_summary']['valid'].values) == [True, False, True],\
        'Soiling interval validity differs from expected when a large negative step\
        is incorporated into the data'

    assert 0.936932 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio different from expected when a large negative step is incorporated into the data'


def test_soiling_srr_max_negative_slope_error(normalized_daily, insolation):
    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_daily, insolation, reps=10,
                                          max_relative_slope_error=45.0)

    assert list(soiling_info['soiling_interval_summary']['valid'].values) == [True, True, False],\
        'Soiling interval validity differs from expected when max_relative_slope_error=45.0'

    assert 0.954569 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio different from expected when max_relative_slope_error=45.0'

def test_soiling_srr_with_nan_interval(normalized_daily, insolation, times):
    '''
    Previous versions had a bug which would have raised an error when an entire interval
    was NaN. See https://github.com/NREL/rdtools/issues/129
    '''
    reps = 10
    normalized_corrupt = normalized_daily.copy()
    normalized_corrupt[26:50] = np.nan
    np.random.seed(1977)
    sr, sr_ci, soiling_info = soiling_srr(normalized_corrupt, insolation, reps=reps)
    assert 0.948792 == pytest.approx(sr, abs=1e-6),\
        'Soiling ratio different from expected value when an entire interval was NaN'
