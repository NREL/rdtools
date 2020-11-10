import pandas as pd
import numpy as np
from rdtools.soiling import soiling_srr
from rdtools.soiling import SRRAnalysis
from rdtools.soiling import annual_soiling_ratios
from rdtools.soiling import monthly_soiling_rates
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
    expected_summary_columns = ['start', 'end', 'soiling_rate', 'soiling_rate_low', 'soiling_rate_high',
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
    expected_means = pd.Series({'soiling_rate': -0.002644544,
                                'soiling_rate_low': -0.002847504,
                                'soiling_rate_high': -0.002455915,
                                'inferred_start_loss': 1.020124,
                                'inferred_end_loss': 0.9566552,
                                'length': 24.0,
                                'valid': 1.0})
    expected_means = expected_means[['soiling_rate', 'soiling_rate_low', 'soiling_rate_high',
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


def test_soiling_srr_min_interval_length(normalized_daily, insolation):
    'Test that a long minimum interval length prevents finding shorter intervals'
    with pytest.raises(NoValidIntervalError):
        np.random.seed(1977)
        # normalized_daily intervals are 25 days long, so min=26 should fail:
        _ = soiling_srr(normalized_daily, insolation, confidence_level=68.2,
                        reps=10, min_interval_length=26)

    # but min=24 should be fine:
    _ = soiling_srr(normalized_daily, insolation, confidence_level=68.2,
                    reps=10, min_interval_length=24)


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

    assert 0.958761 == pytest.approx(sr, abs=1e-6),\
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


def test_soiling_srr_kwargs(monkeypatch, normalized_daily, insolation):
    '''
    Make sure that all soiling_srr parameters get passed on to SRRAnalysis and
    SRRAnalysis.run(), i.e. all necessary inputs to SRRAnalysis are provided by
    soiling_srr.  Done by removing the SRRAnalysis default param values
    and making sure everything still runs.
    '''
    # the __defaults__ attr is the tuple of default values in py3
    monkeypatch.delattr(SRRAnalysis.__init__, "__defaults__")
    monkeypatch.delattr(SRRAnalysis.run, "__defaults__")
    _ = soiling_srr(normalized_daily, insolation, reps=10)


# ###########################
# annual_soiling_ratios tests
# ###########################


@pytest.fixture()
def multi_year_profiles():
    times = pd.date_range('01-01-2018', '11-30-2019', freq='D')
    data = np.array([0]*365 + [10]*334)
    profiles = [pd.Series(x + data, times) for x in range(10)]

    # make insolation slighly longer to test for proper normalization
    times = pd.date_range('01-01-2018', '12-31-2019', freq='D')
    insolation = 350*[0.8] + (len(times)-350)*[1]
    insolation = pd.Series(insolation, index=times)

    return profiles, insolation


def test_annual_soiling_ratios(multi_year_profiles):
    expected_data = np.array([[2018, 4.5, 1.431, 7.569],
        [2019, 14.5, 11.431, 17.569]])
    expected = pd.DataFrame(data=expected_data,
        columns=['year', 'soiling_ratio_median', 'soiling_ratio_low', 'soiling_ratio_high'])
    expected['year'] = expected['year'].astype(int)
    
    srr_profiles, insolation = multi_year_profiles
    result = annual_soiling_ratios(srr_profiles, insolation)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_annual_soiling_ratios_confidence_interval(multi_year_profiles):
    expected_data = np.array([[2018, 4.5, 0.225, 8.775],
        [2019, 14.5, 10.225, 18.775]])
    expected = pd.DataFrame(data=expected_data,
        columns=['year', 'soiling_ratio_median', 'soiling_ratio_low', 'soiling_ratio_high'])
    expected['year'] = expected['year'].astype(int)

    srr_profiles, insolation = multi_year_profiles
    result = annual_soiling_ratios(srr_profiles, insolation, confidence_level=95)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_annual_soiling_ratios_warning(multi_year_profiles):
    srr_profiles, insolation = multi_year_profiles
    insolation = insolation.iloc[:-200]
    match = ('The indexes of stochastic_soiling_profiles are not entirely contained '
             'within the index of insolation_daily. Every day in stochastic_soiling_profiles '
             'should be represented in insolation_daily. This may cause erroneous results.')
    with pytest.warns(UserWarning, match=match):
        result = annual_soiling_ratios(srr_profiles, insolation)


# ###########################
# monthly_soiling_rates tests
# ###########################


@pytest.fixture()
def soiling_interval_summary():
    starts = ['2019/01/01', '2019/01/16', '2019/02/08', '2019/03/06']
    starts = pd.to_datetime(starts).tz_localize('America/Denver')
    ends = ['2019/01/15', '2019/02/07', '2019/03/05', '2019/04/07']
    ends = pd.to_datetime(ends).tz_localize('America/Denver')
    slopes = [-0.005, -0.002, -0.001, -0.002]
    slopes_low = [-0.0055, -0.0025, -0.0015, -0.003]
    slopes_high = [-0.004, 0, 0, -0.001]
    valids = [True, True, False, True]

    soiling_interval_summary = pd.DataFrame()
    soiling_interval_summary['start'] = starts
    soiling_interval_summary['end'] = ends
    soiling_interval_summary['soiling_rate'] = slopes
    soiling_interval_summary['soiling_rate_low'] = slopes_low
    soiling_interval_summary['soiling_rate_high'] = slopes_high
    soiling_interval_summary['inferred_start_loss'] = np.nan
    soiling_interval_summary['inferred_end_loss'] = np.nan
    soiling_interval_summary['length'] = (ends - starts).days
    soiling_interval_summary['valid'] = valids

    return soiling_interval_summary


def _build_monthly_summary(top_rows):
    '''
    Convienience function to build a full monthly soiling summary
    dataframe from the expected_top_rows which summarize Jan-April
    '''

    all_rows = np.vstack((top_rows, [[1, np.nan, np.nan, np.nan, 0]]*8))

    df = pd.DataFrame(data=all_rows,
                      columns=['month', 'soiling_rate_median', 'soiling_rate_low', 'soiling_rate_high', 'interval_count'])
    df['month'] = range(1, 13)

    return df


def test_monthly_soiling_rates(soiling_interval_summary):
    np.random.seed(1977)
    result = monthly_soiling_rates(soiling_interval_summary)

    expected = np.array([[1.00000000e+00, -2.42103810e-03, -5.00912766e-03, -7.68551806e-04, 2.00000000e+00],
                        [2.00000000e+00, -1.25092837e-03, -2.10091842e-03, -3.97354321e-04, 1.00000000e+00],
                        [3.00000000e+00, -2.00313359e-03, -2.68359541e-03, -1.31927678e-03, 1.00000000e+00],
                        [4.00000000e+00, -1.99729563e-03, -2.68067699e-03, -1.31667446e-03, 1.00000000e+00]])
    expected = _build_monthly_summary(expected)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_monthly_soiling_rates_min_interval_length(soiling_interval_summary):
    np.random.seed(1977)
    result = monthly_soiling_rates(soiling_interval_summary, min_interval_length=20)

    expected = np.array([[1.00000000e+00, -1.24851539e-03, -2.10394564e-03, -3.98358211e-04, 1.00000000e+00],
                        [2.00000000e+00, -1.25092837e-03, -2.10091842e-03, -3.97330424e-04, 1.00000000e+00],
                        [3.00000000e+00, -2.00309454e-03, -2.68359541e-03, -1.31927678e-03, 1.00000000e+00],
                        [4.00000000e+00, -1.99729563e-03, -2.68067699e-03, -1.31667446e-03, 1.00000000e+00]])
    expected = _build_monthly_summary(expected)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_monthly_soiling_rates_max_slope_err(soiling_interval_summary):
    np.random.seed(1977)
    result = monthly_soiling_rates(soiling_interval_summary, max_relative_slope_error=120)

    expected = np.array([[1.00000000e+00, -4.74910923e-03, -5.26236739e-03, -4.23901493e-03, 1.00000000e+00],
                        [2.00000000e+00, np.nan, np.nan, np.nan, 0.00000000e+00],
                        [3.00000000e+00, -2.00074270e-03, -2.68073474e-03, -1.31786434e-03, 1.00000000e+00],
                        [4.00000000e+00, -2.00309454e-03, -2.68359541e-03, -1.31927678e-03, 1.00000000e+00]])
    expected = _build_monthly_summary(expected)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_monthly_soiling_rates_confidence_level(soiling_interval_summary):
    np.random.seed(1977)
    result = monthly_soiling_rates(soiling_interval_summary, confidence_level=95)

    expected = np.array([[1.00000000e+00, -2.42103810e-03, -5.42313113e-03, -1.21156562e-04, 2.00000000e+00],
                        [2.00000000e+00, -1.25092837e-03, -2.43731574e-03, -6.23842627e-05, 1.00000000e+00],
                        [3.00000000e+00, -2.00313359e-03, -2.94998476e-03, -1.04988760e-03, 1.00000000e+00],
                        [4.00000000e+00, -1.99729563e-03, -2.95063841e-03, -1.04869949e-03, 1.00000000e+00]])

    expected = _build_monthly_summary(expected)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_monthly_soiling_rates_reps(soiling_interval_summary):
    np.random.seed(1977)
    result = monthly_soiling_rates(soiling_interval_summary, reps=3)

    expected = np.array([[1.00000000e+00, -2.88594088e-03, -5.03736679e-03, -6.47391131e-04, 2.00000000e+00],
                        [2.00000000e+00, -1.67359565e-03, -2.00504171e-03, -1.33240044e-03, 1.00000000e+00],
                        [3.00000000e+00, -1.22306993e-03, -2.19274892e-03, -1.11793240e-03, 1.00000000e+00],
                        [4.00000000e+00, -1.94675549e-03, -2.42574164e-03, -1.54850795e-03, 1.00000000e+00]])

    expected = _build_monthly_summary(expected)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
