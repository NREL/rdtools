import pandas as pd
import numpy as np
from rdtools import soiling_srr, soiling_cods
from rdtools.soiling import NoValidIntervalError
import pytest


@pytest.fixture()
def times():
    tz = 'Etc/GMT+7'
    times = pd.date_range('2019/01/01', '2021/01/01', freq='D', tz=tz)
    return times


@pytest.fixture()
def normalized_daily(times):
    N = len(times)
    interval_1 = 1 - 0.005 * np.arange(0, 25, 1)
    interval_2 = 1 - 0.002 * np.arange(0, 25, 1)
    interval_3 = 1 - 0.001 * np.arange(0, 25, 1)
    profile = np.concatenate((interval_1, interval_2, interval_3))
    repeated_profile = np.concatenate([profile for _ in range(np.ceil(N / 75))])
    np.random.seed(1977)
    noise = 0.01 * np.random.rand(N)
    normalized_daily = pd.Series(data=repeated_profile[:N], index=times)
    normalized_daily = normalized_daily + noise

    return normalized_daily

# def test_soiling_cods(normalized_daily, insolation, times):

#     reps = 16
#     np.random.seed(1977)
#     sr, sr_ci, deg, deg_ci, result_df = soiling_cods(normalized_daily, reps=reps)
#     print(sr, sr_ci, deg, deg_ci)
#     assert 0.963133 == pytest.approx(sr, abs=1e-6),\
#         'Soiling ratio different from expected value'
#     assert np.array([0.961054, 0.964019]) == pytest.approx(sr_ci, abs=1e-6),\
#         'Confidence interval different from expected value'
#     assert 0.958292 == pytest.approx(soiling_info['exceedance_level'], abs=1e-6),\
#         'Exceedance level different from expected value'
#     assert 0.984079 == pytest.approx(soiling_info['renormalizing_factor'], abs=1e-6),\
#         'Renormalizing factor different from expected value'
#     assert len(soiling_info['stochastic_soiling_profiles']) == reps,\
#         'Length of soiling_info["stochastic_soiling_profiles"] different than expected'
#     assert isinstance(soiling_info['stochastic_soiling_profiles'], list),\
#         'soiling_info["stochastic_soiling_profiles"] is not a list'

#     # Check soiling_info['soiling_interval_summary']
#     expected_summary_columns = ['start', 'end', 'slope', 'slope_low', 'slope_high',
#                                 'inferred_start_loss', 'inferred_end_loss', 'length', 'valid']
#     actual_summary_columns = soiling_info['soiling_interval_summary'].columns.values

#     for x in actual_summary_columns:
#         assert x in expected_summary_columns,\
#             "'{}' not an expected column in soiling_info['soiling_interval_summary']".format(x)
#     for x in expected_summary_columns:
#         assert x in actual_summary_columns,\
#             "'{}' was expected as a column, but not in soiling_info['soiling_interval_summary']".format(x)
#     assert isinstance(soiling_info['soiling_interval_summary'], pd.DataFrame),\
#         'soiling_info["soiling_interval_summary"] not a dataframe'
#     expected_means = pd.Series({'slope': -0.002617290,
#                                 'slope_low': -0.002828525,
#                                 'slope_high': -0.002396639,
#                                 'inferred_start_loss': 1.021514,
#                                 'inferred_end_loss': 0.9572880,
#                                 'length': 24.0,
#                                 'valid': 1.0})
#     expected_means = expected_means[['slope', 'slope_low', 'slope_high',
#                                      'inferred_start_loss', 'inferred_end_loss',
#                                      'length', 'valid']]
#     pd.testing.assert_series_equal(expected_means, soiling_info['soiling_interval_summary'].mean(),
#                                    check_exact=False, check_less_precise=6)

#     # Check soiling_info['soiling_ratio_perfect_clean']
#     pd.testing.assert_index_equal(soiling_info['soiling_ratio_perfect_clean'].index, times, check_names=False)
#     assert 0.967170 == pytest.approx(soiling_info['soiling_ratio_perfect_clean'].mean(), abs=1e-6),\
#         "The mean of soiling_info['soiling_ratio_perfect_clean'] differs from expected"
#     assert isinstance(soiling_info['soiling_ratio_perfect_clean'], pd.Series),\
#         'soiling_info["soiling_ratio_perfect_clean"] not a pandas series'

# def test_soiling_cods_with_nan_interval(normalized_daily, insolation, times):
#     '''
#     Previous versions had a bug which would have raised an error when an entire interval
#     was NaN. See https://github.com/NREL/rdtools/issues/129
#     '''
#     reps = 10
#     normalized_corrupt = normalized_daily.copy()
#     normalized_corrupt[26:50] = np.nan
#     np.random.seed(1977)
#     sr, sr_ci, soiling_info = soiling_srr(normalized_corrupt, insolation, reps=reps)
#     assert 0.947416 == pytest.approx(sr, abs=1e-6),\
#         'Soiling ratio different from expected value when an entire interval was NaN'