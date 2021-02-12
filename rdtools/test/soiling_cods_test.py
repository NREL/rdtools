'''Test methods for the CODS-method to soiling analysis'''
import pandas as pd
import numpy as np
import rdtools.soiling as soiling
import pytest


def test_iterative_signal_decomposition(cods_normalized_daily):
    ''' Test iterative_signal_decomposition with fixed test case '''
    np.random.seed(1977)
    cods = soiling.CODSAnalysis(cods_normalized_daily)
    df_out, degradation, soiling_loss, rs, RMSE, sss, adf_res = \
        cods.iterative_signal_decomposition()
    assert 0.080563 == pytest.approx(degradation, abs=1e-6),\
        'Degradation rate different from expected value'
    assert 3.305137 == pytest.approx(soiling_loss, abs=1e-6),\
        'Soiling loss different from expected value'
    assert 0.999359 == pytest.approx(rs, abs=1e-6),\
        'Residual shift different from expected value'
    assert 0.008144 == pytest.approx(RMSE, abs=1e-6),\
        'RMSE different from expected value'
    assert not sss, \
        'Small soiling signal assertion different from expected value'
    assert 7.019626e-11 == pytest.approx(adf_res[1], abs=1e-6),\
        'p-value of Augmented Dickey-Fuller test different from expected value'

    # Check result dataframe
    expected_columns = \
        ['soiling_ratio', 'soiling_rates', 'cleaning_events',
         'seasonal_component', 'degradation_trend', 'total_model', 'residuals']
    actual_columns = df_out.columns.values
    for x in actual_columns:
        assert x in expected_columns,\
            "'{}' not an expected column in result_df]".format(x)
    for x in expected_columns:
        assert x in actual_columns,\
            "'{}' was expected as a column, but not in result_df".format(x)
    assert isinstance(df_out, pd.DataFrame), 'result_df not a dataframe'
    expected_means = pd.Series({'soiling_ratio': 0.9669486267086722,
                                'soiling_rates': -0.0024630658969236213,
                                'cleaning_events': 0.04644808743169399,
                                'seasonal_component': 1.0001490302365126,
                                'degradation_trend': 1.0008062064560372,
                                'total_model': 0.9672468949656685,
                                'residuals': 0.9993594568230086})
    expected_means = expected_means[
        ['soiling_ratio', 'soiling_rates', 'cleaning_events',
         'seasonal_component', 'degradation_trend', 'total_model', 'residuals']]
    pd.testing.assert_series_equal(expected_means, df_out.mean(),
                                   check_exact=False, check_less_precise=True)


def test_iterative_signal_decompositionwith_nan_interval(cods_normalized_daily):
    ''' Test the CODS algorithm with fixed test case with a NaN period'''
    normalized_corrupt = cods_normalized_daily.copy()
    normalized_corrupt[26:50] = np.nan
    np.random.seed(1977)
    cods = soiling.CODSAnalysis(normalized_corrupt)
    df_out, degradation, soiling_loss, rs, RMSE, sss, adf_res = \
        cods.iterative_signal_decomposition()
    assert -0.004968 == pytest.approx(degradation, abs=1e-5),\
        'Degradation rate different from expected value'
    assert 3.232171 == pytest.approx(soiling_loss, abs=1e-5),\
        'Soiling loss different from expected value'
    assert 1.000108 == pytest.approx(rs, abs=1e-5),\
        'Residual shift different from expected value'
    assert 0.008184 == pytest.approx(RMSE, abs=1e-5),\
        'RMSE different from expected value'
    assert not sss, \
        'Small soiling signal assertion different from expected value'
    assert 1.230754e-8 == pytest.approx(adf_res[1], abs=1e-6),\
        'p-value of Augmented Dickey-Fuller test different from expected value'

    # Check result dataframe
    assert isinstance(df_out, pd.DataFrame), 'result_df not a dataframe'
    expected_means = pd.Series({'soiling_ratio': 0.967678,
                                'soiling_rates': -0.002366,
                                'cleaning_events': 0.045082,
                                'seasonal_component': 1.000192,
                                'degradation_trend': 0.999950,
                                'total_model': 0.967915,
                                'residuals': 1.000108})
    expected_means = expected_means[
        ['soiling_ratio', 'soiling_rates', 'cleaning_events',
         'seasonal_component', 'degradation_trend', 'total_model', 'residuals']]
    pd.testing.assert_series_equal(expected_means, df_out.mean(),
                                   check_exact=False, check_less_precise=True)


def test_soiling_cods(cods_normalized_daily):
    ''' Test the CODS algorithm with fixed test case and 16 repetitions'''
    reps = 16
    np.random.seed(1977)
    sr, sr_ci, deg, deg_ci, result_df = soiling.soiling_cods(cods_normalized_daily, reps=reps)
    assert 0.962207 == pytest.approx(sr, abs=1e-1),\
        'Soiling ratio different from expected value'
    assert np.array([0.96662419, 0.95692131]) == pytest.approx(sr_ci, abs=1e-1),\
        'Confidence interval of SR different from expected value'
    assert 0.09 == pytest.approx(deg, abs=0.2),\
        'Degradation rate different from expected value'
    assert np.array([-0.17143952,  0.39313724]) == pytest.approx(deg_ci, abs=0.2),\
        'Confidence interval of degradation rate different from expected value'

    # Check result dataframe
    expected_summary_columns = \
        ['soiling_ratio', 'soiling_rates', 'cleaning_events',
         'seasonal_component', 'degradation_trend', 'total_model', 'residuals',
         'SR_low', 'SR_high', 'rates_low', 'rates_high', 'bt_soiling_ratio',
         'bt_soiling_rates', 'seasonal_low', 'seasonal_high', 'model_low',
         'model_high']
    actual_summary_columns = result_df.columns.values
    for x in actual_summary_columns:
        assert x in expected_summary_columns,\
            "'{}' not an expected column in result_df]".format(x)
    for x in expected_summary_columns:
        assert x in actual_summary_columns,\
            "'{}' was expected as a column, but not in result_df".format(x)


def test_Kalman_filter_for_SR(cods_normalized_daily):
    '''Test the Kalman Filter method in CODS'''
    cods = soiling.CODSAnalysis(cods_normalized_daily)
    dfk, Ps = cods._Kalman_filter_for_SR(cods_normalized_daily)

    # Check if results are okay
    assert dfk.isna().sum().sum() == 0, "NaNs were found in Kalman Filter results"
    assert (dfk.index == cods_normalized_daily.index).all(), \
        "Index returned from Kalman Filter is not as expected"
    expected_columns = ['raw_pi', 'raw_rates', 'smooth_pi', 'smooth_rates', 'soiling_ratio',
                        'soiling_rates', 'cleaning_events', 'days_since_ce']
    actual_columns = dfk.columns.values
    for x in actual_columns:
        assert x in expected_columns,\
            "'{}' not an expected column in Kalman Filter results]".format(x)
    for x in expected_columns:
        assert x in actual_columns,\
            "'{}' was expected as a column, but not in Kalman Filter results".format(x)
    assert Ps.shape == (732, 2, 2), "Shape of array of covariance matrices (Ps) not as expected"
