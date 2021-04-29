'''Bootstrap module tests.'''

from rdtools.bootstrap import _construct_confidence_intervals,\
    _make_time_series_bootstrap_samples
from rdtools.degradation import degradation_year_on_year


def test_bootstrap_module(cods_normalized_daily, cods_normalized_daily_wo_noise):
    ''' Test make time serie bootstrap samples and construct of confidence intervals. '''
    # Test make bootstrap samples
    bootstrap_samples = _make_time_series_bootstrap_samples(cods_normalized_daily,
                                                           cods_normalized_daily_wo_noise,
                                                           sample_nr=10,
                                                           block_length=90,
                                                           decomposition_type='multiplicative')
    # Check if results are as expected
    assert (bootstrap_samples.index == cods_normalized_daily.index).all(), \
        "Index of bootstrapped signals is not as expected"
    assert bootstrap_samples.shape[1] == 10, "Number of columns in bootstrapped signals is wrong"

    # Test construction of confidence intervals
    confidence_intervals, exceedance_level, metrics = _construct_confidence_intervals(
        bootstrap_samples, degradation_year_on_year, uncertainty_method='none')

    # Check if results are as expected
    assert len(confidence_intervals) == 2, "2 confidence interval bounds not returned"
    assert isinstance(confidence_intervals[0], float) and \
        isinstance(confidence_intervals[1], float), "Confidence interval bounds are not float"
    assert isinstance(exceedance_level, float), "Exceedance level is not float"
    assert len(metrics) == 10, "Length of metrics is not as expected"
    for m in metrics:
        assert isinstance(m, float), "Not all metrics are float"
