"""Tests for rdtools.signal_decomposition."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from rdtools import signal_decomposition  # noqa: E402

T = 365.2425  # samples per year


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_series(n_years=3, rd_pct=-0.5, noise=0.02, seed=0):
    """Daily series with a known linear degradation rate and seasonal variation."""
    n = int(n_years * T)
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    seasonal = 0.05 * np.sin(2 * np.pi * t / T)
    trend = 1.0 + (rd_pct / 100.0) * (t / T)
    noise_arr = rng.normal(0, noise, n)
    y = trend + seasonal + noise_arr
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.Series(y, index=idx)


def _make_short_series(n_years=1.5, rd_pct=-0.5, noise=0.02, seed=1):
    """Shorter series for stability/animation tests."""
    return _make_series(n_years=n_years, rd_pct=rd_pct, noise=noise, seed=seed)


def _make_soiled_series(n=900, accumulation_rate=0.0005, seed=4):
    """Daily series with coherent 120-day log-linear soiling intervals."""
    t = np.arange(n, dtype=float)
    ratio = np.exp(-accumulation_rate * (t % 120))
    trend = 1 - 0.004 * t / T
    seasonal = 1 + 0.025 * np.sin(2 * np.pi * t / T)
    noise = np.exp(np.random.default_rng(seed).normal(0, 0.003, n))
    index = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.Series(trend * seasonal * ratio * noise, index=index), ratio


# ---------------------------------------------------------------------------
# degradation() — return structure
# ---------------------------------------------------------------------------

class TestDegradationReturnStructure:
    def test_returns_three_tuple(self):
        s = _make_series()
        result = signal_decomposition.degradation(s)
        assert len(result) == 3

    def test_rd_pct_is_scalar(self):
        s = _make_series()
        Rd_pct, _, _ = signal_decomposition.degradation(s)
        assert np.isscalar(Rd_pct)

    def test_rd_ci_is_finite_and_ordered(self):
        s = _make_series()
        _, Rd_CI, _ = signal_decomposition.degradation(s)
        assert Rd_CI.shape == (2,)
        assert np.all(np.isfinite(Rd_CI))
        assert Rd_CI[0] <= Rd_CI[1]

    def test_sd_trend_results_keys_linear(self):
        s = _make_series()
        _, _, info = signal_decomposition.degradation(s, trend_type='linear')
        assert 'rate_pct_yr' in info
        assert 'components' in info
        assert 'y' in info
        assert 'args' in info
        assert 'problem_status' in info

    def test_sd_trend_results_keys_pwl(self):
        s = _make_series()
        _, _, info = signal_decomposition.degradation(s, trend_type='pwl')
        assert 'rate_pre_pct_yr' in info
        assert 'rate_post_pct_yr' in info
        assert 'rate_overall_pct_yr' in info

    def test_sd_trend_results_keys_monotone(self):
        s = _make_series()
        _, _, info = signal_decomposition.degradation(s, trend_type='monotone')
        assert 'rate_overall_pct_yr' in info
        assert 'rate_instantaneous_pct_yr' in info
        assert 'rate_yearly_pct_yr' in info
        assert 'year_boundaries' in info

    def test_components_keys(self):
        s = _make_series()
        _, _, info = signal_decomposition.degradation(s)
        for k in ('x1', 'x2', 'x3', 'fit'):
            assert k in info['components']

    def test_y_stored_is_original(self):
        s = _make_series()
        _, _, info = signal_decomposition.degradation(s)
        np.testing.assert_array_equal(info['y'], s.values)

    def test_args_stored(self):
        s = _make_series()
        _, _, info = signal_decomposition.degradation(
            s, trend_type='pwl', loss='l1', numharmonics=4
        )
        assert info['args']['trend_type'] == 'pwl'
        assert info['args']['loss'] == 'l1'
        assert info['args']['numharmonics'] == 4

    def test_problem_status_optimal(self):
        s = _make_series()
        _, _, info = signal_decomposition.degradation(s)
        assert info['problem_status'] in ('optimal', 'optimal_inaccurate')


# ---------------------------------------------------------------------------
# degradation() — rate recovery per trend_type
# ---------------------------------------------------------------------------

class TestRateRecovery:
    """Check that each trend_type recovers a known degradation rate.

    Tolerances are generous (0.3 %/yr) because we're testing recovery not
    precision — the solver is deterministic but the synthetic series has noise
    and seasonal structure that can shift the estimate slightly.
    """
    TOL = 0.3  # %/yr

    @pytest.mark.parametrize("rd_true", [-0.5, -1.0])
    def test_linear(self, rd_true):
        s = _make_series(n_years=3, rd_pct=rd_true, noise=0.01)
        Rd_pct, _, info = signal_decomposition.degradation(s, trend_type='linear')
        assert abs(Rd_pct - rd_true) < self.TOL
        assert Rd_pct == info['rate_pct_yr']

    def test_pwl_overall(self):
        s = _make_series(n_years=3, rd_pct=-0.5, noise=0.01)
        Rd_pct, _, info = signal_decomposition.degradation(s, trend_type='pwl')
        assert abs(Rd_pct - (-0.5)) < self.TOL
        assert Rd_pct == info['rate_overall_pct_yr']

    def test_monotone_overall(self):
        s = _make_series(n_years=3, rd_pct=-0.5, noise=0.01)
        Rd_pct, _, info = signal_decomposition.degradation(s, trend_type='monotone')
        assert abs(Rd_pct - (-0.5)) < self.TOL
        assert Rd_pct == info['rate_overall_pct_yr']


# ---------------------------------------------------------------------------
# degradation() — each loss type solves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss", ['l2', 'l1', 'huber', 'quantile'])
def test_all_loss_types_solve(loss):
    s = _make_series()
    Rd_pct, Rd_CI, info = signal_decomposition.degradation(s, loss=loss)
    assert np.isfinite(Rd_pct)
    assert info['problem_status'] in ('optimal', 'optimal_inaccurate')


# ---------------------------------------------------------------------------
# degradation() — log_transform path
# ---------------------------------------------------------------------------

def test_log_transform_solves():
    s = _make_series()
    Rd_pct, _, info = signal_decomposition.degradation(s, log_transform=True)
    assert np.isfinite(Rd_pct)
    assert 'rate_pct_yr' in info


def test_log_transform_components_positive():
    """Back-transformed components should be positive in the original domain."""
    s = _make_series()
    _, _, info = signal_decomposition.degradation(s, log_transform=True)
    assert np.all(info['components']['x2'] > 0)
    assert np.all(info['components']['fit'] > 0)


# ---------------------------------------------------------------------------
# degradation() — ValueError on bad inputs
# ---------------------------------------------------------------------------

def test_bad_trend_type_raises():
    s = _make_series()
    with pytest.raises(ValueError, match="trend_type"):
        signal_decomposition.degradation(s, trend_type='quadratic')


def test_bad_loss_raises():
    s = _make_series()
    with pytest.raises(ValueError, match="loss"):
        signal_decomposition.degradation(s, loss='cauchy')


# ---------------------------------------------------------------------------
# Optional soiling component
# ---------------------------------------------------------------------------

def test_soiling_low_level_problem_is_parameterized_and_nonpositive():
    y = np.log(_make_series(n_years=2.1, noise=0).to_numpy())
    build = signal_decomposition.make_problem(
        y,
        loss='huber',
        numharmonics=3,
        huber_M=0.05,
        include_soiling=True,
        lam_soiling_down=1e-3,
        lam_soiling_value=10 ** -5.8,
    )
    assert build['problem'].is_dcp()
    assert build['problem'].is_dpp()
    assert set(build['parameters']) == {
        'lam_soiling_down', 'lam_soiling_value'
    }
    build['problem'].solve(solver=signal_decomposition.cp.CLARABEL)
    assert np.max(build['variables']['soiling'].value) <= 1e-7


def test_soiling_low_level_weights_are_explicit():
    with pytest.raises(ValueError, match='lam_soiling_down'):
        signal_decomposition.make_problem(
            np.ones(100), include_soiling=True
        )


def test_soiling_candidate_metrics_require_valid_interior_neighbors():
    t = np.arange(800)
    base = -0.0002 * (t % 100)
    paths = np.array([scale * base for scale in np.linspace(0.96, 1.04, 9)])
    statuses = ['optimal'] * 9
    statuses[1] = 'solver_error'
    metrics = signal_decomposition._soiling_candidate_metrics(
        paths, statuses, T
    )
    assert not metrics.loc[0, 'qualifies']
    assert not metrics.loc[1, 'qualifies']
    assert metrics.loc[2, 'qualifies']
    assert not metrics.loc[8, 'qualifies']
    assert metrics.loc[2, 'worst_neighbor_correlation'] > 0.99


def test_soiling_rejects_configuration_outside_validated_preset():
    with pytest.raises(ValueError, match='validated configuration'):
        signal_decomposition.degradation(
            _make_series(), include_soiling=True, loss='l2', n_bootstrap=0
        )


def test_soiling_short_record_warns_but_solves():
    with pytest.warns(UserWarning, match='shorter than two years'):
        _, _, info = signal_decomposition.degradation(
            _make_short_series(), include_soiling=True, n_bootstrap=0
        )
    assert 'soiling' in info


def test_soiling_null_returns_actual_no_soiling_fit():
    clean = _make_series(n_years=2.2, noise=0.001)
    _, _, info = signal_decomposition.degradation(
        clean, include_soiling=True, n_bootstrap=0
    )
    soiling = info['soiling']
    assert not soiling['selector']['detected']
    assert soiling['selector']['final_model'] == 'no_soiling'
    np.testing.assert_array_equal(info['components']['soiling'], 1)
    assert soiling['loss']['time_averaged_loss_pct'] == 0
    assert len(soiling['selector']['fallback_soiling_component_log']) == len(clean)


def test_soiling_selector_detects_coherent_signal_and_reports_metrics():
    series, _ = _make_soiled_series()
    insolation = pd.Series(
        4 + np.sin(2 * np.pi * np.arange(len(series)) / T),
        index=series.index,
    )
    _, _, info = signal_decomposition.degradation(
        series,
        include_soiling=True,
        insolation_daily=insolation,
        n_bootstrap=0,
    )
    soiling = info['soiling']
    assert soiling['selector']['detected']
    assert soiling['selector']['selected_candidate_index'] == 1
    assert len(soiling['selector']['candidate_metrics']) == 9
    assert soiling['loss']['time_averaged_loss_pct'] > 0
    assert soiling['loss']['insolation_weighted_loss_pct'] > 0
    assert list(soiling['loss']['quarterly']['quarter']) == [1, 2, 3, 4]
    assert list(soiling['rate_summary']['period']) == [
        'overall', 'Q1', 'Q2', 'Q3', 'Q4'
    ]
    assert (soiling['intervals'].loc[
        soiling['intervals']['valid'], 'soiling_rate_pct_day'
    ] <= 0).all()


def test_soiling_loss_metrics_time_and_insolation_weighting():
    index = pd.date_range('2020-01-01', periods=366, freq='D')
    ratio = np.ones(366)
    ratio[index.quarter == 1] = 0.9
    insolation = pd.Series(1.0, index=index)
    insolation.loc[index.quarter == 1] = 2.0
    metrics = signal_decomposition._soiling_loss_metrics(
        ratio, index, insolation
    )
    assert metrics['time_averaged_loss_pct'] == pytest.approx(100 * 9.1 / 366)
    expected_weighted = 100 * 18.2 / (366 + 91)
    assert metrics['insolation_weighted_loss_pct'] == pytest.approx(expected_weighted)
    q1 = metrics['quarterly'].set_index('quarter').loc[1]
    assert q1['time_averaged_loss_pct'] == pytest.approx(10)
    assert q1['insolation_weighted_loss_pct'] == pytest.approx(10)


def test_soiling_interval_validity_and_day_weighted_summary():
    index = pd.date_range('2020-01-01', periods=30, freq='D')
    log_ratio = np.r_[
        -0.001 * np.arange(10),
        -0.0005 * np.arange(5),
        0.0002 * np.arange(15),
    ]
    # Material recoveries start the second and third intervals.
    log_ratio[10:] += 0.01 - log_ratio[10]
    log_ratio[15:] += 0.01 - (log_ratio[15] - log_ratio[14])
    intervals = signal_decomposition._soiling_intervals(
        np.exp(log_ratio), index
    )
    assert intervals.loc[0, 'valid']
    assert not intervals.loc[1, 'valid']  # shorter than seven days
    assert not intervals.loc[2, 'valid']  # positive fitted slope
    summary = signal_decomposition._soiling_rate_summary(intervals, index)
    overall = summary.set_index('period').loc['overall']
    assert overall['interval_count'] == 1
    assert overall['day_count'] == 10
    assert overall['median_rate_pct_day'] < 0


def test_soiling_plot_has_component_panel():
    series, _ = _make_soiled_series()
    _, _, info = signal_decomposition.degradation(
        series, include_soiling=True, n_bootstrap=0
    )
    fig = signal_decomposition.plot_decomposition(info)
    assert len(fig.axes) == 5
    plt.close(fig)


def test_soiling_bootstrap_reports_final_model_metric_intervals():
    series, _ = _make_soiled_series()
    _, _, info = signal_decomposition.degradation(
        series,
        include_soiling=True,
        n_bootstrap=4,
        block_size=30,
        random_state=2,
    )
    loss_ci = info['soiling']['loss']['time_averaged_loss_ci']
    rate_ci = info['soiling']['rate_summary'].set_index('period').loc[
        'overall', ['rate_ci_low', 'rate_ci_high']
    ].to_numpy(dtype=float)
    assert np.all(np.isfinite(loss_ci))
    assert np.all(np.isfinite(rate_ci))
    assert loss_ci[0] <= loss_ci[1]
    assert rate_ci[0] <= rate_ci[1]


def test_soiling_null_bootstrap_has_zero_width_metric_intervals():
    clean = _make_series(n_years=2.2, noise=0.001)
    _, _, info = signal_decomposition.degradation(
        clean,
        include_soiling=True,
        n_bootstrap=4,
        block_size=30,
        random_state=3,
    )
    assert not info['soiling']['selector']['detected']
    np.testing.assert_array_equal(
        info['soiling']['loss']['time_averaged_loss_ci'], [0, 0]
    )
    rate_ci = info['soiling']['rate_summary'].set_index('period').loc[
        'overall', ['rate_ci_low', 'rate_ci_high']
    ].to_numpy(dtype=float)
    np.testing.assert_array_equal(rate_ci, [0, 0])


def test_soiling_report_includes_detection_losses_and_rate():
    series, _ = _make_soiled_series()
    _, _, info = signal_decomposition.degradation(
        series, include_soiling=True, n_bootstrap=0
    )
    report = signal_decomposition.format_degradation_report(info)
    assert '**Soiling detected:** yes' in report
    assert 'Time-averaged loss' in report
    assert 'Median local rate' in report


# ---------------------------------------------------------------------------
# Plotting smoke tests
# ---------------------------------------------------------------------------

def test_plot_decomposition_returns_figure():
    s = _make_series()
    _, _, info = signal_decomposition.degradation(s)
    fig = signal_decomposition.plot_decomposition(info)
    assert isinstance(fig, plt.Figure)
    plt.close('all')


def test_format_degradation_report_linear():
    s = _make_series()
    _, _, info = signal_decomposition.degradation(s, trend_type='linear')
    report = signal_decomposition.format_degradation_report(info)
    assert isinstance(report, str)
    assert 'Degradation report' in report
    assert '%/yr' in report


def test_format_degradation_report_pwl():
    s = _make_series()
    _, _, info = signal_decomposition.degradation(s, trend_type='pwl')
    report = signal_decomposition.format_degradation_report(info)
    assert 'Pre-breakpoint' in report
    assert 'Post-breakpoint' in report


def test_format_degradation_report_monotone():
    s = _make_series()
    _, _, info = signal_decomposition.degradation(s, trend_type='monotone')
    report = signal_decomposition.format_degradation_report(info)
    assert 'Year' in report


# ---------------------------------------------------------------------------
# Stability analysis smoke tests (1.5-year series)
# ---------------------------------------------------------------------------

def test_get_valid_endpoints():
    s = _make_short_series()
    endpoints = signal_decomposition.get_valid_endpoints(s.values, step=30, T=T)
    assert len(endpoints) >= 2
    assert endpoints[0] >= int(T)
    assert endpoints[-1] <= len(s)


def test_analyze_fit_stability_returns_dict():
    s = _make_short_series()
    stability = signal_decomposition.analyze_fit_stability(
        s.values,
        make_problem_kwargs={'trend_type': 'linear', 'loss': 'l2'},
        step=30,
    )
    for key in ('n_values', 'rates', 'rate_history', 'x2_snapshots',
                'x2_delta', 'x2_rmsd', 'rate_delta', 'converged_at',
                'convergence_tol'):
        assert key in stability


def test_plot_stability_returns_figure():
    s = _make_short_series()
    stability = signal_decomposition.analyze_fit_stability(
        s.values,
        make_problem_kwargs={'trend_type': 'linear', 'loss': 'l2'},
        step=30,
    )
    fig = signal_decomposition.plot_stability(stability)
    assert isinstance(fig, plt.Figure)
    plt.close('all')
