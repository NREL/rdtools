"""Tests for rdtools.signal_decomposition."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from rdtools import signal_decomposition

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

    def test_rd_ci_is_nan_stub(self):
        s = _make_series()
        _, Rd_CI, _ = signal_decomposition.degradation(s)
        assert Rd_CI.shape == (2,)
        assert np.all(np.isnan(Rd_CI))

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
