"""
Signal decomposition degradation analysis for PV systems.

Seasonal-trend decomposition via convex optimization:

    y = x1 (seasonal) + x2 (trend) + x3 (residual)

Trend types: ``'linear'``, ``'pwl'`` (piecewise-linear), ``'monotone'``.
Loss functions: ``'l2'``, ``'l1'``, ``'huber'``, ``'quantile'``.
Missing data is handled natively via a masked equality constraint.

Public entry point::

    from rdtools.signal_decomposition import degradation
    Rd_pct, Rd_CI, sd_trend_results = degradation(energy_normalized)

Diagnostic / stability tools::

    from rdtools.signal_decomposition import (
        analyze_fit_stability, plot_stability,
        get_valid_endpoints, animate_degradation,
        plot_decomposition, format_degradation_report,
    )
"""

import inspect
import warnings
from itertools import pairwise

import cvxpy as cp
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spcqe


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_kwargs(func, locals_dict):
    """Return a dict of only the parameters that belong to *func*."""
    return {k: locals_dict[k] for k in inspect.signature(func).parameters}


def _end_drop_weights(N, frac_start=0.90, frac_end=0.95, end_scale=1.0, base=0.0):
    """
    Weights for the first-difference (drop) penalty, length N-1.

    base       : weight in the bulk of the series (0.0 = no extra penalty there).
    end_scale  : weight reached in the end region. LARGER = stronger prior
                 that end drops are unlikely -> flatter tail.
    frac_start : fraction of N where the end region begins.
    frac_end   : fraction of N where end_scale is fully reached.
    """
    L = N - 1
    pos = (np.arange(L) + 0.5) / N            # location of each difference
    ramp = np.clip((pos - frac_start) / (frac_end - frac_start), 0.0, 1.0)
    return base + (end_scale - base) * ramp


# ---------------------------------------------------------------------------
# Core decomposition
# ---------------------------------------------------------------------------

def make_problem(
    y,
    numharmonics=6,
    trend_type='linear',
    loss='l2',
    lam_seasonal=1e-1,
    lam_trend=1e0,
    lam_end=0.0,
    end_frac=(0.90, 0.95),
    q=0.75,
    huber_M=1.0,
    include_soiling=False,
    lam_soiling_down=None,
    lam_soiling_value=None,
    T=365.2425,
):
    """
    Build the convex seasonal-trend decomposition problem.

    Solves ``y = x1 + x2 + x3`` plus an optional named soiling component
    ``s``, giving ``y = x1 + x2 + s + x3``, where:

    - **x1** – Seasonal component expressed as ``B @ theta`` (truncated Fourier
      basis), regularised with ``lam_seasonal * ||W @ theta||_2^2``.
    - **x2** – Trend component, one of:

      - ``'linear'``   : global affine ``a + b*t``
      - ``'pwl'``      : piecewise-linear with one breakpoint after the first
        year (index ``int(T)``), continuity enforced at the knot
      - ``'monotone'`` : free signal constrained to be non-increasing
        (``diff(x2) <= 0``), regularised on second differences plus an
        optional end-drop penalty (see *lam_end* / *end_frac*)

    - **x3** – Residual, penalised according to *loss*:

      - ``'l2'``       : ``(1/N) * ||x3||_2^2``
      - ``'l1'``       : ``(1/N) * ||x3||_1``
      - ``'huber'``    : ``(1/N) * sum(huber_M(x3))``
      - ``'quantile'`` : pinball loss at quantile level *q*

    - **s** – Optional nonpositive log-soiling component. Its downward
      increments receive an L2 penalty and its magnitude a weighted L1
      penalty. The weights default to one, reproducing the convex model used
      by the soiling selector. After selection, :func:`degradation` updates
      these weights for two cleaning-interval IRL1 debiasing solves.

    NaN entries in *y* are treated as missing: the equality constraint
    ``y == x1 + x2 + x3`` is imposed only at non-NaN indices.

    Parameters
    ----------
    y : array-like, shape (N,)
        Observed signal.
    numharmonics : int
        Number of Fourier harmonic pairs for the seasonal component.
    trend_type : str
        ``'linear'``, ``'pwl'``, or ``'monotone'``.
    loss : str
        ``'l2'``, ``'l1'``, ``'huber'``, or ``'quantile'``.
    lam_seasonal : float
        Regularisation weight on Fourier coefficients.
    lam_trend : float
        Regularisation weight on trend smoothness (second differences for
        ``'monotone'``; slope magnitude for ``'linear'`` and ``'pwl'``).
    lam_end : float
        End-drop penalty weight; only used when ``trend_type='monotone'``.
        Adds a weighted penalty on first differences (drops) in the tail
        region of the series to discourage large, late-record declines that
        may be artefacts of noise or missing data. ``0.0`` disables the
        penalty entirely (default).
    end_frac : tuple of float, (frac_start, frac_end)
        Fraction-of-record positions at which the end-drop penalty ramps
        from zero to *lam_end*. Only used when ``trend_type='monotone'``
        and ``lam_end > 0``. Default ``(0.90, 0.95)``.
    q : float
        Quantile level in (0, 1); only used when ``loss='quantile'``.
    huber_M : float
        Huber threshold; only used when ``loss='huber'``.
    include_soiling : bool
        Include the nonpositive log-soiling component.
    lam_soiling_down : float or None
        L2 weight on downward soiling increments. Required when
        ``include_soiling=True``.
    lam_soiling_value : float or None
        L1 weight on the soiling component magnitude. Required when
        ``include_soiling=True``.
    T : float
        Period length in samples (default 365.2425 for daily data).

    Returns
    -------
    dict with keys:

    - ``'problem'``   : :class:`cvxpy.Problem` — call ``.solve()`` before
      inspecting variables.
    - ``'variables'`` : dict of CVXPY expressions (``'theta'``, ``'x1'``,
      ``'x2'``, ``'x3'``, optional ``'soiling'``, plus trend-specific scalars).
    - ``'parameters'``: reusable CVXPY soiling-weight parameters when
      ``include_soiling=True``.
    - ``'args'``      : dict of the kwargs actually passed to this function.
    """
    y = np.asarray(y, dtype=float)
    N = len(y)
    t = np.arange(N, dtype=float)

    # ------------------------------------------------------------------
    # 1.  Seasonal component  x1 = B @ theta
    # ------------------------------------------------------------------
    B = spcqe.make_basis_matrix(numharmonics, N, [T])[:, 1:]
    W = spcqe.make_regularization_matrix(numharmonics, lam_seasonal, [T]).tocsr()[:, 1:]

    theta = cp.Variable(B.shape[1], name='theta')
    x1 = B @ theta
    seasonal_reg = cp.sum_squares(W @ theta)

    # ------------------------------------------------------------------
    # 2.  Trend component  x2
    # ------------------------------------------------------------------
    knot = int(T)

    if trend_type == 'linear':
        trend_coeffs = cp.Variable(2, name='trend_coeffs')
        x2 = trend_coeffs[0] + trend_coeffs[1] * t
        trend_constraints = []
        trend_reg = lam_trend * cp.sum_squares(trend_coeffs[1])

    elif trend_type == 'pwl':
        a0 = cp.Variable(name='a0')
        b0 = cp.Variable(name='b0')
        b1 = cp.Variable(name='b1')
        hinge = np.maximum(t - knot, 0.0)
        x2 = a0 + b0 * t + b1 * hinge
        trend_constraints = []
        trend_reg = lam_trend * cp.square(b1)

    elif trend_type == 'monotone':
        x2 = cp.Variable(N, name='x2_monotone')
        trend_constraints = [cp.diff(x2) <= 0]
        drop = cp.diff(x2)
        w = _end_drop_weights(N, frac_start=end_frac[0], frac_end=end_frac[1],
                              end_scale=lam_end)
        trend_reg = (lam_trend * cp.sum_squares(cp.diff(x2, k=2))
                     + cp.sum_squares(cp.multiply(w, drop)))

    else:
        raise ValueError(
            f"trend_type must be 'linear', 'pwl', or 'monotone'; got '{trend_type}'"
        )

    # ------------------------------------------------------------------
    # 2b. Optional nonpositive soiling component
    # ------------------------------------------------------------------
    if include_soiling:
        if lam_soiling_down is None or lam_soiling_value is None:
            raise ValueError(
                "lam_soiling_down and lam_soiling_value are required when "
                "include_soiling=True"
            )
        soiling = cp.Variable(N, name='soiling')
        soiling_down = cp.Parameter(
            nonneg=True, value=float(lam_soiling_down), name='lam_soiling_down'
        )
        # lam_soiling_value is a validated fixed coefficient. Keeping it a
        # numeric constant avoids a product of two Parameters below, which
        # would be convex but not DPP.
        soiling_value = float(lam_soiling_value)
        soiling_value_weights = cp.Parameter(
            N, nonneg=True, value=np.ones(N), name='soiling_value_weights'
        )
        soiling_constraints = [soiling <= 0]
        soiling_reg = (
            soiling_down * cp.norm2(cp.neg(cp.diff(soiling)))
            + soiling_value * cp.sum(
                cp.multiply(soiling_value_weights, cp.abs(soiling))
            )
        )
    else:
        soiling = 0
        soiling_constraints = []
        soiling_reg = 0

    # ------------------------------------------------------------------
    # 3.  Residual  x3 = y - x1 - x2 (- soiling)
    # ------------------------------------------------------------------
    good_data = ~np.isnan(y)
    x3 = cp.Variable(N)

    if loss == 'l2':
        data_fidelity = (1.0 / N) * cp.sum_squares(x3)

    elif loss == 'l1':
        data_fidelity = (1.0 / N) * cp.norm1(x3)

    elif loss == 'huber':
        data_fidelity = (1.0 / N) * cp.sum(cp.huber(x3, huber_M))

    elif loss == 'quantile':
        data_fidelity = 2 * (1.0 / N) * (
            q * cp.sum(cp.pos(x3)) + (1 - q) * cp.sum(cp.pos(-x3))
        )

    else:
        raise ValueError(
            f"loss must be 'l2', 'l1', 'huber', or 'quantile'; got '{loss}'"
        )

    # ------------------------------------------------------------------
    # 4.  Objective and problem
    # ------------------------------------------------------------------
    objective = cp.Minimize(data_fidelity + seasonal_reg + trend_reg + soiling_reg)
    constraints = trend_constraints + soiling_constraints
    constraints.append(y[good_data] == (x1 + x2 + soiling + x3)[good_data])
    problem = cp.Problem(objective, constraints)

    if trend_type == 'linear':
        variables = {'theta': theta, 'trend_coeffs': trend_coeffs,
                     'x1': x1, 'x2': x2, 'x3': x3}
    elif trend_type == 'pwl':
        variables = {'theta': theta, 'a0': a0, 'b0': b0, 'b1': b1,
                     'x1': x1, 'x2': x2, 'x3': x3}
    elif trend_type == 'monotone':
        variables = {'theta': theta, 'x1': x1, 'x2': x2, 'x3': x3}

    if include_soiling:
        variables['soiling'] = soiling

    out = {
        'problem': problem,
        'variables': variables,
        'args': _get_kwargs(make_problem, locals()),
    }
    if include_soiling:
        out['parameters'] = {
            'lam_soiling_down': soiling_down,
            'soiling_value_weights': soiling_value_weights,
        }
    return out


def prepare_input(y, log_transform=False, floor=1e-6):
    """
    Prepare *y* for :func:`make_problem`.

    If ``log_transform=True``, takes ``log(y)`` and additionally masks
    non-positive values (which would produce ``-inf``).

    Parameters
    ----------
    y : array-like
        Input signal.
    log_transform : bool
        Apply natural-log transform before decomposition.
    floor : float
        Values at or below this are treated as missing before taking log.

    Returns
    -------
    numpy.ndarray
        Transformed signal with NaN where values are invalid.
    """
    y = np.asarray(y, dtype=float).copy()
    if log_transform:
        y[y <= floor] = np.nan
        y = np.log(y)
    return y


def recover_components(variables, log_transform=False):
    """
    Back-transform solved component arrays to the original domain.

    Parameters
    ----------
    variables : dict
        The ``'variables'`` dict from :func:`make_problem` after solving.
    log_transform : bool
        Must match the flag used when building the problem.

    Returns
    -------
    dict with keys ``'x1'``, ``'x2'``, ``'x3'``, ``'fit'``, plus ``'soiling'``
    when a soiling component is present.
    """
    x1 = variables['x1'].value
    x2 = variables['x2'].value
    x3 = variables['x3'].value
    has_soiling = 'soiling' in variables
    soiling = variables['soiling'].value if has_soiling else None

    if log_transform:
        x1 = np.exp(x1)
        x2 = np.exp(x2)
        x3 = np.exp(x3)
        if has_soiling:
            soiling = np.exp(soiling)
            fit = x1 * x2 * soiling
        else:
            fit = x1 * x2
    else:
        fit = x1 + x2 + soiling if has_soiling else x1 + x2

    out = {'x1': x1, 'x2': x2, 'x3': x3, 'fit': fit}
    if has_soiling:
        out['soiling'] = soiling
    return out


_SOILING_LAM_DOWN_GRID = np.logspace(-3.5, -1.5, 9)
_SOILING_LAM_VALUE = 10 ** -5.8
_SOILING_Q75_MIN = 0.0075
_SOILING_MAX_RECOVERIES_PER_YEAR = 30.0
_SOILING_MAX_NEIGHBOR_NRMSE = 0.75
_SOILING_MIN_NEIGHBOR_CORRELATION = 0.90
_SOILING_RECOVERY_THRESHOLD = 0.005
_SOILING_IRL1_EPSILON = 0.01
_SOILING_IRL1_ITERATIONS = 2


def _centered_correlation(left, right):
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-12:
        return np.nan
    return float(np.dot(left, right) / denominator)


def _soiling_candidate_metrics(paths, statuses, T):
    """Return frozen structural metrics for one regularization path."""
    n_candidates, n_samples = paths.shape
    years = max((n_samples - 1) / T, 1 / T)
    rows = []
    for index, (weight, path, status) in enumerate(
            zip(_SOILING_LAM_DOWN_GRID, paths, statuses)):
        finite = np.all(np.isfinite(path))
        loss = -path if finite else np.full(n_samples, np.nan)
        q75 = float(np.quantile(loss, 0.75)) if finite else np.nan
        recoveries = (
            float(np.sum(np.diff(path) >= _SOILING_RECOVERY_THRESHOLD) / years)
            if finite else np.nan
        )
        nrmse = np.nan
        correlation = np.nan
        if 0 < index < n_candidates - 1 and finite:
            left = paths[index - 1]
            right = paths[index + 1]
            if np.all(np.isfinite(left)) and np.all(np.isfinite(right)):
                scale = max(q75, 0.001)
                nrmse = max(
                    np.sqrt(np.mean((path - left) ** 2)) / scale,
                    np.sqrt(np.mean((path - right) ** 2)) / scale,
                )
                correlations = [
                    _centered_correlation(path, left),
                    _centered_correlation(path, right),
                ]
                if np.all(np.isfinite(correlations)):
                    correlation = min(correlations)
        qualifies = bool(
            0 < index < n_candidates - 1
            and status in ('optimal', 'optimal_inaccurate')
            and np.isfinite(q75)
            and q75 >= _SOILING_Q75_MIN
            and recoveries <= _SOILING_MAX_RECOVERIES_PER_YEAR
            and nrmse <= _SOILING_MAX_NEIGHBOR_NRMSE
            and correlation >= _SOILING_MIN_NEIGHBOR_CORRELATION
        )
        rows.append({
            'candidate_index': index,
            'lam_soiling_down': float(weight),
            'log10_lam_soiling_down': float(np.log10(weight)),
            'status': status,
            'q75_loss': q75,
            'recoveries_per_year': recoveries,
            'worst_neighbor_nrmse': nrmse,
            'worst_neighbor_correlation': correlation,
            'qualifies': qualifies,
        })
    return pd.DataFrame(rows)


def _solve_soiling_path(build, y, T):
    """Solve the frozen path and return metrics, selection, and fallback."""
    problem = build['problem']
    parameter = build['parameters']['lam_soiling_down']
    variables = build['variables']
    observed = np.isfinite(y)
    paths = []
    statuses = []
    for weight in _SOILING_LAM_DOWN_GRID:
        parameter.value = float(weight)
        status = 'solver_error'
        path = np.full(len(y), np.nan)
        try:
            problem.solve(solver=cp.CLARABEL, warm_start=True)
            status = problem.status
            values = [variables[key].value for key in ('x1', 'x2', 'soiling', 'x3')]
            if status in ('optimal', 'optimal_inaccurate') and all(
                    value is not None and np.all(np.isfinite(value)) for value in values):
                reconstructed = sum(np.asarray(value) for value in values)
                if np.allclose(
                        reconstructed[observed], y[observed], rtol=1e-5, atol=1e-6):
                    path = np.asarray(variables['soiling'].value, dtype=float).copy()
                else:
                    status = 'inconsistent_reconstruction'
        except cp.error.SolverError:
            pass
        paths.append(path)
        statuses.append(status)
    paths = np.asarray(paths)
    metrics = _soiling_candidate_metrics(paths, statuses, T)
    qualifying = metrics.index[metrics['qualifies']].to_numpy()
    selected_index = int(qualifying[0]) if len(qualifying) else None
    return metrics, selected_index, paths[-1].copy()


def _soiling_interval_value_weights(path, epsilon=_SOILING_IRL1_EPSILON):
    """Return one IRL1 value weight per inferred cleaning interval.

    A material upward log step starts a new cleaning-to-cleaning interval.
    Every sample in an interval receives

    ``epsilon / (quantile(abs(path[interval]), 0.75) + epsilon)``.

    Thus a clean or nearly clean interval retains the original L1 penalty
    (weight near one), while an already-established soiling interval receives
    less value shrinkage. Weights depend only on the preceding estimate, never
    on observations or truth, so the next subproblem remains convex.
    """
    path = np.asarray(path, dtype=float)
    recoveries = np.flatnonzero(
        np.diff(path) >= _SOILING_RECOVERY_THRESHOLD
    ) + 1
    boundaries = np.r_[0, recoveries, len(path)]
    weights = np.ones(len(path), dtype=float)
    for start, stop in pairwise(boundaries):
        depth = np.quantile(np.abs(path[start:stop]), 0.75)
        weights[start:stop] = epsilon / (depth + epsilon)
    return weights


def _refine_soiling_irl1(
        build, y, iterations=_SOILING_IRL1_ITERATIONS,
        epsilon=_SOILING_IRL1_EPSILON, warn_on_failure=True):
    """Debias a selected soiling fit with interval-weighted IRL1 solves.

    The input problem must already contain the soiling component selected by
    the frozen regularization-path rule. Each iteration derives fixed interval
    weights from the preceding soiling estimate, assigns the vector CVXPY
    Parameter, and resolves the otherwise unchanged convex decomposition.
    This is a sequence of convex programs; it does not alter path selection or
    turn a structural null into a positive result.

    If an iteration fails validation, restore and resolve the preceding
    successful weighting rather than returning variables from a failed solve.
    The returned diagnostics state how many of the requested iterations were
    completed.
    """
    problem = build['problem']
    variables = build['variables']
    parameter = build['parameters']['soiling_value_weights']
    observed = np.isfinite(y)
    previous = np.asarray(variables['soiling'].value, dtype=float).copy()
    last_weights = np.ones(len(previous), dtype=float)
    rows = []
    for iteration in range(1, iterations + 1):
        weights = _soiling_interval_value_weights(previous, epsilon)
        parameter.value = weights
        status = 'solver_error'
        valid = False
        try:
            problem.solve(solver=cp.CLARABEL, warm_start=True)
            status = problem.status
            values = [variables[key].value for key in ('x1', 'x2', 'soiling', 'x3')]
            if status in ('optimal', 'optimal_inaccurate') and all(
                    value is not None and np.all(np.isfinite(value)) for value in values):
                reconstructed = sum(np.asarray(value) for value in values)
                valid = np.allclose(
                    reconstructed[observed], y[observed], rtol=1e-5, atol=1e-6
                )
        except cp.error.SolverError:
            pass
        rows.append({
            'iteration': iteration,
            'status': status,
            'valid': bool(valid),
            'weight_min': float(np.min(weights)),
            'weight_median': float(np.median(weights)),
        })
        if not valid:
            if warn_on_failure:
                warnings.warn(
                    f'Soiling IRL1 iteration {iteration} failed; returning '
                    f'the preceding successful fit.',
                    UserWarning,
                    stacklevel=2,
                )
            parameter.value = last_weights
            problem.solve(solver=cp.CLARABEL, warm_start=True)
            break
        previous = np.asarray(variables['soiling'].value, dtype=float).copy()
        last_weights = weights.copy()
    return {
        'method': 'cleaning_interval_irl1',
        'epsilon': float(epsilon),
        'requested_iterations': int(iterations),
        'completed_iterations': int(sum(row['valid'] for row in rows)),
        'iterations': pd.DataFrame(rows),
    }


def _soiling_intervals(soiling_ratio, index, min_interval_days=7):
    """Fit compound local rates between material recovery events."""
    ratio = np.asarray(soiling_ratio, dtype=float)
    log_ratio = np.log(np.clip(ratio, 1e-12, None))
    recoveries = np.flatnonzero(np.diff(log_ratio) >= _SOILING_RECOVERY_THRESHOLD) + 1
    boundaries = np.r_[0, recoveries, len(ratio)]
    rows = []
    for interval_index, (start, stop) in enumerate(pairwise(boundaries)):
        positions = np.arange(start, stop)
        valid = np.isfinite(log_ratio[positions])
        positions = positions[valid]
        if len(positions) >= 2:
            slope = np.polyfit(positions - positions[0], log_ratio[positions], 1)[0]
            rate = float((np.exp(slope) - 1) * 100)
        else:
            rate = np.nan
        length = stop - start
        is_valid = bool(length >= min_interval_days and np.isfinite(rate) and rate <= 0)
        recovery = (
            float((np.exp(log_ratio[stop] - log_ratio[stop - 1]) - 1) * 100)
            if stop < len(ratio) else np.nan
        )
        rows.append({
            'interval': interval_index,
            'start': index[start],
            'end': index[stop - 1],
            'length_days': length,
            'soiling_rate_pct_day': rate,
            'start_soiling_ratio': ratio[start],
            'end_soiling_ratio': ratio[stop - 1],
            'subsequent_recovery_pct': recovery,
            'valid': is_valid,
        })
    return pd.DataFrame(rows)


def _soiling_rate_summary(intervals, index):
    """Day-weight interval rates overall and by climatological quarter."""
    rows = []
    periods = [('overall', np.ones(len(index), dtype=bool))]
    periods.extend(
        (f'Q{quarter}', index.quarter == quarter) for quarter in range(1, 5)
    )
    for label, period_mask in periods:
        rates = []
        contributing_intervals = 0
        for _, interval in intervals[intervals['valid']].iterrows():
            overlap = (
                (index >= interval['start'])
                & (index <= interval['end'])
                & period_mask
            ).sum()
            contributing_intervals += int(overlap > 0)
            rates.extend([interval['soiling_rate_pct_day']] * int(overlap))
        rows.append({
            'period': label,
            'median_rate_pct_day': float(np.median(rates)) if rates else np.nan,
            'rate_ci_low': np.nan,
            'rate_ci_high': np.nan,
            'interval_count': contributing_intervals,
            'day_count': len(rates),
        })
    return pd.DataFrame(rows)


def _soiling_loss_metrics(
        soiling_ratio, index, insolation_daily=None, warn_missing_insolation=True):
    """Time-average losses and optional insolation-weighted losses."""
    ratio = np.asarray(soiling_ratio, dtype=float)
    quarters = index.quarter
    rows = []
    for quarter in range(1, 5):
        mask = quarters == quarter
        rows.append({
            'quarter': quarter,
            'time_averaged_loss_pct': float(100 * np.nanmean(1 - ratio[mask])),
        })
    output = {
        'time_averaged_loss_pct': float(100 * np.nanmean(1 - ratio)),
        'time_averaged_loss_ci': np.array([np.nan, np.nan]),
        'quarterly': pd.DataFrame(rows),
    }
    output['quarterly']['time_averaged_loss_ci_low'] = np.nan
    output['quarterly']['time_averaged_loss_ci_high'] = np.nan
    if insolation_daily is not None:
        if not isinstance(insolation_daily, pd.Series):
            raise TypeError('insolation_daily must be a pandas.Series')
        if not isinstance(insolation_daily.index, pd.DatetimeIndex):
            raise TypeError('insolation_daily must have a DatetimeIndex')
        if insolation_daily.index.has_duplicates:
            raise ValueError('insolation_daily index must not contain duplicates')
        weights = insolation_daily.reindex(index).to_numpy(dtype=float)
        if np.any(weights[np.isfinite(weights)] < 0):
            raise ValueError('insolation_daily must be nonnegative')

        def weighted(mask):
            valid = mask & np.isfinite(ratio) & np.isfinite(weights) & (weights > 0)
            if not valid.any():
                return np.nan
            weighted_ratio = (
                np.sum(weights[valid] * ratio[valid]) / np.sum(weights[valid])
            )
            return float(100 * (1 - weighted_ratio))

        output['insolation_weighted_loss_pct'] = weighted(np.ones(len(index), dtype=bool))
        output['insolation_weighted_loss_ci'] = np.array([np.nan, np.nan])
        quarterly_weighted = [
            weighted(quarters == quarter) for quarter in range(1, 5)
        ]
        output['quarterly']['insolation_weighted_loss_pct'] = quarterly_weighted
        missing_quarters = [
            str(quarter) for quarter, value in enumerate(quarterly_weighted, 1)
            if not np.isfinite(value)
        ]
        if missing_quarters and warn_missing_insolation:
            warnings.warn(
                'No positive, finite insolation is available for quarter(s) '
                + ', '.join(missing_quarters)
                + '; corresponding insolation-weighted losses are NaN.',
                UserWarning,
                stacklevel=2,
            )
        output['quarterly']['insolation_weighted_loss_ci_low'] = np.nan
        output['quarterly']['insolation_weighted_loss_ci_high'] = np.nan
    return output


# ---------------------------------------------------------------------------
# Rate extraction
# ---------------------------------------------------------------------------

def extract_degradation_rate(variables, trend_type, T=365.2425):
    """
    Extract degradation rate(s) from a solved decomposition problem.

    Returns rates as average percentage change per year, where 1 year = *T*
    samples. A negative rate means the signal is declining (degrading).

    For the monotone trend type, small solver floating-point errors can
    accumulate over long flat runs of the trend signal, causing the
    non-increasing constraint to be violated by amounts at the solver
    tolerance (~1e-6). The extracted x2 is projected onto the set of
    non-increasing sequences via cumulative minimum before any rates are
    computed. This has no visible effect on plots but prevents spurious
    positive rates in the output.

    Parameters
    ----------
    variables : dict
        The ``'variables'`` dict returned by :func:`make_problem` after solving.
    trend_type : str
        ``'linear'``, ``'pwl'``, or ``'monotone'``.
    T : float
        Samples per year (default 365.2425).

    Returns
    -------
    dict
        Keys depend on *trend_type*:

        - linear: ``'rate_pct_yr'``
        - pwl: ``'rate_pre_pct_yr'``, ``'rate_post_pct_yr'``,
          ``'rate_overall_pct_yr'``
        - monotone: ``'rate_overall_pct_yr'``,
          ``'rate_instantaneous_pct_yr'``, ``'rate_yearly_pct_yr'``,
          ``'year_boundaries'``
    """
    x2 = variables['x2'].value

    if x2 is None:
        raise ValueError("x2.value is None — has the problem been solved?")

    if trend_type == 'monotone':
        x2 = np.minimum.accumulate(x2)

    N = len(x2)
    x2_0 = x2[0]
    x2_end = x2[-1]
    span = N - 1

    def to_pct_yr(delta_per_sample):
        return (delta_per_sample * T / x2_0) * 100.0

    if trend_type == 'linear':
        b = variables['trend_coeffs'].value[1]
        return {'rate_pct_yr': to_pct_yr(b)}

    elif trend_type == 'pwl':
        b0 = variables['b0'].value
        b1 = variables['b1'].value
        overall_slope = (x2_end - x2_0) / span
        return {
            'rate_pre_pct_yr':     to_pct_yr(b0),
            'rate_post_pct_yr':    to_pct_yr(b0 + b1),
            'rate_overall_pct_yr': to_pct_yr(overall_slope),
        }

    elif trend_type == 'monotone':
        overall_slope = (x2_end - x2_0) / span
        rate_overall = to_pct_yr(overall_slope)

        dx = np.diff(x2)
        rate_inst = (dx * T / x2_0) * 100.0

        boundaries = np.arange(0, N, T)
        boundaries = np.append(boundaries, N)
        boundaries = np.round(boundaries).astype(int)
        boundaries = np.clip(boundaries, 0, N)

        yearly_rates = []
        for i in range(len(boundaries) - 1):
            i0, i1 = boundaries[i], boundaries[i + 1]
            seg_len = i1 - i0
            if seg_len < 2:
                continue
            seg_slope = (x2[i1 - 1] - x2[i0]) / (seg_len - 1)
            yearly_rates.append(to_pct_yr(seg_slope))

        return {
            'rate_overall_pct_yr':       rate_overall,
            'rate_instantaneous_pct_yr': rate_inst,
            'rate_yearly_pct_yr':        np.array(yearly_rates),
            'year_boundaries':           boundaries,
        }

    else:
        raise ValueError(
            f"trend_type must be 'linear', 'pwl', or 'monotone'; got '{trend_type}'"
        )


def extract_degradation_rate_log(variables, trend_type, T=365.2425):
    """
    Degradation rate extraction for log-transformed fits.

    In log space the trend is ``log(x2)``, so the chord slope is already a
    log-ratio. The annualised compound rate is::

        rate = (exp(slope * T) - 1) * 100%

    which equals ``(x2(N-1)/x2(0))^(T/(N-1)) - 1`` in the original domain —
    a standard compound annual rate, independent of the reference level.

    Interface is identical to :func:`extract_degradation_rate`; pass
    log-space variables directly (do not exponentiate first).

    Parameters
    ----------
    variables : dict
        The ``'variables'`` dict returned by :func:`make_problem` after solving.
    trend_type : str
        ``'linear'``, ``'pwl'``, or ``'monotone'``.
    T : float
        Samples per year (default 365.2425).

    Returns
    -------
    dict
        Same keys as :func:`extract_degradation_rate`, plus ``'year_labels'``
        (``'full'`` / ``'partial'``) for the monotone case.
    """
    def to_pct_yr_log(slope_per_sample):
        return (np.exp(slope_per_sample * T) - 1) * 100.0

    x2_log = variables['x2'].value
    if x2_log is None:
        raise ValueError("x2.value is None — has the problem been solved?")

    if trend_type == 'monotone':
        x2_log = np.minimum.accumulate(x2_log)

    N = len(x2_log)
    span = N - 1

    if trend_type == 'linear':
        b = variables['trend_coeffs'].value[1]
        return {'rate_pct_yr': to_pct_yr_log(b)}

    elif trend_type == 'pwl':
        b0 = variables['b0'].value
        b1 = variables['b1'].value
        overall_slope = (x2_log[-1] - x2_log[0]) / span
        return {
            'rate_pre_pct_yr':     to_pct_yr_log(b0),
            'rate_post_pct_yr':    to_pct_yr_log(b0 + b1),
            'rate_overall_pct_yr': to_pct_yr_log(overall_slope),
        }

    elif trend_type == 'monotone':
        overall_slope = (x2_log[-1] - x2_log[0]) / span
        dx = np.diff(x2_log)

        MIN_SEGMENT_FRACTION = 0.5
        boundaries = np.arange(0, N, T)
        boundaries = np.append(boundaries, N)
        boundaries = np.clip(np.round(boundaries).astype(int), 0, N)

        yearly_rates = []
        yearly_labels = []
        for i in range(len(boundaries) - 1):
            i0, i1 = boundaries[i], boundaries[i + 1]
            seg_len = i1 - i0
            if seg_len < 2:
                continue
            seg_slope = min((x2_log[i1 - 1] - x2_log[i0]) / (seg_len - 1), 0.0)
            yearly_rates.append(to_pct_yr_log(seg_slope))
            yearly_labels.append(
                'partial' if seg_len < MIN_SEGMENT_FRACTION * T else 'full'
            )

        return {
            'rate_overall_pct_yr':       to_pct_yr_log(overall_slope),
            'rate_instantaneous_pct_yr': (np.exp(dx * T) - 1) * 100.0,
            'rate_yearly_pct_yr':        np.array(yearly_rates),
            'year_labels':               yearly_labels,
            'year_boundaries':           boundaries,
        }

    else:
        raise ValueError(
            f"trend_type must be 'linear', 'pwl', or 'monotone'; got '{trend_type}'"
        )


_MIN_SUCCESS_FRAC = 0.5


def _bootstrap_ci(
    fit,
    residuals,
    nan_mask,
    make_problem_args,
    trend_type,
    log_transform,
    T,
    n_bootstrap,
    block_size,
    confidence_level,
    random_state,
    soiling_context=None,
):
    if n_bootstrap == 0:
        return {}, {}

    rng = np.random.default_rng(random_state)
    if block_size is None:
        block_size = int(T)

    valid_idx = np.where(~nan_mask)[0]
    res_valid = residuals[valid_idx]
    M = res_valid.size
    L = min(block_size, M)
    n_blocks = int(np.ceil(M / L))

    extractor = extract_degradation_rate_log if log_transform else extract_degradation_rate
    _skip = {'rate_instantaneous_pct_yr'}

    collected = {}
    soiling_collected = {}
    for _ in range(n_bootstrap):
        starts = rng.integers(0, M - L + 1, size=n_blocks)
        resampled = np.concatenate([res_valid[s:s + L] for s in starts])[:M]

        y_star = fit.copy()
        y_star[valid_idx] += resampled
        y_star[nan_mask] = np.nan

        try:
            b = make_problem(y_star, **make_problem_args)
            b['problem'].solve(solver=cp.CLARABEL)
            if b['problem'].status not in ('optimal', 'optimal_inaccurate'):
                continue
            # Bootstrap the final estimator, including its two deterministic
            # post-selection IRL1 steps. Re-running only the uniform-L1 model
            # would give intervals for a different estimator than the point fit.
            if soiling_context is not None and not soiling_context['null_model']:
                _refine_soiling_irl1(b, y_star, warn_on_failure=False)
                if b['problem'].status not in ('optimal', 'optimal_inaccurate'):
                    continue
            rates = extractor(b['variables'], trend_type, T=T)
            for k, v in rates.items():
                if k.startswith('rate_') and k not in _skip:
                    collected.setdefault(k, []).append(v)
            if soiling_context is not None:
                if soiling_context['null_model']:
                    ratio = np.ones(len(y_star))
                else:
                    ratio = np.exp(np.asarray(b['variables']['soiling'].value))
                index = soiling_context['index']
                loss_metrics = _soiling_loss_metrics(
                    ratio, index, soiling_context['insolation_daily'],
                    warn_missing_insolation=False,
                )
                intervals = _soiling_intervals(ratio, index)
                rate_summary = _soiling_rate_summary(intervals, index)
                values = {
                    'time_loss_overall': loss_metrics['time_averaged_loss_pct'],
                    **{
                        f'time_loss_Q{row.quarter}': row.time_averaged_loss_pct
                        for row in loss_metrics['quarterly'].itertuples()
                    },
                    **{
                        f'rate_{row.period}': row.median_rate_pct_day
                        for row in rate_summary.itertuples()
                    },
                }
                if 'insolation_weighted_loss_pct' in loss_metrics:
                    values['insolation_loss_overall'] = loss_metrics[
                        'insolation_weighted_loss_pct'
                    ]
                    values.update({
                        f'insolation_loss_Q{row.quarter}':
                            row.insolation_weighted_loss_pct
                        for row in loss_metrics['quarterly'].itertuples()
                    })
                for key, value in values.items():
                    if np.isfinite(value):
                        soiling_collected.setdefault(key, []).append(value)
        except Exception:
            continue

    lower_pct = (100 - confidence_level) / 2
    upper_pct = 100 - lower_pct

    n_success = len(next(iter(collected.values()), []))
    if n_success < _MIN_SUCCESS_FRAC * n_bootstrap:
        return {}, {}

    rate_ci = {
        k: np.percentile(np.array(v), [lower_pct, upper_pct], axis=0)
        for k, v in collected.items()
    }
    soiling_ci = {
        key: np.percentile(values, [lower_pct, upper_pct])
        for key, values in soiling_collected.items()
        if len(values) >= _MIN_SUCCESS_FRAC * n_bootstrap
    }
    return rate_ci, soiling_ci


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------

def get_valid_endpoints(y_full, step=10, T=365.2425):
    """
    Return window lengths for stability analysis, snapped to valid samples.

    Nominal window lengths are ``range(int(T), len(y_full)+1, step)`` plus
    the full length. Each is snapped backward to the nearest non-NaN sample
    and deduplicated.

    Parameters
    ----------
    y_full : array-like
        Full time series (NaN where data is missing).
    step : int
        Nominal spacing between window lengths in samples.
    T : float
        Samples per year; sets the minimum window length.

    Returns
    -------
    numpy.ndarray of int
        Unique, sorted window lengths each ending on a non-NaN sample.
    """
    y_full = np.asarray(y_full, dtype=float)
    N = len(y_full)
    valid_ix = np.where(~np.isnan(y_full))[0]

    if len(valid_ix) == 0:
        raise ValueError("y_full contains no valid (non-NaN) samples.")

    nominal = np.arange(int(T), N + 1, step)
    if nominal[-1] != N:
        nominal = np.append(nominal, N)

    snapped = []
    for n in nominal:
        candidates = valid_ix[valid_ix <= n]
        if len(candidates) == 0:
            continue
        snapped.append(candidates[-1] + 1)

    snapped = np.unique(snapped)
    snapped = snapped[snapped >= int(T)]
    return snapped


def analyze_fit_stability(
    y_full,
    make_problem_kwargs,
    step=10,
    T=365.2425,
):
    """
    Analyze how stable the decomposition estimates are as the data record grows.

    For each window ``y_full[:n]`` (where *n* is drawn from
    :func:`get_valid_endpoints`), solves the decomposition and records the
    trend component ``x2`` and the extracted degradation rates. Stability is
    quantified as how much each quantity changes between successive windows,
    measured over the region of overlap.

    Progress is printed to stdout every 20 windows.

    Parameters
    ----------
    y_full : array-like
        Full time series.
    make_problem_kwargs : dict
        Keyword arguments passed to :func:`make_problem` (do not include
        ``'y'``).
    step : int
        Nominal sample spacing between window lengths.
    T : float
        Samples per year.

    Returns
    -------
    dict with keys:

    - ``'n_values'``        : ndarray, shape (F,) — window lengths solved
    - ``'rates'``           : list of dict, length F — rate dict per window
      (None if solver failed for that window)
    - ``'rate_history'``    : dict of ndarray, shape (F,) — one entry per
      scalar rate key, NaN where not available
    - ``'x2_snapshots'``    : ndarray, shape (F, N) — trend vectors padded
      with NaN beyond the window edge
    - ``'x2_delta'``        : ndarray, shape (F-1,) — mean absolute change
      in x2 over the overlap, normalised by mean absolute x2
    - ``'x2_rmsd'``         : ndarray, shape (F-1,) — RMS difference over
      the overlap, normalised by RMS of x2
    - ``'rate_delta'``      : dict of ndarray, shape (F-1,) — absolute
      change in each scalar rate between successive windows
    - ``'converged_at'``    : dict — for each scalar rate key, the window
      length at which the rate first settles within ``convergence_tol``
      %/yr of the final value for all subsequent windows (None if never)
    - ``'convergence_tol'`` : float — tolerance used for ``converged_at``
    """
    y_full = np.asarray(y_full, dtype=float)
    make_problem_kwargs = {k: v for k, v in make_problem_kwargs.items() if k != 'y'}
    N = len(y_full)
    trend_type = make_problem_kwargs.get('trend_type', 'linear')
    frame_lengths = get_valid_endpoints(y_full, step=step, T=T)
    F = len(frame_lengths)

    convergence_tol = 0.1

    print(f"Solving {F} windows ...")
    x2_snapshots = np.full((F, N), np.nan)
    rates_list = []

    for i, n in enumerate(frame_lengths):
        if i % 20 == 0:
            print(f"  {i + 1}/{F}  (n={n})")
        build = make_problem(y_full[:n], **make_problem_kwargs)
        prob, variables = build['problem'], build['variables']
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if prob.status not in ('optimal', 'optimal_inaccurate'):
            rates_list.append(None)
            continue

        x2 = variables['x2'].value
        if trend_type == 'monotone':
            x2 = np.minimum.accumulate(x2)

        x2_snapshots[i, :n] = x2
        rates_list.append(extract_degradation_rate(variables, trend_type, T=T))

    print("Done.")

    # Rate history — collect each scalar key into its own array
    scalar_keys = set()
    for r in rates_list:
        if r is not None:
            scalar_keys.update(k for k, v in r.items() if np.isscalar(v))

    rate_history = {k: np.full(F, np.nan) for k in scalar_keys}
    for i, r in enumerate(rates_list):
        if r is None:
            continue
        for k in scalar_keys:
            if k in r:
                rate_history[k][i] = r[k]

    # x2 stability — normalised mean-absolute and RMS difference over overlap
    x2_delta = np.full(F - 1, np.nan)
    x2_rmsd = np.full(F - 1, np.nan)

    for i in range(F - 1):
        n_overlap = frame_lengths[i]
        a = x2_snapshots[i, :n_overlap]
        b = x2_snapshots[i + 1, :n_overlap]
        valid = ~(np.isnan(a) | np.isnan(b))
        if valid.sum() < 2:
            continue
        diff = b[valid] - a[valid]
        ref = a[valid]
        x2_delta[i] = np.mean(np.abs(diff)) / (np.mean(np.abs(ref)) + 1e-12)
        x2_rmsd[i] = np.sqrt(np.mean(diff**2)) / (np.sqrt(np.mean(ref**2)) + 1e-12)

    # Rate delta — absolute change in each scalar rate between windows
    rate_delta = {}
    for k, arr in rate_history.items():
        delta = np.full(F - 1, np.nan)
        for i in range(F - 1):
            if not (np.isnan(arr[i]) or np.isnan(arr[i + 1])):
                delta[i] = abs(arr[i + 1] - arr[i])
        rate_delta[k] = delta

    # Convergence detection
    converged_at = {}
    for k, arr in rate_history.items():
        valid_arr = arr[~np.isnan(arr)]
        if len(valid_arr) == 0:
            converged_at[k] = None
            continue
        final = valid_arr[-1]
        within = np.abs(arr - final) <= convergence_tol
        last_out = np.where(~within)[0]
        if len(last_out) == 0:
            converged_at[k] = int(frame_lengths[0])
        elif last_out[-1] == F - 1:
            converged_at[k] = None
        else:
            converged_at[k] = int(frame_lengths[last_out[-1] + 1])

    return {
        'n_values':        frame_lengths,
        'rates':           rates_list,
        'rate_history':    rate_history,
        'x2_snapshots':    x2_snapshots,
        'x2_delta':        x2_delta,
        'x2_rmsd':         x2_rmsd,
        'rate_delta':      rate_delta,
        'converged_at':    converged_at,
        'convergence_tol': convergence_tol,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_stability(stability, T=365.2425, figsize=(12, 10)):
    """
    Four-panel stability summary plot.

    - Row 1: All trend fits (``x2`` snapshots) coloured by window length.
    - Row 2: Normalised RMSD of ``x2`` between successive windows.
    - Row 3: Scalar rate history with convergence markers.
    - Row 4: Absolute rate change between successive windows.

    Parameters
    ----------
    stability : dict
        Output of :func:`analyze_fit_stability`.
    T : float
        Samples per year (used for x-axis labels only).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_values = stability['n_values']
    F = len(n_values)
    mid_n = (n_values[:-1] + n_values[1:]) / 2

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=False)
    fig.subplots_adjust(hspace=0.35)

    colours = cm.plasma(np.linspace(0.15, 0.9, F))

    # Row 1: x2 spaghetti
    ax = axes[0]
    snapshots = stability['x2_snapshots']
    for i in range(F):
        row = snapshots[i]
        valid = ~np.isnan(row)
        if valid.any():
            ax.plot(np.where(valid)[0], row[valid],
                    color=colours[i], lw=0.6, alpha=0.6)
    ax.set_ylabel('x2  (trend)')
    ax.set_title('Trend fits by window length', fontsize=9)
    sm = plt.cm.ScalarMappable(
        cmap='plasma',
        norm=plt.Normalize(n_values[0], n_values[-1]),
    )
    plt.colorbar(sm, ax=ax, label='Window length (samples)', pad=0.01)
    ax.spines[['top', 'right']].set_visible(False)

    # Row 2: x2 RMSD
    ax = axes[1]
    ax.plot(mid_n, stability['x2_rmsd'], color='#2a7de0', lw=1.2)
    ax.fill_between(mid_n, stability['x2_rmsd'], alpha=0.15, color='#2a7de0')
    ax.set_ylabel('Normalised RMSD')
    ax.set_xlabel('Window length (samples)')
    ax.set_title('x2 change between successive windows (normalised)', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    # Row 3: rate history
    ax = axes[2]
    palette = ['#e05c2a', '#27a865', '#2a7de0', '#9b59b6']
    for (k, arr), col in zip(stability['rate_history'].items(), palette):
        ax.plot(n_values, arr, lw=1.2, color=col, label=k)
        c_at = stability['converged_at'].get(k)
        if c_at is not None:
            ax.axvline(c_at, color=col, lw=0.8, ls='--', alpha=0.6)
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.set_ylabel('%/yr')
    ax.set_xlabel('Window length (samples)')
    ax.set_title(
        f"Rate history  (dashed = converged within "
        f"±{stability['convergence_tol']:.2f} %/yr of final value)",
        fontsize=9,
    )
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)

    # Row 4: absolute rate delta
    ax = axes[3]
    for (k, arr), col in zip(stability['rate_delta'].items(), palette):
        ax.plot(mid_n, arr, lw=1.0, color=col, label=k, alpha=0.8)
    ax.set_ylabel('|Δ rate|  (%/yr)')
    ax.set_xlabel('Window length (samples)')
    ax.set_title('Absolute rate change between successive windows', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    return fig


def plot_trend(sd_trend_results, energy_normalized, figsize=(8, 5)):
    """
    Plot normalised daily energy with the fitted trend overlaid.

    Parameters
    ----------
    sd_trend_results : dict
        Output of :func:`degradation` (the third return value).
    energy_normalized : pandas.Series
        Normalised daily energy with a datetime index — typically
        ``TrendAnalysis.sensor_aggregated_performance`` or
        ``TrendAnalysis.clearsky_aggregated_performance``.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt.style.use('tableau-colorblind10')
    x2 = sd_trend_results['components']['x2']

    fig, ax = plt.subplots(figsize=figsize)
    energy_normalized.plot(ax=ax, lw=0.8,
                           label='normalized daily energy', marker='.',
                           ls='none', ms=1)
    ax.plot(energy_normalized.index, x2, lw=1.5, label='trend')
    ax.legend(fontsize=9, framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


def plot_decomposition(sd_trend_results, figsize=(12, 10)):
    """
    Seasonal-trend decomposition plot with an optional soiling panel.

    - Row 1: Measured signal and fit (``x1 + x2``).
    - Row 2: Seasonal component ``x1``.
    - Row 3: Trend component ``x2``.
    - Optional row 4: Soiling ratio.
    - Final row: Residual ``x3``.

    Parameters
    ----------
    sd_trend_results : dict
        Output of :func:`degradation` (the third return value). Must contain
        ``'y'``, ``'components'`` keys.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt.style.use('tableau-colorblind10')
    y = sd_trend_results['y']
    components = sd_trend_results['components']

    x1 = components['x1']
    x2 = components['x2']
    x3 = components['x3']
    fit = components['fit']
    t = np.arange(len(y))

    # Auto-detect linear vs. log-transform space for residual reference
    x3_ref = 1.0 if (np.nanmedian(x3) > 0.5) else 0.0

    has_soiling = 'soiling' in components
    n_rows = 5 if has_soiling else 4
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    ax = axes[0]
    ax.plot(t, y, lw=0.8, label='Measured y', zorder=1)
    fit_label = 'Fit (seasonal × trend × soiling)' if has_soiling else 'Fit (x1+x2)'
    ax.plot(t, fit, lw=1.5, label=fit_label, zorder=2)
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
    ax.set_title('Decomposition', fontsize=11, fontweight='bold')

    ax = axes[1]
    ax.plot(t, x1, lw=1.2)
    ax.axhline(x3_ref, color='black', lw=0.5, ls='--')
    ax.set_ylabel('x1  (seasonal)')

    ax = axes[2]
    ax.plot(t, x2, lw=1.5)
    ax.set_ylabel('x2  (trend)')

    residual_axis = 3
    if has_soiling:
        ax = axes[3]
        ax.plot(t, components['soiling'], lw=1.2)
        ax.axhline(1.0, color='black', lw=0.5, ls='--')
        ax.set_ylabel('soiling ratio')
        residual_axis = 4

    ax = axes[residual_axis]
    ax.fill_between(t, x3, x3_ref,
                    where=(x3 >= x3_ref), alpha=0.5, lw=0)
    ax.fill_between(t, x3, x3_ref,
                    where=(x3 < x3_ref), alpha=0.5, lw=0)
    ax.axhline(x3_ref, color='black', lw=0.7)
    ax.set_ylabel('x3  (residual)')
    ax.set_xlabel('Sample index')

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    fig.align_ylabels(axes)
    plt.tight_layout()
    return fig


def animate_degradation(
    y_full,
    make_problem_kwargs,
    output_path='degradation_animation.mp4',
    fps=8,
    step=10,
    T=365.2425,
    figsize=(12, 10),
    dpi=150,
):
    """
    Save an animation of the decomposition fit as the data record grows.

    Each frame solves :func:`make_problem` on ``y_full[:n]`` for *n* in
    ``range(int(T), len(y_full)+1, step)``, then renders a 4-panel
    decomposition plot. The full-length x-axis is fixed so the fit visually
    grows from left to right.

    Progress is printed to stdout every 20 frames.

    Output format is determined by the extension of *output_path*:

    - ``.mp4`` — requires system ``ffmpeg``
    - ``.gif`` — uses :class:`matplotlib.animation.PillowWriter` (Pillow dep)

    Parameters
    ----------
    y_full : array-like
        Full time series.
    make_problem_kwargs : dict
        Keyword arguments passed to :func:`make_problem` on every frame
        (do not include ``'y'``).
    output_path : str
        Destination file path; extension sets the format.
    fps : int
        Frames per second.
    step : int
        Samples to advance per frame; smaller = smoother but slower.
    T : float
        Samples per year; sets the minimum crop length.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Resolution of each rendered frame.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    y_full = np.asarray(y_full, dtype=float)
    make_problem_kwargs = {k: v for k, v in make_problem_kwargs.items() if k != 'y'}
    N = len(y_full)
    t_full = np.arange(N)

    min_n = int(T)
    frame_lengths = list(range(min_n, N + 1, step))
    if frame_lengths[-1] != N:
        frame_lengths.append(N)

    print(f"Solving {len(frame_lengths)} frames ...")
    solutions = []
    for i, n in enumerate(frame_lengths):
        if i % 20 == 0:
            print(f"  frame {i + 1}/{len(frame_lengths)}  (n={n})")
        build = make_problem(y_full[:n], **make_problem_kwargs)
        prob, variables = build['problem'], build['variables']
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status not in ('optimal', 'optimal_inaccurate'):
            solutions.append(None)
        else:
            solutions.append({
                'n':  n,
                'x1': variables['x1'].value,
                'x2': variables['x2'].value,
                'x3': variables['x3'].value,
            })
    print("Done solving. Rendering animation ...")

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    trend_type = make_problem_kwargs.get('trend_type', 'linear')
    loss = make_problem_kwargs.get('loss', 'l2')
    fig.suptitle(
        f"Degradation decomposition — trend: {trend_type}, loss: {loss}",
        fontsize=11, fontweight='bold', y=1.01,
    )

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0, N - 1)

    axes[0].set_ylabel('y')
    axes[1].set_ylabel('x1  (seasonal)')
    axes[2].set_ylabel('x2  (trend)')
    axes[3].set_ylabel('x3  (residual)')
    axes[3].set_xlabel('Sample index')

    axes[0].plot(t_full, y_full, color='#dddddd', lw=0.6, zorder=1)
    axes[0].axhline(0, color='black', lw=0.4, ls='--')
    axes[1].axhline(0, color='black', lw=0.5, ls='--')
    axes[3].axhline(0, color='black', lw=0.7)

    vlines = [ax.axvline(min_n, color='#888888', lw=0.8, ls=':') for ax in axes]

    line_fit, = axes[0].plot([], [], color='#e05c2a', lw=1.5,
                             label='Fit (x1+x2)', zorder=2)
    line_x1, = axes[1].plot([], [], color='#2a7de0', lw=1.2)
    line_x2, = axes[2].plot([], [], color='#27a865', lw=1.5)

    fill_state = {'pos': None, 'neg': None}

    axes[0].legend(
        handles=[
            plt.Line2D([0], [0], color='#dddddd', lw=1.5, label='Measured y'),
            plt.Line2D([0], [0], color='#e05c2a', lw=1.5, label='Fit (x1+x2)'),
        ],
        loc='upper right', fontsize=7, framealpha=0.7,
    )
    fig.align_ylabels(axes)

    def update(frame_idx):
        sol = solutions[frame_idx]
        if sol is None:
            return

        n = sol['n']
        t = np.arange(n)
        x1 = sol['x1']
        x2 = sol['x2']
        x3 = sol['x3']
        fit = x1 + x2

        line_fit.set_data(t, fit)
        line_x1.set_data(t, x1)
        line_x2.set_data(t, x2)

        for key in ('pos', 'neg'):
            if fill_state[key] is not None:
                fill_state[key].remove()

        fill_state['pos'] = axes[3].fill_between(
            t, x3, 0, where=(x3 >= 0), color='#e05c2a', alpha=0.5, lw=0,
        )
        fill_state['neg'] = axes[3].fill_between(
            t, x3, 0, where=(x3 < 0), color='#2a7de0', alpha=0.5, lw=0,
        )

        for vl in vlines:
            vl.set_xdata([n, n])

        for ax, arr in zip(axes[1:], [x1, x2, x3]):
            lo, hi = np.nanmin(arr), np.nanmax(arr)
            pad = max((hi - lo) * 0.1, 1e-6)
            ax.set_ylim(lo - pad, hi + pad)

        fit_lo = np.nanmin([np.nanmin(y_full[:n]), np.nanmin(fit)])
        fit_hi = np.nanmax([np.nanmax(y_full[:n]), np.nanmax(fit)])
        pad0 = max((fit_hi - fit_lo) * 0.1, 1e-6)
        axes[0].set_ylim(fit_lo - pad0, fit_hi + pad0)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_lengths),
        interval=1000 // fps,
        blit=False,
    )

    suffix = output_path.rsplit('.', 1)[-1].lower()
    if suffix == 'gif':
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)

    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved → {output_path}")
    return ani


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_degradation_report(sd_trend_results):
    """
    Format a degradation report as a Markdown string.

    Reads rate keys and analysis arguments directly from *sd_trend_results*.

    Parameters
    ----------
    sd_trend_results : dict
        Output of :func:`degradation` (the third return value).

    Returns
    -------
    str
        Markdown-formatted degradation report.
    """
    args = sd_trend_results['args']
    trend_type = args['trend_type']

    if trend_type == 'linear':
        body = f"| Overall | {sd_trend_results['rate_pct_yr']:+.3f} %/yr |"

    elif trend_type == 'pwl':
        body = (
            f"| Pre-breakpoint  | {sd_trend_results['rate_pre_pct_yr']:+.3f} %/yr |\n"
            f"| Post-breakpoint | {sd_trend_results['rate_post_pct_yr']:+.3f} %/yr |\n"
            f"| Overall         | {sd_trend_results['rate_overall_pct_yr']:+.3f} %/yr |"
        )

    elif trend_type == 'monotone':
        yearly_rows = "\n".join(
            f"| Year {i + 1:>2d}  | {r:+.3f} %/yr |"
            for i, r in enumerate(sd_trend_results['rate_yearly_pct_yr'])
        )
        body = (
            f"| Overall | {sd_trend_results['rate_overall_pct_yr']:+.3f} %/yr |\n"
            f"{yearly_rows}"
        )

    else:
        raise ValueError(
            f"trend_type must be 'linear', 'pwl', or 'monotone'; got '{trend_type}'"
        )

    end_penalty_str = (
        f" | **λ end:** {args['lam_end']:.2e}"
        if trend_type == 'monotone' and args.get('lam_end', 0.0) > 0
        else ""
    )

    soiling_section = ""
    if 'soiling' in sd_trend_results:
        soiling = sd_trend_results['soiling']
        selector = soiling['selector']
        loss = soiling['loss']
        state = 'yes' if selector['detected'] else 'no (neutral model)'
        weight = selector['selected_lam_soiling_down']
        weight_text = f"{weight:.3e}" if weight is not None else 'n/a'
        rate = soiling['rate_summary'].set_index('period').loc['overall']
        quarterly_loss = loss['quarterly'].set_index('quarter')
        quarterly_rate = soiling['rate_summary'].set_index('period')
        quarterly_rows = []
        for quarter in range(1, 5):
            loss_row = quarterly_loss.loc[quarter]
            rate_row = quarterly_rate.loc[f'Q{quarter}']
            weighted = (
                f" | {loss_row['insolation_weighted_loss_pct']:.3f} %"
                if 'insolation_weighted_loss_pct' in quarterly_loss else ''
            )
            quarterly_rows.append(
                f"| Q{quarter} | {loss_row['time_averaged_loss_pct']:.3f} %"
                f"{weighted} | {rate_row['median_rate_pct_day']:.4f} %/day |"
            )
        has_weighted_loss = 'insolation_weighted_loss_pct' in loss
        weighted_header = ' | Insolation-weighted loss' if has_weighted_loss else ''
        weighted_separator = '|---:' if has_weighted_loss else ''
        soiling_section = (
            f"\n## Soiling report\n\n"
            f"**Soiling detected:** {state} | **selected λ down:** {weight_text}\n\n"
            f"| Metric | Estimate |\n|---|---:|\n"
            f"| Time-averaged loss | {loss['time_averaged_loss_pct']:.3f} % |\n"
            f"| Median local rate | {rate['median_rate_pct_day']:.4f} %/day |\n"
        )
        if 'insolation_weighted_loss_pct' in loss:
            soiling_section += (
                f"| Insolation-weighted loss | "
                f"{loss['insolation_weighted_loss_pct']:.3f} % |\n"
            )
        soiling_section += (
            f"\n| Quarter | Time-averaged loss{weighted_header} | Median local rate |\n"
            f"|---|---:{weighted_separator}|---:|\n"
            + "\n".join(quarterly_rows)
            + "\n"
        )

    return (
        f"## Degradation report\n\n"
        f"**Trend type:** `{trend_type}` | **Loss:** `{args['loss']}` | "
        f"**Harmonics:** {args['numharmonics']} | "
        f"**λ seasonal:** {args['lam_seasonal']:.2e} | "
        f"**λ trend:** {args['lam_trend']:.2e}"
        f"{end_penalty_str}\n\n"
        f"| Segment | Rate |\n"
        f"|---|---|\n"
        f"{body}\n\n"
        f"> Rates expressed as %/yr relative to the trend value at the first sample.\n"
        f"> Negative values indicate degradation.\n"
        f"{soiling_section}"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def degradation(
    energy_normalized,
    trend_type='linear',
    loss=None,
    numharmonics=None,
    lam_seasonal=1e-1,
    lam_trend=1e0,
    lam_end=0.0,
    end_frac=(0.90, 0.95),
    q=0.75,
    huber_M=None,
    include_soiling=False,
    insolation_daily=None,
    log_transform=None,
    confidence_level=68.2,
    n_bootstrap=500,
    block_size=None,
    random_state=None,
):
    """
    Signal-decomposition degradation analysis.

    Decomposes *energy_normalized* into seasonal, trend, and residual
    components via convex optimisation and returns the overall degradation
    rate of the trend component. By default this performs one decomposition
    without soiling. With ``include_soiling=True``, the final algorithm is:

    1. solve the validated nine-weight uniform-L1 soiling path;
    2. apply the frozen coherence/materiality selector;
    3. if no candidate qualifies, solve and return the structural no-soiling
       model;
    4. otherwise, refit the selected model twice using cleaning-interval IRL1
       value weights with epsilon 0.01, leaving every other component,
       constraint, and selected downward weight unchanged.

    IRL1 reduces amplitude shrinkage after detection; it never participates in
    detection. Bootstrap replicates repeat the same final two-step refinement.

    The soiling component represents conventional dry soil or sand
    accumulation interrupted by discrete cleaning/recovery events. It is not a
    biological-fouling model. Persistent partial outages can produce similar
    downward level shifts and must be detected or corrected upstream before
    using this option.

    Assumes daily aggregation (``T = 365.2425`` samples/year). Non-daily
    ``aggregation_freq`` support is future work.

    The confidence interval is computed via a moving-block bootstrap over the
    fitted residuals, respecting residual autocorrelation. See ``n_bootstrap``,
    ``block_size``, and ``confidence_level`` for controls.

    Parameters
    ----------
    energy_normalized : pandas.Series
        Time series of insolation-weighted aggregated normalized PV energy,
        indexed by date. NaN entries are treated as missing data.
    trend_type : str
        Trend model: ``'linear'``, ``'pwl'`` (piecewise-linear with one
        breakpoint after the first year), or ``'monotone'`` (non-increasing).
    loss : str or None
        Residual loss. ``None`` selects ``'huber'`` with ``huber_M=0.05``.
    numharmonics : int or None
        Number of Fourier harmonic pairs. ``None`` selects 6 normally and 3
        when ``include_soiling=True``.
    lam_seasonal : float
        Regularisation weight on Fourier coefficients.
    lam_trend : float
        Regularisation weight on trend smoothness.
    lam_end : float
        End-drop penalty weight; only used when ``trend_type='monotone'``.
        See :func:`make_problem` for details. ``0.0`` disables it (default).
    end_frac : tuple of float
        ``(frac_start, frac_end)`` ramp positions for the end-drop penalty.
        Only used when ``trend_type='monotone'`` and ``lam_end > 0``.
        Default ``(0.90, 0.95)``.
    q : float
        Quantile level in (0, 1); only used when ``loss='quantile'``.
    huber_M : float or None
        Huber threshold. ``None`` selects 0.05. This default is scaled for
        normalized PV performance near unity.
    include_soiling : bool
        Run the validated soiling path and selector. This option requires the
        validated linear/Huber/log configuration; conflicting model options
        raise ``ValueError``. The model assumes conventional dry accumulation
        with discrete recoveries and upstream handling of partial outages; it
        is not intended for biological fouling. Default False.
    insolation_daily : pandas.Series or None
        Daily insolation aligned by date. When provided, insolation-weighted
        soiling losses are returned in addition to time-averaged losses.
    log_transform : bool or None
        If True, apply a natural-log transform before decomposing. The
        returned components are back-transformed to the original domain.
        Rates are computed as compound annual rates via
        :func:`extract_degradation_rate_log`. ``None`` selects True. Set False
        explicitly to fit an additive decomposition in ratio space.
    confidence_level : float
        Confidence level for ``Rd_CI`` in percent (e.g. ``68.2`` for ≈1σ,
        ``95`` for 95%). The interval is the empirical
        ``[(100 - confidence_level)/2, 100 - (100 - confidence_level)/2]``
        percentiles of the bootstrap distribution. Default ``68.2``.
    n_bootstrap : int
        Number of bootstrap replicates. Larger values give a more stable CI
        estimate at proportionally higher cost. Default ``500``. Set to ``0``
        to skip CI estimation entirely — useful for fast interactive exploration.
    block_size : int or None
        Length (in samples) of each contiguous block used in the moving-block
        bootstrap. ``None`` (default) uses ``int(T)`` ≈ one year, which
        preserves low-frequency residual dependence relevant for trend-slope
        uncertainty.
    random_state : int or None
        Seed for the random number generator. ``None`` (default) gives a
        different result each call; an integer makes the CI reproducible.

    Returns
    -------
    Rd_pct : float
        Overall degradation rate in %/year. For ``trend_type='linear'`` this
        is ``rate_pct_yr``; for ``'pwl'`` and ``'monotone'`` it is
        ``rate_overall_pct_yr``.
    Rd_CI : numpy.ndarray, shape (2,)
        ``[lower, upper]`` confidence interval on the overall degradation rate
        (%/year) at ``confidence_level``, computed via moving-block bootstrap.
        Returns ``[nan, nan]`` if ``n_bootstrap=0`` or fewer than half of
        bootstrap solves succeed.
    sd_trend_results : dict
        Full results dict. Keys:

        - Rate keys (flat-merged from :func:`extract_degradation_rate` or
          :func:`extract_degradation_rate_log`; vary by *trend_type*)
        - ``'ci_<rate_key>'``: bootstrap CI arrays for each bootstrapped rate,
          shape ``(2,)`` for scalar rates or ``(2, n)`` for array rates.
          Present only when ``n_bootstrap > 0`` and enough solves succeed.
          Examples: ``'ci_rate_pct_yr'`` (linear),
          ``'ci_rate_pre_pct_yr'`` / ``'ci_rate_post_pct_yr'`` /
          ``'ci_rate_overall_pct_yr'`` (pwl),
          ``'ci_rate_overall_pct_yr'`` / ``'ci_rate_yearly_pct_yr'`` (monotone).
        - ``'components'``: dict with ``'x1'``, ``'x2'``, ``'x3'``,
          ``'fit'`` arrays in the original (non-log) domain
        - ``'y'``: original input values (pre-log-transform) as ndarray
        - ``'args'``: dict of kwargs passed to :func:`make_problem`
        - ``'problem_status'``: solver status string
        - ``'soiling'`` when requested: selector and IRL1-refinement
          diagnostics, daily soiling ratio/rate estimates,
          cleaning-to-cleaning intervals, overall and quarterly rate
          summaries, and time-averaged loss metrics. Optional
          insolation-weighted losses are included when *insolation_daily* is
          supplied. Negative rates denote soiling accumulation.

    Raises
    ------
    ValueError
        If the solver does not return ``'optimal'`` or
        ``'optimal_inaccurate'`` status.
    """
    if include_soiling:
        resolved = {
            'loss': 'huber' if loss is None else loss,
            'numharmonics': 3 if numharmonics is None else numharmonics,
            'huber_M': 0.05 if huber_M is None else huber_M,
            'log_transform': True if log_transform is None else log_transform,
        }
        incompatible = []
        if trend_type != 'linear':
            incompatible.append("trend_type='linear'")
        if resolved['loss'] != 'huber':
            incompatible.append("loss='huber'")
        if resolved['numharmonics'] != 3:
            incompatible.append('numharmonics=3')
        if not np.isclose(resolved['huber_M'], 0.05):
            incompatible.append('huber_M=0.05')
        if resolved['log_transform'] is not True:
            incompatible.append('log_transform=True')
        if not np.isclose(lam_seasonal, 0.1):
            incompatible.append('lam_seasonal=0.1')
        if not np.isclose(lam_trend, 1.0):
            incompatible.append('lam_trend=1.0')
        if incompatible:
            raise ValueError(
                'include_soiling=True requires the validated configuration: '
                + ', '.join(incompatible)
            )
        if len(energy_normalized) < 2 * 365.2425:
            warnings.warn(
                'Soiling selection on records shorter than two years is not '
                'validated; results may be less stable.',
                UserWarning,
                stacklevel=2,
            )
        if not isinstance(energy_normalized.index, pd.DatetimeIndex):
            raise ValueError('include_soiling=True requires a DatetimeIndex')
    else:
        resolved = {
            'loss': 'huber' if loss is None else loss,
            'numharmonics': 6 if numharmonics is None else numharmonics,
            'huber_M': 0.05 if huber_M is None else huber_M,
            'log_transform': True if log_transform is None else log_transform,
        }
    loss = resolved['loss']
    numharmonics = resolved['numharmonics']
    huber_M = resolved['huber_M']
    log_transform = resolved['log_transform']

    energy_normalized = energy_normalized.sort_index()
    y = energy_normalized.values

    T = 365.2425
    y_input = prepare_input(y, log_transform=log_transform)

    common_build_args = dict(
        numharmonics=numharmonics, trend_type=trend_type, loss=loss,
        lam_seasonal=lam_seasonal, lam_trend=lam_trend, lam_end=lam_end,
        end_frac=end_frac, q=q, huber_M=huber_M, T=T,
    )
    selector = None
    if include_soiling:
        path_build = make_problem(
            y_input, include_soiling=True,
            lam_soiling_down=_SOILING_LAM_DOWN_GRID[0],
            lam_soiling_value=_SOILING_LAM_VALUE,
            **common_build_args,
        )
        candidate_metrics, selected_index, fallback_component = _solve_soiling_path(
            path_build, y_input, T
        )
        detected = selected_index is not None
        refinement = None
        if detected:
            selected_weight = float(_SOILING_LAM_DOWN_GRID[selected_index])
            path_build['parameters']['lam_soiling_down'].value = selected_weight
            path_build['problem'].solve(solver=cp.CLARABEL, warm_start=True)
            path_build['args']['lam_soiling_down'] = selected_weight
            # Detection is now frozen. Debias only the selected positive fit;
            # null records never enter this IRL1 sequence.
            refinement = _refine_soiling_irl1(path_build, y_input)
            build = path_build
        else:
            selected_weight = None
            build = make_problem(y_input, include_soiling=False, **common_build_args)
            build['problem'].solve(solver=cp.CLARABEL)
        selector = {
            'detected': detected,
            'final_model': 'soiling' if detected else 'no_soiling',
            'selected_candidate_index': selected_index,
            'selected_lam_soiling_down': selected_weight,
            'null_reason': None if detected else 'no_coherent_soiling',
            'fallback_candidate_index': len(_SOILING_LAM_DOWN_GRID) - 1,
            'fallback_lam_soiling_down': float(_SOILING_LAM_DOWN_GRID[-1]),
            'fallback_soiling_component_log': fallback_component,
            'lam_soiling_value': _SOILING_LAM_VALUE,
            'refinement': refinement,
            'candidate_metrics': candidate_metrics,
            'thresholds': {
                'q75_loss_min': _SOILING_Q75_MIN,
                'max_recoveries_per_year': _SOILING_MAX_RECOVERIES_PER_YEAR,
                'max_neighbor_nrmse': _SOILING_MAX_NEIGHBOR_NRMSE,
                'min_neighbor_correlation': _SOILING_MIN_NEIGHBOR_CORRELATION,
            },
        }
    else:
        build = make_problem(y_input, include_soiling=False, **common_build_args)
        build['problem'].solve(solver=cp.CLARABEL)

    prob = build['problem']
    variables = build['variables']

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise ValueError(
            f"Solver did not converge: status='{prob.status}'. "
            "Check that the input series has sufficient valid data."
        )

    extractor = extract_degradation_rate_log if log_transform else extract_degradation_rate
    rates = extractor(variables, trend_type, T=T)

    if trend_type == 'linear':
        Rd_pct = rates['rate_pct_yr']
    else:
        Rd_pct = rates['rate_overall_pct_yr']

    fit_work = variables['x1'].value + variables['x2'].value
    if include_soiling and selector['detected']:
        fit_work = fit_work + variables['soiling'].value
    residuals_work = variables['x3'].value
    nan_mask = np.isnan(y_input)

    ci_dict, soiling_ci = _bootstrap_ci(
        fit=fit_work,
        residuals=residuals_work,
        nan_mask=nan_mask,
        make_problem_args={k: v for k, v in build['args'].items() if k != 'y'},
        trend_type=trend_type,
        log_transform=log_transform,
        T=T,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        confidence_level=confidence_level,
        random_state=random_state,
        soiling_context=(
            {
                'index': energy_normalized.index,
                'insolation_daily': insolation_daily,
                'null_model': not selector['detected'],
            }
            if include_soiling else None
        ),
    )

    rate_ci_key = 'rate_pct_yr' if trend_type == 'linear' else 'rate_overall_pct_yr'
    Rd_CI = ci_dict.get(rate_ci_key, np.array([np.nan, np.nan]))

    components = recover_components(variables, log_transform=log_transform)
    soiling_results = None
    if include_soiling:
        if selector['detected']:
            ratio = components['soiling']
        else:
            ratio = np.ones(len(y), dtype=float)
            components['soiling'] = ratio
            components['fit'] = components['x1'] * components['x2']
        index = energy_normalized.index
        log_ratio = np.log(np.clip(ratio, 1e-12, None))
        changes = np.diff(log_ratio)
        daily_rate = np.full(len(ratio), np.nan)
        accumulating = changes <= 0
        daily_rate[np.flatnonzero(accumulating) + 1] = (
            np.exp(changes[accumulating]) - 1
        ) * 100
        cleaning = np.r_[False, changes >= _SOILING_RECOVERY_THRESHOLD]
        daily = pd.DataFrame({
            'soiling_ratio': ratio,
            'soiling_rate_pct_day': daily_rate,
            'cleaning_event': cleaning,
        }, index=index)
        intervals = _soiling_intervals(ratio, index)
        rate_summary = _soiling_rate_summary(intervals, index)
        loss_metrics = _soiling_loss_metrics(ratio, index, insolation_daily)
        loss_metrics['time_averaged_loss_ci'] = soiling_ci.get(
            'time_loss_overall', np.array([np.nan, np.nan])
        )
        for row_index, quarter in enumerate(range(1, 5)):
            time_ci = soiling_ci.get(
                f'time_loss_Q{quarter}', np.array([np.nan, np.nan])
            )
            loss_metrics['quarterly'].loc[
                row_index,
                ['time_averaged_loss_ci_low', 'time_averaged_loss_ci_high'],
            ] = time_ci
            if 'insolation_weighted_loss_pct' in loss_metrics:
                insolation_ci = soiling_ci.get(
                    f'insolation_loss_Q{quarter}', np.array([np.nan, np.nan])
                )
                loss_metrics['quarterly'].loc[row_index, [
                    'insolation_weighted_loss_ci_low',
                    'insolation_weighted_loss_ci_high',
                ]] = insolation_ci
        if 'insolation_weighted_loss_pct' in loss_metrics:
            loss_metrics['insolation_weighted_loss_ci'] = soiling_ci.get(
                'insolation_loss_overall', np.array([np.nan, np.nan])
            )
        for row_index, period in enumerate(rate_summary['period']):
            rate_ci = soiling_ci.get(
                f'rate_{period}', np.array([np.nan, np.nan])
            )
            rate_summary.loc[row_index, ['rate_ci_low', 'rate_ci_high']] = rate_ci
        soiling_results = {
            'selector': selector,
            'daily': daily,
            'loss': loss_metrics,
            'intervals': intervals,
            'rate_summary': rate_summary,
        }

    sd_trend_results = {
        **rates,
        'components':     components,
        'y':              y,
        'args':           {**build['args'], 'log_transform': log_transform},
        'problem_status': prob.status,
        **{f'ci_{k}': v for k, v in ci_dict.items()},
    }
    if include_soiling:
        sd_trend_results['soiling'] = soiling_results

    return Rd_pct, Rd_CI, sd_trend_results
