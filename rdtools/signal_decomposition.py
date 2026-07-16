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

import cvxpy as cp
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
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
    T=365.2425,
):
    """
    Build the convex seasonal-trend decomposition problem.

    Solves ``y = x1 + x2 + x3`` where:

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
    T : float
        Period length in samples (default 365.2425 for daily data).

    Returns
    -------
    dict with keys:

    - ``'problem'``   : :class:`cvxpy.Problem` — call ``.solve()`` before
      inspecting variables.
    - ``'variables'`` : dict of CVXPY expressions (``'theta'``, ``'x1'``,
      ``'x2'``, ``'x3'``, plus trend-specific scalars).
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
    # 3.  Residual  x3 = y - x1 - x2
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
    objective = cp.Minimize(data_fidelity + seasonal_reg + trend_reg)
    constraints = trend_constraints
    constraints.append(y[good_data] == (x1 + x2 + x3)[good_data])
    problem = cp.Problem(objective, constraints)

    if trend_type == 'linear':
        variables = {'theta': theta, 'trend_coeffs': trend_coeffs,
                     'x1': x1, 'x2': x2, 'x3': x3}
    elif trend_type == 'pwl':
        variables = {'theta': theta, 'a0': a0, 'b0': b0, 'b1': b1,
                     'x1': x1, 'x2': x2, 'x3': x3}
    elif trend_type == 'monotone':
        variables = {'theta': theta, 'x1': x1, 'x2': x2, 'x3': x3}

    return {
        'problem': problem,
        'variables': variables,
        'args': _get_kwargs(make_problem, locals()),
    }


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
    dict with keys ``'x1'``, ``'x2'``, ``'x3'``, ``'fit'``.
    """
    x1 = variables['x1'].value
    x2 = variables['x2'].value
    x3 = variables['x3'].value

    if log_transform:
        x1 = np.exp(x1)
        x2 = np.exp(x2)
        x3 = np.exp(x3)
        fit = x1 * x2
    else:
        fit = x1 + x2

    return {'x1': x1, 'x2': x2, 'x3': x3, 'fit': fit}


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
):
    if n_bootstrap == 0:
        return np.array([np.nan, np.nan])

    rng = np.random.default_rng(random_state)
    if block_size is None:
        block_size = int(T)

    valid_idx = np.where(~nan_mask)[0]
    res_valid = residuals[valid_idx]
    M = res_valid.size
    L = min(block_size, M)
    n_blocks = int(np.ceil(M / L))

    extractor = extract_degradation_rate_log if log_transform else extract_degradation_rate
    rate_key = 'rate_pct_yr' if trend_type == 'linear' else 'rate_overall_pct_yr'

    collected = []
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
            rates = extractor(b['variables'], trend_type, T=T)
            collected.append(rates[rate_key])
        except Exception:
            continue

    if len(collected) < _MIN_SUCCESS_FRAC * n_bootstrap:
        return np.array([np.nan, np.nan])

    lower_pct = (100 - confidence_level) / 2
    upper_pct = 100 - lower_pct
    return np.array(np.percentile(collected, [lower_pct, upper_pct]))


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
    Four-panel seasonal-trend decomposition plot.

    - Row 1: Measured signal and fit (``x1 + x2``).
    - Row 2: Seasonal component ``x1``.
    - Row 3: Trend component ``x2``.
    - Row 4: Residual ``x3``.

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

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    ax = axes[0]
    ax.plot(t, y, lw=0.8, label='Measured y', zorder=1)
    ax.plot(t, fit, lw=1.5, label='Fit (x1+x2)', zorder=2)
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

    ax = axes[3]
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
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def degradation(
    energy_normalized,
    trend_type='linear',
    loss='l2',
    numharmonics=6,
    lam_seasonal=1e-1,
    lam_trend=1e0,
    lam_end=0.0,
    end_frac=(0.90, 0.95),
    q=0.75,
    huber_M=1.0,
    log_transform=False,
    confidence_level=68.2,
    n_bootstrap=500,
    block_size=None,
    random_state=None,
):
    """
    Signal-decomposition degradation analysis.

    Decomposes *energy_normalized* into seasonal, trend, and residual
    components via convex optimisation and returns the overall degradation
    rate of the trend component.

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
    loss : str
        Residual loss: ``'l2'``, ``'l1'``, ``'huber'``, or ``'quantile'``.
    numharmonics : int
        Number of Fourier harmonic pairs for the seasonal component.
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
    huber_M : float
        Huber threshold; only used when ``loss='huber'``.
    log_transform : bool
        If True, apply a natural-log transform before decomposing. The
        returned components are back-transformed to the original domain.
        Rates are computed as compound annual rates via
        :func:`extract_degradation_rate_log`.
    confidence_level : float
        Confidence level for ``Rd_CI`` in percent (e.g. ``68.2`` for ≈1σ,
        ``95`` for 95%). The interval is the empirical
        ``[(100 - confidence_level)/2, 100 - (100 - confidence_level)/2]``
        percentiles of the bootstrap distribution. Default ``68.2``.
    n_bootstrap : int
        Number of bootstrap replicates. Larger values give a more stable CI
        estimate at proportionally higher cost. Default ``500``. Set to ``0``
        to skip the CI entirely and return ``[nan, nan]`` immediately — useful
        for fast interactive exploration.
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
        ``[lower, upper]`` confidence interval on the degradation rate
        (%/year) at ``confidence_level``, computed via moving-block bootstrap.
        Returns ``[nan, nan]`` if fewer than half of bootstrap solves succeed.
    sd_trend_results : dict
        Full results dict. Keys:

        - Rate keys (flat-merged from :func:`extract_degradation_rate` or
          :func:`extract_degradation_rate_log`; vary by *trend_type*)
        - ``'components'``: dict with ``'x1'``, ``'x2'``, ``'x3'``,
          ``'fit'`` arrays in the original (non-log) domain
        - ``'y'``: original input values (pre-log-transform) as ndarray
        - ``'args'``: dict of kwargs passed to :func:`make_problem`
        - ``'problem_status'``: solver status string

    Raises
    ------
    ValueError
        If the solver does not return ``'optimal'`` or
        ``'optimal_inaccurate'`` status.
    """
    energy_normalized = energy_normalized.sort_index()
    y = energy_normalized.values

    T = 365.2425
    y_input = prepare_input(y, log_transform=log_transform)

    build = make_problem(
        y_input,
        numharmonics=numharmonics,
        trend_type=trend_type,
        loss=loss,
        lam_seasonal=lam_seasonal,
        lam_trend=lam_trend,
        lam_end=lam_end,
        end_frac=end_frac,
        q=q,
        huber_M=huber_M,
        T=T,
    )

    prob = build['problem']
    variables = build['variables']
    prob.solve(solver=cp.CLARABEL)

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
    residuals_work = variables['x3'].value
    nan_mask = np.isnan(y_input)

    Rd_CI = _bootstrap_ci(
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
    )

    components = recover_components(variables, log_transform=log_transform)

    sd_trend_results = {
        **rates,
        'components':     components,
        'y':              y,
        'args':           build['args'],
        'problem_status': prob.status,
    }

    return Rd_pct, Rd_CI, sd_trend_results
