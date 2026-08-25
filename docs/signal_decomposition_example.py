import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import rdtools
    from rdtools import signal_decomposition as sd

    return mo, pd, plt, rdtools, sd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Signal decomposition degradation analysis (PVDAQ system 4)

    This notebook applies the signal-decomposition degradation estimator
    (`rdtools.signal_decomposition.degradation`) to PVDAQ system 4 — the
    same dataset used in
    [`TrendAnalysis_example.ipynb`](TrendAnalysis_example.ipynb) and
    [`TrendAnalysis_hybrid_example.ipynb`](TrendAnalysis_hybrid_example.ipynb).

    The estimator decomposes the daily normalised-energy series into three
    components solved simultaneously via convex optimisation:

    $$y = x_1 + x_2 + x_3$$

    - **$x_1$** — seasonal component, expressed as a truncated Fourier basis
    - **$x_2$** — trend component (linear, piecewise-linear, or monotone)
    - **$x_3$** — residual noise, penalised by a user-selected loss function

    The widget panel below lets you explore how each modelling choice
    (trend type, loss, regularisation) affects the fitted decomposition.
    All parameter changes trigger an automatic re-solve; typical wall time
    per solve is well under a second.

    The standard `TrendAnalysis` workflow is used throughout: data
    normalisation and filtering happen once inside `sensor_analysis`; only
    the CVXPY solve is reactive.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Import and prepare data

    Loads PVDAQ system 4 from the local pickle cache (ignored by git) or
    falls back to the released CSV. Column names and site metadata follow
    the convention in `TrendAnalysis_example.ipynb`.
    """)
    return


@app.cell
def _(pd):
    file_url = (
        'https://github.com/NatLabRockies/rdtools/releases/download/3.0.0/'
        'pvdaq_system_4_2010-2016_subset_soil_signal.csv'
    )
    cache_file = 'PVDAQ_system_4_2010-2016_subset_soilsignal.pickle'

    try:
        df = pd.read_pickle(cache_file)
    except FileNotFoundError:
        df = pd.read_csv(file_url, index_col=0, parse_dates=True)
        df.to_pickle(cache_file)

    df = df.rename(columns={
        'ac_power':      'power_ac',
        'ambient_temp':  'Tamb',
        'poa_irradiance': 'poa',
    })

    meta = {
        'latitude':          39.7406,
        'longitude':        -105.1774,
        'timezone':          'Etc/GMT+7',
        'gamma_pdc':         -0.0034,
        'azimuth':           180,
        'tilt':              40,
        'power_dc_rated':    1000.0,
        'temp_model_params': 'open_rack_glass_polymer',
    }

    df.index = df.index.tz_localize(meta['timezone'])
    freq = pd.infer_freq(df.index[:10])

    print(f"loaded {len(df):,} rows, {df.index.min()} → {df.index.max()}")
    return df, freq, meta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Configure `TrendAnalysis`

    Same construction as `TrendAnalysis_example.ipynb`. We pass `power_ac`
    directly (no synthetic soiling) so the fitted rate reflects the raw
    record.
    """)
    return


@app.cell
def _(df, freq, meta, rdtools):
    ta = rdtools.TrendAnalysis(
        df['power_ac'], df['poa'],
        temperature_ambient=df['Tamb'],
        gamma_pdc=meta['gamma_pdc'],
        interp_freq=freq,
        windspeed=df['wind_speed'],
        power_dc_rated=meta['power_dc_rated'],
        temperature_model=meta['temp_model_params'],
    )
    return (ta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Year-on-year baseline

    A standard YoY fit over the whole record for comparison. The
    signal-decomposition rate reported in §4 should be read against this
    baseline — differences arise from how each method handles the seasonal
    component and the loss function used.
    """)
    return


@app.cell
def _(ta):
    ta.sensor_analysis(analyses=['yoy_degradation'])
    yoy = ta.results['sensor']['yoy_degradation']
    ci_yoy = yoy['rd_confidence_interval']
    print(
        f"YoY whole-series:  Rd = {yoy['p50_rd']:+.3f} %/yr   "
        f"68% CI = [{ci_yoy[0]:+.3f}, {ci_yoy[1]:+.3f}]"
    )
    return (yoy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Interactive signal decomposition

    Adjust the controls below to explore different trend models and loss
    functions. Each change re-runs `ta.sensor_analysis` with the new
    parameters — the solve is reactive and typically completes in under a
    second.

    **Trend types**

    - `linear` — global affine trend $x_2 = a + bt$; one degradation rate
    - `pwl` — piecewise-linear with one breakpoint at the end of year 1;
      separate pre- and post-breakpoint rates
    - `monotone` — free non-increasing signal regularised on second
      differences; per-year and overall rates

    **Loss functions**

    - `l2` — least squares; optimal for Gaussian noise
    - `l1` — absolute value; robust to sparse large outliers
    - `huber` — interpolates between L2 (small residuals) and L1 (large);
      threshold set by *Huber M*
    - `quantile` — fits the $q$-th conditional quantile; use $q < 0.5$
      for a lower-envelope trend

    **Regularisation**

    $\lambda_\text{seasonal}$ controls how tightly the Fourier coefficients
    are penalised toward zero (large → smoother seasonal shape).
    $\lambda_\text{trend}$ penalises the trend slope or curvature (large →
    flatter or more slowly varying trend).
    """)
    return


@app.cell
def _(mo):
    trend_type_radio = mo.ui.radio(
        options=['linear', 'pwl', 'monotone'],
        value='linear',
        label='Trend type',
    )
    loss_radio = mo.ui.radio(
        options=['l2', 'l1', 'huber', 'quantile'],
        value='huber',
        label='Loss function',
    )
    numharmonics_slider = mo.ui.slider(
        start=1, stop=20, step=1, value=3,
        label='Num harmonics',
        show_value=True,
    )
    lam_seasonal_slider = mo.ui.slider(
        start=-4, stop=2, step=0.1, value=-1,
        label='λ seasonal (log₁₀)',
        show_value=True,
    )
    lam_trend_slider = mo.ui.slider(
        start=-4, stop=4, step=0.1, value=0,
        label='λ trend (log₁₀)',
        show_value=True,
    )
    q_slider = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.75,
        label='Quantile q',
        show_value=True,
    )
    huber_m_slider = mo.ui.slider(
        start=0.01, stop=2.0, step=0.01, value=0.05,
        label='Huber M',
        show_value=True,
    )
    log_toggle = mo.ui.switch(label='Log-transform input', value=True)
    lam_end_slider = mo.ui.slider(
        start=0.0, stop=10.0, step=0.1, value=0.0,
        label='λ end-drop',
        show_value=True,
    )

    loss_specific = mo.vstack([
        mo.md('**Loss-specific parameters**'),
        mo.hstack([q_slider, huber_m_slider], justify='start', gap='2rem'),
        mo.md(
            '_q is used only for `quantile` loss; '
            'M is used only for `huber` loss._'
        ),
    ])
    monotone_specific = mo.vstack([
        mo.md('**Monotone end-drop penalty**'),
        lam_end_slider,
        mo.md(
            '_λ end-drop penalises large drops in the final ~10 % of the record. '
            'Only used for `monotone` trend type; 0 disables it._'
        ),
    ])
    controls = mo.vstack([
        mo.md('## Decomposition controls'),
        mo.hstack([trend_type_radio, loss_radio], justify='start', gap='4rem'),
        mo.md('---'),
        mo.md('**Harmonics & regularisation**'),
        mo.hstack(
            [numharmonics_slider, lam_seasonal_slider, lam_trend_slider],
            justify='start',
            gap='2rem',
        ),
        mo.md('---'),
        loss_specific,
        mo.md('---'),
        monotone_specific,
        mo.md('---'),
        log_toggle,
        mo.md('---'),
    ])
    return (
        controls,
        huber_m_slider,
        lam_end_slider,
        lam_seasonal_slider,
        lam_trend_slider,
        log_toggle,
        loss_radio,
        numharmonics_slider,
        q_slider,
        trend_type_radio,
    )


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(
    huber_m_slider,
    lam_end_slider,
    lam_seasonal_slider,
    lam_trend_slider,
    log_toggle,
    loss_radio,
    numharmonics_slider,
    q_slider,
    ta,
    trend_type_radio,
):
    sd_kwargs = {
        'trend_type':    trend_type_radio.value,
        'loss':          loss_radio.value,
        'numharmonics':  numharmonics_slider.value,
        'lam_seasonal':  10 ** lam_seasonal_slider.value,
        'lam_trend':     10 ** lam_trend_slider.value,
        'lam_end':       lam_end_slider.value,
        'q':             q_slider.value,
        'huber_M':       huber_m_slider.value,
        'log_transform': log_toggle.value,
    }
    ta.sensor_analysis(analyses=['signal_decomposition'],
                       sd_kwargs={**sd_kwargs, 'n_bootstrap': 0},
                       skip_preprocess=True)
    results = ta.results['sensor']['signal_decomposition']
    rd = results['rd_pct']
    print(f"Rd = {rd:+.3f} %/yr  |  solver: {results['sd_trend_results']['problem_status']}")
    return results, sd_kwargs


@app.cell
def _(mo, results, sd):
    mo.md(
        sd.format_degradation_report(results['sd_trend_results'])
    )
    return


@app.cell
def _(plt, results, sd, ta):
    sd.plot_trend(results['sd_trend_results'], ta.sensor_aggregated_performance)
    plt.ylim(0.6, 1)
    plt.gcf()
    return


@app.cell
def _(plt, results, sd):
    sd.plot_decomposition(results['sd_trend_results'])
    _fig = plt.gcf()
    _fig.get_axes()[0]\
        .set_ylim(0.6, 1)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Confidence interval

    Run a moving-block bootstrap (500 replicates, block length ≈ 1 year) on the
    residuals of the current fit to estimate a 68.2 % confidence interval on the
    degradation rate. Runtime is roughly 500× the point-estimate solve time
    (~15-30 sec on a typical machine).

    Press the button to run. Uses the same model parameters as the widget
    settings above. The table below compares the result with the year-on-year
    baseline from §3.
    """)
    return


@app.cell
def _(mo):
    run_ci = mo.ui.run_button(label='run CI bootstrap (500 replicates)')
    run_ci
    return (run_ci,)


@app.cell
def _(mo, run_ci, sd_kwargs, ta, yoy):
    mo.stop(not run_ci.value)

    ta.sensor_analysis(
        analyses=['signal_decomposition'],
        sd_kwargs=sd_kwargs,
        skip_preprocess=True,
    )
    _ci_res = ta.results['sensor']['signal_decomposition']
    _yoy    = yoy

    _rd_sd  = _ci_res['rd_pct']
    _ci_sd  = _ci_res['rd_confidence_interval']
    _rd_yoy = _yoy['p50_rd']
    _ci_yoy = _yoy['rd_confidence_interval']

    _hdr = f"{'Method':<26} {'Rd (%/yr)':>10}  {'68% CI':}"
    _sep = '-' * 58
    _row_yoy = f"{'YoY (whole series)':<26} {_rd_yoy:>+10.3f}  [{_ci_yoy[0]:+.3f}, {_ci_yoy[1]:+.3f}]"
    _row_sd  = f"{'Signal decomp':<26} {_rd_sd:>+10.3f}  [{_ci_sd[0]:+.3f}, {_ci_sd[1]:+.3f}]"
    print(f"{_hdr}\n{_sep}\n{_row_yoy}\n{_row_sd}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Fit stability

    How much does the estimated degradation rate change as the available
    data record grows? `analyze_fit_stability` solves the decomposition for
    every valid window from one year up to the full record and plots:

    1. All trend fits $x_2$ coloured by window length (spaghetti plot)
    2. Normalised RMSD of $x_2$ between successive windows
    3. Rate history with convergence markers (±0.1 %/yr of the final value)
    4. Absolute rate change between successive windows

    Press the button to run. Uses the same parameters as the current
    widget settings above.
    """)
    return


@app.cell
def _(mo):
    run_stability = mo.ui.run_button(label='run stability analysis')
    run_stability
    return (run_stability,)


@app.cell
def _(mo, results, run_stability, sd):
    mo.stop(not run_stability.value)

    stability = sd.analyze_fit_stability(
        results['sd_trend_results']['y'],
        results['sd_trend_results']['args'],
    )
    stability_fig = sd.plot_stability(stability)
    return (stability_fig,)


@app.cell
def _(stability_fig):
    stability_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Degradation animation

    Renders a frame-by-frame animation of the decomposition fit as the
    data record grows, saved to `degradation_animation.mp4` (requires
    system `ffmpeg`) or `.gif` (requires Pillow).

    Press the button to render. Uses the same parameters as the current
    widget settings. Expect ~30–60 s for the full PVDAQ 4 record at the
    default step size.
    """)
    return


@app.cell
def _(mo):
    run_ani = mo.ui.run_button(label='run animation')
    run_ani
    return (run_ani,)


@app.cell
def _(mo, results, run_ani, sd):
    mo.stop(not run_ani.value)

    sd.animate_degradation(
        results['sd_trend_results']['y'],
        results['sd_trend_results']['args'],
        output_path='degradation_animation.mp4',
        fps=12,
        step=5,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Conditional quantile sweep

    Solves the decomposition at several quantile levels
    ($q \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$) and overlays each fitted
    quantile estimate on the input data — a "conditional quantile time
    series". Loss is forced to `quantile`; every other setting (trend
    type, harmonics, regularisation, log-transform, Huber M — where
    relevant) is wired from the widget panel above.

    Press the button to run.
    """)
    return


@app.cell
def _(mo):
    run_qsweep = mo.ui.run_button(label='run quantile sweep')
    run_qsweep
    return (run_qsweep,)


@app.cell
def _(
    huber_m_slider,
    lam_end_slider,
    lam_seasonal_slider,
    lam_trend_slider,
    log_toggle,
    mo,
    numharmonics_slider,
    plt,
    run_qsweep,
    ta,
    trend_type_radio,
):
    mo.stop(not run_qsweep.value)

    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Settings shared across all solves, wired from the widgets.
    # Loss is forced to 'quantile' since we are sweeping q.
    base_kwargs = {
        'trend_type':    trend_type_radio.value,
        'loss':          'quantile',
        'numharmonics':  numharmonics_slider.value,
        'lam_seasonal':  10 ** lam_seasonal_slider.value,
        'lam_trend':     10 ** lam_trend_slider.value,
        'lam_end':       lam_end_slider.value,
        'huber_M':       huber_m_slider.value,
        'log_transform': log_toggle.value,
    }

    quantile_fits = {}
    for q in quantiles:
        sd_kwargs_q = {**base_kwargs, 'q': q, 'n_bootstrap': 0}
        ta.sensor_analysis(
            analyses=['signal_decomposition'],
            sd_kwargs=sd_kwargs_q,
            skip_preprocess=True,
        )
        res_q = ta.results['sensor']['signal_decomposition']
        tr = res_q['sd_trend_results']

        # Fitted conditional quantile curve = seasonal + trend (x1 + x2).
        # Swap to `tr['x2']` alone if you want only the trend component.
        # estimate = tr['components']['x1'] + tr['components']['x2']
        estimate = tr['components']['x2']

        quantile_fits[q] = {
            'estimate': estimate,
            'rd_pct':   res_q['rd_pct'],
            'status':   tr['problem_status'],
        }
        print(
            f"q = {q:.2f}  Rd = {res_q['rd_pct']:+.3f} %/yr  "
            f"|  solver: {tr['problem_status']}"
        )

    # Build the conditional quantile time series plot.
    y_input = ta.sensor_aggregated_performance

    fig_qsweep, ax_qsweep = plt.subplots(figsize=(11, 5))
    ax_qsweep.plot(
        y_input.index, y_input.values,
        '.', color='0.7', markersize=2, alpha=0.5,
        label='input (aggregated)',
    )

    cmap = plt.get_cmap('viridis')
    for i, q in enumerate(quantiles):
        est = quantile_fits[q]['estimate']
        ax_qsweep.plot(
            y_input.index, est,
            color=cmap(i / (len(quantiles) - 1)),
            linewidth=1.8,
            label=f'q = {q:.1f}  ({quantile_fits[q]["rd_pct"]:+.2f} %/yr)',
        )

    ax_qsweep.set_xlabel('date')
    ax_qsweep.set_ylabel('normalised energy')
    ax_qsweep.set_title(
        f'Conditional quantile time series  '
        f'(trend: {trend_type_radio.value})'
    )
    ax_qsweep.legend(loc='best', fontsize=8, ncol=2)
    ax_qsweep.grid(True, alpha=0.3)
    fig_qsweep.tight_layout()
    plt.ylim(0.6, 1)
    return (fig_qsweep,)


@app.cell
def _(fig_qsweep):
    fig_qsweep
    return


if __name__ == "__main__":
    app.run()
