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

    return mo, pd, rdtools, sd


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
    return


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
        value='l2',
        label='Loss function',
    )
    numharmonics_slider = mo.ui.slider(
        start=1, stop=20, step=1, value=6,
        label='Num harmonics',
        show_value=True,
    )
    lam_seasonal_slider = mo.ui.slider(
        start=-4, stop=2, step=0.25, value=-1,
        label='λ seasonal (log₁₀)',
        show_value=True,
    )
    lam_trend_slider = mo.ui.slider(
        start=-4, stop=4, step=0.25, value=0,
        label='λ trend (log₁₀)',
        show_value=True,
    )
    q_slider = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.75,
        label='Quantile q',
        show_value=True,
    )
    huber_m_slider = mo.ui.slider(
        start=0.01, stop=2.0, step=0.01, value=0.2,
        label='Huber M',
        show_value=True,
    )
    log_toggle = mo.ui.switch(label='Log-transform input', value=False)

    loss_specific = mo.vstack([
        mo.md('**Loss-specific parameters**'),
        mo.hstack([q_slider, huber_m_slider], justify='start', gap='2rem'),
        mo.md(
            '_q is used only for `quantile` loss; '
            'M is used only for `huber` loss._'
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
        log_toggle,
        mo.md('---'),
    ])
    return (
        controls,
        huber_m_slider,
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
        'q':             q_slider.value,
        'huber_M':       huber_m_slider.value,
        'log_transform': log_toggle.value,
    }
    ta.sensor_analysis(analyses=['signal_decomposition'], sd_kwargs=sd_kwargs,
                       skip_preprocess=True)
    results = ta.results['sensor']['signal_decomposition']
    rd = results['rd_pct']
    print(f"Rd = {rd:+.3f} %/yr  |  solver: {results['sd_trend_results']['problem_status']}")
    return (results,)


@app.cell
def _(mo, results, sd):
    mo.md(
        sd.format_degradation_report(results['sd_trend_results'])
    )
    return


@app.cell
def _(results, sd):
    sd.plot_decomposition(results['sd_trend_results'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Fit stability

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
    ## 6. Degradation animation

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


if __name__ == "__main__":
    app.run()
