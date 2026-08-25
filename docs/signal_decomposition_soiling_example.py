import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import rdtools
    from rdtools import signal_decomposition as sd

    return mo, np, pd, plt, rdtools, sd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Signal decomposition with soiling

    This notebook applies the optional soiling component in
    `rdtools.signal_decomposition.degradation` to two versions of the same
    PVDAQ system 4 record:

    1. the measured AC power, without an added synthetic soiling signal; and
    2. the measured AC power multiplied by the synthetic soiling factor
       distributed with the RdTools example dataset.

    Holding the site, weather, filters, and normalization workflow fixed makes
    this a direct test of the soiling option. The first case demonstrates the
    structural-null behavior. The second provides a known reference component
    against which the fitted soiling ratio can be compared.

    With `include_soiling=True`, RdTools solves a nine-candidate regularization
    path and selects the first structurally coherent component. If no candidate
    qualifies, it performs a true seasonal-plus-degradation decomposition and
    returns an exact unity soiling factor. This is intentionally more work than
    the default degradation-only analysis, which remains a single decomposition.

    The component is designed for conventional dry soil or sand accumulation
    interrupted by discrete cleaning events. Biological growth on modules is a
    different process, and persistent partial outages must be detected or
    corrected upstream because their level shifts can resemble soiling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load the established PVDAQ example dataset

    PVDAQ system 4 does not experience substantial natural soiling in this
    record. The released dataset therefore includes a synthetic `soiling`
    column for method demonstrations. The file is downloaded once and cached
    locally, following the existing RdTools documentation examples.
    """)
    return


@app.cell
def _(pd):
    file_url = (
        "https://github.com/NatLabRockies/rdtools/releases/download/3.0.0/"
        "pvdaq_system_4_2010-2016_subset_soil_signal.csv"
    )
    cache_file = "PVDAQ_system_4_2010-2016_subset_soilsignal.pickle"

    try:
        pvdaq = pd.read_pickle(cache_file)
    except FileNotFoundError:
        pvdaq = pd.read_csv(file_url, index_col=0, parse_dates=True)
        pvdaq.to_pickle(cache_file)

    pvdaq = pvdaq.rename(columns={
        "ac_power": "power_ac",
        "ambient_temp": "Tamb",
        "poa_irradiance": "poa",
    })

    metadata = {
        "timezone": "Etc/GMT+7",
        "gamma_pdc": -0.0034,
        "power_dc_rated": 1000.0,
        "temp_model_params": "open_rack_glass_polymer",
    }

    if pvdaq.index.tz is None:
        pvdaq.index = pvdaq.index.tz_localize(metadata["timezone"])
    input_frequency = pd.infer_freq(pvdaq.index[:10])
    pvdaq["power_with_soiling"] = pvdaq["power_ac"] * pvdaq["soiling"]

    print(
        f"Loaded {len(pvdaq):,} rows from {pvdaq.index.min()} "
        f"through {pvdaq.index.max()}."
    )
    return input_frequency, metadata, pvdaq


@app.cell
def _(mo, plt, pvdaq):
    fig_input, axes_input = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, constrained_layout=True
    )
    axes_input[0].plot(
        pvdaq.index, pvdaq["power_ac"], ".", ms=1, alpha=0.15,
        label="measured AC power",
    )
    axes_input[0].plot(
        pvdaq.index, pvdaq["power_with_soiling"], ".", ms=1, alpha=0.15,
        label="with synthetic soiling",
    )
    axes_input[0].set_ylabel("AC power")
    axes_input[0].legend()
    axes_input[1].plot(pvdaq.index, pvdaq["soiling"], lw=0.8)
    axes_input[1].axhline(1, color="black", lw=0.6, ls="--")
    axes_input[1].set_ylabel("synthetic\nsoiling ratio")
    axes_input[1].set_xlabel("date")
    mo.vstack([
        mo.md(
            "The synthetic factor is applied multiplicatively to the measured "
            "power; all weather channels are unchanged."
        ),
        fig_input,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Normalize and filter both cases identically

    Each case gets its own `TrendAnalysis` instance because its PV power input
    differs. Configuration and environmental inputs are otherwise identical.
    `TrendAnalysis` automatically passes daily aggregated insolation into the
    signal-decomposition backend, so both time-averaged and
    insolation-weighted loss metrics are available.
    """)
    return


@app.cell
def _(input_frequency, metadata, pvdaq, rdtools):
    shared_arguments = {
        "poa_global": pvdaq["poa"],
        "temperature_ambient": pvdaq["Tamb"],
        "gamma_pdc": metadata["gamma_pdc"],
        "interp_freq": input_frequency,
        "windspeed": pvdaq["wind_speed"],
        "power_dc_rated": metadata["power_dc_rated"],
        "temperature_model": metadata["temp_model_params"],
    }

    analysis_without_synthetic = rdtools.TrendAnalysis(
        pvdaq["power_ac"], **shared_arguments
    )
    analysis_with_synthetic = rdtools.TrendAnalysis(
        pvdaq["power_with_soiling"], **shared_arguments
    )
    return analysis_with_synthetic, analysis_without_synthetic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Run the soiling-enabled decomposition

    The validated model preset and selector thresholds are deliberately not
    exposed as tuning controls. Calling `include_soiling=True` requests the
    frozen log-Huber model and coherence sweep. We initially set
    `n_bootstrap=0` so this section isolates the point estimate; uncertainty is
    added later without changing the selected model.
    """)
    return


@app.cell
def _(analysis_with_synthetic, analysis_without_synthetic):
    analysis_without_synthetic.sensor_analysis(
        analyses=["signal_decomposition"],
        sd_kwargs={"include_soiling": True, "n_bootstrap": 0},
    )
    result_without_synthetic = analysis_without_synthetic.results["sensor"][
        "signal_decomposition"
    ]

    analysis_with_synthetic.sensor_analysis(
        analyses=["signal_decomposition"],
        sd_kwargs={"include_soiling": True, "n_bootstrap": 0},
    )
    result_with_synthetic = analysis_with_synthetic.results["sensor"][
        "signal_decomposition"
    ]
    return result_with_synthetic, result_without_synthetic


@app.cell
def _(pd, result_with_synthetic, result_without_synthetic):
    def summarize_case(label, result):
        info = result["sd_trend_results"]
        soiling = info["soiling"]
        selector = soiling["selector"]
        loss = soiling["loss"]
        overall_rate = soiling["rate_summary"].set_index("period").loc["overall"]
        return {
            "case": label,
            "detected": selector["detected"],
            "selected_lam_down": selector["selected_lam_soiling_down"],
            "degradation_pct_yr": result["rd_pct"],
            "time_averaged_loss_pct": loss["time_averaged_loss_pct"],
            "insolation_weighted_loss_pct": loss[
                "insolation_weighted_loss_pct"
            ],
            "median_local_rate_pct_day": overall_rate[
                "median_rate_pct_day"
            ],
            "valid_intervals": int(soiling["intervals"]["valid"].sum()),
        }

    comparison = pd.DataFrame([
        summarize_case("measured PVDAQ", result_without_synthetic),
        summarize_case("PVDAQ + synthetic soiling", result_with_synthetic),
    ]).set_index("case")
    print(comparison.to_string())
    comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The important first comparison is the selector decision. A null result is
    not a heavily regularized soiling curve: it is a newly solved model without
    a soiling component, accompanied by an exact all-ones factor. The synthetic
    case should select an interior regularization weight and report nonzero
    loss and accumulation-rate estimates.
    """)
    return


@app.cell
def _(pd, result_with_synthetic, result_without_synthetic):
    metrics_without = result_without_synthetic["sd_trend_results"]["soiling"][
        "selector"
    ]["candidate_metrics"].assign(case="measured PVDAQ")
    metrics_with = result_with_synthetic["sd_trend_results"]["soiling"][
        "selector"
    ]["candidate_metrics"].assign(case="PVDAQ + synthetic soiling")
    selector_metrics = pd.concat(
        [metrics_without, metrics_with], ignore_index=True
    ).set_index(["case", "candidate_index"])
    selector_metrics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Compare fitted and known soiling components

    The synthetic reference is aggregated to daily resolution and aligned to
    the filtered signal-decomposition index. It is a useful known component,
    but not a claim that all real low-to-moderate soiling follows this exact
    generator. In particular, the interpretation here is dry accumulation and
    cleaning, not biological fouling.
    The unmodified PVDAQ case has a reference ratio of one because no synthetic
    loss was added.
    """)
    return


@app.cell
def _(
    analysis_with_synthetic,
    analysis_without_synthetic,
    np,
    pd,
    plt,
    pvdaq,
    result_with_synthetic,
    result_without_synthetic,
):
    clean_index = analysis_without_synthetic.sensor_aggregated_performance.index
    soiled_index = analysis_with_synthetic.sensor_aggregated_performance.index
    known_daily_soiling = pvdaq["soiling"].resample("D").mean()
    known_soiling_aligned = known_daily_soiling.reindex(soiled_index)

    estimated_without = result_without_synthetic["sd_trend_results"][
        "components"
    ]["soiling"]
    estimated_with = result_with_synthetic["sd_trend_results"]["components"][
        "soiling"
    ]

    known_ratio = known_soiling_aligned.to_numpy()
    component_rmse = float(np.sqrt(np.nanmean(
        (estimated_with - known_ratio) ** 2
    )))
    component_mae = float(np.nanmean(np.abs(estimated_with - known_ratio)))

    insolation = analysis_with_synthetic.sensor_aggregated_insolation.reindex(
        soiled_index
    ).to_numpy()
    valid_weight = (
        np.isfinite(known_ratio) & np.isfinite(insolation) & (insolation > 0)
    )
    known_time_loss = float(100 * np.nanmean(1 - known_ratio))
    known_insolation_loss = float(100 * (
        1 - np.sum(insolation[valid_weight] * known_ratio[valid_weight])
        / np.sum(insolation[valid_weight])
    ))
    estimated_loss = result_with_synthetic["sd_trend_results"]["soiling"][
        "loss"
    ]

    component_scores = pd.Series({
        "RMSE (soiling ratio)": component_rmse,
        "MAE (soiling ratio)": component_mae,
        "known time-averaged loss (%)": known_time_loss,
        "estimated time-averaged loss (%)": estimated_loss[
            "time_averaged_loss_pct"
        ],
        "known insolation-weighted loss (%)": known_insolation_loss,
        "estimated insolation-weighted loss (%)": estimated_loss[
            "insolation_weighted_loss_pct"
        ],
    }, name="synthetic-component recovery")

    fig_components, axes_components = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, constrained_layout=True
    )
    axes_components[0].plot(
        clean_index, estimated_without, lw=1.1, label="estimated"
    )
    axes_components[0].axhline(
        1, color="black", lw=0.7, ls="--", label="no added synthetic loss"
    )
    axes_components[0].set_title("Measured PVDAQ")
    axes_components[0].set_ylabel("soiling ratio")
    axes_components[0].legend()

    axes_components[1].plot(
        soiled_index, known_soiling_aligned, color="black", lw=0.8,
        label="known synthetic component",
    )
    axes_components[1].plot(
        soiled_index, estimated_with, color="C3", lw=1.1,
        label="estimated component",
    )
    axes_components[1].set_title("PVDAQ + synthetic soiling")
    axes_components[1].set_ylabel("soiling ratio")
    axes_components[1].set_xlabel("date")
    axes_components[1].legend()
    return component_scores, fig_components


@app.cell
def _(component_scores, fig_components, mo):
    mo.vstack([fig_components, component_scores])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Losses by calendar quarter

    Time-averaged loss is always returned. Because this analysis runs through
    `TrendAnalysis`, aligned daily insolation is also available and the result
    includes SRR-style insolation-weighted loss. Q1–Q4 values pool the same
    calendar quarter across all years in the record. These first fits use
    `n_bootstrap=0`, so this point-estimate table intentionally omits CI fields;
    uncertainty is computed in §8.
    """)
    return


@app.cell
def _(pd, result_with_synthetic, result_without_synthetic):
    quarterly_without = result_without_synthetic["sd_trend_results"]["soiling"][
        "loss"
    ]["quarterly"].assign(case="measured PVDAQ")
    quarterly_with = result_with_synthetic["sd_trend_results"]["soiling"][
        "loss"
    ]["quarterly"].assign(case="PVDAQ + synthetic soiling")
    quarterly_losses = pd.concat(
        [quarterly_without, quarterly_with], ignore_index=True
    ).set_index(["case", "quarter"])[[
        "time_averaged_loss_pct",
        "insolation_weighted_loss_pct",
    ]]
    quarterly_losses
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Local soiling rates

    Negative `%/day` values indicate accumulation. Cleaning-to-cleaning
    intervals shorter than seven days, or intervals with a positive fitted
    slope, are retained for diagnostics but marked invalid. Summary medians
    weight valid interval rates by their calendar-day exposure overall and in
    each climatological quarter. As above, the point-estimate table omits the
    CI fields until the bootstrap is run. These local rates describe dry
    accumulation between detected recoveries; they should not be interpreted
    as biological-growth rates.
    """)
    return


@app.cell
def _(pd, result_with_synthetic, result_without_synthetic):
    rates_without = result_without_synthetic["sd_trend_results"]["soiling"][
        "rate_summary"
    ].assign(case="measured PVDAQ")
    rates_with = result_with_synthetic["sd_trend_results"]["soiling"][
        "rate_summary"
    ].assign(case="PVDAQ + synthetic soiling")
    rate_comparison = pd.concat(
        [rates_without, rates_with], ignore_index=True
    ).set_index(["case", "period"])[[
        "median_rate_pct_day",
        "interval_count",
        "day_count",
    ]]
    rate_comparison
    return


@app.cell
def _(result_with_synthetic):
    valid_soiling_intervals = result_with_synthetic["sd_trend_results"][
        "soiling"
    ]["intervals"].query("valid")
    valid_soiling_intervals
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Full decomposition plots

    The added panel is the multiplicative soiling ratio. In the null case it is
    exactly one; in the detected case it participates in the displayed fit as
    `seasonal × trend × soiling`.
    """)
    return


@app.cell
def _(mo, result_without_synthetic, sd):
    figure_without = sd.plot_decomposition(
        result_without_synthetic["sd_trend_results"]
    )
    mo.vstack([mo.md("### Measured PVDAQ"), figure_without])
    return


@app.cell
def _(mo, result_with_synthetic, sd):
    figure_with = sd.plot_decomposition(
        result_with_synthetic["sd_trend_results"]
    )
    mo.vstack([mo.md("### PVDAQ + synthetic soiling"), figure_with])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Bootstrap uncertainty

    The selector is not rerun inside bootstrap replicates. Each replicate fits
    only the terminal model selected for that data stream. Use the control to
    balance runtime and interval stability, then press the button. Script mode
    runs four replicates as a lightweight execution check; interactive mode
    uses the selected count.
    """)
    return


@app.cell
def _(mo):
    bootstrap_count = mo.ui.slider(
        start=20, stop=500, step=20, value=100,
        label="bootstrap replicates", show_value=True,
    )
    run_bootstrap = mo.ui.run_button(label="run final-model bootstrap")
    mo.hstack([bootstrap_count, run_bootstrap], justify="start", gap="2rem")
    return bootstrap_count, run_bootstrap


@app.cell
def _(mo):
    script_mode = mo.app_meta().mode == "script"
    return (script_mode,)


@app.cell
def _(
    analysis_with_synthetic,
    analysis_without_synthetic,
    bootstrap_count,
    mo,
    pd,
    run_bootstrap,
    script_mode,
):
    mo.stop(not (script_mode or run_bootstrap.value))
    replicate_count = 4 if script_mode else bootstrap_count.value

    analysis_without_synthetic.sensor_analysis(
        analyses=["signal_decomposition"],
        sd_kwargs={
            "include_soiling": True,
            "n_bootstrap": replicate_count,
            "random_state": 42,
        },
        skip_preprocess=True,
    )
    bootstrap_without = analysis_without_synthetic.results["sensor"][
        "signal_decomposition"
    ]

    analysis_with_synthetic.sensor_analysis(
        analyses=["signal_decomposition"],
        sd_kwargs={
            "include_soiling": True,
            "n_bootstrap": replicate_count,
            "random_state": 42,
        },
        skip_preprocess=True,
    )
    bootstrap_with = analysis_with_synthetic.results["sensor"][
        "signal_decomposition"
    ]

    bootstrap_rows = []
    for label, result in [
        ("measured PVDAQ", bootstrap_without),
        ("PVDAQ + synthetic soiling", bootstrap_with),
    ]:
        soiling = result["sd_trend_results"]["soiling"]
        loss = soiling["loss"]
        quarterly = loss["quarterly"].set_index("quarter")
        rates = soiling["rate_summary"].set_index("period")
        rate = rates.loc["overall"]
        bootstrap_rows.append({
            "case": label,
            "period": "overall",
            "time_loss_pct": loss["time_averaged_loss_pct"],
            "time_loss_ci_low": loss["time_averaged_loss_ci"][0],
            "time_loss_ci_high": loss["time_averaged_loss_ci"][1],
            "insolation_loss_pct": loss["insolation_weighted_loss_pct"],
            "insolation_loss_ci_low": loss["insolation_weighted_loss_ci"][0],
            "insolation_loss_ci_high": loss["insolation_weighted_loss_ci"][1],
            "median_rate_pct_day": rate["median_rate_pct_day"],
            "rate_ci_low": rate["rate_ci_low"],
            "rate_ci_high": rate["rate_ci_high"],
        })
        for quarter in range(1, 5):
            loss_row = quarterly.loc[quarter]
            rate_row = rates.loc[f"Q{quarter}"]
            bootstrap_rows.append({
                "case": label,
                "period": f"Q{quarter}",
                "time_loss_pct": loss_row["time_averaged_loss_pct"],
                "time_loss_ci_low": loss_row["time_averaged_loss_ci_low"],
                "time_loss_ci_high": loss_row["time_averaged_loss_ci_high"],
                "insolation_loss_pct": loss_row["insolation_weighted_loss_pct"],
                "insolation_loss_ci_low": loss_row["insolation_weighted_loss_ci_low"],
                "insolation_loss_ci_high": loss_row["insolation_weighted_loss_ci_high"],
                "median_rate_pct_day": rate_row["median_rate_pct_day"],
                "rate_ci_low": rate_row["rate_ci_low"],
                "rate_ci_high": rate_row["rate_ci_high"],
            })
    bootstrap_summary = pd.DataFrame(bootstrap_rows).set_index(["case", "period"])
    bootstrap_summary
    return


if __name__ == "__main__":
    app.run()
