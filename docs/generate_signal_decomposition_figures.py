"""Generate static figures for the signal-decomposition documentation.

Run from the repository root with::

    pixi run -e dev python docs/generate_signal_decomposition_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rdtools


DOCS_DIR = Path(__file__).resolve().parent
IMAGE_DIR = DOCS_DIR / "sphinx" / "source" / "_images"


def fitted_limits(values, margin_fraction=0.15, minimum_margin=0.01):
    """Return plotting limits centered on a fitted component's range."""
    finite = np.asarray(values)[np.isfinite(values)]
    margin = max(np.ptp(finite) * margin_fraction, minimum_margin)
    return finite.min() - margin, finite.max() + margin


def central_limits(values, lower=1, upper=99, margin_fraction=0.1):
    """Return robust plotting limits without changing the fitted data."""
    low, high = np.nanpercentile(values, [lower, upper])
    margin = max((high - low) * margin_fraction, 0.005)
    return low - margin, high + margin


def load_pvdaq_system_4():
    """Load the public PVDAQ record used by the documentation notebooks."""
    data = pd.read_csv(
        DOCS_DIR / "PVDAQ_system_4_2010-2016_subset_soilsignal.csv",
        index_col=0,
        parse_dates=True,
    ).rename(
        columns={
            "ac_power": "power_ac",
            "ambient_temp": "Tamb",
            "poa_irradiance": "poa",
        }
    )
    data.index = data.index.tz_localize("Etc/GMT+7")
    return data


def make_analysis(data):
    """Construct the same sensor analysis used in the SD examples."""
    return rdtools.TrendAnalysis(
        data["power_ac"],
        data["poa"],
        temperature_ambient=data["Tamb"],
        gamma_pdc=-0.0034,
        interp_freq=pd.infer_freq(data.index[:10]),
        windspeed=data["wind_speed"],
        power_dc_rated=1000.0,
        temperature_model="open_rack_glass_polymer",
    )


def main():
    """Fit the documented SD models and save their component summaries."""
    data = load_pvdaq_system_4()
    analysis = make_analysis(data)
    analysis.sensor_analysis(
        analyses=["signal_decomposition"],
        sd_kwargs={"n_bootstrap": 0},
    )
    basic = analysis.results["sensor"]["signal_decomposition"]
    basic_info = basic["sd_trend_results"]
    basic_dates = analysis.sensor_aggregated_performance.index
    basic_components = basic_info["components"]
    figure, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(
        basic_dates,
        analysis.sensor_aggregated_performance,
        color="0.55",
        linewidth=0.7,
        alpha=0.55,
        label="Normalized daily performance",
    )
    axes[0].plot(
        basic_dates,
        basic_components["fit"],
        color="#0072B2",
        linewidth=2,
        label="Seasonal + trend fit",
    )
    axes[0].set_ylabel("Performance")
    axes[0].legend(loc="lower left")
    axes[0].set_title("Basic signal decomposition of public PVDAQ system 4")
    axes[0].set_ylim(fitted_limits(basic_components["fit"]))

    component_panels = (
        ("x1", "Seasonal ratio", "#009E73"),
        ("x2", "Degradation trend", "#D55E00"),
        ("x3", "Residual ratio", "#7A5195"),
    )
    for axis, (key, label, color) in zip(axes[1:], component_panels):
        axis.plot(basic_dates, basic_components[key], color=color, linewidth=1.5)
        axis.set_ylabel(label)
    axes[1].axhline(1, color="0.5", linestyle=":", linewidth=1)
    axes[3].axhline(1, color="0.5", linestyle=":", linewidth=1)
    axes[3].set_ylim(central_limits(basic_components["x3"]))
    axes[3].set_xlabel("Date")
    figure.tight_layout()
    figure.savefig(
        IMAGE_DIR / "signal_decomposition_basic.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Establish the PVDAQ record's own PWL trend, then apply a known
    # multiplicative early-life burn-in signal to the measured power. Taking
    # the ratio of the two recovered trends isolates recovery of the injected
    # signal from the trend already present in the public record.
    analysis.sensor_analysis(
        analyses=["signal_decomposition"],
        sd_kwargs={"trend_type": "pwl", "n_bootstrap": 0},
        skip_preprocess=True,
    )
    baseline = analysis.results["sensor"]["signal_decomposition"]

    elapsed_days = (
        data.index - data.index.min()
    ).total_seconds() / (24 * 60 * 60)
    first_year = np.minimum(elapsed_days, 365.2425)
    later_years = np.maximum(elapsed_days - 365.2425, 0)
    injected_ratio = np.exp(
        np.log(0.96) * first_year / 365.2425
        + np.log(0.995) * later_years / 365.2425
    )
    altered_data = data.copy()
    altered_data["power_ac"] *= injected_ratio

    altered_analysis = make_analysis(altered_data)
    altered_analysis.sensor_analysis(
        analyses=["signal_decomposition"],
        sd_kwargs={"trend_type": "pwl", "n_bootstrap": 0},
    )
    altered = altered_analysis.results["sensor"]["signal_decomposition"]
    altered_info = altered["sd_trend_results"]
    baseline_info = baseline["sd_trend_results"]
    dates = altered_analysis.sensor_aggregated_performance.index

    recovered_ratio = (
        altered_info["components"]["x2"]
        / baseline_info["components"]["x2"]
    )
    recovered_ratio /= recovered_ratio[0]
    daily_elapsed = (dates - dates.min()).days.to_numpy()
    known_ratio = np.exp(
        np.log(0.96) * np.minimum(daily_elapsed, 365.2425) / 365.2425
        + np.log(0.995) * np.maximum(daily_elapsed - 365.2425, 0) / 365.2425
    )

    figure, axes = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    observed = altered_analysis.sensor_aggregated_performance
    axes[0].plot(
        dates, observed, color="0.55", linewidth=0.7, alpha=0.55,
        label="Normalized daily performance",
    )
    axes[0].plot(
        dates,
        altered_info["components"]["fit"],
        color="#0072B2",
        linewidth=2,
        label="Seasonal + PWL trend fit",
    )
    axes[0].set_ylabel("Performance ratio")
    axes[0].set_title("Year-1 breakpoint model on public PVDAQ + synthetic burn-in")
    axes[0].legend(loc="lower left")
    axes[0].set_ylim(fitted_limits(altered_info["components"]["fit"]))

    axes[1].plot(
        dates, known_ratio, color="black", linestyle="--", linewidth=2,
        label="Known injected profile",
    )
    axes[1].plot(
        dates, recovered_ratio, color="#D55E00", linewidth=2,
        label="Recovered incremental trend",
    )
    axes[1].axvline(
        dates.min() + pd.Timedelta(days=365.2425),
        color="0.5", linestyle=":", linewidth=1.5,
    )
    axes[1].set_ylabel("Added effect")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(
        IMAGE_DIR / "signal_decomposition_year1_breakpoint.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
