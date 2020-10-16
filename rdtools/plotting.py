'''Functions for plotting degradation and soiling analysis results.'''

import matplotlib.pyplot as plt
import numpy as np
import warnings

def degradation_summary_plots(yoy_rd, yoy_ci, yoy_info, normalized_yield,
                              hist_xmin=None, hist_xmax=None, bins=None,
                              scatter_ymin=None, scatter_ymax=None,
                              plot_color=None, summary_title=None,
                              scatter_alpha=0.5):
    '''
    Create plots (scatter plot and histogram) that summarize degradation
    analysis results.

    Parameters
    ----------
    yoy_rd : float
        rate of relative performance change in %/yr
    yoy_ci : float
        one-sigma confidence interval of degradation rate estimate
    yoy_info : dict
        a dictionary with keys:

        * YoY_values - pandas series of right-labeled year on year slopes
        * renormalizing_factor - float value used to recenter data
        * exceedance_level - the degradation rate that was outperformed with
          a probability given by the ``exceedance_prob`` parameter in
          the :py:func:`.degradation.degradation_year_on_year`

    normalized_yield : pd.Series
         PV yield data that is normalized, filtered and aggregated
    hist_xmin : float, optional
        lower limit of x-axis for the histogram
    hist_xmax : float, optional
        upper limit of x-axis for the histogram
    bins : int, optional
        Number of bins in the histogram distribution. If omitted,
        ``len(yoy_values) // 40`` will be used
    scatter_ymin : float, optional
        lower limit of y-axis for the scatter plot
    scatter_ymax : float, optional
        upper limit of y-axis for the scatter plot
    plot_color : str, optional
        color of the summary plots
    summary_title : str, optional
        overall title for summary plots
    scatter_alpha : float, default 0.5
        Transparency of the scatter plot

    Note
    ----
    It should be noted that the yoy_rd, yoy_ci and yoy_info are the outputs
    from :py:func:`.degradation.degradation_year_on_year`.

    Returns
    -------
    fig : matplotlib Figure
        Figure with two axes
    '''

    yoy_values = yoy_info['YoY_values']

    if bins is None:
        bins = len(yoy_values) // 40

    bins = int(min(bins, len(yoy_values)))

    # Calculate the degradation line
    start = normalized_yield.index[0]
    end = normalized_yield.index[-1]
    years = (end - start).days / 365.25
    yoy_values = yoy_info['YoY_values']

    x = [start, end]
    y = [1, 1 + (yoy_rd * years) / 100.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    ax2.hist(yoy_values, label='YOY', bins=bins, color=plot_color)
    ax2.axvline(x=yoy_rd, color='black', linestyle='dashed', linewidth=3)

    ax2.set_xlim(hist_xmin, hist_xmax)

    label = (
        ' $R_{d}$ = %.2f%%/yr \n'
        'confidence interval: \n'
        '%.2f to %.2f %%/yr' % (yoy_rd, yoy_ci[0], yoy_ci[1])
    )
    ax2.annotate(label, xy=(0.5, 0.7), xycoords='axes fraction',
                 bbox=dict(facecolor='white', edgecolor=None, alpha=0))
    ax2.set_xlabel('Annual degradation (%)')

    renormalized_yield = normalized_yield / yoy_info['renormalizing_factor']
    ax1.plot(renormalized_yield.index, renormalized_yield, 'o',
             color=plot_color, alpha=scatter_alpha)
    ax1.plot(x, y, 'k--', linewidth=3)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Renormalized energy')

    ax1.set_ylim(scatter_ymin, scatter_ymax)

    fig.autofmt_xdate()

    if summary_title is not None:
        fig.suptitle(summary_title)

    return fig


def soiling_monte_carlo_plot(soiling_info, normalized_yield, point_alpha=0.5,
                             profile_alpha=0.05, ymin=None, ymax=None,
                             profiles=None, point_color=None,
                             profile_color='C1'):
    '''
    Create figure to visualize Monte Carlo of soiling profiles used in the SRR
    analysis.

    .. warning::
        The soiling module is currently experimental. The API, results,
        and default behaviors may change in future releases (including MINOR
        and PATCH releases) as the code matures.

    Parameters
    ----------
    soiling_info : dict
        ``soiling_info`` returned by :py:meth:`.soiling.SRRAnalysis.run` or
        :py:func:`.soiling.soiling_srr`.
    normalized_yield : pd.Series
        PV yield data that is normalized, filtered and aggregated.
    point_alpha : float, default 0.5
        tranparency of the ``normalized_yield`` points
    profile_alpha : float, default 0.05
        transparency of each profile
    ymin : float, optional
        minimum y coordinate
    ymax : float, optional
        maximum y coordinate
    profiles : int, optional
        the number of stochastic profiles to plot.  If not specified, plot
        all profiles.
    point_color : str, optional
        color of the normalized_yield points
    profile_color : str, default 'C1'
        color of the stochastic profiles

    Returns
    -------
    fig : matplotlib Figure
    '''
    warnings.warn(
        'The soiling module is currently experimental. The API, results, '
        'and default behaviors may change in future releases (including MINOR '
        'and PATCH releases) as the code matures.'
    )

    fig, ax = plt.subplots()
    renormalized = normalized_yield / soiling_info['renormalizing_factor']
    ax.plot(renormalized.index, renormalized, 'o', alpha=point_alpha,
            color=point_color)
    ax.set_ylim(ymin, ymax)

    if profiles is not None:
        to_plot = soiling_info['stochastic_soiling_profiles'][:profiles]
    else:
        to_plot = soiling_info['stochastic_soiling_profiles']
    for profile in to_plot:
        ax.plot(profile.index, profile, color=profile_color,
                alpha=profile_alpha)
    ax.set_ylabel('Renormalized energy')
    fig.autofmt_xdate()

    return fig


def soiling_interval_plot(soiling_info, normalized_yield, point_alpha=0.5,
                          profile_alpha=1, ymin=None, ymax=None,
                          point_color=None, profile_color=None):
    '''
    Create figure to visualize valid soiling profiles used in the SRR analysis.

    .. warning::
        The soiling module is currently experimental. The API, results,
        and default behaviors may change in future releases (including MINOR
        and PATCH releases) as the code matures.

    Parameters
    ----------
    soiling_info : dict
        ``soiling_info`` returned by :py:meth:`.soiling.SRRAnalysis.run` or
        :py:func:`.soiling.soiling_srr`.
    normalized_yield : pd.Series
        PV yield data that is normalized, filtered and aggregated.
    point_alpha : float, default 0.5
        tranparency of the ``normalized_yield`` points
    profile_alpha : float, default 1
        transparency of soiling profile
    ymin : float, optional
        minimum y coordinate
    ymax : float, optional
        maximum y coordinate
    point_color : str, optional
        color of the ``normalized_yield`` points
    profile_color : str, optional
        color of the soiling intervals

    Returns
    -------
    fig : matplotlib Figure
    '''
    warnings.warn(
        'The soiling module is currently experimental. The API, results, '
        'and default behaviors may change in future releases (including MINOR '
        'and PATCH releases) as the code matures.'
    )

    sratio = soiling_info['soiling_ratio_perfect_clean']
    fig, ax = plt.subplots()
    renormalized = normalized_yield / soiling_info['renormalizing_factor']
    ax.plot(renormalized.index, renormalized, 'o')
    ax.plot(sratio.index, sratio, 'o')
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Renormalized energy')

    fig.autofmt_xdate()

    return fig


def soiling_rate_histogram(soiling_info, bins=None):
    '''
    Create histogram of soiling rates found in the SRR analysis.

    .. warning::
        The soiling module is currently experimental. The API, results,
        and default behaviors may change in future releases (including MINOR
        and PATCH releases) as the code matures.

    Parameters
    ----------
    soiling_info : dict
        ``soiling_info`` returned by :py:meth:`.soiling.SRRAnalysis.run` or
        :py:func:`.soiling.soiling_srr`.
    bins : int
        number of histogram bins to use

    Returns
    -------
    fig : matplotlib Figure
    '''
    warnings.warn(
        'The soiling module is currently experimental. The API, results, '
        'and default behaviors may change in future releases (including MINOR '
        'and PATCH releases) as the code matures.'
    )

    soiling_summary = soiling_info['soiling_interval_summary']
    fig, ax = plt.subplots()
    ax.hist(100.0 * soiling_summary.loc[soiling_summary['valid'], 'soiling_rate'],
            bins=bins)
    ax.set_xlabel('Soiling rate (%/day)')
    ax.set_ylabel('Count')

    return fig


def availability_summary_plots(power_system, power_subsystem, loss_total,
                               energy_cumulative, energy_expected_rescaled,
                               outage_info):
    """
    Create a figure summarizing the availability analysis results.

    Because all of the parameters to this function are products of an
    AvailabilityAnalysis object, it is usually easier to use
    :py:meth:`.availability.AvailabilityAnalysis.plot` instead of running
    this function manually.

    .. warning::
        The availability module is currently experimental. The API, results,
        and default behaviors may change in future releases (including MINOR
        and PATCH releases) as the code matures.

    Parameters
    ----------
    power_system : pd.Series
        Timeseries total system power.

    power_subsystem : pd.DataFrame
        Timeseries power data, one column per subsystem.

    loss_total : pd.Series
        Timeseries system lost power.

    energy_cumulative : pd.Series
        Timeseries system cumulative energy.

    energy_expected_rescaled : pd.Series
        Timeseries expected energy, rescaled to match actual energy. This
        reflects interval energy, not cumulative.

    outage_info : pd.DataFrame
        A dataframe with information about system outages.

    Returns
    -------
    fig : matplotlib Figure

    See Also
    --------
    rdtools.availability.AvailabilityAnalysis.plot

    Examples
    --------
    >>> aa = AvailabilityAnalysis(...)
    >>> aa.run()
    >>> fig = rdtools.plotting.availability_summary_plots(aa.power_system,
    ...     aa.power_subsystem, aa.loss_total, aa.energy_cumulative,
    ...     aa.energy_expected_rescaled, aa.outage_info)
    """
    warnings.warn(
        'The availability module is currently experimental. The API, results, '
        'and default behaviors may change in future releases (including MINOR '
        'and PATCH releases) as the code matures.'
    )

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[:, 1], sharex=ax1)

    # inverter power
    power_system.plot(ax=ax1)
    ax1.set_ylabel('Inverter Power [kW]')
    # meter power
    power_subsystem.plot(ax=ax2)
    ax2.set_ylabel('System power [kW]')
    # lost power
    loss_total.plot(ax=ax3)
    ax3.set_ylabel('Estimated lost power [kW]')

    # cumulative energy
    energy_cumulative.plot(ax=ax4, label='Reported Production')

    # we'll use the index value to only set legend entries for the first
    # outage we plot.  Just in case the index has some other values, we'll
    # reset it here:
    outage_info = outage_info.reset_index(drop=True)
    for i, row in outage_info.iterrows():
        # matplotlib ignores legend entries starting with underscore, so we
        # can use that to hide duplicate entries
        prefix = "_" if i > 0 else ""
        start, end = row[['start', 'end']]
        start_energy = row['energy_start']
        expected_energy = row['energy_expected']
        lo, hi = np.abs(expected_energy - row[['ci_lower', 'ci_upper']])
        expected_curve = energy_expected_rescaled[start:end].cumsum()
        expected_curve += start_energy
        expected_curve.plot(c='tab:orange', ax=ax4,
                            label=prefix + 'Expected Production')
        energy_end = expected_curve.iloc[-1]
        ax4.errorbar([end], [energy_end], [[lo], [hi]], c='k',
                     label=prefix + 'Uncertainty')
    ax4.legend()
    ax4.set_ylabel('Cumulative Energy [kWh]')
    return fig
