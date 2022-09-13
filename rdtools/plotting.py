'''Functions for plotting degradation and soiling analysis results.'''

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np
import warnings


def degradation_summary_plots(yoy_rd, yoy_ci, yoy_info, normalized_yield,
                              hist_xmin=None, hist_xmax=None, bins=None,
                              scatter_ymin=None, scatter_ymax=None,
                              plot_color=None, summary_title=None,
                              scatter_alpha=0.5, detailed=False):
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

    normalized_yield : pandas.Series
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
    detailed : bool, optional
        Include extra information in the returned figure:

        * Color code points by the number of times they get used in calculating
          Rd slopes.  Default color: 2 times (as a start and endpoint). Green:
          1 time. Red: 0 times.
        * The number of year-on-year slopes contributing to the histogram.

    Note
    ----
    It should be noted that the yoy_rd, yoy_ci and yoy_info are the outputs
    from :py:func:`.degradation.degradation_year_on_year`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with two axes
    '''

    yoy_values = yoy_info['YoY_values']

    if bins is None:
        bins = len(yoy_values) // 40

    bins = int(min(bins, len(yoy_values)))

    if plot_color is None:
        plot_color = 'C0'

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
    if detailed:
        n = yoy_values.notnull().sum()
        label += '\n' + f'n = {n}'

    ax2.annotate(label, xy=(0.5, 0.6), xycoords='axes fraction',
                 bbox=dict(facecolor='white', edgecolor=None, alpha=0))
    ax2.set_xlabel('Annual degradation (%)')

    renormalized_yield = normalized_yield / yoy_info['renormalizing_factor']
    if detailed:
        colors = yoy_info['usage_of_points'].map({0: 'red', 1: 'green', 2: plot_color})
    else:
        colors = plot_color
    ax1.scatter(renormalized_yield.index, renormalized_yield,
                c=colors, alpha=scatter_alpha)

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
    normalized_yield : pandas.Series
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
    fig : matplotlib.figure.Figure
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
    normalized_yield : pandas.Series
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
    fig : matplotlib.figure.Figure
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
    fig : matplotlib.figure.Figure
    '''
    warnings.warn(
        'The soiling module is currently experimental. The API, results, '
        'and default behaviors may change in future releases (including MINOR '
        'and PATCH releases) as the code matures.'
    )

    soiling_summary = soiling_info['soiling_interval_summary']
    fig, ax = plt.subplots()
    ax.hist(100.0 * soiling_summary.loc[soiling_summary['valid'],
                                        'soiling_rate'], bins=bins)
    ax.set_xlabel('Soiling rate (%/day)')
    ax.set_ylabel('Count')

    return fig


def tune_filter_plot(signal, mask, display_web_browser=False):
    """
    This function allows the user to visualize filtered data in
    a Plotly plot, after tweaking the function's different
    parameters. The plot of signal colored according to mask
    can be zoomed in on, for an in-depth look.

    Parameters
    ----------
    signal : pandas.Series
        Index of the Pandas series is a Pandas datetime index. Usually
        this is PV power or energy, but other signals will work.
    mask : pandas.Series
        Pandas series of booleans, where included data periods
        are marked as True, and omitted-data periods occurs are
        marked as False. Should have the same detetime index as signal.
    display_web_browser : boolean, default False
        When set to True, the Plotly graph is displayed in the
        user's web browser.

    Returns
    ---------
    Interactive Plotly graph, with the masked time series for the filter.
    """
    # Get the names of the series and the datetime index
    column_name = signal.name
    if column_name is None:
        column_name = 'signal'
        signal = signal.rename(column_name)
    index_name = signal.index.name
    if index_name is None:
        index_name = 'datetime'
        signal = signal.rename_axis(index_name)
    # Visualize the power_ac time series, delineating clipping periods
    # using the clipping_mask series. Use plotly to visualize.
    df = pd.DataFrame(signal)
    # Add the mask as a column
    df['mask'] = mask
    df = df.reset_index()
    fig = px.scatter(df, x=index_name, y=column_name, color='mask',
                     color_discrete_map={
                         True: "blue",
                         False: "goldenrod"},
                     )
    # If display_web_browser is set to True, the time series with clipping
    # is rendered via the web browser.
    if display_web_browser is True:
        fig.show(renderer="browser")
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
    power_system : pandas.Series
        Timeseries total system power.

    power_subsystem : pandas.DataFrame
        Timeseries power data, one column per subsystem.

    loss_total : pandas.Series
        Timeseries system lost power.

    energy_cumulative : pandas.Series
        Timeseries system cumulative energy.

    energy_expected_rescaled : pandas.Series
        Timeseries expected energy, rescaled to match actual energy. This
        reflects interval energy, not cumulative.

    outage_info : pandas.DataFrame
        A dataframe with information about system outages.

    Returns
    -------
    fig : matplotlib.figure.Figure

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

    # meter power
    ax1.plot(power_system.index, power_system.values)
    ax1.set_ylabel('System power [kW]')
    # inverter power
    ax2.plot(power_subsystem.index, power_subsystem.values)
    ax2.set_ylabel('Inverter power [kW]')
    # lost power
    ax3.plot(loss_total.index, loss_total.values)
    ax3.set_ylabel('Estimated lost power [kW]')

    # cumulative energy
    ax4.plot(energy_cumulative.index, energy_cumulative.values,
             label='Reported Production')

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
        ax4.plot(expected_curve.index, expected_curve.values, c='tab:orange',
                 label=prefix + 'Expected Production')
        energy_end = expected_curve.iloc[-1]
        ax4.errorbar([end], [energy_end], [[lo], [hi]], c='k',
                     label=prefix + 'Uncertainty')
    ax4.legend()
    ax4.set_ylabel('Cumulative Energy [kWh]')
    return fig


def degradation_timeseries_plot(yoy_info, rolling_days=365, include_ci=True,
                                fig=None, plot_color=None, ci_color=None, **kwargs):
    '''
    Plot resampled time series of degradation trend with time

    Parameters
    ----------
    yoy_info : dict
        a dictionary with keys:

        * YoY_values - pandas series of right-labeled year on year slopes
    rolling_days: int, default 365
        Number of days for rolling window. Note that the window must contain
        at least 50% of datapoints to be included in rolling plot.
    include_ci : bool, default True
        calculate and plot 2-sigma confidence intervals along with rolling median
    fig     : matplotlib, optional
        fig object to add new plot to (first set of axes only)
    plot_color : str, optional
        color of the timeseries trendline
    ci_color : str, optional
        color of the confidence interval 'fuzz'
    kwargs :
        Extra parameters passed to matplotlib.pyplot.axis.plot()

    Note
    ----
    It should be noted that ``yoy_info`` is an output
    from :py:func:`rdtools.degradation.degradation_year_on_year`.

    Returns
    -------
    matplotlib.figure.Figure
    '''

    def _bootstrap(x, percentile, reps):
        # stolen from degradation_year_on_year
        n1 = len(x)
        xb1 = np.random.choice(x, (n1, reps), replace=True)
        mb1 = np.nanmedian(xb1, axis=0)
        return np.percentile(mb1, percentile)

    try:
        results_values = yoy_info['YoY_values']

    except KeyError:
        raise KeyError("yoy_info input dictionary does not contain key `YoY_values`.")

    if plot_color is None:
        plot_color = 'tab:orange'
    if ci_color is None:
        ci_color = 'C0'

    roller = results_values.rolling(f'{rolling_days}d', min_periods=rolling_days//2)
    # unfortunately it seems that you can't return multiple values in the rolling.apply() kernel.
    # TODO: figure out some workaround to return both percentiles in a single pass
    if include_ci:
        ci_lower = roller.apply(_bootstrap, kwargs={'percentile': 2.5, 'reps': 100}, raw=True)
        ci_upper = roller.apply(_bootstrap, kwargs={'percentile': 97.5, 'reps': 100}, raw=True)
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]
    if include_ci:
        ax.fill_between(ci_lower.index, ci_lower, ci_upper, color=ci_color)
    ax.plot(roller.median(), color=plot_color, **kwargs)
    ax.axhline(results_values.median(), c='k', ls='--')
    plt.ylabel('Degradation trend (%/yr)')
    fig.autofmt_xdate()

    return fig
