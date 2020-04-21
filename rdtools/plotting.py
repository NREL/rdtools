'''Functions for plotting degradation and soiling analysis results.'''

import matplotlib.pyplot as plt


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

    soiling_summary = soiling_info['soiling_interval_summary']
    fig, ax = plt.subplots()
    ax.hist(100.0 * soiling_summary.loc[soiling_summary['valid'], 'slope'],
            bins=bins)
    ax.set_xlabel('Soiling rate (%/day)')
    ax.set_ylabel('Count')

    return fig
