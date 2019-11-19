import matplotlib.pyplot as plt


def degradation_summary_plots(yoy_rd, yoy_ci, yoy_info, normalized_yield,
                              hist_xmin=None, hist_xmax=None, scatter_ymin=None,
                              scatter_ymax=None, plot_color=None, summary_title=None,
                              scatter_alpha=0.5):
    '''
    Description
    -----------
    Return a figure containing plots (scatter plot and histogram) that summarize
    degradation analysis results

    Parameters
    ----------
    yoy_rd : numeric
        Rate of relative performance change in %/yr
    yoy_ci : numeric
        Confidence interval of degradation rate estimate
    yoy_info : dict
        Information from year on year degradation rate calculation. Usually
        obtained from degradation_year_on_year(). Items must include
        'YoY_values' : pd.Series of right-labeled year on year slopes
        'renormalizing_factor' : numeric of value used to recenter data
    normalized_yield : pandas.Series
        Time series of PV yield data that is normalized, filtered and aggregated.
    hist_xmin : numeric
        Lower limit of x-axis for the histogram
    hist_xmax : numeric
        Upper limit of x-axis for the histogram
    scatter_ymin : numeric
        Lower limit of y-axis for the scatter plot
    scatter_ymax : numeric
        Upper limit of y-axis for the scatter plot
    plot_color : Matplotlib color specification
        Color to use for plots
    summary_title : str
        Overall title for summary plots
    scatter_alpha : numeric
        Transparency of the scatter plot, must be between 0 and 1

    Returns
    -------
    matplotlib.Figure.figure

    Notes
    -----
    The yoy_rd, yoy_ci and yoy_info are the outputs from
    the degradation_year_on_year() function of the degradation module

    '''

    # Calculate the degradation line
    start = normalized_yield.index[0]
    end = normalized_yield.index[-1]
    years = (end - start).days / 365.25
    yoy_values = yoy_info['YoY_values']

    x = [start, end]
    y = [1, 1 + (yoy_rd * years) / 100.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    ax2.hist(yoy_values, label='YOY', bins=len(
        yoy_values) // 40, color=plot_color)
    ax2.axvline(x=yoy_rd, color='black', linestyle='dashed', linewidth=3)

    ax2.set_xlim(hist_xmin, hist_xmax)

    ax2.annotate(u' $R_{d}$ = %.2f%%/yr \n confidence interval: \n %.2f to %.2f %%/yr'
                 % (yoy_rd, yoy_ci[0], yoy_ci[1]), xy=(0.5, 0.7), xycoords='axes fraction',
                 bbox=dict(facecolor='white', edgecolor=None, alpha=0))
    ax2.set_xlabel('Annual degradation (%)')

    ax1.plot(normalized_yield.index, normalized_yield /
             yoy_info['renormalizing_factor'], 'o', color=plot_color, alpha=scatter_alpha)
    ax1.plot(x, y, 'k--', linewidth=3)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Renormalized energy')

    ax1.set_ylim(scatter_ymin, scatter_ymax)

    fig.autofmt_xdate()

    if summary_title is not None:
        fig.suptitle(summary_title)
    plt.show()

    return fig


def soiling_monte_carlo_plot(soiling_info, normalized_yield, point_alpha=0.5, profile_alpha=0.05,
                             ymin=None, ymax=None, profiles=None, point_color=None, profile_color='C1'):
    '''
    Description
    -----------
    Return a figure to visualize Monte Carlo of soiling profiles used in the SRR analysis.

    Parameters
    ----------
    soiling_info : dict
        soiling_info returned by srr_analysis.run() or soiling_srr()
    normalized_yield : pandas.Series
        Time series of PV yield data that is normalized, filtered and aggregated.
    point_alpha : numeric
        Tranparency of the normalized_yield points, must be between 0 and 1
    profile_alpha : numeric
        Transparency of each profile, must be between 0 and 1
    ymin : numeric
        Minimum y coordinate
    ymax : numeric
        Maximum y coordinate
    profiles : int
        Number of stochasitc profiles to plot
    point_color : Matplotlib color specification
        Color of the normalized_yield points
    profile_color : Matplotlib color specification
        Color of the stochastic profiles

    Returns
    -------
    matplotlib.Figure.figure
    '''

    fig, ax = plt.subplots()
    ax.plot(normalized_yield.index, normalized_yield / soiling_info['renormalizing_factor'], 'o',
            alpha=point_alpha, color=point_color)
    ax.set_ylim(ymin, ymax)

    if profiles is not None:
        to_plot = soiling_info['stochastic_soiling_profiles'][:profiles]
    else:
        to_plot = soiling_info['stochastic_soiling_profiles']
    for profile in to_plot:
        ax.plot(profile.index, profile, color=profile_color, alpha=profile_alpha)
    ax.set_ylabel('Renormalized energy')
    fig.autofmt_xdate()

    return fig


def soiling_interval_plot(soiling_info, normalized_yield, point_alpha=0.5, profile_alpha=1,
                          ymin=None, ymax=None, point_color=None, profile_color=None):
    '''
    Description
    -----------
    Return figure to visualize valid soiling profiles used in the SRR analysis.

    Parameters
    ----------
    soiling_info : dict
        soiling_info returned by srr_analysis.run() or soiling_srr()
    normalized_yield : pandas.Series
        Time series of PV yield data that is normalized, filtered and aggregated.
    point_alpha : numeric
        Tranparency of the normalized_yield points, must be between 0 and 1
    profile_alpha : numeric
        Transparency of each profile, must be between 0 and 1
    ymin : numeric
        Minimum y coordinate
    ymax : numeric
        Maximum y coordinate
    profiles : int
        Number of stochasitc profiles to plot
    point_color : Matplotlib color specification
        Color of the normalized_yield points
    profile_color : Matplotlib color specification
        Color of the stochastic profiles

    Returns
    -------
    matplotlib.figure.Figure
    '''

    sratio = soiling_info['soiling_ratio_perfect_clean']
    fig, ax = plt.subplots()
    ax.plot(normalized_yield.index, normalized_yield / soiling_info['renormalizing_factor'], 'o')
    ax.plot(sratio.index, sratio, 'o')
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Renormalized energy')

    fig.autofmt_xdate()

    return fig


def soiling_rate_histogram(soiling_info, bins=None):
    '''
    Description
    -----------
    Return a figure containing a histogram of soiling rates found in the SRR analysis.

    Parameters
    ----------
    soiling_info : dict
        soiling_info returned by srr_analysis.run() or soiling_srr()
    bins : int
        number of histogram bins to use

    Returns
    -------
    matplotlib.figure.Figure
    '''

    soiling_summary = soiling_info['soiling_interval_summary']
    fig, ax = plt.subplots()
    ax.hist(100.0 * soiling_summary.loc[soiling_summary['valid'], 'slope'], bins=bins)
    ax.set_xlabel('Soiling rate (%/day)')
    ax.set_ylabel('Count')

    return fig
