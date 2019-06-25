import matplotlib.pyplot as plt


def degradation_summary_plots(yoy_rd, yoy_ci, yoy_info, normalized_yield,
                              hist_xmin=None, hist_xmax=None, scatter_ymin=None,
                              scatter_ymax=None, plot_color=None, summary_title=None,
                              scatter_alpha=0.5):
    '''
    Description
    -----------
    Function to create plots (scatter plot and histogram) that summarize degradation analysis results

    Parameters
    ----------
    yoy_rd: rate of relative performance change in %/yr (float)
    yoy_ci: one-sigma confidence interval of degradation rate estimate (float)
    yoy_info: dict
        ('YoY_values') pandas series of right-labeled year on year slopes
        ('renormalizing_factor') float of value used to recenter data
        ('exceedance_level') the degradation rate that was outperformed with
        a probability given by the exceedance_prob parameter in
        the degradation_year_on_year function of the degradation module
    normalized_yield: Pandas Time Series (numeric)
         cotaining PV yield data that is normalized, filtered and aggregated.
    hist_xmin: lower limit of x-axis for the histogram (numeric)
    hist_xmax: upper limit of x-axis for the histogram (numeric)
    scatter_ymin: lower limit of y-axis for the scatter plot (numeric)
    scatter_ymax: upper limit of y-axis for the scatter plot (numeric)
    plot_color: color of the summary plots
    summary_title: overall title for summary plots (string)
    scatter_alpha: Transparency of the scatter plot (numeric)
    It should be noted that the yoy_rd, yoy_ci and yoy_info are the outputs from
    the degradation_year_on_year function of the degradation module

    Returns
    -------
    Figure with two axes
    '''

    yoy_values = yoy_info['YoY_values']

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
    Creates figure to visualize Monte Carlo of soiling profiles used in the SRR analysis.

    Parameters
    ----------
    soiling_info: soiling_info returned by srr_analysis.run() or soiling_srr() (dict)
    normalized_yield: Pandas Time Series (numeric) cotaining PV yield data that is normalized,
                      filtered and aggregated.
    point_alpha: tranparency of the normalized_yield points (numeric)
    profile_alpha: transparency of each profile (numeric)
    ymin: minimum y coordinate (numeric)
    ymax: maximum y coordinate (numeric)
    profiles: the number of stochasitc profiles to plot (int)
    point_color: color of the normalized_yield points
    profile_color: color of the stochastic profiles

    Returns
    -------
    Figure
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
    Creates figure to visualize valid soiling profiles used in the SRR analysis.

    Parameters
    ----------
    soiling_info: soiling_info returned by srr_analysis.run() or soiling_srr() (dict)
    normalized_yield: Pandas Time Series (numeric) cotaining PV yield data that is normalized,
                      filtered and aggregated.
    point_alpha: tranparency of the normalized_yield points (numeric)
    profile_alpha: transparency of soiling profile (numeric)
    ymin: minimum y coordinate (numeric)
    ymax: maximum y coordinate (numeric)
    point_color: color of the normalized_yield points
    profile_color: color of the soiling intervals

    Returns
    -------
    Figure
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
    Creates histogram of soiling rates found in the SRR analysis.

    Parameters
    ----------
    soiling_info: soiling_info returned by srr_analysis.run() or soiling_srr() (dict)
    bins: number of histogram bins to use (int)

    Returns
    -------
    Figure
    '''

    soiling_summary = soiling_info['soiling_interval_summary']
    fig, ax = plt.subplots()
    ax.hist(100.0 * soiling_summary.loc[soiling_summary['valid'], 'slope'], bins=bins)
    ax.set_xlabel('Soiling rate (%/day)')
    ax.set_ylabel('Count')

    return fig
