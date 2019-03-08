''' Plotting Module
This module contains plotting functions for common diagonostic plots.
'''

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 12,
                           'lines.markeredgewidth': 0,
                           'lines.markersize': 2
                           })

def xyplot(x, y, set_xlimits=False, set_ylimits=False,\
           xmin=None, xmax=None, ymin=None, ymax=None,\
           xlabel='', ylabel='', plot_title='',\
           autoformat_xdate=True, fmt='o', alpha=0.5, figsize=(4.5,3)):
    '''
    Description
    -----------
    Simple (x,y) scatter plot for Pandas Time Series

    Parameters
    ----------
    x: Pandas Time Series (numeric)
    y: Pandas Time Series (numeric)
    set_xlimits: whether to set limits on the x-axis (boolean) 
    set_ylimits: whether to set limits on the y-axis (boolean) 
    xmin: lower limit of x-axis (numeric)
    xmax: upper limit of x-axis (numeric)
    ymin: lower limit of y-axis (numeric)
    ymax: upper limit of y-axis (numeric)
    xlabel: label for x-axis (string)
    ylabel: label for y-axis (string)
    plot_title: title for plot (string)
    autoformat_xdate: whether to automaticaly format x-axis datetime labels (boolean)
    fmt: basic formatting like color, marker and linestyle (string)
    alpha: transparency, 0.0 transparent through 1.0 opaque (numeric)
    figsize: width, height in inches (tuple(numeric))


    Returns
    -------
    (fig, ax):  figure and axes handles from subplots
    Displays a figure
    '''

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, fmt, alpha = alpha)
    
    if (set_xlimits):
        if (xmin is None):
            xmin = min(x)
        
        if (xmax is None):
            xmax = max(x)
        ax.set_xlim(xmin,xmax)
        
    if (set_ylimits):   
        if (ymin is None):
            ymin = min(y)
        
        if (ymax is None):
            ymax = max(y)        

        ax.set_ylim(ymin,ymax)
    
    if (autoformat_xdate):
        fig.autofmt_xdate()
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.title(plot_title)
    
    plt.show()
    return fig,ax
    
def xy2plot(x, y1, y2,\
            set_xlimits=False, set_ylimits=False,\
            xmin=None, xmax=None,\
            ymin=None, ymax=None,\
            xlabel='', ylabel='', autoformat_xdate=True,\
            fmt1='o', alpha1=0.5, fmt2='o', alpha2=0.5,\
            figsize=(4.5,3), plot_title ='', with_legend=True,\
            legend_loc='upper center',legend_bbox_to_anchor=None,\
            legend_shadow = False, legend_ncol=1,\
            label1=None, label2=None):
    '''
    Description
    -----------
    Simple (x,y1,y2) scatter plot for Pandas Time Series

    Parameters
    ----------
    x: Pandas Time Series (numeric)
    y1: Pandas Time Series (numeric)
    y2: Pandas Time Series (numeric)
    set_xlimits: whether to set limits on the x-axis (boolean) 
    set_ylimits: whether to set limits on the y-axis (boolean) 
    xmin: lower limit of x-axis (numeric)
    xmax: upper limit of x-axis (numeric)
    ymin: lower limit of y-axis (numeric)
    ymax: upper limit of y-axis (numeric)
    xlabel: label for x-axis (string)
    ylabel: label for y-axis (string)
    autoformat_xdate: whether to automaticaly format x-axis datetime labels (boolean)
    fmt1: basic formatting like color, marker and linestyle for y1 (string)
    alpha1: transparency, 0.0 transparent through 1.0 opaque for y1 (numeric)
    fmt2: basic formatting like color, marker and linestyle for y2 (string)
    alpha2: transparency, 0.0 transparent through 1.0 opaque for y2 (numeric)
    figsize: width, height in inches (tuple(numeric))
    plot_title: title for plot (string)
    with_legend: whether to create legend (boolean)
    legend_loc: legend location (string)
    legend_bbox_to_anchor: parameters for manual legend placement (tuple(numeric))
    legend_shadow: whether to include legend shadow (boolean)
    legend_ncol: number of columns in legend (integer)
    label1: label for y1 (string)
    label2: label for y2 (string)


    Returns
    -------
    Displays a figure
    '''

    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, y1, fmt1, alpha = alpha1, label = label1)
    ax.plot(x, y2, fmt2, alpha = alpha2, label = label2)
    
    if (set_xlimits):
        if (xmin is None):
            xmin = min(x)
        
        if (xmax is None):
            xmax = max(x)
            
        ax.set_xlim(xmin,xmax)
        
    if (set_ylimits):   
        if (ymin is None):
            ymin = min(min(y1), min(y2))
        
        if (ymax is None):
            ymax = max(max(y1), max(y2))    

        ax.set_ylim(ymin,ymax)

    if (autoformat_xdate):
        fig.autofmt_xdate()
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)    
    
    plt.title(plot_title)
    
    if (with_legend):
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor,\
                  shadow=legend_shadow, ncol=legend_ncol) 

    plt.show()
    return fig,ax

def degradation_summary_plots(x, y, yoy_info, yoy_rd, yoy_ci, daily,\
                              set_hist_xlimits=False, set_scatter_ylimits=False,\
                              hist_xmin=-30, hist_xmax=45, scatter_ymin=0.5, scatter_ymax=1.2,\
                              plot_color='blue', summary_title=None):
    '''
    Description
    -----------
    Function to create plots (scatter plot and histogram) that summarize degradation analysis results

    Parameters
    ----------
    x: list of inital and final indices of the aggregated data (timestamp)
    y: list of initial and final relative performances (numeric)
    yoy_info: dict
            ('YoY_values') pandas series of right-labeled year on year slopes
            ('renormalizing_factor') float of value used to recenter data
            ('exceedance_level') the degradation rate that was outperformed with
            a probability given by the exceedance_prob parameter in
            the degradation_year_on_year function of the degradation module
    yoy_rd: rate of relative performance change in %/yr (float)
    yoy_ci: one-sigma confidence interval of degradation rate estimate (float)
    daily: Pandas Time Series (numeric) 
         cotaining data that is normalized, filtered and aggregated on the daily scale.
    set_hist_xlimits: whether to set limits on the x-axis for the histogram (boolean) 
    set_scatter_ylimits: whether to set limits on the y-axis for the scatter plot (boolean) 
    hist_xmin: lower limit of x-axis for the histogram (numeric)
    hist_xmax: upper limit of x-axis for the histogram (numeric)
    scatter_ymin: lower limit of y-axis for the scatter plot (numeric)
    scatter_ymax: upper limit of y-axis for the scatter plot (numeric)
    plot_color: color of the summary plots
    summary_title: overall title for summary plots (string)

    It should be noted that the yoy_rd, yoy_ci and yoy_info are the outputs from 
    the degradation_year_on_year function of the degradation module


    Returns
    -------
    Displays two figures summarizing degradation analysis results
    '''

    yoy_values = yoy_info['YoY_values']
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 3))
    ax2.hist(yoy_values, label='YOY', bins=len(yoy_values)/40, color = plot_color)
    ax2.axvline(x=yoy_rd, color='black', linestyle='dashed', linewidth=3)
    
    if (set_hist_xlimits):
        ax2.set_xlim(hist_xmin,hist_xmax)
    
    ax2.annotate( u' $R_{d}$ = %.2f%%/yr \n confidence interval: \n %.2f to %.2f %%/yr' 
             %(yoy_rd, yoy_ci[0], yoy_ci[1]),  xy=(0.5, 0.7), xycoords='axes fraction',
            bbox=dict(facecolor='white', edgecolor=None, alpha = 0))
    ax2.set_xlabel('Annual degradation (%)');

    ax1.plot(daily.index, daily/yoy_info['renormalizing_factor'], 'o', color = plot_color, alpha = 0.5)
    ax1.plot(x, y, 'k--', linewidth=3)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Renormalized Energy')
    
    if (set_scatter_ylimits):
        ax1.set_ylim(scatter_ymin, scatter_ymax)
    
    fig.autofmt_xdate()

    fig.suptitle(summary_title);
    plt.show()
    
    #return fig, (ax1, ax2)  # possibly return figure and axes handles for further plot tweaks?
