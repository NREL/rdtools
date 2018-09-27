''' Plotting Module

This module contains plotting functions.
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

def degradation_summary_plots(x, y, yoy_info, yoy_rd, yoy_ci, daily,\
                              set_hist_xlimits=False, set_scatter_ylimits=False,\
                              hist_xmin=-30, hist_xmax=45, scatter_ymin=0.5, scatter_ymax=1.2,\
                              plot_color='blue', summary_title=None):

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
