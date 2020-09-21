'''Functions for filtering and subsetting PV system data.'''

import numpy as np
import pandas as pd
import plotly.express as px
#Import plotly for viewing in the browser
import plotly.io as pio
pio.renderers.default = "browser"

def normalized_filter(energy_normalized, energy_normalized_low=0.01,
                      energy_normalized_high=None):
    '''
    Select normalized yield between ``low_cutoff`` and ``high_cutoff``

    Parameters
    ----------
    energy_normalized : pd.Series
        Normalized energy measurements.
    energy_normalized_low : float, default 0.01
        The lower bound of acceptable values.
    energy_normalized_high : float, optional
        The upper bound of acceptable values.

    Returns
    -------
    pd.Series
        Boolean Series of whether the given measurement is within acceptable
        bounds.
    '''

    if energy_normalized_low is None:
        energy_normalized_low = -np.inf
    if energy_normalized_high is None:
        energy_normalized_high = np.inf

    return ((energy_normalized > energy_normalized_low) &
            (energy_normalized < energy_normalized_high))


def poa_filter(poa_global, poa_global_low=200, poa_global_high=1200):
    '''
    Filter POA irradiance readings outside acceptable measurement bounds.

    Parameters
    ----------
    poa_global : pd.Series
        POA irradiance measurements.
    poa_global_low : float, default 200
        The lower bound of acceptable values.
    poa_global_high : float, default 1200
        The upper bound of acceptable values.

    Returns
    -------
    pd.Series
        Boolean Series of whether the given measurement is within acceptable
        bounds.
    '''
    return (poa_global > poa_global_low) & (poa_global < poa_global_high)


def tcell_filter(temperature_cell, temperature_cell_low=-50,
                 temperature_cell_high=110):
    '''
    Filter temperature readings outside acceptable measurement bounds.

    Parameters
    ----------
    temperature_cell : pd.Series
        Cell temperature measurements.
    temperature_cell_low : float, default -50
        The lower bound of acceptable values.
    temperature_cell_high : float, default 110
        The upper bound of acceptable values.

    Returns
    -------
    pd.Series
        Boolean Series of whether the given measurement is within acceptable
        bounds.
    '''
    return ((temperature_cell > temperature_cell_low) &
            (temperature_cell < temperature_cell_high))


def geometric_clip_filter(power_ac, clipping_percentile_cutoff = 0.8,
                          daily_max_percentile_cutoff = 0.9,
                          first_order_derivative_threshold = None):
    """
    Mask a time series, delineating clipping and non-clipping periods.
    Returns the time series with clipping periods omitted, as well as a boolean mask, 
    with 'True' delineating clipping, and 'False' delineating regular periods.
    
    Parameters
    ----------
    power_ac : pd.Series
        AC power in Watts. Index of the Pandas series is a Pandas datetime index.
    clipping_percentile_cutoff: float, default 0.8
        Cutoff value for the percentile (for the whole time series) where clipping takes place. 
        So, for example, if the cutoff is set to 0.8, then any value in the normalized time series less than 
        0.8 will not be considered clipping. The higher the threshold, the more data omitted.
    daily_max_percentile_cutoff: float, default 0.9
        Cutoff value for the the daily percentile where clipping takes place. So, for example,
        if the cutoff is set to 0.9, then any value in a normalized daily time series that is 
        less than 90% the max daily value will not be considered clipping. The higher the threshold,
        the more data omitted.
    first_order_derivative_threshold : float, default None,
        Cutoff value for the derivative threshold. The higher the value, the less stringent 
        the function is on defining clipping periods. Represents the cutoff for the first-order
        derivative across two data points. Default is set to None, where the threshold is derived 
        based on an experimental equation, which varies threshold by sampling frequency.
        
    
    Returns
    -------
    pd.Series: Filtered ac_power time series, with clipping periods excluded.
    pd. Series: Boolean mask time series for clipping, with True indicating a clipping period
        and False representing a non-clipping period
    """
    #Check that it's a Pandas series with a datetime index. If not, raise an error.
    if not isinstance(power_ac.index, pd.DatetimeIndex):
        raise TypeError('Must be a Pandas series with a datetime index.')
    #Get the names of the series and the datetime index
    column_name = power_ac.name
    if column_name is None:
        column_name = 'power_ac'
        power_ac = power_ac.rename(column_name)   
    index_name = power_ac.index.name
    if index_name is None:
        index_name = 'datetime'
        power_ac = power_ac.rename_axis(index_name)   
    #Get the sampling frequency of the time series
    time_series_sampling_frequency = power_ac.index.to_series(keep_tz=True).diff().astype('timedelta64[m]').mode()[0]
    #Based on the sampling frequency, adjust the first order derivative threshold. This is a 
    #default equation that is experimentally derived from PV Fleets data. Value can also be
    #manually set by the user.
    if first_order_derivative_threshold == None:
        first_order_derivative_threshold = (0.00005 * time_series_sampling_frequency) + 0.0009
    #Generate a dataframe for the series
    dataframe = pd.DataFrame(power_ac)
    #Set all negative values in the data stream to 0
    dataframe.loc[dataframe[power_ac.name]<0, column_name] = 0
    #Remove anything more than 3 standard deviations from the mean (outliers)
    #OR use IQR calculation to remove outliers (Q1 - 5*IQR) or (Q3 - 5*IQR)
    mean = np.mean(dataframe[column_name], axis=0)
    std = np.std(dataframe[column_name], axis=0)
    Q1 = np.quantile(dataframe[dataframe[column_name]>0][column_name], 0.25)
    Q3 = np.quantile(dataframe[dataframe[column_name]>0][column_name], 0.75)
    IQR = Q3 - Q1
    #Outlier cleaning statement--set outliers to 0
    dataframe.loc[(abs(mean - dataframe[column_name]) > (3*std)) &
                  (dataframe[column_name] <= 0) & 
                  (dataframe[column_name] >= (Q3 + 5*IQR))] = 0
    #Min-max normalize the time series
    scaled_column = 'scaled_' + column_name
    dataframe[scaled_column] = (dataframe[column_name] - dataframe[column_name].min()) / (dataframe[column_name].max() - dataframe[column_name].min())
    #Get the first-order derivative of the time series over one time shift
    dataframe['first_order_derivative_backward'] = dataframe[scaled_column] - dataframe[scaled_column].shift(1)
    dataframe['first_order_derivative_forward'] = dataframe[scaled_column] - dataframe[scaled_column].shift(-1)
    #Get the date of the reading
    dataframe['date'] = pd.to_datetime(dataframe.index).date
    #Daily max calculations
    dataframe['daily_max_' + scaled_column] = dataframe.groupby('date')[scaled_column].transform("max")   
    dataframe['daily_max_percentile_' + scaled_column] = dataframe[scaled_column] / dataframe['daily_max_' + scaled_column]   
    ##############################################
    #Logic for clipping:
    #-Normalized value must be greater than or equal to clipping_percentile_cutoff value
    #-First-order derivative (either backward- or forward-calculated) less than 
    #   first_order_derivative_threshold value
    #-Value within X percentile of daily max value
    ##############################################
    dataframe['daily_max_percentile_threshold'] =  (dataframe['daily_max_percentile_' + scaled_column] >= daily_max_percentile_cutoff)
    dataframe['top_percentile_threshold_value'] = (dataframe[scaled_column] >= clipping_percentile_cutoff)
    #First order derivative--compare to hourly average
    dataframe['low_val_threshold_first_order_derivative_forward'] =  abs(dataframe['first_order_derivative_backward']) <= first_order_derivative_threshold 
    dataframe['low_val_threshold_first_order_derivative_backward'] =  abs(dataframe['first_order_derivative_forward']) <= first_order_derivative_threshold 
    #Set default mask to False
    dataframe[scaled_column +"_clipping_mask"] = False
    #Boolean statement for detecting clipping sequences
    dataframe.loc[((dataframe['daily_max_percentile_threshold'] == True) & 
                   (dataframe['top_percentile_threshold_value'] == True) & 
                   ((dataframe['low_val_threshold_first_order_derivative_forward'] == True) | 
                    (dataframe['low_val_threshold_first_order_derivative_backward'] == True))), scaled_column +"_clipping_mask"] = True
    #Count the subsequent mask categories in the sequence
    dataframe['subgroup'] = (dataframe[scaled_column +'_clipping_mask'] != dataframe[scaled_column +'_clipping_mask'].shift(1)).cumsum()
    #Count the subgroup column
    dataframe['subgroup_count'] = dataframe.groupby("subgroup")["subgroup"].transform('count')
    #Remove any clipping sequences that are less than two subsequent readings in length
    dataframe.loc[(dataframe[scaled_column +'_clipping_mask'] == True) & (dataframe['subgroup_count'] <= 2), scaled_column +"_clipping_mask"] = False 
    #Find daily threshold for clipping, and set anything within +-0.01 as a clipped value
    clipping_cutoff_df = dataframe.groupby(['date', scaled_column +'_clipping_mask'])[scaled_column].min().reset_index()
    clipping_cutoff_df = clipping_cutoff_df[clipping_cutoff_df[scaled_column +'_clipping_mask'] == True].rename(columns={scaled_column: 'clipping_cutoff'})
    #Merge back with the main dataframe
    dataframe = pd.merge(dataframe.reset_index(), clipping_cutoff_df[['date', 'clipping_cutoff']], on = ['date'], how = 'left')
    dataframe.set_index(index_name, inplace = True)
    dataframe.loc[(dataframe['clipping_cutoff'] - dataframe[scaled_column]) <= 0.01, scaled_column +'_clipping_mask'] = True
    return pd.Series(dataframe[dataframe[scaled_column +'_clipping_mask'] == False][column_name]), pd.Series(dataframe[scaled_column +'_clipping_mask'])


def tune_geometric_clip_filter_plot(power_ac, clipping_percentile_cutoff = 0.8,
                                    daily_max_percentile_cutoff = 0.9,
                                    first_order_derivative_threshold = None,
                                    display_web_browser = True):
    """
    This function allows the user to visualize a clipping filter in a matplotlib plot, after tweaking 
    the function's different hyperparameters. The plot can be zoomed in on, for an in-depth look at
    clipping in the AC power time series.
    
    Parameters
    ----------
    power_ac : pd.Series
        AC power in Watts. Index of the Pandas series is a Pandas datetime index.
    clipping_percentile_cutoff: float, default 0.8
        Cutoff value for the percentile (for the whole time series) where clipping takes place. 
        So, for example, if the cutoff is set to 0.8, then any value in the normalized time series less than 
        0.8 will not be considered clipping. The higher the threshold, the more data omitted.
    daily_max_percentile_cutoff: float, default 0.9
        Cutoff value for the the daily percentile where clipping takes place. So, for example,
        if the cutoff is set to 0.9, then any value in a normalized daily time series that is 
        less than 90% the max daily value will not be considered clipping. The higher the threshold,
        the more data omitted.
    first_order_derivative_threshold : float, default None,
        Cutoff value for the derivative threshold. The higher the value, the less stringent 
        the function is on defining clipping periods. Represents the cutoff for the first-order
        derivative across two data points. Default is set to None, where the threshold is derived 
        based on an experimental equation, which varies threshold by sampling frequency.        
    
    Returns 
    ---------
    Interactive Plotly graph, with the masked time series for clipping. Returned via web browser.
    """
    #First run the time series through the geometric_clip_filter mask.
    filtered_power_ac, clipping_mask = geometric_clip_filter(power_ac, clipping_percentile_cutoff,
                                                             daily_max_percentile_cutoff,
                                                             first_order_derivative_threshold)
    #Get the names of the series and the datetime index
    column_name = power_ac.name
    if column_name is None:
        column_name = 'power_ac'
        power_ac = power_ac.rename(column_name)   
    index_name = power_ac.index.name
    if index_name is None:
        index_name = 'datetime'
        power_ac = power_ac.rename_axis(index_name)   
    #Visualize the power_ac time series, delineating clipping periods using the clipping_mask series.
    #Use plotly to visualize.
    df = pd.DataFrame(power_ac)
    #Add the clipping mask as a column
    df['clipping_mask'] = clipping_mask
    df = df.reset_index()
    fig = px.scatter(df, x = index_name, y = column_name, color= 'clipping_mask')
    #If display_web_browser is set to True, the time series with clipping is rendered via 
    #the web browser.
    if display_web_browser == True:
        pio.renderers.default = "browser"
        fig.show()
    return fig


def clip_filter(power_ac, quantile=0.98):
    '''
    Filter data points likely to be affected by clipping
    with power greater than or equal to 99% of the `quant`
    quantile.

    Parameters
    ----------
    power_ac : pd.Series
        AC power in Watts
    quantile : float, default 0.98
        Value for upper threshold quantile

    Returns
    -------
    pd.Series
        Boolean Series of whether the given measurement is below 99% of the
        quantile filter.
    '''
    v = power_ac.quantile(quantile)
    return (power_ac < v * 0.99)


def csi_filter(poa_global_measured, poa_global_clearsky, threshold=0.15):
    '''
    Filtering based on clear-sky index (csi)

    Parameters
    ----------
    poa_global_measured : pd.Series
        Plane of array irradiance based on measurments
    poa_global_clearsky : pd.Series
        Plane of array irradiance based on a clear sky model
    threshold : float, default 0.15
        threshold for filter

    Returns
    -------
    pd.Series
        Boolean Series of whether the clear-sky index is within the threshold
        around 1.
    '''

    csi = poa_global_measured / poa_global_clearsky
    return (csi >= 1.0 - threshold) & (csi <= 1.0 + threshold)
