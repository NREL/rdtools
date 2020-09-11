'''Functions for filtering and subsetting PV system data.'''

import numpy as np
import pandas as pd

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
                          first_order_derivative_threshold = 0.0045):
    """
    Mask a time series, delineating clipping and non-clipping periods.
    Returns the time series with clipping periods omitted, as well as a boolean mask, 
    with 'True' delineating clipping, and 'False' delineating regular periods.
    
    Parameters
    ----------
    power_ac : pd.Series
        AC power in Watts. Index of the Pandas series is a Pandas datetime index.
    clipping_percentile_cutoff: float, default 0.8
        Cutoff value for the percentile for where clipping takes place. So, for example,
        if the cutoff is set to 0.8, then any value in the normalized time series less than 
        0.8 will not be considered clipping.
    derivative_threshold : float, default 0.0045
        Cutoff value for the derivative threshold. The higher the value, the less stringent 
        the function is on defining clipping periods. Represents the cutoff for the first-order
        derivative across two data points.
    
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
    #Generate a dataframe for the series
    dataframe = pd.DataFrame(power_ac)
    #Set all negative values in the data stream to 0
    dataframe[dataframe[power_ac.name]<0][column_name] = 0
    #Remove anything more than 3 standard deviations from the mean (outliers)
    #OR use IQR calculation to remove outliers (Q1 - 5*IQR) or (Q3 - 5*IQR)
    mean = np.mean(dataframe[column_name], axis=0)
    std = np.std(dataframe[column_name], axis=0)
    Q1 = np.quantile(dataframe[column_name], 0.25)
    Q3 = np.quantile(dataframe[column_name], 0.75)
    IQR = Q3 - Q1
    #Outlier removal statement
    dataframe = dataframe[(abs(mean - dataframe[column_name]) < (3*std)) &
                          (dataframe[column_name] > (Q1 - 5*IQR)) & 
                          (dataframe[column_name] < (Q3 + 5*IQR))]
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
    dataframe['daily_max_difference_' + scaled_column] = dataframe['daily_max_' + scaled_column] - dataframe[scaled_column]  
    #7 day rolling max calculations
    dataframe['seven_day_rolling_max_daily_value_' + scaled_column] = dataframe.groupby('date')[scaled_column].rolling(min_periods = 1, center = True,  window = 7).max().reset_index(0, drop=True)    
    dataframe['seven_day_max_difference_' + scaled_column] = dataframe['seven_day_rolling_max_daily_value_' + scaled_column] - dataframe[scaled_column]  
    ##############################################
    #Logic for clipping:
    #-Normalized value must be greater than or equal to clipping_percentile_cutoff value
    #-First-order derivative (either backward- or forward-calculated) less than 
    #   first_order_derivative_threshold value
    #-Value within 0.02 normalized units of max daily value
    #-Value within 0.02 normalized units of max 7-day value
    ##############################################
    dataframe['daily_max_difference_threshold'] =  (dataframe['daily_max_difference_' + scaled_column] <= 0.02)
    dataframe['seven_day_max_difference_threshold'] =  (dataframe['seven_day_max_difference_' + scaled_column] <= 0.02)
    dataframe['top_percentile_threshold_value'] = (dataframe[scaled_column] >= clipping_percentile_cutoff)
    #First order derivative--compare to hourly average
    dataframe['low_val_threshold_first_order_derivative_forward'] =  abs(dataframe['first_order_derivative_backward']) <= first_order_derivative_threshold 
    dataframe['low_val_threshold_first_order_derivative_backward'] =  abs(dataframe['first_order_derivative_forward']) <= first_order_derivative_threshold 
    #Set default mask to False
    dataframe[scaled_column +"_clipping_mask"] = False
    #Boolean statement for detecting clipping sequences
    dataframe.loc[((dataframe['daily_max_difference_threshold'] == True) & 
                   (dataframe['seven_day_max_difference_threshold'] == True) & 
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
