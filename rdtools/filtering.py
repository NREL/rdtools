'''Functions for filtering and subsetting PV system data.'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import joblib
import os
import xgboost as xgb

#Load in the XGBoost clipping model using joblib.
xgboost_clipping_model = joblib.load((os.path.dirname(__file__)) + "/models/xgboost_clipping_model.dat")


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


def logic_clip_filter(power_ac, 
                      mounting_type = 'Fixed', 
                      derivative_cutoff = 0.2):
    '''
    This filter is a logic-based filter that is used to filter out 
    clipping periods in AC power and AC energy time series. 
    The AC power or energy time series is filtered based on the 
    maximum derivative over a rolling window, as compared to a user-set 
    derivate_cutoff (default set to 0.2).  The size of the
    rolling window is increased when the system is a tracked system.
    
    Parameters
    ----------
    power_ac : pd.Series
        Pandas time series, representing PV system power or energy with 
        a pandas datetime index.
    mounting_type : String 
        String representing the mounting configuration associated with the 
        AC power/energy time series. Can either be "Fixed" or "Tracking".
        Default set to 'Fixed'.
    derivative_cutoff : Float
        Cutoff for max derivative threshold. Defaults to 0.2; however, values 
        as high as 0.4 have been tested and shown to be effective.  The higher 
        the cutoff, the more values in the dataset that will be determined as 
        clipping.
        
    Returns
    -------
    pd.Series
        Boolean Series of whether the given measurement is deterimined as clipping.
        True values delineate clipping periods, and False values delineate non-
        clipping periods.
    '''  
    #Check that it's a Pandas series with a datetime index. If not, raise an error.
    if not isinstance(power_ac.index, pd.DatetimeIndex):
        raise TypeError('Must be a Pandas series with a datetime index.')
    #Check the other input variables to ensure that they are the correct format
    if (mounting_type != "Tracking") & (mounting_type != "Fixed"):
        raise ValueError("Variable mounting_type must be string 'Tracking' or 'Fixed'.")
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
    time_series_sampling_frequency = power_ac.index.to_series().diff().astype('timedelta64[m]').mode()[0]
    #Make copies of the original inputs for the cases that the data is changes for clipping evaluation
    original_time_series_sampling_frequency = time_series_sampling_frequency
    power_copy = power_ac.copy()
    #duplicate indexes are dropped and the a frequency is given to the index
    #testing shows that pandas.rolling(periods) performs consistently only when the datetime index has a set frequency
    power_ac = power_ac.reset_index().drop_duplicates(subset = power_ac.index.name, 
                                                      keep = 'first').set_index(power_ac.index.name)
    freq_string = str(time_series_sampling_frequency) + 'T'
    if time_series_sampling_frequency >=10:    
        power_ac = power_ac.asfreq(freq_string)
    #High frequency data (less than 10 minutes) has demonstrated potential to have more noise than low
    #frequency  data. Therefore, the  data is resampled to a 15-minute average before running 
    #the filter.
    if time_series_sampling_frequency < 10:
        power_ac = power_ac.resample('15T').mean()
        time_series_sampling_frequency = 15
    #Tracked PV systems typically have much flatter output over the course of the central hours of the day,
    #as compared to fixed tilt systems. This function determines clipping by looking for a near-zero derivative
    #over 3 periods for a fixed tilt system, and over 5 periods for a tracked system with a sampling frequency
    #more frequent than once every 30 minutes.
    if (mounting_type == "Tracking") & (time_series_sampling_frequency < 30):
        roll_periods = 5
    else:
        roll_periods = 3
    #Replace the lower 10% of daily data with NaN's
    daily = 0.1 * power_ac.resample('D').max()
    power_ac['tenP_daily'] = daily.reindex(index = power_ac.index, method='ffill')
    power_ac.loc[power_ac[column_name] < power_ac['tenP_daily'], column_name]= np.nan   
    power_ac = power_ac[column_name]
    
    #Calculate the maximum value over a forward-rolling window
    max_roll = power_ac.iloc[::-1].rolling(roll_periods).max()
    max_roll = max_roll.reindex(power_ac.index)
    #calculated the minimum value over a forward-rolling window
    min_roll = power_ac.iloc[::-1].rolling(roll_periods).min()
    min_roll = min_roll.reindex(power_ac.index)
    #Calculate the maximum derivative within the foward-rolling window
    deriv_max = (max_roll - min_roll) / ((max_roll + min_roll) / 2) * 100
    #Determine clipping values based on the maximum derivative in the rolling window,
    #and the user-specified derivative threshold
    roll_clip_mask = (deriv_max < derivative_cutoff)
    #the following applies the clipping determination to all data points within
    #the rolling window
    #Get max derivative at a certain timestamp, looks at the periods     
    clipping = roll_clip_mask.copy()
    if roll_periods > 1:
        i=0
        for c in roll_clip_mask:
            if c==True: 
                clipping[i+1] = True
                clipping[i+2] = True
                if roll_periods > 3:
                    clipping[i + 3] = True
                if roll_periods > 4:    
                    clipping[i + 4] = True
            i=i+1
    # high frequency was resampled to 15 minute average data
    #the following lines apply the 15 minute clipping filter to the
    #original 15 minute data resulting in a clipping filter on the orginal
    #data   
    if original_time_series_sampling_frequency < 10:
        power_ac = power_copy.copy()
        clipping = clipping.reindex(index=power_ac.index,
                                    method='ffill')
        #Subset the series where clipping filter == True
        clip_pwr = power_ac[clipping]
        clip_pwr = clip_pwr.reindex(index = power_ac.index,
                                    fill_value = np.nan)
        #Set any values within the clipping max + clipping min threshold as clipping. This is done 
        #specifically for capturing the noise for high frequency data sets. 
        daily_mean = clip_pwr.resample('D').mean()
        daily_std = clip_pwr.resample('D').std()
        daily_clipping_max = daily_mean + 2 * daily_std
        daily_clipping_max = daily_clipping_max.reindex(index=power_ac.index,method='ffill')
        daily_clipping_min = daily_mean - 2 * daily_std
        daily_clipping_min = daily_clipping_min.reindex(index=power_ac.index,method='ffill')
    else:  
        #find the maximum and minimum power level where clipping is found each day
        clip_pwr = power_ac[clipping]
        clip_pwr = clip_pwr.reindex(index=power_ac.index,fill_value=np.nan)
        daily_clipping_max = clip_pwr.resample('D').max()
        daily_clipping_min = clip_pwr.resample('D').min()
        daily_clipping_min = daily_clipping_min.reindex(index=power_ac.index,method='ffill')
        daily_clipping_max = daily_clipping_max.reindex(index=power_ac.index,method='ffill')
    #set all values to clipping that are between the maximum and minimum power levels
    #where clipping was found on a daily basis
    #note that this bins data as clipping that is not necessary clipped
    #for example, for one day, daily data on a 50 kW is found to be clipped at 49kW
    # and 30 kW.  This is atypical for PV systems but such a finding brings concern
    #to what is  happening on the said.   It was thought to be conservative to label
    #all data between 30 and 49kW as clipping so as to not use this data in further analysis
    final_clip=(daily_clipping_min <= power_ac) & (power_ac <= daily_clipping_max)
    final_clip=final_clip.reindex(index = power_copy.index,
                                  fill_value = False)
    return final_clip


def xgboost_clip_filter(power_ac, 
                        mounting_type = 'Fixed'):
    """
    This function generates the features to run through the XGBoost
    clipping model, and generates model outputs.

    Parameters
    ----------
    power_ac : pd.Series
        Pandas time series, representing PV system power or energy with 
        a pandas datetime index.
    mounting_type : String 
        String representing the mounting configuration associated with the 
        AC power/energy time series. Can either be "Fixed" or "Tracking".
        Default set to 'Fixed'.

    Returns
    -------
    pd.Series
        Boolean Series of whether the given measurement is deterimined as clipping.
        True values delineate clipping periods, and False values delineate non-
        clipping periods.
    """
    power_ac_df = power_ac.to_frame()
    power_ac_df['mounting_config'] = mounting_type
    #Get the sampling frequency (as a continuous feature variable)
    power_ac_df['sampling_frequency'] = power_ac_df.index.to_series().diff().astype('timedelta64[m]').mode()[0]
    #Min-max normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    power_ac_df['scaled_value'] = min_max_scaler.fit_transform(power_ac_df[['value']])    
    #Get the rolling derivative
    sampling_frequency = power_ac_df['sampling_frequency'].iloc[0] 
    if sampling_frequency < 10:
        rolling_window = 5
    elif (sampling_frequency >= 10) and (sampling_frequency < 60):
        rolling_window = 3#max(int(round(60/sampling_frequency, 0)), 1)
    else:
        rolling_window = 2
    power_ac_df['rolling_average'] = power_ac_df['scaled_value'].rolling(window = rolling_window,
                                                                         center = True).mean()
    #First-order derivative
    power_ac_df['first_order_derivative_backward'] = power_ac_df.scaled_value.diff()
    power_ac_df['first_order_derivative_forward'] = power_ac_df.scaled_value.shift(-1).diff()    
    #First order derivative for the rolling average
    power_ac_df['first_order_derivative_backward_rolling_avg'] = power_ac_df.rolling_average.diff()
    power_ac_df['first_order_derivative_forward_rolling_avg'] = power_ac_df.rolling_average.shift(-1).diff()    
    
    #Rolling max derivative over time
    max_roll = power_ac_df['scaled_value'].iloc[::-1].rolling(rolling_window).max()
    max_roll = max_roll.reindex(power_ac_df.index)
    #calculated the minimum value over a set forward rolling window
    min_roll = power_ac_df['scaled_value'].iloc[::-1].rolling(rolling_window).min()
    min_roll = min_roll.reindex(power_ac_df.index)
    #calculate the maximum derivative within the set foward rolling window
    power_ac_df['deriv_max'] = (max_roll - min_roll) / ((max_roll + min_roll) / 2) * 100
    #Get the max value for the day and see how each value compares
    power_ac_df['date'] = list(pd.to_datetime(pd.Series(power_ac_df.index)).dt.date)
    power_ac_df['daily_max'] = power_ac_df.groupby(['date'])['scaled_value'].transform(max)
    #Get percentage of daily max
    power_ac_df['percent_daily_max'] = power_ac_df['scaled_value'] / power_ac_df['daily_max']
    #Convert tracking/fixed tilt to a boolean variable
    power_ac_df.loc[power_ac_df['mounting_config'] == 'Tracking', 'mounting_config_bool'] = 1
    power_ac_df.loc[power_ac_df['mounting_config'] == 'Fixed', 'mounting_config_bool'] = 0
    #Subset the dataframe to only include model inputs
    power_ac_df = power_ac_df[['first_order_derivative_backward',
                               'first_order_derivative_forward',
                               'first_order_derivative_backward_rolling_avg',
                               'first_order_derivative_forward_rolling_avg',
                               'sampling_frequency',
                               'mounting_config_bool','scaled_value',
                               'rolling_average','daily_max', 
                               'percent_daily_max', 'deriv_max']]
    #Run the power_ac_df dataframe through the XGBoost ML model,
    #and return boolean outputs
    xgb_predictions = pd.Series(xgboost_clipping_model.predict(power_ac_df).astype(bool))
    #Add datetime as an index
    xgb_predictions.index = power_ac_df.index
    return xgb_predictions

