'''Functions for filtering and subsetting PV system data.'''

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from numbers import Number

# Load in the XGBoost clipping model using joblib.
xgboost_clipping_model = None
model_path = (os.path.dirname(__file__)) + \
                             "/models/xgboost_clipping_model.dat"


def _load_xgboost_clipping_model():
    global xgboost_clipping_model
    if xgboost_clipping_model is None:
        xgboost_clipping_model = joblib.load(model_path)
    return xgboost_clipping_model


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


def clip_filter(power_ac, model="quantile", **kwargs):
    """
    Master wrapper for running one of the desired clipping filters.
    The default filter run is the quantile clipping filter.
    
    Parameters
    ----------
    power_ac : pd.Series
        Pandas time series, representing PV system with
        a pandas datetime index.
    model : string, default 'quantile_clip_filter'
        Clipping filter model to run. Can be 'quantile',
        'xgboost', or 'logic'.
    kwargs :
        Additional clipping filter args, specific to the model being
        used. Keyword must be passed with value.

    Returns
    -------
    pd.Series
        Boolean Series of whether to include the point because it is not
        clipping.
        True values delineate non-clipping periods, and False values delineate
        clipping periods.
    """
    if isinstance(model, Number):
        quantile = model
        warnings.warn("Function clip_filter is now a wrapper for different "
                      "clipping filters. To reproduce prior behavior, "
                      "parameters have been interpreted as model= "
                      "'quantile_clip_filter', quantile={quantile}. "
                      "This syntax will be removed in a future version.",
                      DeprecationWarning)
        kwargs['quantile'] = quantile
        model = 'quantile'

    if (model == 'quantile'):
        clip_mask = quantile_clip_filter(power_ac, **kwargs)
    elif model == 'xgboost':
        clip_mask = xgboost_clip_filter(power_ac, **kwargs)
    elif model == 'logic':
        clip_mask = logic_clip_filter(power_ac, **kwargs)
    else:
        raise ValueError(
            "Variable model must be 'quantile', "
            "'xgboost', or 'logic'.")
    return clip_mask


def quantile_clip_filter(power_ac, quantile=0.98):
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


def _format_clipping_time_series(power_ac, mounting_type):
    """
    Format an AC power time series appropriately for
    either the logic_clip_filter function or the xgboost_clip_filter
    function.
    
    Parameters
    ----------
    power_ac : pd.Series
        Pandas time series, representing PV system power with
        a pandas datetime index.
    mounting_type : String
        String representing the mounting configuration associated with the
        AC power time series. Can either be "fixed" or "single_axis_tracking".
        Default set to 'fixed'.
        
    Returns
    -------
    pd.Series
        AC power time series
    String
        AC Power time series name
    String
        Datetime Index name
    """
    # Check that it's a Pandas series with a datetime index.
    # If not, raise an error.
    if not isinstance(power_ac.index, pd.DatetimeIndex):
        raise TypeError('Must be a Pandas series with a datetime index.')
    # Check the other input variables to ensure that they are the
    # correct format
    if (mounting_type != "single_axis_tracking") & (mounting_type != "fixed"):
        raise ValueError(
            "Variable mounting_type must be string 'single_axis_tracking' or 'fixed'.")
    # Check that there is enough data in the dataframe. Must be greater than
    # 10 readings.
    if len(power_ac) <= 10:
        raise Exception('<=10 readings in the time series, cannot run filter.')
    # Get the names of the series and the datetime index
    column_name = 'value'
    power_ac = power_ac.rename(column_name)
    index_name = power_ac.index.name
    if index_name is None:
        index_name = 'datetime'
        power_ac = power_ac.rename_axis(index_name)
    return power_ac, power_ac.index.name


def _calculate_max_rolling_range(power_ac, roll_periods):
    """
    This function calculates the maximum range over a rolling
    time period for an AC power time series. A pandas series of
    the rolling range is returned.
    
    Parameters
    ----------
    power_ac : pd.Series
        Pandas time series, representing PV system power with
        a pandas datetime index.
    roll_periods: Int
        Number of readings to calculate the rolling maximum range on.

    Returns
    -------
    pd.Series
        Time series of the rolling maximum range.
    """
    # Calculate the maximum value over a forward-rolling window
    max_roll = power_ac.iloc[::-1].rolling(roll_periods).max()
    max_roll = max_roll.reindex(power_ac.index)
    # Calculate the minimum value over a forward-rolling window
    min_roll = power_ac.iloc[::-1].rolling(roll_periods).min()
    min_roll = min_roll.reindex(power_ac.index)
    # Calculate the maximum rolling range within the foward-rolling window
    rolling_range_max = (max_roll - min_roll) / ((max_roll + min_roll) / 2) * 100
    return rolling_range_max


def logic_clip_filter(power_ac,
                      mounting_type='fixed',
                      rolling_range_max_cutoff=0.2,
                      roll_periods=None):
    '''
    This filter is a logic-based filter that is used to filter out
    clipping periods in AC power time series.
    The AC power time series is filtered based on the
    maximum range over a rolling window, as compared to a user-set
    rolling_range_max_cutoff (default set to 0.2).  The size of the
    rolling window is increased when the system is a tracked system.
    
    Parameters
    ----------
    power_ac : pd.Series
        Pandas time series, representing PV system power with
        a pandas datetime index.
    mounting_type: string, default 'fixed'
        String representing the mounting configuration associated with the
        AC power time series. Can either be "fixed" or "single_axis_tracking".
        Default set to 'fixed'.
    rolling_range_max_cutoff : float, default 0.2
        Cutoff for max rolling range threshold. Defaults to 0.2; however,
        values as high as 0.4 have been tested and shown to be effective.
        The higher the cutoff, the more values in the dataset that will be
        determined as clipping.
    roll_periods: Integer.
        Number of periods to examine when looking for a near-zero derivative
        in the time series derivative. If roll_periods = 3, the system looks
        for a near-zero derivative over 3 consecutive readings. Default value
        is set to None, so the function uses default logic: it looks for a
        near-zero derivative over 3 periods for a fixed tilt system, and over
        5 periods for a tracked system with a sampling frequency more frequent
        than once every 30 minutes.

    Returns
    -------
    pd.Series
        Boolean Series of whether to include the point because it is not
        clipping.
        True values delineate non-clipping periods, and False values delineate
        clipping periods.
    '''
    # Format the power time series
    power_ac, index_name = _format_clipping_time_series(power_ac,
                                                        mounting_type)
    # Get the sampling frequency of the time series
    time_series_sampling_frequency = power_ac.index.to_series().diff()\
        .astype('timedelta64[m]').mode()[0]
    # Make copies of the original inputs for the cases that the data is
    # changes for clipping evaluation
    original_time_series_sampling_frequency = time_series_sampling_frequency
    power_copy = power_ac.copy()
    # Drop duplciate indices
    power_ac = pd.DataFrame(power_ac[~power_ac.index.duplicated(keep='first')])
    freq_string = str(time_series_sampling_frequency) + 'T'
    if time_series_sampling_frequency >= 10:
        power_ac = power_ac.resample(freq_string).asfreq()
    # High frequency data (less than 10 minutes) has demonstrated
    # potential to have more noise than low frequency  data.
    # Therefore, the  data is resampled to a 15-minute average
    # before running the filter.
    if time_series_sampling_frequency < 10:
        power_ac = power_ac.resample('15T').mean()
        time_series_sampling_frequency = 15
    # If a value for roll_periods is not designated, the function uses
    # the current default logic to set the roll_periods value.
    if roll_periods is None:
        if (mounting_type == "single_axis_tracking") & \
         (time_series_sampling_frequency < 30):
            roll_periods = 5
        else:
            roll_periods = 3
    # Replace the lower 25% of daily data with NaN's
    daily = 0.25 * power_ac.resample('D').max()
    power_ac['ten_percent_daily'] = daily.reindex(index=power_ac.index,
                                                  method='ffill')
    power_ac.loc[power_ac['value'] < power_ac['ten_percent_daily'],
                 'value'] = np.nan
    power_ac = power_ac['value']
    # Calculate the maximum rolling range for the power time series.
    rolling_range_max = _calculate_max_rolling_range(power_ac, roll_periods)
    # Determine clipping values based on the maximum rolling range in
    # the rolling window, and the user-specified rolling range threshold
    roll_clip_mask = (rolling_range_max < rolling_range_max_cutoff)
    # Set values within roll_periods values from a True instance
    # as True as well
    clipping = (roll_clip_mask.rolling(roll_periods).sum() >= 1)
    # High frequency was resampled to 15-minute average data.
    # The following lines apply the 15-minute clipping filter to the
    # original 15-minute data resulting in a clipping filter on the original
    # data.
    if (original_time_series_sampling_frequency < 10):
        power_ac = power_copy.copy()
        clipping = clipping.reindex(index=power_copy.index,
                                    method='ffill')
        # Subset the series where clipping filter == True
        clip_pwr = power_ac[clipping]
        clip_pwr = clip_pwr.reindex(index=power_copy.index,
                                    fill_value=np.nan)
        # Set any values within the clipping max + clipping min threshold
        # as clipping. This is done specifically for capturing the noise
        # for high frequency data sets.
        daily_mean = clip_pwr.resample('D').mean()
        daily_std = clip_pwr.resample('D').std()
        daily_clipping_max = daily_mean + 2 * daily_std
        daily_clipping_max = daily_clipping_max.reindex(index=power_copy.index,
                                                        method='ffill')
        daily_clipping_min = daily_mean - 2 * daily_std
        daily_clipping_min = daily_clipping_min.reindex(index=power_copy.index,
                                                        method='ffill')
    else:
        # Find the maximum and minimum power level where clipping is
        # detected each day.
        clip_pwr = power_ac[clipping]
        clip_pwr = clip_pwr.reindex(index=power_copy.index,
                                    fill_value=np.nan)
        daily_clipping_max = clip_pwr.resample('D').max()
        daily_clipping_min = clip_pwr.resample('D').min()
        daily_clipping_min = daily_clipping_min.reindex(index=power_copy.index,
                                                        method='ffill')
        daily_clipping_max = daily_clipping_max.reindex(index=power_copy.index,
                                                        method='ffill')
    power_ac = power_ac.reindex(index=power_copy.index,
                                method='ffill')
    # Set all values to clipping that are between the maximum and minimum
    # power levels where clipping was found on a daily basis.
    final_clip = (daily_clipping_min <= power_ac) & \
        (power_ac <= daily_clipping_max)
    return ~final_clip


def xgboost_clip_filter(power_ac,
                        mounting_type='fixed'):
    """
    This function generates the features to run through the XGBoost
    clipping model, and generates model outputs.

    Parameters
    ----------
    power_ac : pd.Series
        Pandas time series, representing PV system power with
        a pandas datetime index.
    mounting_type: string, default 'fixed'
        String representing the mounting configuration associated with the
        AC power time series. Can either be "fixed" or "single_axis_tracking".

    Returns
    -------
    pd.Series
        Boolean Series of whether to include the point because it is not
        clipping.
        True values delineate non-clipping periods, and False values delineate
        clipping periods.
    """
    # Load in the XGBoost model
    xgboost_clipping_model = _load_xgboost_clipping_model()
    # Format the power time series
    power_ac, index_name = _format_clipping_time_series(power_ac,
                                                        mounting_type)
    # Convert the Pandas series to a dataframe, with mounting_type as an
    # additional column.
    power_ac_df = power_ac.to_frame()
    power_ac_df['mounting_config'] = mounting_type
    # Get the sampling frequency (as a continuous feature variable)
    power_ac_df['sampling_frequency'] = power_ac_df.index.to_series()\
        .diff().astype('timedelta64[m]').mode()[0]
    # Min-max normalize
    max_min_diff = (power_ac_df['value'].max() - power_ac_df['value'].min())
    power_ac_df['scaled_value'] = (power_ac_df['value'] -
                                   power_ac_df['value'].min()) / max_min_diff
    # Get the rolling derivative
    sampling_frequency = power_ac_df['sampling_frequency'].iloc[0]
    if sampling_frequency < 10:
        rolling_window = 5
    elif (sampling_frequency >= 10) and (sampling_frequency < 60):
        rolling_window = 3
    else:
        rolling_window = 2
    power_ac_df['rolling_average'] = power_ac_df['scaled_value']\
        .rolling(window=rolling_window, center=True).mean()
    # First-order derivative
    power_ac_df['first_order_derivative_backward'] = power_ac_df\
        .scaled_value.diff()
    power_ac_df['first_order_derivative_forward'] = power_ac_df\
        .scaled_value.shift(-1).diff()
    # First order derivative for the rolling average
    power_ac_df['first_order_derivative_backward_rolling_avg'] = \
        power_ac_df.rolling_average.diff()
    power_ac_df['first_order_derivative_forward_rolling_avg'] = \
        power_ac_df.rolling_average.shift(-1).diff()
    # Calculate the maximum rolling range for the power time series.
    power_ac_df['deriv_max'] = _calculate_max_rolling_range(
        power_ac=power_ac_df['scaled_value'], roll_periods=rolling_window)
    # Get the max value for the day and see how each value compares
    power_ac_df['date'] = list(pd.to_datetime(pd.Series(
        power_ac_df.index)).dt.date)
    power_ac_df['daily_max'] = power_ac_df.groupby(
        ['date'])['scaled_value'].transform(max)
    # Get percentage of daily max
    power_ac_df['percent_daily_max'] = power_ac_df['scaled_value'] \
        / power_ac_df['daily_max']
    # Convert single-axis tracking/fixed tilt to a boolean variable
    power_ac_df.loc[power_ac_df['mounting_config'] == "single_axis_tracking",
                    'mounting_config_bool'] = 1
    power_ac_df.loc[power_ac_df['mounting_config'] == 'fixed',
                    'mounting_config_bool'] = 0
    # Subset the dataframe to only include model inputs
    power_ac_df = power_ac_df[['first_order_derivative_backward',
                               'first_order_derivative_forward',
                               'first_order_derivative_backward_rolling_avg',
                               'first_order_derivative_forward_rolling_avg',
                               'sampling_frequency',
                               'mounting_config_bool', 'scaled_value',
                               'rolling_average', 'daily_max',
                               'percent_daily_max', 'deriv_max']].dropna()
    # Run the power_ac_df dataframe through the XGBoost ML model,
    # and return boolean outputs
    xgb_predictions = pd.Series(xgboost_clipping_model.predict(
        power_ac_df).astype(bool))
    # Add datetime as an index
    xgb_predictions.index = power_ac_df.index
    xgb_predictions = xgb_predictions.reindex(power_ac.index,
                                              fill_value=False)
    return ~xgb_predictions
