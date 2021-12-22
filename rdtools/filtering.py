'''Functions for filtering and subsetting PV system data.'''

import numpy as np
import pandas as pd
import os
import warnings
from numbers import Number
import rdtools
import xgboost as xgb

# Load in the XGBoost clipping model using joblib.
xgboost_clipping_model = None
model_path = os.path.join(os.path.dirname(__file__),
                          "models", "xgboost_clipping_model.json")


def _load_xgboost_clipping_model():
    global xgboost_clipping_model
    if xgboost_clipping_model is None:
        xgboost_clipping_model = xgb.XGBClassifier()
        xgboost_clipping_model.load_model(model_path)
    return xgboost_clipping_model


def normalized_filter(energy_normalized, energy_normalized_low=0.01,
                      energy_normalized_high=None):
    '''
    Select normalized yield between ``low_cutoff`` and ``high_cutoff``

    Parameters
    ----------
    energy_normalized : pandas.Series
        Normalized energy measurements.
    energy_normalized_low : float, default 0.01
        The lower bound of acceptable values.
    energy_normalized_high : float, optional
        The upper bound of acceptable values.

    Returns
    -------
    pandas.Series
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
    poa_global : pandas.Series
        POA irradiance measurements.
    poa_global_low : float, default 200
        The lower bound of acceptable values.
    poa_global_high : float, default 1200
        The upper bound of acceptable values.

    Returns
    -------
    pandas.Series
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
    temperature_cell : pandas.Series
        Cell temperature measurements.
    temperature_cell_low : float, default -50
        The lower bound of acceptable values.
    temperature_cell_high : float, default 110
        The upper bound of acceptable values.

    Returns
    -------
    pandas.Series
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
    poa_global_measured : pandas.Series
        Plane of array irradiance based on measurments
    poa_global_clearsky : pandas.Series
        Plane of array irradiance based on a clear sky model
    threshold : float, default 0.15
        threshold for filter

    Returns
    -------
    pandas.Series
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
    power_ac : pandas.Series
        Pandas time series, representing PV system power or energy.
        For best performance, timestamps should be in local time.
    model : str, default 'quantile'
        Clipping filter model to run. Can be 'quantile',
        'xgboost', or 'logic'. Note: using the xgboost model can
        result in errors on some systems. These can often be alleviated
        by using conda to install xgboost, see
        https://anaconda.org/conda-forge/xgboost.
    kwargs :
        Additional clipping filter args, specific to the model being
        used. Keyword must be passed with value.

    Returns
    -------
    pandas.Series
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
                      f"'quantile_clip_filter', quantile={quantile}. "
                      "This syntax will be removed in a future version.",
                      rdtools._deprecation.rdtoolsDeprecationWarning)
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
    with power or energy greater than or equal to 99% of the `quant`
    quantile.

    Parameters
    ----------
    power_ac : pandas.Series
        AC power or AC energy time series
    quantile : float, default 0.98
        Value for upper threshold quantile

    Returns
    -------
    pandas.Series
        Boolean Series of whether the given measurement is below 99% of the
        quantile filter.
    '''
    v = power_ac.quantile(quantile)
    return (power_ac < v * 0.99)


def _format_clipping_time_series(power_ac, mounting_type):
    """
    Format an AC power or AC energy time series appropriately for
    either the logic_clip_filter function or the xgboost_clip_filter
    function.

    Parameters
    ----------
    power_ac : pandas.Series
        Pandas time series, representing PV system power or energy.
        For best performance, timestamps should be in local time.
    mounting_type : str
        String representing the mounting configuration associated with the
        AC power or energy time series. Can either be "fixed" or
        "single_axis_tracking".

    Returns
    -------
    pandas.Series
        AC power or AC energy time series
    str
        AC Power or AC energy time series name
    """
    # Check that it's a Pandas series with a datetime index.
    # If not, raise an error.
    if not isinstance(power_ac.index, pd.DatetimeIndex):
        raise TypeError('Must be a Pandas series with a datetime index.')
    # Check if the time series is tz-aware. If not, throw a
    # warning.
    has_timezone = pd.Series(power_ac.index).apply(lambda t: t.tzinfo is not None)
    # Throw a warning that we're expecting time zone-localized data,
    # if no time zone is specified.
    if not has_timezone.all():
        warnings.warn("Function expects timestamps in local time. "
                      "For best results pass a time-zone-localized "
                      "time series localized to the correct local time zone.")
    # Check the other input variables to ensure that they are the
    # correct format
    if (mounting_type != "single_axis_tracking") & (mounting_type != "fixed"):
        raise ValueError(
            "Variable mounting_type must be string 'single_axis_tracking' or "
            "'fixed'.")
    # Check if the datetime index is out of order. If it is, throw an
    # error.
    if not all(power_ac.sort_index().index == power_ac.index):
        raise IndexError(
            "Time series index has not been sorted. Implement the "
            "sort_index() method to the time series to rerun this function.")
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


def _check_data_sampling_frequency(power_ac):
    """
    Check the data sampling frequency of the time series. If the sampling
    frequency is not >=95% consistent, the time series is flagged with a
    warning.

    Parameters
    ----------
    power_ac : pandas.Series
        Pandas time series, representing PV system power or energy.
        For best performance, timestamps should be in local time.

    Returns
    -------
    None
    """
    # Get the sampling frequency counts--if the sampling frequency is not
    # consistently >=95% the same, then throw a warning.
    sampling_frequency_df = pd.DataFrame(power_ac.index.to_series()
                                         .diff().astype('timedelta64[s]')
                                         .value_counts())/len(power_ac)
    sampling_frequency_df.columns = ["count"]
    if (sampling_frequency_df["count"] < .95).all():
        warnings.warn("Variable sampling frequency across time series. "
                      "Less than 95% of the time series is sampled at the "
                      "same interval. This function was not tested "
                      "on variable frequency data--use at your own risk!")
    return


def _calculate_max_rolling_range(power_ac, roll_periods):
    """
    This function calculates the maximum range over a rolling
    time period for an AC power or energy time series. A pandas series of
    the rolling range is returned.

    Parameters
    ----------
    power_ac : pandas.Series
        Pandas time series, representing PV system power or energy.
        For best performance, timestamps should be in local time.
    roll_periods: int
        Number of readings to calculate the rolling maximum range on.

    Returns
    -------
    pandas.Series
        Time series of the rolling maximum range.
    """
    # Calculate the maximum value over a forward-rolling window
    max_roll = power_ac.iloc[::-1].rolling(roll_periods).max()
    max_roll = max_roll.reindex(power_ac.index)
    # Calculate the minimum value over a forward-rolling window
    min_roll = power_ac.iloc[::-1].rolling(roll_periods).min()
    min_roll = min_roll.reindex(power_ac.index)
    # Calculate the maximum rolling range within the foward-rolling window
    rolling_range_max = (max_roll - min_roll)/((max_roll + min_roll)/2)*100
    return rolling_range_max


def _apply_overall_clipping_threshold(power_ac,
                                      clipping_mask,
                                      clipped_power_ac):
    """
    Apply an overall clipping threshold to the data. This
    additional logic sets an overall threshold in the dataset
    where all points above this threshold are labeled as clipping
    periods.

    Parameters
    ----------
    power_ac : pandas.Series
        Pandas time series, representing PV system power or energy.
        For best performance, timestamps should be in local time.
    clipping_mask : pandas.Series
        Boolean mask of the AC power or energy time series, where clipping
        periods are labeled as True and non-clipping periods are
        labeled as False. Has a datetime index.
    clipped_power_ac: pandas.Series
        Pandas time series, representing PV system power or energy filtered
        where only clipping periods occur. Has a pandas datetime index.

    Returns
    -------
    clipping_mask : pandas.Series
        Boolean mask of clipping/non-clipping periods, after applying
        the overall clipping threshold to the mask. Clipping
        periods are labeled as True and non-clipping periods are
        labeled as False. Has a pandas datetime index.
    """
    upper_bound_pdiff = abs((power_ac.quantile(.99) -
                             clipped_power_ac.quantile(.99))
                            / ((power_ac.quantile(.99) +
                                clipped_power_ac.quantile(.99))/2))
    percent_clipped = len(clipped_power_ac)/len(power_ac)*100
    if (upper_bound_pdiff < 0.005) & (percent_clipped > 4):
        max_clip = (power_ac >= power_ac.quantile(0.99))
        clipping_mask = (clipping_mask | max_clip)
    return clipping_mask


def logic_clip_filter(power_ac,
                      mounting_type='fixed',
                      rolling_range_max_cutoff=0.2,
                      roll_periods=None):
    '''
    This filter is a logic-based filter that is used to filter out
    clipping periods in AC power or energy time series. It is based
    on the method presented in [1]. A boolean filter is returned
    based on the maximum range over a rolling window, as compared to
    a user-set rolling_range_max_cutoff (default set to 0.2). Periods
    where the relative maximum difference between any two points is
    less than rolling_range_max_cutoff are flagged as clipping and used
    to set daily clipping levels for the final mask.

    Parameters
    ----------
    power_ac : pandas.Series
        Pandas time series, representing PV system power or energy.
        For best performance, timestamps should be in local time.
    mounting_type: str, default 'fixed'
        String representing the mounting configuration associated with the
        AC power or energy time series. Can either be "fixed" or
        "single_axis_tracking". Default set to 'fixed'.
    rolling_range_max_cutoff : float, default 0.2
        Relative fractional cutoff for max rolling range threshold. When the
        relative maximum range in any interval is below this cutoff, the interval
        is determined to be clipping. Defaults to 0.2; however, values as high as
        0.4 have been tested and shown to be effective. The higher the cutoff, the
        more values in the dataset that will be determined as clipping.
    roll_periods: int, optional
        Number of periods to examine when looking for a near-zero derivative
        in the time series derivative. If roll_periods = 3, the system looks
        for a near-zero derivative over 3 consecutive readings. Default value
        is set to None, so the function uses default logic: it looks for a
        near-zero derivative over 3 periods for a fixed tilt system, and over
        5 periods for a tracked system with a sampling frequency more frequent
        than once every 30 minutes.

    Returns
    -------
    pandas.Series
        Boolean Series of whether to include the point because it is not
        clipping.
        True values delineate non-clipping periods, and False values delineate
        clipping periods.

    References
    ----------
    .. [1] Perry K., Muller, M., and Anderson K. "Performance comparison of clipping
       detection techniques in AC power time series", 2021 IEEE 48th Photovoltaic
       Specialists Conference (PVSC). DOI: 10.1109/PVSC43889.2021.9518733.
    '''
    # Throw a warning that this is still an experimental filter
    warnings.warn("The logic-based filter is an experimental clipping filter "
                  "that is still under development. The API, results, and "
                  "default behaviors may change in future releases (including "
                  "MINOR and PATCH). Use at your own risk!")
    # Format the time series
    power_ac, index_name = _format_clipping_time_series(power_ac,
                                                        mounting_type)
    # Test if the data sampling frequency is variable, and flag it if the time
    # series sampling frequency is less than 95% consistent.
    _check_data_sampling_frequency(power_ac)
    # Get the sampling frequency of the time series
    time_series_sampling_frequency = power_ac.index.to_series().diff()\
        .astype('timedelta64[m]').mode()[0]
    # Make copies of the original inputs for the cases that the data is
    # changes for clipping evaluation
    original_time_series_sampling_frequency = time_series_sampling_frequency
    power_ac_copy = power_ac.copy()
    # Drop duplicate indices
    power_ac = power_ac.reset_index().drop_duplicates(
        subset=power_ac.index.name,
        keep='first').set_index(power_ac.index.name)
    freq_string = str(time_series_sampling_frequency) + 'T'
    # Set days with the majority of frozen data to null.
    daily_std = power_ac.resample('D').std() / power_ac.resample('D').mean()
    power_ac['daily_std'] = daily_std.reindex(index=power_ac.index,
                                              method='ffill')
    power_ac.loc[power_ac['daily_std'] < 0.1,
                 'value'] = np.nan
    power_ac.drop('daily_std',
                  axis=1,
                  inplace=True)
    power_cleaned = power_ac['value'].copy()
    power_cleaned = power_cleaned.reindex(power_ac_copy.index,
                                          fill_value=np.nan)
    # High frequency data (less than 10 minutes) has demonstrated
    # potential to have more noise than low frequency  data.
    # Therefore, the  data is resampled to a 15-minute median
    # before running the filter.
    if time_series_sampling_frequency >= 10:
        power_ac = rdtools.normalization.interpolate(power_ac,
                                                     freq_string)
    else:
        power_ac = power_ac.resample('15T').median()
        time_series_sampling_frequency = 15
    # If a value for roll_periods is not designated, the function uses
    # the current default logic to set the roll_periods value.
    if roll_periods is None:
        if (mounting_type == "single_axis_tracking") & \
          (time_series_sampling_frequency < 30):
            roll_periods = 5
        else:
            roll_periods = 3
    # Replace the lower 10% of daily data with NaN's
    daily = 0.1 * power_ac.resample('D').max()
    power_ac['ten_percent_daily'] = daily.reindex(index=power_ac.index,
                                                  method='ffill')
    power_ac.loc[power_ac['value'] < power_ac['ten_percent_daily'],
                 'value'] = np.nan
    power_ac = power_ac['value']
    # Calculate the maximum rolling range for the time series.
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
        clipping = clipping.reindex(index=power_ac_copy.index,
                                    method='ffill')
        # Subset the series where clipping filter == True
        clip_pwr = power_ac_copy[clipping]
        clip_pwr = clip_pwr.reindex(index=power_ac_copy.index,
                                    fill_value=np.nan)
        # Set any values within the clipping max + clipping min threshold
        # as clipping. This is done specifically for capturing the noise
        # for high frequency data sets.
        daily_mean = clip_pwr.resample('D').mean()
        df_daily = daily_mean.to_frame(name='mean')
        df_daily['clipping_max'] = clip_pwr.groupby(pd.Grouper(freq='D')
                                                    ).quantile(0.99)
        df_daily['clipping_min'] = clip_pwr.groupby(pd.Grouper(freq='D')
                                                    ).quantile(0.075)
        daily_clipping_max = df_daily['clipping_max'].reindex(
            index=power_ac_copy.index, method='ffill')
        daily_clipping_min = df_daily['clipping_min'].reindex(
            index=power_ac_copy.index, method='ffill')
    else:
        # Find the maximum and minimum power_ac level where clipping is
        # detected each day.
        clipping = clipping.reindex(index=power_ac_copy.index,
                                    method='ffill')
        clip_pwr = power_ac_copy[clipping]
        clip_pwr = clip_pwr.reindex(index=power_ac_copy.index,
                                    fill_value=np.nan)
        daily_clipping_max = clip_pwr.resample('D').max()
        daily_clipping_min = clip_pwr.resample('D').min()
        daily_clipping_min = daily_clipping_min.reindex(
            index=power_ac_copy.index, method='ffill')
        daily_clipping_max = daily_clipping_max.reindex(
            index=power_ac_copy.index, method='ffill')
    # Set all values to clipping that are between the maximum and minimum
    # power_ac levels where clipping was found on a daily basis.
    clipping_difference = (daily_clipping_max -
                           daily_clipping_min)/daily_clipping_max
    final_clip = ((daily_clipping_min <= power_ac_copy) &
                  (power_ac_copy <= daily_clipping_max) &
                  (clipping_difference <= 0.025)) \
        | ((power_ac_copy <= daily_clipping_max*1.0025) &
           (power_ac_copy >= daily_clipping_max*0.9975) &
           (clipping_difference > 0.025))\
        | ((power_ac_copy <= daily_clipping_min*1.0025) &
           (power_ac_copy >= daily_clipping_min*0.9975) &
           (clipping_difference > 0.025))
    final_clip = final_clip.reindex(index=power_ac_copy.index,
                                    fill_value=False)
    # Check for an overall clipping threshold that should apply to all data
    clip_power_ac = power_ac_copy[final_clip]
    final_clip = _apply_overall_clipping_threshold(power_cleaned,
                                                   final_clip,
                                                   clip_power_ac)
    return ~final_clip


def _calculate_xgboost_model_features(df, sampling_frequency):
    """
    Calculate the features that will be fed into the XGBoost model.

    Parameters
    ----------
    df: pandas.DataFrame
        Pandas dataframe, containing the AC power or energy time series
        under the 'value' column.
    sampling_frequency: int
        Sampling frequency of the AC power or energy time series.

    Returns
    -------
    pandas.DataFrame
        Pandas dataframe, containing all of the features in the XGBoost
        model.
    """
    # Min-max normalize
    max_min_diff = (df['value'].max() - df['value'].min())
    df['scaled_value'] = (df['value'] - df['value'].min()) / max_min_diff
    if sampling_frequency < 10:
        rolling_window = 5
    elif (sampling_frequency >= 10) and (sampling_frequency < 60):
        rolling_window = 3
    else:
        rolling_window = 2
    df['rolling_average'] = df['scaled_value']\
        .rolling(window=rolling_window, center=True).mean()
    # First-order derivative
    df['first_order_derivative_backward'] = df.scaled_value.diff()
    df['first_order_derivative_forward'] = df.scaled_value.shift(-1).diff()
    # First order derivative for the rolling average
    df['first_order_derivative_backward_rolling_avg'] = \
        df.rolling_average.diff()
    df['first_order_derivative_forward_rolling_avg'] = \
        df.rolling_average.shift(-1).diff()
    # Calculate the maximum rolling range for the power or energy time series.
    df['deriv_max'] = _calculate_max_rolling_range(
        power_ac=df['scaled_value'], roll_periods=rolling_window)
    # Get the max value for the day and see how each value compares
    df['date'] = list(pd.to_datetime(pd.Series(df.index)).dt.date)
    df['daily_max'] = df.groupby(['date'])['scaled_value'].transform(max)
    # Get percentage of daily max
    df['percent_daily_max'] = df['scaled_value'] / (df['daily_max'] + .00001)
    # Get the standard deviation, median and mean of the first order
    # derivative over the rolling_window period
    df['deriv_backward_rolling_stdev'] = \
        df['first_order_derivative_backward']\
        .rolling(window=rolling_window, center=True).std()
    df['deriv_backward_rolling_mean'] = \
        df['first_order_derivative_backward']\
        .rolling(window=rolling_window, center=True).mean()
    df['deriv_backward_rolling_median'] = \
        df['first_order_derivative_backward']\
        .rolling(window=rolling_window, center=True).median()
    df['deriv_backward_rolling_max'] = \
        df['first_order_derivative_backward']\
        .rolling(window=rolling_window, center=True).max()
    df['deriv_backward_rolling_min'] = \
        df['first_order_derivative_backward']\
        .rolling(window=rolling_window, center=True).min()
    return df


def xgboost_clip_filter(power_ac,
                        mounting_type='fixed'):
    """
    This function generates the features to run through the XGBoost
    clipping model, runs the data through the model, and generates
    model outputs.

    Parameters
    ----------
    power_ac : pandas.Series
        Pandas time series, representing PV system power or energy.
        For best performance, timestamps should be in local time.
    mounting_type: str, default 'fixed'
        String representing the mounting configuration associated with the
        AC power or energy time series. Can either be "fixed" or
        "single_axis_tracking".

    Returns
    -------
    pandas.Series
        Boolean Series of whether to include the point because it is not
        clipping.
        True values delineate non-clipping periods, and False values delineate
        clipping periods.

    References
    ----------
    .. [1] Perry K., Muller, M., and Anderson K. "Performance comparison of clipping
       detection techniques in AC power time series", 2021 IEEE 48th Photovoltaic
       Specialists Conference (PVSC). DOI: 10.1109/PVSC43889.2021.9518733.
    """
    # Throw a warning that this is still an experimental filter
    warnings.warn("The XGBoost filter is an experimental clipping filter "
                  "that is still under development. The API, results, and "
                  "default behaviors may change in future releases (including "
                  "MINOR and PATCH). Use at your own risk!")
    # Load in the XGBoost model
    xgboost_clipping_model = _load_xgboost_clipping_model()
    # Format the power or energy time series
    power_ac, index_name = _format_clipping_time_series(power_ac,
                                                        mounting_type)
    # Test if the data sampling frequency is variable, and flag it if the time
    # series sampling frequency is less than 95% consistent.
    _check_data_sampling_frequency(power_ac)
    # Get the most common sampling frequency
    sampling_frequency = int(power_ac.index.to_series().diff()
                             .astype('timedelta64[m]').mode()[0])
    freq_string = str(sampling_frequency) + "T"
    # Min-max normalize
    # Resample the series based on the most common sampling frequency
    power_ac_interpolated = rdtools.normalization.interpolate(power_ac,
                                                              freq_string)
    # Convert the Pandas series to a dataframe.
    power_ac_df = power_ac_interpolated.to_frame()
    # Get the sampling frequency (as a continuous feature variable)
    power_ac_df['sampling_frequency'] = sampling_frequency
    # If the data sampling frequency of the series is more frequent than
    # once every five minute, resample at 5-minute intervals before
    # plugging into the model
    if sampling_frequency < 5:
        power_ac_df = power_ac_df.resample('5T').mean()
        power_ac_df['sampling_frequency'] = 5
    # Add mounting type as a column
    power_ac_df['mounting_config'] = mounting_type
    # Generate the features for the model.
    power_ac_df = _calculate_xgboost_model_features(power_ac_df,
                                                    sampling_frequency)
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
                               'percent_daily_max', 'deriv_max',
                               'deriv_backward_rolling_stdev',
                               'deriv_backward_rolling_mean',
                               'deriv_backward_rolling_median',
                               'deriv_backward_rolling_min',
                               'deriv_backward_rolling_max']].dropna()
    # Run the power_ac_df dataframe through the XGBoost ML model,
    # and return boolean outputs
    xgb_predictions = pd.Series(xgboost_clipping_model.predict(
        power_ac_df).astype(bool))
    # Add datetime as an index
    xgb_predictions.index = power_ac_df.index
    # Reindex with the original data index. Re-adjusts to original
    # data frequency.
    xgb_predictions = xgb_predictions.reindex(index=power_ac.index,
                                              method='ffill')
    xgb_predictions = xgb_predictions.fillna(False)
    # Regenerate the features with the original sampling frequency
    # (pre-resampling or interpolation).
    power_ac_df = power_ac.to_frame()
    power_ac_df = _calculate_xgboost_model_features(power_ac_df,
                                                    sampling_frequency)
    # Add back in XGB predictions for the original dataframe
    power_ac_df['xgb_predictions'] = xgb_predictions.astype(bool)
    power_ac_df_clipping = power_ac_df[power_ac_df['xgb_predictions']
                                       .fillna(False)]
    # Make everything between the
    # max and min values found for clipping each day as clipping.
    power_ac_df_clipping_max = power_ac_df_clipping['scaled_value']\
        .resample('D').max()
    power_ac_df_clipping_min = power_ac_df_clipping['scaled_value']\
        .resample('D').min()
    power_ac_df['daily_clipping_min'] = power_ac_df_clipping_min.reindex(
        index=power_ac_df.index, method='ffill')
    power_ac_df['daily_clipping_max'] = power_ac_df_clipping_max.reindex(
        index=power_ac_df.index, method='ffill')
    if sampling_frequency < 5:
        power_ac_df['daily_clipping_max_threshold'] = \
            (power_ac_df['daily_clipping_max'] * .96)
        power_ac_df['clipping cutoff'] = \
            power_ac_df[['daily_clipping_min',
                         'daily_clipping_max_threshold']].max(axis=1)
        final_clip = ((power_ac_df['clipping cutoff'] <=
                       power_ac_df['scaled_value'])
                      & (power_ac_df['percent_daily_max'] >= .9)
                      & (power_ac_df['scaled_value'] <=
                         power_ac_df['daily_clipping_max'] * 1.0025)
                      & (power_ac_df['scaled_value'] >= .1))
    else:
        final_clip = ((power_ac_df['daily_clipping_min'] <=
                       power_ac_df['scaled_value'])
                      & (power_ac_df['percent_daily_max'] >= .95)
                      & (power_ac_df['scaled_value'] <=
                         power_ac_df['daily_clipping_max'] * 1.0025)
                      & (power_ac_df['scaled_value'] >= .1))
    final_clip = final_clip.reindex(index=power_ac.index, fill_value=False)
    return ~(final_clip.astype(bool))
