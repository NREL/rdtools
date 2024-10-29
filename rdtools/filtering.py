"""Functions for filtering and subsetting PV system data."""

import numpy as np
import pandas as pd
import os
import warnings
import pvlib
from scipy.interpolate import interp1d
import rdtools
import xgboost as xgb

# Load in the XGBoost clipping model using joblib.
xgboost_clipping_model = None
model_path = os.path.join(
    os.path.dirname(__file__), "models", "xgboost_clipping_model.json"
)


def _load_xgboost_clipping_model():
    global xgboost_clipping_model
    if xgboost_clipping_model is None:
        xgboost_clipping_model = xgb.XGBClassifier()
        xgboost_clipping_model.load_model(model_path)
    return xgboost_clipping_model


def normalized_filter(
    energy_normalized, energy_normalized_low=0.01, energy_normalized_high=None
):
    """
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
    """

    if energy_normalized_low is None:
        energy_normalized_low = -np.inf
    if energy_normalized_high is None:
        energy_normalized_high = np.inf

    return (energy_normalized > energy_normalized_low) & (
        energy_normalized < energy_normalized_high
    )


def poa_filter(poa_global, poa_global_low=200, poa_global_high=1200):
    """
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
    """
    return (poa_global > poa_global_low) & (poa_global < poa_global_high)


def tcell_filter(temperature_cell, temperature_cell_low=-50, temperature_cell_high=110):
    """
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
    """
    return (temperature_cell > temperature_cell_low) & (
        temperature_cell < temperature_cell_high
    )


def clearsky_filter(poa_global_measured, poa_global_clearsky, model='pvlib', **kwargs):
    """
    Wrapper function for running either the CSI or pvlib clearsky filter.

    Parameters
    ----------
    poa_global_measured : pandas.Series
        Plane of array irradiance based on measurments
    poa_global_clearsky : pandas.Series
        Plane of array irradiance based on a clear sky model
    model : str, default 'pvlib'
        Clearsky filter model to be applied. Can be 'pvlib' or 'csi'.
    kwargs :
        Additional clearsky filter args, specific to the filter being
        used. Keyword must be passed with value.

    Returns
    -------
    pandas.Series
        Boolean Series of whether or not the given time is clear
        based on the selected filter.

    See Also
    --------
    csi_filter : Filtering based on clear-sky index (csi).
    pvlib_clearsky_filter : Filtering based on pvlib's clearsky model.
    """

    if model == "csi":
        clearsky_mask = csi_filter(poa_global_measured, poa_global_clearsky, **kwargs)
    elif model == "pvlib":
        clearsky_mask = pvlib_clearsky_filter(poa_global_measured, poa_global_clearsky, **kwargs)
    else:
        raise ValueError("Clearsky filter must be 'pvlib' or 'csi'.")
    return clearsky_mask


def csi_filter(poa_global_measured, poa_global_clearsky, threshold=0.15):
    """
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
    """

    csi = poa_global_measured / poa_global_clearsky
    return (csi >= 1.0 - threshold) & (csi <= 1.0 + threshold)


def pvlib_clearsky_filter(
    poa_global_measured,
    poa_global_clearsky,
    window_length=90,
    mean_diff=75,
    max_diff=75,
    lower_line_length=-45,
    upper_line_length=80,
    var_diff=0.032,
    slope_dev=75,
    lookup_parameters=False,
    **kwargs,
):
    """
    Filtering based on the Reno and Hansen method for clear-sky filtering
    as implimented in pvlib. Requires a regular time series with uniform
    time steps.

    Parameters
    ----------
    poa_global_measured : pandas.Series
        Plane of array irradiance based on measurments
    poa_global_clearsky : pandas.Series
        Plane of array irradiance based on a clear sky model
    window_length : int, default 10
        Length of sliding time window in minutes. Must be greater than 2
        periods.
    mean_diff : float, default 75
        Threshold value for agreement between mean values of measured
        and clearsky in each interval, see Eq. 6 in [1]. [W/m2]
    max_diff : float, default 75
        Threshold value for agreement between maxima of measured and
        clearsky values in each interval, see Eq. 7 in [1]. [W/m2]
    lower_line_length : float, default -5
        Lower limit of line length criterion from Eq. 8 in [1].
        Criterion satisfied when lower_line_length < line length difference
        < upper_line_length.
    upper_line_length : float, default 10
        Upper limit of line length criterion from Eq. 8 in [1].
    var_diff : float, default 0.005
        Threshold value in Hz for the agreement between normalized
        standard deviations of rate of change in irradiance, see Eqs. 9
        through 11 in [1].
    slope_dev : float, default 8
        Threshold value for agreement between the largest magnitude of
        change in successive values, see Eqs. 12 through 14 in [1].
    lookup_parameters : bool, default False
        Look up the recomended parameters [2] based on the
        frequency of poa_global_measured. If poa_global_measured has a defined
        frequency, this overrides the values of window_length, max_diff,
        var_diff, and slope_dev. For frequencies below 1 minute or greater than
        30, the lookup uses the recomended parameters for 1 or 30 minutes
        respectively. If poa_global_measured doesn't have a defined frequency,
        the passed or default values of the parameters are used.
    kwargs :
        Additional arguments passed to pvlib.clearsky.detect_clearsky
        return_components is set to False and not passed.

    Returns
    -------
    pandas.Series
        Boolean Series of whether or not the given time is clear.

    References
    ----------
    [1] M.J. Reno and C.W. Hansen, Renewable Energy 90, pp. 520-531 (2016)
    [2] D.C. Jordan and C.W. Hansen, Renewable Energy 209 pp. 393-400 (2023)


    """

    if lookup_parameters and poa_global_measured.index.freq:
        frequencies = np.array([1, 5, 15, 30])
        windows = np.array([50, 60, 90, 120])
        max_diffs = np.array([60, 65, 75, 90])
        var_diffs = np.array([0.005, 0.01, 0.032, 0.07])
        slope_devs = np.array([50, 60, 75, 96])

        windows_interp = interp1d(
            frequencies,
            windows,
            fill_value=(windows[0], windows[-1]),
            bounds_error=False,
        )
        max_diffs_interp = interp1d(
            frequencies,
            max_diffs,
            fill_value=(max_diffs[0], max_diffs[-1]),
            bounds_error=False,
        )
        var_diffs_interp = interp1d(
            frequencies,
            var_diffs,
            fill_value=(var_diffs[0], var_diffs[-1]),
            bounds_error=False,
        )
        slope_devs_interp = interp1d(
            frequencies,
            slope_devs,
            fill_value=(slope_devs[0], slope_devs[-1]),
            bounds_error=False,
        )

        freq_minutes = poa_global_measured.index.freq.nanos / 10**9 / 60
        window_length = windows_interp(freq_minutes)
        max_diff = max_diffs_interp(freq_minutes)
        var_diff = var_diffs_interp(freq_minutes)
        slope_dev = slope_devs_interp(freq_minutes)

    df = pd.concat([poa_global_measured, poa_global_clearsky], axis=1, join="outer")
    df.columns = ["measured", "clearsky"]

    kwargs["return_components"] = False
    mask = pvlib.clearsky.detect_clearsky(
        df["measured"],
        df["clearsky"],
        window_length=window_length,
        mean_diff=mean_diff,
        max_diff=max_diff,
        lower_line_length=lower_line_length,
        upper_line_length=upper_line_length,
        var_diff=var_diff,
        slope_dev=slope_dev,
        **kwargs,
    )
    return mask


def clip_filter(power_ac, model="logic", **kwargs):
    """
    Master wrapper for running one of the desired clipping filters.
    The default filter run is the quantile clipping filter.

    Parameters
    ----------
    power_ac : pandas.Series
        Pandas time series, representing PV system power or energy.
        For best performance, timestamps should be in local time.
    model : str, default 'logic'
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

    if model == "quantile":
        clip_mask = quantile_clip_filter(power_ac, **kwargs)
    elif model == "xgboost":
        clip_mask = xgboost_clip_filter(power_ac, **kwargs)
    elif model == "logic":
        clip_mask = logic_clip_filter(power_ac, **kwargs)
    else:
        raise ValueError("Variable model must be 'quantile', " "'xgboost', or 'logic'.")
    return clip_mask


def quantile_clip_filter(power_ac, quantile=0.98):
    """
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
    """
    v = power_ac.quantile(quantile)
    return power_ac < v * 0.99


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
        raise TypeError("Must be a Pandas series with a datetime index.")
    # Check if the time series is tz-aware. If not, throw a
    # warning.
    has_timezone = pd.Series(power_ac.index).apply(lambda t: t.tzinfo is not None)
    # Throw a warning that we're expecting time zone-localized data,
    # if no time zone is specified.
    if not has_timezone.all():
        warnings.warn(
            "Function expects timestamps in local time. "
            "For best results pass a time-zone-localized "
            "time series localized to the correct local time zone."
        )
    # Check the other input variables to ensure that they are the
    # correct format
    if (mounting_type != "single_axis_tracking") & (mounting_type != "fixed"):
        raise ValueError(
            "Variable mounting_type must be string 'single_axis_tracking' or "
            "'fixed'."
        )
    # Check if the datetime index is out of order. If it is, throw an
    # error.
    if not all(power_ac.sort_index().index == power_ac.index):
        raise IndexError(
            "Time series index has not been sorted. Implement the "
            "sort_index() method to the time series to rerun this function."
        )
    # Check that there is enough data in the dataframe. Must be greater than
    # 10 readings.
    if len(power_ac) <= 10:
        raise Exception("<=10 readings in the time series, cannot run filter.")
    # Get the names of the series and the datetime index
    column_name = "value"
    power_ac = power_ac.rename(column_name)
    index_name = power_ac.index.name
    if index_name is None:
        index_name = "datetime"
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
    sampling_frequency_df = pd.DataFrame(
        power_ac.index.to_series().diff().astype("timedelta64[s]").value_counts()
    ) / len(power_ac)
    sampling_frequency_df.columns = ["count"]
    if (sampling_frequency_df["count"] < 0.95).all():
        warnings.warn(
            "Variable sampling frequency across time series. "
            "Less than 95% of the time series is sampled at the "
            "same interval. This function was not tested "
            "on variable frequency data--use at your own risk!"
        )
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
    rolling_range_max = (max_roll - min_roll) / ((max_roll + min_roll) / 2) * 100
    return rolling_range_max


def _apply_overall_clipping_threshold(power_ac, clipping_mask, clipped_power_ac):
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
    upper_bound_pdiff = abs(
        (power_ac.quantile(0.99) - clipped_power_ac.quantile(0.99))
        / ((power_ac.quantile(0.99) + clipped_power_ac.quantile(0.99)) / 2)
    )
    percent_clipped = len(clipped_power_ac) / len(power_ac) * 100
    if (upper_bound_pdiff < 0.005) & (percent_clipped > 4):
        max_clip = power_ac >= power_ac.quantile(0.99)
        clipping_mask = clipping_mask | max_clip
    return clipping_mask


def logic_clip_filter(
    power_ac, mounting_type="fixed", rolling_range_max_cutoff=0.2, roll_periods=None
):
    """
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
    """
    # Format the time series
    power_ac, index_name = _format_clipping_time_series(power_ac, mounting_type)
    # Test if the data sampling frequency is variable, and flag it if the time
    # series sampling frequency is less than 95% consistent.
    _check_data_sampling_frequency(power_ac)
    # Get the sampling frequency of the time series
    time_series_sampling_frequency = (
        power_ac.index.to_series().diff() / pd.Timedelta("60s")
    ).mode()[0]
    # Make copies of the original inputs for the cases that the data is
    # changes for clipping evaluation
    original_time_series_sampling_frequency = time_series_sampling_frequency
    power_ac_copy = power_ac.copy()
    # Drop duplicate indices
    power_ac = (
        power_ac.reset_index()
        .drop_duplicates(subset=power_ac.index.name, keep="first")
        .set_index(power_ac.index.name)
    )
    freq_string = str(time_series_sampling_frequency) + "min"
    # Set days with the majority of frozen data to null.
    daily_std = power_ac.resample("D").std() / power_ac.resample("D").mean()
    power_ac["daily_std"] = daily_std.reindex(index=power_ac.index, method="ffill")
    power_ac.loc[power_ac["daily_std"] < 0.1, "value"] = np.nan
    power_ac.drop("daily_std", axis=1, inplace=True)
    power_cleaned = power_ac["value"].copy()
    power_cleaned = power_cleaned.reindex(power_ac_copy.index, fill_value=np.nan)
    # High frequency data (less than 10 minutes) has demonstrated
    # potential to have more noise than low frequency  data.
    # Therefore, the  data is resampled to a 15-minute median
    # before running the filter.
    if time_series_sampling_frequency >= 10:
        power_ac = rdtools.normalization.interpolate(power_ac, freq_string)
    else:
        power_ac = power_ac.resample("15min").median()
        time_series_sampling_frequency = 15
    # If a value for roll_periods is not designated, the function uses
    # the current default logic to set the roll_periods value.
    if roll_periods is None:
        if (mounting_type == "single_axis_tracking") & (
            time_series_sampling_frequency < 30
        ):
            roll_periods = 5
        else:
            roll_periods = 3
    # Replace the lower 10% of daily data with NaN's
    daily = 0.1 * power_ac.resample("D").max()
    power_ac["ten_percent_daily"] = daily.reindex(index=power_ac.index, method="ffill")
    power_ac.loc[power_ac["value"] < power_ac["ten_percent_daily"], "value"] = np.nan
    power_ac = power_ac["value"]
    # Calculate the maximum rolling range for the time series.
    rolling_range_max = _calculate_max_rolling_range(power_ac, roll_periods)
    # Determine clipping values based on the maximum rolling range in
    # the rolling window, and the user-specified rolling range threshold
    roll_clip_mask = rolling_range_max < rolling_range_max_cutoff
    # Set values within roll_periods values from a True instance
    # as True as well
    clipping = roll_clip_mask.rolling(roll_periods).sum() >= 1
    # High frequency was resampled to 15-minute average data.
    # The following lines apply the 15-minute clipping filter to the
    # original 15-minute data resulting in a clipping filter on the original
    # data.
    if original_time_series_sampling_frequency < 10:
        clipping = clipping.reindex(index=power_ac_copy.index, method="ffill")
        # Subset the series where clipping filter == True
        clip_pwr = power_ac_copy[clipping]
        clip_pwr = clip_pwr.reindex(index=power_ac_copy.index, fill_value=np.nan)
        # Set any values within the clipping max + clipping min threshold
        # as clipping. This is done specifically for capturing the noise
        # for high frequency data sets.
        daily_mean = clip_pwr.resample("D").mean()
        df_daily = daily_mean.to_frame(name="mean")
        df_daily["clipping_max"] = clip_pwr.groupby(pd.Grouper(freq="D")).quantile(0.99)
        df_daily["clipping_min"] = clip_pwr.groupby(pd.Grouper(freq="D")).quantile(
            0.075
        )
        daily_clipping_max = df_daily["clipping_max"].reindex(
            index=power_ac_copy.index, method="ffill"
        )
        daily_clipping_min = df_daily["clipping_min"].reindex(
            index=power_ac_copy.index, method="ffill"
        )
    else:
        # Find the maximum and minimum power_ac level where clipping is
        # detected each day.
        clipping = clipping.reindex(index=power_ac_copy.index, method="ffill")
        clip_pwr = power_ac_copy[clipping]
        clip_pwr = clip_pwr.reindex(index=power_ac_copy.index, fill_value=np.nan)
        daily_clipping_max = clip_pwr.resample("D").max()
        daily_clipping_min = clip_pwr.resample("D").min()
        daily_clipping_min = daily_clipping_min.reindex(
            index=power_ac_copy.index, method="ffill"
        )
        daily_clipping_max = daily_clipping_max.reindex(
            index=power_ac_copy.index, method="ffill"
        )
    # Set all values to clipping that are between the maximum and minimum
    # power_ac levels where clipping was found on a daily basis.
    clipping_difference = (daily_clipping_max - daily_clipping_min) / daily_clipping_max
    final_clip = (
        (
            (daily_clipping_min <= power_ac_copy)
            & (power_ac_copy <= daily_clipping_max)
            & (clipping_difference <= 0.025)
        )
        | (
            (power_ac_copy <= daily_clipping_max * 1.0025)
            & (power_ac_copy >= daily_clipping_max * 0.9975)
            & (clipping_difference > 0.025)
        )
        | (
            (power_ac_copy <= daily_clipping_min * 1.0025)
            & (power_ac_copy >= daily_clipping_min * 0.9975)
            & (clipping_difference > 0.025)
        )
    )
    final_clip = final_clip.reindex(index=power_ac_copy.index, fill_value=False)
    # Check for an overall clipping threshold that should apply to all data
    clip_power_ac = power_ac_copy[final_clip]
    final_clip = _apply_overall_clipping_threshold(
        power_cleaned, final_clip, clip_power_ac
    )
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
    max_min_diff = df["value"].max() - df["value"].min()
    df["scaled_value"] = (df["value"] - df["value"].min()) / max_min_diff
    if sampling_frequency < 10:
        rolling_window = 5
    elif (sampling_frequency >= 10) and (sampling_frequency < 60):
        rolling_window = 3
    else:
        rolling_window = 2
    df["rolling_average"] = (
        df["scaled_value"].rolling(window=rolling_window, center=True).mean()
    )
    # First-order derivative
    df["first_order_derivative_backward"] = df.scaled_value.diff()
    df["first_order_derivative_forward"] = df.scaled_value.shift(-1).diff()
    # First order derivative for the rolling average
    df["first_order_derivative_backward_rolling_avg"] = df.rolling_average.diff()
    df["first_order_derivative_forward_rolling_avg"] = df.rolling_average.shift(
        -1
    ).diff()
    # Calculate the maximum rolling range for the power or energy time series.
    df["deriv_max"] = _calculate_max_rolling_range(
        power_ac=df["scaled_value"], roll_periods=rolling_window
    )
    # Get the max value for the day and see how each value compares
    df["date"] = list(pd.to_datetime(pd.Series(df.index)).dt.date)
    df["daily_max"] = df.groupby(["date"])["scaled_value"].transform("max")

    # Get percentage of daily max
    df["percent_daily_max"] = df["scaled_value"] / (df["daily_max"] + 0.00001)
    # Get the standard deviation, median and mean of the first order
    # derivative over the rolling_window period
    df["deriv_backward_rolling_stdev"] = (
        df["first_order_derivative_backward"]
        .rolling(window=rolling_window, center=True)
        .std()
    )
    df["deriv_backward_rolling_mean"] = (
        df["first_order_derivative_backward"]
        .rolling(window=rolling_window, center=True)
        .mean()
    )
    df["deriv_backward_rolling_median"] = (
        df["first_order_derivative_backward"]
        .rolling(window=rolling_window, center=True)
        .median()
    )
    df["deriv_backward_rolling_max"] = (
        df["first_order_derivative_backward"]
        .rolling(window=rolling_window, center=True)
        .max()
    )
    df["deriv_backward_rolling_min"] = (
        df["first_order_derivative_backward"]
        .rolling(window=rolling_window, center=True)
        .min()
    )
    return df


def xgboost_clip_filter(power_ac, mounting_type="fixed"):
    """
    This filter uses and XGBoost model to filter out
    clipping periods in AC power or energy time series.

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
    # Load in the XGBoost model
    xgboost_clipping_model = _load_xgboost_clipping_model()
    # Format the power or energy time series
    power_ac, index_name = _format_clipping_time_series(power_ac, mounting_type)
    # Test if the data sampling frequency is variable, and flag it if the time
    # series sampling frequency is less than 95% consistent.
    _check_data_sampling_frequency(power_ac)
    # Get the most common sampling frequency
    sampling_frequency = int(
        (power_ac.index.to_series().diff() / pd.Timedelta("60s")).mode()[0]
    )
    freq_string = str(sampling_frequency) + "min"
    # Min-max normalize
    # Resample the series based on the most common sampling frequency
    power_ac_interpolated = rdtools.normalization.interpolate(power_ac, freq_string)
    # Convert the Pandas series to a dataframe.
    power_ac_df = power_ac_interpolated.to_frame()
    # Get the sampling frequency (as a continuous feature variable)
    power_ac_df["sampling_frequency"] = sampling_frequency
    # If the data sampling frequency of the series is more frequent than
    # once every five minute, resample at 5-minute intervals before
    # plugging into the model
    if sampling_frequency < 5:
        power_ac_df = power_ac_df.resample("5min").mean()
        power_ac_df["sampling_frequency"] = 5
    # Add mounting type as a column
    power_ac_df["mounting_config"] = mounting_type
    # Generate the features for the model.
    power_ac_df = _calculate_xgboost_model_features(power_ac_df, sampling_frequency)
    # Convert single-axis tracking/fixed tilt to a boolean variable
    power_ac_df.loc[
        power_ac_df["mounting_config"] == "single_axis_tracking", "mounting_config_bool"
    ] = 1
    power_ac_df.loc[
        power_ac_df["mounting_config"] == "fixed", "mounting_config_bool"
    ] = 0
    # Subset the dataframe to only include model inputs
    power_ac_df = power_ac_df[
        [
            "first_order_derivative_backward",
            "first_order_derivative_forward",
            "first_order_derivative_backward_rolling_avg",
            "first_order_derivative_forward_rolling_avg",
            "sampling_frequency",
            "mounting_config_bool",
            "scaled_value",
            "rolling_average",
            "daily_max",
            "percent_daily_max",
            "deriv_max",
            "deriv_backward_rolling_stdev",
            "deriv_backward_rolling_mean",
            "deriv_backward_rolling_median",
            "deriv_backward_rolling_min",
            "deriv_backward_rolling_max",
        ]
    ].dropna()
    # Run the power_ac_df dataframe through the XGBoost ML model,
    # and return boolean outputs
    xgb_predictions = pd.Series(
        xgboost_clipping_model.predict(power_ac_df).astype(bool)
    )
    # Add datetime as an index
    xgb_predictions.index = power_ac_df.index
    # Reindex with the original data index. Re-adjusts to original
    # data frequency.
    xgb_predictions = xgb_predictions.reindex(index=power_ac.index, method="ffill")
    xgb_predictions.loc[xgb_predictions.isnull()] = False

    # Regenerate the features with the original sampling frequency
    # (pre-resampling or interpolation).
    power_ac_df = power_ac.to_frame()
    power_ac_df = _calculate_xgboost_model_features(power_ac_df, sampling_frequency)
    # Add back in XGB predictions for the original dataframe
    power_ac_df["xgb_predictions"] = xgb_predictions.astype(bool)
    power_ac_df_clipping = power_ac_df[power_ac_df["xgb_predictions"].fillna(False)]
    # Make everything between the
    # max and min values found for clipping each day as clipping.
    power_ac_df_clipping_max = power_ac_df_clipping["scaled_value"].resample("D").max()
    power_ac_df_clipping_min = power_ac_df_clipping["scaled_value"].resample("D").min()
    power_ac_df["daily_clipping_min"] = power_ac_df_clipping_min.reindex(
        index=power_ac_df.index, method="ffill"
    )
    power_ac_df["daily_clipping_max"] = power_ac_df_clipping_max.reindex(
        index=power_ac_df.index, method="ffill"
    )
    if sampling_frequency < 5:
        power_ac_df["daily_clipping_max_threshold"] = (
            power_ac_df["daily_clipping_max"] * 0.96
        )
        power_ac_df["clipping cutoff"] = power_ac_df[
            ["daily_clipping_min", "daily_clipping_max_threshold"]
        ].max(axis=1)
        final_clip = (
            (power_ac_df["clipping cutoff"] <= power_ac_df["scaled_value"])
            & (power_ac_df["percent_daily_max"] >= 0.9)
            & (
                power_ac_df["scaled_value"]
                <= power_ac_df["daily_clipping_max"] * 1.0025
            )
            & (power_ac_df["scaled_value"] >= 0.1)
        )
    else:
        final_clip = (
            (power_ac_df["daily_clipping_min"] <= power_ac_df["scaled_value"])
            & (power_ac_df["percent_daily_max"] >= 0.95)
            & (
                power_ac_df["scaled_value"]
                <= power_ac_df["daily_clipping_max"] * 1.0025
            )
            & (power_ac_df["scaled_value"] >= 0.1)
        )
    final_clip = final_clip.reindex(index=power_ac.index, fill_value=False)
    return ~(final_clip.astype(bool))


def two_way_window_filter(
    series, roll_period=pd.to_timedelta("7 Days"), outlier_threshold=0.03
):
    """
    Removes anomalies based on forward and backward window of the rolling median. Points beyond
    outlier_threshold from both the forward and backward-looking median are excluded by the filter.
    Designed for use after the aggregation step in the RdTools trend analysis workflows.

    Parameters
    ----------
    series: pandas.Series
        Pandas time series to be filtered.
    roll_period : int or timedelta, default 7 days
        The window to use for backward and forward
        rolling medians for detecting outliers.
    outlier_threshold : default is 0.03 meaning 3%

    Returns
    -------
    pandas.Series
        Boolean Series excluding anomalies
    """

    series = series / series.quantile(0.99)
    backward_median = series.rolling(roll_period, min_periods=5, closed="both").median()
    forward_median = (
        series.loc[::-1].rolling(roll_period, min_periods=5, closed="both").median()
    )

    backward_dif = abs(series - backward_median)
    forward_dif = abs(series - forward_median)

    # This is a change from Matt's original logic, which can exclude
    # points with a NaN median
    backward_dif.fillna(0, inplace=True)
    forward_dif.fillna(0, inplace=True)

    dif_min = backward_dif.combine(forward_dif, min, 0)

    mask = dif_min < outlier_threshold

    return mask


def insolation_filter(insolation, quantile=0.1):
    """
    A simple quantile filter. Primary application in RdTools is to exclude
    low insolation points after the aggregation step in the trend analysis
    workflows.

    Parameters
    ----------
    insolation: pandas.Series
        Pandas time series to be filtered. Usually insolation.
    quantile : float, default 0.1
        the minimum quantile above which data is kept.

    Returns
    -------
    pandas.Series
        Boolean Series excluding points below the quantile threshold
    """

    limit = insolation.quantile(quantile)
    mask = insolation >= limit
    return mask


def hampel_filter(series, k="14d", t0=3):
    """
    Hampel outlier designed for use after the aggregation step
    in the RdTools trend analysis workflows, but broadly
    applicable.

    Parameters
    ----------
    series : pandas.Series
        daily normalized time series
    k : int or time offset string e.g. 'd', default 14d
        size of window including the sample; 14d is equal to 7 days on either
        side of value
    t0 : int, default 3
        Threshold value, defaults to 3 sigma Pearson's rule.
    Returns
    -------
    pandas.Series
        Boolean Series of whether the given measurement is within t0 sigma of the
        rolling median.  False points indicate outliers to be excluded.
    """
    # Hampel Filter
    L = 1.4826
    rolling_median = series.rolling(k, center=True, min_periods=1).median()
    difference = np.abs(rolling_median - series)
    median_abs_deviation = difference.rolling(k, center=True, min_periods=1).median()
    threshold = t0 * L * median_abs_deviation
    return difference <= threshold


def _tukey_fence(series, k=1.5):
    "Calculates the upper and lower tukey fences from a pandas series"
    p25 = series.quantile(0.25)
    p75 = series.quantile(0.75)
    iqr = p75 - p25
    upper_fence = k * iqr + p75
    lower_fence = p25 - 1.5 * iqr
    return lower_fence, upper_fence


def directional_tukey_filter(series, roll_period=pd.to_timedelta("7 Days"), k=1.5):
    """
    Performs a forward and backward looking rolling Tukey filter. Points more than k*IQR
    above the third quartile or below the first quartile are classified as outliers. Points
    must only pass one of either the forward or backward looking filters to be kept. Designed
    for use after the aggregation step in the RdTools trend analysis workflows


    Parameters
    ----------
    series: pandas.Series
        Pandas time series to be filtered.
    roll_period : int or timedelta, default 7 days
        The window to use for backward and forward
        rolling medians for detecting outliers.
    k : float
        The Tukey parameter. Points more than k*IQR above the third quartile
        or below the first quartile are classified as outliers.

    Returns
    -------
    pandas.Series
        Boolean Series excluding anomalies
    """

    backward_median = series.rolling(roll_period, min_periods=5, closed="both").median()
    forward_median = (
        series.loc[::-1].rolling(roll_period, min_periods=5, closed="both").median()
    )
    backward_dif = series - backward_median
    forward_dif = series - forward_median

    backward_dif_lower, backward_dif_upper = _tukey_fence(backward_dif, k)
    forward_dif_lower, forward_dif_upper = _tukey_fence(forward_dif, k)

    mask = ((forward_dif > forward_dif_lower) & (forward_dif < forward_dif_upper)) | (
        (backward_dif > backward_dif_lower) & (backward_dif < backward_dif_upper)
    )
    return mask


def hour_angle_filter(series, lat, lon, min_hour_angle=-30, max_hour_angle=30):
    """
    Creates a filter based on the hour angle of the sun (15 degrees per hour)

    Parameters
    ----------
    series: pandas.Series
        Pandas time series to be filtered
    lat: float
        location latitude
    lon: float
        location longitude
    min_hour_angle: float
        minimum hour angle to include
    max_hour_angle: float
        maximum hour angle to include

    Returns
    -------
    pandas.Series
        Boolean Series excluding points outside the specified hour
        angle range

    """

    times = series.index
    spa = pvlib.solarposition.get_solarposition(times, lat, lon)
    eot = spa["equation_of_time"]
    hour_angle = pvlib.solarposition.hour_angle(times, lon, eot)
    hour_angle = pd.Series(hour_angle, index=times)
    mask = (hour_angle >= min_hour_angle) & (hour_angle <= max_hour_angle)

    return mask
