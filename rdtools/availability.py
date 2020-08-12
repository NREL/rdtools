"""
Functions for detecting and quantifying production loss from photovoltaic
system downtime events.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def loss_from_power(subsystem_power, system_power, low_threshold=None,
                    system_power_limit=None):
    """
    Estimate timeseries production loss from system downtime events by
    comparing subsystem power data to total system power (e.g. inverter power
    to meter power). This implements the "power comparison" method from [1]_.

    Because this method is based on peer-to-peer comparison at each timestamp,
    it is not suitable for full system outages (i.e., at least one inverter
    must be reporting along with the system meter).

    Parameters
    ----------
    subsystem_power : pd.DataFrame
        Timeseries power data, one column per subsystem. In the typical case,
        this is inverter AC power data.

    system_power : pd.Series
        Timeseries total system power. In the typical case, this is meter
        power data. The index must match ``subsystem_power``.

    low_threshold : float or pd.Series, optional
        An optional threshold used to naively classify subsystems as online.
        If the threshold is a scalar, it will be used for all subsystems. For
        subsystems with different capacities, a pandas Series may be passed
        with index values matching the columns in ``subsystem_power``. Units
        must match ``subsystem_power`` and ``system_power``.
        If omitted, the limit is calculated for each subsystem independently
        as 0.001 times the 99th percentile of its power data.

    system_power_limit : float or pd.Series, optional
        An optional maximum system power used as an upper limit for
        (system_power + lost_power) so that the maximum system capacity or
        interconnection limit is not exceeded. If omitted, that check is
        skipped.

    Returns
    -------
    p_loss : pd.Series
        Estimated timeseries power loss due to subsystem downtime. The index
        matches the input power data.

    See Also
    --------
    rdtools.availability.loss_from_energy

    Notes
    -----
    This method's ability to detect short-duration outages is limited by the
    resolution of the power data. For instance, 15-minute averages would not
    be able to resolve the rapid power cycling of an intermittent inverter.
    Additionally, the loss at the edges of an outage may be underestimated
    because of masking by the interval averages.

    References
    ----------
    .. [1] Anderson K. and Blumenthal R. "Overcoming communications outages in
       inverter downtime analysis", 2020 IEEE 47th Photovoltaic Specialists
       Conference (PVSC).
    """

    subsystem_power = subsystem_power.fillna(0)
    system_power = system_power.clip(lower=0)

    # Part A
    if low_threshold is None:
        low_threshold = subsystem_power.quantile(0.99) / 1000

    looks_online = subsystem_power > low_threshold
    reporting = subsystem_power[looks_online]
    relative_sizes = reporting.divide(reporting.mean(axis=1), axis=0).median()
    normalized_subsystem_powers = reporting.divide(relative_sizes, axis=1)
    mean_subsystem_power = normalized_subsystem_powers.mean(axis=1)

    virtual_full_power = mean_subsystem_power * subsystem_power.shape[1]

    system_delta = 1 - system_power / virtual_full_power

    subsystem_fraction = relative_sizes / relative_sizes.sum()
    smallest_delta = subsystem_power.le(low_threshold) \
                                    .replace(False, np.nan) \
                                    .multiply(subsystem_fraction) \
                                    .min(axis=1) \
                                    .fillna(1)  # use safe value of 100%
    is_downtime = system_delta > (0.75 * smallest_delta)
    is_downtime[looks_online.all(axis=1)] = False

    # Part B
    lowest_possible = (looks_online.multiply(subsystem_fraction)).sum(axis=1)
    f_online = (system_power / virtual_full_power).clip(lower=lowest_possible,
                                                        upper=1)
    p_loss = (1 - f_online) / f_online * system_power
    p_loss[~is_downtime] = 0

    if system_power_limit is not None:
        limit_exceeded = p_loss + system_power > system_power_limit
        loss = system_power_limit - system_power[limit_exceeded]
        p_loss.loc[limit_exceeded] = loss.clip(lower=0)

    return p_loss.fillna(0)


def loss_from_energy(power, energy, subsystem_power, expected_power):
    """
    Estimate total production loss from system downtime events by
    comparing system production recovered from cumulative production data with
    expected production from an energy model. This implements the "expected
    energy" method from [1]_.

    This function is useful for full system outages when no system data is
    available at all. However, it does require cumulative production data
    recorded at the device level and only reports estimated lost production
    for entire outages rather than timeseries lost power.

    Parameters
    ----------
    power : pd.Series
        Timeseries power data for a system.

    energy : pd.Series
        Timeseries cumulative energy data for the system. These values
        must be recorded at the device level (rather than summed by a
        downstream device like a datalogger or DAS provider) to preserve its
        integrity across communication interruptions. Units must match
        ``power`` integrated to hourly energy (e.g. if ``power`` is in kW then
        ``energy`` must be in kWh).

    subsystem_power : pd.DataFrame
        Timeseries power data, one column per subsystem. In the typical case,
        this is inverter AC power data.

    expected_power : pd.Series
        Expected system power data with the same index as the measured data.
        This can be modeled from on-site weather measurements if there is no
        risk of instrument calibration or data gaps. However, because full
        system outages often cause weather data to be lost as well, it may
        be more useful to use data from an independent weather station or
        satellite-based weather provider.

    Returns
    -------
    outage_info : pd.DataFrame
        A dataframe of records about each detected outage, one row per outage.
        The primary columns of interest are ``type``, which can be either
        ``'real'`` or ``'comms'`` and reports whether the outage was determined
        to be a real outage with lost production or just a communications
        interruption with no production impact; and ``loss`` which reports
        the estimated production loss for the outage.

    See Also
    --------
    rdtools.availability.loss_from_power

    Notes
    -----
    This method's ability to detect short-duration outages is limited by the
    resolution of the system data. For instance, 15-minute averages would not
    be able to resolve the rapid power cycling of an intermittent inverter.
    Additionally, the loss at the edges of an outage may be underestimated
    because of masking by the interval averages.

    References
    ----------
    .. [1] Anderson K. and Blumenthal R. "Overcoming communications outages in
       inverter downtime analysis", 2020 IEEE 47th Photovoltaic Specialists
       Conference (PVSC).
    """
    df = pd.DataFrame({
        'Meter_kW': power,
        'Expected Power': expected_power,
        'Meter_kWh': energy,
    })

    power_loss = loss_from_power(subsystem_power, power)
    system_performing_normally = (power_loss == 0) & (power > 0)

    # filter out nighttime as well, since night intervals shouldn't count
    subset = system_performing_normally & (df['Expected Power'] > 0)

    # rescale expected energy to better match actual production.
    # this shifts the error distributions so that as interval length increases,
    # error -> 0
    scaling_subset = df.loc[subset, ['Expected Power', 'Meter_kW']].sum()
    scaling_factor = (
        scaling_subset['Expected Power'] / scaling_subset['Meter_kW']
    )
    df['Expected Power'] /= scaling_factor
    df['Expected Energy'] = df['Expected Power'] / 4
    df['Meter_kWh_interval'] = df['Meter_kW'] / 4

    df_subset = df.loc[subset, :]

    # window length is "number of daytime intervals".
    # Note: the logspace bounds are currently kind of arbitrary
    window_lengths = np.logspace(np.log10(12),
                                 np.log10(0.75*len(df_subset)),
                                 10).astype(int)
    results_list = []
    for window_length in window_lengths:
        rolling = df_subset.rolling(window=window_length, center=True).sum()
        actual = rolling['Meter_kWh_interval']
        expected = rolling['Expected Energy']
        # remove the nans at beginning and end because of minimum window length
        actual = actual[~np.isnan(actual)]
        expected = expected[~np.isnan(expected)]
        temp = pd.DataFrame({
            'actual': actual,
            'expected': expected,
            'window length': window_length
        })
        results_list.append(temp)

    df_error = pd.concat(results_list)
    df_error['error'] = df_error['actual'] / df_error['expected'] - 1

    upper = df_error.groupby('window length')['error'].quantile(0.99)
    lower = df_error.groupby('window length')['error'].quantile(0.01)

    # functions to predict the confidence interval for a given outage length.
    # linear interp inside the range, nearest neighbor outside the range
    def interp(series):
        return interp1d(series.index, series.values,
                        fill_value=(series.values[0], series.values[-1]),
                        bounds_error=False)
    # functions mapping number of intervals (outage length) to error bounds
    predict_upper = interp(upper)
    predict_lower = interp(lower)

    # Calculate boolean series to indicate full outages.
    # done this way (instead of inspecting power data, e.g.) so that outages
    # can span across nights.
    full_outage = ~(df['Meter_kWh'] > 0).fillna(False)

    # Find expected production and associated uncertainty for each outage
    diff = full_outage.astype(int).diff()
    starts = df.index[diff == 1].tolist()
    ends = df.index[diff == -1].tolist()
    steps = diff[~diff.isnull() & (diff != 0)]
    if steps[0] == -1:
        # data starts in an outage
        starts.insert(0, df.index[0])
    if steps[-1] == 1:
        # data ends in an outage
        ends.append(df.index[-1])

    outage_data = []
    for start, end in zip(starts, ends):
        df_outage = df.loc[start:end, :]

        daylight_intervals = (df_outage['Expected Power'] > 0).sum()

        # df.loc[start, 'Meter_kWh'] is the first value in the outage.
        # to get the starting energy, need to get previous value:
        start_minus_one = df.index[df.index.get_loc(start)-1]

        data = {
            'start': start,
            'end': end,
            'duration': end - start,
            'intervals': len(df_outage),
            'daylight_intervals': daylight_intervals,
            'error_lower': predict_lower(daylight_intervals),
            'error_upper': predict_upper(daylight_intervals),
            'expected_energy': df_outage['Expected Energy'].sum(),
            'start_energy': df.loc[start_minus_one, 'Meter_kWh'],
            'end_energy': df.loc[end, 'Meter_kWh'],
        }
        outage_data.append(data)

    df_outages = pd.DataFrame(outage_data)
    # pandas < 0.25.0 sorts columns alphabetically.  revert to dict order:
    df_outages = df_outages[data.keys()]

    df_outages['actual_energy'] = (
            df_outages['end_energy'] - df_outages['start_energy']
    )
    df_outages['ci_lower'] = (
        (1 + df_outages['error_lower']) * df_outages['expected_energy']
    )
    df_outages['ci_upper'] = (
        (1 + df_outages['error_upper']) * df_outages['expected_energy']
    )
    df_outages['type'] = np.where(
        df_outages['actual_energy'] < df_outages['ci_lower'],
        'real',
        'comms')
    df_outages['loss'] = np.where(
        df_outages['type'] == 'real',
        df_outages['expected_energy'] - df_outages['actual_energy'],
        0)

    return df_outages
