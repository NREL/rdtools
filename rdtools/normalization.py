'''Functions for normalizing, rescaling, and regularizing PV system data.'''

import pandas as pd
import pvlib
import numpy as np
from scipy.optimize import minimize
import warnings
from rdtools._deprecation import deprecated


class ConvergenceError(Exception):
    '''Rescale optimization did not converge'''
    pass


def normalize_with_expected_power(pv, power_expected, poa_global,
                                  pv_input='power'):
    '''
    Normalize PV power or energy based on expected PV power.

    Parameters
    ----------
    pv : pd.Series
        Right-labeled time series PV energy or power. If energy, should *not*
        be cumulative, but only for preceding time step. Type (energy or power)
        must be specified in the ``pv_input`` parameter.
    power_expected : pd.Series
        Right-labeled time series of expected PV power. (Note: Expected energy
        is not supported.)
    poa_global : pd.Series
        Right-labeled time series of plane-of-array irradiance associated with
        ``expected_power``
    pv_input : {'power' or 'energy'}
        Specifies the type of input used for ``pv`` parameter. Default: 'power'

    Returns
    -------
    energy_normalized : pd.Series
        Energy normalized based on ``power_expected``
    insolation : pd.Series
        Insolation associated with each normalized point

    '''

    freq = _check_series_frequency(pv, 'pv')

    if pv_input == 'power':
        energy = energy_from_power(pv, freq, power_type='right_labeled')
    elif pv_input == 'energy':
        energy = pv.copy()
        energy.name = 'energy_Wh'
    else:
        raise ValueError("Unexpected value for pv_input. pv_input should be 'power' or 'energy'.")

    model_tds, mean_model_td = _delta_index(power_expected)
    measure_tds, mean_measure_td = _delta_index(energy)

    # Case in which the model less frequent than the measurements
    if mean_model_td > mean_measure_td:
        power_expected = interpolate(power_expected, pv.index)
        poa_global = interpolate(poa_global, pv.index)

    energy_expected = energy_from_power(power_expected, freq, power_type='right_labeled')
    insolation = energy_from_power(poa_global, freq, power_type='right_labeled')

    energy_normalized = energy / energy_expected

    index_union = energy_normalized.index.union(insolation.index)
    energy_normalized = energy_normalized.reindex(index_union)
    insolation = insolation.reindex(index_union)

    return energy_normalized, insolation


def pvwatts_dc_power(poa_global, power_dc_rated, temperature_cell=None,
                     poa_global_ref=1000, temperature_cell_ref=25,
                     gamma_pdc=None):
    '''
    PVWatts v5 Module Model: DC power given effective poa poa_global, module
    nameplate power, and cell temperature. This function differs from the PVLIB
    implementation by allowing cell temperature to be an optional parameter.

    Parameters
    ----------
    poa_global : pd.Series
        Total effective plane of array irradiance.
    power_dc_rated : float
        Rated DC power of array in watts
    temperature_cell : pd.Series, optional
        Measured or derived cell temperature [degrees Celsius].
        Time series assumed to be same frequency as ``poa_global``.
        If omitted, the temperature term will be ignored.
    poa_global_ref : float, default 1000
        Reference irradiance at standard test condition [W/m**2].
    temperature_cell_ref : float, default 25
        Reference temperature at standard test condition [degrees Celsius].
    gamma_pdc : float, default None
        Linear array efficiency temperature coefficient [1 / degree Celsius].
        If omitted, the temperature term will be ignored.

    Note
    ----
    All series are assumed to be right-labeled, meaning that the recorded
    value at a given timestamp refers to the previous time interval

    Returns
    -------
    power_dc : pd.Series
        DC power in watts determined by PVWatts v5 equation.
    '''

    power_dc = power_dc_rated * poa_global / poa_global_ref

    if temperature_cell is not None and gamma_pdc is not None:
        temperature_factor = (
            1 + gamma_pdc * (temperature_cell - temperature_cell_ref)
        )
        power_dc = power_dc * temperature_factor

    return power_dc


def normalize_with_pvwatts(energy, pvwatts_kws):
    '''
    Normalize system AC energy output given measured poa_global and
    meteorological data. This method uses the PVWatts V5 module model.

    Energy timeseries and poa_global timeseries can be different granularities.

    Parameters
    ----------
    energy : pd.Series
        Energy time series to be normalized in watt hours.
        Must be a right-labeled regular time series.
    pvwatts_kws : dict
        Dictionary of parameters used in the pvwatts_dc_power function.  See
        Other Parameters.

    Other Parameters
    ------------------
    poa_global : pd.Series
        Total effective plane of array irradiance.
    power_dc_rated : float
        Rated DC power of array in watts
    temperature_cell : pd.Series, optional
        Measured or derived cell temperature [degrees Celsius].
        Time series assumed to be same frequency as `poa_global`.
        If omitted, the temperature term will be ignored.
    poa_global_ref : float, default 1000
        Reference irradiance at standard test condition [W/m**2].
    temperature_cell_ref : float, default 25
        Reference temperature at standard test condition [degrees Celsius].
    gamma_pdc : float, default None
        Linear array efficiency temperature coefficient [1 / degree Celsius].
        If omitted, the temperature term will be ignored.

    Note
    ----
    All series are assumed to be right-labeled, meaning that the recorded
    value at a given timestamp refers to the previous time interval

    Returns
    -------
    energy_normalized : pd.Series
        Energy divided by PVWatts DC energy.
    insolation : pd.Series
        Insolation associated with each normalized point
    '''

    power_dc = pvwatts_dc_power(**pvwatts_kws)
    irrad = pvwatts_kws['poa_global']

    energy_normalized, insolation = normalize_with_expected_power(energy, power_dc, irrad, pv_input='energy')

    return energy_normalized, insolation


@deprecated(since='2.0.0', removal='3.0.0',
            alternative='normalize_with_expected_power')
def sapm_dc_power(pvlib_pvsystem, met_data):
    '''
    Use Sandia Array Performance Model (SAPM) and PVWatts to compute the
    effective DC power using measured irradiance, ambient temperature, and wind
    speed. Effective irradiance and cell temperature are calculated with SAPM,
    and DC power with PVWatts.

    Parameters
    ----------
    pvlib_pvsystem : pvlib-python LocalizedPVSystem object
        Object contains orientation, geographic coordinates, equipment
        constants (including DC rated power in watts).  The object must also
        specify either the ``temperature_model_parameters`` attribute or both
        ``racking_model`` and ``module_type`` attributes to infer the temperature model parameters.
    met_data : pd.DataFrame
        Measured irradiance components, ambient temperature, and wind speed.
        Expected met_data DataFrame column names:
        ['DNI', 'GHI', 'DHI', 'Temperature', 'Wind Speed']

    Note
    ----
    All series are assumed to be right-labeled, meaning that the recorded
    value at a given timestamp refers to the previous time interval

    Returns
    -------
    power_dc : pd.Series
        DC power in watts derived using Sandia Array Performance Model and
        PVWatts.
    effective_poa : pd.Series
        Effective irradiance calculated with SAPM
    '''

    solar_position = pvlib_pvsystem.get_solarposition(met_data.index)

    total_irradiance = pvlib_pvsystem\
        .get_irradiance(solar_position['zenith'],
                        solar_position['azimuth'],
                        met_data['DNI'],
                        met_data['GHI'],
                        met_data['DHI'])

    aoi = pvlib_pvsystem.get_aoi(solar_position['zenith'],
                                 solar_position['azimuth'])

    airmass = pvlib_pvsystem\
        .get_airmass(solar_position=solar_position, model='kastenyoung1989')
    airmass_absolute = airmass['airmass_absolute']

    effective_irradiance = pvlib.pvsystem\
        .sapm_effective_irradiance(poa_direct=total_irradiance['poa_direct'],
                                   poa_diffuse=total_irradiance['poa_diffuse'],
                                   airmass_absolute=airmass_absolute,
                                   aoi=aoi,
                                   module=pvlib_pvsystem.module)

    temp_cell = pvlib_pvsystem\
        .sapm_celltemp(total_irradiance['poa_global'],
                       met_data['Temperature'],
                       met_data['Wind Speed'])

    power_dc = pvlib_pvsystem\
        .pvwatts_dc(g_poa_effective=effective_irradiance,
                    temp_cell=temp_cell)

    return power_dc, effective_irradiance


@deprecated(since='2.0.0', removal='3.0.0',
            alternative='normalize_with_expected_power')
def normalize_with_sapm(energy, sapm_kws):
    '''
    Normalize system AC energy output given measured met_data and
    meteorological data. This method relies on the Sandia Array Performance
    Model (SAPM) to compute the effective DC energy using measured irradiance,
    ambient temperature, and wind speed.

    Energy timeseries and met_data timeseries can be different granularities.

    Parameters
    ----------
    energy : pd.Series
        Energy time series to be normalized  in watt hours.
        Must be a right-labeled regular time series.
    sapm_kws : dict
        Dictionary of parameters required for sapm_dc_power function. See
        Other Parameters.

    Other Parameters
    ---------------
    pvlib_pvsystem : pvlib-python LocalizedPVSystem object
        Object contains orientation, geographic coordinates, equipment
        constants (including DC rated power in watts).  The object must also
        specify either the ``temperature_model_parameters`` attribute or both
        ``racking_model`` and ``module_type`` to infer the model parameters.
    met_data : pd.DataFrame
        Measured met_data, ambient temperature, and wind speed.  Expected
        column names are ['DNI', 'GHI', 'DHI', 'Temperature', 'Wind Speed']

    Note
    ----
    All series are assumed to be right-labeled, meaning that the recorded
    value at a given timestamp refers to the previous time interval

    Returns
    -------
    energy_normalized : pd.Series
        Energy divided by Sandia Model DC energy.
    insolation : pd.Series
        Insolation associated with each normalized point
    '''

    power_dc, irrad = sapm_dc_power(**sapm_kws)

    energy_normalized, insolation = normalize_with_expected_power(energy, power_dc, irrad, pv_input='energy')

    return energy_normalized, insolation


def _delta_index(series):
    '''
    Takes a pandas series with a DatetimeIndex as input and
    returns (time step sizes, average time step size) in hours

    Parameters
    ----------
    series : pd.Series
        A pandas timeseries

    Returns
    -------
    deltas : pd.Series
        A timeseries representing the timestep sizes of ``series``
    mean : float
        The average timestep
    '''

    if series.index.freq is None:
        # If there is no frequency information, explicitly calculate interval
        # sizes. Length of each interval calculated by using 'int64' to convert
        # to nanoseconds.
        hours = pd.Series(series.index.astype('int64') / (10.0**9 * 3600.0))
        hours.index = series.index
        deltas = hours.diff()
    else:
        # If there is frequency information, pandas shift can be used to gain
        # a meaningful interval for the first element of the timeseries
        # Length of each interval calculated by using 'int64' to convert to
        # nanoseconds.
        deltas = (series.index - series.index.shift(-1)).astype('int64') / \
                 (10.0**9 * 3600.0)
    return deltas, np.mean(deltas.dropna())


delta_index = deprecated('2.0.0', removal='3.0.0')(_delta_index)


def irradiance_rescale(irrad, irrad_sim, max_iterations=100,
                       method='iterative', convergence_threshold=1e-6):
    '''
    Attempt to rescale modeled irradiance to match measured irradiance on
    clear days.

    Parameters
    ----------
    irrad : pd.Series
        measured irradiance time series
    irrad_sim : pd.Series
        modeled/simulated irradiance time series
    max_iterations : int, default 100
        The maximum number of times to attempt rescale optimization.
        Ignored if ``method = 'single_opt'``
    method : str, default 'iterative'
        The calculation method to use. 'single_opt' implements the
        irradiance_rescale of rdtools v1.1.3 and earlier. 'iterative'
        implements a more stable calculation that may yield different results
        from the single_opt method.
    convergence_threshold : float, default 1e-6
        The acceptable iteration-to-iteration scaling factor difference to
        determine convergence.  If the threshold is not reached after
        ``max_iterations``, raise
        :py:exc:`rdtools.normalization.ConvergenceError`.
        Must be greater than zero.  Only used if ``method=='iterative'``.

    Returns
    -------
    pd.Series
        Rescaled modeled irradiance time series
    '''

    if method == 'iterative':
        def _rmse(fact):
            """
            Calculates RMSE with a given rescale fact(or) according to global
            filt(er)
            """
            rescaled_irrad_sim = fact * irrad_sim
            difference = rescaled_irrad_sim[filt] - irrad[filt]
            rmse = np.sqrt((difference**2.0).mean())
            return rmse

        def _single_rescale(irrad, irrad_sim, guess):
            "Optimizes rescale factor once"
            global filt
            csi = irrad / (guess * irrad_sim)  # clear sky index
            filt = (csi >= 0.8) & (csi <= 1.2) & (irrad > 200)
            min_result = minimize(_rmse, guess, method='Nelder-Mead')

            factor = min_result['x'][0]
            return factor

        # Calculate an initial guess for the rescale factor
        factor = (np.percentile(irrad.dropna(), 90) /
                  np.percentile(irrad_sim.dropna(), 90))
        prev_factor = 1.0

        # Iteratively run the optimization,
        # recalculating the clear sky filter each time
        iteration = 0
        while abs(factor - prev_factor) > convergence_threshold:
            iteration += 1
            if iteration > max_iterations:
                msg = 'Rescale did not converge within max_iterations'
                raise ConvergenceError(msg)
            prev_factor = factor
            factor = _single_rescale(irrad, irrad_sim, factor)

        return factor * irrad_sim

    elif method == 'single_opt':
        def _rmse(fact):
            rescaled_irrad_sim = fact * irrad_sim
            csi = irrad / rescaled_irrad_sim
            filt = (csi >= 0.8) & (csi <= 1.2)
            difference = rescaled_irrad_sim[filt] - irrad[filt]
            rmse = np.sqrt((difference**2.0).mean())
            return rmse

        guess = np.percentile(irrad.dropna(), 90) / \
            np.percentile(irrad_sim.dropna(), 90)
        min_result = minimize(_rmse, guess, method='Nelder-Mead')
        factor = min_result['x'][0]

        out_irrad = factor * irrad_sim
        return out_irrad

    else:
        raise ValueError('Invalid method')


def _check_series_frequency(series, series_description):
    '''
    Returns the inferred frequency of a pandas series, raises ValueError
    using ``series_description`` if it can't.

    Parameters
    ----------
    series : pd.Series
        The timeseries to infer the frequency of.
    series_description : str
        The description to use when raising an error.

    Returns
    -------
    freq : pandas Offsets string
        The inferred index frequency
    '''

    if series.index.freq is None:
        freq = pd.infer_freq(series.index)
        if freq is None:
            error_string = ('Could not infer frequency of ' +
                            series_description +
                            ', which must be a regular time series')
            raise ValueError(error_string)
    else:
        freq = series.index.freq
    return freq


check_series_frequency = deprecated('2.0.0', removal='3.0.0')(_check_series_frequency)


def _t_step_nanoseconds(time_series):
    '''
    return a series of right labeled differences in the index of time_series
    in nanoseconds
    '''
    t_steps = np.diff(time_series.index.astype('int64')).astype('float')
    t_steps = np.insert(t_steps, 0, np.nan)
    t_steps = pd.Series(index=time_series.index, data=t_steps)
    return t_steps


def energy_from_power(power, target_frequency=None, max_timedelta=None, power_type='right_labeled'):
    '''
    Returns a regular right-labeled energy time series in units of Wh per
    interval from a power time series. For instantaneous timeseries, a
    trapezoidal sum is used. For right labeled time series, a rectangular sum
    is used. NaN is filled where the gap between input data points exceeds
    ``max_timedelta``. Power_series should
    be given in Watts.

    Parameters
    ----------
    power : pd.Series
        Time series of power in Watts
    target_frequency : DatetimeOffset or frequency string, default None
        The frequency of the energy time series to be returned.
        If omitted, use the median timestep of ``power``, or if ``power`` has
        fewer than two elements, use ``power.index.freq`.
    max_timedelta : pd.Timedelta, default None
        The maximum allowed gap between power measurements. If the gap between
        consecutive power measurements exceeds ``max_timedelta``, NaN will be
        returned for that interval. If omitted, ``max_timedelta`` is set
        internally to the median time delta in ``power``. Ignored when ``power``
        has fewer than two elements.
    power_type : {'right_labeled', 'instantaneous'}
        The labeling convention used in power. Default: 'right_labeled'

    Returns
    -------
    pd.Series
        right-labeled energy in Wh per interval
    '''

    if not isinstance(power.index, pd.DatetimeIndex):
        raise ValueError('power must be a pandas series with a '
                         'DatetimeIndex')

    if len(power) <= 1:
        # just one value, doesn't make sense to interpolate or trapz aggregate.
        # use the index frequency to determine the appropriate timescale
        if power_type == 'instantaneous':
            raise ValueError("power_type='instantaneous' is incompatible with single element "
                             "power. Use power_type='right-labeled'")
        if target_frequency is None:
            if power.index.freq is None:
                raise ValueError('Could not determine period of input power')

            target_frequency = power.index.freq
        # just raise if it's a non-fixed frequency
        interval_length_ns = \
            pd.tseries.frequencies.to_offset(target_frequency).nanos

        energy = power * interval_length_ns / 1e9 / 3600  # ns to s to h
        energy.name = 'energy_Wh'
        return energy

    t_steps = _t_step_nanoseconds(power)
    median_step_ns = t_steps.median()

    if target_frequency is None:
        # 'N' is the Pandas offset alias for ns
        target_frequency = str(int(median_step_ns)) + 'N'

    if max_timedelta is None:
        max_interval_nanoseconds = median_step_ns
    else:
        max_interval_nanoseconds = max_timedelta.total_seconds() * 10.0**9
    # set max_timedelta for use in interpolate and _aggregate
    max_timedelta = pd.to_timedelta(f'{max_interval_nanoseconds} nanos')
    try:
        freq_interval_size_ns = \
            pd.tseries.frequencies.to_offset(target_frequency).nanos
    except ValueError as e:
        if 'is a non-fixed frequency' in str(e):
            temp_ind = pd.date_range(power.index[0],
                                     power.index[-1],
                                     freq=target_frequency)
            temp_series = pd.Series(data=1, index=temp_ind)
            temp_diffs = _t_step_nanoseconds(temp_series)
            freq_interval_size_ns = temp_diffs.median()
        else:
            raise

    if freq_interval_size_ns <= median_step_ns:
        power = interpolate(power, target_frequency, max_timedelta)
    energy = _aggregate(power, target_frequency, max_timedelta, power_type)

    # Set the frequency if we can
    try:
        energy.index.freq = pd.infer_freq(energy.index)
    except ValueError:
        pass

    # enforce max_timedelta
    t_steps = t_steps.reindex(energy.index, method='backfill')
    energy.loc[t_steps > max_interval_nanoseconds] = np.nan

    energy.name = 'energy_Wh'

    return energy


def _aggregate(time_series, target_frequency, max_timedelta, series_type):
    '''
    Returns a right-labeled series with frequency target_frequency generated by
    aggregating ``time_series`` (in units of hours). For instantaneous timeseries,
    a trapezoidal sum is used. For right labeled time series, a rectangular sum
    is used. If any interval in ``time_series`` is greater than ``max_timedelta``,
    it is omitted from the sum.

    Parameters
    ----------
    time_series : pd.Series
    target_frequency : DatetimeOffset, or frequency string
        The frequency of the accumulated series to be returned.
    max_timedelta : pd.Timedelta, default None
        The maximum allowed gap between power measurements. If the gap between
        consecutive power measurements exceeds ``max_timedelta``, no energy value
        will be returned for that interval.
    series_type : {'right_labeled', 'instantaneous'}
        The labeling convention of time_series


    Returns
    -------
    pd.Series
        right-labeled aggregated time_series in _*hours per interval
    '''

    # series that has same index as desired output
    output_dummy = time_series.resample(target_frequency,
                                        closed='right',
                                        label='right').sum()

    union_index = time_series.index.union(output_dummy.index)
    time_series = time_series.dropna()

    values = time_series.values

    # Identify gaps (including from nans) bigger than max_time_delta
    timestamps = time_series.index.astype('int64').values
    timestamps = pd.Series(timestamps, index=time_series.index)
    t_diffs = timestamps.diff()
    # Keep track of the gap size but with refilled NaNs and new
    # timestamps from target freq
    t_diffs = t_diffs.reindex(union_index, method='bfill')

    max_interval_nanoseconds = max_timedelta.total_seconds() * 10.0**9

    gap_mask = t_diffs > max_interval_nanoseconds

    time_series = time_series.reindex(union_index)
    t_diffs = np.diff(time_series.index.astype('int64').values)
    t_diffs_hours = t_diffs / 10**9 / 3600.0
    if series_type == 'instantaneous':
        # interpolate with trapz sum
        time_series = time_series.interpolate(method='time')
        time_series[gap_mask] = np.nan
        values = time_series.values
        series_sum = (values[1:] + values[:-1]) / 2 * t_diffs_hours
    elif series_type == 'right_labeled':
        # bfill and rectangular sum
        time_series = time_series.bfill()
        time_series[gap_mask] = np.nan
        values = time_series.values
        series_sum = values[1:] * t_diffs_hours
    else:
        raise ValueError("series_type must be either 'instantaneous' or 'right_labeled', "
                         "not '{}'".format(series_type))

    series_sum = pd.Series(data=series_sum, index=time_series.index[1:])

    aggregated = series_sum.resample(target_frequency,
                                     closed='right',
                                     label='right').sum(min_count=1)

    return aggregated


def _interpolate_series(time_series, target_index, max_timedelta=None,
                        warning_threshold=0.1):
    '''
    Returns an interpolation of time_series onto target_index, NaN is returned
    for times associated with gaps in time_series longer than ``max_timedelta``.

    Parameters
    ----------
    time_series : pd.Series
        Original values to be used in generating the interpolation
    target_index : pd.DatetimeIndex
        the index onto which the interpolation is to be made
    max_timedelta : pd.Timedelta, default None
        The maximum allowed gap between values in time_series. Times associated
        with gaps longer than ``max_timedelta`` are excluded from the output. If
        omitted, ``max_timedelta`` is set internally to two times the median
        time delta in ``time_series``.
    warning_threshold : float, default 0.1
        The fraction of data exclusion above which a warning is raised. With
        the default value of 0.1, a warning will be raised if the fraction
        of data excluded because of data gaps longer than ``max_timedelta`` is
        above than 10%.

    Returns
    -------
    pd.Series

    Note
    ----
    Timezone information in the DatetimeIndexes is handled automatically,
    however both ``time_series`` and ``target_index`` should be time zone aware or
    they should both be time zone naive.

    '''

    # note the name of the input, so we can use it for the output
    original_name = time_series.name

    # copy, rename, and make df from input
    time_series = time_series.copy()
    time_series.name = 'data'
    df = pd.DataFrame(time_series)
    df = df.dropna()

    # convert to integer index and calculate the size of gaps in input
    timestamps = df.index.astype('int64')
    df['timestamp'] = timestamps
    df['gapsize_ns'] = df['timestamp'].diff()
    df.index = timestamps

    valid_indput_index = df.index.copy()

    if max_timedelta is None:
        max_interval_nanoseconds = 2 * df['gapsize_ns'].median()
    else:
        max_interval_nanoseconds = max_timedelta.total_seconds() * 10.0**9

    fraction_excluded = (df['gapsize_ns'] > max_interval_nanoseconds).mean()
    if fraction_excluded > warning_threshold:
        warnings.warn("Fraction of excluded data "
                      f"({100*fraction_excluded:0.02f}%) "
                      "exceeded threshold",
                      UserWarning)

    # put data on index that includes both original and target indicies
    target_timestamps = target_index.astype('int64')
    union_index = df.index.append(target_timestamps)
    union_index = union_index.drop_duplicates(keep='first')
    df = df.reindex(union_index)
    df = df.sort_index()

    # calculate the gap size in the original data (timestamps)
    df['gapsize_ns'] = df['gapsize_ns'].fillna(method='bfill')
    df.loc[valid_indput_index, 'gapsize_ns'] = 0

    # perform the interpolation when the max gap size criterion is satisfied
    df_valid = df[df['gapsize_ns'] <= max_interval_nanoseconds].copy()
    df_valid['interpolated_data'] = \
        df_valid['data'].interpolate(method='index')

    df['interpolated_data'] = df_valid['interpolated_data']

    out = pd.Series(df['interpolated_data'])
    out = out.loc[target_timestamps]
    out.name = original_name
    out.index = pd.to_datetime(out.index, utc=True).tz_convert(target_index.tz)
    out = out.reindex(target_index)

    return out


def interpolate(time_series, target, max_timedelta=None, warning_threshold=0.1):
    '''
    Returns an interpolation of time_series, excluding times associated with
    gaps in each column of time_series longer than max_timedelta; NaNs are
    returned within those gaps.

    Parameters
    ----------
    time_series : pd.Series, pd.DataFrame
        Original values to be used in generating the interpolation
    target : pd.DatetimeIndex, DatetimeOffset, or frequency string

        * If DatetimeIndex: the index onto which the interpolation is to be
          made
        * If DatetimeOffset or frequency string: the frequency at which to
          resample and interpolate
    max_timedelta : pd.Timedelta, default None
        The maximum allowed gap between values in ``time_series``. Times
        associated with gaps longer than ``max_timedelta`` are excluded from the
        output. If omitted, ``max_timedelta`` is set internally to two times
        the median time delta in ``time_series``.
    warning_threshold : float, default 0.1
        The fraction of data exclusion above which a warning is raised. With
        the default value of 0.1, a warning will be raised if the fraction
        of data excluded because of data gaps longer than ``max_timedelta`` is
        above than 10%.

    Returns
    -------
    pd.Series or pd.DataFrame (matching type of time_series) with DatetimeIndex

    Note
    ----
    Timezone information in the DatetimeIndexes is handled automatically,
    however both ``time_series`` and ``target`` should be time zone aware or they
    should both be time zone naive.
    '''

    if isinstance(target, pd.DatetimeIndex):
        target_index = target
    elif isinstance(target, (pd.tseries.offsets.DateOffset, str)):
        target_index = pd.date_range(time_series.index.min(),
                                     time_series.index.max(),
                                     freq=target)

    if (time_series.index.tz is None) ^ (target_index.tz is None):
        raise ValueError('Either time_series or target is time-zone aware but '
                         'the other is not. Both must be time-zone aware or '
                         'both must be time-zone naive.')

    if isinstance(time_series, pd.Series):
        out = _interpolate_series(time_series, target_index, max_timedelta,
                                 warning_threshold)
    elif isinstance(time_series, pd.DataFrame):
        out_list = []
        for col in time_series.columns:
            ts = time_series[col]
            series = _interpolate_series(ts, target_index, max_timedelta,
                                        warning_threshold)
            out_list.append(series)
        out = pd.concat(out_list, axis=1)
    else:
        raise ValueError('time_series must be a Pandas Series or DataFrame')

    return out
