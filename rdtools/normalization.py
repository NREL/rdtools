''' Energy Normalization Module

This module contains functions to help normalize AC energy output with measured
poa_global in preparation for calculating PV system degradation.
'''

import pandas as pd
import pvlib
import numpy as np
from scipy.optimize import minimize
import warnings


class ConvergenceError(Exception):
    pass


def pvwatts_dc_power(poa_global, P_ref, T_cell=None, G_ref=1000, T_ref=25, gamma_pdc=None):
    '''
    PVWatts v5 Module Model: DC power given effective poa poa_global, module
    nameplate power, and cell temperature. This function differs from the PVLIB
    implementation by allowing cell temperature to be an optional parameter.

    Note: If T_cell or gamma_pdc are omitted, the temperature term will be
          ignored.

    Parameters
    ----------
    poa_global: Pandas Series (numeric)
        Total effective plane of array irradiance.
    P_ref: numeric
        Rated DC power of array in watts
    T_cell: Pandas Series (numeric)
        Measured or derived cell temperature [degrees celsius].
        Time series assumed to be same frequency as poa_global.
    G_ref: numeric, default value is 1000
        Reference irradiance at standard test condition [W/m**2].
    T_ref: numeric, default value is 25
        Reference temperature at standard test condition [degrees celsius].
    gamma_pdc: numeric, default is None
        Linear array efficiency temperature coefficient [1 / degree celsius].

    Note: All series are assumed to be right-labeled, meaning that the recorded value
          at a given timestamp refers ot the previous time interval

    Returns
    -------
    dc_power: Pandas Series (numeric)
        DC power in watts determined by PVWatts v5 equation.
    '''

    dc_power = P_ref * poa_global / G_ref

    if T_cell is not None and gamma_pdc is not None:
        temperature_factor = 1 + gamma_pdc * (T_cell - T_ref)
        dc_power = dc_power * temperature_factor

    return dc_power


def normalize_with_pvwatts(energy, pvwatts_kws):
    '''
    Normalize system AC energy output given measured poa_global and
    meteorological data. This method uses the PVWatts V5 module model.

    Energy timeseries and poa_global timeseries can be different granularities.

    Parameters
    ----------
    energy: Pandas Series (numeric)
        Energy time series to be normalized in watt hours.
        Must be a right-labeled regular time series.
    pvwatts_kws: dictionary
        Dictionary of parameters used in the pvwatts_dc_power function.

        PVWatts Parameters
        ------------------
        poa_global: Pandas Series (numeric)
            Total effective plane of array irradiance.
        P_ref: numeric
            Rated DC power of array in watts.
        T_cell: Pandas Series (numeric)
            Measured or derived cell temperature [degrees celsius].
            Time series assumed to be same frequency as poa_global.
        G_ref: numeric, default value is 1000
            Reference irradiance at standard test condition [W/m**2].
        T_ref: numeric, default value is 25
            Reference temperature at standard test condition [degrees celsius].
        gamma_pdc: numeric, default is None
            Linear array efficiency temperature coefficient [1 / degree celsius].
    Note: All series are assumed to be right-labeled, meaning that the recorded value
          at a given timestamp refers ot the previous time interval

    Returns
    -------
    tuple (normalized_energy, insolation)
        normalized_energy: Pandas Series (numeric)
            Energy divided by PVWatts DC energy.
        insolation: Pandas Series (numeric)
            Insolation associated with each normalized point
    '''

    freq = check_series_frequency(energy, 'energy')

    dc_power = pvwatts_dc_power(**pvwatts_kws)
    irrad = pvwatts_kws['poa_global']

    model_tds, mean_model_td = delta_index(dc_power)
    irrad_tds, mean_irrad_td = delta_index(irrad)
    measure_tds, mean_measure_td = delta_index(energy)

    if mean_model_td <= mean_measure_td:
        energy_dc = dc_power * model_tds
        energy_dc = energy_dc.resample(freq).sum()
        energy_dc = energy_dc.reindex(energy.index, method='nearest')

        insolation = irrad * irrad_tds
        insolation = insolation.resample(freq).sum()
        insolation = insolation.reindex(energy.index, method='nearest')

    elif mean_model_td > mean_measure_td:
        dc_power = dc_power.resample(freq).asfreq()
        dc_power = dc_power.interpolate()
        dc_power = dc_power.reindex(energy.index, method='nearest')
        energy_dc = dc_power * measure_tds  # timedelta is that of measurement due to reindex

        irrad = irrad.resample(freq).asfreq()
        irrad = irrad.interpolate()
        irrad = irrad.reindex(energy.index, method='nearest')
        insolation = irrad * measure_tds  # timedelta is that of measurement due to reindex

    normalized_energy = energy / energy_dc

    return normalized_energy, insolation


def sapm_dc_power(pvlib_pvsystem, met_data):
    '''
    Use Sandia Array Performance Model (SAPM) and PVWatts to compute the
    effective DC power using measured irradiance, ambient temperature, and wind
    speed. Effective irradiance and cell temperature are calculated with SAPM,
    and DC power with PVWatts.

    Parameters
    ----------
    pvlib_pvsystem: pvlib-python LocalizedPVSystem object
        Object contains orientation, geographic coordinates, equipment
        constants (including DC rated power in watts).
    met_data: Pandas DataFrame (numeric)
        Measured irradiance components, ambient temperature, and wind speed.
        Expected met_data DataFrame column names:
            ['DNI', 'GHI', 'DHI', 'Temperature', 'Wind Speed']
    Note: All series are assumed to be right-labeled, meaning that the recorded value
          at a given timestamp refers ot the previous time interval

    Returns
    -------
    tuple (dc_power, effective_poa)
        dc_power: Pandas Series (numeric)
            DC power in watts derived using Sandia Array Performance Model and PVWatts.
        effective_poa: Pandas Series (numeric)
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

    effective_poa = pvlib.pvsystem\
        .sapm_effective_irradiance(poa_direct=total_irradiance['poa_direct'],
                                   poa_diffuse=total_irradiance['poa_diffuse'],
                                   airmass_absolute=airmass_absolute,
                                   aoi=aoi,
                                   module=pvlib_pvsystem.module,
                                   reference_irradiance=1)

    temp_cell = pvlib_pvsystem\
        .sapm_celltemp(irrad=total_irradiance['poa_global'],
                       wind=met_data['Wind Speed'],
                       temp=met_data['Temperature'])

    dc_power = pvlib_pvsystem\
        .pvwatts_dc(g_poa_effective=effective_poa,
                    temp_cell=temp_cell['temp_cell'])

    return dc_power, effective_poa


def normalize_with_sapm(energy, sapm_kws):
    '''
    Normalize system AC energy output given measured met_data and
    meteorological data. This method relies on the Sandia Array Performance
    Model (SAPM) to compute the effective DC energy using measured irradiance,
    ambient temperature, and wind speed.

    Energy timeseries and met_data timeseries can be different granularities.

    Parameters
    ----------
    energy: Pandas Series (numeric)
        Energy time series to be normalized  in watt hours.
        Must be a right-labeled regular time series.
    sapm_kws: dictionary
        Dictionary of parameters required for sapm_dc_power function.

        SAPM Parameters
        ---------------
        pvlib_pvsystem: pvlib-python LocalizedPVSystem object
            Object contains orientation, geographic coordinates, equipment
            constants.
        met_data: Pandas DataFrame (numeric)
            Measured met_data, ambient temperature, and wind speed.
    Note: All series are assumed to be right-labeled, meaning that the recorded value
          at a given timestamp refers ot the previous time interval
    Returns
    -------
    tuple (normalized_energy, insolation)
        normalized_energy: Pandas Series (numeric)
            Energy divided by Sandia Model DC energy.
        insolation: Pandas Series (numeric)
            Insolation associated with each normalized point
    '''

    freq = check_series_frequency(energy, 'energy')

    dc_power, irrad = sapm_dc_power(**sapm_kws)

    model_tds, mean_model_td = delta_index(dc_power)
    irrad_tds, mean_irrad_td = delta_index(irrad)
    measure_tds, mean_measure_td = delta_index(energy)

    if mean_model_td <= mean_measure_td:
        energy_dc = dc_power * model_tds
        energy_dc = energy_dc.resample(freq).sum()
        energy_dc = energy_dc.reindex(energy.index, method='nearest')

        insolation = irrad * irrad_tds
        insolation = insolation.resample(freq).sum()
        insolation = insolation.reindex(energy.index, method='nearest')

    elif mean_model_td > mean_measure_td:
        dc_power = dc_power.resample(freq).asfreq()
        dc_power = dc_power.interpolate()
        dc_power = dc_power.reindex(energy.index, method='nearest')
        energy_dc = dc_power * measure_tds  # timedelta is that of measurment due to reindex

        irrad = irrad.resample(freq).asfreq()
        irrad = irrad.interpolate()
        irrad = irrad.reindex(energy.index, method='nearest')
        insolation = irrad * measure_tds  # timedelta is that of measurment due to reindex

    normalized_energy = energy / energy_dc

    return normalized_energy, insolation


def delta_index(series):
    '''
    Takes a panda series with a DatetimeIndex as input and
    returns (time step sizes, average time step size) in hours
    '''

    if series.index.freq is None:
        # If there is no frequency information, explicily calculate interval sizes
        # Length of each interval calculated by using 'int64' to convert to nanoseconds
        hours = pd.Series(series.index.astype('int64') / (10.0**9 * 3600.0))
        hours.index = series.index
        deltas = hours.diff()
    else:
        # If there is frequency information, pandas shift can be used to gain a meaningful
        # interful for the first element of the timeseries
        # Length of each interval calculated by using 'int64' to convert to nanoseconds
        deltas = (series.index - series.index.shift(-1)).astype('int64') / (10.0**9 * 3600.0)
    return deltas, np.mean(deltas.dropna())


def irradiance_rescale(irrad, modeled_irrad, max_iterations=100, method=None):
    '''
    Attempts to rescale modeled irradiance to match measured irradiance on clear days
    Parameters
    ----------
    irrad: Pandas Series (numeric)
        measured irradiance time series
    modeled_irrad: Pandas Series (numeric)
        modeled irradiance time series
    max_iterations: (int)
        The maximum number of times to attempt rescale optimization, default 100.
        Ignored if method = 'single_opt'
    method: (str)
        The caclulation method to use. 'single_opt' implements the irradiance_rescale of
        rdtools v1.1.3 and earlier. 'iterative' implements a more stable calculation
        that may yield different results from the single_opt method. Default None issues
        a warning then uses the iterative calculation.

    Returns
    -------
    Pandas Series (numeric): resacaled modeled irradiance time series
    '''

    if method is None:
        warnings.warn("The underlying calculations for irradiance_rescale have changed "
                      "which may affect results. To revert to the version of irradiance_rescale "
                      "from rdtools v1.1.3 or earlier, use method = 'single_opt'. ")
        method = 'iterative'

    if method == 'iterative':
        def _rmse(fact):
            "Calculates RMSE with a given rescale fact(or) according to global filt(er)"
            rescaled_modeled_irrad = fact * modeled_irrad
            rmse = np.sqrt(((rescaled_modeled_irrad[filt] - irrad[filt]) ** 2.0).mean())
            return rmse

        def _single_rescale(irrad, modeled_irrad, guess):
            "Optimizes rescale factor once"
            global filt
            csi = irrad / (guess * modeled_irrad)  # clear sky index
            filt = (csi >= 0.8) & (csi <= 1.2) & (irrad > 200)
            min_result = minimize(_rmse, guess, method='Nelder-Mead')

            factor = min_result['x'][0]
            return factor

        # Calculate an initial guess for the rescale factor
        factor = np.percentile(irrad.dropna(), 90) / np.percentile(modeled_irrad.dropna(), 90)

        # Iteratively run the optimization, recalculating the clear sky filter each time
        convergence_threshold = 10**-6
        for i in range(max_iterations):
            prev_factor = factor
            factor = _single_rescale(irrad, modeled_irrad, factor)
            delta = abs(factor - prev_factor)
            if delta < convergence_threshold:
                break

        if delta >= convergence_threshold:
            raise ConvergenceError('Rescale did not converge within max_iterations')
        else:
            return factor * modeled_irrad

    elif method == 'single_opt':
        def _rmse(fact):
            rescaled_modeled_irrad = fact * modeled_irrad
            csi = irrad / rescaled_modeled_irrad
            filt = (csi >= 0.8) & (csi <= 1.2)
            rmse = np.sqrt(((rescaled_modeled_irrad[filt] - irrad[filt]) ** 2.0).mean())
            return rmse

        guess = np.percentile(irrad.dropna(), 90) / np.percentile(modeled_irrad.dropna(), 90)
        min_result = minimize(_rmse, guess, method='Nelder-Mead')
        factor = min_result['x'][0]

        out_irrad = factor * modeled_irrad
        return out_irrad

    else:
        raise ValueError('Invalid method')


def check_series_frequency(series, series_description):
    '''Returns the inferred frequency of a pandas series, raises ValueError
    using series_description if it can't. series_description should be a string'''

    if series.index.freq is None:
        freq = pd.infer_freq(series.index)
        if freq is None:
            error_string = ('Could not infer frequency of ' + series_description +
                            ', which must be a regular time series')
            raise ValueError(error_string)
    else:
        freq = series.index.freq
    return freq
