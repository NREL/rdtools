''' Energy Normalization Module

This module contains functions to help normalize AC energy output with measured
poa_global in preparation for calculating PV system degradation.
'''

import pandas as pd
import pvlib
import numpy as np


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
        Rated DC power of array.
        T_cell: Pandas Series (numeric)
        Measured or derived cell temperature [degrees celsius].
        Time series assumed to be same frequency as poa_global.
    G_ref: numeric, default value is 1000
        Reference irradiance at standard test condition [W/m**2].
    T_ref: numeric, default value is 25
        Reference temperature at standard test condition [degrees celsius].
    gamma_pdc: numeric, default is None
        Linear array efficiency temperature coefficient [1 / degree celsius].

    Note: All series are assumed to be right-labeled

    Returns
    -------
    dc_power: Pandas Series (numeric)
        DC power determined by PVWatts v5 equation.
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
        Energy time series to be normalized in power * hours.
    pvwatts_kws: dictionary
        Dictionary of parameters used in the pvwatts_dc_power function.

        PVWatts Parameters
        ------------------
        poa_global: Pandas Series (numeric)
            Total effective plane of array irradiance.
        P_ref: numeric
            Rated DC power of array.
        T_cell: Pandas Series (numeric)
            Measured or derived cell temperature [degrees celsius].
            Time series assumed to be same frequency as poa_global.
        G_ref: numeric, default value is 1000
            Reference irradiance at standard test condition [W/m**2].
        T_ref: numeric, default value is 25
            Reference temperature at standard test condition [degrees celsius].
        gamma_pdc: numeric, default is None
            Linear array efficiency temperature coefficient [1 / degree celsius].
    Note: All series are assumed to be right-labeled

    Returns
    -------
    normalized_energy: Pandas Series (numeric)
        Energy divided by PVWatts DC energy.
    insolation:: Pandas Series (numeric)
        Insolation associated with each normalized point
    '''

    if energy.index.freq is None:
        freq = pd.infer_freq(energy.index)
    else:
        freq = energy.index.freq

    dc_power = pvwatts_dc_power(**pvwatts_kws)
    irrad = pvwatts_kws['poa_global']

    # Length of each right labeled interval
    model_tds = (dc_power.index - dc_power.index.shift(-1)).astype('int64') / (10.0**9 * 3600.0)
    irrad_tds = (irrad.index - irrad.index.shift(-1)).astype('int64') / (10.0**9 * 3600.0)
    measure_tds = (energy.index - energy.index.shift(-1)).astype('int64') / (10.0**9 * 3600.0)

    mean_model_td = np.mean(model_tds)
    mean_measure_td = np.mean(measure_tds)

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
        constants.
    met_data: Pandas DataFrame (numeric)
        Measured irradiance components, ambient temperature, and wind speed.
        Expected met_data DataFrame column names:
            ['DNI', 'GHI', 'DHI', 'Temperature', 'Wind Speed']
    Note: All series are assumed to be right-labeled

    Returns
    -------
    dc_power: Pandas Series (numeric)
        DC power derived using Sandia Array Performance Model.
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
        Energy time series to be normalized  in power * hours.
    sapm_kws: dictionary
        Dictionary of parameters required for sapm_dc_power function.

        SAPM Parameters
        ---------------
        pvlib_pvsystem: pvlib-python LocalizedPVSystem object
            Object contains orientation, geographic coordinates, equipment
            constants.
        met_data: Pandas DataFrame (numeric)
            Measured met_data, ambient temperature, and wind speed.
    Note: All series are assumed to be right-labeled
    Returns
    -------
    normalized_energy: Pandas Series (numeric)
        Energy divided by Sandia Model DC energy.
    insolation:: Pandas Series (numeric)
        Insolation associated with each normalized point
    '''

    if energy.index.freq is None:
        freq = pd.infer_freq(energy.index)
    else:
        freq = energy.index.freq

    dc_power, irrad = sapm_dc_power(**sapm_kws)

    # Length of each right labeled interval
    model_tds = (dc_power.index - dc_power.index.shift(-1)).astype('int64') / (10.0**9 * 3600.0)
    irrad_tds = (irrad.index - irrad.index.shift(-1)).astype('int64') / (10.0**9 * 3600.0)
    measure_tds = (energy.index - energy.index.shift(-1)).astype('int64') / (10.0**9 * 3600.0)

    mean_model_td = np.mean(model_tds)
    mean_measure_td = np.mean(measure_tds)

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
