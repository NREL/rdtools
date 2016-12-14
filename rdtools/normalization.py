''' Energy Normalization Module

This module contains functions to help normalize AC energy output with measured
irradiance in preparation for calculating PV system degradation.
'''

import pandas as pd
import pvlib


def pvwatts_dc_power(poa_global, P_stc, T_cell=None,
                     G_stc=1000, T_stc=25, gamma_pdc=None):
    '''
    PVWatts v5 Module Model: DC power given effective poa irradiance, module
    nameplate power, and cell temperature.

    Note: If either T_cell or gamma_pdc is not provided, the temperature term
          will be ignored.

    Parameters
    ----------
    poa_global: Pandas Series (numeric)
        Total effective plane of array irradiance.
    P_stc: numeric
        Module nameplate power at standard test condition.
    T_cell: Pandas Series (numeric)
        Measured or derived cell temperature [degrees celsius].
        Time series assumed to be same frequency as poa_global.
    G_stc: numeric, default value is 1000
        Reference irradiance at standard test condition [W/m**2].
    T_stc: numeric, default value is 25
        Reference temperature at standard test condition [degrees celsius].
    gamma_pdc: numeric, default is None
        Linear array efficiency temperature coefficient [1 / degree celsius].

    Returns
    -------
    p_dc: Pandas Series (numeric)
        DC power determined by PVWatts v5 equation.
    '''

    dc_power = P_stc * poa_global / G_stc

    if T_cell is not None and gamma_pdc is not None:
        temperature_factor = 1 + gamma_pdc*(T_cell - T_stc)
        dc_power = dc_power * temperature_factor

    return dc_power


def normalize_with_sapm(pvlib_pvsystem, energy, irradiance):
    '''
    Normalize system AC energy output given measured irradiance and
    meteorological data. This method relies on the Sandia Array Performance
    Model (SAPM) to compute the effective DC energy using measured irradiance,
    ambient temperature, and wind speed.

    Energy timeseries and irradiance timeseries can be different granularities.

    Parameters
    ----------
    pvlib_pvsystem: pvlib-python LocalizedPVSystem object
        Object contains orientation, geographic coordinates, equipment constants.
    energy: Pandas Series (numeric)
        Energy time series to be normalized.
    irradiance: Pandas DataFrame (numeric)
        Measured irradiance, ambient temperature, and wind speed.

    Returns
    -------
    normalized_energy: Pandas Series (numeric)
        Energy divided by Sandia Model DC energy.
    '''

    if energy.index.freq is None:
        freq = pd.infer_freq(energy.index)
    else:
        freq = energy.index.freq

    p_dc = sapm_dc_power(pvlib_pvsystem, irradiance)

    energy_dc = p_dc.resample(freq).sum()
    normalized_energy = energy / energy_dc

    return normalized_energy


def sapm_dc_power(pvlib_pvsystem, irradiance):
    '''
    Use Sandia Array Performance Model (SAPM) to compute
    the effective DC power using measured irradiance, ambient temperature, and
    wind speed.

    Parameters
    ----------
    pvlib_pvsystem: pvlib-python LocalizedPVSystem object
        Object contains orientation, geographic coordinates, equipment constants.
    irradiance: Pandas DataFrame (numeric)
        Measured irradiance components, ambient temperature, and wind speed.

    Returns
    -------
    p_dc: Pandas Series (numeric)
        DC power derived using Sandia Array Performance Model.
    '''

    solar_position = pvlib_pvsystem.get_solarposition(irradiance.index)

    total_irradiance = pvlib_pvsystem\
        .get_irradiance(solar_position['zenith'],
                        solar_position['azimuth'],
                        irradiance['DNI'],
                        irradiance['GHI'],
                        irradiance['DHI'])

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
                       wind=irradiance['Wind Speed'],
                       temp=irradiance['Temperature'])

    p_dc = pvlib_pvsystem\
        .pvwatts_dc(g_poa_effective=effective_poa,
                    temp_cell=temp_cell['temp_cell'])

    return p_dc
