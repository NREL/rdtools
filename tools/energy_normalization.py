''' Energy Normalization Module

This module contains functions to help normalize AC energy output with measured
irradiance in preparation for calculating PV system degradation.
'''

import pandas as pd
import pvlib


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

    if energy.index.freq is None:
        freq = pd.infer_freq(energy.index)
    else:
        freq = energy.index.freq

    energy_dc = p_dc.resample(freq).sum()
    normalized_energy = energy / energy_dc

    return normalized_energy
