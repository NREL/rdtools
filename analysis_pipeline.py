from datetime import timedelta
import pandas as pd
import numpy as np
import pvlib
import rdtools
from timezonefinder import TimezoneFinder

import logging
import sys
logging.root.handlers = []
logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s| %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
    
def get_timezone(latitude, longitude):
    """
    returns (str) timezone
    """
    if not TimezoneFinder:
        logging.info("Could not lookup timezone for location {}, {}".format(latitude, longitude))
        return None

    tf = TimezoneFinder()
    tz = tf.timezone_at(lat=latitude, lng=longitude)
    logging.info("timezone for location {}, {} is {}".format(latitude, longitude, tz))
    
    return tz

def rename_columns(df):
    """
    Best guess at renaming columns
    """
    try:
        df.columns = [col.decode('utf-8') for col in df.columns]
    except AttributeError:
        pass  # Python 3 strings are already unicode literals

    columns = [name.lower() for name in df.columns]

    rename_dictionary = {}
    for name in columns:
        if 'power' in name: 
            rename_dictionary[name] = 'power'
        elif 'temperature' in name or 'tamb' in name:
            rename_dictionary[name] = 'Tamb'
        elif 'wind' in name and 'speed' in name:
            rename_dictionary[name] = 'wind_speed'
        elif 'ghi' in name or ('global' in name and 'horizontal' in name):
            rename_dictionary[name] = 'ghi'
        elif 'dhi' in name or ('diffuse' in name and 'horizontal' in name):
            rename_dictionary[name] = 'dhi'
        elif 'datatime' in name or 'timestamp' in name:
            rename_dictionary[name] = 'timestamp'

    df.columns = columns
    df.rename(columns = rename_dictionary, inplace=True)
    df = df[rename_dictionary.values()]

    return df

class AnalysisPipeline(object):
    def __init__(self, input_filename, system_metadata, clearsky = True):
        '''
        input_filename: csv with power time-series
        system_metdata: dictionary with the following keys: 
                        - systemid   (string. unique identifier) 
                        - latitude
                        - longitude
                        - tilt       (degrees)
                        - azimuth    (degrees)
                        - pdc        (DC capacity in Watts)
        '''

        self.system_metadata = system_metadata
        self.df = rename_columns(pd.read_csv(input_filename))
        self.clearsky = clearsky

    def _process_data(self):

        self.timezone = get_timezone( self.system_metadata['latitude'], 
                                      self.system_metadata['longitude'] )

        self.df.index = pd.to_datetime(self.df.timestamp)
        self.df.index = self.df.index.tz_localize(self.timezone, ambiguous='infer')

        # There is some missing data, but we can infer the frequency from the first several data points
        # And then set the frequency of the dataframe
        freq = pd.infer_freq(self.df.index[:10])
        logging.info('Inferred frequency = {}'.format(freq))
        self.df = self.df.resample(freq).median()

        # Convert power from kilowatts to watts
        self.df['power'] = self.df.power * 1000.0 

        # Calculate energy yield in Wh
        self.df['energy'] = self.df.power * pd.to_timedelta(self.df.power.index.freq).total_seconds()/(3600.0)

        logging.info("setting poa and cell temperature from pvlib")
        poa, cell_temperature = self._get_variables_from_pvlib(self.df, clearsky_variables = False)

        self.df['poa'] = poa.values

        normalized_energy, insolation = self._normalize(self.df.energy, self.df.poa, cell_temperature)
        self.df['normalized_energy'] = normalized_energy.values
        self.df['insolation'] = insolation.values

        df = self._remove_outliers(normalized_energy, self.df.power, self.df.poa, cell_temperature)

        logging.info('removed outliers')

        clearsky_df = None
        if self.clearsky:
            logging.info("setting clearsky poa and cell temperature from pvlib")
            clearsky_poa, clearsky_cell_temperature = self._get_variables_from_pvlib(self.df, clearsky_variables = True)
            clearsky_normalized_energy, clearsky_insolation = self._normalize(self.df.energy, clearsky_poa, clearsky_cell_temperature)

            self.df['clearsky_normalized_energy'] = clearsky_normalized_energy
            self.df['clearsky_insolation'] = clearsky_insolation

            clearsky_df = self._remove_outliers(clearsky_normalized_energy, self.df.power, clearsky_poa, clearsky_cell_temperature)

        return df, clearsky_df


    def _get_poa_and_Tcell(self, dhi, dni, ghi, Tamb, wind_speed, solar_zenith, solar_azimuth):

        sky = pvlib.irradiance.isotropic(self.system_metadata['tilt'], dhi)
        beam = pvlib.irradiance.beam_component(self.system_metadata['tilt'], 
                                               self.system_metadata['azimuth'], 
                                               solar_zenith, solar_azimuth, dni)
        
        poa = beam + sky
        if 'poa' in self.df.columns:
            poa = rdtools.irradiance_rescale(poa, self.df.poa, method='iterative')

        df_temp = pvlib.pvsystem.sapm_celltemp(poa, wind_speed, Tamb, 
                                               model = self.system_metadata['temp_model'])
        cell_temperature = df_temp.temp_cell

        return poa, cell_temperature

    def _get_variables_from_pvlib(self, df, clearsky_variables):

        loc = pvlib.location.Location(self.system_metadata['latitude'], 
                                      self.system_metadata['longitude'], 
                                      tz = self.timezone)

        solar_position = loc.get_solarposition(df.index)
        solar_zenith = solar_position.zenith
        solar_azimuth = solar_position.azimuth

        if not clearsky_variables:
            dhi = self.df.dhi
            ghi = self.df.ghi
            dni = (ghi - dhi)/np.cos(np.deg2rad(solar_zenith))
            return self._get_poa_and_Tcell(dhi, dni, ghi, 
                                           df.Tamb, df.wind_speed,
                                           solar_zenith, solar_azimuth)

        else: 
            clearsky_irradiance = loc.get_clearsky(df.index, solar_position=solar_position)

            clearsky_Tamb = rdtools.get_clearsky_tamb(df.index, 
                                                      self.system_metadata['latitude'], 
                                                      self.system_metadata['longitude'])

            return self._get_poa_and_Tcell(clearsky_irradiance.dhi, clearsky_irradiance.dni, 
                                           clearsky_irradiance.ghi, clearsky_Tamb, 0,
                                           solar_zenith, solar_azimuth)


    def _normalize(self, energy, poa, cell_temperature):
        pvwatts_kws = {
                "poa_global" : poa,
                "P_ref" : self.system_metadata['pdc'],
                "T_cell" : cell_temperature,
                "G_ref" : 1000,
                "T_ref": 25,
                "gamma_pdc" : self.system_metadata['tempco']
                }

        # Calculate the normaliztion, the function also returns the relevant insolation for
        # each point in the normalized PV energy timeseries
        normalized_energy, insolation = rdtools.normalize_with_pvwatts(energy, pvwatts_kws)

        return normalized_energy, insolation

    def _remove_outliers(self, normalized_energy, power, poa, cell_temperature):
        logging.info('removing outliers')

        nz_mask = (normalized_energy > 0)
        poa_mask = rdtools.poa_filter(poa)
        tcell_mask = rdtools.tcell_filter(cell_temperature)
        clip_mask = rdtools.clip_filter(power)

        csi_mask = nz_mask.copy()
        if 'clearsky_insolation' in self.df.columns:
            csi_mask = rdtools.csi_filter(self.df.insolation, self.df.clearsky_insolation)

        df = self.df[nz_mask & poa_mask & tcell_mask & clip_mask & csi_mask]

        return df

    def yoy_degradation(self, normalized_energy, insolation, confidence_level):

        daily = rdtools.aggregation_insol(normalized_energy, 
                                          insolation, 
                                          frequency = 'D')

        yoy_rd, yoy_confidence_interval, yoy_info = rdtools.degradation_year_on_year(daily, confidence_level)

        return yoy_rd, yoy_confidence_interval

    def calculate_yoy_degradation(self, confidence_level=68.2):
        logging.info('processing data')
        df, clearsky_df = self._process_data() 

        logging.info('calculating year on year degradation from weather data')
        yoy_rd, yoy_confidence_interval = self.yoy_degradation(df.normalized_energy, 
                                                               df.insolation, 
                                                               confidence_level)

        clearsky_yoy_rd = None
        clearsky_yoy_confidence_interval = None
        if self.clearsky:
            logging.info('calculating year on year degradation using clearsky variables')
            clearsky_yoy_rd, clearsky_yoy_confidence_interval = self.yoy_degradation(clearsky_df.clearsky_normalized_energy, 
                                                                                     clearsky_df.clearsky_insolation, 
                                                                                     confidence_level)

        return {self.system_metadata['systemid']: (yoy_rd, yoy_confidence_interval, clearsky_yoy_rd, clearsky_yoy_confidence_interval)}
