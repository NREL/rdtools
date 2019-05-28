'''
This module contains functions and classes for object-oriented end-to-end analysis
'''
import pvlib
import pandas as pd
from . import normalization
from . import filtering
from . import aggregation
from . import degradation
from . import soiling
from . import clearsky_temperature


class system_analysis():
    '''
    Class for end-to-end analysis
    '''
    def __init__(self, pv, poa, cell_temperature, temperature_coefficient,
                 aggregation_freq='D', pv_input='power', pvlib_location=None, clearsky_poa=None,
                 clearsky_cell_temperature=None, temperature_model=None, pv_azimuth=None,
                 pv_tilt=None, pv_nameplate=None, interp_freq=None, max_interp_timedelta=None):

        if interp_freq is not None:
            pv = normalization.interpolate(pv, interp_freq, max_interp_timedelta)
            poa = normalization.interpolate(poa, interp_freq, max_interp_timedelta)
            cell_temperature = normalization.interpolate(cell_temperature, interp_freq, max_interp_timedelta)
            if clearsky_poa is not None:
                clearsky_poa = normalization.interpolate(clearsky_poa, interp_freq, max_interp_timedelta)
            if clearsky_cell_temperature is not None:
                clearsky_cell_temperature = normalization.interpolate(clearsky_cell_temperature, interp_freq, max_interp_timedelta)
            if isinstance(pv_azimuth, (pd.Series, pd.DataFrame)):
                pv_azimuth = normalization.interpolate(pv_azimuth, interp_freq, max_interp_timedelta)
            if isinstance(pv_tilt, (pd.Series, pd.DataFrame)):
                pv_tilt = normalization.interpolate(pv_tilt, interp_freq, max_interp_timedelta)

        if pv_input == 'power':
            self.pv_power = pv
            self.pv_energy = normalization.energy_from_power(pv)
        elif pv_input == 'energy':
            self.pv_power = None
            self.pv_energy = pv

        self.poa = poa
        self.cell_temperature = cell_temperature
        self.temperature_coefficient = temperature_coefficient
        self.aggregation_freq = aggregation_freq
        self.pvlib_location = pvlib_location
        self.clearsky_cell_temepratur = clearsky_cell_temperature
        self.clearsky_poa = clearsky_poa
        self.temperature_model = temperature_model
        self.pv_azimuth = pv_azimuth
        self.pv_tilt = pv_tilt
        self.pv_nameplate = pv_nameplate
        self.results = {}

        # Initialize to use default filter parameters
        self.filter_params = {
            'normalized_filter': {},
            'poa_filter': {},
            'tcell_filter': {},
            'clip_filter': {},
            'ad_hoc_filter': None  # use this to include an explict filter
        }

    def pvwatts_norm(self, poa, cell_temperature):
        if self.pv_nameplate is None:
            renorm = True
            pv_nameplate = 1.0
        else:
            renorm = False

        pvwatts_kws = {"poa_global": poa,
                       "P_ref": pv_nameplate,
                       "T_cell": cell_temperature,
                       "G_ref": 1000,
                       "T_ref": 25,
                       "gamma_pdc": self.temperature_coefficient}

        normalized, insolation = normalization.normalize_with_pvwatts(self.pv_energy, pvwatts_kws)

        if renorm:
            # Normalize to the 95th percentile for convienience, this is renomalized out
            # in the calculations but is relevant to normalized_filter()
            normalized = normalized / normalized.quantile(0.95)

        return normalized, insolation

    def filter(self, normalized, case):  # Consider making self.sensor_normalized and self.clearsky_normalized
        bool_filter = True

        if case == 'sensor':
            poa = self.poa
            cell_temp = self.cell_temperature
        if case == 'clearsky':
            poa = self.clearsky_poa
            cell_temp = self.clearsky_cell_temperature

        if 'normalized_filter' in self.filter_params.keys():
            f = filtering.normalized_filter(normalized, **self.filter_params['normalized_filter'])
            bool_filter = bool_filter & f
        if 'poa_filter' in self.filter_params.keys():
            f = filtering.poa_filter(poa, **self.filter_params['poa_filter'])
            bool_filter = bool_filter & f
        if 'tcell_filter' in self.filter_params.keys():
            f = filtering.tcell_filter(cell_temp, **self.filter_params['tcell_filter'])
            bool_filter = bool_filter & f
        if 'clip_filter' in self.filter_params.keys():
            if self.pv_power is None:
                raise ValueError('PV power (not energy) is required for the clipping filter. Either omit the clipping filter,'
                                 'provide PV power at instantiation, or explicitly assign system_analysis.pv_power.')
            f = filtering.clip_filter(self.pv_power, **self.filter_params['clip_filter'])
            bool_filter = bool_filter & f
        if 'ad_hoc_filter' in self.filter_params.keys():
            if self.filter_params['ad_hoc_filter'] is not None:
                bool_filter = bool_filter & self.filter_params['ad_hoc_filter']

        if case == 'sensor':
            self.sensor_filter = bool_filter
        elif case == 'clearsky':
            self.clearsky_filter = bool_filter

    def aggregate(self, normalized, insolation):
        aggregated = aggregation.aggregation_insol(normalized, insolation, self.aggregation_freq)
        aggregated_insolation = insolation.resample(self.aggregation_freq).sum()

        return aggregated, aggregated_insolation

    def yoy_degradation(self, aggregated, **kwargs):

        yoy_rd, yoy_ci, yoy_info = degradation.degradation_year_on_year(aggregated, **kwargs)

        yoy_results = {
            'p50_rd': yoy_rd,
            'rd_confidence_interval': yoy_ci,
            'calc_info': yoy_info
        }

        return yoy_results

    def srr_soiling(self, aggregated, aggregated_insolation, **kwargs):
        if aggregated.index.freq != 'D' or aggregated_insolation.index.freq != 'D':
            raise ValueError('Soiling SRR analysis requires daily aggregatation.')

        sr, sr_ci, soiling_info = soiling.soiling_srr(aggregated, aggregated_insolation, **kwargs)

        srr_results = {
            'p50_sratio': sr,
            'sratio_confidence_interval': sr_ci,
            'calc_info': soiling_info
        }

        return srr_results

    def sensor_preprocess(self):
        normalized, insolation = self.pvwatts_norm(self.poa, self.cell_temperature)
        self.filter(normalized, 'sensor')
        aggregated, aggregated_insolation = self.aggregate(normalized[self.sensor_filter], insolation[self.sensor_filter])
        self.sensor_aggregated_performance = aggregated
        self.sensor_aggregated_insolation = aggregated_insolation

    def sensor_analysis(self, analyses=['yoy_degradation'], yoy_kwargs={}, srr_kwargs={}):
        self.sensor_preprocess()
        sensor_results = {}

        if 'yoy_degradation' in analyses:
            yoy_results = self.yoy_degradation(self.sensor_aggregated_performance, **yoy_kwargs)
            sensor_results['yoy_degradation'] = yoy_results

        if 'srr_soiling' in analyses:
            srr_results = self.srr_soiling(self.sensor_aggregated_performance,
                                           self.sensor_aggregated_insolation,
                                           **srr_kwargs)
            sensor_results['srr_soiling'] = srr_results

        self.results['sensor'] = sensor_results

















