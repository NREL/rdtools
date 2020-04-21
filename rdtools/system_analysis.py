'''
This module contains functions and classes for object-oriented end-to-end analysis
'''
import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import normalization
from . import filtering
from . import aggregation
from . import degradation
from . import soiling
from . import clearsky_temperature
from . import plotting


class SystemAnalysis():
    '''
    Class for end-to-end analysis

    Parameters
    ----------
    pv : pd.Series
        Right-labeled time series PV energy or power. If energy, should *not*
        be cumulative, but only for preceding time step.
    poa : pd.Series
        Right-labeled time series measured plane of array irradiance in W/m^2
    cell_temperature : pd.Series
        Right-labeled time series of cell temperature in Celsius. In practice,
        back of module temperature works as a good approximation.
    ambient_temperature : pd.Series
        Right-labeled time Series of ambient temperature in Celsius
    temperature_coefficient : numeric
        Fractional PV power temperature coefficient
    aggregation_freq : str or Pandas DateOffset object
        Pandas frequency specification with which to aggregate normalized PV
        data for analysis
    pv_input : str
        'power' or 'energy' to specify type of input used for pv parameter
    pvlib_location : pvlib.location.Location
        Used for calculating clearsky temperature and irradiance
    clearsky_poa : pd.Series
        Right-labeled time Series of clear-sky plane of array irradiance
    clearsky_cell_temperature : pd.Series
        Right-labeled time series of cell temperature in clear-sky conditions
        in Celsius. In practice, back of module temperature works as a good
        approximation.
    clearsky_ambient_temperature : pd.Series
            Right-label time series of ambient temperature in clear sky conditions
        in Celsius
    windspeed : pd.Series
        Right-labeled Pandas Time Series or numeric indicating wind speed in
        m/s for use in calculating cell temperature from ambient default value
        of 0 neglects the wind in this calculation
    albedo : numeric
        Albedo to be used in irradiance transposition calculations
    temperature_model : str
        Model parameter pvlib.pvsystem.sapm_celltemp() used in calculating cell
        temperature from ambient
    pv_azimuth : numeric
        Azimuth of PV array in degrees from north
    pv_tilt : numeric
        Tilt of PV array in degrees from horizontal
    pv_nameplate : numeric
        Nameplate DC rating of PV array in Watts
    interp_freq : str or Pandas DateOffset object
        Pandas frequency specification used to interpolate all pandas.Series
        passed at instantiation. We recommend using the natural frequency of the
        data, rather than up or down sampling. Analysis requires regular time series.
    max_timedelta : datetime.timedelta
        The maximum gap in the data to be interpolated/integrated across when
        interpolating or calculating energy from power

    Attributes
    ----------
    (not all attributes documented here)
    filter_parameters: dict
        parameters to be passed to rdtools.filtering functions. Keys are the
        names of the rdtools.filtering functions. Values are dicts of parameters
        to be passed to those functions. Also has a special key `ad_hoc_filter`
        the associated value is a boolean mask joined with the rest of the filters.
        filter_parameters defaults to empty dicts for each function in rdtools.filtering,
        in which case those functions use default parameter values,  `ad_hoc_filter`
        defaults to None. See examples for more information.
    results : dict
        Nested dict used to store the results of methods ending with `_analysis`

    '''

    def __init__(self, pv, poa=None, cell_temperature=None, ambient_temperature=None,
                 temperature_coefficient=None, aggregation_freq='D', pv_input='power', pvlib_location=None,
                 clearsky_poa=None, clearsky_cell_temperature=None, clearsky_ambient_temperature=None,
                 windspeed=0, albedo=0.25, temperature_model=None, pv_azimuth=None, pv_tilt=None,
                 pv_nameplate=None, interp_freq=None, max_timedelta=None):

        if interp_freq is not None:
            pv = normalization.interpolate(pv, interp_freq, max_timedelta)
            if poa is not None:
                poa = normalization.interpolate(poa, interp_freq, max_timedelta)
            if cell_temperature is not None:
                cell_temperature = normalization.interpolate(cell_temperature, interp_freq, max_timedelta)
            if ambient_temperature is not None:
                ambient_temperature = normalization.interpolate(ambient_temperature, interp_freq, max_timedelta)
            if clearsky_poa is not None:
                clearsky_poa = normalization.interpolate(clearsky_poa, interp_freq, max_timedelta)
            if clearsky_cell_temperature is not None:
                clearsky_cell_temperature = normalization.interpolate(clearsky_cell_temperature, interp_freq, max_timedelta)
            if clearsky_ambient_temperature is not None:
                clearsky_ambient_temperature = normalization.interpolate(clearsky_ambient_temperature, interp_freq, max_timedelta)
            if isinstance(pv_azimuth, (pd.Series, pd.DataFrame)):
                pv_azimuth = normalization.interpolate(pv_azimuth, interp_freq, max_timedelta)
            if isinstance(pv_tilt, (pd.Series, pd.DataFrame)):
                pv_tilt = normalization.interpolate(pv_tilt, interp_freq, max_timedelta)

        if pv_input == 'power':
            self.pv_power = pv
            self.pv_energy = normalization.energy_from_power(pv, max_timedelta=max_timedelta)
        elif pv_input == 'energy':
            self.pv_power = None
            self.pv_energy = pv

        self.cell_temperature = cell_temperature
        self.ambient_temperature = ambient_temperature
        self.clearsky_cell_temperature = clearsky_cell_temperature
        self.clearsky_ambient_temperature = clearsky_ambient_temperature

        self.poa = poa
        self.temperature_coefficient = temperature_coefficient
        self.aggregation_freq = aggregation_freq
        self.pvlib_location = pvlib_location
        self.clearsky_poa = clearsky_poa
        self.windspeed = windspeed
        self.albedo = albedo
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
            'csi_filter': {},
            'ad_hoc_filter': None  # use this to include an explict filter
        }

    def calc_clearsky_poa(self, times=None, rescale=True, **kwargs):
        '''
        Calculate clearsky plane-of-array irradiance and stores in self.clearsky_poa

        Parameters
        ----------
        times : pandas.DateTimeIndex
            times on for which to calculate clearsky poa
        rescale : bool
            Whether to attempt to rescale clearsky irradiance to measured
        model : str
            Model for pvlib.irradiance.get_total_irradiance()
        kwargs :
            Extra parameters passed to pvlib.irradiance.get_total_irradiance()

        Returns
        -------
        None
        '''
        if times is None:
            times = self.poa.index
        if self.pvlib_location is None:
            raise ValueError('pvlib location must be provided')
        if self.pv_tilt is None or self.pv_azimuth is None:
            raise ValueError('pv_tilt and pv_azimuth must be provided')
        if times is not self.poa.index and rescale is True:
            raise ValueError('rescale=True can only be used when clearsky poa is on same index as poa')

        loc = self.pvlib_location
        sun = loc.get_solarposition(times)
        clearsky = loc.get_clearsky(times, solar_position=sun)

        clearsky_poa = pvlib.irradiance.get_total_irradiance(self.pv_tilt, self.pv_azimuth, sun['apparent_zenith'],
                                                             sun['azimuth'], clearsky['dni'], clearsky['ghi'],
                                                             clearsky['dhi'], albedo=self.albedo, **kwargs)
        clearsky_poa = clearsky_poa['poa_global']

        if rescale is True:
            clearsky_poa = normalization.irradiance_rescale(self.poa, clearsky_poa, method='iterative')

        self.clearsky_poa = clearsky_poa

    def calc_cell_temperature(self, poa, windspeed, ambient_temperature):
        '''
        Return cell temperature calculated from ambient conditions.

        Parameters
        ----------
        poa : numeric
            Plane of array irradiance in W/m^2
        windspeed = numeric
            Wind speed in m/s
        ambient_temperature : numeric
            Ambient temperature in Celsius

        Returns
        -------
        numeric
            calculated cell temperature
        '''
        if self.temperature_model is None:
            cell_temp = pvlib.pvsystem.sapm_celltemp(poa, windspeed, ambient_temperature)
        else:
            cell_temp = pvlib.pvsystem.sapm_celltemp(poa, windspeed, ambient_temperature, model=self.temperature_model)
        cell_temp = cell_temp['temp_cell']

        return cell_temp

    def calc_clearsky_tamb(self):
        '''
        Calculate clear-sky ambient temperature and store in self.clearsky_ambient_temperature
        '''
        times = self.clearsky_poa.index
        if self.pvlib_location is None:
            raise ValueError('pvlib location must be provided')
        loc = self.pvlib_location

        cs_amb_temp = clearsky_temperature.get_clearsky_tamb(times, loc.latitude, loc.longitude)

        self.clearsky_ambient_temperature = cs_amb_temp

    def pvwatts_norm(self, poa, cell_temperature):
        '''
        Normalize PV energy to that expected from a PVWatts model.

        Parameters
        ---------
        poa : numeric
            plane of array irradiance in W/m^2
        cell_temperature : numeric
            cell temperature in Celsius

        Returns
        -------
        pandas.Series
            Normalized pv energy
        pandas.Series
            Associated insolation
        '''
        if self.pv_nameplate is None:
            renorm = True
            pv_nameplate = 1.0
        else:
            renorm = False
            pv_nameplate = self.pv_nameplate

        if self.temperature_coefficient is None:
            raise ValueError('Temperature coeffcient must be available to perform pvwatts_norm')

        pvwatts_kws = {"poa_global": poa,
                       "P_ref": pv_nameplate,
                       "T_cell": cell_temperature,
                       "G_ref": 1000,
                       "T_ref": 25,
                       "gamma_pdc": self.temperature_coefficient}

        normalized, insolation = normalization.normalize_with_pvwatts(self.pv_energy, pvwatts_kws)

        if renorm:
            # Normalize to the 95th percentile for convenience, this is renormalized out
            # in the calculations but is relevant to normalized_filter()
            x = normalized[np.isfinite(normalized)]
            normalized = normalized / x.quantile(0.95)

        return normalized, insolation

    def filter(self, normalized, case):
        '''
        Calculate filters based on those in rdtools.filtering. Uses
        self.filter_params, which is a dict, the keys of which are names of
        functions in rdtools.filtering, and the values of which are dicts
        containing the associated parameters with which to run the filtering
        functions. See examples for details on how to modify filter parameters.

        Parameters
        ----------
        normalized : pandas.Series
            Time series of normalized PV energy
        case : str
            'sensor' or 'clearsky' which filtering protocol to apply. Affects
            whether filtering.csi_filter() is used and whether result is stored
            in self.sensor_filter or self.clearsky_filter)

        Returns
        -------
        None
        '''
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
            if poa is None:
                raise ValueError('poa must be available to use poa_filter')
            f = filtering.poa_filter(poa, **self.filter_params['poa_filter'])
            bool_filter = bool_filter & f
        if 'tcell_filter' in self.filter_params.keys():
            if cell_temp is None:
                raise ValueError('Cell temperature must be available to use tcell_filter')
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
        if case == 'clearsky':
            if self.poa is None or self.clearsky_poa is None:
                raise ValueError('Both poa and clearsky_poa must be available to do clearsky filtering with csi_filter')
            f = filtering.csi_filter(self.poa, self.clearsky_poa, **self.filter_params['csi_filter'])
            bool_filter = bool_filter & f

        if case == 'sensor':
            self.sensor_filter = bool_filter
        elif case == 'clearsky':
            self.clearsky_filter = bool_filter

    def aggregate(self, normalized, insolation):
        '''
        Return insolation-weighted normalized PV energy and the associated aggregated insolation

        Parameters
        ----------
        normalized : pandas.Series
            Time series of normalized PV energy
        insolation : pandas.Series
            Time Series of insolation associated with each `normalized` point

        Returns
        -------
        pandas.Series
            Insolation-weighted aggregated normalized PV energy
        pandas.Series
            Aggregated insolation
        '''
        aggregated = aggregation.aggregation_insol(normalized, insolation, self.aggregation_freq)
        aggregated_insolation = insolation.resample(self.aggregation_freq).sum()

        return aggregated, aggregated_insolation

    def yoy_degradation(self, aggregated, **kwargs):
        '''
        Perform year-on-year degradation analysis on insolation-weighted
        aggregated energy yield.

        Parameters
        ----------
        aggregated : pandas.Series
            Time Series of insolation-weighted aggregated normalized PV energy
        kwargs :
            Extra parameters passed to degradation.degradation_year_on_year()

        Returns
        -------
        dict
            Year-on-year degradation results with keys:
            'p50_rd' : The median year-on-year degradation rate
            'rd_confidence_interval' : lower and upper bounds of degradation
                                       rate confidence interval as a list
            'calc_info': Dict of detailed results
                         (see degradation.degradation_year_on_year() docs)
        '''

        yoy_rd, yoy_ci, yoy_info = degradation.degradation_year_on_year(aggregated, **kwargs)

        yoy_results = {
            'p50_rd': yoy_rd,
            'rd_confidence_interval': yoy_ci,
            'calc_info': yoy_info
        }

        return yoy_results

    def srr_soiling(self, aggregated, aggregated_insolation, **kwargs):
        '''
        Perform stochastic rate and recovery soiling analysis.

        Parameters
        ---------
        aggregated : pandas.Series
            Time Series of insolation-weighted aggregated normalized PV energy
        aggregated_insolation : pandas.Series
            Time Series of insolation, aggregated at same level as `aggregated`
        kwargs :
            Extra parameters passed to soiling.soiling_srr()

        Returns
        -------
        dict
            Soiling results with keys:
            'p50_sratio' : The median insolation-weighted soiling ratio
            'sratio_confidence_interval' : list of lower and upper bounds of
                                          insolation-weighted soiling ratio
                                          confidence interval
            'calc_info' : Dict of detailed results (see soiling.soiling_srr() docs)
        '''
        if aggregated.index.freq != 'D' or aggregated_insolation.index.freq != 'D':
            raise ValueError('Soiling SRR analysis requires daily aggregation.')

        sr, sr_ci, soiling_info = soiling.soiling_srr(aggregated, aggregated_insolation, **kwargs)

        srr_results = {
            'p50_sratio': sr,
            'sratio_confidence_interval': sr_ci,
            'calc_info': soiling_info
        }

        return srr_results

    def sensor_preprocess(self):
        '''
        Perform sensor-based normalization, filtering, and aggregation work flow.
        '''
        if self.poa is None:
            raise ValueError('poa must be available to perform sensor_preprocess')
        if self.cell_temperature is None and self.ambient_temperature is None:
            raise ValueError('either cell or ambient temperature must be available to perform sensor_preprocess')
        if self.cell_temperature is None:
            self.cell_temperature = self.calc_cell_temperature(self.poa, self.windspeed, self.ambient_temperature)
        normalized, insolation = self.pvwatts_norm(self.poa, self.cell_temperature)
        self.filter(normalized, 'sensor')
        aggregated, aggregated_insolation = self.aggregate(normalized[self.sensor_filter], insolation[self.sensor_filter])
        self.sensor_aggregated_performance = aggregated
        self.sensor_aggregated_insolation = aggregated_insolation

    def clearsky_preprocess(self):
        '''
        Perform clear-sky-based normalization, filtering, and aggregation work flow
        '''
        if self.clearsky_poa is None:
            self.calc_clearsky_poa(model='isotropic')
        if self.clearsky_cell_temperature is None:
            if self.clearsky_ambient_temperature is None:
                self.calc_clearsky_tamb()
            self.clearsky_cell_temperature = self.calc_cell_temperature(self.clearsky_poa, 0, self.clearsky_ambient_temperature)
            # Note example notebook uses windspeed=0 in the clearskybranch
        cs_normalized, cs_insolation = self.pvwatts_norm(self.clearsky_poa, self.clearsky_cell_temperature)
        self.filter(cs_normalized, 'clearsky')
        cs_aggregated, cs_aggregated_insolation = self.aggregate(cs_normalized[self.clearsky_filter], cs_insolation[self.clearsky_filter])
        self.clearsky_aggregated_performance = cs_aggregated
        self.clearsky_aggregated_insolation = cs_aggregated_insolation

    def sensor_analysis(self, analyses=['yoy_degradation'], yoy_kwargs={}, srr_kwargs={}):
        '''
        Perform entire sensor-based analysis workflow. Results are stored in self.results['sensor']

        Parameters
        ---------
        analyses : list of str
            Analyses to perform, valid entries are 'yoy_degradation' and 'srr_soiling'
        yoy_kwargs : dict
            kwargs to pass to degradation.degradation_year_on_year()
        srr_kwargs : dict
            kwargs to pass to soiling.soiling_srr()

        Returns
        -------
        None
        '''

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

    def clearsky_analysis(self, analyses=['yoy_degradation'], yoy_kwargs={}, srr_kwargs={}):
        '''
        Perform entire clear-sky-based analysis workflow. Results are stored in self.results['clearsky']

        Parameters
        ---------
        analyses : list of str
            Analyses to perform, valid entries are 'yoy_degradation' and 'srr_soiling'
        yoy_kwargs : dict
            kwargs to pass to degradation.degradation_year_on_year()
        srr_kwargs : dict
            kwargs to pass to soiling.soiling_srr()

        Returns
        -------
        None
        '''

        self.clearsky_preprocess()
        clearsky_results = {}

        if 'yoy_degradation' in analyses:
            yoy_results = self.yoy_degradation(self.clearsky_aggregated_performance, **yoy_kwargs)
            clearsky_results['yoy_degradation'] = yoy_results

        if 'srr_soiling' in analyses:
            srr_results = self.srr_soiling(self.clearsky_aggregated_performance,
                                           self.clearsky_aggregated_insolation,
                                           **srr_kwargs)
            clearsky_results['srr_soiling'] = srr_results

        self.results['clearsky'] = clearsky_results

    def plot_degradation_summary(self, result_to_plot, **kwargs):
        '''
        Return a figure of a scatter plot and a histogram summarizing degradation rate analysis.

        Parameters
        ----------
        result_to_plot : str
            The workflow result to plot, allowed values are 'sensor' and 'clearsky'
        kwargs :
            Extra parameters passed to plotting.degradation_summary_plots()

        Returns
        -------
        matplotlib.figure.Figure

        '''

        if result_to_plot == 'sensor':
            results_dict = self.results['sensor']['yoy_degradation']
            aggregated = self.sensor_aggregated_performance
        elif result_to_plot == 'clearsky':
            results_dict = self.results['clearsky']['yoy_degradation']
            aggregated = self.clearsky_aggregated_performance

        fig = plotting.degradation_summary_plots(results_dict['p50_rd'], results_dict['rd_confidence_interval'],
                                                 results_dict['calc_info'], aggregated, **kwargs)
        return fig

    def plot_soiling_monte_carlo(self, result_to_plot, **kwargs):
        '''
        Return a figure visualizing the Monte Carlo of soiling profiles used in
        stochastic rate and recovery soiling analysis.

        Parameters
        ----------
        result_to_plot : str
            The workflow result to plot, allowed values are 'sensor' and 'clearsky'
        kwargs :
            Extra parameters passed to plotting.soiling_monte_carlo_plot()

        Returns
        -------
        matplotlib.figure.Figure
        '''

        if result_to_plot == 'sensor':
            results_dict = self.results['sensor']['srr_soiling']
            aggregated = self.sensor_aggregated_performance
        elif result_to_plot == 'clearsky':
            results_dict = self.results['clearsky']['srr_soiling']
            aggregated = self.clearsky_aggregated_performance

        fig = plotting.soiling_monte_carlo_plot(results_dict['calc_info'], aggregated, **kwargs)

        return fig

    def plot_soiling_interval(self, result_to_plot, **kwargs):
        '''
        Return a figure visualizing the valid soiling intervals used in
        stochastic rate and recovery soiling analysis.

        Parameters
        ----------
        result_to_plot : str
            The workflow result to plot, allowed values are 'sensor' and 'clearsky'
        kwargs :
            Extra parameters passed to plotting.soiling_interval_plot()

        Returns
        -------
        matplotlib.figure.Figure
        '''

        if result_to_plot == 'sensor':
            results_dict = self.results['sensor']['srr_soiling']
            aggregated = self.sensor_aggregated_performance
        elif result_to_plot == 'clearsky':
            results_dict = self.results['clearsky']['srr_soiling']
            aggregated = self.clearsky_aggregated_performance

        fig = plotting.soiling_interval_plot(results_dict['calc_info'], aggregated, **kwargs)

        return fig

    def plot_soiling_rate_histogram(self, result_to_plot, **kwargs):
        '''
        Return a histogram of soiling rates found in the stochastic rate and recovery
        soiling analysis

        Parameters
        ----------
        result_to_plot : str
            The workflow result to plot, allowed values are 'sensor' and 'clearsky'
        kwargs :
            Extra parameters passed to plotting.soiling_rate_histogram()

        Returns
        -------
        matplotlib.figure.Figure
        '''

        if result_to_plot == 'sensor':
            results_dict = self.results['sensor']['srr_soiling']
        elif result_to_plot == 'clearsky':
            results_dict = self.results['clearsky']['srr_soiling']

        fig = plotting.soiling_rate_histogram(results_dict['calc_info'], **kwargs)

        return fig

    def plot_pv_vs_irradiance(self, poa_type, alpha=0.01, **kwargs):
        '''
        Plot PV energy vs irradiance, useful in diagnosing things like timezone problems or
        transposition errors.

        Parameters
        ----------
        poa_type: str
            The plane of array irradiance type to plot, allowed values are
            'sensor' and 'clearsky'
        alpha : numeric
            transparency of the scatter plot
        kwargs :
            Extra parameters passed to matplotlib.pyplot.axis.plot()

        Returns
        -------
        matplotlib.figure.Figure
        '''

        if poa_type == 'sensor':
            poa = self.poa
        elif poa_type == 'clearsky':
            poa = self.clearsky_poa

        to_plot = pd.merge(pd.DataFrame(poa), pd.DataFrame(self.pv_energy), left_index=True, right_index=True)

        fig, ax = plt.subplots()
        ax.plot(to_plot.iloc[:, 0], to_plot.iloc[:, 1], 'o', alpha=alpha, **kwargs)
        ax.set_xlim(0, 1500)
        ax.set_xlabel('Irradiance (W/m$^2$)')
        ax.set_ylabel('PV Energy (Wh/timestep)')

        return fig
