'''
This module contains functions and classes for object-oriented
end-to-end analysis
'''
import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdtools import normalization, filtering, aggregation, degradation
from rdtools import clearsky_temperature, plotting
import warnings


class TrendAnalysis():
    '''
    Class for end-to-end degradation and soiling analysis using
    :py:meth:`~rdtools.TrendAnalysis.sensor_analysis` or
    :py:meth:`~rdtools.TrendAnalysis.clearsky_analysis`

    Parameters
    ----------
    pv : pandas.Series
        Right-labeled time series PV energy or power. If energy, should *not*
        be cumulative, but only for preceding time step.
    poa_global : pandas.Series
        Right-labeled time series measured plane of array irradiance in W/m^2
    temperature_cell : pandas.Series
        Right-labeled time series of cell temperature in Celsius. In practice,
        back of module temperature works as a good approximation.
    temperature_ambient : pandas.Series
        Right-labeled time Series of ambient temperature in Celsius
    gamma_pdc : float
        Fractional PV power temperature coefficient
    aggregation_freq : str or pandas.tseries.offsets.DateOffset
        Pandas frequency specification with which to aggregate normalized PV
        data for analysis. For more information, see
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    pv_input : str
        'power' or 'energy' to specify type of input used for pv parameter
    windspeed : numeric
        Right-labeled Pandas Time Series or single numeric value indicating wind
        speed in m/s for use in calculating cell temperature from ambient default
        value of 0 neglects the wind in this calculation
    power_expected : pandas.Series
        Right-labeled time series of expected PV power. (Note: Expected energy
        is not supported.)
    temperature_model : str or dict
        Model parameters for :py:func:`pvlib.temperature.sapm_cell`. Used in calculating cell
        temperature from ambient. If string, must be a valid entry
        for sapm model in :py:data:`pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS`. If dict, must
        have keys 'a', 'b', 'deltaT'. See :py:func:`pvlib.temperature.sapm_cell` documentation
        for details.
    power_dc_rated : float
        Nameplate DC rating of PV array in Watts. If omitted, pv output will be internally
        normalized in the normalization step based on it's 95th percentile
        (see TrendAnalysis._pvwatts_norm() source).
    interp_freq : str or pandas.tseries.offsets.DateOffset
        Pandas frequency specification used to interpolate the input PV power
        or energy. We recommend using the natural frequency of the
        data, rather than up or down sampling. Analysis requires regular time series.
        For more information see
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    max_timedelta : pandas.Timedelta
        The maximum gap in the data to be interpolated/integrated across when
        interpolating or calculating energy from power

    Attributes
    ----------
    (not all attributes documented here)
    filter_params: dict
        parameters to be passed to rdtools.filtering functions. Keys are the
        names of the rdtools.filtering functions. Values are dicts of parameters
        to be passed to those functions. Also has a special key `ad_hoc_filter`
        the associated value is a boolean mask joined with the rest of the filters.
        filter_params defaults to empty dicts for each function in rdtools.filtering,
        in which case those functions use default parameter values,  `ad_hoc_filter`
        defaults to None. See examples for more information.
    results : dict
        Nested dict used to store the results of methods ending with `_analysis`
    '''

    def __init__(self, pv, poa_global=None, temperature_cell=None, temperature_ambient=None,
                 gamma_pdc=None, aggregation_freq='D', pv_input='power',
                 windspeed=0, power_expected=None, temperature_model=None,
                 power_dc_rated=None, interp_freq=None, max_timedelta=None):

        if interp_freq is not None:
            pv = normalization.interpolate(pv, interp_freq, max_timedelta)

        if poa_global is not None:
            poa_global = normalization.interpolate(
                poa_global, pv.index, max_timedelta)
        if temperature_cell is not None:
            temperature_cell = normalization.interpolate(
                temperature_cell, pv.index, max_timedelta)
        if temperature_ambient is not None:
            temperature_ambient = normalization.interpolate(
                temperature_ambient, pv.index, max_timedelta)
        if power_expected is not None:
            power_expected = normalization.interpolate(
                power_expected, pv.index, max_timedelta)
        if isinstance(windspeed, pd.Series):
            windspeed = normalization.interpolate(
                windspeed, pv.index, max_timedelta)

        if pv_input == 'power':
            self.pv_power = pv
            self.pv_energy = normalization.energy_from_power(
                pv, max_timedelta=max_timedelta)
        elif pv_input == 'energy':
            self.pv_power = None
            self.pv_energy = pv

        self.temperature_cell = temperature_cell
        self.temperature_ambient = temperature_ambient
        self.poa_global = poa_global
        self.gamma_pdc = gamma_pdc
        self.aggregation_freq = aggregation_freq
        self.windspeed = windspeed
        self.power_expected = power_expected
        self.temperature_model = temperature_model
        self.power_dc_rated = power_dc_rated
        self.interp_freq = interp_freq
        self.max_timedelta = max_timedelta
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
        # remove tcell_filter from list if power_expected is passed in
        if power_expected is not None and temperature_cell is None:
            del self.filter_params['tcell_filter']

    def set_clearsky(self, pvlib_location=None, pv_azimuth=None, pv_tilt=None,
                     poa_global_clearsky=None, temperature_cell_clearsky=None,
                     temperature_ambient_clearsky=None, albedo=0.25,
                     solar_position_method='nrel_numpy'):
        '''
        Initialize values for a clearsky analysis which requires configuration
        of location and orientation details. If optional parameters `poa_global_clearsky`,
        `temperature_ambient_clearsky` are not passed, they will be modeled
        based on location and orientation.

        Parameters
        ----------
        pvlib_location : pvlib.location.Location
            Used for calculating clearsky temperature and irradiance
        pv_azimuth : numeric
            Azimuth of PV array in degrees from north. Can be right-labeled
            Pandas Time Series or single numeric value.
        pv_tilt : numeric
            Tilt of PV array in degrees from horizontal. Can be right-labeled
            Pandas Time Series or single numeric value.
        poa_global_clearsky : pandas.Series
            Right-labeled time Series of clear-sky plane of array irradiance
        temperature_cell_clearsky : pandas.Series
            Right-labeled time series of cell temperature in clear-sky conditions
            in Celsius. In practice, back of module temperature works as a good
            approximation.
        temperature_ambient_clearsky : pandas.Series
            Right-label time series of ambient temperature in clear sky conditions
            in Celsius
        albedo : numeric
            Albedo to be used in irradiance transposition calculations. Can be right-labeled
            Pandas Time Series or single numeric value.
        solar_position_method : str, default 'nrel_numpy'
            Optional method name to pass to :py:func:`pvlib.solarposition.get_solarposition`.
            Switching methods may improve calculation time.
        '''
        max_timedelta = self.max_timedelta

        if poa_global_clearsky is not None:
            poa_global_clearsky = normalization.interpolate(
                poa_global_clearsky, self.pv_energy.index, max_timedelta)
        if temperature_cell_clearsky is not None:
            temperature_cell_clearsky = normalization.interpolate(
                temperature_cell_clearsky, self.pv_energy.index, max_timedelta)
        if temperature_ambient_clearsky is not None:
            temperature_ambient_clearsky = normalization.interpolate(
                temperature_ambient_clearsky, self.pv_energy.index, max_timedelta)
        if isinstance(pv_azimuth, (pd.Series, pd.DataFrame)):
            pv_azimuth = normalization.interpolate(
                pv_azimuth, self.pv_energy.index, max_timedelta)
        if isinstance(pv_tilt, (pd.Series, pd.DataFrame)):
            pv_tilt = normalization.interpolate(
                pv_tilt, self.pv_energy.index, max_timedelta)

        self.pvlib_location = pvlib_location
        self.pv_azimuth = pv_azimuth
        self.pv_tilt = pv_tilt
        self.poa_global_clearsky = poa_global_clearsky
        self.temperature_cell_clearsky = temperature_cell_clearsky
        self.temperature_ambient_clearsky = temperature_ambient_clearsky
        self.albedo = albedo
        self.solar_position_method = solar_position_method

    def _calc_clearsky_poa(self, times=None, rescale=True, **kwargs):
        '''
        Calculate clearsky plane-of-array irradiance and stores in self.poa_global_clearsky

        Parameters
        ----------
        times : pandas.DateTimeIndex
            times on for which to calculate clearsky poa.  If not provided then
            it will be simulated at 1-minute frequency and averaged to match the
            index of self.poa_global
        rescale : bool
            Whether to attempt to rescale clearsky irradiance to measured
        kwargs :
            Extra parameters passed to pvlib.irradiance.get_total_irradiance()

        Returns
        -------
        None
        '''
        aggregate = False
        if times is None:
            times = pd.date_range(self.poa_global.index.min(), self.poa_global.index.max(),
                                  freq='1min')
            aggregate = True

        if self.pvlib_location is None:
            raise ValueError(
                'pvlib location must be provided using set_clearsky()')
        if self.pv_tilt is None or self.pv_azimuth is None:
            raise ValueError(
                'pv_tilt and pv_azimuth must be provided using set_clearsky()')

        loc = self.pvlib_location
        solar_position_kwargs = {}
        if self.solar_position_method:
            solar_position_kwargs['method'] = self.solar_position_method
        sun = loc.get_solarposition(times, **solar_position_kwargs)
        clearsky = loc.get_clearsky(times, solar_position=sun)

        clearsky_poa = pvlib.irradiance.get_total_irradiance(
            self.pv_tilt,
            self.pv_azimuth,
            sun['apparent_zenith'],
            sun['azimuth'],
            clearsky['dni'],
            clearsky['ghi'],
            clearsky['dhi'],
            albedo=self.albedo,
            **kwargs)
        clearsky_poa = clearsky_poa['poa_global']

        if aggregate:
            interval_id = pd.Series(range(len(self.poa_global)), index=self.poa_global.index)
            interval_id = interval_id.reindex(times, method='backfill')
            clearsky_poa = clearsky_poa.groupby(interval_id).mean()
            clearsky_poa.index = self.poa_global.index
            clearsky_poa.iloc[0] = np.nan

        if rescale is True:
            if not clearsky_poa.index.equals(self.poa_global.index):
                raise ValueError(
                    'rescale=True can only be used when clearsky poa is on same index as poa')

            clearsky_poa = normalization.irradiance_rescale(
                self.poa_global, clearsky_poa, method='iterative')

        self.poa_global_clearsky = clearsky_poa

    def _calc_cell_temperature(self, poa_global, temperature_ambient, windspeed):
        '''
        Return cell temperature calculated from ambient conditions.

        Parameters
        ----------
        poa_global : numeric
            Plane of array irradiance in W/m^2
        temperature_ambient : numeric
            Ambient temperature in Celsius
        windspeed = numeric
            Wind speed in m/s

        Returns
        -------
        numeric
            calculated cell temperature
        '''

        try:  # workflow for pvlib >= 0.7

            if self.temperature_model is None:
                self.temperature_model = "open_rack_glass_polymer"  # default

            # check if self.temperature_model is a string or dict with keys 'a', 'b' and 'deltaT'
            if isinstance(self.temperature_model, str):
                model_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
                    'sapm'][self.temperature_model]
            elif (isinstance(self.temperature_model, dict) &
                  ('a' in self.temperature_model) &
                  ('b' in self.temperature_model) &
                  ('deltaT' in self.temperature_model)):
                model_params = self.temperature_model
            else:
                raise ValueError('pvlib temperature_model entry is neither '
                                 'a string nor a dictionary with correct '
                                 'entries. Try "open_rack_glass_polymer"')
            cell_temp = pvlib.temperature.sapm_cell(poa_global=poa_global,
                                                    temp_air=temperature_ambient,
                                                    wind_speed=windspeed,
                                                    **model_params
                                                    )
        except AttributeError as e:
            print('Error: PVLib > 0.7 required')
            raise e
        return cell_temp

    def _calc_clearsky_tamb(self):
        '''
        Calculate clear-sky ambient temperature and store in self.temperature_ambient_clearsky
        '''
        times = self.poa_global_clearsky.index
        if self.pvlib_location is None:
            raise ValueError(
                'pvlib location must be provided using set_clearsky()')
        loc = self.pvlib_location

        cs_amb_temp = clearsky_temperature.get_clearsky_tamb(
            times, loc.latitude, loc.longitude)

        self.temperature_ambient_clearsky = cs_amb_temp

    def _pvwatts_norm(self, poa_global, temperature_cell):
        '''
        Normalize PV energy to that expected from a PVWatts model.

        Parameters
        ---------
        poa_global : numeric
            plane of array irradiance in W/m^2
        temperature_cell : numeric
            cell temperature in Celsius

        Returns
        -------
        pandas.Series
            Normalized pv energy
        pandas.Series
            Associated insolation
        '''

        if self.power_dc_rated is None:
            renorm = True
            power_dc_rated = 1.0
        else:
            renorm = False
            power_dc_rated = self.power_dc_rated

        if self.gamma_pdc is None:
            warnings.warn('Temperature coefficient not passed in to TrendAnalysis'
                          '. No temperature correction will be conducted.')
        pvwatts_kws = {"poa_global": poa_global,
                       "power_dc_rated": power_dc_rated,
                       "temperature_cell": temperature_cell,
                       "poa_global_ref": 1000,
                       "temperature_cell_ref": 25,
                       "gamma_pdc": self.gamma_pdc}

        energy_normalized, insolation = normalization.normalize_with_pvwatts(
            self.pv_energy, pvwatts_kws)

        if renorm:
            # Normalize to the 95th percentile for convenience, this is renormalized out
            # in the calculations but is relevant to normalized_filter()
            x = energy_normalized[np.isfinite(energy_normalized)]
            energy_normalized = energy_normalized / x.quantile(0.95)

        return energy_normalized, insolation

    def _filter(self, energy_normalized, case):
        '''
        Calculate filters based on those in rdtools.filtering. Uses
        self.filter_params, which is a dict, the keys of which are names of
        functions in rdtools.filtering, and the values of which are dicts
        containing the associated parameters with which to run the filtering
        functions. See examples for details on how to modify filter parameters.

        Parameters
        ----------
        energy_normalized : pandas.Series
            Time series of normalized PV energy
        case : str
            'sensor' or 'clearsky' which filtering protocol to apply. Affects
            whether filtering.csi_filter() is used and whether result is stored
            in self.sensor_filter or self.clearsky_filter)

        Returns
        -------
        None
        '''
        # Combining filters is non-trivial because of the possibility of index
        # mismatch.  Adding columns to an existing dataframe performs a left index
        # join, but probably we actually want an outer join.  We can get an outer
        # join by keeping this as a dictionary and converting it to a dataframe all
        # at once.  However, we add a default value of True, with the same index as
        # energy_normalized, so that the output is still correct even when all
        # filters have been disabled.
        filter_components = {'default': pd.Series(True, index=energy_normalized.index)}

        if case == 'sensor':
            poa = self.poa_global
            cell_temp = self.temperature_cell
        if case == 'clearsky':
            poa = self.poa_global_clearsky
            cell_temp = self.temperature_cell_clearsky

        if 'normalized_filter' in self.filter_params:
            f = filtering.normalized_filter(
                energy_normalized, **self.filter_params['normalized_filter'])
            filter_components['normalized_filter'] = f
        if 'poa_filter' in self.filter_params:
            if poa is None:
                raise ValueError('poa must be available to use poa_filter')
            f = filtering.poa_filter(poa, **self.filter_params['poa_filter'])
            filter_components['poa_filter'] = f
        if 'tcell_filter' in self.filter_params:
            if cell_temp is None:
                raise ValueError(
                    'Cell temperature must be available to use tcell_filter')
            f = filtering.tcell_filter(
                cell_temp, **self.filter_params['tcell_filter'])
            filter_components['tcell_filter'] = f
        if 'clip_filter' in self.filter_params:
            if self.pv_power is None:
                raise ValueError('PV power (not energy) is required for the clipping filter. '
                                 'Either omit the clipping filter, provide PV power at '
                                 'instantiation, or explicitly assign TrendAnalysis.pv_power.')
            f = filtering.clip_filter(
                self.pv_power, **self.filter_params['clip_filter'])
            filter_components['clip_filter'] = f
        if case == 'clearsky':
            if self.poa_global is None or self.poa_global_clearsky is None:
                raise ValueError('Both poa_global and poa_global_clearsky must be available to '
                                 'do clearsky filtering with csi_filter')
            f = filtering.csi_filter(
                self.poa_global, self.poa_global_clearsky, **self.filter_params['csi_filter'])
            filter_components['csi_filter'] = f

        # note: the previous implementation using the & operator treated NaN
        # filter values as False, so we do the same here for consistency:
        filter_components = pd.DataFrame(filter_components).fillna(False)

        # apply special checks to ad_hoc_filter, as it is likely more prone to user error
        if self.filter_params.get('ad_hoc_filter', None) is not None:
            ad_hoc_filter = self.filter_params['ad_hoc_filter']

            if ad_hoc_filter.isnull().any():
                warnings.warn('ad_hoc_filter contains NaN values; setting to False (excluding)')
                ad_hoc_filter = ad_hoc_filter.fillna(False)

            if not filter_components.index.equals(ad_hoc_filter.index):
                warnings.warn('ad_hoc_filter index does not match index of other filters; missing '
                              'values will be set to True (kept). Align the index with the index '
                              'of the filter_components attribute to prevent this warning')
                ad_hoc_filter = ad_hoc_filter.reindex(filter_components.index).fillna(True)

            filter_components['ad_hoc_filter'] = ad_hoc_filter

        bool_filter = filter_components.all(axis=1)
        filter_components = filter_components.drop(columns=['default'])
        if case == 'sensor':
            self.sensor_filter = bool_filter
            self.sensor_filter_components = filter_components
        elif case == 'clearsky':
            self.clearsky_filter = bool_filter
            self.clearsky_filter_components = filter_components

    def _filter_check(self, post_filter):
        '''
        post-filter check for requisite 730 days of data

        Parameters
        ----------
        post_filter : pandas.Series
            Time series filtered by boolean output from self.filter
        '''

        if post_filter.empty:
            post_filter_length = pd.Timedelta('0d')
        else:
            post_filter_length = post_filter.index[-1] - post_filter.index[0]
        if post_filter_length < pd.Timedelta('730d'):
            raise ValueError(
                "Less than two years of data left after filtering")

    def _aggregate(self, energy_normalized, insolation):
        '''
        Return insolation-weighted normalized PV energy and the associated aggregated insolation

        Parameters
        ----------
        energy_normalized : pandas.Series
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
        aggregated = aggregation.aggregation_insol(
            energy_normalized, insolation, self.aggregation_freq)
        aggregated_insolation = insolation.resample(
            self.aggregation_freq).sum()

        return aggregated, aggregated_insolation

    def _yoy_degradation(self, energy_normalized, **kwargs):
        '''
        Perform year-on-year degradation analysis on insolation-weighted
        aggregated energy yield.

        Parameters
        ----------
        energy_normalized : pandas.Series
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
        self._filter_check(energy_normalized)
        yoy_rd, yoy_ci, yoy_info = degradation.degradation_year_on_year(
            energy_normalized, **kwargs)

        yoy_results = {
            'p50_rd': yoy_rd,
            'rd_confidence_interval': yoy_ci,
            'calc_info': yoy_info
        }

        return yoy_results

    def _srr_soiling(self, energy_normalized_daily, insolation_daily, **kwargs):
        '''
        Perform stochastic rate and recovery soiling analysis.

        Parameters
        ---------
        energy_normalized_daily : pandas.Series
            Time Series of insolation-weighted aggregated normalized PV energy
        insolation_daily : pandas.Series
            Time Series of insolation, aggregated at same level as energy_normalized_daily
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

        from rdtools import soiling

        daily_freq = pd.tseries.offsets.Day()
        if (energy_normalized_daily.index.freq != daily_freq or
                insolation_daily.index.freq != daily_freq):
            raise ValueError(
                'Soiling SRR analysis requires daily aggregation.')

        sr, sr_ci, soiling_info = soiling.soiling_srr(
            energy_normalized_daily, insolation_daily, **kwargs)

        srr_results = {
            'p50_sratio': sr,
            'sratio_confidence_interval': sr_ci,
            'calc_info': soiling_info
        }

        return srr_results

    def _sensor_preprocess(self):
        '''
        Perform sensor-based normalization, filtering, and aggregation.
        If optional parameter self.power_expected is passed in,
        normalize_with_expected_power will be used instead of pvwatts.
        '''
        if self.poa_global is None:
            raise ValueError(
                'poa_global must be available to perform _sensor_preprocess')

        if self.power_expected is None:
            # Thermal details required if power_expected is not manually set.
            if self.temperature_cell is None and self.temperature_ambient is None:
                raise ValueError('either cell or ambient temperature must be available '
                                 'to perform _sensor_preprocess')
            if self.temperature_cell is None:
                self.temperature_cell = self._calc_cell_temperature(
                    self.poa_global, self.temperature_ambient, self.windspeed)
            energy_normalized, insolation = self._pvwatts_norm(
                self.poa_global, self.temperature_cell)
        else:  # self.power_expected passed in by user
            energy_normalized, insolation = normalization.normalize_with_expected_power(
                self.pv_energy, self.power_expected, self.poa_global, pv_input='energy')
        self._filter(energy_normalized, 'sensor')
        aggregated, aggregated_insolation = self._aggregate(
            energy_normalized[self.sensor_filter], insolation[self.sensor_filter])
        self.sensor_aggregated_performance = aggregated
        self.sensor_aggregated_insolation = aggregated_insolation

    def _clearsky_preprocess(self):
        '''
        Perform clear-sky-based normalization, filtering, and aggregation.
        If optional parameter self.power_expected is passed in,
        normalize_with_expected_power will be used instead of pvwatts.
        '''
        try:
            if self.poa_global_clearsky is None:
                self._calc_clearsky_poa(model='isotropic')
        except AttributeError:
            raise AttributeError("No poa_global_clearsky. 'set_clearsky' must be run " +
                                 "prior to 'clearsky_analysis'")
        if self.temperature_cell_clearsky is None:
            if self.temperature_ambient_clearsky is None:
                self._calc_clearsky_tamb()
            self.temperature_cell_clearsky = self._calc_cell_temperature(
                self.poa_global_clearsky, self.temperature_ambient_clearsky, 0)
            # Note example notebook uses windspeed=0 in the clearskybranch
        if self.power_expected is None:
            cs_normalized, cs_insolation = self._pvwatts_norm(
                self.poa_global_clearsky, self.temperature_cell_clearsky)
        else:  # self.power_expected passed in by user
            cs_normalized, cs_insolation = normalization.normalize_with_expected_power(
                self.pv_energy, self.power_expected, self.poa_global_clearsky, pv_input='energy')
        self._filter(cs_normalized, 'clearsky')
        cs_aggregated, cs_aggregated_insolation = self._aggregate(
            cs_normalized[self.clearsky_filter], cs_insolation[self.clearsky_filter])
        self.clearsky_aggregated_performance = cs_aggregated
        self.clearsky_aggregated_insolation = cs_aggregated_insolation

    def sensor_analysis(self, analyses=['yoy_degradation'], yoy_kwargs={}, srr_kwargs={}):
        '''
        Perform entire sensor-based analysis workflow.
        Results are stored in self.results['sensor']

        Parameters
        ---------
        analyses : list
            Analyses to perform as a list of strings. Valid entries are 'yoy_degradation'
            and 'srr_soiling'
        yoy_kwargs : dict
            kwargs to pass to :py:func:`rdtools.degradation.degradation_year_on_year`
        srr_kwargs : dict
            kwargs to pass to :py:func:`rdtools.soiling.soiling_srr`

        Returns
        -------
        None
        '''

        self._sensor_preprocess()
        sensor_results = {}

        if 'yoy_degradation' in analyses:
            yoy_results = self._yoy_degradation(
                self.sensor_aggregated_performance, **yoy_kwargs)
            sensor_results['yoy_degradation'] = yoy_results

        if 'srr_soiling' in analyses:
            srr_results = self._srr_soiling(self.sensor_aggregated_performance,
                                            self.sensor_aggregated_insolation,
                                            **srr_kwargs)
            sensor_results['srr_soiling'] = srr_results

        self.results['sensor'] = sensor_results

    def clearsky_analysis(self, analyses=['yoy_degradation'], yoy_kwargs={}, srr_kwargs={}):
        '''
        Perform entire clear-sky-based analysis workflow. Results are stored
        in self.results['clearsky']

        Parameters
        ---------
        analyses : list
            Analyses to perform as a list of strings. Valid entries are 'yoy_degradation'
            and 'srr_soiling'
        yoy_kwargs : dict
            kwargs to pass to :py:func:`rdtools.degradation.degradation_year_on_year`
        srr_kwargs : dict
            kwargs to pass to :py:func:`rdtools.soiling.soiling_srr`

        Returns
        -------
        None
        '''

        self._clearsky_preprocess()
        clearsky_results = {}

        if 'yoy_degradation' in analyses:
            yoy_results = self._yoy_degradation(
                self.clearsky_aggregated_performance, **yoy_kwargs)
            clearsky_results['yoy_degradation'] = yoy_results

        if 'srr_soiling' in analyses:
            srr_results = self._srr_soiling(self.clearsky_aggregated_performance,
                                            self.clearsky_aggregated_insolation,
                                            **srr_kwargs)
            clearsky_results['srr_soiling'] = srr_results

        self.results['clearsky'] = clearsky_results

    def plot_degradation_summary(self, case, **kwargs):
        '''
        Return a figure of a scatter plot and a histogram summarizing degradation rate analysis.

        Parameters
        ----------
        case : str
            The workflow result to plot, allowed values are 'sensor' and 'clearsky'
        kwargs :
            Extra parameters passed to :py:func:`rdtools.plotting.degradation_summary_plots`

        Returns
        -------
        matplotlib.figure.Figure
        '''

        if case == 'sensor':
            results_dict = self.results['sensor']['yoy_degradation']
            aggregated = self.sensor_aggregated_performance
        elif case == 'clearsky':
            results_dict = self.results['clearsky']['yoy_degradation']
            aggregated = self.clearsky_aggregated_performance
        else:
            raise ValueError("case must be either 'sensor' or 'clearsky'")

        fig = plotting.degradation_summary_plots(
            results_dict['p50_rd'],
            results_dict['rd_confidence_interval'],
            results_dict['calc_info'], aggregated, **kwargs)
        return fig

    def plot_soiling_monte_carlo(self, case, **kwargs):
        '''
        Return a figure visualizing the Monte Carlo of soiling profiles used in
        stochastic rate and recovery soiling analysis.

        Parameters
        ----------
        case : str
            The workflow result to plot, allowed values are 'sensor' and 'clearsky'
        kwargs :
            Extra parameters passed to :py:func:`rdtools.plotting.soiling_monte_carlo_plot`

        Returns
        -------
        matplotlib.figure.Figure
        '''

        if case == 'sensor':
            results_dict = self.results['sensor']['srr_soiling']
            aggregated = self.sensor_aggregated_performance
        elif case == 'clearsky':
            results_dict = self.results['clearsky']['srr_soiling']
            aggregated = self.clearsky_aggregated_performance
        else:
            raise ValueError("case must be either 'sensor' or 'clearsky'")

        fig = plotting.soiling_monte_carlo_plot(
            results_dict['calc_info'], aggregated, **kwargs)

        return fig

    def plot_soiling_interval(self, case, **kwargs):
        '''
        Return a figure visualizing the valid soiling intervals used in
        stochastic rate and recovery soiling analysis.

        Parameters
        ----------
        case : str
            The workflow result to plot, allowed values are 'sensor' and 'clearsky'
        kwargs :
            Extra parameters passed to :py:func:`rdtools.plotting.soiling_interval_plot`

        Returns
        -------
        matplotlib.figure.Figure
        '''

        if case == 'sensor':
            results_dict = self.results['sensor']['srr_soiling']
            aggregated = self.sensor_aggregated_performance
        elif case == 'clearsky':
            results_dict = self.results['clearsky']['srr_soiling']
            aggregated = self.clearsky_aggregated_performance
        else:
            raise ValueError("case must be either 'sensor' or 'clearsky'")

        fig = plotting.soiling_interval_plot(
            results_dict['calc_info'], aggregated, **kwargs)

        return fig

    def plot_soiling_rate_histogram(self, case, **kwargs):
        '''
        Return a histogram of soiling rates found in the stochastic rate and recovery
        soiling analysis

        Parameters
        ----------
        case : str
            The workflow result to plot, allowed values are 'sensor' and 'clearsky'
        kwargs :
            Extra parameters passed to :py:func:`rdtools.plotting.soiling_rate_histogram`

        Returns
        -------
        matplotlib.figure.Figure
        '''

        if case == 'sensor':
            results_dict = self.results['sensor']['srr_soiling']
        elif case == 'clearsky':
            results_dict = self.results['clearsky']['srr_soiling']
        else:
            raise ValueError("case must be either 'sensor' or 'clearsky'")

        fig = plotting.soiling_rate_histogram(
            results_dict['calc_info'], **kwargs)

        return fig

    def plot_pv_vs_irradiance(self, case, alpha=0.01, **kwargs):
        '''
        Plot PV energy vs irradiance, useful in diagnosing things like timezone problems or
        transposition errors.

        Parameters
        ----------
        case: str
            The plane of array irradiance type to plot, allowed values are
            'sensor' and 'clearsky'
        alpha : float
            transparency of the scatter plot
        kwargs :
            Extra parameters passed to matplotlib.pyplot.axis.plot()

        Returns
        -------
        matplotlib.figure.Figure
        '''

        if case == 'sensor':
            poa = self.poa_global
        elif case == 'clearsky':
            poa = self.poa_global_clearsky
        else:
            raise ValueError("case must be either 'sensor' or 'clearsky'")

        to_plot = pd.merge(pd.DataFrame(poa), pd.DataFrame(
            self.pv_energy), left_index=True, right_index=True)

        fig, ax = plt.subplots()
        ax.plot(to_plot.iloc[:, 0], to_plot.iloc[:, 1],
                'o', alpha=alpha, **kwargs)
        ax.set_xlim(0, 1500)
        ax.set_xlabel('Irradiance (W/m$^2$)')
        ax.set_ylabel('PV Energy (Wh/timestep)')
        return fig

    def plot_degradation_timeseries(self, case, rolling_days=365, **kwargs):
        '''
        Plot resampled time series of degradation trend with time

        Parameters
        ----------
        case: str
            The workflow result to plot, allowed values are 'sensor' and 'clearsky'
        rolling_days: int, default 365
            Number of days for rolling window. Note that the window must contain
            at least 50% of datapoints to be included in rolling plot.
        kwargs :
            Extra parameters passed to :py:func:`rdtools.plotting.degradation_timeseries_plot`

        Returns
        -------
        matplotlib.figure.Figure
        '''

        if case == 'sensor':
            yoy_info = self.results['sensor']['yoy_degradation']['calc_info']
        elif case == 'clearsky':
            yoy_info = self.results['clearsky']['yoy_degradation']['calc_info']
        else:
            raise ValueError("case must be either 'sensor' or 'clearsky'")

        fig = plotting.degradation_timeseries_plot(yoy_info, rolling_days, **kwargs)
        return fig
