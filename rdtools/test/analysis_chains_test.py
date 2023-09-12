from rdtools import TrendAnalysis, normalization
from conftest import assert_isinstance, assert_warnings
import pytest
import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@pytest.fixture
def basic_parameters():
    # basic parameters (no time series data) for the TrendAnalysis class

    parameters = dict(
        gamma_pdc=-0.005,
        temperature_model={'a': -3.47, 'b': -0.0594, 'deltaT': 3}
    )

    return parameters


@pytest.fixture
def cs_input():
    # basic parameters (no time series data) for the TrendAnalysis class
    loc = pvlib.location.Location(-23.762028, 133.874886,
                                  tz='Australia/North')
    cs_input = dict(
        pvlib_location=loc,
        pv_tilt=20,
        pv_azimuth=0,
        solar_position_method='ephemeris',  # just to improve test execution speed
    )

    return cs_input


@pytest.fixture
def degradation_trend(basic_parameters, cs_input):
    # smooth linear multi-year decline from 1.0 from degradation_test.py

    # hide this import inside the function so that pytest doesn't find it
    # and run the degradation tests as a part of this module
    from degradation_test import DegradationTestCase

    rd = -0.05
    input_freq = 'H'
    degradation_trend = DegradationTestCase.get_corr_energy(rd, input_freq)
    tz = cs_input['pvlib_location'].tz
    return degradation_trend.tz_localize(tz)


@pytest.fixture
def sensor_parameters(basic_parameters, degradation_trend):
    # basic parameters plus time series data
    power = degradation_trend
    poa_global = power * 1000
    temperature_ambient = power * 0 + 25
    basic_parameters['pv'] = power
    basic_parameters['poa_global'] = poa_global
    basic_parameters['temperature_ambient'] = temperature_ambient
    basic_parameters['interp_freq'] = 'H'
    return basic_parameters


@pytest.fixture
def sensor_analysis(sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters)
    rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
    return rd_analysis


@pytest.fixture
def sensor_analysis_exp_power(sensor_parameters):
    power_expected = normalization.pvwatts_dc_power(sensor_parameters['poa_global'],
                                                    power_dc_rated=1)
    sensor_parameters['power_expected'] = power_expected
    rd_analysis = TrendAnalysis(**sensor_parameters)
    rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
    return rd_analysis


def test_interpolation(basic_parameters, degradation_trend):

    power = degradation_trend
    shifted_index = power.index + pd.to_timedelta('8 minutes')

    dummy_series = power * 0 + 25
    dummy_series.index = shifted_index

    basic_parameters['pv'] = power
    basic_parameters['poa_global'] = dummy_series
    basic_parameters['temperature_ambient'] = dummy_series
    basic_parameters['temperature_cell'] = dummy_series
    basic_parameters['windspeed'] = dummy_series
    basic_parameters['power_expected'] = dummy_series
    basic_parameters['interp_freq'] = 'H'

    rd_analysis = TrendAnalysis(**basic_parameters)

    pd.testing.assert_index_equal(rd_analysis.pv_energy.index,
                                  rd_analysis.poa_global.index[1:])
    pd.testing.assert_index_equal(rd_analysis.pv_energy.index,
                                  rd_analysis.temperature_ambient.index[1:])
    pd.testing.assert_index_equal(rd_analysis.pv_energy.index,
                                  rd_analysis.temperature_cell.index[1:])
    pd.testing.assert_index_equal(rd_analysis.pv_energy.index,
                                  rd_analysis.windspeed.index[1:])
    pd.testing.assert_index_equal(rd_analysis.pv_energy.index,
                                  rd_analysis.power_expected.index[1:])

    rd_analysis.set_clearsky(pv_azimuth=dummy_series,
                             pv_tilt=dummy_series,
                             poa_global_clearsky=dummy_series,
                             temperature_cell_clearsky=dummy_series,
                             temperature_ambient_clearsky=dummy_series)

    pd.testing.assert_index_equal(rd_analysis.pv_energy.index,
                                  rd_analysis.pv_azimuth.index)
    pd.testing.assert_index_equal(rd_analysis.pv_energy.index,
                                  rd_analysis.pv_tilt.index)
    pd.testing.assert_index_equal(rd_analysis.pv_energy.index,
                                  rd_analysis.poa_global_clearsky.index)
    pd.testing.assert_index_equal(rd_analysis.pv_energy.index,
                                  rd_analysis.temperature_cell_clearsky.index)
    pd.testing.assert_index_equal(rd_analysis.pv_energy.index,
                                  rd_analysis.temperature_ambient_clearsky.index)


def test_sensor_analysis(sensor_analysis):
    yoy_results = sensor_analysis.results['sensor']['yoy_degradation']
    rd = yoy_results['p50_rd']
    ci = yoy_results['rd_confidence_interval']

    assert -1 == pytest.approx(rd, abs=1e-2)
    assert [-1, -1] == pytest.approx(ci, abs=1e-2)


def test_sensor_analysis_energy(sensor_parameters, sensor_analysis):
    sensor_parameters['pv'] = sensor_analysis.pv_energy
    sensor_parameters['pv_input'] = 'energy'
    sensor_analysis2 = TrendAnalysis(**sensor_parameters)
    sensor_analysis2.pv_power = sensor_analysis.pv_power
    sensor_analysis2.sensor_analysis(analyses=['yoy_degradation'])
    yoy_results = sensor_analysis2.results['sensor']['yoy_degradation']
    rd = yoy_results['p50_rd']
    ci = yoy_results['rd_confidence_interval']

    assert -1 == pytest.approx(rd, abs=1e-2)
    assert [-1, -1] == pytest.approx(ci, abs=1e-2)


def test_sensor_analysis_exp_power(sensor_analysis_exp_power):
    yoy_results = sensor_analysis_exp_power.results['sensor']['yoy_degradation']
    rd = yoy_results['p50_rd']
    ci = yoy_results['rd_confidence_interval']

    assert 0 == pytest.approx(rd, abs=1e-2)
    assert [0, 0] == pytest.approx(ci, abs=1e-2)


def test_sensor_analysis_power_dc_rated(sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
    yoy_results = rd_analysis.results['sensor']['yoy_degradation']
    rd = yoy_results['p50_rd']
    ci = yoy_results['rd_confidence_interval']

    assert -1 == pytest.approx(rd, abs=1e-2)
    assert [-1, -1] == pytest.approx(ci, abs=1e-2)


def test_sensor_analysis_ad_hoc_filter(sensor_parameters):
    # by excluding all but a few points, we should trigger the <2yr error
    filt = pd.Series(False, index=sensor_parameters['pv'].index)
    filt.iloc[-100:] = True
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.filter_params['ad_hoc_filter'] = filt
    with pytest.raises(ValueError, match="Less than two years of data left after filtering"):
        rd_analysis.sensor_analysis(analyses=['yoy_degradation'])


def test_sensor_analysis_aggregated_ad_hoc_filter(sensor_parameters):
    # by excluding all but a few points, we should trigger the <2yr error
    filt = pd.Series(False,
                     index=sensor_parameters['pv'].index)
    filt = filt.resample('1D').first().dropna(how='all')
    filt.iloc[-500:] = True
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.filter_params_aggregated['ad_hoc_filter'] = filt
    with pytest.raises(ValueError, match="Less than two years of data left after filtering"):
        rd_analysis.sensor_analysis(analyses=['yoy_degradation'])


def test_filter_components(sensor_parameters):
    poa = sensor_parameters['poa_global']
    poa_filter = (poa > 200) & (poa < 1200)
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
    assert (poa_filter ==
            rd_analysis.sensor_filter_components['poa_filter']).all()


def test_aggregated_filter_components(sensor_parameters):
    daily_ad_hoc_filter = pd.Series(True,
                                    index=sensor_parameters['pv'].index)
    daily_ad_hoc_filter[:600] = False
    daily_ad_hoc_filter = daily_ad_hoc_filter.resample(
        '1D').first().dropna(how='all')
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.filter_params = {}  # disable all index-based filters
    rd_analysis.filter_params_aggregated['ad_hoc_filter'] = daily_ad_hoc_filter
    rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
    assert (daily_ad_hoc_filter ==
            rd_analysis.sensor_filter_components_aggregated['ad_hoc_filter']).all()


def test_filter_components_no_filters(sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.filter_params = {}  # disable all filters
    rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
    expected = pd.Series(True, index=rd_analysis.pv_energy.index)
    pd.testing.assert_series_equal(rd_analysis.sensor_filter, expected)
    assert rd_analysis.sensor_filter_components.empty


def test_aggregated_filter_components_no_filters(sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.filter_params = {}  # disable all index-based filters
    rd_analysis.filter_params_aggregated = {}  # disable all daily filters
    rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
    expected = pd.Series(True, index=rd_analysis.pv_energy.index)
    daily_expected = expected.resample('1D').first().dropna(how='all')
    pd.testing.assert_series_equal(rd_analysis.sensor_filter_aggregated,
                                   daily_expected)
    assert rd_analysis.sensor_filter_components.empty


@pytest.mark.parametrize('workflow', ['sensor', 'clearsky'])
def test_filter_ad_hoc_warnings(workflow, sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.set_clearsky(pvlib_location=pvlib.location.Location(40, -80),
                             poa_global_clearsky=rd_analysis.poa_global)
    # warning for incomplete index
    ad_hoc_filter = pd.Series(True, index=sensor_parameters['pv'].index[:-5])
    rd_analysis.filter_params['ad_hoc_filter'] = ad_hoc_filter
    with pytest.warns(UserWarning, match='ad_hoc_filter index does not match index'):
        if workflow == 'sensor':
            rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
            components = rd_analysis.sensor_filter_components
        else:
            rd_analysis.clearsky_analysis(analyses=['yoy_degradation'])
            components = rd_analysis.clearsky_filter_components

    # missing values set to True
    assert components['ad_hoc_filter'].all()

    # warning about NaNs
    ad_hoc_filter = pd.Series(True, index=sensor_parameters['pv'].index)
    ad_hoc_filter.iloc[10] = np.nan
    rd_analysis.filter_params['ad_hoc_filter'] = ad_hoc_filter
    with pytest.warns(UserWarning, match='ad_hoc_filter contains NaN values; setting to False'):
        if workflow == 'sensor':
            rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
            components = rd_analysis.sensor_filter_components
        else:
            rd_analysis.clearsky_analysis(analyses=['yoy_degradation'])
            components = rd_analysis.clearsky_filter_components

    # NaN values set to False
    assert not components['ad_hoc_filter'].iloc[10]
    assert components.drop(components.index[10])['ad_hoc_filter'].all()


@pytest.mark.parametrize('workflow', ['sensor', 'clearsky'])
def test_aggregated_filter_ad_hoc_warnings(workflow, sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.set_clearsky(pvlib_location=pvlib.location.Location(40, -80),
                             poa_global_clearsky=rd_analysis.poa_global)
    # disable all filters outside of CSI
    rd_analysis.filter_params = {'csi_filter': {}}
    # warning for incomplete index
    daily_ad_hoc_filter = pd.Series(True,
                                    index=sensor_parameters['pv'].index[:-5])
    daily_ad_hoc_filter = daily_ad_hoc_filter.resample(
        '1D').first().dropna(how='all')
    rd_analysis.filter_params_aggregated['ad_hoc_filter'] = daily_ad_hoc_filter
    with pytest.warns(UserWarning, match='ad_hoc_filter index does not match index'):
        if workflow == 'sensor':
            rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
            components = rd_analysis.sensor_filter_components_aggregated
        else:
            rd_analysis.clearsky_analysis(analyses=['yoy_degradation'])
            components = rd_analysis.clearsky_filter_components_aggregated

    # missing values set to True
    assert components['ad_hoc_filter'].all()

    # warning about NaNs
    rd_analysis_2 = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis_2.set_clearsky(pvlib_location=pvlib.location.Location(40, -80),
                               poa_global_clearsky=rd_analysis_2.poa_global)
    # disable all filters outside of CSI
    rd_analysis_2.filter_params = {'csi_filter': {}}
    daily_ad_hoc_filter = pd.Series(True, index=sensor_parameters['pv'].index)
    daily_ad_hoc_filter = daily_ad_hoc_filter.resample(
        '1D').first().dropna(how='all')
    daily_ad_hoc_filter.iloc[10] = np.nan
    rd_analysis_2.filter_params_aggregated['ad_hoc_filter'] = daily_ad_hoc_filter
    with pytest.warns(UserWarning, match='ad_hoc_filter contains NaN values; setting to False'):
        if workflow == 'sensor':
            rd_analysis_2.sensor_analysis(analyses=['yoy_degradation'])
            components = rd_analysis_2.sensor_filter_components_aggregated
        else:
            rd_analysis_2.clearsky_analysis(analyses=['yoy_degradation'])
            components = rd_analysis_2.clearsky_filter_components_aggregated

    # NaN values set to False
    assert not components['ad_hoc_filter'].iloc[10]
    assert components.drop(components.index[10])['ad_hoc_filter'].all()


def test_cell_temperature_model_invalid(sensor_parameters):
    wind = pd.Series(0, index=sensor_parameters['pv'].index)
    sensor_parameters.pop('temperature_model')
    rd_analysis = TrendAnalysis(**sensor_parameters, windspeed=wind,
                                temperature_model={'bad': True})
    with pytest.raises(ValueError, match='pvlib temperature_model entry is neither'):
        rd_analysis.sensor_analysis()


def test_no_gamma_pdc(sensor_parameters):
    sensor_parameters.pop('gamma_pdc')
    rd_analysis = TrendAnalysis(**sensor_parameters)

    with pytest.warns(UserWarning) as record:
        rd_analysis.sensor_analysis()

    assert_warnings(["Temperature coefficient not passed"], record)


@pytest.fixture
def clearsky_parameters(basic_parameters, sensor_parameters,
                        cs_input, degradation_trend):
    # clear-sky weather data.  Uses TrendAnalysis's internal clear-sky
    # functions to generate the data.
    rd_analysis = TrendAnalysis(**sensor_parameters)
    rd_analysis.set_clearsky(**cs_input)
    rd_analysis._clearsky_preprocess()
    poa = rd_analysis.poa_global_clearsky
    clearsky_parameters = basic_parameters
    clearsky_parameters['poa_global'] = poa
    clearsky_parameters['pv'] = poa * degradation_trend
    return clearsky_parameters


@pytest.fixture
def clearsky_analysis(cs_input, clearsky_parameters):
    rd_analysis = TrendAnalysis(**clearsky_parameters)
    rd_analysis.set_clearsky(**cs_input)
    rd_analysis.clearsky_analysis(analyses=['yoy_degradation'])
    return rd_analysis


@pytest.fixture
def clearsky_optional(cs_input, clearsky_analysis):
    # optional parameters to exercise other branches
    times = clearsky_analysis.poa_global.index
    extras = dict(
        poa_global_clearsky=clearsky_analysis.poa_global_clearsky,
        temperature_cell_clearsky=clearsky_analysis.temperature_cell_clearsky,
        temperature_ambient_clearsky=clearsky_analysis.temperature_ambient_clearsky,
        pv_tilt=pd.Series(cs_input['pv_tilt'], index=times),
        pv_azimuth=pd.Series(cs_input['pv_azimuth'], index=times),
        solar_position_method='ephemeris',  # just to improve test execution speed
    )
    return extras


def test_clearsky_analysis(clearsky_analysis):
    yoy_results = clearsky_analysis.results['clearsky']['yoy_degradation']
    ci = yoy_results['rd_confidence_interval']
    rd = yoy_results['p50_rd']
    print(ci)
    assert -4.70 == pytest.approx(rd, abs=1e-2)
    assert [-4.71, -4.69] == pytest.approx(ci, abs=1e-2)


def test_clearsky_analysis_optional(clearsky_analysis, clearsky_parameters, clearsky_optional):

    clearsky_analysis.set_clearsky(**clearsky_optional)
    clearsky_analysis.clearsky_analysis()
    yoy_results = clearsky_analysis.results['clearsky']['yoy_degradation']
    ci = yoy_results['rd_confidence_interval']
    rd = yoy_results['p50_rd']
    print(f'ci:{ci}')
    assert -4.70 == pytest.approx(rd, abs=1e-2)
    assert [-4.71, -4.69] == pytest.approx(ci, abs=1e-2)


@pytest.fixture
def clearsky_analysis_exp_power(clearsky_parameters, clearsky_optional):
    power_expected = normalization.pvwatts_dc_power(clearsky_parameters['poa_global'],
                                                    power_dc_rated=1)
    clearsky_parameters['power_expected'] = power_expected
    rd_analysis = TrendAnalysis(**clearsky_parameters)
    rd_analysis.set_clearsky(**clearsky_optional)
    rd_analysis.clearsky_analysis(analyses=['yoy_degradation'])
    return rd_analysis


def test_clearsky_analysis_exp_power(clearsky_analysis_exp_power):
    yoy_results = clearsky_analysis_exp_power.results['clearsky']['yoy_degradation']
    rd = yoy_results['p50_rd']
    ci = yoy_results['rd_confidence_interval']

    assert -5.128 == pytest.approx(rd, abs=1e-2)
    assert [-5.128, -5.127] == pytest.approx(ci, abs=1e-2)


def test_no_set_clearsky(clearsky_parameters):
    rd_analysis = TrendAnalysis(**clearsky_parameters)
    with pytest.raises(AttributeError, match="No poa_global_clearsky. 'set_clearsky' must be run"):
        rd_analysis.clearsky_analysis()


def test_solar_position_method_passthrough(sensor_analysis, mocker):
    # verify that the solar_position_method kwarg is passed through to pvlib correctly
    spy = mocker.spy(pvlib.solarposition, 'get_solarposition')
    for method in ['nrel_numpy', 'ephemeris']:
        sensor_analysis.set_clearsky(pvlib.location.Location(40, -80), pv_tilt=20, pv_azimuth=180,
                                     solar_position_method=method)
        sensor_analysis._calc_clearsky_poa()
        assert spy.call_args[1]['method'] == method


def test_index_mismatch():
    # GH #277
    times = pd.date_range('2019-01-01', '2022-01-01', freq='15min')
    pv = pd.Series(1.0, index=times)
    # low-frequency weather inputs
    dummy_series = pd.Series(1.0, index=times[::4])
    keys = ['poa_global', 'temperature_cell',
            'temperature_ambient', 'power_expected', 'windspeed']
    kwargs = {key: dummy_series.copy() for key in keys}
    rd_analysis = TrendAnalysis(pv, **kwargs)
    for key in keys:
        interpolated_series = getattr(rd_analysis, key)
        assert interpolated_series.index.equals(times)

    cs_keys = ['poa_global_clearsky', 'temperature_cell_clearsky', 'temperature_ambient_clearsky',
               'pv_azimuth', 'pv_tilt']
    cs_kwargs = {key: dummy_series.copy() for key in cs_keys}
    rd_analysis.set_clearsky(**cs_kwargs)
    for key in cs_keys:
        interpolated_series = getattr(rd_analysis, key)
        assert interpolated_series.index.equals(times[1:])


@pytest.fixture
def soiling_parameters(basic_parameters, soiling_normalized_daily, cs_input):
    # parameters for soiling analysis with TrendAnalysis
    power = soiling_normalized_daily.resample('1h').interpolate()
    return dict(
        pv=power,
        poa_global=power * 0 + 1000,
        temperature_cell=power * 0 + 25,
        gamma_pdc=0,
        interp_freq='D',
    )


@pytest.fixture
def soiling_analysis_sensor(soiling_parameters):
    soiling_analysis = TrendAnalysis(**soiling_parameters)
    np.random.seed(1977)
    soiling_analysis.sensor_analysis(analyses=['srr_soiling'],
                                     srr_kwargs={'reps': 10})
    return soiling_analysis


@pytest.fixture
def soiling_analysis_clearsky(soiling_parameters, cs_input):
    soiling_analysis = TrendAnalysis(**soiling_parameters)
    soiling_analysis.set_clearsky(**cs_input)
    np.random.seed(1977)
    with pytest.warns(UserWarning, match='20% or more of the daily data'):
        soiling_analysis.clearsky_analysis(analyses=['srr_soiling'],
                                           srr_kwargs={'reps': 10})
    return soiling_analysis


def test_srr_soiling(soiling_analysis_sensor):
    srr_results = soiling_analysis_sensor.results['sensor']['srr_soiling']
    sratio = srr_results['p50_sratio']
    ci = srr_results['sratio_confidence_interval']
    renorm_factor = srr_results['calc_info']['renormalizing_factor']
    print(f'soiling ci:{ci}')
    assert 0.965 == pytest.approx(sratio, abs=1e-3), \
        'Soiling ratio different from expected value in TrendAnalysis.srr_soiling'
    assert [0.96, 0.97] == pytest.approx(ci, abs=1e-2), \
        'Soiling confidence interval different from expected value in TrendAnalysis.srr_soiling'
    assert 0.974 == pytest.approx(renorm_factor, abs=1e-3), \
        'Renormalization factor different from expected value in TrendAnalysis.srr_soiling'


def test_plot_degradation(sensor_analysis):
    assert_isinstance(
        sensor_analysis.plot_degradation_summary('sensor'), plt.Figure)
    assert_isinstance(
        sensor_analysis.plot_pv_vs_irradiance('sensor'), plt.Figure)


def test_plot_cs(clearsky_analysis):
    assert_isinstance(
        clearsky_analysis.plot_degradation_summary('clearsky'), plt.Figure)
    assert_isinstance(
        clearsky_analysis.plot_pv_vs_irradiance('clearsky'), plt.Figure)


def test_plot_soiling(soiling_analysis_sensor):
    assert_isinstance(
        soiling_analysis_sensor.plot_soiling_monte_carlo('sensor'), plt.Figure)
    assert_isinstance(
        soiling_analysis_sensor.plot_soiling_interval('sensor'), plt.Figure)
    assert_isinstance(
        soiling_analysis_sensor.plot_soiling_rate_histogram('sensor'), plt.Figure)


def test_plot_soiling_cs(soiling_analysis_clearsky):
    assert_isinstance(
        soiling_analysis_clearsky.plot_soiling_monte_carlo('clearsky'), plt.Figure)
    assert_isinstance(
        soiling_analysis_clearsky.plot_soiling_interval('clearsky'), plt.Figure)
    assert_isinstance(
        soiling_analysis_clearsky.plot_soiling_rate_histogram('clearsky'), plt.Figure)


def test_errors(sensor_parameters, clearsky_analysis):

    rdtemp = TrendAnalysis(sensor_parameters['pv'])
    with pytest.raises(ValueError, match='poa_global must be available'):
        rdtemp._sensor_preprocess()

    # no temperature
    rdtemp = TrendAnalysis(sensor_parameters['pv'],
                           poa_global=sensor_parameters['poa_global'])
    with pytest.raises(ValueError, match='either cell or ambient temperature'):
        rdtemp._sensor_preprocess()

    # clearsky analysis with no tilt/azm
    clearsky_analysis.pv_tilt = None
    clearsky_analysis.poa_global_clearsky = None
    with pytest.raises(ValueError, match='pv_tilt and pv_azimuth must be provided'):
        clearsky_analysis._clearsky_preprocess()

    # clearsky analysis with no pvlib.loc
    clearsky_analysis.pvlib_location = None
    with pytest.raises(ValueError, match='pvlib location must be provided'):
        clearsky_analysis._clearsky_preprocess()


@pytest.mark.parametrize('method_name', ['plot_degradation_summary',
                                         'plot_soiling_monte_carlo',
                                         'plot_soiling_interval',
                                         'plot_soiling_rate_histogram',
                                         'plot_pv_vs_irradiance'])
def test_plot_errors(method_name, sensor_analysis):
    func = getattr(sensor_analysis, method_name)
    with pytest.raises(ValueError, match="case must be either 'sensor' or 'clearsky'"):
        func(case='bad')


def test_plot_degradation_timeseries(sensor_analysis, clearsky_analysis):
    assert_isinstance(
        sensor_analysis.plot_degradation_timeseries('sensor'), plt.Figure)
    assert_isinstance(
        clearsky_analysis.plot_degradation_timeseries('clearsky'), plt.Figure)
