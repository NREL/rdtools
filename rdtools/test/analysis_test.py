from rdtools import RdAnalysis, normalization
from soiling_test import normalized_daily, times
from plotting_test import assert_isinstance
import pytest
import pvlib
import pandas as pd
import matplotlib.pyplot as plt


@pytest.fixture
def basic_parameters():
    # basic parameters (no time series data) for the RdAnalysis class

    parameters = dict(
        temperature_coefficient=-0.005,
        temperature_model={'a': -3.47, 'b': -0.0594, 'deltaT': 3}
    )

    return parameters


@pytest.fixture
def cs_input():
    # basic parameters (no time series data) for the RdAnalysis class
    loc = pvlib.location.Location(-23.762028, 133.874886,
                                  tz='Australia/North')
    cs_input = dict(
        pvlib_location=loc,
        pv_tilt=20,
        pv_azimuth=0
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
    poa = power * 1000
    ambient_temperature = power * 0 + 25
    basic_parameters['pv'] = power
    basic_parameters['poa'] = poa
    basic_parameters['ambient_temperature'] = ambient_temperature
    basic_parameters['interp_freq'] = 'H'
    return basic_parameters


@pytest.fixture
def sensor_analysis(sensor_parameters):
    rd_analysis = RdAnalysis(**sensor_parameters)
    rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
    return rd_analysis


@pytest.fixture
def sensor_analysis_exp_power(sensor_parameters):
    power_expected = normalization.pvwatts_dc_power(sensor_parameters['poa'],
                                                    power_dc_rated=1)
    sensor_parameters['power_expected'] = power_expected
    rd_analysis = RdAnalysis(**sensor_parameters)
    rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
    return rd_analysis


def test_sensor_analysis(sensor_analysis):
    yoy_results = sensor_analysis.results['sensor']['yoy_degradation']
    rd = yoy_results['p50_rd']
    ci = yoy_results['rd_confidence_interval']

    assert -1 == pytest.approx(rd, abs=1e-2)
    assert [-1, -1] == pytest.approx(ci, abs=1e-2)


def test_sensor_analysis_energy(sensor_parameters, sensor_analysis):
    sensor_parameters['pv'] = sensor_analysis.pv_energy
    sensor_parameters['pv_input'] = 'energy'
    sensor_analysis2 = RdAnalysis(**sensor_parameters)
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


@pytest.fixture
def clearsky_parameters(basic_parameters, sensor_parameters,
                        cs_input, degradation_trend):
    # clear-sky weather data.  Uses RdAnalysis's internal clear-sky
    # functions to generate the data.
    rd_analysis = RdAnalysis(**sensor_parameters)
    rd_analysis.set_clearsky(**cs_input)
    rd_analysis.clearsky_preprocess()
    poa = rd_analysis.clearsky_poa
    clearsky_parameters = basic_parameters
    clearsky_parameters['poa'] = poa
    clearsky_parameters['pv'] = poa * degradation_trend
    return clearsky_parameters


@pytest.fixture
def clearsky_analysis(cs_input, clearsky_parameters):
    rd_analysis = RdAnalysis(**clearsky_parameters)
    rd_analysis.set_clearsky(**cs_input)
    rd_analysis.clearsky_analysis(analyses=['yoy_degradation'])
    return rd_analysis


@pytest.fixture
def clearsky_optional(cs_input, clearsky_analysis):
    # optional parameters to exercise other branches
    times = clearsky_analysis.poa.index
    extras = dict(
        clearsky_poa=clearsky_analysis.clearsky_poa,
        clearsky_cell_temperature=clearsky_analysis.clearsky_cell_temperature,
        clearsky_ambient_temperature=clearsky_analysis.clearsky_ambient_temperature,
        pv_tilt=pd.Series(cs_input['pv_tilt'], index=times),
        pv_azimuth=pd.Series(cs_input['pv_azimuth'], index=times)
    )
    return extras


def test_clearsky_analysis(clearsky_analysis):
    yoy_results = clearsky_analysis.results['clearsky']['yoy_degradation']
    ci = yoy_results['rd_confidence_interval']
    rd = yoy_results['p50_rd']
    print(ci)
    assert -4.73 == pytest.approx(rd, abs=1e-2)
    assert [-4.74, -4.72] == pytest.approx(ci, abs=1e-2)


def test_clearsky_analysis_optional(clearsky_analysis, clearsky_parameters, clearsky_optional):

    clearsky_analysis.set_clearsky(**clearsky_optional)
    clearsky_analysis.clearsky_analysis()
    yoy_results = clearsky_analysis.results['clearsky']['yoy_degradation']
    ci = yoy_results['rd_confidence_interval']
    rd = yoy_results['p50_rd']
    print(f'ci:{ci}')
    assert -4.73 == pytest.approx(rd, abs=1e-2)
    assert [-4.74, -4.72] == pytest.approx(ci, abs=1e-2)


@pytest.fixture
def soiling_parameters(basic_parameters, normalized_daily, cs_input):
    # parameters for soiling analysis with RdAnalysis
    power = normalized_daily.resample('1h').interpolate()
    return dict(
        pv=power,
        poa=power * 0 + 1000,
        cell_temperature=power * 0 + 25,
        temperature_coefficient=0,
        interp_freq='D',
    )


@pytest.fixture
def soiling_analysis_sensor(soiling_parameters):
    soiling_analysis = RdAnalysis(**soiling_parameters)
    soiling_analysis.sensor_analysis(analyses=['srr_soiling'],
                                     srr_kwargs={'reps': 10})
    return soiling_analysis


@pytest.fixture
def soiling_analysis_clearsky(soiling_parameters, cs_input):
    soiling_analysis = RdAnalysis(**soiling_parameters)
    soiling_analysis.set_clearsky(**cs_input)
    soiling_analysis.clearsky_analysis(analyses=['srr_soiling'],
                                       srr_kwargs={'reps': 10})
    return soiling_analysis


def test_srr_soiling(soiling_analysis_sensor):
    srr_results = soiling_analysis_sensor.results['sensor']['srr_soiling']
    sratio = srr_results['p50_sratio']
    ci = srr_results['sratio_confidence_interval']
    renorm_factor = srr_results['calc_info']['renormalizing_factor']
    print(f'soiling ci:{ci}')
    assert 0.959 == pytest.approx(sratio, abs=1e-3),\
        'Soiling ratio different from expected value in RdAnalysis.srr_soiling'
    assert [0.95, 0.96] == pytest.approx(ci, abs=1e-2),\
        'Soiling confidence interval different from expected value in RdAnalysis.srr_soiling'
    assert 0.974 == pytest.approx(renorm_factor, abs=1e-3),\
        'Renormalization factor different from expected value in RdAnalysis.srr_soiling'


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

    rdtemp = RdAnalysis(sensor_parameters['pv'])
    with pytest.raises(ValueError, match='poa must be available'):
        rdtemp.sensor_preprocess()

    # no temperature
    rdtemp = RdAnalysis(sensor_parameters['pv'],
                        poa=sensor_parameters['poa'])
    with pytest.raises(ValueError, match='either cell or ambient temperature'):
        rdtemp.sensor_preprocess()

    # clearsky analysis with no tilt/azm
    clearsky_analysis.pv_tilt = None
    clearsky_analysis.clearsky_poa = None
    with pytest.raises(ValueError, match='pv_tilt and pv_azimuth must be provided'):
        clearsky_analysis.clearsky_preprocess()

    # clearsky analysis with no pvlib.loc
    clearsky_analysis.pvlib_location = None
    with pytest.raises(ValueError, match='pvlib location must be provided'):
        clearsky_analysis.clearsky_preprocess()
