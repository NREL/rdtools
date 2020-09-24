from rdtools import analysis
from soiling_test import normalized_daily, times
from plotting_test import assert_isinstance
import pytest
import pvlib
import pandas as pd
import matplotlib.pyplot as plt


@pytest.fixture
def basic_parameters():
    # basic parameters (no time series data) for the RdAnalysis class
    loc = pvlib.location.Location(-23.762028, 133.874886,
                                  tz='Australia/North')
    parameters = dict(
        temperature_coefficient=-0.005,
        pvlib_location=loc,
        pv_tilt=20,
        pv_azimuth=0,
        temperature_model={'a': -3.47, 'b': -0.0594, 'deltaT': 3}
    )
    return parameters


@pytest.fixture
def degradation_trend(basic_parameters):
    # smooth linear multi-year decline from 1.0 from degradation_test.py

    # hide this import inside the function so that pytest doesn't find it
    # and run the degradation tests as a part of this module
    from degradation_test import DegradationTestCase

    rd = -0.05
    input_freq = 'H'
    degradation_trend = DegradationTestCase.get_corr_energy(rd, input_freq)
    tz = basic_parameters['pvlib_location'].tz
    return degradation_trend.tz_localize(tz)


@pytest.fixture
def sensor_parameters(basic_parameters, degradation_trend):
    # basic parameters plus time series data
    power = degradation_trend
    poa = power*1000
    ambient_temperature = power*0+25
    basic_parameters['pv'] = power
    basic_parameters['poa'] = poa
    basic_parameters['ambient_temperature'] = ambient_temperature
    return basic_parameters


@pytest.fixture
def sensor_analysis(sensor_parameters):
    rd_analysis = analysis.RdAnalysis(**sensor_parameters)
    rd_analysis.sensor_analysis(analyses=['yoy_degradation'])
    return rd_analysis


def test_sensor_analysis(sensor_analysis):
    yoy_results = sensor_analysis.results['sensor']['yoy_degradation']
    rd = yoy_results['p50_rd']
    ci = yoy_results['rd_confidence_interval']

    assert -1 == pytest.approx(rd, abs=1e-2)
    assert [-1, -1] == pytest.approx(ci, abs=1e-2)


@pytest.fixture
def clearsky_parameters(basic_parameters, sensor_parameters,
                        degradation_trend):
    # clear-sky weather data.  Uses RdAnalysis's internal clear-sky
    # functions to generate the data.
    rd_analysis = analysis.RdAnalysis(**sensor_parameters)
    rd_analysis.clearsky_preprocess()
    poa = rd_analysis.clearsky_poa
    basic_parameters['poa'] = poa
    basic_parameters['pv'] = poa * degradation_trend
    return basic_parameters


@pytest.fixture
def clearsky_analysis(clearsky_parameters):
    rd_analysis = analysis.RdAnalysis(**clearsky_parameters)
    rd_analysis.clearsky_analysis(analyses=['yoy_degradation'])
    return rd_analysis


@pytest.fixture
def clearsky_optional(clearsky_parameters, clearsky_analysis):
    # optional parameters to exercise other branches
    times = clearsky_analysis.poa.index
    extras = dict(
        clearsky_poa=clearsky_analysis.clearsky_poa,
        clearsky_cell_temperature=clearsky_analysis.clearsky_cell_temperature,
        clearsky_ambient_temperature=clearsky_analysis.clearsky_ambient_temperature,
        pv=clearsky_analysis.pv_energy,

        # series orientation instead of scalars to exercise interpolation
        pv_tilt=pd.Series(clearsky_parameters['pv_tilt'], index=times),
        pv_azimuth=pd.Series(clearsky_parameters['pv_azimuth'], index=times)
    )
    # merge dicts, favoring new params over the ones in clearsky_parameters
    return {**clearsky_parameters, **extras}


def test_clearsky_analysis(clearsky_analysis):
    yoy_results = clearsky_analysis.results['clearsky']['yoy_degradation']
    ci = yoy_results['rd_confidence_interval']
    rd = yoy_results['p50_rd']

    assert -4.744 == pytest.approx(rd, abs=1e-3)
    assert [-4.756, -4.734] == pytest.approx(ci, abs=1e-3)


def test_clearsky_analysis_optional(clearsky_parameters, clearsky_optional):
    rd_analysis = analysis.RdAnalysis(**clearsky_optional,
                                      pv_input='energy')
    rd_analysis.pv_power = clearsky_parameters['pv']
    rd_analysis.clearsky_analysis()
    yoy_results = rd_analysis.results['clearsky']['yoy_degradation']
    ci = yoy_results['rd_confidence_interval']
    rd = yoy_results['p50_rd']

    assert -4.744 == pytest.approx(rd, abs=1e-2)
    assert [-4.756, -4.734] == pytest.approx(ci, abs=1e-3)


@pytest.fixture
def soiling_parameters(basic_parameters, normalized_daily):
    # parameters for soiling analysis with RdAnalysis
    power = normalized_daily.resample('1h').interpolate()
    return dict(
        pv=power,
        poa=power * 0 + 1000,
        cell_temperature=power * 0 + 25,
        pvlib_location=basic_parameters['pvlib_location'],
        temperature_coefficient=0,
        interp_freq='D',
        pv_tilt=basic_parameters['pv_tilt'],
        pv_azimuth=basic_parameters['pv_azimuth']
    )


@pytest.fixture
def soiling_analysis_sensor(soiling_parameters):
    soiling_analysis = analysis.RdAnalysis(**soiling_parameters)
    soiling_analysis.sensor_analysis(analyses=['srr_soiling'],
                                     srr_kwargs={'reps': 10})
    return soiling_analysis


@pytest.fixture
def soiling_analysis_clearsky(soiling_parameters):
    soiling_analysis = analysis.RdAnalysis(**soiling_parameters)
    soiling_analysis.clearsky_analysis(analyses=['srr_soiling'],
                                       srr_kwargs={'reps': 10})
    return soiling_analysis


def test_srr_soiling(soiling_analysis_sensor):
    srr_results = soiling_analysis_sensor.results['sensor']['srr_soiling']
    sratio = srr_results['p50_sratio']
    ci = srr_results['sratio_confidence_interval']
    renorm_factor = srr_results['calc_info']['renormalizing_factor']

    assert 0.9583 == pytest.approx(sratio, abs=1e-4),\
        'Soiling ratio different from expected value in RdAnalysis.srr_soiling'
    assert [0.9552, 0.9607] == pytest.approx(ci, abs=1e-4),\
        'Soiling confidence interval different from expected value in RdAnalysis.srr_soiling'
    assert 0.97417 == pytest.approx(renorm_factor, abs=1e-4),\
        'Renormalization factor different from expected value in RdAnalysis.srr_soiling'


def test_plot_degradation(sensor_analysis):
    assert_isinstance(sensor_analysis.plot_degradation_summary('sensor'), plt.Figure)
    assert_isinstance(sensor_analysis.plot_pv_vs_irradiance('sensor'), plt.Figure)


def test_plot_cs(clearsky_analysis):
    assert_isinstance(clearsky_analysis.plot_degradation_summary('clearsky'), plt.Figure)
    assert_isinstance(clearsky_analysis.plot_pv_vs_irradiance('clearsky'), plt.Figure)


def test_plot_soiling(soiling_analysis_sensor):
    assert_isinstance(soiling_analysis_sensor.plot_soiling_monte_carlo('sensor'), plt.Figure)
    assert_isinstance(soiling_analysis_sensor.plot_soiling_interval('sensor'), plt.Figure)    
    assert_isinstance(soiling_analysis_sensor.plot_soiling_rate_histogram('sensor'), plt.Figure) 


def test_plot_soiling_cs(soiling_analysis_clearsky):
    assert_isinstance(soiling_analysis_clearsky.plot_soiling_monte_carlo('clearsky'), plt.Figure)
    assert_isinstance(soiling_analysis_clearsky.plot_soiling_interval('clearsky'), plt.Figure)    
    assert_isinstance(soiling_analysis_clearsky.plot_soiling_rate_histogram('clearsky'), plt.Figure) 


def test_errors(sensor_parameters, clearsky_analysis):

    rdtemp = analysis.RdAnalysis(sensor_parameters['pv'])
    with pytest.raises(ValueError, match='poa must be available'):
        rdtemp.sensor_preprocess()

    # no temperature
    rdtemp = analysis.RdAnalysis(sensor_parameters['pv'],
                                 poa=sensor_parameters['poa'])
    with pytest.raises(ValueError, match='either cell or ambient temperature'):
        rdtemp.sensor_preprocess()

    # clearsky analysis with no pvlib.loc
    clearsky_analysis.pvlib_location = None
    clearsky_analysis.clearsky_poa = None
    with pytest.raises(ValueError, match='pvlib location must be provided'):
        clearsky_analysis.clearsky_preprocess()
