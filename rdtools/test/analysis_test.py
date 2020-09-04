from rdtools import analysis
from soiling_test import normalized_daily, times
from plotting_test import assert_isinstance
import pytest
import pvlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

meta = {"latitude": -23.762028,
    "longitude": 133.874886,
    "timezone": 'Australia/North',
    "tempco": -0.005,
    "azimuth": 0,
    "tilt": 20}

"""
#########
##  THE FOLLOWING ARE OLD VERSIONS OF TESTS NOT USING FIXTURES. CAN BE REMOVED
#########
## From soiling_test.  Not needed if test_srr_soiling is removed
#@pytest.fixture()
def times2():
    tz = 'Etc/GMT+7'
    return pd.date_range('2019/01/01', '2019/03/16', freq='D', tz=tz)

#@pytest.fixture()
def normalized_daily2(times2):
    interval_1 = 1 - 0.005 * np.arange(0, 25, 1)
    interval_2 = 1 - 0.002 * np.arange(0, 25, 1)
    interval_3 = 1 - 0.001 * np.arange(0, 25, 1)
    profile = np.concatenate((interval_1, interval_2, interval_3))
    np.random.seed(1977)
    noise = 0.01 * np.random.rand(75)
    data = pd.Series(data=profile, index=times2)
    data = data + noise

    return data

def assert_isinstance(obj, klass):
    assert isinstance(obj, klass), f'got {type(obj)}, expected {klass}'

@pytest.fixture()
def times():
    tz = 'Etc/GMT+7'
    times = pd.date_range('2019/01/01', '2019/03/16', freq='D', tz=tz)
    return times


@pytest.fixture()
def normalized_daily(times):
    interval_1 = 1 - 0.005 * np.arange(0, 25, 1)
    interval_2 = 1 - 0.002 * np.arange(0, 25, 1)
    interval_3 = 1 - 0.001 * np.arange(0, 25, 1)
    profile = np.concatenate((interval_1, interval_2, interval_3))
    np.random.seed(1977)
    noise = 0.01 * np.random.rand(75)
    normalized_daily = pd.Series(data=profile, index=times)
    normalized_daily = normalized_daily + noise

    return normalized_daily


def test_srr_soiling():  # duplicate of test_srr_soiling_fixture
    reps = 10
    
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    
    power = normalized_daily2(times2()).resample('1h').interpolate()
    poa = power*0+1000

    rdSoiling = analysis.RdAnalysis(power,poa,ambient_temperature = power*0+25, pvlib_location=loc,
                        temperature_coefficient=0, interp_freq='D' ) 
    
    rdSoiling.sensor_analysis(analyses=['srr_soiling'], srr_kwargs={'reps':reps})
    
    srr_results = rdSoiling.results['sensor']['srr_soiling']
   
    assert 0.9583 == pytest.approx(srr_results['p50_sratio'], abs=1e-4),\
        'Soiling ratio different from expected value in RdAnalysis.srr_soiling'  
    assert [0.9552, 0.9607] == pytest.approx(srr_results['sratio_confidence_interval'], abs=1e-4),\
        'Soiling confidence interval different from expected value in RdAnalysis.srr_soiling' 
    assert 0.97417 == pytest.approx(srr_results['calc_info']['renormalizing_factor'], abs=1e-4),\
        'Renormalization factor different from expected value in RdAnalysis.srr_soiling' 
"""

    
@pytest.fixture()
def get_energy():
    from degradation_test import DegradationTestCase  #re-use degradation test fixture
    rd = -0.05
    input_freq = 'H'
    corr_energy = DegradationTestCase.get_corr_energy(rd, input_freq) # degradation_test.py
    return corr_energy

@pytest.fixture()
def getRd(get_energy):
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    power = get_energy.tz_localize(meta['timezone'])    
    rd = analysis.RdAnalysis(power, power*1000, ambient_temperature = power*0+25, temperature_coefficient=meta['tempco'],
                pvlib_location=loc, pv_tilt=meta['tilt'], pv_azimuth=meta['azimuth'], 
                interp_freq='D', temperature_model = {'a': -3.47, 'b': -0.0594, 'deltaT': 3}) # temperature_model = "open_rack_glass_glass"
    return rd
    
@pytest.fixture()
def getRdCS(get_energy):
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    power = get_energy.tz_localize(meta['timezone'])
    # initialize the RdAnalysis class
    temp = analysis.RdAnalysis(power, power*1000, pvlib_location=loc, 
                        pv_tilt=meta['tilt'], pv_azimuth=meta['azimuth'], 
                        temperature_coefficient=0 )
    # get clearsky expected
    temp.clearsky_preprocess()  # get expected clear-sky values in clearsky_poa.
    cs = temp.clearsky_poa
    
    rdCS = analysis.RdAnalysis(power*cs, cs, temperature_coefficient=meta['tempco'],
            pvlib_location=loc, pv_tilt=meta['tilt'], pv_azimuth=meta['azimuth'], 
            temperature_model = {'a': -3.47, 'b': -0.0594, 'deltaT': 3}
            )
    return rdCS

@pytest.fixture()
def getRdSoiling(normalized_daily):
    
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    power = normalized_daily.resample('1h').interpolate()
    poa = power*0+1000
    rdSoiling = analysis.RdAnalysis(power,poa, cell_temperature = power*0+25, pvlib_location=loc,
                        temperature_coefficient=0, interp_freq='D',
                        pv_tilt=meta['tilt'], pv_azimuth=meta['azimuth'] ) 
    return rdSoiling
    

def test_sensor_analysis_fixture(getRd):

    getRd.sensor_analysis(analyses=['yoy_degradation'])

    yoy_results = getRd.results['sensor']['yoy_degradation']

    assert -1 == pytest.approx(yoy_results['p50_rd'], abs=1e-2)
    assert [-1, -1] == pytest.approx(yoy_results['rd_confidence_interval'], abs=1e-2)


def test_clearsky_analysis_fixture(getRdCS):

    getRdCS.clearsky_analysis()
    cs_yoy_results = getRdCS.results['clearsky']['yoy_degradation']

    assert -4.744 == pytest.approx(cs_yoy_results['p50_rd'], abs=1e-3)
    assert [-4.75, -4.73] == pytest.approx(cs_yoy_results['rd_confidence_interval'], abs=1e-2)
    
    # Re-run while passing some of the clearsky values back into the original instance to improve test coverage
    poa = getRdCS.poa
    cspoa = getRdCS.clearsky_poa
    cscell = getRdCS.clearsky_cell_temperature
    csamb = getRdCS.clearsky_ambient_temperature
    pv = getRdCS.pv_energy
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    
    rdCS2 = analysis.RdAnalysis(pv, poa=poa, clearsky_cell_temperature=cscell,
            clearsky_poa=cspoa, clearsky_ambient_temperature=csamb, pv_input='energy', 
            pvlib_location=loc, pv_tilt=pd.Series(data=meta['tilt'], index=pv.index), 
            pv_azimuth=pd.Series(data=meta['azimuth'], index=pv.index),
            temperature_model = {'a': -3.47, 'b': -0.0594, 'deltaT': 3}, interp_freq='1H'
            )
    rdCS2.pv_power = getRdCS.pv_power
    rdCS2.clearsky_analysis()
    cs_yoy_results2 = rdCS2.results['clearsky']['yoy_degradation']
    assert -5.1278 == pytest.approx(cs_yoy_results2['p50_rd'], abs=1e-3)
    assert [-5.1285, -5.1269] == pytest.approx(cs_yoy_results2['rd_confidence_interval'], abs=1e-3)    
    
"""
def test_clearsky_analysis(get_energy): #duplicate of above

    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    power = get_energy.tz_localize(meta['timezone'])
    # initialize the RdAnalysis class
    temp = analysis.RdAnalysis(power, power*1000, pvlib_location=loc, 
                        pv_tilt=meta['tilt'], pv_azimuth=meta['azimuth'], 
                        temperature_coefficient=0 )
    # get clearsky expected
    temp.clearsky_preprocess()  # get expected clear-sky values in clearsky_poa.
    cs = temp.clearsky_poa
    
    rd = analysis.RdAnalysis(power*cs, cs, temperature_coefficient=meta['tempco'],
            pvlib_location=loc, pv_tilt=meta['tilt'], pv_azimuth=meta['azimuth'], 
            temperature_model = {'a': -3.47, 'b': -0.0594, 'deltaT': 3}
            )
    
    rd.clearsky_analysis()
    cs_yoy_results = rd.results['clearsky']['yoy_degradation']

    assert -4.744 == pytest.approx(cs_yoy_results['p50_rd'], abs=1e-3)
    assert [-4.75, -4.73] == pytest.approx(cs_yoy_results['rd_confidence_interval'], abs=1e-2)
"""


def test_srr_soiling_fixture(getRdSoiling):  # same as test_srr_soiling just using test fixtures.
    
    getRdSoiling.sensor_analysis(analyses=['srr_soiling'], srr_kwargs={'reps':10})
    
    srr_results = getRdSoiling.results['sensor']['srr_soiling']
   
    assert 0.9583 == pytest.approx(srr_results['p50_sratio'], abs=1e-4),\
        'Soiling ratio different from expected value in RdAnalysis.srr_soiling'  
    assert [0.9552, 0.9607] == pytest.approx(srr_results['sratio_confidence_interval'], abs=1e-4),\
        'Soiling confidence interval different from expected value in RdAnalysis.srr_soiling' 
    assert 0.97417 == pytest.approx(srr_results['calc_info']['renormalizing_factor'], abs=1e-4),\
        'Renormalization factor different from expected value in RdAnalysis.srr_soiling' 

def test_plot_degradation(getRd):
    getRd.sensor_analysis(analyses=['yoy_degradation'])
    assert_isinstance(getRd.plot_degradation_summary('sensor'), plt.Figure)
    assert_isinstance(getRd.plot_pv_vs_irradiance('sensor'), plt.Figure)

def test_plot_cs(getRdCS):
    getRdCS.clearsky_analysis(analyses=['yoy_degradation'])
    assert_isinstance(getRdCS.plot_degradation_summary('clearsky'), plt.Figure)
    assert_isinstance(getRdCS.plot_pv_vs_irradiance('clearsky'), plt.Figure)   
    
def test_plot_soiling(getRdSoiling):
    getRdSoiling.sensor_analysis(analyses=['srr_soiling'], srr_kwargs={'reps':10})
    assert_isinstance(getRdSoiling.plot_soiling_monte_carlo('sensor'), plt.Figure)
    assert_isinstance(getRdSoiling.plot_soiling_interval('sensor'), plt.Figure)    
    assert_isinstance(getRdSoiling.plot_soiling_rate_histogram('sensor'), plt.Figure) 
    
def test_plot_soiling_cs(getRdSoiling):
    getRdSoiling.clearsky_analysis(analyses=['srr_soiling'], srr_kwargs={'reps':10})
    assert_isinstance(getRdSoiling.plot_soiling_monte_carlo('clearsky'), plt.Figure)
    assert_isinstance(getRdSoiling.plot_soiling_interval('clearsky'), plt.Figure)    
    assert_isinstance(getRdSoiling.plot_soiling_rate_histogram('clearsky'), plt.Figure) 
    
def test_errors(get_energy, getRdCS):
    # clearsky analysis with no pvlib.loc or tilt or azimuth
    power = get_energy.tz_localize(meta['timezone'])
    rdtemp = analysis.RdAnalysis(power)
    with pytest.raises(ValueError):
        rdtemp.sensor_preprocess() # no POA error
    rdtemp = analysis.RdAnalysis(power,power*1000)
    with pytest.raises(ValueError):
        rdtemp.sensor_preprocess() # no temperature error
    getRdCS.pvlib_location=None
    with pytest.raises(ValueError):
        getRdCS.clearsky_preprocess() #pvlib location must be provided
        