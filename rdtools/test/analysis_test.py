from rdtools import analysis
#from soiling_test import times,normalized_daily, insolation
import pytest
import pvlib
import numpy as np
import pandas as pd

meta = {"latitude": -23.762028,
    "longitude": 133.874886,
    "timezone": 'Australia/North',
    "tempco": -0.005,
    "azimuth": 0,
    "tilt": 20}

## From soiling_test
#@pytest.fixture()
def times():
    tz = 'Etc/GMT+7'
    times = pd.date_range('2019/01/01', '2019/03/16', freq='D', tz=tz)
    return times


#@pytest.fixture()
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

@pytest.fixture()
def times_fixture():
    tz = 'Etc/GMT+7'
    times = pd.date_range('2019/01/01', '2019/03/16', freq='D', tz=tz)
    return times

@pytest.fixture()
def normalized_daily_fixture(times_fixture):
    interval_1 = 1 - 0.005 * np.arange(0, 25, 1)
    interval_2 = 1 - 0.002 * np.arange(0, 25, 1)
    interval_3 = 1 - 0.001 * np.arange(0, 25, 1)
    profile = np.concatenate((interval_1, interval_2, interval_3))
    np.random.seed(1977)
    noise = 0.01 * np.random.rand(75)
    normalized_daily = pd.Series(data=profile, index=times)
    normalized_daily = normalized_daily + noise

    return normalized_daily


#@pytest.fixture()
def get_energy(rd = -0.05):
    from degradation_test import DegradationTestCase  #re-use degradation test fixture
    input_freq = 'H'
    corr_energy = DegradationTestCase.get_corr_energy(rd, input_freq) # degradation_test.py
    return corr_energy



def test_sensor_analysis():

    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    power = get_energy().tz_localize(meta['timezone'])
    
    rd = analysis.RdAnalysis(power, power*1000, ambient_temperature = power*0+25, temperature_coefficient=meta['tempco'],
                pvlib_location=loc, pv_tilt=meta['tilt'], pv_azimuth=meta['azimuth'], 
                interp_freq='D', temperature_model = {'a': -3.47, 'b': -0.0594, 'deltaT': 3}) # temperature_model = "open_rack_glass_glass"

    rd.sensor_analysis(analyses=['yoy_degradation'])

    yoy_results = rd.results['sensor']['yoy_degradation']

    assert -1 == pytest.approx(yoy_results['p50_rd'], abs=1e-2)
    assert [-1, -1] == pytest.approx(yoy_results['rd_confidence_interval'], abs=1e-2)


def test_clearsky_analysis():

    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    power = get_energy().tz_localize(meta['timezone'])
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


def test_srr_soiling():  
    reps = 10
    np.random.seed(1977)
    
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    power = normalized_daily(times()).resample('1h').interpolate()
    poa = power*0+1000

    rd = analysis.RdAnalysis(power,poa,ambient_temperature = power*0+25, pvlib_location=loc,
                        temperature_coefficient=0, interp_freq='D' ) 

    rd.sensor_analysis(analyses=['srr_soiling'], srr_kwargs={'reps':reps})
    
    srr_results = rd.results['sensor']['srr_soiling']
   
    assert 0.9583 == pytest.approx(srr_results['p50_sratio'], abs=1e-4),\
        'Soiling ratio different from expected value in RdAnalysis.srr_soiling'  
    assert [0.9552, 0.9607] == pytest.approx(srr_results['sratio_confidence_interval'], abs=1e-4),\
        'Soiling confidence interval different from expected value in RdAnalysis.srr_soiling' 
    assert 0.97417 == pytest.approx(srr_results['calc_info']['renormalizing_factor'], abs=1e-4),\
        'Renormalization factor different from expected value in RdAnalysis.srr_soiling' 


def test_srr_soiling_fixture(normalized_daily_fixture):  
    reps = 10
    np.random.seed(1977)
    
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    power = normalized_daily_fixture.resample('1h').interpolate()
    poa = power*0+1000

    rd = analysis.RdAnalysis(power,poa,ambient_temperature = power*0+25, pvlib_location=loc,
                        temperature_coefficient=0, interp_freq='D' ) 

    rd.sensor_analysis(analyses=['srr_soiling'], srr_kwargs={'reps':reps})
    
    srr_results = rd.results['sensor']['srr_soiling']
   
    assert 0.9583 == pytest.approx(srr_results['p50_sratio'], abs=1e-4),\
        'Soiling ratio different from expected value in RdAnalysis.srr_soiling'  
    assert [0.9552, 0.9607] == pytest.approx(srr_results['sratio_confidence_interval'], abs=1e-4),\
        'Soiling confidence interval different from expected value in RdAnalysis.srr_soiling' 
    assert 0.97417 == pytest.approx(srr_results['calc_info']['renormalizing_factor'], abs=1e-4),\
        'Renormalization factor different from expected value in RdAnalysis.srr_soiling' 

