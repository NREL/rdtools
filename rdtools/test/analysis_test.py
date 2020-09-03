from rdtools import analysis
import pytest
import pvlib


#@pytest.fixture()
def get_energy():
    from degradation_test import DegradationTestCase  #re-use degradation test fixture
    rd = -0.05
    input_freq = 'H'
    corr_energy = DegradationTestCase.get_corr_energy(rd, input_freq) # degradation_test.py
    return corr_energy

def test_sensor_analysis():
    meta = {"latitude": -23.762028,
        "longitude": 133.874886,
        "timezone": 'Australia/North',
        "tempco": -0.005,
        "azimuth": 0,
        "tilt": 20}
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    power = get_energy().tz_localize(meta['timezone'])
    
    rd = analysis.RdAnalysis(power, power*1000, ambient_temperature = power*0+25, temperature_coefficient=meta['tempco'],
                pvlib_location=loc, pv_tilt=meta['tilt'], pv_azimuth=meta['azimuth'], 
                interp_freq='D', temperature_model = {'a': -3.47, 'b': -0.0594, 'deltaT': 3}) # temperature_model = "open_rack_glass_glass"
    # need cell or ambient temp for sensor_analysis
    rd.sensor_analysis(analyses=['yoy_degradation']) #, 'srr_soiling'

    yoy_results = rd.results['sensor']['yoy_degradation']

    assert -1 == pytest.approx(yoy_results['p50_rd'], abs=1e-2)
    assert [-1, -1] == pytest.approx(yoy_results['rd_confidence_interval'], abs=1e-2)


def test_clearsky_analysis():
    meta = {"latitude": -23.762028,
        "longitude": 133.874886,
        "timezone": 'Australia/North',
        "tempco": -0.005,
        "azimuth": 0,
        "tilt": 20}
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
    power = get_energy().tz_localize(meta['timezone'])
    
    rd = analysis.RdAnalysis(power, power*1000, temperature_coefficient=meta['tempco'],
                pvlib_location=loc, pv_tilt=meta['tilt'], pv_azimuth=meta['azimuth'], 
                )
    # need cell or ambient temp for sensor_analysis

    rd.clearsky_analysis()
    cs_yoy_results = rd.results['clearsky']['yoy_degradation']

    assert -3.8677 == pytest.approx(cs_yoy_results['p50_rd'], abs=1e-4)
    assert [-4.7, -3.8] == pytest.approx(cs_yoy_results['rd_confidence_interval'], abs=1e-1)


def test_srr_soiling():  
    pass
    


