''' Clearsky Filtering Module

module contains functions for clearsky filtering 

'''

import pandas as pd
import pvlib

def get_clearsky_irrad(system_loc, times):    
    '''
    Given some location and times, get the clearsky values

    Parameters
    ----------
    system_loc: pvlib-python LocalizedPVSystem object
        Object contains orientation, geographic coordinates, equipment
        constants.    	    
    times: DatetimeIndex of measurement time series
    '''
    location = pvlib.location.Location(system_loc.latitude,system_loc.longitude) # timezone here too?
    clearsky = location.get_clearsky(times)
    return clearsky

def get_clearsky_poa(system_loc, clearsky):
    '''
    Use PV-LIB to simulate what a fixed system w/ certain configuration would see.

    Parameters
    ----------
    system_loc: pvlib-python LocalizedPVSystem object
        Object contains orientation, geographic coordinates, equipment
        constants.    	    
    clearsky: Pandas DataFrame (numeric)
         clear sky estimates of GHI, DNI, and/or DHI 
    '''
    location = pvlib.location.Location(system_loc.latitude,system_loc.longitude) # timezone here too?
    
    times = clearsky.index
    
    solar_position = location.get_solarposition(times)
    solar_position = solar_position[solar_position.index.duplicated()==False] # necessary?
    
    sim=system_loc.get_irradiance(solar_position['apparent_zenith'],solar_position['azimuth'],clearsky['dni'],clearsky['ghi'],clearsky['dhi'])
    return sim['poa_global']
