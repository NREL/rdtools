''' Clearsky Filtering Module

module contains functions for clearsky filtering 

'''

import pandas as pd
import pvlib
import numpy as np

def get_period(times):
    '''
    Determine the period of some time series.
    
    Parameters
    ----------
    times: DatetimeIndex of measurement time series
    '''    
    times_diff = times[1:]-times[0:-1]
    if len(times_diff.unique()) is not 1: # need there to be just one timestep
        raise(ValueError('The timestep is not constant; cannot compute the series period.'))
    period = times_diff[0] / pd.Timedelta(minutes=1)
    return period

def get_clearsky_irrad(system_loc, times, correct_bin_labelling=False):
    '''
    Given some location and times, get the clearsky values

    Parameters
    ----------
    system_loc: pvlib-python LocalizedPVSystem object
        Object contains orientation, geographic coordinates, equipment
        constants.    	    
    times: DatetimeIndex of measurement time series
    correct_bin_labelling: Boolean
        Whether clearsky values should be taken from times offset by
        half a period from the reported index.
    '''
    location = pvlib.location.Location(system_loc.latitude,system_loc.longitude)
    
    if correct_bin_labelling:
        period = get_period(times)
        times_shifted = times + pd.Timedelta(minutes=period/2.0)
        clearsky = location.get_clearsky(times_shifted)
        clearsky.index = times
    
    else:
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
    
    times = clearsky.index.tz_convert('utc')
    
    solar_position = location.get_solarposition(times)

    sim=system_loc.get_irradiance(solar_position['apparent_zenith'],solar_position['azimuth'],clearsky['dni'],clearsky['ghi'],clearsky['dhi'])
    return sim['poa_global']

def detect_clearsky_params(w):
    '''
    Estimate parameters for pvlib detect_clearsky

    Parameters
    ----------
    w : measurement period in minutes
    '''
    dur = max(15,w*3) # moving window period in minutes
    mean_diff = max(75,w/2.0)
    max_diff = max(120,w/1.5)
    lower_line_length = min(-25,-1.0*w/2.0)
    upper_line_length = max(30,w/2.0)

    return dur, mean_diff, max_diff, lower_line_length, upper_line_length
    
def remove_cloudy_times(df,irrad,system_loc,viz=False,correct_bin_labelling=False,return_when_clear=False):
    '''
    Filter based on when clearsky times are detected.
    
    Parameters
    ----------
    df: Pandas DataFrame or Series to be filtered
    irrad: Series
        POA irradiance values with which to filter data
    system_loc: pvlib-python LocalizedPVSystem object
        Object contains orientation, geographic coordinates, equipment
        constants.    	    
    viz: Boolean
        Whether the unfiltered and filtered irradiance should be plotted
    correct_bin_labelling: Boolean
        Whether clearsky values should be taken from times offset by
        half a period from the reported index.
    return_when_clear: Boolean
        Whether the Boolean output of detect_clearsky will be returned, which 
        can be used as an input to remove_cloudy_days.
    '''
    
    # Get the clearsky irradiance
    clearsky = get_clearsky_irrad(system_loc, irrad.index.tz_convert('utc'), correct_bin_labelling=correct_bin_labelling)
    clearsky_poa = get_clearsky_poa(system_loc, clearsky)
    
    # Get the period
    w = get_period(irrad.index)
    
    # Get the parameters for detect_clearsky based on the data period
    dur, mean_diff, max_diff, lower_line_length, upper_line_length = detect_clearsky_params(w)
    
    # Determine which times are clear
    is_clear = pvlib.clearsky.detect_clearsky(irrad.values,clearsky_poa,irrad.index,dur,mean_diff=mean_diff,max_diff=max_diff,lower_line_length=lower_line_length,upper_line_length=upper_line_length,var_diff=5000,slope_dev=9999)

    # Remove rows corresponding to cloudy times
    df_filtered = df.copy()[is_clear==True]
    
    # Plot the unfiltered and filtered data
    if viz:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(111)
        ax.plot(clearsky['dni'],label='Simulated DNI',color='g')
        ax.plot(irrad,lw=1,color='gray',label='Measured irradiance')
        ax.plot(irrad[is_clear==True],'o',color='y',label='Clearsky points')
        ax.legend()
        ax.set_ylabel('Irradiance [W/m^2]')
        plt.show()
        
    if return_when_clear:
        return df_filtered, is_clear
    
    return df_filtered
    
def remove_cloudy_days(df,is_clear,start_time='8:00',end_time='16:00',thresh=0.8):
    '''
    Filter out entire days that were determined to be cloudy.
    
    Parameters
    ----------
    df: Pandas DataFrame or Series to be filtered
    is_clear: Boolean array
        Returned by pvlib's detect_clearksy; can be obtained from
        remove_cloudy_times by setting return_when_clear==True.
    start_time: time
        Start of the daily period in which to consider the instantaneous
        clearness values.
    end_time: time
        End of the daily period in which to consider the instantaneous
        clearness values.
    thresh: float
        Portion of datapoints between start_time and end_time that must be
        clear for a day to be called clear.
    '''
    
    if len(df)!=len(is_clear):
        raise ValueError('Inputs df and is_clear must be the same length.')
    
    # get the unique dates
    unique_dates = np.unique(df.index.date)
    
    # initialize the series
    clear_days = pd.Series(index=unique_dates) 
    is_clear_series = pd.Series(index=df.index,data=is_clear) # just the Boolean "is clear", now with an index
    
    # For each date, look at how many times during the day are clear, and call
    # the whole day clear or not.
    for date in unique_dates:
        inst_for_this_day = is_clear_series[is_clear_series.index.date==date].between_time(start_time,end_time) # instantaneous datapoints during this day
        num_true = len(inst_for_this_day[inst_for_this_day==True]) # how many are clear
        num_false = len(inst_for_this_day[inst_for_this_day==False]) # how many are cloudy
        clear_days[date] = float(num_true)/float(num_false+num_true) >= thresh

    # now up-sample this back to the original index
    clear_days.index = pd.to_datetime(clear_days.index).tz_localize(df.index.tz)
    clear_times = clear_days.reindex(index=df.index,method='ffill')
    df_filtered = df[clear_times==True]
    
    return df_filtered