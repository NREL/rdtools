''' Clearsky Filtering Module

module contains functions for clearsky filtering 

'''

import pandas as pd
import pvlib
import numpy as np
import datetime
from datetime import datetime
import collections

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
    location = pvlib.location.Location(system_loc.latitude,system_loc.longitude, tz=times.tzinfo)
    
    if correct_bin_labelling:
        period = get_period(times)
        times_shifted = times + pd.Timedelta(minutes=period/2.0)
        clearsky = location.get_clearsky(times_shifted)
    else:
        clearsky = location.get_clearsky(times)
    
    if isinstance(clearsky, collections.OrderedDict):
       clearsky = pd.DataFrame.from_dict(clearsky)
       
    clearsky.index = times

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
    
    if (clearsky.index.tzinfo is None):
       raise ValueError('Time zone information required for clearsky times')
    
    times = clearsky.index

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
    if (irrad.index.tzinfo is None):
       raise ValueError('Time zone information required for times of irradiance measurement')
       
    clearsky = get_clearsky_irrad(system_loc, irrad.index, correct_bin_labelling=correct_bin_labelling)
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
        fig = plt.figure(figsize=(20,12))
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
    
def remove_cloudy_days(df,is_clear,start_time='7:00',end_time='17:00',thresh=0.8,return_clear_days=False):
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
    
    if (df.index.tzinfo is None):
       raise ValueError('Time zone information required for dataframe DatetimeIndex')

    # redefine start_time and end_time for timezone of df.index
    start_time = datetime.strptime(start_time, '%H:%M').time()
    start_time = start_time.replace(tzinfo=df.index.tzinfo)
    end_time = datetime.strptime(end_time, '%H:%M').time()
    end_time = end_time.replace(tzinfo=df.index.tzinfo)

    # get the unique dates
    unique_dates = np.unique(df.index.date)
    
    # initialize the series
    clear_days = pd.Series(index=unique_dates) 
    is_clear_series = pd.Series(index=df.index,data=is_clear) # just the Boolean "is clear", now with an index
    
    filtered1 = is_clear_series.between_time(start_time,end_time)
    
    def enough_points_for_today(s):
        num_true = len(s[s==True])
        num_false = len(s[s==False])
        portion_true = float(num_true)/float(num_false+num_true)        
        return portion_true>thresh
        
    clear_days = filtered1.resample('1D').apply(enough_points_for_today)
    
    '''
    # For each date, look at how many times during the day are clear, and call
    # the whole day clear or not.
    for date in unique_dates:
        inst_for_this_day = is_clear_series[is_clear_series.index.date==date].between_time(start_time,end_time) # instantaneous datapoints during this day
        num_true = len(inst_for_this_day[inst_for_this_day==True]) # how many are clear
        num_false = len(inst_for_this_day[inst_for_this_day==False]) # how many are cloudy
        
        # call a day cloudy if there are no valid points to look at between start_time and end_time
        if (num_false+num_true)==0:
            clear_days[date] = False
        else:
            clear_days[date] = float(num_true)/float(num_false+num_true) >= thresh
    '''

    # now up-sample this back to the original index
    #clear_days.index = pd.to_datetime(clear_days.index).tz_localize(df.index.tz)
    clear_times = clear_days.reindex(index=df.index,method='ffill')
    df_filtered = df[clear_times==True]
    
    if return_clear_days:
        return df_filtered,clear_days
    
    return df_filtered

def remove_cloudy_days_from_curve(df,energy,quant=0.90,viz=False,return_when_clear=False):
    '''
    Filter out cloudy days just based on the shape of the power curve.
    
    This uses the empirical result that all clearsky days fall around the same
    straight line on a plot of daily max energy vs. energy curve arc length.
    
    Parameters
    ----------
    df: Pandas DataFrame or Series to be filtered
    energy: Series
        Energy or power values with which to filter data 	    
    quant: float
        Quantile threshold for splitting clear and cloudy days
    viz: Boolean
        Whether the unfiltered and filtered irradiance should be plotted
    return_when_clear: Boolean
        Whether the Boolean output of when clearsky is detected should be
        returned.    
    '''
    
    def sweep_slopes(daily_metrics):
        slopes = np.linspace(0,1,51)
        ds = slopes[1]-slopes[0]
        ndiff = pd.Series(index=slopes)
        for s in slopes:
            slope_above = s+ds
            slope = s
            slope_below = s-ds
            
            pred_above = daily_metrics['linelength']*slope_above
            pred = daily_metrics['linelength']*slope
            pred_below = daily_metrics['linelength']*slope_below
            
            n_above = len(daily_metrics[(daily_metrics['max']>=pred)&(daily_metrics['max']<=pred_above)])
            n_below = len(daily_metrics[(daily_metrics['max']<=pred)&(daily_metrics['max']>=pred_below)])
            
            ndiff.loc[s] = n_below-n_above
            
        best_slope = ndiff.idxmax()
        
        #clear_days = daily_metrics[(daily_metrics['ratio']>=best_slope-ds)&(daily_metrics['max']<=pred_above)]
            
        return best_slope,ndiff
        
    # df which will store the metrics for each day
    daily_metrics = pd.DataFrame(index = energy.copy().resample('1D').mean().index)
    
    # each daily metric will be some sort of resampling of that day's values
    resampler = energy.copy().resample('1D')

    # function to compute the arc length for a day's energy curve
    def line_length(s):
        x = (((s.diff()**2 + 0))**0.5).sum()
        #x = ((s.diff()**2 + 0).sum())**0.5
        return x    
    
    import matplotlib.pyplot as plt
    # calculate the daily line length, max, and their ratio
    daily_metrics['linelength'] = resampler.apply(line_length)
    daily_metrics['max'] = resampler.max()
    daily_metrics['ratio'] = (daily_metrics['max']/daily_metrics['linelength']).replace([np.inf,-np.inf],np.nan)
    #daily_metrics = daily_metrics[daily_metrics['max']>=daily_metrics['max'].quantile(0.95)/2]
                                  
    best_slope,ndiff = sweep_slopes(daily_metrics)
    plt.figure()
    ndiff.plot()
    plt.title(str(best_slope))
    best_slope = .5
    
    # select clear days based on the ratio
    #clear_days = daily_metrics['ratio'] > daily_metrics['ratio'].quantile(quant)
    #clear_days = daily_metrics['ratio'] > (daily_metrics['ratio'].min() + quant*(best_slope - daily_metrics['ratio'].min()))
    tol = 0.03
    good_above = (daily_metrics['ratio'] > (best_slope*(1-tol)))
    good_below = (daily_metrics['ratio'] < (best_slope*(1+tol)))
    good_max = daily_metrics['max'] > 0.4*daily_metrics['max'].quantile(0.95)
    
    clear_days = good_above & good_below & good_max
    print(daily_metrics['ratio'])
        
    # go from clear DAYS to clear TIMES
    clear_inst = clear_days.reindex(index=energy.index,method='ffill')
    
    df_filtered = df.copy()
    df_filtered = df_filtered[clear_inst==True]
    
    # plot the daily points and filtered curves if necessary
    if viz:
        
        import matplotlib.pyplot as plt
        
        from sklearn.linear_model import LinearRegression
        print(daily_metrics.loc[clear_days==True,'linelength'].to_frame())
        model = LinearRegression().fit(daily_metrics.loc[clear_days==True,'linelength'].to_frame(),daily_metrics.loc[clear_days==True,'max'].to_frame())
        print(model.coef_)
        print(model.intercept_)
        x_line = [0.,daily_metrics.loc[clear_days==True,'linelength'].max()]
        y_line = model.predict(pd.DataFrame(x_line))        
        
        fig = plt.figure(figsize=(12,9))
        
        ax1 = fig.add_subplot(221)
        topline = [x*best_slope*(1+tol) for x in x_line]
        bottomline = [x*best_slope*(1-tol) for x in x_line]
        ax1.fill_between(x_line,bottomline,topline,facecolor='g',alpha=0.5)
        ax1.scatter(daily_metrics['linelength'],daily_metrics['max'],color='gray',label='cloudy days')
        ax1.scatter(daily_metrics['linelength'][clear_days==True],daily_metrics['max'][clear_days==True],color='g',label='clearsky days')
        ax1.plot(x_line,y_line,color='r')
        ax1.legend()
        ax1.set_xlabel('daily line length')
        ax1.set_ylabel('daily max')
        
        ax2 = fig.add_subplot(222)
        ax2.plot(energy,color='gray')
        ax2.plot(energy[clear_inst==True],'o',color='g')
        
        ax3 = fig.add_subplot(223)
        daily_metrics['ratio'].hist(bins=100,ax=ax3)
        
        ax4 = fig.add_subplot(224)
        daily_metrics['max'].hist(bins=100,ax=ax4)
        
        fig.suptitle('Picked '+str(len(clear_days[clear_days==True]))+' clear days out of '+str(len(clear_days)))        

        plt.show()
        plt.pause(0.1)
        plt.show()
                
    if return_when_clear:
        return df_filtered, clear_inst
        
    else:
        return df_filtered