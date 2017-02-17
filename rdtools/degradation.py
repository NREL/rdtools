''' Degradation Module

This module contains functions to calculate the degradation rate of
photovoltaic systems.
'''

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.api as sm


def degradation_ols(normalized_energy):
    '''
    Description
    -----------
    OLS routine

    Parameters
    ----------
    normalized_energy: Pandas Time Series (numeric)
        Daily or lower frequency time series of normalized system ouput.

    Returns
    -------
    degradation_values: dictionary
        Contains values for annual degradation rate as %/year ('Rd_pct'),
        slope, intercept, root mean square error of regression ('rmse'),
        standard error of the slope ('slope_stderr') and intercept ('intercept_stderr'),
        and least squares RegressionResults object ('ols_results')
    '''

    normalized_energy.name = 'normalized_energy'
    df = normalized_energy.to_frame()
    
    #calculate a years column as x value for regression, ignoreing leap years
    day_diffs = (df.index - df.index[0])
    df['days'] = day_diffs.astype('timedelta64[s]')/(60*60*24)
    df['years'] = df.days/365.0
    
    #add intercept-constant to the exogeneous variable
    df = sm.add_constant(df)
    
    #perform regression
    ols_model = sm.OLS(endog = df.normalized_energy, exog = df.loc[:,['const','years']],
                       hasconst = True, missing = 'drop' )
    
    results = ols_model.fit()
    
    # collect intercept and slope
    b, m = results.params
    
    # rate of degradation in terms of percent/year
    Rd_pct = 100.0 * m / b
    
    #Calculate RMSE
    rmse = np.sqrt(results.mse_resid)
    
    #Collect standrd errors
    stderr_b, stderr_m = results.bse

    degradation = {
        'Rd_pct': Rd_pct,
        'slope': m,
        'intercept': b,
        'rmse': rmse,
        'slope_stderr': stderr_m,
        'intercept_stderr': stderr_b,
        'ols_result': results
    }

    return degradation

      
def degradation_classical_decomposition(normalized_energy):
    '''
    Description
    -----------
    Classical decomposition routine from Dirk Jordan, Chris Deline, and Michael Deceglie

    Parameters
    ----------
    normalized_energy: Pandas Time Series (numeric)
        Daily or lower frequency time series of normalized system ouput.
        Must be regular time series.

    Returns
    -------
    degradation_values: dictionary
        Contains values for annual degradation rate as %/year ('Rd_pct'),
        slope, intercept, root mean square error of regression ('rmse'),
        standard error of the slope ('slope_stderr') and intercept ('intercept_stderr'),
        least squares RegressionResults object ('ols_results'),
        pandas series for the annual rolling mean ('series'),
        Mann-Kendall test trend ('mk_test_trend')
    '''
    
    normalized_energy.name = 'normalized_energy'
    df = normalized_energy.to_frame()
    
    df_check_freq = df.copy()

    df_check_freq = df_check_freq.dropna()

    if df_check_freq.index.freq is None:
        raise ValueError('Classical decomposition requires a regular time series with'
                         ' defined frequency and no missing data.')

    #calculate a years column as x value for regression, ignoreing leap years
    day_diffs = (df.index - df.index[0])
    df['days'] = day_diffs.astype('timedelta64[s]')/(60*60*24)
    df['years'] = df.days/365.0
    
    #Compute yearly rolling mean to isolate trend component using moving average
    it = df.iterrows()
    energy_ma = []
    for i, row in it:
        if row.years-0.5 >= min(df.years) and row.years+0.5 <= max(df.years):
            roll = df[(df.years <= row.years+0.5) & (df.years >= row.years-0.5)]
            energy_ma.append(roll.normalized_energy.mean())
        else:
            energy_ma.append(np.nan)
    
    df['energy_ma'] = energy_ma
    
    #add intercept-constant to the exogeneous variable
    df = sm.add_constant(df)
    
    #perform regression
    ols_model = sm.OLS(endog = df.energy_ma, exog = df.loc[:,['const','years']],
                       hasconst = True, missing = 'drop' )
    
    results = ols_model.fit()
    
    # collect intercept and slope
    b, m = results.params
    
    # rate of degradation in terms of percent/year
    Rd_pct = 100.0 * m / b
    
    #Calculate RMSE
    rmse = np.sqrt(results.mse_resid)
    
    #Collect standrd errors
    stderr_b, stderr_m = results.bse

    #Perform Mann-Kendall 
    test_trend, h, p, z = _mk_test(df.energy_ma.dropna(), alpha=0.05)

    degradation = {
        'Rd_pct': Rd_pct,
        'slope': m,
        'intercept': b,
        'rmse': rmse,
        'slope_stderr': stderr_m,
        'intercept_stderr': stderr_b,
        'ols_result': results,
        'series': df.energy_ma,
        'mk_test_trend': test_trend
    }

    return degradation
     


def degradation_year_on_year(normalized_energy, freq = 'D'):
    '''
    Description
    -----------
    Year-on-year decomposition method

    Parameters
    ----------
    normalized_energy: Pandas data series (numeric) containing corrected performance ratio
        timeseries index in monthly format              
    freq: string to specify aggregation frequency, default value 'D' (daily)
    
    Returns
    -------
    degradation_values: dictionary
        Contains values for median degradation rate and standard error
        'Rd_med', 'Rd_stderr_pct', 'YoY_filtered'
        where YoY_filtered is list containing year on year data for
	specified frequency
    '''
    
    if freq == 'MS':
        # monthly (month start)
        normalized_energy = normalized_energy.resample('MS').mean()
        YearSampleSize = 12
    elif freq == 'M':
        # monthly (month end)
        normalized_energy = normalized_energy.resample('M').mean()
        YearSampleSize = 12
    elif freq == 'W':
        # weekly
        normalized_energy = normalized_energy.resample('W').mean()
        YearSampleSize = 52
    elif freq == 'H':
        # hourly
        normalized_energy = normalized_energy.resample('H').mean()
        YearSampleSize = 8760
    elif freq in ['D', '30T', '15T', 'T']:
        # sample to daily by default
        normalized_energy = normalized_energy.resample('D').mean()
        YearSampleSize = 365
    else:
        raise Exception('Frequency {} not supported'.format(freq))

    # year-on-year approach
    YoYresult = normalized_energy.diff(YearSampleSize) / normalized_energy * 100
    
    def remove_outliers(x): 
        '''
        Description
        -----------
        Remove data points greater or smaller than 100: the system can only lose 100%/year,
        however arbitrary large number can originate by division close to zero!
   
        Parameters
        ----------
        x: float, element of list

        Returns
        -------
        x: float x if absolute value of x is < 100
        '''
        if x < 100 and x > -100:
            return x    
        
    YoY_filtered1 = filter(remove_outliers, YoYresult)
  
    med1 = np.median(YoY_filtered1)                       
    
    # bootstrap to determine 95% CI for the 2 different outlier removal methods
    n1 = len(YoY_filtered1)
    reps = 1000
    xb1 = np.random.choice(YoY_filtered1, (n1, reps), replace=True)
    mb1 = np.median(xb1, axis=0)
    mb1.sort()
    lpc1 = np.percentile(mb1, 16)
    upc1 = np.percentile(mb1, 84)
    unc1 = np.round(upc1 - lpc1,2)         
      
    print '\nDegradation and 68% confidence interval YOY approach:'
    print 'YOY1: Rd = {:.2f} +/- {:.2f}'.format(med1, unc1)
    
    degradation_values = {
        'Rd_median': med1,
        'Rd_stderr_pct': unc1,
        'YoY_filtered':YoY_filtered1
        }
    return degradation_values
      
    
def _mk_test(x, alpha = 0.05):  
    '''
    Description
    -----------
    Mann-Kendall test of significance for trend (used in classical decomposition function)

    Parameters
    ----------
    x: a vector of data type float
    alpha: float, significance level (0.05 default)

    Returns
    -------
    trend: string, tells the trend (increasing, decreasing or no trend)
    h: boolean, True (if trend is present) or False (if trend is absence)
    p: float, p value of the significance test
    z: float, normalized test statistics 
    '''

    from scipy.stats import norm
    
    n = len(x)

    # calculate S
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:
        # there is no tie
        var_s = (n * (n - 1) * (2 * n + 5))/18
    else:
        # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n * (n - 1) * (2 * n + 5) + \
                np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
            z = 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)

    # calculate the p_value for two tail test
    p = 2 * (1 - norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1 - alpha / 2) 

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, z


