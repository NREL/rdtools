''' Degradation Module

This module contains functions to calculate the degradation rate of
photovoltaic systems.
'''

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.api as sm


def degradation_with_ols(normalized_energy):
    '''
    Description
    -----------
    OLS routine

    Parameters
    ----------
    normalized_energy: Pandas Series (numeric)
        Monthly time series of normalized system ouput.

    Returns
    -------
    degradation: dictionary
        Contains degradation rate and standard errors of regression
    '''

    y = normalized_energy

    # remove NaN values
    y = y.dropna()

    # number of examples
    N = len(y)

    # integer-months, the exogeneous variable
    months = np.arange(0, len(y))

    # add intercept-constant to the exogeneous variable
    X = sm.add_constant(months)
    columns = ['constant', 'months']
    exog = pd.DataFrame(X, index = y.index, columns = columns)

    # fit linear model
    ols_model = sm.OLS(endog = y, exog = exog, hasconst = True)
    results = ols_model.fit()

    # collect intercept and slope
    b, m = results.params

    # rate of degradation in terms of percent/year
    Rd_pct = 100 * (m * 12) / b

    # root mean square error
    rmse = np.sqrt(np.power(y - results.predict(exog = exog), 2).sum() / (N - 2))

    # total sum of squares of the time variable
    tss_months = np.power(months - months.mean(), 2).sum()

    # standard error of the slope and intercept
    stderr_b = rmse * np.sqrt((1 / (N - 1)) + months.mean()**2 / tss_months)
    stderr_m = rmse * np.sqrt(1 / tss_months)

    # standard error of the regression
    stderr_Rd = np.sqrt((stderr_m * 12 / b)**2 + ((-12 * m / b**2) * stderr_b)**2)
    stderr_Rd_pct = 100 * stderr_Rd

    degradation = {
        'Rd_pct': Rd_pct,
        'slope': m,
        'intercept': b,
        'rmse': rmse,
        'slope_stderr': stderr_m,
        'intercept_stderr': stderr_b,
        'Rd_stderr_pct': stderr_Rd_pct,
    }

    return degradation

      
def degradation_classical_decomposition(normalized_energy, interpolate_flag = False):
    '''
    Description
    -----------
    Classical decomposition routine from Dirk Jordan and Chris Deline

    Parameters
    ----------
    normalized_energy: Pandas Series (numeric) containing corrected performance ratio, 
        with index being monthly frequency in timestamp format
    interpolate_flag: boolean flag to either interpolate missing data or fill 
        with the median value (default)

    Returns
    -------
    degradation_values: dictionary
        Contains values for annual degradation rate and standard error
        'Rd_pct', 'slope', 'intercept', 'Rd_stderr_pct', 'dataframe'
        where dataframe contains annual rolling mean values
    '''
    
    normalized_energy.name = 'normalized_energy'
    dataframe = normalized_energy.to_frame()

    # try to extract month information from the dataframe index. Fill missing data with median value
    energy_median = normalized_energy.median()

    # check for DatetimeIndex
    if isinstance(dataframe.index, pd.DatetimeIndex):
        dataframe = dataframe.resample('MS').mean()
        
        if interpolate_flag:
	    dataframe = dataframe.interpolate()
        else:
            # append the median value to missing months
            dataframe = dataframe.fillna(value = energy_median)

    # assume an array of months
    dataframe['Month'] = np.arange(0, len(dataframe))

    if pd.infer_freq(dataframe.index) == 'MS' and len(normalized_energy) > 12:
        dataframe = dataframe.rolling(window = 12, center = True).mean()
        
    dataframe = dataframe.dropna()
    y2 = dataframe['normalized_energy']
    x2 = dataframe['Month']
    


    if len(dataframe) <= 2:
        print '\nNot enough data for seasonal decomposition:'
        return {}
        
    # OLS regression with constant
    x2 = sm.add_constant(x2)
    model2 = sm.OLS(y2,x2).fit()
    b_cd, m_cd = model2.params
    Rd_cd, SE_Rd_cd = ols_rd_uncertainty(model2)    

    print '\nDegradation and Standard Error of Classical decomposition:'
    print 'Rd = {:.2f} +/- {:.2f}'.format(Rd_cd, SE_Rd_cd)  

    degradation_values = {
    'Rd_pct': Rd_cd,
    'slope': m_cd,
    'intercept': b_cd,
    'Rd_stderr_pct': SE_Rd_cd,
    'Dataframe':dataframe
    }
    return degradation_values
     


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
      
def degradation_ARIMA(normalized_energy):
    '''
    Description
    -----------
    ARIMA approach - Seasonal decompostion is a special type of ARIMA model

    Parameters
    ----------
    normalized_energy: Pandas data series (numeric) containing corrected performance ratio

    Returns
    -------
    degradation_values: dictionary
        Contains values for annual degradation rate and standard error
        'Rd_pct', 'test_trend', 'p', 'Rd_stderr_pct', 'dataframe'
    '''
    res = sm.tsa.seasonal_decompose(normalized_energy, freq=12)
    y_decomp = res.trend.dropna()        
    df4 = pd.DataFrame({'MS':y_decomp.index,'normalized_energy':y_decomp.values})
    df4['Month'] = range(0, len(df4))
    y3=df4['normalized_energy']
    x3=df4['Month']           
    
    
    
    if len(df4) <=2:
        print '\nNot enough data for ARIMA decomposition:'
        return {}
       
    # OLS regression with constant
    x3 = sm.add_constant(x3)
    model3 = sm.OLS(y3,x3).fit()
    Rd_decomp, SE_Rd_decomp = ols_rd_uncertainty(model3)    

    test_trend,h,p,z = _mk_test(y_decomp, alpha = 0.05)  
        
    degradation_values = {
    'Rd_pct': Rd_decomp,
    'test_trend': test_trend,
    'p': p,
    'Rd_stderr_pct': SE_Rd_decomp,
    'Dataframe':df4,
    }
    return degradation_values
    
    
def _mk_test(x, alpha = 0.05):  
    '''
    Description
    -----------
    Mann-Kendall test of significance for trend (used in ARIMA function)

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

def ols_rd_uncertainty(ols_model):
    '''
    Description
    -----------
    OLS uncertainty calculation

    Parameters
    ----------
    a simple ordinary least squares model created using
    statsmodel.api.OLS 

    Returns
    -------
    Rd_decomp: float, annual degradation rate (in percentage)
    SE_Rd_decomp: float, standard error of annual degradation rate (in percentage)
    '''

    model = ols_model

    b_decomp = model.params['const']
    m_decomp = model.params['Month']
    SE_b_decomp = model.bse['const']
    SE_m_decomp = model.bse['Month']
    
    Rd_decomp = (m_decomp * 12) / b_decomp * 100
    SE_Rd_decomp = np.sqrt(np.power(SE_m_decomp * 12 / b_decomp, 2) + np.power((- 12 * m_decomp / b_decomp**2) * SE_b_decomp, 2))*100

    return Rd_decomp, SE_Rd_decomp
