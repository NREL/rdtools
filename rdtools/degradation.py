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

    if pd.infer_freq(normalized_energy.index) == 'MS' and len(y) > 12:
        # apply 12-month rolling mean
        y = y.rolling(12, center=True).mean()

    # remove NaN values
    y = y.dropna()

    # number of examples
    N = len(y)

    # integer-months, the exogeneous variable
    months = np.arange(0, len(y))

    # add intercept-constant to the exogeneous variable
    X = sm.add_constant(months)
    columns = ['constant', 'months']
    exog = pd.DataFrame(X, index=y.index, columns=columns)

    # fit linear model
    ols_model = sm.OLS(endog=y, exog=exog, hasconst=True)
    results = ols_model.fit()

    # collect intercept and slope
    b, m = results.params

    # rate of degradation in terms of percent/year
    Rd_pct = 100 * (m * 12) / b

    # root mean square error
    rmse = np.sqrt(np.power(y - results.predict(exog=exog), 2).sum() / (N - 2))

    # total sum of squares of the time variable
    tss_months = np.power(months - months.mean(), 2).sum()

    # standard error of the slope and intercept
    stderr_b = rmse * np.sqrt((1/(N-1)) + months.mean()**2 / tss_months)
    stderr_m = rmse * np.sqrt(1 / tss_months)

    # standard error of the regression
    stderr_Rd = np.sqrt((stderr_m * 12/b)**2 + ((-12*m / b**2) * stderr_b)**2)
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

      
###  NEW classical decomposition routine from Chris Deline
def degradation_classical_decomposition(normalized_energy, interpolate_flag = False):
    # (degradation) = degradation_classical_decomposition(normalized_energy, interpolate_flag = False)
    #
    # classical decomponsition method using statsmodel.api
    # from Dirk Jordan (NREL) 12/10/16
    # input parameters:  
    #       dataframe['normalized_energy'] : corrected performance ratio.
    #       dataframe.index : timestamp format, monthly frequency
    #       interpolate_flag:  boolean flag to either interpolate missing data or fill with the median value (default)
    #output parameters:
    #   degradation : tuple of annual degradation rate and standard error
    #              'Rd_pct', 'slope', 'intercept', 'Rd_stderr_pct', 'dataframe'  
    #   dataframe:  dataframe with annual rolling mean values
    
    # TODO:  check timebase for other input time options (day, week)    

    
    ################################################################################################################################
    # Classical decomposition approach 

    #need to choose fill methods: interpolate, or replace NaN's with the median value.
    #interpolate_flag = True
    
    
    dataframe = pd.DataFrame()
    dataframe['normalized_energy'] = normalized_energy
    #try to extract month information from the dataframe index. Fill missing data with median value
    try:
        energy_median = normalized_energy.median()
        dataframe = dataframe.resample('MS').mean()
        
        if interpolate_flag:
            dataframe = dataframe.interpolate()    
        else:  #append the median value to missing months
            dataframe = dataframe.fillna(value=energy_median)
        
        dataframe['Month'] = np.arange(0, len(dataframe))

    except:
        # if you don't pass a datetime index, it assumes that there is data for each month
        dataframe['Month'] = np.arange(0, len(dataframe))
        
    if pd.infer_freq(dataframe.index) == 'MS' and len(normalized_energy) > 12:
        dataframe = dataframe.rolling(window = 12, center = True).mean()
        
    dataframe = dataframe.dropna()
    y2=dataframe['normalized_energy']
    x2 = dataframe['Month']
    


    if len(dataframe) ==0:
        degradation_values = {}

        print '\nNot enough data for seasonal decomposition:'
        
        
    else:        
        # OLS regression with constant
        x2=sm.add_constant(x2)
        model2 = sm.OLS(y2,x2).fit()
    
        b_cd = model2.params['const']
        m_cd = model2.params['Month']
        SE_b_cd = model2.bse['const']
        SE_m_cd = model2.bse['Month']
    
        Rd_cd = (m_cd * 12) / b_cd * 100
        SE_Rd_cd = np.sqrt(np.power(SE_m_cd * 12/b_cd, 2) + np.power((-12*m_cd/b_cd**2) * SE_b_cd, 2))*100

        print '\nDegradation and Standard Error of Classical decomposition:'
        print 'Rd = {:.2f} +/- {:.2f}'.format(Rd_cd, SE_Rd_cd)  

        degradation_values = {
        'Rd_pct': Rd_cd,
        'slope': m_cd,
        'intercept': b_cd,
        #'rmse': rmse,
        #'slope_stderr': stderr_m,
        #'intercept_stderr': stderr_b,
        'Rd_stderr_pct': SE_Rd_cd,
        'Dataframe':dataframe
        }
    return(degradation_values)
     


def degradation_year_on_year(normalized_energy):
    # (YOY_median,YOY_unc,YoY_filtered1) = degradation_year_on_year(normalized_energy)
    #
    # year-on-year  decomponsition method 
    # from Dirk Jordan (NREL) 12/10/16
    # input parameters:  
    #       series['normalized_energy'] : corrected performance ratio series
    #       index:  timeseries index in monthly format
    #       
    #output parameters:
    #   (YOY_median,YOY_unc) : tuple of median degradation rate and standard error
    # TODO:  check timebase for other input time options (day, week)    

    ################################################################################################################################
    # YOY approach
    
    # ensure we have monthly data
    normalized_energy = normalized_energy.resample('MS').mean()
    

    #  year-on-year approach.
    YoYresult = []
    for index in range(normalized_energy.size-12):
        YoYresult.append( (normalized_energy[index+12]-normalized_energy[index])/+normalized_energy[index]*100 )
    
    #print YoYresult    
    #YoY = pd.Series(YoYresult)
       
    #Remove data points greater or smaller than 100: the system can only lose 100%/year,
    #however arbitray large number can originate by division close to zero!

    def remove_outliers(x): 
        if x < 100 and x > -100:
            return x    
        
    YoY_filtered1 = filter(remove_outliers, YoYresult)
    #print YoY_filtered1    
  
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
        #'slope': m_cd,
        #'intercept': b_cd,
        #'rmse': rmse,
        #'slope_stderr': stderr_m,
        #'intercept_stderr': stderr_b,
        'Rd_stderr_pct': unc1,
        'YoY_filtered':YoY_filtered1
        }
    return(degradation_values)
      
    # End YOY approach
    ################################################################################################################################

      
def degradation_ARIMA(normalized_energy):
    # (Rd_decomp,SE_Rd_decomp,test_trend,p,df4) = classical_decomposition(normalized_energy,Month)
    #
    # ARIMA seasonal decomponsition method using statsmodel.api
    # from Dirk Jordan (NREL) 12/10/16
    # input parameters:  
    #       dataframe['normalized_energy'] : corrected performance ratio.
    #output parameters:
    #   (Rd_decomp,SE_Rd_decomp,test_trend,p) : tuple of annual degradation rate and standard error
    # TODO:  check timebase for other input time options (day, week)    

    ################################################################################################################################
    # ARIMA approach - Seasonal decompostion is a special type of ARIMA model   

    # deal with missing values
    #y = y.interpolate(inplace=True)
    
    res = sm.tsa.seasonal_decompose(normalized_energy, freq=12)
    #resplot = res.plot()         
    y_decomp = res.trend.dropna()        
    df4 = pd.DataFrame({'MS':y_decomp.index,'normalized_energy':y_decomp.values})
    df4['Month'] = range(0, len(df4))
    y3=df4['normalized_energy']
    x3=df4['Month']           
    
    
    
    if len(df4) <=2:
        Rd_decomp = 11111
        SE_Rd_decomp = 11111
        test_trend = 11111
        p = 11111
        print '\nNot enough data for ARIMA decomposition:'
       
    else:
 
       # OLS regression with constant
        x3=sm.add_constant(x3)
        model3 = sm.OLS(y3,x3).fit()
    
        b_decomp = model3.params['const']
        m_decomp = model3.params['Month']
        SE_b_decomp = model3.bse['const']
        SE_m_decomp = model3.bse['Month']
    
        Rd_decomp = (m_decomp * 12) / b_decomp * 100
        SE_Rd_decomp = np.sqrt(np.power(SE_m_decomp * 12/b_decomp, 2) + np.power((-12*m_decomp/b_decomp**2) * SE_b_decomp, 2))*100

        #print '\nDegradation and Standard Error of ARIMA decomposition:'
        #print 'Rd = {:.2f} +/- {:.2f}'.format(Rd_decomp, SE_Rd_decomp)              
       
        test_trend,h,p,z = _mk_test(y_decomp,alpha=0.05)  
        #print test_trend, h, p    
    
        
        degradation_values = {
        'Rd_pct': Rd_decomp,
        'test_trend': test_trend,
        'p': p,
        #'rmse': rmse,
        #'slope_stderr': stderr_m,
        #'intercept_stderr': stderr_b,
        'Rd_stderr_pct': SE_Rd_decomp,
        'Dataframe':df4,
        }
    return(degradation_values)


    
    
#Mann-Kendall test of significance for trend (used in ARIMA function)
def _mk_test(x, alpha = 0.05):  

#    Input:
#        x:   a vector of data
#        alpha: significance level (0.05 default)
#    Output:
#        trend: tells the trend (increasing, decreasing or no trend)
#        h: True (if trend is present) or False (if trend is absence)
#        p: p value of the significance test
#        z: normalized test statistics 

    from scipy.stats import norm
    
    n = len(x)

    # calculate S 
    s = 0
    for k in range(n-1):
        for j in range(k+1,n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s>0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
            z = 0
    elif s<0:
        z = (s + 1)/np.sqrt(var_s)

    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z))) # two tail test
    h = abs(z) > norm.ppf(1-alpha/2) 

    if (z<0) and h:
        trend = 'decreasing'
    elif (z>0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, z