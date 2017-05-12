""" Degradation Module Tests. """

import unittest, sys

import pandas as pd
import numpy as np

from rdtools import degradation_ols, degradation_classical_decomposition, degradation_year_on_year

class DegradationTestCase(unittest.TestCase):
    ''' Unit tests for degradation module. 
        This code works with python but not with nosetests
    '''

    def __init__(self, testname, freq, list_CD_YOY_freq):
        super(DegradationTestCase, self).__init__(testname)
        self.input_freq = freq
        self.list_CD_YOY_input_freq = list_CD_YOY_freq
    
    def get_ols_corr_energy(self, rd, input_freq):
        ''' Create input for degradation_ols function, depending on frequency. 
            Allowed frequencies for this degradation function are 'MS', 'M', 'W', 'D'
            'H', 'T' and 'S'.
        '''
        if (input_freq == 'MS'):
            x = pd.date_range(start='2012-01-01', end='2012-03-01', freq=input_freq)
            N = len(x)
            months = np.arange(N)
            y = np.ones(N) * np.power(1 + rd/12, months)
        elif (input_freq == 'M'):
            x = pd.date_range(start='2012-01-31', end='2012-03-31', freq=input_freq)
            N = len(x)
            months = np.arange(N)
            y = np.ones(N) * np.power(1 + rd/12, months)
        elif (input_freq == 'W'):
            x = pd.date_range(start='2012-01-01', end='2012-01-21', freq=input_freq)
            N = len(x)
            weeks = np.arange(N)
            y = np.ones(N) * np.power(1 + rd/52, weeks)
        elif (input_freq == 'D'):
            x = pd.date_range(start='2012-01-01', end='2012-01-03', freq=input_freq)
            N = len(x)
            days = np.arange(N)
            y = np.ones(N) * np.power(1 + rd/365, days)
        elif (input_freq == 'H'):
            x = pd.date_range(start='2012-01-01 00:00:00', end='2012-01-02 00:00:00', freq=input_freq)
            N = len(x)
            hours = np.arange(N)
            y = np.ones(N) * np.power(1 + rd/(365 * 24), hours)
        elif (input_freq == 'T'):
            x = pd.date_range(start='2012-01-01 00:00:00', end='2012-01-01 00:50:00', freq=input_freq)
            N = len(x)
            minutes = np.arange(N)
            y = np.ones(N) * np.power(1 + rd/(365 * 24 * 60), minutes)
        elif (input_freq == 'S'):
            x = pd.date_range(start='2012-01-01 00:00:00', end='2012-01-01 00:10:00', freq=input_freq)
            N = len(x)
            seconds = np.arange(N)
            y = np.ones(N) * np.power(1 + rd/(365 * 24 * 60 * 60), seconds)
        else :
            sys.exit("Unknown frequency type")
        
        corr_energy = pd.Series(data=y, index=x)
        return corr_energy

    def get_CD_YOY_corr_energy(self, rd, input_freq):
        ''' Create input for degradation_classical_decomposition and degradation_year_on_year
            functions, depending on frequency. Allowed frequencies for both of these degradation 
            functions  are 'MS', 'M', 'W' and 'D'.
        '''
        if (input_freq == 'MS'):
            x = pd.date_range(start='2012-01-01', end='2015-01-01', freq=input_freq)
            N = len(x)
            months = np.arange(N)
            y = np.ones(N) * np.power(1 + rd/12, months)
        elif (input_freq == 'M'):
            x = pd.date_range(start='2012-01-31', end='2015-01-31', freq=input_freq)
            N = len(x)
            months = np.arange(N)
            y = np.ones(N) * np.power(1 + rd/12, months)
        elif (input_freq == 'W'):
            x = pd.date_range(start='2012-01-01', end='2015-01-01', freq=input_freq)
            N = len(x)
            weeks = np.arange(N)
            y = np.ones(N) * np.power(1 + rd/52, weeks)
        elif (input_freq == 'D'):
            x = pd.date_range(start='2012-01-01', end='2015-01-01', freq=input_freq)
            N = len(x)
            days = np.arange(N)
            y = np.ones(N) * np.power(1 + rd/365, days)
        else :
            sys.exit("Unknown frequency type")

        corr_energy = pd.Series(data=y, index=x)
        return corr_energy

    def setUp(self):
        # define module constants and parameters

        self.rd = -0.005

        if self.input_freq in self.list_CD_YOY_input_freq:
            self.test_CD_YOY_corr_energy = self.get_CD_YOY_corr_energy(self.rd, self.input_freq)
        
        self.test_ols_corr_energy = self.get_ols_corr_energy(self.rd, self.input_freq)
             

    def tearDown(self):
        pass
    
    def test_degradation_with_ols(self):
        ''' Test degradation with ols. '''

        funcName = sys._getframe().f_code.co_name
        print '\r', 'Running ', funcName

        # test ols degradation calc
        print 'Frequency: ', self.input_freq
        rd_result = degradation_ols(self.test_ols_corr_energy)
        self.assertAlmostEqual(rd_result[0], 100*self.rd, places=1)
        print 'Actual: ', 100*self.rd
        print 'Estimated: ', rd_result[0]

        # TODO
        # - support for different time series frequencies
        # - inputs

    def test_degradation_classical_decomposition(self):
        ''' Test degradation with classical decomposition. '''

        funcName = sys._getframe().f_code.co_name
        print '\r', 'Running ', funcName

        # test classical decomposition degradation calc
        print 'Frequency: ', self.input_freq
        rd_result = degradation_classical_decomposition(self.test_CD_YOY_corr_energy)
        self.assertAlmostEqual(rd_result[0], 100*self.rd, places=1)
        print 'Actual: ', 100*self.rd
        print 'Estimated: ', rd_result[0]

        # TODO
        # - support for different time series frequencies
        # - inputs

    def test_degradation_year_on_year(self):
        ''' Test degradation with year on year approach. '''

        funcName = sys._getframe().f_code.co_name
        print '\r', 'Running ', funcName

        # test classical decomposition degradation calc
        print 'Frequency: ', self.input_freq
        rd_result = degradation_year_on_year(self.test_CD_YOY_corr_energy)
        self.assertAlmostEqual(rd_result[0], 100*self.rd, places=1)
        print 'Actual: ', 100*self.rd
        print 'Estimated: ', rd_result[0]

        # TODO
        # - support for different time series frequencies
        # - inputs


if __name__ == '__main__':

    # Allowed frequencies for degradation_ols
    list_ols_input_freq = ['MS','M','W','D','H','T','S']
    # Allowed frequencies for degradation_classical_decomposition and degradation_year_on_year
    list_CD_YOY_input_freq = ['MS','M','W','D']

    for input_freq in list_ols_input_freq:
        # Depending on frequency, run all tests or only test_degradation_with_ols
        suite = unittest.TestSuite()
        suite.addTest(DegradationTestCase("test_degradation_with_ols", \
                      input_freq, list_CD_YOY_input_freq))
        if input_freq in list_CD_YOY_input_freq:
            suite.addTest(DegradationTestCase("test_degradation_year_on_year",\
                          input_freq, list_CD_YOY_input_freq))
            suite.addTest(DegradationTestCase("test_degradation_classical_decomposition",\
                          input_freq, list_CD_YOY_input_freq))

        runner = unittest.TextTestRunner()
        runner.run(suite)
