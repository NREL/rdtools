import unittest
import pandas as pd
import numpy as np
from rdtools import qcrad

class QCRadTestCase(unittest.TestCase):

    def setUp(self):
        output = pd.DataFrame(
        columns=['ghi', 'dhi', 'dni', 'solar_zenith', 'dni_extra',
                 'ghi_limit_flag', 'dhi_limit_flag', 'dni_limit_flag',
                 'consistent_components', 'diffuse_ratio_limit'],
        data=np.array([[-100, 100, 100, 30, 1370, 0, 1, 1, 0, 0],
                       [100, -100, 100, 30, 1370, 1, 0, 1, 0, 0],
                       [100, 100, -100, 30, 1370, 1, 1, 0, 0, 1],
                       [1000, 100, 900, 0, 1370, 1, 1, 1, 1, 1],
                       [1000, 200, 800, 15, 1370, 1, 1, 1, 1, 1],
                       [1000, 200, 800, 60, 1370, 0, 1, 1, 0, 1],
                       [1000, 300, 850, 80, 1370, 0, 0, 1, 0, 1],
                       [1000, 500, 800, 90, 1370, 0, 0, 1, 0, 1],
                       [500, 100, 1100, 0, 1370, 1, 1, 1, 0, 1],
                       [1000, 300, 1200, 0, 1370, 1, 1, 1, 0, 1],
                       [500, 600, 100, 60, 1370, 1, 1, 1, 0, 0],
                       [500, 600, 400, 80, 1370, 0, 0, 1, 0, 0],
                       [500, 500, 300, 80, 1370, 0, 0, 1, 1, 1],
                       [0, 0, 0, 93, 1370, 1, 1, 1, 0, 0]]))
        dtypes = ['float64', 'float64', 'float64', 'float64', 'float64',
                  'bool', 'bool', 'bool', 'bool', 'bool']
        for (col, typ) in zip(output.columns, dtypes):
            output[col] = output[col].astype(typ)

        self.irradiance_QCRad = output

    def test_check_irradiance_limits(self):
        ghi_limits, dhi_limits, dni_limits = qcrad.check_irradiance_limits(
            self.irradiance_QCRad['solar_zenith'],
            self.irradiance_QCRad['dni_extra'],
            ghi=self.irradiance_QCRad['ghi'])

        self.assertListEqual(
            ghi_limits.tolist(),
            self.irradiance_QCRad['ghi_limit_flag'].tolist()
        )
        self.assertIsNone(dhi_limits)
        self.assertIsNone(dni_limits)

        ghi_limits, dhi_limits, dni_limits = qcrad.check_irradiance_limits(
            self.irradiance_QCRad['solar_zenith'],
            self.irradiance_QCRad['dni_extra'],
            ghi=self.irradiance_QCRad['ghi'],
            dhi=self.irradiance_QCRad['dhi'],
            dni=self.irradiance_QCRad['dni']
        )
        self.assertListEqual(
            dhi_limits.tolist(),
            self.irradiance_QCRad['dhi_limit_flag'].tolist()
        )
        self.assertListEqual(
            dni_limits.tolist(),
            self.irradiance_QCRad['dni_limit_flag'].tolist()
        )

        ghi_limits, dhi_limits, dni_limits = qcrad.check_irradiance_limits(
            self.irradiance_QCRad['solar_zenith'],
            self.irradiance_QCRad['dni_extra']
        )
        self.assertIsNone(ghi_limits)
        self.assertIsNone(dni_limits)
        self.assertIsNone(dhi_limits)

    def test_check_ghi_limits(self):
        expected = self.irradiance_QCRad
        ghi_out_expected = expected['ghi_limit_flag']
        ghi_out = qcrad.check_ghi_limits(expected['ghi'],
                                           expected['solar_zenith'],
                                           expected['dni_extra'])
        self.assertListEqual(ghi_out.tolist(), ghi_out_expected.tolist())

    def test_check_dhi_limits(self):
        expected = self.irradiance_QCRad
        dhi_out_expected = expected['dhi_limit_flag']
        dhi_out = qcrad.check_dhi_limits(expected['dhi'],
                                           expected['solar_zenith'],
                                           expected['dni_extra'])
        self.assertListEqual(dhi_out.tolist(), dhi_out_expected.tolist())

    def test_check_dni_limits(self):
        expected = self.irradiance_QCRad
        dni_out_expected = expected['dni_limit_flag']
        dni_out = qcrad.check_dni_limits(expected['dni'],
                                           expected['solar_zenith'],
                                           expected['dni_extra'])
        self.assertListEqual(dni_out.tolist(), dni_out_expected.tolist())

    def test_check_irradiance_consistency_QCRad(self):
        expected = self.irradiance_QCRad
        cons_comp, diffuse = qcrad.check_irradiance_consistency(
            expected['ghi'], expected['solar_zenith'], expected['dni_extra'],
            expected['dhi'], expected['dni'])
        self.assertListEqual(cons_comp.tolist(),
                             expected['consistent_components'].tolist())
        self.assertListEqual(diffuse.tolist(),
                             expected['diffuse_ratio_limit'].tolist())
