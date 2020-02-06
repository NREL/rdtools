import pandas as pd
import numpy as np
from rdtools import qcrad

import pytest
from pandas.util.testing import assert_series_equal


@pytest.fixture
def irradiance():
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

    return output


def test_check_irradiance_limits(irradiance):
    ghi_limits, dhi_limits, dni_limits = qcrad.check_irradiance_limits(
        irradiance['solar_zenith'],
        irradiance['dni_extra'],
        ghi=irradiance['ghi'])

    assert_series_equal(
        ghi_limits,
        irradiance['ghi_limit_flag']
    )
    assert dhi_limits is None
    assert dni_limits is None

    ghi_limits, dhi_limits, dni_limits = qcrad.check_irradiance_limits(
        irradiance['solar_zenith'],
        irradiance['dni_extra'],
        ghi=irradiance['ghi'],
        dhi=irradiance['dhi'],
        dni=irradiance['dni']
    )

    assert_series_equal(
        dhi_limits,
        irradiance['dhi_limit_flag']
    )
    assert_series_equal(
        dni_limits,
        irradiance['dni_limit_flag']
    )

    ghi_limits, dhi_limits, dni_limits = qcrad.check_irradiance_limits(
        irradiance['solar_zenith'],
        irradiance['dni_extra']
    )

    assert ghi_limits is None
    assert dni_limits is None
    assert dhi_limits is None

def test_check_ghi_limits(irradiance):
    expected = irradiance
    ghi_out_expected = expected['ghi_limit_flag']
    ghi_out = qcrad.check_ghi_limits(expected['ghi'],
                                     expected['solar_zenith'],
                                     expected['dni_extra'])
    assert_series_equal(ghi_out, ghi_out_expected)

def test_check_dhi_limits(irradiance):
    expected = irradiance
    dhi_out_expected = expected['dhi_limit_flag']
    dhi_out = qcrad.check_dhi_limits(expected['dhi'],
                                     expected['solar_zenith'],
                                     expected['dni_extra'])
    assert_series_equal(dhi_out, dhi_out_expected)

def test_check_dni_limits(irradiance):
    expected = irradiance
    dni_out_expected = expected['dni_limit_flag']
    dni_out = qcrad.check_dni_limits(expected['dni'],
                                     expected['solar_zenith'],
                                     expected['dni_extra'])
    assert_series_equal(dni_out, dni_out_expected)

def test_check_irradiance_consistency_QCRad(irradiance):
    expected = irradiance
    cons_comp, diffuse = qcrad.check_irradiance_consistency(
        expected['ghi'], expected['solar_zenith'], expected['dni_extra'],
        expected['dhi'], expected['dni'])
    assert_series_equal(cons_comp, expected['consistent_components'])
    assert_series_equal(diffuse, expected['diffuse_ratio_limit'])
