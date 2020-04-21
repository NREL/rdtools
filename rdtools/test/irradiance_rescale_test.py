import pandas as pd
from pandas.testing import assert_series_equal
from rdtools import irradiance_rescale
from rdtools.normalization import ConvergenceError
import pytest


@pytest.fixture
def simple_irradiance():
    times = pd.date_range('2019-06-01 12:00', freq='15T', periods=5)
    time_series = pd.Series([1, 2, 3, 4, 5], index=times)
    return time_series


@pytest.mark.parametrize("method", ['iterative', 'single_opt'])
def test_rescale(method, simple_irradiance):
    # test basic functionality
    modeled = simple_irradiance
    measured = 1.05 * simple_irradiance
    rescaled = irradiance_rescale(measured, modeled, method=method)
    expected = measured
    assert_series_equal(rescaled, expected, check_exact=False)



def test_max_iterations(simple_irradiance):
    # check that max_iterations is actually used.  passing zero should fail
    with pytest.raises(UnboundLocalError):
        _ = irradiance_rescale(simple_irradiance, simple_irradiance,
                               max_iterations=0, method='iterative')


def test_non_convergence(simple_irradiance):
    # use iterative method without enough iterations to converge
    measured = simple_irradiance * 100  # method expects irrad > 200
    modeled = measured.copy()
    modeled.iloc[2] *= 1.1
    modeled.iloc[3] *= 1.3
    modeled.iloc[4] *= 0.8

    with pytest.raises(ConvergenceError):
        _ = irradiance_rescale(measured, modeled, method='iterative',
                               max_iterations=2)

    _ = irradiance_rescale(measured, modeled, method='iterative',
                           max_iterations=10)
