import pandas as pd
from pandas.testing import assert_series_equal
from rdtools import irradiance_rescale
from rdtools.normalization import ConvergenceError
import pytest


@pytest.fixture
def simple_irradiance():
    times = pd.date_range("2019-06-01 12:00", freq="15T", periods=5)
    time_series = pd.Series([1, 2, 3, 4, 5], index=times, dtype=float)
    return time_series


@pytest.mark.parametrize("method", ["iterative", "single_opt", "error"])
def test_rescale(method, simple_irradiance):
    # test basic functionality
    if method == "error":
        pytest.raises(
            ValueError,
            irradiance_rescale,
            simple_irradiance,
            simple_irradiance * 1.05,
            method=method,
        )
    else:
        modeled = simple_irradiance
        measured = 1.05 * simple_irradiance
        rescaled = irradiance_rescale(measured, modeled, method=method)
        expected = measured
        assert_series_equal(rescaled, expected, check_exact=False)


def test_max_iterations(simple_irradiance):
    # use iterative method without enough iterations to converge
    measured = simple_irradiance * 100  # method expects irrad > 200
    modeled = measured.copy()
    modeled.iloc[2] *= 1.1
    modeled.iloc[3] *= 1.3
    modeled.iloc[4] *= 0.8

    with pytest.raises(ConvergenceError):
        _ = irradiance_rescale(measured, modeled, method="iterative", max_iterations=2)

    _ = irradiance_rescale(measured, modeled, method="iterative", max_iterations=10)


def test_max_iterations_zero(simple_irradiance):
    # zero is sort of a special case, test it separately

    # test series already close enough
    true_factor = 1.0 + 1e-8
    rescaled = irradiance_rescale(
        simple_irradiance,
        simple_irradiance * true_factor,
        max_iterations=0,
        method="iterative",
    )
    assert_series_equal(rescaled, simple_irradiance, check_exact=False)

    # tighten threshold so that it isn't already close enough
    with pytest.raises(ConvergenceError):
        _ = irradiance_rescale(
            simple_irradiance,
            simple_irradiance * true_factor,
            max_iterations=0,
            convergence_threshold=1e-9,
            method="iterative",
        )


def test_convergence_threshold(simple_irradiance):
    # can't converge if threshold is negative
    with pytest.raises(ConvergenceError):
        _ = irradiance_rescale(
            simple_irradiance,
            simple_irradiance * 1.05,
            max_iterations=5,  # reduced count for speed
            convergence_threshold=-1,
            method="iterative",
        )
