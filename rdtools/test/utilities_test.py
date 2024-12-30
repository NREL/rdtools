import pandas as pd
import numpy as np
import pytest
from rdtools.utilities import robust_quantile, robust_median, robust_mean


@pytest.fixture
def data():
    data_zeros = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_nan = pd.Series([np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return data_zeros, data_nan


def test_robust_quantile(data):
    data_zeros, data_nan = data
    quantile = 0.5
    expected_result = 5.5
    assert expected_result == robust_quantile(data_zeros, quantile)
    assert expected_result == robust_quantile(data_nan, quantile)

    quantile = 0.25
    expected_result = 3.25
    assert expected_result == robust_quantile(data_zeros, quantile)
    assert expected_result == robust_quantile(data_nan, quantile)

    quantile = 0.75
    expected_result = 7.75
    assert expected_result == robust_quantile(data_zeros, quantile)
    assert expected_result == robust_quantile(data_nan, quantile)


def test_robust_median(data):
    data_zeros, data_nan = data
    expected_result = 5.5
    assert expected_result == robust_median(data_zeros)
    assert expected_result == robust_median(data_nan)


def test_robust_mean(data):
    data_zeros, data_nan = data
    expected_result = 5.5
    assert expected_result == robust_mean(data_zeros)
    assert expected_result == robust_mean(data_nan)
