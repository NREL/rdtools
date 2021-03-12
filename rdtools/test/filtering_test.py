""" Filtering Module Tests. """

import pytest

import pandas as pd
import numpy as np

from rdtools import (csi_filter,
                     poa_filter,
                     tcell_filter,
                     clip_filter,
                     normalized_filter,
                     logic_clip_filter,
                     xgboost_clip_filter)


def test_csi_filter():
    ''' Unit tests for clear sky index filter.'''

    measured_poa = np.array([1, 1, 0, 1.15, 0.85])
    clearsky_poa = np.array([1, 2, 1, 1.00, 1.00])
    filtered = csi_filter(measured_poa,
                          clearsky_poa,
                          threshold=0.15)
    # Expect clearsky index is filtered with threshold of +/- 0.15.
    expected_result = np.array([True, False, False, True, True])
    assert filtered.tolist() == expected_result.tolist()


def test_poa_filter():
    ''' Unit tests for plane of array insolation filter.'''

    measured_poa = np.array([201, 1199, 500, 200, 1200])
    filtered = poa_filter(measured_poa,
                          poa_global_low=200,
                          poa_global_high=1200)
    # Expect high and low POA cutoffs to be non-inclusive.
    expected_result = np.array([True, True, True, False, False])
    assert filtered.tolist() == expected_result.tolist()


def test_tcell_filter():
    ''' Unit tests for cell temperature filter.'''

    tcell = np.array([-50, -49, 0, 109, 110])
    filtered = tcell_filter(tcell,
                            temperature_cell_low=-50,
                            temperature_cell_high=110)

    # Expected high and low tcell cutoffs to be non-inclusive.
    expected_result = np.array([False, True, True, True, False])
    assert filtered.tolist() == expected_result.tolist()


def test_clip_filter():
    ''' Unit tests for inverter clipping filter.'''

    power = pd.Series(np.arange(1, 101))
    # Note: Power is expected to be Series object because clip_filter makes
    #       use of the Series.quantile() method.
    filtered = clip_filter(power, quantile=0.98)
    # Expect 99% of the 98th quantile to be filtered
    expected_result = power < (98 * 0.99)
    assert ((expected_result == filtered).all())


@pytest.fixture
def generate_power_time_series():
    power_no_datetime_index = pd.Series(np.arange(1, 101))
    power_datetime_index = pd.Series(np.arange(1, 101))
    # Add datetime index to second series
    time_range = pd.date_range('2016-12-02T11:00:00.000Z',
                               '2017-06-06T07:00:00.000Z', freq='H')
    power_datetime_index.index = pd.to_datetime(time_range[:100])
    # Note: Power is expected to be Series object with a datetime index.
    return power_no_datetime_index, power_datetime_index


def test_logic_clip_filter(generate_power_time_series):
    ''' Unit tests for geometric clipping filter.'''
    power_no_datetime_index, power_datetime_index = \
        generate_power_time_series
    # Test that a Type Error is raised when a pandas series
    # without a datetime index is used.
    pytest.raises(TypeError,  logic_clip_filter,
                  power_no_datetime_index)
    # Expect none of the sequence to be clipped (as it's
    # constantly increasing)
    filtered, mask = logic_clip_filter(power_datetime_index)
    assert (not mask.all())


def test_xgboost_clip_filter(generate_power_time_series):
    ''' Unit tests for geometric clipping filter.'''
    power_no_datetime_index, power_datetime_index = \
        generate_power_time_series
    
    # Test that a Type Error is raised when a pandas series
    # without a datetime index is used.
    pytest.raises(TypeError,  xgboost_clip_filter,
                  power_no_datetime_index)
    # Expect none of the sequence to be clipped (as it's
    # constantly increasing)
    filtered, mask = xgboost_clip_filter(power_datetime_index)
    assert (not mask.all())


def test_normalized_filter_default():
    pd.testing.assert_series_equal(normalized_filter(pd.Series([-5, 5])),
                                   pd.Series([False, True]))
    pd.testing.assert_series_equal(normalized_filter(
                        pd.Series([-1e6, 1e6]),
                        energy_normalized_low=None,
                        energy_normalized_high=None),
                        pd.Series([True, True]))

    pd.testing.assert_series_equal(normalized_filter(
                                pd.Series([-2, 2]),
                                energy_normalized_low=-1,
                                energy_normalized_high=1),
                                pd.Series([False, False]))

    eps = 1e-16
    pd.testing.assert_series_equal(normalized_filter(
                        pd.Series([0.01 - eps, 0.01 + eps, 1e308])),
                        pd.Series([False, True, True]))
