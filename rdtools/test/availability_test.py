"""
Test suite for inverter availability code.
"""

import pytest
from pandas.testing import assert_series_equal

from rdtools.availability import loss_from_power, loss_from_energy

import pvlib
import pandas as pd
import numpy as np
import itertools

# Values to parametrize power tests across.  One test will be run for each
# combination. Can't be careless about expanding this list because of
# combinatorial explosion.
PARAMETER_SPACE = list(itertools.product(
    [0, np.nan],  # values that power data takes during comms outage
    [0, np.nan, 0.001, -0.001],  # values during real downtime
    [1.0, 3.0],  # relative inverter capacities
    [1, 3],  # the number of inverters per system
    [False, True],  # whether any systems are really offline
    [False, True],  # whether a comms outage occurs
))

# display names for the test cases.  default is just 0..N
PARAMETER_IDS = ["_".join(map(str, p)) for p in PARAMETER_SPACE]


@pytest.fixture(params=PARAMETER_SPACE, ids=PARAMETER_IDS)
def power_data(request):
    """
    Generate power test cases corresponding to cover different system designs
    and data artifacts caused by outages. This fixture is parametrized across
    many of combinations (~hundreds) in the PARAMETER_SPACE list.

    Returns a tuple:
        - inverter_power, dataframe
        - meter_power, series
        - expected_loss, series
    """
    # unpack the parameters:
    comms_value, outage_value, relative_sizing, n_inverter, \
        has_power_outage, has_comms_outage = request.param

    # a few days of clearsky irradiance for creating a plausible power signal
    times = pd.date_range('2019-01-01', '2019-01-05 23:59', freq='15min',
                          tz='US/Eastern')
    location = pvlib.location.Location(40, -80)
    clearsky = location.get_clearsky(times)

    # just set base inverter power = ghi+clipping for simplicity
    base_power = clearsky['ghi'].clip(upper=0.8*clearsky['ghi'].max())

    inverter_power = pd.DataFrame({'inv1': base_power})
    if n_inverter == 3:
        inverter_power['inv2'] = base_power / relative_sizing
        inverter_power['inv3'] = base_power * relative_sizing

    expected_loss = pd.Series(0, index=times, dtype=float)

    if has_power_outage:
        date = '2019-01-01'
        # this expected_loss calculation is not exactly the same as what the
        # function uses, but it is quite close.  Need to do the comparison
        # with appropriate precision.
        expected_loss.loc[date] = inverter_power.loc[date, 'inv1']
        lim = inverter_power['inv1'].quantile(0.99) / 1000
        expected_loss[inverter_power['inv1'] < lim] = 0
        inverter_power.loc[date, 'inv1'] = outage_value

        # special case:  if n_inv == 1, the method can't estimate the loss:
        if n_inverter == 1:
            expected_loss.loc[date] = 0

    # meter_power reflects real inverter-level outages, but not
    # inverter-level comms outages, so do the sum before adding comms outages:
    meter_power = inverter_power.sum(axis=1)

    if has_comms_outage:
        inverter_power.loc['2019-01-02', 'inv1'] = comms_value

    return inverter_power, meter_power, expected_loss


def test_loss_from_power(power_data):
    # implicitly sweeps across the parameter space because power_data is
    # parametrized
    inverter_power, meter_power, expected_loss = power_data
    actual_loss = loss_from_power(inverter_power, meter_power)
    # pandas <1.1.0 as no atol/rtol parameters, so just use np.round instead:
    assert_series_equal(np.round(expected_loss, 1),
                        np.round(actual_loss, 1))

