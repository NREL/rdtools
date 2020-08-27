"""
Test suite for inverter availability functions.
"""

import pytest
from pandas.testing import assert_series_equal

from rdtools.availability import loss_from_power, loss_from_energy

import pvlib
import pandas as pd
import numpy as np
import itertools
import datetime

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

    The method is to generate some inverter power signals of varying scales,
    introduce power outages to some of them, calculate the system meter power
    as the summed inverter power, and then add inverter communication outages.

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
    # use haurwitz to avoid dependency on `tables`
    clearsky = location.get_clearsky(times, model='haurwitz')

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


@pytest.fixture
def dummy_power_data():
    # one inverter off half the time, one always online
    N = 10
    df = pd.DataFrame({
        'inv1': [0] * (N//2) + [1] * (N//2),
        'inv2': [1] * N,
    }, index=pd.date_range('2019-01-01', freq='h', periods=N))
    return df, df.sum(axis=1)


def test_loss_from_power_threshold(dummy_power_data):
    # test low_threshold parameter.
    # negative threshold means the inverter is never classified as offline
    inverter_power, meter_power = dummy_power_data
    actual_loss = loss_from_power(inverter_power, meter_power,
                                  low_threshold=-1)
    assert actual_loss.sum() == 0


def test_loss_from_power_limit(dummy_power_data):
    # test system_power_limit parameter.
    # set it unrealistically low to verify it constrains the loss.
    # real max power is 2, real max loss is 1, so setting limit=1.5 sets max
    # loss to 0.5
    inverter_power, meter_power = dummy_power_data
    actual_loss = loss_from_power(inverter_power, meter_power,
                                  system_power_limit=1.5)
    assert actual_loss.max() == pytest.approx(0.5, abs=0.01)


# %%

ENERGY_PARAMETER_SPACE = list(itertools.product(
    [0, np.nan],  # outage value for power
    [0, np.nan, None],  # value for cumulative energy (None means real value)
    [0, 0.25, 0.5, 0.75, 1.0],  # fraction of comms outage that is power outage
))
# display names for the test cases.  default is just 0..N
ENERGY_PARAMETER_IDS = ["_".join(map(str, p)) for p in ENERGY_PARAMETER_SPACE]


def _generate_energy_data(power_value, energy_value, outage_fraction):
    """
    Generate an artificial mixed communication/power outage.
    """
    # a few days of clearsky irradiance for creating a plausible power signal
    times = pd.date_range('2019-01-01', '2019-01-15 23:59', freq='15min',
                          tz='US/Eastern')
    location = pvlib.location.Location(40, -80)
    # use haurwitz to avoid dependency on `tables`
    clearsky = location.get_clearsky(times, model='haurwitz')

    # just set base inverter power = ghi+clipping for simplicity
    base_power = clearsky['ghi'].clip(upper=0.8*clearsky['ghi'].max())

    inverter_power = pd.DataFrame({
        'inv0': base_power,
        'inv1': base_power*0.7,
        'inv2': base_power*1.3,
    })
    expected_power = inverter_power.sum(axis=1)
    # dawn/dusk points
    expected_power[expected_power < 10] = 0
    # add noise and bias to the expected power signal
    np.random.seed(2020)
    expected_power *= 1.05 + np.random.normal(0, scale=0.05, size=len(times))

    # calculate what part of the comms outage is a power outage
    comms_outage = slice('2019-01-03 00:00', '2019-01-06 00:00')
    start = times.get_loc(comms_outage.start)
    stop = times.get_loc(comms_outage.stop)
    power_outage = slice(start, int(start + outage_fraction * (stop-start)))
    expected_loss = inverter_power.iloc[power_outage, :].sum().sum() / 4
    inverter_power.iloc[power_outage, :] = 0
    meter_power = inverter_power.sum(axis=1)
    meter_energy = meter_power.cumsum() / 4
    # add an offset because in practice cumulative meter data never
    # actually starts at 0:
    meter_energy += 100

    meter_power[comms_outage] = power_value
    if energy_value is not None:
        meter_energy[comms_outage] = energy_value
    inverter_power.loc[comms_outage, :] = power_value

    expected_type = 'real' if outage_fraction > 0 else 'comms'

    return (meter_power,
            meter_energy,
            inverter_power,
            expected_power,
            expected_loss,
            expected_type)


@pytest.fixture(params=ENERGY_PARAMETER_SPACE, ids=ENERGY_PARAMETER_IDS)
def energy_data(request):
    # fixture sweeping across the entire parameter space
    power_value, energy_value, outage_fraction = request.param
    return _generate_energy_data(power_value, energy_value, outage_fraction)


@pytest.fixture
def energy_data_outage_single():
    # fixture only using a single parameter combination, for simpler tests.
    # has one real outage.
    outage_value, outage_fraction = np.nan, 0.25
    return _generate_energy_data(outage_value, outage_value, outage_fraction)


@pytest.fixture
def energy_data_comms_single():
    # fixture only using a single parameter combination, for simpler tests.
    # has one comms outage.
    outage_value, outage_fraction = np.nan, 0
    return _generate_energy_data(outage_value, outage_value, outage_fraction)


def test_loss_from_energy(energy_data):
    # test single outage
    (meter_power,
     meter_energy,
     inverter_power,
     expected_power,
     expected_loss,
     expected_type) = energy_data

    outage_info = loss_from_energy(meter_power, meter_energy, inverter_power,
                                   expected_power)

    # only one outage
    assert len(outage_info) == 1
    outage_info = outage_info.iloc[0, :]

    # outage was correctly classified:
    assert outage_info['type'] == expected_type

    # outage loss is accurate to 5% of the true value:
    assert outage_info['loss'] == pytest.approx(expected_loss, rel=0.05)


def test_loss_from_energy_multiple(energy_data):
    # test multiple outages
    (meter_power,
     meter_energy,
     inverter_power,
     expected_power,
     _, _) = energy_data

    date = '2019-01-08'
    meter_power.loc[date] = 0
    meter_energy.loc[date] = 0
    inverter_power.loc[date] = 0
    outage_info = loss_from_energy(meter_power, meter_energy, inverter_power,
                                   expected_power)
    assert len(outage_info) == 2


@pytest.mark.parametrize('side', ['start', 'end'])
def test_loss_from_energy_startend(side, energy_data_outage_single):
    # data starts or ends in an outage
    (meter_power,
     meter_energy,
     inverter_power,
     expected_power,
     _, _) = energy_data_outage_single

    if side == 'start':
        # an outage all day on the 1st, so technically the outage extends to
        # sunrise on the 2nd, but it doesn't wrap around to the previous dusk
        date = '2019-01-01'
        expected_start = '2019-01-01 08:00'
        expected_end = '2019-01-02 08:00'
        idx = 0
    else:
        # last day doesn't have a "sunrise on the next day", so it doesn't
        # wrap around
        date = '2019-01-15'
        expected_start = '2019-01-14 17:15'
        expected_end = '2019-01-15 23:45'
        idx = -1

    meter_power.loc[date] = 0
    meter_energy.loc[date] = 0
    inverter_power.loc[date] = 0
    outage_info = loss_from_energy(meter_power, meter_energy, inverter_power,
                                   expected_power)
    actual_start = outage_info['start'].iloc[idx].strftime('%Y-%m-%d %H:%M')
    actual_end = outage_info['end'].iloc[idx].strftime('%Y-%m-%d %H:%M')
    assert actual_start == expected_start
    assert actual_end == expected_end


def test_loss_from_energy_quantiles(energy_data_comms_single):
    # exercise the quantiles parameter
    (meter_power,
     meter_energy,
     inverter_power,
     expected_power,
     _, _) = energy_data_comms_single

    # first make sure it gets picked up as a comms outage with normal quantiles
    outage_info = loss_from_energy(meter_power, meter_energy, inverter_power,
                                   expected_power, quantiles=(0.01, 0.99))
    assert outage_info['type'].values[0] == 'comms'

    # set the lower quantile very high so that the comms outage gets
    # classified as a real outage
    outage_info = loss_from_energy(meter_power, meter_energy, inverter_power,
                                   expected_power, quantiles=(0.999, 0.9999))
    assert outage_info['type'].values[0] == 'real'
