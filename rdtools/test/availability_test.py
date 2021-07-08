"""
Test suite for inverter availability functions.
"""

import pytest
from pandas.testing import assert_series_equal
from conftest import assert_isinstance

from rdtools.availability import AvailabilityAnalysis

import pvlib
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

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
    many (~hundreds) combinations in the PARAMETER_SPACE list.

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


def test__calc_loss_subsystem(power_data):
    # implicitly sweeps across the parameter space because power_data is
    # parametrized
    inverter_power, meter_power, expected_loss = power_data
    # these values aren't relevant to this test, but the timeseries are
    # checked for timestamp consistency so just pass in dummy data:
    energy_cumulative = pd.Series(np.nan, meter_power.index)
    power_expected = pd.Series(np.nan, meter_power.index)
    aa = AvailabilityAnalysis(meter_power,
                              inverter_power,
                              energy_cumulative=energy_cumulative,
                              power_expected=power_expected)
    aa._calc_loss_subsystem(low_threshold=None, relative_sizes=None,
                            power_system_limit=None)
    actual_loss = aa.loss_subsystem
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
    # return dummy data for cumulative energy and expected power
    dummy = pd.Series(np.nan, df.index)
    return df, df.sum(axis=1), dummy


def test_calc_loss_subsystem_threshold(dummy_power_data):
    # test low_threshold parameter.
    # negative threshold means the inverter is never classified as offline
    inverter_power, meter_power, dummy = dummy_power_data
    aa = AvailabilityAnalysis(meter_power,
                              inverter_power,
                              energy_cumulative=dummy,
                              power_expected=dummy)
    aa._calc_loss_subsystem(low_threshold=-1, relative_sizes=None,
                            power_system_limit=None)
    actual_loss = aa.loss_subsystem
    assert actual_loss.sum() == 0


def test_calc_loss_subsystem_limit(dummy_power_data):
    # test system_power_limit parameter.
    # set it unrealistically low to verify it constrains the loss.
    # real max power is 2, real max loss is 1, so setting limit=1.5 sets max
    # loss to 0.5
    inverter_power, meter_power, dummy = dummy_power_data
    aa = AvailabilityAnalysis(meter_power,
                              inverter_power,
                              energy_cumulative=dummy,
                              power_expected=dummy)
    aa._calc_loss_subsystem(low_threshold=None, relative_sizes=None,
                            power_system_limit=1.5)
    actual_loss = aa.loss_subsystem
    assert actual_loss.max() == pytest.approx(0.5, abs=0.01)


@pytest.fixture
def difficult_data():
    # a nasty dataset with lots of downtime and almost no periods where
    # the two inverters are online simultaneously, so calculating the
    # relative sizes automatically gives the wrong answer.

    # generate a plausible clear-sky power signal
    times = pd.date_range('2019-01-01', '2019-01-06', freq='15min',
                          tz='US/Eastern', closed='left')
    location = pvlib.location.Location(40, -80)
    clearsky = location.get_clearsky(times, model='haurwitz')
    # just scale GHI to power for simplicity
    base_power = 2.5*clearsky['ghi']
    # but require a minimum irradiance to turn on, simulating start-up voltage
    base_power[clearsky['ghi'] < 20] = 0

    # 1 and 3, so the relative sizing is 0.5 and 1.5 (1/2 and 3/2)
    df = pd.DataFrame({
        'inv1_power': base_power,
        'inv2_power': base_power * 3,
    })
    relative_sizes = {'inv1_power': 0.5, 'inv2_power': 1.5}

    # inv1 offline days 1 & 2; inv2 offline days 3 & 4.  Both online on day 5
    df.loc['2019-01-01', 'inv1_power'] = 0
    df.loc['2019-01-02', 'inv1_power'] = 0
    df.loc['2019-01-03', 'inv2_power'] = 0
    df.loc['2019-01-04', 'inv2_power'] = 0

    # no need for communication outages here, so just take meter=inv sum
    expected_power = meter_power = df.sum(axis=1)

    return df, meter_power, expected_power, relative_sizes


def test_calc_loss_subsystem_relative_sizes(difficult_data):
    # test that manually passing in relative_sizes improves the results
    # for pathological datasets with tons of downtime
    invs, meter, expected, relative_sizes = difficult_data
    aa = AvailabilityAnalysis(meter,
                              invs,
                              energy_cumulative=meter.cumsum()/4,
                              power_expected=expected)
    # verify that results are bad by default -- without the correction, the
    # two inverters are weighted equally, so availability will be 50% when
    # only one is online
    aa.run(rollup_period='d')
    ava = aa.results['availability']
    assert np.allclose(ava.iloc[0:4], 0.5)
    assert np.allclose(ava.iloc[4], 1.0)

    # now use the correct relative_sizes
    aa.run(rollup_period='d', relative_sizes=relative_sizes)
    ava = aa.results['availability']
    assert np.allclose(ava.iloc[0:2], 0.75)
    assert np.allclose(ava.iloc[2:4], 0.25)
    assert np.allclose(ava.iloc[4], 1.0)

# %%


def test__calc_loss_system(energy_data):
    # test single outage
    (meter_power,
     meter_energy,
     inverter_power,
     expected_power,
     expected_loss,
     expected_type) = energy_data

    aa = AvailabilityAnalysis(meter_power, inverter_power,
                              meter_energy, expected_power)
    aa.run()
    outage_info = aa.outage_info

    # only one outage
    assert len(outage_info) == 1
    outage_info = outage_info.iloc[0, :]

    # outage was correctly classified:
    assert outage_info['type'] == expected_type

    # outage loss is accurate to 5% of the true value:
    assert outage_info['loss'] == pytest.approx(expected_loss, rel=0.05)


def test__calc_loss_system_multiple(energy_data):
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
    aa = AvailabilityAnalysis(meter_power, inverter_power,
                              meter_energy, expected_power)
    aa.run()
    outage_info = aa.outage_info
    assert len(outage_info) == 2


@pytest.mark.parametrize('side', ['start', 'end'])
def test__calc_loss_system_startend(side, energy_data_outage_single):
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
        expected_start = '2019-01-01 00:00'
        expected_end = '2019-01-02 07:45'
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

    aa = AvailabilityAnalysis(meter_power, inverter_power,
                              meter_energy, expected_power)
    aa.run()
    outage_info = aa.outage_info
    actual_start = outage_info['start'].iloc[idx].strftime('%Y-%m-%d %H:%M')
    actual_end = outage_info['end'].iloc[idx].strftime('%Y-%m-%d %H:%M')
    assert actual_start == expected_start
    assert actual_end == expected_end


def test__calc_loss_system_quantiles(energy_data_comms_single):
    # exercise the quantiles parameter
    (meter_power,
     meter_energy,
     inverter_power,
     expected_power,
     _, _) = energy_data_comms_single

    # first make sure it gets picked up as a comms outage with normal quantiles
    aa = AvailabilityAnalysis(meter_power, inverter_power,
                              meter_energy, expected_power)
    aa.run(quantiles=(0.01, 0.99))
    outage_info = aa.outage_info
    assert outage_info['type'].values[0] == 'comms'

    # set the lower quantile very high so that the comms outage gets
    # classified as a real outage
    aa = AvailabilityAnalysis(meter_power, inverter_power,
                              meter_energy, expected_power)
    aa.run(quantiles=(0.999, 0.9999))
    outage_info = aa.outage_info
    assert outage_info['type'].values[0] == 'real'


# %% plotting

def test_plot(availability_analysis_object):
    result = availability_analysis_object.plot()
    assert_isinstance(result, plt.Figure)


# %% errors

def test_plot_norun(dummy_power_data):
    _, _, dummy = dummy_power_data
    aa = AvailabilityAnalysis(dummy, dummy, dummy, dummy)
    # don't call run, just go straight to plot
    with pytest.raises(TypeError, match="No results to plot"):
        aa.plot()


def test_availability_analysis_index_mismatch(energy_data_outage_single):
    # exercise the timeseries index check
    (meter_power,
     meter_energy,
     inverter_power,
     expected_power,
     _, _) = energy_data_outage_single

    base_kwargs = {
        'power_system': meter_power,
        'power_subsystem': inverter_power,
        'energy_cumulative': meter_energy,
        'power_expected': expected_power,
    }
    # verify that the check works for any of the timeseries inputs
    for key in base_kwargs.keys():
        kwargs = base_kwargs.copy()
        value = kwargs.pop(key)
        value_shortened = value.iloc[1:]
        kwargs[key] = value_shortened
        with pytest.raises(ValueError, match='timeseries indexes must match'):
            _ = AvailabilityAnalysis(**kwargs)


def test_availability_analysis_doublecount_loss(availability_analysis_object):
    # test that a warning is emitted when loss is found at both the
    # system and subsystem levels. I don't know how to trigger the warning
    # with real data, so we'll "hack" the analysis object:
    loss = pd.Series(0, index=availability_analysis_object.power_system.index)
    loss.iloc[0] = 1
    availability_analysis_object.loss_system = loss
    availability_analysis_object.loss_subsystem = loss
    match = 'Loss detected simultaneously at both system and subsystem levels.'
    with pytest.warns(UserWarning, match=match):
        availability_analysis_object._combine_losses()
