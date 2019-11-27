"""
Unit tests for inverter availability functions
"""

import pytest

import pandas as pd
import numpy as np
import pvlib
import math

from rdtools import is_online, downtime_loss, signal_to_profile


# make up some test clear-sky meter power data
st = '2019-01-01 00:00'
ed = '2019-02-01 00:00'
kWdc = 100.0
gamma = -0.004
lat, lon = 40, -80
tilt = 20
azi = 180
idx = pd.date_range(st, ed, freq='15T', closed='left', tz='US/Eastern')
location = pvlib.location.Location(lat, lon)
solpos = location.get_solarposition(idx)
clearsky = location.get_clearsky(idx, solar_position=solpos)
poa = pvlib.irradiance.get_total_irradiance(tilt,
                                            azi,
                                            solpos['zenith'],
                                            solpos['azimuth'],
                                            clearsky['dni'],
                                            clearsky['ghi'],
                                            clearsky['dhi'])
poa = poa['poa_global']
tamb = pd.Series(index=idx, data=25)
tcell = pvlib.pvsystem.sapm_celltemp(poa, 0, tamb)
tcell = tcell['temp_cell']

meter = kWdc * poa/1000 * (1 + gamma*(tcell - 25))
is_daylight = solpos['elevation'] > 5


def test_is_online_single_inverter_online():
    # one inverter, always online during day
    inv = meter.to_frame()
    mask = is_online(inv, meter)
    uptime = mask.loc[is_daylight, :].mean().mean()
    assert math.isclose(uptime, 1)


@pytest.mark.parametrize("offline_value", [0, np.nan, 1e-3])
def test_is_online_single_inverter_offline(offline_value):
    # one inverter offline with various readings for a day
    inv = meter.copy().to_frame()
    inv.loc['2019-01-10', :] = offline_value

    meter_adjusted = meter.copy()
    meter_adjusted.loc['2019-01-10'] = 0

    mask = is_online(inv, meter_adjusted)
    uptime = mask.loc[is_daylight, :].mean().mean()
    assert math.isclose(uptime, 30/31, rel_tol=.01)


@pytest.mark.parametrize("relative_sizing", [0.25, 0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("offline_value", [0, np.nan, 1e-3])
@pytest.mark.parametrize("offline_invs, expected_uptime", [
        ([],      1),                   # no downtime
        ([0],     1 - (1/31) / 3),      # individual inverters offline
        ([1],     1 - (1/31) / 3),
        ([2],     1 - (1/31) / 3),
        ([0,1],   1 - (1/31) * 2 / 3),  # first two inverters offline
        ([0,1,2], 1 - (1/31))           # all three inverters offline
])
def test_is_online_multiple_inverters_offline(relative_sizing,
                                              offline_value,
                                              offline_invs,
                                              expected_uptime):
    """
    Test if losses.is_online can handle:
        - any number of inverters offline
        - wide range of relative inverter sizing
        - various symptoms of being offline (0, nan, 0 < epsilon << 100%)
        - inverters that look offline but are really just not communicating
          (doesn't work if some are comms and some are actually offline)
    """
    # N inverters offline with various readings for one day
    inv = meter / (relative_sizing + 1 + 1/relative_sizing)
    online_invs = pd.DataFrame({
        0: inv * relative_sizing,
        1: inv,
        2: inv / relative_sizing
    })

    invs = online_invs.copy()
    invs.loc['2019-01-10', offline_invs] = offline_value

    # meter greater than inverter sum, just a comms outage
    mask = is_online(invs, meter)
    uptime = mask.loc[is_daylight, :].mean().mean()
    assert math.isclose(uptime, 1)

    # meter matches inverter sum, aka real downtime
    invs_zero = online_invs.copy()
    invs_zero.loc['2019-01-10', offline_invs] = 0
    meter_adjusted = invs_zero.sum(axis=1)
    mask = is_online(invs, meter_adjusted)
    uptime = mask.loc[is_daylight, :].mean().mean()
    assert math.isclose(uptime, expected_uptime, rel_tol=0.01)


def test_downtime_loss():
    """
    Test if losses.downtime_loss can handle:
        - single inverters offline
        - all inverters offline
        - differently-sized inverters
        - inverter comms outages
    """
    ideal_inverter_power = pd.DataFrame({
        'inv0': (meter/3).clip(upper=20),
        'inv1': (2*meter/3).clip(upper=40)
    })
    expected_power = ideal_inverter_power.sum(axis=1)
    production_profile = signal_to_profile(expected_power)

    inverter_power = ideal_inverter_power.copy()
    # introduce various outages
    inverter_power.loc['2019-01-03', 'inv0'] = 0
    inverter_power.loc['2019-01-05 10:00':'2019-01-05 14:00', 'inv1'] = 0
    inverter_power.loc['2019-01-07 12:00':'2019-01-08 10:00', :] = 0
    inverter_power.loc['2019-01-06'] = 0

    # this one is just an inverter comms outage though
    meter_adjusted = inverter_power.sum(axis=1)
    meter_adjusted['2019-01-06'] = expected_power['2019-01-06']

    online_mask = is_online(inverter_power, meter_adjusted)
    lost_power = downtime_loss(
            inverter_power,
            meter_adjusted,
            online_mask,
            expected_power,
            production_profile,
            is_daylight
    )
    daily_loss = lost_power.resample('d').sum() / 4
    observed = daily_loss['2019-01-03']
    expected = ideal_inverter_power.loc['2019-01-03',
                                        'inv0'].sum() / 4
    assert abs(observed / expected) - 1 < 0.01

    observed = daily_loss['2019-01-05']
    expected = ideal_inverter_power.loc['2019-01-05 10:00':'2019-01-05 14:00',
                                        'inv1'].sum() / 4
    assert abs(observed / expected) - 1 < 0.01

    observed = daily_loss['2019-01-07'] + daily_loss['2019-01-08']
    expected = ideal_inverter_power.loc['2019-01-07 12:00':'2019-01-08 10:00',
                                        :].sum().sum() / 4
    assert abs(observed / expected) - 1 < 0.01

    observed = daily_loss['2019-01-06']
    assert observed == 0


def test_downtime_loss_mixed():
    """
    The ultimate test:  some inverters offline, some only non-communicative.

    Jan 3: inv0 offline, inv1 not communicating
    Jan 4: inv1 offline, inv2+inv3 not communicating
    Jan 5: inv2 offline, inv1 not communicating
    Jan 6: inv3 offline, no others communicating
    """
    ideal_inverter_power = pd.DataFrame({
        'inv0': 0.1 * meter,
        'inv1': 0.2 * meter,
        'inv2': 0.3 * meter,
        'inv3': 0.4 * meter,
    })
    expected_power = ideal_inverter_power.sum(axis=1)
    production_profile = signal_to_profile(expected_power)

    inverter_power = ideal_inverter_power.copy()
    # introduce production outages
    inverter_power.loc['2019-01-03', 'inv0'] = 0
    inverter_power.loc['2019-01-04', 'inv1'] = 0
    inverter_power.loc['2019-01-05', 'inv2'] = 0
    inverter_power.loc['2019-01-06', 'inv3'] = 0

    # record meter data before applying inverter comms outages
    meter_adjusted = inverter_power.sum(axis=1)

    # introduce comms outages
    inverter_power.loc['2019-01-03', 'inv1'] = 0
    inverter_power.loc['2019-01-03', ['inv2', 'inv3']] = 0
    inverter_power.loc['2019-01-03', 'inv1'] = 0
    inverter_power.loc['2019-01-03', :] = 0

    online_mask = is_online(inverter_power, meter_adjusted)
    lost_power = downtime_loss(
            inverter_power,
            meter_adjusted,
            online_mask,
            expected_power,
            production_profile,
            is_daylight
    )
    daily_loss = lost_power.resample('d').sum() / 4

    observed = daily_loss['2019-01-03']
    expected = ideal_inverter_power.loc['2019-01-03',
                                        'inv0'].sum() / 4
    assert abs(observed / expected) - 1 < 0.01

    expected = ideal_inverter_power.loc['2019-01-04',
                                        'inv1'].sum() / 4
    assert abs(observed / expected) - 1 < 0.01

    expected = ideal_inverter_power.loc['2019-01-05',
                                        'inv2'].sum() / 4
    assert abs(observed / expected) - 1 < 0.01

    expected = ideal_inverter_power.loc['2019-01-06',
                                        'inv3'].sum() / 4
    assert abs(observed / expected) - 1 < 0.01
