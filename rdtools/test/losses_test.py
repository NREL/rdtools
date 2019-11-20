"""
Unit tests for inverter availability functions
"""

import pytest

import pandas as pd
import numpy as np
import pvlib
import math

from rdtools import is_online


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


def test_is_online_multiple_inverters_online():
    # three inverters, always online during day
    inv = meter / 3
    online_invs = pd.DataFrame({
        0: inv,
        1: inv,
        2: inv
    })

    mask = is_online(online_invs, meter)
    uptime = mask.loc[is_daylight, :].mean().mean()
    assert math.isclose(uptime, 1)


@pytest.mark.parametrize("offline_value", [0, np.nan, 1e-3])
@pytest.mark.parametrize("offline_invs, expected_uptime", [
        ([0], 1 - (1/31) / 3),  # individual inverters offline
        ([1], 1 - (1/31) / 3),
        ([2], 1 - (1/31) / 3),
        ([0,1],   1 - (1/31) * 2 / 3),  # first two inverters offline
        ([0,1,2], 1 - (1/31))           # all three inverters offline
])
def test_is_online_multiple_inverters_offline(offline_value,
                                              offline_invs,
                                              expected_uptime):
    # N inverters offline with various readings for one day
    inv = meter / 3
    online_invs = pd.DataFrame({
        0: inv,
        1: inv,
        2: inv
    })

    invs = online_invs.copy()
    invs.loc['2019-01-10', offline_invs] = offline_value

    # meter greater than inverter sum, just a comms outage
    mask = is_online(invs, meter)
    uptime = mask.loc[is_daylight, :].mean().mean()
    assert math.isclose(uptime, 1)

    # meter matches inverter sum, aka real downtime
    meter_adjusted = meter.copy()
    meter_adjusted *= (1 - len(offline_invs)/3)
    mask = is_online(invs, meter_adjusted)
    uptime = mask.loc[is_daylight, :].mean().mean()
    assert math.isclose(uptime, expected_uptime, rel_tol=0.01)
