import pytest
import pandas as pd

from rdtools.clearsky_temperature import get_clearsky_tamb


@pytest.fixture
def china_cs_tamb():
    dt = pd.date_range('2015-01-01', '2015-02-01', freq='15min', tz='Asia/Shanghai')
    china_west = get_clearsky_tamb(dt, 37.951721, 80.609843)
    china_east = get_clearsky_tamb(dt, 36.693692, 117.699686)
    return china_west, china_east


def test_hour_offset(china_cs_tamb):
    # Test for shifting temperature peak with longitude for same timezone
    china_west, china_east = china_cs_tamb

    df = pd.DataFrame(index=china_west.index)
    df['west'] = china_west
    df['east'] = china_east
    df['hour'] = df.index.hour

    west_hottest_hour = df.sort_values(by='west', ascending=False)['hour'].iloc[0]
    east_hottest_hour = df.sort_values(by='east', ascending=False)['hour'].iloc[0]

    assert west_hottest_hour > 12
    assert east_hottest_hour > 12
    assert west_hottest_hour > east_hottest_hour


