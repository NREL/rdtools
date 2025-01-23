import pytest
import datetime
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


def test_not_on_land():
    # test that specifying a point in the ocean returns NaN and warns
    dt = pd.date_range('2015-01-01', freq='15min', periods=1, tz='UTC')
    with pytest.warns(UserWarning, match='possibly invalid Lat/Lon coordinates'):
        ocean_cs_tamb = get_clearsky_tamb(dt, 40, -60)
    assert ocean_cs_tamb.isnull().all()


def test_with_tricky_timezones():
    # Some timezones have DST shifts at midnight, which
    # can lead to NonExistentTimeError. This tests for the
    # problem in issue #372

    tz = 'America/Santiago'
    start_date = datetime.datetime(2018, 8, 10, 0, 0, 0)
    end_date = datetime.datetime(2018, 8, 14, 23, 0, 0)
    freq = "h"
    lat = -24
    lon = -70

    times = pd.date_range(start=start_date, end=end_date, freq=freq)
    times = times.tz_localize(tz=tz, ambiguous='infer',
                              nonexistent='shift_forward')
    get_clearsky_tamb(times, lat, lon)
