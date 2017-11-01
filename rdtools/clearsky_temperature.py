import h5py
from numpy import arange
from datetime import timedelta
import pandas as pd
import pkg_resources
import numpy as np


def get_clearsky_tamb(times, latitude, longitude, window_size=40, gauss_std=20):
    '''
    Description
    -----------
    Estimates the ambient temperature at latitude and longitude for the given times

    Parameters
    ----------
    times:       DateTimeIndex in local time
    latitude:    float degrees
    longitude:   float degrees

    Returns
    -------
    pandas Series of clear sky ambient temperature

    Reference
    ---------
    Uses data from images created by Jesse Allen, NASA's Earth Observatory
    using data courtesy of the MODIS Land Group.
    https://neo.sci.gsfc.nasa.gov/view.php?datasetId=MOD_LSTD_CLIM_M
    https://neo.sci.gsfc.nasa.gov/view.php?datasetId=MOD_LSTN_CLIM_M
    '''

    filepath = pkg_resources.resource_filename('rdtools', 'data/temperature.hdf5')

    buffer = timedelta(days=window_size)
    freq_actual = pd.infer_freq(times)
    dt_daily = pd.date_range(times[0] - buffer, times[-1] + buffer, freq='D')

    f = h5py.File(filepath, "r")

    a = f['temperature']['day']
    b = f['temperature']['night']

    lons = len(a[:, 0, 0])
    lats = len(a[0, :, 0])

    lon_temp = longitude - 180
    if lon_temp < 0:
        lon_temp += 360
    lon_index = round(float(lons) * float(lon_temp) / 360.0)
    lat_index = round(float(lats) * (90.0 - float(latitude)) / 180.0)

    df = pd.DataFrame(index=dt_daily)
    df['month'] = df.index.month

    ave_day = []
    ave_night = []

    radius = 0
    for k in range(12):

        day = _get_pixel_value(a, lon_index, lat_index, k, radius)
        night = _get_pixel_value(b, lon_index, lat_index, k, radius)

        if day == float("NaN"):
            day = a[:, lat_index, k]
        if night == float("NaN"):
            night = a[:, lat_index, k]

        ave_day.append(day)
        ave_night.append(night)

    for i in range(12):
        df.loc[df['month'] == i + 1, 'day'] = ave_day[i]
        df.loc[df['month'] == i + 1, 'night'] = ave_night[i]

    df = df.rolling(window=window_size, win_type='gaussian', min_periods=1, center=True).mean(std=gauss_std)

    df = df.resample(freq_actual).interpolate(method='linear')
    df['month'] = df.index.month

    df = df.reindex(times, method='nearest')

    utc_offsets = [y.utcoffset().total_seconds() / 3600.0 for y in df.index]

    def solar_noon_offset(utc_offset):
        return longitude / 180.0 * 12.0 - utc_offset

    df['solar_noon_offset'] = solar_noon_offset(np.array(utc_offsets))

    df['hour_of_day'] = df.index.hour + df.index.minute / 60.0
    df['Clear Sky Temperature (C)'] = _get_temperature(df['hour_of_day'].values, df['night'].values,\
                                                       df['day'].values, df['solar_noon_offset'].values)
    return df['Clear Sky Temperature (C)']


def _get_pixel_value(data, i, j, k, radius):
    list = []
    for x in arange(i - radius, i + radius + 1):
        if x < 0 or x >= len(data[:, 0, 0]):
            continue
        for y in arange(j - radius, j + radius + 1):
            if y < 0 or y >= len(data[0, :, 0]):
                continue

            value = data[x, y, k]
            if value == float("NaN"):
                continue

            list.append(value)

    if len(list) == 0:
        return float("NaN")

    return pd.Series(list).median()


def _get_temperature(hour_of_day, night_temp, day_temp, solar_noon_offset):
    hour_offset = 8.0 + solar_noon_offset
    temp_scaler = 0.7
    t_diff = day_temp - night_temp
    t_ave = (day_temp + night_temp) / 2.0
    v = np.cos((hour_of_day + hour_offset) / 24.0 * 2.0 * np.pi)
    t = t_diff * 0.5 * v * temp_scaler + t_ave
    return t
