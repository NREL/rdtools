import h5py
from numpy import arange
from datetime import timedelta, datetime
import os
import math
import pandas as pd
import pkg_resources





def get_clearsky_tamb(times, latitude, longitude, utc_offset):
    '''
    :param times:       DateTimeIndex in local time
    :param latitude:    float degrees
    :param longitude:   float degrees
    :return:            pandas Series of cell sky ambient temperature
    '''


    filepath = pkg_resources.resource_filename('rdtools', 'data/temperature.hdf5')

    buffer = timedelta(days=80)
    interval = times[1] - times[0]
    points_per_day = int(timedelta(days=1)/interval)
    dt = pd.date_range(times[0] - buffer, times[-1] + buffer, freq=interval)

    #print model


    f = h5py.File(filepath, "r")

    a = f['temperature']['day']
    b = f['temperature']['night']

    lons = len(a[:, 0, 0])
    lats = len(a[0, :, 0])

    lon_temp = longitude - 180
    if lon_temp  < 0:
        lon_temp += 360
    lon_index = round(float(lons) * float(lon_temp) / 360.0)
    lat_index = round(float(lats) * (90.0 - float(latitude)) / 180.0)

    #print lons, lats, lon_index, lat_index


    df = pd.DataFrame(index=dt)
    df['month'] = df.index.month

    ave_day = []
    ave_night = []

    radius = 0
    for k in range(12):

        day = _get_pixel_value(a,lon_index,lat_index,k,radius)
        night = _get_pixel_value(b, lon_index, lat_index, k, radius)

        if day == float("NaN"):
            day = a[:,lat_index,k]
        if night == float("NaN"):
            night = a[:,lat_index,k]

        ave_day.append(day)
        ave_night.append(night)


    #print ave_day, ave_night



    for i in range(12):
        df.loc[df['month']== i+1, 'day'] = ave_day[i]
        df.loc[df['month'] == i+1, 'night'] = ave_night[i]


    df = df.rolling(window=40 * points_per_day, win_type='gaussian').mean(std=20 * points_per_day)
    df = df[(df.index >= times[0]) & (df.index <= times[-1])]

    utc_offsets = [y.utcoffset().total_seconds()/3600.0 for y in df.index]
    solar_noon_offset = lambda utc_offset : longitude / 180.0 * 12.0 - utc_offset
    df['solar_noon_offset'] = [solar_noon_offset(utc_offset) for utc_offset in utc_offsets]

    df['hour_of_day'] = df.index.hour + df.index.minute / 60.0
    df['Clear Sky Temperature (C)'] = df.apply(lambda x:
                                               _get_temperature(x['hour_of_day'], x['night'], 
                                                                x['day'],x['solar_noon_offset']), axis=1)
    return df['Clear Sky Temperature (C)']


def _get_pixel_value(data, i, j, k, radius):
    list = []
    for x in arange(i-radius,i+radius+1):
        if x < 0 or x >= len(data[:, 0, 0]):
            continue
        for y in arange(j - radius, j + radius + 1):
            if y < 0 or y >= len(data[0,:,0]):
                continue

            value = data[x, y, k]
            if value == float("NaN"):
                continue

            list.append(value)

    if len(list) == 0:
        return float("NaN")

    return pd.Series(list).median()


def _get_temperature(hour_of_day, night_temp, day_temp, solar_noon_offset):
    hour_offset = -4.0 + solar_noon_offset
    temp_scaler = 0.7
    t_diff = day_temp - night_temp
    t_ave = (day_temp + night_temp) / 2.0
    v = math.cos((hour_of_day + hour_offset) / 24.0 * 2.0 * math.pi)
    t = t_diff * 0.5 * v * temp_scaler + t_ave
    return t


