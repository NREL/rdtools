import unittest
import pandas as pd
from datetime import datetime

from rdtools.clearsky_temperature import get_clearsky_tamb


class ClearSkyTemperatureTestCase(unittest.TestCase):
    '''Unit tests for clearsky_temperature module'''

    def setUp(self):

        dt = pd.date_range(datetime(2015,1,1), datetime(2015,2,1), freq='15min', tz = 'Asia/Shanghai')

        self.china_west = get_clearsky_tamb(dt, 37.951721, 80.609843)
        self.china_east = get_clearsky_tamb(dt, 36.693692, 117.699686)

        #self.china_west.to_csv("west.csv")
        #self.china_east.to_csv("east.csv")


    # Test for shifting temperature peak with longitude for same timezone
    def test_hour_offset(self):

        df = pd.DataFrame(index = self.china_west.index)
        df['west'] = self.china_west
        df['east'] = self.china_east
        df['hour'] = df.index.hour

        west_hottest_hour = df.sort_values(by='west', ascending=False)['hour'].iloc[0]
        east_hottest_hour = df.sort_values(by='east', ascending=False)['hour'].iloc[0]

        #print west_hottest_hour , east_hottest_hour

        self.assertTrue(west_hottest_hour > 12)
        self.assertTrue(east_hottest_hour > 12)
        self.assertTrue(west_hottest_hour > east_hottest_hour)

# TODO:
# Test irradiance_rescale


if __name__ == '__main__':
    unittest.main()
