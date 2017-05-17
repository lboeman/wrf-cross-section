"""An object for containing the source data for
wind cross section plots
"""
from math import pi
from numpy import cos, sin
from netCDF4 import Dataset
from metpy.calc import pressure_to_height_std
from metpy.units import units
import sys
import logging
import datetime
import pandas as pd
import numpy as np
import mysql.connector
import mylogin


class CrossSectionData(object):
    """
    Attributes
    ----------
    wrf_data: netCDF4 Dataset
        A handle for the wrf_data being used.
    source_data: pandas Dataframe
        A dataframe to be used as the source for a bokeh plot.
    location_dict: dictionary
        A dictionary containing information on each wind station
        including lat, lon, origin coordinates and terminal
        coordinates for vertical(v1,v2) and horizontal(h1,h2) series.
    station: str
        The key of the current station being used in location_dict.
    lats: numpy ndarray
        The latitude variable from the WRF data.
    lons: numpy ndarray
        The longitude variable from the WRF data.
    init: datetime
    orientation: str
        "vertical" or "horizontal"
    times: set of ints
        The time steps present in the dataframe.
    """
    spread = 10

    def _tunnel_dist(self, lat_ar, lon_ar, lat, lon):
        """
        Finds the index of the closest point assuming a spherical earth.
        Parameters
        ----------
        lat_ar: numpy array
            The array of the corresponding netCDF's latitude variable
        lon_ar: numpy array
            The arrray of the corresponding netCDF's longitude variable
        lat: float
            The target latitude
        lon: float
            The target longitude
        Returns
        -------
        List
            A list of ints, length 2 representing the y,x coordinates in
            the wrf file.
        """
        rad_factor = pi/180.0  # cos/sin require angles in rads
        lats = lat_ar[:] * rad_factor
        lons = lon_ar[:] * rad_factor
        lat_rad = lat * rad_factor
        lon_rad = lon * rad_factor

        clat, clon = cos(lats), cos(lons)
        slat, slon = sin(lats), sin(lons)
        delx = cos(lat_rad)*cos(lon_rad) - clat*clon
        dely = cos(lat_rad)*sin(lon_rad) - clat*slon
        delz = sin(lat_rad) - slat
        dist_sq = delx**2 + dely**2 + delz**2
        idx = dist_sq.argmin()  # 1d index of the minimum value
        y, x = np.unravel_index(idx, lats.shape)
        return [y, x]

    def _gather_location_data(self):
        """Gathers information about each wind station
        in the database and adds it to location_dict.
        """
        # Mysql setup
        mysql_login = mylogin.get_login_info("selectonly")
        mysql_login['database'] = 'utility_data'
        cnx = mysql.connector.connect(**mysql_login)
        query = 'SELECT name,lat,lon,elevation FROM \
                measurementDescriptions WHERE type = "Wind"'
        cursor = cnx.cursor()
        cursor.execute(query)

        # Get the lat/lons for each wind stations
        self.location_dict = {}
        for row in cursor.fetchall():
            if row[0] == 'Total Wind':
                continue
            self.location_dict[row[0]] = {'lat': float(row[1]),
                                          'lon': float(row[2]),
                                          'elevation': float(row[3])}
        cursor.close()

    def _create_terminal_points(self):
        """
        Sets up the terminal points for cross sections in the
        location_dict attribute. Should only be called by the
        constructor during instantiation.

        """
        spread = self.spread
        for station in self.location_dict.values():
                lat = station['lat']
                lon = station['lon']
                station['origin'] = self._tunnel_dist(self.lats,
                                                      self.lons,
                                                      lat, lon)
                station["v1"] = [station['origin'][0]-spread,
                                 station['origin'][1]]
                station["v2"] = [station['origin'][0]+spread,
                                 station['origin'][1]]
                station["h1"] = [station['origin'][0],
                                 station['origin'][1]-spread]
                station["h2"] = [station['origin'][0],
                                 station['origin'][1]+spread]

    def __init__(self, filename):
        try:
            self.wrf_data = Dataset(filename)
        except:
            logging.exception("WRF file does not exist.")
            sys.exit(1)

        self.lats = self.wrf_data.variables['XLAT']
        self.lons = self.wrf_data.variables['XLONG']

        self.source_data = pd.DataFrame()
        self._gather_location_data()
        self._create_terminal_points()
        self._station = self.location_dict.keys()[0]
        self._orientation = 'vertical'
        self.times = set()
        self.init = datetime.datetime.strptime(
                            self.wrf_data.START_DATE,
                            '%Y-%m-%d_%X'
                            )

    @property
    def station(self):
        """Getter for the station property"""
        return self._station

    @station.setter
    def station(self, new_station):
        """Setter for station property. If the value has changed
        then reset times and source_data.
        """
        if new_station == self._station:
            return None  # do nothing

        self._station = new_station
        self.source_data = pd.DataFrame()
        self.times = set()

    @property
    def orientation(self):
        """Getter for orientation"""
        return self._orientation

    @orientation.setter
    def orientation(self, new_orientation):
        """Setter for orientation, if value changed, reset
        times and source_data
        """
        if new_orientation == self._orientation:
            return None  # do nothing
        self._orientation = new_orientation
        self.source_data = pd.DataFrame()
        self.times = set()

    def list_stations(self):
        return self.location_dict.keys()

    def update_source(self, time):
        """Rebuilds source_data for the current station and
        orientation for the time interval.
        """

        if time in self.times:
            return
        else:
            self.times.add(time)
        station = self.location_dict[self.station]
        origin = station['origin']
        time_format = '%y/%m/%d %H:%M:%SZ'

        if self._orientation == 'vertical':
            y_range = range(station['v1'][0], station['v2'][0])
            x_range = origin[1]

            latlons = ['%0.2f,%0.2f' %
                       (self.lats[y, origin[1]],
                        self.lons[y, origin[1]])
                       for y in y_range]

        else:
            y_range = origin[0]
            x_range = range(station['h1'][1], station['h2'][1])

            latlons = ['%0.2f,%0.2f' %
                       (self.lats[origin[0], x],
                        self.lons[origin[0], x])
                       for x in x_range]

        pr = (self.wrf_data['PB'][time, :, y_range, x_range] +
              self.wrf_data['P'][time, :, y_range, x_range])

        wx = self.wrf_data['U'][time, :, y_range, x_range]

        wy = self.wrf_data['V'][time, :, y_range, x_range]

        height = pressure_to_height_std(pr * units.pascal)
        height = np.array(height)
        height = height * 1000  # km to m
        wspd = np.sqrt(wx**2 + wy**2)
        # calculate the height at each index
        heights = np.diff(height, axis=0)
        # Append heights for last index, this just reuses
        # the second to last value
        last_value = np.reshape(heights[-1, :], (1, self.spread*2))
        heights = np.vstack((heights, last_value))

        # build the source dataframe
        ys = height+(heights/2)
        i = str(time)
        self.source_data['y'+i] = ys.ravel()
        self.source_data['h'+i] = heights.ravel()
        self.source_data['w'+i] = wspd.ravel()
        valid_time = self.init+datetime.timedelta(hours=time)
        self.source_data['t'+i] = (valid_time).strftime(time_format)
        if 'll' not in self.source_data.index:
            self.source_data['ll'] = latlons * self.wrf_data['P'].shape[1]
