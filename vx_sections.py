from netCDF4 import Dataset
from math import pi
from numpy import cos, sin
from wrf import to_np, getvar, CoordPair, vertcross
from wrf.destag import destagger
from metpy.calc import pressure_to_height_std as pth
from metpy.units import units
from datetime import timedelta
import matplotlib
import os
import math
import datetime
import numpy as np
import mysql.connector
import mylogin
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# the directory to store figures in
FIG_DIR = "wind_cross_sections"


def get_altitude_index(a, alt):
    """Returns the index of the desired altitude in a.
    parameters
    ----------
    a: array-like object
        The sequence of altitudes to search through
    alt: float
        The desired altitude to look for

    """
    i = 0
    for v in a:
        if v >= alt:
            break
        i += 1
    return i


def get_weather_file():
    """ Returns the location of current weather data as a string
    """
    location = '/a4/uaren/%s/WRFGFS_06Z/wrfsolar_d02_hourly.nc'
    now = datetime.datetime.now()
    return location % (now.strftime("%Y/%m/%d"))


def tunnel_dist(lat_ar, lon_ar, lat, lon):
    """ tunnel distance: finds the index of the closest point assuming
        a spherical earth
        parameters
        ----------
        lat_ar: numpy array
            the array of the corresponding netCDF's latitude variable
        lon_ar: numpy array
            the arrray of the corresponding netCDF's longitude variable
        lat: float
            the target latitude
        lon: float
            the target longitude
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


def get_wspd(u, v):
    """Returns total windspeed.
    """
    time = datetime.datetime.now()
    wspd = np.sqrt(u[:]**2+v[:]**2)
    print(datetime.datetime.now()-time)
    return wspd


def main():
    # Mysql setup
    mysql_login = mylogin.get_login_info("selectonly")
    mysql_login['database'] = 'utility_data'
    cnx = mysql.connector.connect(**mysql_login)
    query = 'SELECT name,lat,lon,elevation FROM \
            measurementDescriptions WHERE type = "Wind"'
    cursor = cnx.cursor()
    cursor.execute(query)

    # Get all the lat/lons of wind stations
    loc_dict = {}
    for row in cursor.fetchall():
        if row[0] == 'Total Wind':
            continue
        loc_dict[row[0]] = {'lat': float(row[1]),
                            'lon': float(row[2]),
                            'elevation': float(row[3])}
    cursor.close()
    filename = get_weather_file()
    wrf_data = Dataset(filename)

    # create the figure directory
    if not os.path.isdir(FIG_DIR):
        os.mkdir(FIG_DIR)
    # get the init time of the file
    init = datetime.datetime.strptime(wrf_data.START_DATE, '%Y-%m-%d_%X')

    lats = wrf_data.variables['XLAT']
    lons = wrf_data.variables['XLONG']

    # The spread over which to create the cross section in degrees
    SPREAD = .5
    # Add start and end points for both north-south and east-west cross section
    for station in loc_dict:
        lat = loc_dict[station]['lat']
        lon = loc_dict[station]['lon']
        loc_dict[station]['origin'] = tunnel_dist(lats, lons, lat, lon)
        loc_dict[station]["v1"] = tunnel_dist(lats, lons, lat-SPREAD, lon)
        loc_dict[station]["v2"] = tunnel_dist(lats, lons, lat+SPREAD, lon)
        loc_dict[station]["h1"] = tunnel_dist(lats, lons, lat, lon-SPREAD)
        loc_dict[station]["h2"] = tunnel_dist(lats, lons, lat, lon+SPREAD)
    # DEBUG STUFF
    print("---Location Dict---\n")
    print(loc_dict)
    print('\n' + filename)
    # get z-variable(pressure), convert it to altitude
    time = datetime.datetime.now()
    print('getting z')
    z = getvar(wrf_data, "p", units="mb", meta=False, timeidx=None)
    a = pth(z * units.mbar)
    a = np.array(a)
    print('got z')
    print(datetime.datetime.now()-time)
    time = datetime.datetime.now()
    # get Windspeed
    # wspd = getvar(wrf_data, "wspd_wdir", units="kt", meta=False)[0,:]
    print('getting u')
    u = destagger(wrf_data['U'][:], -1)
    print('got u')
    print(datetime.datetime.now()-time)
    time = datetime.datetime.now()
    print('getting v')
    v = destagger(wrf_data['V'][:], -2)
    print('got v')
    print(datetime.datetime.now()-time)
    time = datetime.datetime.now()
    print('getting wspdv')
    wspd = get_wspd(u, v)
    print('got v')
    print(datetime.datetime.now()-time)
    time = datetime.datetime.now()
    print('Data collected. Creating plots')
    # loop through locations and create cross sections
    for location in loc_dict:
        figure_path = '%s/%s' % (FIG_DIR, station)
        station = loc_dict[location]
        if not os.path.isdir(figure_path):
            os.mkdir(figure_path)
        for closeup in(False, True):
            if(closeup):
                figure_path = figure_path + '/2k'
                if not os.path.isdir(figure_path):
                    os.mkdir(figure_path)
            for orientation in ('v', 'h'):
                for i in range(0, wspd.shape[0]):
                    if not os.path.isdir("%s/%s" % (figure_path, orientation)):
                        os.mkdir("%s/%s" % (figure_path, orientation))
                    # get the index of the station's altitude as well as the index of 2km above.
                    station_altitude_idx = station['elevation']/(a[i, -1, station['v1'][0], station['v1'][1]]*10)

                    start = CoordPair(x=station[orientation+'1'][1], y=station[orientation+'1'][0])
                    end = CoordPair(x=station[orientation+'2'][1], y=station[orientation+'2'][0])
                    wspd_cross = vertcross(wspd[i], a[i], wrfin=wrf_data, cache=None,
                                           start_point=start, end_point=end, latlon=False, meta=True)
                    # generate labels for the x axis - not entirely accurate due to the interpolation
                    # done by vertcross()
                    x_axis = wspd_cross.dim_1
                    if(orientation == 'v'):
                        xlabel_idxs = [((start.x+end.x)/2, y) for y in x_axis.values+station[orientation+'1'][0]]
                        # find the minimum altitude in the range, and find it's index on the y axis
                        min_alt_y = a[i, 0, xlabel_idxs[0][1]:xlabel_idxs[-1][1], (start.x+end.x)/2].argmin()
                        lower_lim = (a[i, 0, xlabel_idxs[0][1]+min_alt_y, (start.x+end.x)/2]
                                     / a[i, -1, station['v1'][0], station['v1'][1]])*100
                    else:
                        xlabel_idxs = [(x, (start.y+end.y)/2) for x in x_axis.values+loc_dict[station][orientation+'1'][1]]
                        min_alt_x = a[i, 0, (start.x+end.x)/2, xlabel_idxs[0][0]:xlabel_idxs[-1][0]].argmin()
                        lower_lim = (a[i, 0, (start.x+end.x)/2, xlabel_idxs[0][1]+min_alt_x]
                                     / a[i, -1, station['v1'][0], station['v1'][1]])*100
                    latlons = ['%0.2f,\n%0.2f' % (lats[y, x], lons[y, x]) for x, y in xlabel_idxs]

                    # Create the figure
                    fig = plt.figure(figsize=(12, 6))
                    ax = plt.axes()

                    max_wspd = 60
                    if closeup:
                        max_wspd = 25
                        y_limit = (station['elevation']+2000)/(a[-1, station['origin'][0], station['origin'][1]]*10)
                        plt.ylim(lower_lim, y_limit)
                    # Make the contour plot
                    wspd_contours = ax.contourf(to_np(wspd_cross), levels=np.linspace(0, max_wspd), extend='max')
                    wspd_contours.set_clim(0, max_wspd)

                    # Add the color bar
                    bar = plt.colorbar(wspd_contours, ax=ax)
                    bar.set_label("Windspeed (kt)")
                    # Set y axis labels
                    ylabel_idx = ax.get_yticks()/100
                    ylabels = ylabel_idx*a[i, -1, station['v1'][0], station['v1'][1]]*1000
                    ax.set_yticklabels(ylabels.astype(int))
                    # Set x axis labels
                    xlabel_idx = ax.get_xticks()[:-1].astype(int).tolist()
                    xlabels = [latlons[x] for x in xlabel_idx]
                    ax.set_xticklabels(xlabels, rotation=0)

                    ax.set_xlabel("Latitude, Longitude", fontsize=12)
                    ax.set_ylabel("Altitude (m)", fontsize=12)
                    orientation_tag = 'North-South'

                    if(orientation == 'h'):
                        orientation_tag = 'West-East'
                    if closeup:
                        orientation_tag += '-2km'
                    plt.title("Vertical Cross Section of Wind Speed (kt) \n %s(%0.2f,%0.2f) %s"
                              % (location, station['lat'], station['lon'], orientation_tag))
                    plt.tight_layout()
                    if(orientation == 'h'):
                        plt.plot(math.fabs(station['origin'][1]-start.x),
                                 station_altitude_idx,
                                 marker='x', markersize=6, color="red", label=station)
                    else:
                        plt.plot(math.fabs(station['origin'][0]-start.y),
                                 station_altitude_idx,
                                 marker='x', markersize=6, color="red", label=station)
                    plt.legend(loc=3)
                    plt.text(.65, -.15, 'Initialized:' + init.strftime('%y/%m/%d %H:%M:%S') +
                             ' Valid:' + (init+timedelta(hours=i)).strftime('%y/%m/%d %H:%M:%S'),
                             transform=ax.transAxes, backgroundcolor='w')
                    figname = '%s/%s/%s.png' % (figure_path, orientation, (init+timedelta(hours=i)).strftime('%y%m%d%H%M%S'))
                    plt.savefig(figname)
                    plt.close(fig)
        break
    print('all plots created')
    print(datetime.datetime.now()-time)


if __name__ == "__main__":
    main()
