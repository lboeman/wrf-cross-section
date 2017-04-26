from netCDF4 import Dataset
from math import pi
from numpy import cos, sin
from metpy.calc import pressure_to_height_std as pth
from metpy.units import units
from datetime import timedelta
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Viridis256
from math import pi

from bokeh.io import show
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
    CustomJS,
    Slider
)
from bokeh.layouts import column, widgetbox
import os
import math
import datetime
import numpy as np
import pandas as pd
import mysql.connector
import mylogin
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

    # The spread over which to create the cross section in wrf grid points
    # ~1.8km each
    SPREAD = 10

    # Add start and end points for both north-south and east-west cross section
    for station in loc_dict:
        lat = loc_dict[station]['lat']
        lon = loc_dict[station]['lon']
        loc_dict[station]['origin'] = tunnel_dist(lats, lons, lat, lon)
        loc_dict[station]["v1"] = [loc_dict[station]['origin'][0]-SPREAD,loc_dict[station]['origin'][1]]
        loc_dict[station]["v2"] = [loc_dict[station]['origin'][0]+SPREAD,loc_dict[station]['origin'][1]]
        loc_dict[station]["h1"] = [loc_dict[station]['origin'][0],loc_dict[station]['origin'][1]-SPREAD]
        loc_dict[station]["h2"] = [loc_dict[station]['origin'][0],loc_dict[station]['origin'][1]+SPREAD]
    # DEBUG STUFF
    print "---Location Dict---\n"
    print loc_dict
    print '\n' + filename
    for location in loc_dict:
        figure_path = '%s/%s' % (FIG_DIR, location)
        station = loc_dict[location]
        if not os.path.isdir(figure_path):
            os.mkdir(figure_path)
        for o in ['v','h']:
            dframe = pd.DataFrame()
            ll_set = False
            plot_title = location
            if o=='v':
                plot_title = plot_title+' South-North'
            else:
                plot_title = plot_title+' West-East'
            for t in range(0,wrf_data['PB'].shape[0]):
                if o=='v':
                    pr = wrf_data['PB'][t,:,station['v1'][0]:station['v2'][0],station['origin'][1]]+wrf_data['P'][t,:,station['v1'][0]:station['v2'][0],station['origin'][1]]
                    wx = wrf_data['U'][t,:,station['v1'][0]:station['v2'][0],station['origin'][1]]
                    wy = wrf_data['V'][t,:,station['v1'][0]:station['v2'][0],station['origin'][1]]
                    latlons = ['%0.2f,%0.2f' % (lats[y, station['origin'][1]],
                                                lons[y, station['origin'][1]])
                                                for y in range(station['v1'][0],station['v2'][0])]
                else:
                    pr = wrf_data['PB'][t,:,station['origin'][0],station['h1'][1]:station['h2'][1]]+wrf_data['P'][t,:,station['origin'][0],station['h1'][1]:station['h2'][1]]
                    wx = wrf_data['U'][t,:,station['origin'][0],station['h1'][1]:station['h2'][1]]
                    wy = wrf_data['V'][t,:,station['origin'][0],station['h1'][1]:station['h2'][1]]
                    latlons = ['%0.2f,%0.2f' % (lats[station['origin'][0],x],
                                                lons[station['origin'][0],x])
                                                for x in range(station['h1'][1],station['h2'][1])]

                a = pth(pr * units.pascal) # convert pressure to height
                a = np.array(a)
                a = a * 1000 # km to m
                wspd = np.sqrt(wx**2+wy**2)
                heights=[]
                # calculate the height at each index
                for x in range(0,len(a)-1):
                    heights.extend(a[x+1]-a[x]) 
                heights.extend([heights[-1]]*20)

                # build the source dataframe
                ys = a.ravel()
                # center the y values
                ys = [ys[i]+heights[i]/2 for i in range(0,len(ys))]
                i = str(t)
                dframe['y'+i] = ys
                dframe['h'+i] = heights
                dframe['w'+i] = wspd.ravel()
                dframe['t'+i] = (init+timedelta(hours=t)).strftime('%y/%m/%d %H:%M:%S')
                dframe['ll'] = latlons * wrf_data['P'].shape[1]
                print t
            # Setup bokeh plot
            init_title = plot_title+' \n Initialized:'+init.strftime('%y/%m/%d %H:%M:%S')
            init_title = init_title+' Valid:'+(init+timedelta(hours=t)).strftime('%y/%m/%d %H:%M:%S')
            mapper = LinearColorMapper(palette=Viridis256, low=0, high=40)
            TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save"
            source = ColumnDataSource(dframe)
            pf = figure(title=init_title,
                       x_range=latlons[0:(2*SPREAD)+1],
                       plot_width=900, plot_height=500,
                       x_axis_label="Latitude, Longitude",
                       y_axis_label="Altitude(m)",
                       tools=TOOLS, toolbar_location='above')
            rects=pf.rect(x='ll',y='y0',
                   width=1,height='h0',
                   fill_color={'field': 'w0', 'transform': mapper},
                   source=source,
                   line_color=None
                   )
            pf.cross(x=[latlons[10]],y=[station['elevation']],size=10,color="#FF0000",legend=location)
            color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                                 ticker=BasicTicker(desired_num_ticks=10),
                                 formatter=PrintfTickFormatter(format="%d m/s"),
                                 label_standoff=10, border_line_color=None, location=(0, 0))
            pf.xaxis.major_label_orientation = pi / 4
            pf.add_layout(color_bar, 'right')
            pf.select_one(HoverTool).tooltips = [
                 ('position', '@ll'),
                 ('wspd', '@wind'),
            ]
            cbcallback = CustomJS(args=dict(),code="""
                mapper.high = slider.value;
                """)
            cbslider = Slider(start=10, end=100, value=40, step=10, title="Colorbar Max", callback=cbcallback)
            cbcallback.args['mapper'] = mapper
            cbcallback.args['slider'] = cbslider
            tcallback = CustomJS(args=dict(rects=rects,source=source,mapper=mapper,plot=pf),code="""
                console.log(pf.title);
                rects.glyph.height.field = "h"+slider.value;
                rects.glyph.y.field = "y"+slider.value;
                rects.glyph.fill_color.field = "w"+slider.value;
                var title = plot.title.slice(0,-17);
                plot.title = title+source["t"+slider.value];
                source.trigger("change");
            """)
            tslider = Slider(start=0, end=wrf_data['PB'].shape[0], value=0,step=1,title="timestep",callback=tcallback)
            tcallback.args['slider'] = tslider
            output_file('%s/%s/%s.html' % (figure_path,o,location+(init+timedelta(hours=t)).strftime('%y%m%d%H%M%S')))
            layout = column(
                pf,
                widgetbox(cbslider),
                widgetbox(tslider)
            )
            show(layout)
        break
    print 'all plots created'


if __name__ == "__main__":
    main()
