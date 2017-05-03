from netCDF4 import Dataset
from numpy import cos, sin
from metpy.calc import pressure_to_height_std as pth
from metpy.units import units
from datetime import timedelta
from bokeh.plotting import (figure, output_file)
from bokeh.palettes import Viridis256
from math import pi
from dateutil.parser import parse
from bokeh.io import show
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter as ptf,
    ColorBar,
    CustomJS,
    Slider
)
from bokeh.layouts import column, widgetbox
from serverscripts.utils import (handle_exception,
                                 basic_logging_config,
                                 log_to_file
                                 )
import argparse
import logging
import sys
import os
import datetime
import numpy as np
import pandas as pd
import mysql.connector
import mylogin

# the directory to store figures in
FIG_DIR = "wind_cross_sections"
DATA_DIR = "/a4"


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


def get_weather_file(time, model):
    """ Returns the location of current weather data as a string, None
        if the file does not exist
    parameters
    ----------
    time: datetime object
        The desired time for the model run
    model: string
        The model to search for i.e. 'GFS'
    """
    location = '%s/uaren/%s/WRF%s_%sZ/wrfsolar_d02_hourly.nc'
    location = location % (DATA_DIR, time.strftime("%Y/%m/%d"),
                           model, time.strftime("%H"))
    if not(os.path.isfile(location)):
        return None
    else:
        return location


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


def createTerminalPoints(loc_dict, lats, lons, spread):
    for station in loc_dict:
            lat = loc_dict[station]['lat']
            lon = loc_dict[station]['lon']
            loc_dict[station]['origin'] = tunnel_dist(lats, lons, lat, lon)
            loc_dict[station]["v1"] = [loc_dict[station]['origin'][0]-spread,
                                       loc_dict[station]['origin'][1]]
            loc_dict[station]["v2"] = [loc_dict[station]['origin'][0]+spread,
                                       loc_dict[station]['origin'][1]]
            loc_dict[station]["h1"] = [loc_dict[station]['origin'][0],
                                       loc_dict[station]['origin'][1]-spread]
            loc_dict[station]["h2"] = [loc_dict[station]['origin'][0],
                                       loc_dict[station]['origin'][1]+spread]
    return loc_dict


def generatePlots(time, model, s=None):
    """Creates bokeh cross-sections at all timesteps found in the
    weather data, for both South-North and West-East orientations.

    parameters
    ----------
    time: Datetime
        The init time of the wrf data to use.
    model: string
        The model to use.
    s: string
        The name of the station to produce. Used when you would
        like only one plot.
    """
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

    # If the user passed in a station, check that it exists
    if s is not None:
        if s in loc_dict:
            loc_dict = {s:loc_dict[s]}
            logging.debug('Using station %s.\n %s' % (s,loc_dict))
        else:
            logging.error('Given station does not exist.')
            sys.exit(1)

    filename = get_weather_file(time, model)
    if filename is not None:
        logging.info('Using file: %s' % filename)
    else:
        logging.error("WRF file does not exist.")
        sys.exit(1)
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
    loc_dict = createTerminalPoints(loc_dict, lats, lons, SPREAD)

    # loop through the locations and generate plots
    for location in loc_dict:
        figure_path = os.path.join(FIG_DIR, location)
        station = loc_dict[location]
        logging.info('Creating plots for %s.' % location)

        if not os.path.isdir(figure_path):
            os.mkdir(figure_path)

        # Repeat plotting code for v(South-North), and h(West-East)
        for o in ['v', 'h']:
            figure_path = os.path.join(FIG_DIR, location, o)

            if not os.path.isdir(figure_path):
                os.mkdir(figure_path)

            ll_set = False
            plot_title = location

            if o == 'v':
                plot_title = plot_title+' South-North'
            else:
                plot_title = plot_title+' West-East'

            # Build the dataframe
            dframe = pd.DataFrame()
            # Gather data for each time index
            origin = station['origin']
            tf = '%y/%m/%d %H:%M:%SZ'  # time format
            for t in range(0, wrf_data['PB'].shape[0]):
                if o == 'v':
                    v1 = station['v1']
                    v2 = station['v2']
                    pr = (wrf_data['PB'][t, :,
                                         v1[0]:v2[0],
                                         origin[1]] +
                          wrf_data['P'][t, :,
                                        v1[0]:v2[0],
                                        origin[1]])

                    wx = wrf_data['U'][t, :,
                                       v1[0]:v2[0],
                                       origin[1]]

                    wy = wrf_data['V'][t, :,
                                       v1[0]:v2[0],
                                       origin[1]]
                    latlons = ['%0.2f,%0.2f' %
                               (lats[y, origin[1]],
                                lons[y, origin[1]])
                               for y in range(v1[0],
                                              v2[0])]
                else:
                    h1 = station['h1']
                    h2 = station['h2']
                    pr = (wrf_data['PB'][t, :, origin[0], h1[1]:h2[1]] +
                          wrf_data['P'][t, :, origin[0], h1[1]:h2[1]])
                    wx = wrf_data['U'][t, :, origin[0], h1[1]:h2[1]]
                    wy = wrf_data['V'][t, :, origin[0], h1[1]:h2[1]]
                    latlons = ['%0.2f,%0.2f' % (lats[origin[0], x],
                                                lons[origin[0], x])
                               for x in range(h1[1], h2[1])]

                a = pth(pr * units.pascal)  # convert pressure to height
                a = np.array(a)
                a = a * 1000  # km to m
                wspd = np.sqrt(wx**2 + wy**2)
                heights = []
                # calculate the height at each index
                for x in range(0, len(a)-1):
                    heights.extend(a[x+1]-a[x])

                # Append heights for last index, this just reuses
                # the second to last value
                heights.extend([heights[-1]]*20)

                # build the source dataframe
                ys = a.ravel()
                ys = [ys[i]+heights[i]/2 for i in range(0, len(ys))]
                i = str(t)
                dframe['y'+i] = ys
                dframe['h'+i] = heights
                dframe['w'+i] = wspd.ravel()
                dframe['t'+i] = (init+timedelta(hours=t)).strftime(tf)
                if ll_set is not True:
                    dframe['ll'] = latlons * wrf_data['P'].shape[1]
                    ll_set = True
                logging.debug(t)

            # Setup bokeh plot
            init_title = (plot_title+'  Initialized: ' +
                          init.strftime(tf))
            init_title = (init_title+'    Valid: ' +
                          (init+timedelta(hours=t)).strftime(tf))
            mapper = LinearColorMapper(palette=Viridis256, low=0, high=40)
            TOOLS = "pan,wheel_zoom,reset,hover,save"
            source = ColumnDataSource(dframe)
            # Initialize bokeh figure
            pf = figure(title=init_title,
                        x_range=latlons[0:(2*SPREAD)+1],
                        plot_width=1000, plot_height=600,
                        x_axis_label="Latitude, Longitude",
                        y_axis_label="Altitude(m)",
                        tools=TOOLS, toolbar_location='left',
                        toolbar_sticky=False)
            # Plot rectangles representing each point of data
            rects = pf.rect(x='ll', y='y0',
                            width=1, height='h0',
                            fill_color={'field': 'w0', 'transform': mapper},
                            source=source,
                            line_color=None)
            # Plot the position  of the station as a red x
            pf.cross(x=[latlons[10]], y=[station['elevation']],
                     size=10, color="#FF0000", legend=location)
            color_bar = ColorBar(color_mapper=mapper,
                                 major_label_text_font_size="8pt",
                                 ticker=BasicTicker(desired_num_ticks=10),
                                 formatter=ptf(format="%d m/s"),
                                 label_standoff=10,
                                 border_line_color=None,
                                 location=(0, 0))
            pf.xaxis.major_label_orientation = pi / 4
            pf.add_layout(color_bar, 'right')
            pf.select_one(HoverTool).tooltips = [
                 ('position', '@ll'),
                 ('wspd', '@w0'),
            ]

            cbcallback = CustomJS(args=dict(),
                                  code="""
                                       mapper.high = slider.value;
                                       """)
            cbslider = Slider(start=10, end=100,
                              value=40, step=10,
                              title="Colorbar Max",
                              callback=cbcallback)
            cbcallback.args['mapper'] = mapper
            cbcallback.args['slider'] = cbslider

            ht = pf.select_one(HoverTool)
            # The Javascript callback
            tcallback = CustomJS(args=dict(ht=ht, rects=rects,
                                           source=source, plot=pf),
                                 code="""
                var data = source.get("data");
                rects.glyph.height.field = "h"+slider.value;
                rects.glyph.y.field = "y"+slider.value;
                rects.glyph.fill_color.field = "w"+slider.value;
                ht.tooltips[1][1] = "@w"+slider.value;
                var title = plot.attributes.title.attributes
                .text.slice(0,-18);
                plot.attributes.title
                .attributes.text = title+data["t"+slider.value][0];
                source.trigger("change");
                """)
            tslider = Slider(start=0, end=wrf_data['PB'].shape[0]-1,
                             value=0, step=1, title="timestep",
                             callback=tcallback)
            tcallback.args['slider'] = tslider
            figure_name = os.path.join(FIG_DIR, location,
                                       o, init.strftime('%y%m%d%H%M%S'))
            output_file(figure_name)
            layout = column(
                pf,
                widgetbox(cbslider),
                widgetbox(tslider)
            )
            show(layout)


def main():
    """Parses arguments from the command line and calls
    generatePlots(). Woo.
    """
    # define valid models
    MODELS = ['GFS', 'NAM']
    sys.excepthook = handle_exception
    basic_logging_config()
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create Hourly Bokeh Plots.')
    argparser.add_argument('time',
                           help="Init time of the model to use.")
    argparser.add_argument('model',
                           help="The model to use: GFS or NAM")
    argparser.add_argument('-v', '--verbose',
                           help="Increase logging verbosity.",
                           action="count")
    argparser.add_argument('-s', '--station',
                           help="The Name of a station to plot. For\
                                 plotting a single station.")
    argparser.add_argument('--log-file',
                           help='Path to the file to direct logging < ERROR')
    args = argparser.parse_args()
    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose and args.verbose > 1:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.log_file is not None:
        log_to_file(args.log_file)

    try:
        wrf_time = parse(args.time)
    except:
        logging.exception("Incorrect time format.")
        sys.exit(1)
    if args.model not in MODELS:
        logging.error("Invalid Model")
        sys.exit(1)
    generatePlots(wrf_time, args.model, args.station)


if __name__ == "__main__":
    main()
