from math import pi
from netCDF4 import Dataset
from numpy import cos, sin
from metpy.calc import pressure_to_height_std
from metpy.units import units
from datetime import timedelta
from bokeh.plotting import (figure, output_file)
from bokeh.palettes import Viridis256
from dateutil.parser import parse
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
from serverscripts.utils import (
    handle_exception,
    basic_logging_config,
    log_to_file
)
import sys
import os
import argparse
import logging
import datetime
import numpy as np
import pandas as pd
import mysql.connector
import mylogin

# the directory to store figures in
FIG_DIR = "wind_cross_sections"

# the directory in which to find the data files
DATA_DIR = "/a4"

# The spread over which to create the cross section in wrf grid points
# ~1.8km each
SPREAD = 10


def get_weather_file(time, model):
    """
    Returns the location of current weather data as a string, None
    if the file does not exist.

    Parameters
    ----------
    time: datetime object
         The desired time for the model run.
    model: string
         The model to search for i.e. 'GFS'.

    Returns
    -------
    string
       The path to the desired wrf data file or None.
    """
    location = '%s/uaren/%s/WRF%s_%sZ/wrfsolar_d02_hourly.nc'
    location = location % (DATA_DIR, time.strftime("%Y/%m/%d"),
                           model, time.strftime("%H"))
    return location


def tunnel_dist(lat_ar, lon_ar, lat, lon):
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


def create_terminal_points(loc_dict, lats, lons, spread):
    """
    Takes a dictionary of locations with latitude and longitude and finds
    the origin, and then terminal points for South-North and West-East
    cross sections.

    Parameters
    ----------
    loc_dict: dictionary
        A dictionary where each key is a location with 'lat' and 'lon'.
    lats: array_like
        The latitudes from the wrf file.
    lons: array_like
        The longitudes from the wrf file.
    spread:
        The number of grid points the terminal points should from the origin
        in either direction.

    Returns
    -------
    Dictionary
        Updated location dictionary with terminal points and origin.
    """
    for station in loc_dict.values():
            lat = station['lat']
            lon = station['lon']
            station['origin'] = tunnel_dist(lats, lons, lat, lon)
            station["v1"] = [station['origin'][0]-spread,
                             station['origin'][1]]
            station["v2"] = [station['origin'][0]+spread,
                             station['origin'][1]]
            station["h1"] = [station['origin'][0],
                             station['origin'][1]-spread]
            station["h2"] = [station['origin'][0],
                             station['origin'][1]+spread]
    return loc_dict


def generate_plots(time, model, facility=None):
    """
    Creates bokeh cross-sections at all timesteps found in the
    weather data, for both South-North and West-East orientations.

    Parameters
    ----------
    time: Datetime
        The init time of the wrf data to use.
    model: string
        The model to use.
    facility: string
        The name of the station to produce. Used when you would
        like only one plot. Defaults to None
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
    loc_dict = {}
    for row in cursor.fetchall():
        if row[0] == 'Total Wind':
            continue
        loc_dict[row[0]] = {'lat': float(row[1]),
                            'lon': float(row[2]),
                            'elevation': float(row[3])}
    cursor.close()

    # If the user passed in a station, check that it exists
    if facility is not None:
        if facility in loc_dict:
            loc_dict = {facility: loc_dict[facility]}
            logging.debug('Using station %s.\n %s' % (facility, loc_dict))
        else:
            logging.error('Given station does not exist.')
            sys.exit(1)

    filename = get_weather_file(time, model)

    try:
        wrf_data = Dataset(filename)
    except:
        logging.exception("WRF file does not exist.")
        sys.exit(1)

    # create the figure directory
    if not os.path.isdir(FIG_DIR):
        os.mkdir(FIG_DIR)

    # get the init time of the file
    init = datetime.datetime.strptime(wrf_data.START_DATE, '%Y-%m-%d_%X')

    lats = wrf_data.variables['XLAT']
    lons = wrf_data.variables['XLONG']

    # Add start and end points for both north-south and east-west cross section
    loc_dict = create_terminal_points(loc_dict, lats, lons, SPREAD)

    # loop through the locations and generate plots
    for location in loc_dict:
        figure_path = os.path.join(FIG_DIR, location)
        station = loc_dict[location]
        logging.info('Creating plots for %s.' % location)

        if not os.path.isdir(figure_path):
            os.mkdir(figure_path)

        # Repeat plotting code for v(South-North), and h(West-East)
        for orientation in ['vertical', 'horizontal']:
            figure_path = os.path.join(FIG_DIR, location, orientation)

            if not os.path.isdir(figure_path):
                os.mkdir(figure_path)

            plot_title = location

            if orientation == 'vertical':
                plot_title = plot_title+' South-North'
            else:
                plot_title = plot_title+' West-East'

            # Build the dataframe
            dframe = pd.DataFrame()
            # Gather data for each time index
            origin = station['origin']
            time_format = '%y/%m/%d %H:%M:%SZ'
            for time in range(0, wrf_data['PB'].shape[0]):
                if orientation == 'vertical':
                    y_range = range(station['v1'][0], station['v2'][0])
                    x_range = origin[1]

                    latlons = ['%0.2f,%0.2f' %
                               (lats[y, origin[1]],
                                lons[y, origin[1]])
                               for y in y_range]
                else:
                    y_range = origin[0]
                    x_range = range(station['h1'][1], station['h2'][1])

                    latlons = ['%0.2f,%0.2f' %
                               (lats[origin[0], x],
                                lons[origin[0], x])
                               for x in x_range]
                pr = (wrf_data['PB'][time, :, y_range, x_range] +
                      wrf_data['P'][time, :, y_range, x_range])

                wx = wrf_data['U'][time, :, y_range, x_range]

                wy = wrf_data['V'][time, :, y_range, x_range]

                height = pressure_to_height_std(pr * units.pascal)
                height = np.array(height)
                height = height * 1000  # km to m
                wspd = np.sqrt(wx**2 + wy**2)
                # calculate the height at each index
                heights = np.diff(height, axis=0)
                # Append heights for last index, this just reuses
                # the second to last value
                last_value = np.reshape(heights[-1, :], (1, SPREAD*2))
                heights = np.vstack((heights, last_value))

                # build the source dataframe
                ys = height+(heights/2)
                i = str(time)
                dframe['y'+i] = ys.ravel()
                dframe['h'+i] = heights.ravel()
                dframe['w'+i] = wspd.ravel()
                valid_time = init+timedelta(hours=time)
                dframe['t'+i] = (valid_time).strftime(time_format)
                logging.debug(time)

            dframe['ll'] = latlons * wrf_data['P'].shape[1]
            # Setup bokeh plot
            init_title = ('%s  Initialized: %s   Valid: %s' % (
                          plot_title,
                          init.strftime(time_format),
                          (init+timedelta(hours=time)).strftime(time_format)))
            mapper = LinearColorMapper(palette=Viridis256, low=0, high=40)
            tools = "pan,wheel_zoom,reset,hover,save"
            source = ColumnDataSource(dframe)
            # Initialize bokeh figure
            pf = figure(title=init_title,
                        x_range=latlons[0:(2*SPREAD)+1],
                        plot_width=1000, plot_height=600,
                        x_axis_label="Latitude, Longitude",
                        y_axis_label="Altitude(m)",
                        tools=tools, toolbar_location='left',
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
            bokeh_formatter = PrintfTickFormatter(format="%d m/s")
            color_bar = ColorBar(color_mapper=mapper,
                                 major_label_text_font_size="8pt",
                                 ticker=BasicTicker(desired_num_ticks=10),
                                 formatter=bokeh_formatter,
                                 label_standoff=10,
                                 border_line_color=None,
                                 location=(0, 0))
            pf.xaxis.major_label_orientation = pi / 4
            pf.add_layout(color_bar, 'right')
            pf.select_one(HoverTool).tooltips = [
                 ('position', '@ll'),
                 ('wspd', '@w0'),
                 ('altitude', "@y0{int}"),
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
                ht.tooltips[1][2] = "@y"+slider.value;
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
                                       orientation,
                                       init.strftime('%y%m%d%H%M%S'))
            output_file(figure_name+'.html')
            layout = column(
                pf,
                widgetbox(cbslider),
                widgetbox(tslider)
            )
            show(layout)


def main():
    """Parses arguments from the command line and calls
    generate_plots(). Woo.
    """
    # define valid models
    models = ('GFS', 'NAM')

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
    if args.model not in models:
        logging.error("Invalid Model")
        sys.exit(1)
    generate_plots(wrf_time, args.model, args.station)


if __name__ == "__main__":
    main()
