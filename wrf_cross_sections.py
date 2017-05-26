"""wrf_cross_sections.py

A script which sets up a bokeh figure and interactions for wrf
windspeed cross sections.

Notes
-----
Required environment variables:
MYSQL_CRED: A string to be passed to mylogin.get_login_info()
            defaults to "selectonly"
WRF_DATA_DIRECTORY: the path to the directory in which the wrf
            data can be found. 
"""
import os
import datetime
import numpy as np
import pandas as pd
import pymysql
import mylogin

from math import pi
from datetime import timedelta
from dateutil import parser
from bokeh.plotting import (figure, curdoc)
from bokeh.palettes import Viridis256
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    TextInput,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
    Slider,
    Select,
    RadioGroup,
    Div,
)
from bokeh.layouts import row, widgetbox
from netCDF4 import Dataset
from numpy import cos, sin


# the directory in which to find the data files
DATA_DIR = os.getenv('WRF_DATA_DIRECTORY', '/a4/uaren')

# The spread over which to create the cross section in wrf grid points
# ~1.8km each
SPREAD = 10


def check_date(attr, old, new):
    """Makes sure an appropriately formatted date is input
    """
    try:
        parser.parse(wrf_init_date_input.value)
    except:
        display_message('Unrecognized date format, please use\
                         "yyyy/mm/dd".', 'error')
        return None
    update_datasource(attr, old, new)


def selected_date():
    """builds a datetime object from the current widget
    selections.
    """
    day = parser.parse(wrf_init_date_input.value)
    hour = wrf_init_time_select.value
    hour = timedelta(hours=int(hour[:-1]))
    return day+hour


def pressure_to_height(pressure):
    """Converts pressure in Pa to height
    Totally stollen from metpy.calc.pressure_to_height_std.
    """
    t0 = 288
    gamma = 6.5
    p0 = 101325
    g = 9.806650
    rd = .2870579780696303  # Rd = 8.3144621/28.9644
    return (t0 / gamma) * (1 - (pressure / p0)**(rd * gamma / g))


def tunnel_dist(lat_ar, lon_ar, lat, lon):
    """Finds the index of the closest point assuming a spherical earth.
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


def get_weather_file():
    """
    Returns the location of current weather data as a string, None
    if the file does not exist.

    Returns
    -------
    string
       The path to the desired wrf data file or None.
    """
    wrf_date = selected_date()
    file_date = os.path.join(wrf_date.strftime('%Y'),
                             wrf_date.strftime('%m'),
                             wrf_date.strftime('%d'))
    file_time = wrf_init_time_select.value
    file_model = wrf_model_select.value
    dir_name = f'WRF{file_model}_{file_time}'
    location = os.path.join(DATA_DIR, file_date,
                            dir_name, 'wrfsolar_d02_hourly.nc')
    return location


def open_wrf_file():
    """Returns an netCDF Dataset object

    Returns
    -------
    wrf_file: netCDF4 Dataset
        The opened wrf file.
    """
    try:
        wrf_file = Dataset(get_weather_file())
    except:
        display_message("WRF file does not exist.", 'error')
        return None
    return wrf_file


def colorbar_change(attr, old, new):
    """Callback for cbar_slider, sets the maximum value of
    the colorbar to the slider value
    """
    mapper.high = cbar_slider.value


def update_title(time_index, orientation):
    """ Returns the current name for the plot based on
    station, time and orientation.
    Parameters
    ----------
    time_index: int
        The index of the time dimension from which to
        gather data.
    orientation: string
        "South-North" or "West-East"
    Returns
    -------
    plot_title: string
        Formatted title for the figure based on the
        currently selected widget values and args.
    """
    wrf_init_time = selected_date()
    time_format = '%Y-%m-%d %H:%M:%SZ'
    station_name = station_select.value
    plot_title = '{} ({}) Initialized: {}   Valid: {}'.format(
                  station_name,
                  orientation,
                  wrf_init_time.strftime(time_format),
                  (wrf_init_time +
                   timedelta(hours=time_index)).strftime(time_format))
    return plot_title


def build_dataframe(time_index, station, orientation):
    """Gathers the required variables for plotting from
    the WRF file and organizes them into a dataframe.

    Parameters
    ----------
    time_index: int
        The index of the time dimension from which to
        gather data.
    station: string
        The name of the station to look for. This should
        exist in location_dict.
    orientation: string
        "South-North" or "West-East"
    """
    station_data = location_dict[station]
    wrf_data = open_wrf_file()
    if wrf_data is None:
        return
    new_source = pd.DataFrame()
    lats = wrf_data['XLAT']
    lons = wrf_data['XLONG']
    # If we're opening a file with a shorter time scale,
    # and we're out of bounds, bring the index back in.
    time_index_max = wrf_data['P'].shape[0]-1
    if time_index > time_index_max:
        time_index = time_index_max
        time_slider.value = time_index_max
    time_slider.end = time_index_max
    if orientation == 'South-North':
        y_range = station_data['vertical_range']
        x_range = station_data['origin'][1]

        latlons = ['%0.2f,%0.2f' %
                   (lats[y, station_data['origin'][1]],
                    lons[y, station_data['origin'][1]])
                   for y in y_range]
    else:
        y_range = station_data['origin'][0]
        x_range = station_data['horizontal_range']

        latlons = ['%0.2f,%0.2f' %
                   (lats[station_data['origin'][0], x],
                    lons[station_data['origin'][0], x])
                   for x in x_range]

    pressure = (wrf_data['PB'][time_index, :, y_range, x_range] +
                wrf_data['P'][time_index, :, y_range, x_range])
    wspd_x_component = wrf_data['U'][time_index, :, y_range, x_range]

    wspd_y_component = wrf_data['V'][time_index, :, y_range, x_range]

    height = pressure_to_height(pressure)
    height = np.array(height)
    height = height * 1000  # convert km to m

    wspd = np.sqrt(wspd_x_component**2 +
                   wspd_y_component**2)

    # calculate the height at each index
    heights = np.diff(height, axis=0)

    # Append heights for last index, this just reuses
    # the second to last value
    last_value = np.reshape(heights[-1, :], (1, SPREAD*2))
    heights = np.vstack((heights, last_value))

    # build the source dataframe
    y_values = height+(heights/2)
    new_source['altitude'] = y_values.ravel()
    new_source['height'] = heights.ravel()
    new_source['wspd'] = wspd.ravel()
    new_source['latlons'] = latlons * wrf_data['P'].shape[1]
    wrf_data.close()
    return new_source, latlons


def update_figure(time, orientation, new_x_range):
    """Updates figure's text labels.
    """
    display_message()
    fig.title.text = update_title(time, orientation)
    fig.x_range.factors = new_x_range
    station_pos.data_source.data.update(get_station_data())


def update_datasource(attr, old, new):
    """Build a new dataframe with the values of each widget.
    """
    orientation = orientation_select.labels[orientation_select.active]
    new_data, new_x_range = build_dataframe(
            time_slider.value,
            station_select.value,
            orientation)
    if new_data is None:
        return  # Something happened.
    source.data.update(new_data)
    update_figure(time_slider.value, orientation, new_x_range)


def get_station_data():
    """Returns info about the currently selected station
    based on widget state.
    """
    location = location_dict[station_select.value]
    x_range = fig.x_range.factors
    station_info_dict = {
        'latlons': [x_range[SPREAD]],
        'altitude': [location['elevation']],
        'label': [station_select.value],
    }
    return station_info_dict


def display_message(msg="", msg_type=None):
    """Displays message to the user. Call with no arguments
    to clear the message box.

    Parameters
    ----------
    msg: str
        The message to display.
    msg_type: str
        Really, just pass in "error" if you want the text to
        red.
    """
    if msg_type == "error":
        message_panel.text = f'<p style="color:#F00">{msg}</p>'
    else:
        message_panel.text = f'<p>{msg}</p>'


# Declare Bokeh Widgets

# Panel for displaying messages to users
message_panel = Div()

# Initialization time selector
init_options = ['00Z', '06Z', '12Z', '18Z']
wrf_init_time_select = Select(title="Initialized Time",
                              value=init_options[1],
                              options=init_options)
wrf_init_time_select.on_change('value', update_datasource)

# Initialization date selector
initial_date = datetime.date.today().strftime("%Y/%m/%d")
wrf_init_date_input = TextInput(title="Initialized Date",
                                value=initial_date)
wrf_init_date_input.on_change('value', check_date)

# Model select widget
wrf_model_options = ['GFS', 'NAM']
wrf_model_select = Select(title="Model",
                          value=wrf_model_options[0],
                          options=wrf_model_options)
wrf_model_select.on_change('value', update_datasource)

# Time Select widget
time_slider = Slider(start=0, end=1,
                     value=0, step=1,
                     title="timestep")
time_slider.on_change('value', update_datasource)

# Query Mysql and build a dict of information on each available station
login_info = os.getenv('MYSQL_CRED','selectonly')
mysql_login = mylogin.get_login_info(login_info)
mysql_login['database'] = 'utility_data'
conn = pymysql.connect(**mysql_login)

query = 'SELECT name,lat,lon,elevation FROM \
        measurementDescriptions WHERE type = "Wind"'
cursor = conn.cursor()
cursor.execute(query)

location_dict = {}
for line in cursor.fetchall():
    if line[0] == 'Total Wind':
        continue
    location_dict[line[0]] = {'lat': float(line[1]),
                              'lon': float(line[2]),
                              'elevation': float(line[3])}
cursor.close()
conn.close()
wrf_data = open_wrf_file()
lats = wrf_data['XLAT']
lons = wrf_data['XLONG']
wrf_time_length = wrf_data['P'].shape[0]

for station in location_dict.values():
    station['origin'] = tunnel_dist(lats, lons,
                                    station['lat'],
                                    station['lon'])
    station["vertical_range"] = range(station['origin'][0]-SPREAD,
                                      station['origin'][0]+SPREAD)
    station["horizontal_range"] = range(station['origin'][1]-SPREAD,
                                        station['origin'][1]+SPREAD)
wrf_data.close()
# Define widgets that are dependent on initial data.

# Station Select Widget
station_select_options = list(location_dict.keys())
station_select = Select(title="Station:",
                        value=station_select_options[0],
                        options=station_select_options)
station_select.on_change('value', update_datasource)

#  Orientation widget
orientation_select = RadioGroup(
    labels=["South-North", "West-East"],
    active=0)
orientation_select.on_change('active', update_datasource)

# Colorbar Widget
cbar_slider = Slider(start=10, end=100,
                     value=40, step=10,
                     title="Colorbar Max")
cbar_slider.on_change('value', colorbar_change)

# Define initial data and figure attributes
mapper = LinearColorMapper(palette=Viridis256, low=0, high=40)
tools = "pan,wheel_zoom,box_zoom,reset,hover,save"
initial_source, initial_x_range = build_dataframe(
    time_slider.value,
    station_select.value,
    "South-North"
    )

source = ColumnDataSource(initial_source)
figure_title = update_title(0, "North-South")
bokeh_formatter = PrintfTickFormatter(format="%d m/s")
color_bar = ColorBar(
     color_mapper=mapper,
     major_label_text_font_size="8pt",
     ticker=BasicTicker(desired_num_ticks=10),
     formatter=bokeh_formatter,
     label_standoff=10,
     border_line_color=None,
     location=(0, 0))

# Initialize bokeh figure
fig = figure(
    title=figure_title,
    plot_width=1000,
    plot_height=600,
    x_range=initial_x_range,
    x_axis_label="Latitude, Longitude",
    y_axis_label="Altitude(m)",
    tools=tools, toolbar_location='left',
    toolbar_sticky=False)

fig.add_layout(color_bar, 'right')
fig.xaxis.major_label_orientation = pi / 4
fig.select_one(HoverTool).tooltips = [
     ('position', '@latlons'),
     ('wspd', '@wspd'),
     ('altitude', '@altitude{int}'),
]
# Plot rectangles representing each point of data
rects = fig.rect(x='latlons', y='altitude',
                 width=1, height='height',
                 fill_color={'field': 'wspd', 'transform': mapper},
                 source=source,
                 line_color=None)


# Plot the position  of the station as a red x
station_init = ColumnDataSource(get_station_data())
station_pos = fig.cross(
    x='latlons',
    y='altitude',
    source=station_init,
    size=10, color="#FF0000", legend='label')

# add the widgets to a box and layout with fig
inputs = widgetbox(
    station_select,
    orientation_select,
    time_slider,
    cbar_slider,
    wrf_init_date_input,
    wrf_init_time_select,
    message_panel
    )
layout = row(inputs, fig)
curdoc().add_root(layout)
