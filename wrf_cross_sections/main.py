"""wrf_cross_sections.py

A script which sets up a bokeh figure and interactions for wrf
windspeed cross sections.

Notes
-----
Required environment variables:
WRF_DATA_DIRECTORY: The path to the directory in which the wrf
    data can be found. Assumes the directory has the following
    structure:
        {year}/{month}/{day}/WRF{model}_{init_time}/wrf_d02_hourly.nc
"""
from datetime import timedelta, datetime, date
from math import pi
import os
import sys

from bokeh.plotting import (figure, curdoc)
from bokeh.palettes import Viridis256
from bokeh.models import (
    BasicTicker,
    Button,
    ColorBar,
    ColumnDataSource,
    Div,
    HoverTool,
    TextInput,
    LinearColorMapper,
    PrintfTickFormatter,
    Slider,
    Select,
    RadioGroup,
)
from bokeh.layouts import row, column, widgetbox
import colorcet as cc
from dateutil import parser
from netCDF4 import Dataset
import numpy as np
from numpy import cos, sin
import pandas as pd


# The directory in which to find the data files
DATA_DIR = os.getenv('WRF_DATA_DIRECTORY')
if DATA_DIR is None:
    sys.exit('WRF_DATA_DIRECTORY env variable not set.')

# Path to the elevation file
dirname = os.path.dirname(__file__)
DEFAULT_ELEVATION_FILE = os.path.join(dirname, 'data', 'elevation.csv')
ELEVATION_FILE = os.getenv('CROSS_SECTION_ELEVATION_FILE',
                           DEFAULT_ELEVATION_FILE)


def validate_date(attr, old, new):
    """Valitate the format of the WRF initialization date input by the user.
    """
    try:
        parser.parse(wrf_init_date_input.value)
    except ValueError:
        display_message('Unrecognized date format, please use\
                         "yyyy/mm/dd".', 'error')
        return None
    update_datasource(attr, old, new)


def selected_date():
    """Parses selected widget options and returns a datetime object.
    """
    day = parser.parse(wrf_init_date_input.value)
    hour = wrf_init_time_select.value
    hour = timedelta(hours=int(hour[:-1]))
    return day+hour


def pressure_to_height(pressure):
    """Converts pressure in Pa to height.
    Notes
    -----
    Adapted from metpy.calc.pressure_to_height_std, removed unit
    checking and hardcoded variables to reduce dependencies.
    MetPy sourcode: https://github.com/Unidata/MetPy
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


def get_elevation_data():
    """Returns a pandas dataframe object or None if the file
    does not exist.

    Returns
    -------
    elevation_data: pandas dataframe
        Dataframe of the elevation data.
    """
    try:
        elevation_data = pd.read_csv(ELEVATION_FILE, header=None)
    except FileNotFoundError:
        display_message("Elevation file does not exist.", 'error')
        return None
    return elevation_data


def open_wrf_file():
    """Returns an netCDF Dataset object or None if the file
    does not exist.

    Returns
    -------
    wrf_file: netCDF4 Dataset
        The opened wrf file.

    Notes
    -----
    The returned file must be closed by the user.
    """
    try:
        wrf_file = Dataset(get_weather_file())
    except FileNotFoundError:
        display_message("WRF file does not exist.", 'error')
        return None
    return wrf_file


def colorbar_change(attr, old, new):
    """Callback for cbar_slider, sets the maximum value of
    the colorbar to the slider value, sets the minimum if
    applicable.
    """
    mapper.high = cbar_slider.value

    # water vapor transport uses the x or y wind component, and negative
    # values are possible
    if variable_select.value == 'Water Vapor Transport':
        mapper.low = -cbar_slider.value
    else:
        mapper.low = 0


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
    variable = variable_select.value
    wrf_data = open_wrf_file()
    curr_byte_time = wrf_data["Times"][time_index].tostring()
    wrf_data.close()
    curr_time = datetime.strptime(curr_byte_time.decode('utf-8'),
                                  "%Y-%m-%d_%H:%M:%S")
    plot_title = '{} {} ({}) Initialized: {}   Valid: {}'.format(
                  station_name,
                  variable,
                  orientation,
                  wrf_init_time.strftime(time_format),
                  curr_time.strftime(time_format))
    return plot_title


def wspd_colorbar():
    """Sets the color mapper for wind speed
    """
    cbar_slider.value = 40
    mapper.palette = Viridis256
    bokeh_formatter.format = "%d m/s"


def qvapor_colorbar():
    """Sets the color mapper for water vapor mixing vapor
    """
    cbar_slider.value = 10
    mapper.palette = Viridis256
    bokeh_formatter.format = "%f g/kg"


def vt_colorbar():
    """Sets the color mapper settings for water vapor transfer
    """
    cbar_slider.value = 100
    mapper.palette = cc.coolwarm
    bokeh_formatter.format = "%f g/kg m/s"


def update_variable(attr, old, new):
    """Variable Select callback. Sets the colorbar appropriately
    and then updates data.
    """
    if new == 'Wind Speed':
        wspd_colorbar()
    elif new == 'Water Vapor Mixing Ratio':
        qvapor_colorbar()
    elif new == 'Water Vapor Transport':
        vt_colorbar()
    update_datasource(attr, old, new)


def build_dataframes(time_index, station_name, orientation):
    """Gathers the required variables for plotting from
    the WRF file and organizes them into a dataframe to
    be plotted by bokeh.

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
    wrf_data = open_wrf_file()
    if wrf_data is None:
        return
    lats = wrf_data['XLAT'][:]
    lons = wrf_data['XLONG'][:]
    spread = int(width_slider.value / 2)
    # Get the indices we need for the current station
    station_data = location_dict[station_name]
    station_data['origin'] = tunnel_dist(lats, lons,
                                         station_data['lat'],
                                         station_data['lon'])
    station_data["y_range"] = np.arange(station_data['origin'][0]-spread,
                                        station_data['origin'][0]+spread)
    station_data["x_range"] = np.arange(station_data['origin'][1]-spread,
                                        station_data['origin'][1]+spread)

    # If we're opening a file with a shorter time scale,
    # and we're out of bounds, bring the index back in.
    time_index_max = wrf_data['Times'].shape[0]-1
    if time_index > time_index_max:
        time_index = time_index_max
        time_slider.value = time_index_max

    time_slider.end = time_index_max

    if orientation == 'South-North':
        y_range = station_data['y_range']
        x_range = station_data['origin'][1]

        station_latlons = ['%0.2f,%0.2f' %
                           (lats[y, station_data['origin'][1]],
                            lons[y, station_data['origin'][1]])
                           for y in y_range]
    else:
        y_range = station_data['origin'][0]
        x_range = station_data['x_range']

        station_latlons = ['%0.2f,%0.2f' %
                           (lats[station_data['origin'][0], x],
                            lons[station_data['origin'][0], x])
                           for x in x_range]

    # Calculate pressure from base state and perturbation pressure
    pressure = (wrf_data['PB'][time_index, :, y_range, x_range] +
                wrf_data['P'][time_index, :, y_range, x_range])

    # Convert pressure to altitude
    altitude = pressure_to_height(pressure)
    altitude = np.array(altitude)
    altitude = altitude * 1000  # km to m

    # Calculate the height of each data point
    heights = np.diff(altitude, axis=0)

    # Append heights for last index, this just reuses
    # the second to last value
    last_value = np.reshape(heights[-1, :], (1, spread*2))
    heights = np.vstack((heights, last_value))

    selected_variable = variable_select.value
    if selected_variable == 'Wind Speed':
        wspd_x_component = wrf_data['U'][time_index, :, y_range, x_range]

        wspd_y_component = wrf_data['V'][time_index, :, y_range, x_range]

        variable = np.sqrt(wspd_x_component**2 +
                           wspd_y_component**2)
    elif selected_variable == 'Water Vapor Mixing Ratio':
        # kg/kg to g/kg
        mixing_ratio = wrf_data['QVAPOR'][time_index, :, y_range, x_range]
        mixing_ratio_grams = mixing_ratio * 1000
        variable = mixing_ratio_grams
    elif selected_variable == 'Water Vapor Transport':
        # kg/kg to g/kg
        mixing_ratio = wrf_data['QVAPOR'][time_index, :, y_range, x_range]
        mixing_ratio_grams = mixing_ratio * 1000
        if orientation == 'South-North':
            wspd_component = wrf_data['U'][time_index, :, y_range, x_range]
        else:
            wspd_component = wrf_data['V'][time_index, :, y_range, x_range]
        variable = mixing_ratio_grams * wspd_component

    wrf_data.close()

    # build the source dataframes
    y_values = altitude+(heights/2)  # center the y values of each point
    cross_source = pd.DataFrame()
    cross_source['altitude'] = y_values.ravel()
    cross_source['height'] = heights.ravel()
    cross_source['value'] = variable.ravel()
    cross_source['latlons'] = station_latlons * wrf_data['P'].shape[1]

    elevation_source = pd.DataFrame()
    elevation_source['elevation'] = elevation_data.iloc[y_range, x_range]
    elevation_source['latlons'] = station_latlons
    return cross_source, station_latlons, elevation_source


def update_figure(time, orientation, new_x_range):
    """Updates figure's text labels.
    """
    display_message()
    fig.title.text = update_title(time, orientation)
    fig.x_range.factors = new_x_range
    station_pos.data_source.data.update(get_station_data())


def click_update_datasource():
    """Callback function for buttons to update without passing
    the normal attr, old, new parameters.
    """
    update_datasource(None, None, None)


def update_datasource(attr, old, new):
    """Create a new datasource by selecting data from the WRF file using
    bokeh widgets as parameters.
    """
    orientation = orientation_select.labels[orientation_select.active]
    new_data, new_x_range, new_elev = build_dataframes(
            time_slider.value,
            station_select.value,
            orientation)
    if new_data is None:
        return
    update_figure(time_slider.value, orientation, new_x_range)
    rects.data_source.data.update(new_data)
    elevation.data_source.data['index'] = new_elev.index
    elevation.data_source.data.update(new_elev)


def get_station_data():
    """Returns info about the currently selected station
    based on widget state.
    """
    location = location_dict[station_select.value]
    x_range = fig.x_range.factors
    spread = int(width_slider.value / 2)
    station_info_dict = {
        'latlons': [x_range[spread]],
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


def find_initial():
    """Looks for file to use as initial data.
    """
    initial_model = None
    initial_date = parser.parse('2018/12/07')  # hard coded to demo file
    initial_time = None
    days_to_try = 10
    while initial_model is None:
        dir_date = os.path.join(initial_date.strftime('%Y'),
                                initial_date.strftime('%m'),
                                initial_date.strftime('%d'))
        location = os.path.join(DATA_DIR, dir_date)
        files = os.listdir(location)
        if files is not None:
            for f in files:
                if f[:3] == "WRF":
                    initial_model = f[3:6]
                    if initial_model not in ["NAM", "GFS"]:
                        continue
                    initial_time = f[-3:]
                    initial_date = initial_date.strftime("%Y/%m/%d")
                return initial_time, initial_date, initial_model
        days_to_try -= 1
        initial_date = initial_date-timedelta(days=1)
        if days_to_try <= 0:
            raise Exception("Could not locate forecast newer than 10 days")

# BOKEH WIDGETS
# Here we define bokeh widgets that may provide inital values.
# Below, more widgets are defined that act on or require data.


# Panel for displaying messages to users
message_panel = Div()

# Get an existing file for default values
init_time, init_date, init_model = find_initial()

# Initialization time selector
init_options = ['00Z', '06Z', '12Z', '18Z']
wrf_init_time_select = Select(title="Initialized Time",
                              value=init_time,
                              options=init_options)
wrf_init_time_select.on_change('value', update_datasource)

# Initialization date selector
initial_date = date.today().strftime("%Y/%m/%d")
wrf_init_date_input = TextInput(title="Initialized Date",
                                value=init_date)
wrf_init_date_input.on_change('value', validate_date)

# Model select widget
wrf_model_options = ['GFS', 'NAM']
wrf_model_select = Select(title="Model",
                          value=init_model,
                          options=wrf_model_options)
wrf_model_select.on_change('value', update_datasource)

# Time Select widget
time_slider = Slider(start=0, end=1,
                     value=0, step=1,
                     title="Timestep")
time_slider.on_change('value', update_datasource)

# Width select widget
width_slider = Slider(start=10, end=190,
                      value=20, step=10,
                      title="Cross section width")
# button to apply width changes
width_button = Button(label="Apply Width")
width_button.on_click(click_update_datasource)

# INITIALIZATION DATA

# Location Data
location_dict = {}
location_dict['Tucson - UA'] = {'lat': 32.2319,
                                'lon': -110.9501,
                                'elevation': 728}
location_dict['Red Horse Wind'] = {'lat': 32.285461,
                                   'lon': -110.090711,
                                   'elevation': 1580}
location_dict['Chiricahua Gap'] = {'lat': 31.9,
                                   'lon': -108.5,
                                   'elevation': 1666}
location_dict['San Pedro River - US/Mexico Border'] = {'lat': 31.334,
                                                       'lon': -110.148,
                                                       'elevation': 1310}
elevation_data = get_elevation_data()

wrf_data = open_wrf_file()
lats = wrf_data['XLAT']
lons = wrf_data['XLONG']
wrf_time_length = wrf_data['P'].shape[0]

wrf_data.close()

# Define widgets that are dependent on initial data.

# Variable select widget
variable_select_options = ['Wind Speed',
                           'Water Vapor Mixing Ratio',
                           'Water Vapor Transport']
variable_select = Select(
    title='Variable:',
    value=variable_select_options[0],
    options=variable_select_options)
variable_select.on_change('value', update_variable)

# Station Select Widget
station_select_options = list(location_dict.keys())
station_select = Select(title="Location:",
                        value=station_select_options[0],
                        options=station_select_options)
station_select.on_change('value', update_datasource)

#  Orientation widget
orientation_select = RadioGroup(
    labels=["South-North", "West-East"],
    active=0)
orientation_select.on_change('active', update_datasource)

# Colorbar Widget
cbar_slider = Slider(start=0, end=100,
                     value=40, step=1,
                     title="Colorbar Max")
cbar_slider.on_change('value', colorbar_change)
bokeh_formatter = PrintfTickFormatter(format="%d m/s")

# Define initial data and figure attributes
mapper = LinearColorMapper(palette=Viridis256, low=0, high=40)
tools = "pan,wheel_zoom,box_zoom,reset,hover,save"
initial_cross_df, initial_x_range, initial_elev_df = build_dataframes(
    time_slider.value,
    station_select.value,
    "South-North"
    )
cross_source = ColumnDataSource(initial_cross_df)
elevation_source = ColumnDataSource(initial_elev_df)
figure_title = update_title(0, "South-North")

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
    background_fill_color='Gray',
    tools=tools, toolbar_location='left',
    toolbar_sticky=False)

fig.add_layout(color_bar, 'right')
fig.xaxis.major_label_orientation = pi / 4
fig.select_one(HoverTool).tooltips = [
     ('position', '@latlons'),
     ('value', '@value'),
     ('altitude', '@altitude{int}'),
]

# Plot rectangles representing each point of data
rects = fig.rect(x='latlons', y='altitude',
                 width=1, height='height',
                 fill_color={'field': 'value', 'transform': mapper},
                 source=cross_source,
                 line_color=None)

# Plot surface elevation as a line
elevation = fig.line(x='latlons',
                     y='elevation',
                     line_width=2,
                     line_color='Black',
                     source=elevation_source,
                     legend='Elevation')

# Plot the position  of the station as a red x
station_init = ColumnDataSource(get_station_data())
station_pos = fig.cross(
    x='latlons',
    y='altitude',
    source=station_init,
    size=10, color="#FF0000", legend='label')

# add the widgets to a box and layout with fig
inputs_left = widgetbox(
    station_select,
    variable_select,
    orientation_select,
    time_slider,
    cbar_slider,
    width_slider,
    width_button)

inputs_right = widgetbox(
    wrf_init_date_input,
    wrf_init_time_select,
    wrf_model_select,
    message_panel
    )
widgets = row(inputs_left, inputs_right)
figure = row(fig, sizing_mode='fixed')
container = column(widgets, figure, sizing_mode='fixed')
curdoc().title = "WRF Cross Sections"
curdoc().add_root(container)
