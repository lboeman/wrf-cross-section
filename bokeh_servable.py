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
from CrossSectionData import CrossSectionData
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

def colorbar_change():
    mapper.high = cbar_slider.value

def update_title(time):
    ime_format = '%y/%m/%d %H:%M:%SZ'
    plot_title = bokeh_source.station

    if bokeh_source.orientation == 'vertical':
        plot_title = plot_title+' South-Nworth'
    else:
        plot_title = plot_title+' West-East'
    plot_title = ('%s  Initialized: %s   Valid: %s' % (
                  plot_title,
                  bokeh_source.init.strftime(time_format),
                  (bokeh_source.init+
                   timedelta(hours=time)).strftime(time_format)))
    return plot_title


def update_glyph_sources(i):
    rects.y = 'y%d' % i
    rects.height = 'h%d' % i
    rects.fill_color.field = 'w%d' % i 
    pf.select_one(HoverTool).tooltops = [
        ('position', '@ll'),
        ('wspd', '@w%s' % i),
        ('altitude', '@y%s' % i)
    ]
def update_figure(time):
    pf.title = update_title(time)
    latlons = bokeh.source_data['ll']
    pf.x_range = latlons[0:(2*bokeh_source.spread)+1]

def station_change():
    time = time_slider.value
    new_station = station_select.value
    bokeh_source.station = new_station
    bokeh_source.update(time)
    update_title(time)

def time_change():
    time = time_slider.value
    bokeh_source.update(time)
    update_glyph_sources(time)
    update_figure(time)


def orientation_change():
    bokeh_source.orientation = orientation_select.value
    bokeh_source.update_source(time_slider.value)
       
filename = get_weather_file(time, model)

bokeh_source = CrossSectionData(filename)
bokeh_source.update_source(0) # initialize with the first timestep

# Setup bokeh plot
mapper = LinearColorMapper(palette=Viridis256, low=0, high=40)
tools = "pan,wheel_zoom,reset,hover,save"
source = ColumnDataSource(bokeh_source.source_data)

# Initialize bokeh figure
pf = figure(title=update_title(0),
            x_range=bokeh_source.source_data['ll'][0:(2*bokeh_source.spread)+1],
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
# TODO: fix this shit.
station_pos = pf.cross(x=[latlons[10]], y=[station['elevation']],
         size=10, color="#FF0000", legend=bokeh_source.station)
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

cbar_slider = Slider(start=10, end=100,
                     value=40, step=10,
                     title="Colorbar Max",
                     callback=cbcallback)

# Time Select widget
slider_end = bokeh_source.wrf_data['P'].shape[0]-1
time_slider = Slider(start=0, end=slider_end,
                 value=0, step=1, title="timestep",
                 callback=tcallback)

# Station Select Widget
station_select_options = bokeh_source.list_stations()
station_select = Select(title="Station:",
                        value = station_select_options[0],
                        options = station_select_options)
#  Orientation widget
orientation_select = RadioGroup(
    labels = ["vertical","horizontal"],
    active = 0)

inputs = widgetbox(
    station_select,
    orientation_select,
    time_slider,
    cbar_slider,
    )
layout = row(
    inputs,
    pf
    )
show(layout)
