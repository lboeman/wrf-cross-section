# WRF Cross Sections

Bokeh app for visualizing a cross section of University of Arizona WRF run output. Allows visualization of the following variables:
- Wind Speed
- Water vapor mixing ratio
- Water vapor transport

## Usage
  * Environment Variables:
    * WRF_DATA_DIRECTORY: The path to the directory to find WRF data in. The directory must contain the following structure:

		`<year>/<month>/<day>/WRF<model>_<initialization time>/wrf_d02_hourly.nc`
		
  * Serve cross sections with bokeh using a command like:
```
WRF_DATA_DIRECTORY=<path to wrf files> bokeh serve wrf_cross_sections --port PORT
```
