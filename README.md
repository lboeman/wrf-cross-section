# WRF Cross Sections

Bokeh app for visualizing a cross section of University of Arizona WRF run output. Allows visualization of the following variables:
- Wind Speed
- Water vapor mixing ratio
- Water vapor transport

The following locations are provided: University of Arizona in Tucson Arizona, Chiricahua Gap in New Mexico, and the San Pedro River at the US/Mexico border.

## Setup
This application requires Python 3.6 or greater.

Install requirements using a command such as:
	`pip install -r requirements.txt`


## Usage
  * Environment Variables:
    * WRF_DATA_DIRECTORY: The path to the directory to find WRF data in. The directory must contain the following structure:

		`<year>/<month>/<day>/WRF<model>_<initialization time>/wrf_d02_hourly.nc` where month and day are two digit, zero-padded numbers.
	* CROSS_SECTION_ELEVATION_FILE: The absolute path to a csv file containing the elevations of each grid point in the WRF domain. Defaults to the provided file in the `wrf_cross_sections/data` directory.

  * The app can be run from the root of the source folder with the following command.:
```
WRF_DATA_DIRECTORY=<path to wrf files> ./run
```

The application will then be available at [http://localhost:5006/wrf_cross_sections](http://localhost:5006/wrf_cross_sections)


## SRTM data
Elevation is plotted using NASA Shuttle Radar Topography Mission(SRTM) V1, 3arc-second data. A CSV of elevation data associated with the University of Arizona's WRF model domain is provided in the `wrf_cross_sections/data` directory. A script, `srtm_to_csv.py` is provided in the `scripts` directory for generating these a csv of elevation data for WRF files with different domains.
The script has the following requirements:

- The WRF file includes an *XLAT* and *XLONG* variable.
- There is a directory of 1 degree SRTM `.hgt` files that include data for the domain of your WRF file.

The script may be run from the `scripts` directory with the following command.
```
python srtm_to_csv.py <path to WRF file> <path to SRTM file directory> <path to output file>
```

SRTM data courtesy of the U.S. Geological Survey.

[NASA Shuttle Radar Topography Mission](https://www2.jpl.nasa.gov/srtm/) 
