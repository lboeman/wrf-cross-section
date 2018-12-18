import argparse
import logging
import netCDF4
import numpy as np
import pandas as pd
from pathlib import Path
import sys


def elevations_from_files(wrf_filename, srtm_directory, samples):
    """Iterates over possible latitude, longitude pairs in the
    WRF file and generates a csv of elevations, where the the row
    and column numbers match the indices of their associated
    latitude and longitude in the WRF file.

    NOTE: This function assumes you have already gathered the needed
    SRTM files in the provided directory.
    """
    wrf_file = netCDF4.Dataset(wrf_filename, 'r')
    lats = wrf_file.variables['XLAT'][:]
    lons = wrf_file.variables['XLONG'][:]
    wrf_file.close()

    def elevation_from_file(lat, lon):
        # SRTM files are designated by the latitude and logitude
        # of their bottom left corner, and always with positive
        # values. i.e. the file containing (-6.3 N, -110.23 E)
        # will be labelled S07W111.
        if lat >= 0:
            file_lat = f'N{int(lat):02d}'
        else:
            file_lat = f'S{int(-np.floor(lat)):02d}'
        if lon >= 0:
            file_lon = f'E{int(lon):03d}'
        else:
            file_lon = f'W{int(-np.floor(lon)):03d}'
        srtm_filename = f'{file_lat}{file_lon}.hgt'
        srtm_filepath = srtm_directory.joinpath(srtm_filename)
        with open(srtm_filepath, 'r') as hgt_data:
            elevations = np.fromfile(hgt_data,
                                     np.dtype('>i2'),
                                     samples * samples)
            elevations = elevations.reshape((samples, samples))
        lat_index = int(round((lat - int(lat)) * (samples - 1), 0))
        lon_index = int(round((lon - int(lon)) * (samples - 1), 0))
        return elevations[samples - 1 - lat_index, lon_index].astype(int)
    v_elev_from_file = np.vectorize(elevation_from_file)
    elevations_arr = np.ma.getdata(v_elev_from_file(lats, lons))
    elevations = pd.DataFrame(elevations_arr)
    return elevations


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
                description=('Generate a csv of elevations for each grid '
                             'point of a specified WRF file using NASA '
                             'Shuttle Radar Topography Mission(SRTM) v1 '
                             'data.'))
    parser.add_argument('wrf_file',
                        help='Path of the WRF file.')
    parser.add_argument('srtm_directory',
                        help='Directory containing SRTM files.')
    parser.add_argument('output_file',
                        help='The path of the desired output file.')
    parser.add_argument('-o', action='store_true',
                        help='Overwrite output file if it already exists.')
    parser.add_argument('--srtm_samples', type=int, default=1201,
                        help=('The number of samples in the SRTM file. '
                              'Defaults to 1201 for version 1, 3 arc-second'
                              'files.'))
    args = parser.parse_args()

    wrf_file = Path(args.wrf_file)
    if not wrf_file.exists():
        logger.error(f'WRF file {wrf_file} does not exist.')
        sys.exit(1)

    srtm_directory = Path(args.srtm_directory)
    if not srtm_directory.is_dir():
        logger.error(f'SRTM directory {srtm_directory} does not exist.')
        sys.exit(1)

    output_file = Path(args.output_file)
    if output_file.exists() and not args.o:
        logger.error(f'Outputfile {output_file} already exists.')
        sys.exit(1)

    csv_data = elevations_from_files(args.wrf_file,
                                     srtm_directory,
                                     args.srtm_samples)
    csv_data.to_csv(output_file, index=False)
    logging.info(f'File {output_file.resolve()} written successfully.')
