# -*- coding: utf-8 -*-
"""
Plot misclassified cases under different extreme circle densities (single, few, many circles).

TRACKED_ERROR selects the cases to track on the training set:
  - 'false negative': TRUE records rejected by the single-circle experiment (NMO targets),
  - 'false positive': FALSE records accepted by the many-circle experiment (AWS targets).
The same records are then plotted for all three circle densities, each as an extreme circle
map with a histogram of neighboring rainfall.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import rioxarray

# Import common utilities
import _bootstrap  # noqa: F401
from src.qc_plot_utils import plot_rainfall_event

# Import circle processing utilities
from src.qc_circle_process_utils import (
    find_single_extreme_circle,
    compute_single_circle_extent,
    compute_few_circles_extent,
    compute_many_circles_extent,
    get_circles_for_station,
    construct_validation_circles,
)

# Import data loading utilities
from src.qc_data_loader import (
    AdjacentPrecipSource,
    load_station_info,
    load_circles,
    load_qc_result,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_DIR = SCRIPT_DIR.parent / 'data'
TUNING_TABLE_DIR = SCRIPT_DIR.parent / 'outputs' / 'tables' / 'tuning'
FIGURE_DIR = SCRIPT_DIR.parent / 'outputs' / 'figures'

TRACKED_ERROR = 'false negative'  # 'false negative' or 'false positive'

# Configuration for neighbor data time window (hours before and after target time)
NEIGHBOR_HOR = 2

DATASET = 'train'

CLIMATE_LIMIT = 184.4
OUTLIER_CIRCLES_RADIUS, EXTREME_CIRCLES_RADIUS  = 60, 80

INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'
DEM_FILE = DATA_DIR / 'external' / 'basemap' / 'gd_dem_1km.tif'
# Cached adjacent data of the dataset (data/cache/); the database is used if missing
INPUT_CACHE_FILES = [DATA_DIR / 'cache' / f"gd_validation_data_{DATASET}_adjacent{NEIGHBOR_HOR}h.parquet"]

INPUT_FINE_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{EXTREME_CIRCLES_RADIUS}km.csv"
INPUT_COARSE_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_outlier' / f"neighbor_circles_outlier_{EXTREME_CIRCLES_RADIUS}km.csv"

INPUT_FILES = {
    density: TUNING_TABLE_DIR / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_{density}.csv"
    for density in ['single', 'few', 'many']
}

# Per tracked error: experiment defining the cases, actual label of the cases,
# confusion types (rejected, accepted) with their file-name labels, and output folder
TRACKING = {
    'false negative': {'reference': 'single', 'actual': True,
                       'rejected': ('False negative', 'fn'), 'accepted': ('True positive', 'tp'),
                       'output_dir': FIGURE_DIR / f"cases_circle_gap_fn_{DATASET}"},
    'false positive': {'reference': 'many', 'actual': False,
                       'rejected': ('True negative', 'tn'), 'accepted': ('False positive', 'fp'),
                       'output_dir': FIGURE_DIR / f"cases_circle_gap_fp_{DATASET}"},
}
TRACK = TRACKING[TRACKED_ERROR]
OUTPUT_DIR = TRACK['output_dir']
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_qc_rec(input_file: Path, df_qc_reference: pd.DataFrame) -> pd.DataFrame:
    """
    Load the records of input_file that match the tracked cases of the reference experiment,
    and label each with its confusion type in this experiment.
    """
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    df['ddatetime'] = pd.to_datetime(df['ddatetime'])
    df['stacode'] = df['stacode'].astype(str)

    df_keys = df_qc_reference[['stacode', 'ddatetime']].copy()
    df_keys['stacode'] = df_keys['stacode'].astype(str)
    df_matched = pd.merge(df, df_keys, on=['stacode', 'ddatetime'], how='inner')

    rejected = df_matched['qc_label'] == 'FALSE'
    df_matched.loc[rejected, 'confusion_type'] = TRACK['rejected'][0]
    df_matched.loc[~rejected, 'confusion_type'] = TRACK['accepted'][0]
    return df_matched


def plot_qc_result(source: AdjacentPrecipSource, total_records, df_stations, flag: str,
                   df_qc_result: pd.DataFrame, df_extreme_circles, raster_dem):

    file_labels = dict([TRACK['rejected'], TRACK['accepted']])

    for idx, record in df_qc_result.reset_index(drop=True).iterrows():
        target_stacode = str(record['stacode'])
        ddatetime = record['ddatetime']
        target_precip = float(record['r'])
        confusion_type = record['confusion_type']

        print(f"Plotting map {idx+1}/{total_records} precipitation {target_precip} at {target_stacode}, {ddatetime}")

        # Adjacent hours precipitation data
        df_all_adj = source.get(ddatetime - timedelta(hours=NEIGHBOR_HOR), ddatetime + timedelta(hours=NEIGHBOR_HOR))
        df_all_adj = df_all_adj[df_all_adj['r'] <= CLIMATE_LIMIT]

        # Current hour precipitation data for all stations
        df_all = df_all_adj[df_all_adj['ddatetime'] == ddatetime].copy()
        if df_all.empty:
            continue

        # Get extreme circles containing this station
        df_circles = get_circles_for_station(target_stacode, df_extreme_circles)

        # only plot a single extreme circle that was used
        if flag == 'single':
            df_circles = find_single_extreme_circle(
                extreme_circles=df_circles,
                target_stacode=target_stacode,
                df_station_info=df_stations,
                )

        # Construct validation circles from parsed locations
        df_validation_circles = construct_validation_circles(record['validation_circle_locs'], df_circles)

        # Extract all stations from extreme circles
        extreme_circle_stations = df_circles['neighbors'].explode().unique()

        # Filter 5-hour data for stations within extreme_circles circles
        df_circles_precip = df_all_adj[df_all_adj['stacode'].isin(extreme_circle_stations)]

        # Merge with station locations
        df_circles_precip = pd.merge(
            df_circles_precip,
            df_stations[['stacode', 'lat', 'lon']],
            on='stacode',
            how='left'
        )

        # Select extent computer based on flag
        if flag == 'single':
            map_extent = compute_single_circle_extent(df_circles)
        elif flag == 'few':
            map_extent = compute_few_circles_extent(df_circles)
        elif flag == 'many':
            map_extent = compute_many_circles_extent(df_circles)

        output_file = OUTPUT_DIR / (f"qc_{target_stacode}_{ddatetime.strftime('%Y%m%d%H')}_{target_precip}_"
                                    f"{file_labels[confusion_type]}@{flag}.png")

        plot_rainfall_event(target_stacode, ddatetime, target_precip,
                            df_circles_precip,
                            df_circles,
                            df_validation_circles,
                            confusion_type,
                            dem_data=raster_dem,
                            df_all_extreme_circles=df_extreme_circles,
                            map_extent=map_extent,
                            output_file=output_file)


def main():
    print(f"Tracking {TRACKED_ERROR} cases, starting at {datetime.now()}")

    print("Loading DEM data...")
    raster_dem = rioxarray.open_rasterio(DEM_FILE)

    print("Loading station location info...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    df_stations['stacode'] = df_stations['stacode'].astype(str)

    print("Loading small-gap circles for inspecting extremes...")
    df_fine_extreme_circles = load_circles(INPUT_FINE_EXTREME_CIRCLES)

    print("Loading big-step circles for inspecting extremes...")
    df_coarse_extreme_circles = load_circles(INPUT_COARSE_EXTREME_CIRCLES)

    print("Loading qc predicted result...")
    df_qc_reference = load_qc_result(INPUT_FILES[TRACK['reference']], qc_type=TRACKED_ERROR)
    df_qc = {density: load_qc_rec(INPUT_FILES[density], df_qc_reference) for density in INPUT_FILES}
    circles = {'single': df_fine_extreme_circles, 'few': df_coarse_extreme_circles, 'many': df_fine_extreme_circles}

    total_records = len(df_qc_reference)
    with AdjacentPrecipSource(INPUT_CACHE_FILES, CONFIG_FILE, DB_SECTION, df_stations['stacode']) as source:
        for density in ['single', 'few', 'many']:
            print(f"Plotting {total_records} {TRACKED_ERROR} cases @ {density} circles...")
            plot_qc_result(source, total_records, df_stations, density,
                           df_qc[density], circles[density], raster_dem)

    print(f"\nFinished at {datetime.now()}")


if __name__ == '__main__':
    main()
