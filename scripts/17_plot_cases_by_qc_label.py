# -*- coding: utf-8 -*-
"""
Plot false positive/negative or type1/2/3/4/5 cases from QC validation results: train/test.
Visualizes misclassified precipitation events with extreme circle maps
and precipitation histograms for each case.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import rioxarray

# Import common utilities
import _bootstrap  # noqa: F401
from src.qc_plot_utils import (
    plot_rainfall_event,
)

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
TABLE_DIR = SCRIPT_DIR.parent / 'outputs' / 'tables'
FIGURE_DIR = SCRIPT_DIR.parent / 'outputs' / 'figures'

# Configuration for neighbor data time window (hours before and after target time)
NEIGHBOR_HOR = 2

DATASET = 'test'  # plot false case from "train" dataset or "test" dataset
QC_TYPE = 'false'  # "false": fn or fp or "type1/2/3/4/5"

CLIMATE_LIMIT = 184.4
OUTLIER_CIRCLES_RADIUS, EXTREME_CIRCLES_RADIUS  = 60, 80

# Test mode configuration - choose one: 'single', 'few', or 'many'
MODE = 'many' 

INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'
DEM_FILE = DATA_DIR / 'external' / 'basemap' / 'gd_dem_1km.tif'
# Cached adjacent data (data/cache/); the database is used if none exists.
INPUT_CACHE_FILES = [DATA_DIR / 'cache' / f"gd_validation_data_{DATASET}_adjacent{NEIGHBOR_HOR}h.parquet"]

OUTPUT_BASENAME = f"qc_{QC_TYPE}"

# Configuration based on test mode
if MODE == 'single':
    INPUT_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{EXTREME_CIRCLES_RADIUS}km.csv"
    INPUT_FILE = TABLE_DIR / 'tuning' / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_single.csv"
    OUTPUT_DIR = FIGURE_DIR / f"qc_{QC_TYPE}@single_circle_{DATASET}"

elif MODE == 'few':
    # adopt the outlier circles to substitute extreme circles because outlier circles have larger gaps, thereby few circles
    INPUT_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_outlier' / f"neighbor_circles_outlier_{EXTREME_CIRCLES_RADIUS}km.csv"
    INPUT_FILE = TABLE_DIR / 'tuning' / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_few.csv"
    OUTPUT_DIR = FIGURE_DIR / f"qc_{QC_TYPE}@few_circle_{DATASET}"

elif MODE == 'many':
    INPUT_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{EXTREME_CIRCLES_RADIUS}km.csv"
    # training set: adopted setting written by 08_qc_tune_circle_step.py; test set: 12_qc_evaluate.py
    INPUT_FILE = (TABLE_DIR / 'tuning' / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_many.csv"
                  if DATASET == 'train' else
                  TABLE_DIR / 'validation' / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km.csv")
    OUTPUT_DIR = FIGURE_DIR / f"qc_{QC_TYPE}@many_circle_{DATASET}"

else:
    raise ValueError(f"Invalid MODE: '{MODE}'. Must be 'single', 'few', or 'many'")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_qc_result(source: AdjacentPrecipSource, total_records, df_stations,
                   df_qc_result: pd.DataFrame, df_extreme_circles, raster_dem):

    for idx, record in df_qc_result.iterrows():
        target_stacode = str(record['stacode'])
        ddatetime = record['ddatetime']
        target_precip = float(record['r'])
        confusion_type = record['confusion_type']

        print(f"Plotting map {idx+1}/{total_records} precipitation {target_precip} at {target_stacode}, {ddatetime}")

        # Fetch adjacent hours precipitation data
        df_all_adj = source.get(ddatetime - timedelta(hours=NEIGHBOR_HOR), ddatetime + timedelta(hours=NEIGHBOR_HOR))
        df_all_adj = df_all_adj[df_all_adj['r']<=CLIMATE_LIMIT]

        # Fetch current hour precipitation data for all stations
        df_all = df_all_adj[df_all_adj['ddatetime'] == ddatetime].copy()            
        if df_all.empty:
            continue
        
        # Get extreme circles containing this station
        df_circles = get_circles_for_station(target_stacode, df_extreme_circles)

        # only plot a single extreme circle that was used 
        if MODE == 'single':
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

        # Select extent computer based on test mode
        if MODE == 'single':
            map_extent = compute_single_circle_extent(df_circles)
        elif MODE == 'few':
            map_extent = compute_few_circles_extent(df_circles)
        elif MODE == 'many':
            map_extent = compute_many_circles_extent(df_circles)

        # Save figure
        output_file = OUTPUT_DIR / f"{OUTPUT_BASENAME}_{target_stacode}_{ddatetime.strftime('%Y%m%d%H')}_{str(target_precip)}.png"

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
    """ Main function to process all false positive & false negative cases. """

    print("Loading DEM data...")
    raster_dem = rioxarray.open_rasterio(DEM_FILE)

    print("Loading station location info...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    df_stations['stacode'] = df_stations['stacode'].astype(str)
    
    print("Loading circles for inspecting extremes...")
    df_extreme_circles = load_circles(INPUT_EXTREME_CIRCLES)

    print("Loading qc predicted result...")
    df_qc_result = load_qc_result(INPUT_FILE, qc_type=QC_TYPE)
    total_records = len(df_qc_result)  

    with AdjacentPrecipSource(INPUT_CACHE_FILES, CONFIG_FILE, DB_SECTION, df_stations['stacode']) as source:
        print(f"Plotting {total_records} qc {QC_TYPE} cases...")
        plot_qc_result(source, total_records, df_stations,
                       df_qc_result, df_extreme_circles, raster_dem)
    print(f"\nFinished at {datetime.now()}")


if __name__ == '__main__':
    main()
