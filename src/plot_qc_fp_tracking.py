# -*- coding: utf-8 -*-
"""
Track and plot false positive cases from QC validation results.

Visualizes precipitation events where FALSE values were incorrectly classified as TRUE,
creating figures with extreme circle maps and precipitation histograms.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import rioxarray

# Import common utilities
from qc_plot_utils import (
    plot_rainfall_event,
)

# Import circle processing utilities
from qc_circle_process_utils import (
    find_single_extreme_circle,
    compute_single_circle_extent,
    compute_few_circles_extent,
    compute_many_circles_extent,
    get_circles_for_station,
    construct_validation_circles,
)

# Import data loading utilities
from qc_data_loader import (
    load_db_config,
    create_db_engine,
    load_station_info,
    load_circles,
    fetch_all_station_precip,
    load_qc_result,
)

SRC_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SRC_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_DIR = SRC_DIR.parent / 'data'

# Configuration for neighbor data time window (hours before and after target time)
NEIGHBOR_HOR = 2

DATASET = 'train'

CLIMATE_LIMIT = 184.4
OUTLIER_CIRCLES_RADIUS, EXTREME_CIRCLES_RADIUS  = 60, 80

INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'
DEM_FILE = DATA_DIR / 'external' / 'gd_dem_1km.tif'

INPUT_FINE_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{EXTREME_CIRCLES_RADIUS}km.csv"
INPUT_COARSE_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_outlier' / f"neighbor_circles_outlier_{EXTREME_CIRCLES_RADIUS}km.csv"

INPUT_FILE_SINGLE = DATA_DIR / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_single.csv"
INPUT_FILE_FEW = DATA_DIR / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_few.csv"
INPUT_FILE_MORE = DATA_DIR / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_many.csv"

OUTPUT_DIR = SRC_DIR.parent / 'figures' / f"qc_false_tracking@aws_{DATASET}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_qc_rec(input_file: Path, df_qc_many: pd.DataFrame):
    """
    Load and filter records from INPUT_FILE_MORE that match the false positive cases 
    from the many circle experiment.
    """
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    df['ddatetime'] = pd.to_datetime(df['ddatetime'])
    
    # Create merge keys from df_qc_fn_single
    df_fn_keys = df_qc_many[['stacode', 'ddatetime']].copy()
    df_fn_keys['stacode'] = df_fn_keys['stacode'].astype(str)

    df_matched = pd.merge(df, df_fn_keys, on=['stacode', 'ddatetime'], how='inner' )

    mask = (df_matched['validation'] == False) & (df_matched['qc_label'] == 'FALSE')
    df_matched.loc[mask,'confusion_type'] = 'True negative'

    mask = (df_matched['validation'] == False) & (df_matched['qc_label'] != 'FALSE')
    df_matched.loc[mask,'confusion_type'] = 'False positive'
    
    return df_matched
    

def plot_qc_result(conn, total_records, all_stations, df_stations, flag: str,
                   df_qc_result: pd.DataFrame, df_extreme_circles, raster_dem):

    for idx, record in df_qc_result.iterrows():
        target_stacode = str(record['stacode'])
        ddatetime = record['ddatetime']
        target_precip = float(record['r'])
        confusion_type = record['confusion_type']
        if confusion_type == 'True negative':
            label = 'tn'
        elif confusion_type == 'False positive':
            label = 'fp'

        print(f"Plotting map {idx+1}/{total_records} precipitation {target_precip} at {target_stacode}, {ddatetime}")

        # Fetch adjacent hours precipitation data
        df_all_adj = fetch_all_station_precip(conn, ddatetime - timedelta(hours=NEIGHBOR_HOR), ddatetime + timedelta(hours=NEIGHBOR_HOR), all_stations)
        df_all_adj = df_all_adj[df_all_adj['r']<=CLIMATE_LIMIT]

        # Fetch current hour precipitation data for all stations
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

        if flag == 'many':
            map_extent = compute_many_circles_extent(df_circles)
        elif flag == 'few':
            map_extent = compute_few_circles_extent(df_circles)
        elif flag == 'single':
            map_extent = compute_single_circle_extent(df_circles)
        output_file = OUTPUT_DIR / f"qc_{target_stacode}_{ddatetime.strftime('%Y%m%d%H')}_{str(target_precip)}_{label}@{flag}.png"

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

    print("Loading DEM data...")
    raster_dem = rioxarray.open_rasterio(DEM_FILE)

    print("Loading station location info...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    all_stations = df_stations['stacode']
    
    print("Loading samll-gap circles for inspecting extremes...")
    df_fine_extreme_circles = load_circles(INPUT_FINE_EXTREME_CIRCLES)

    print("Loading big-step circles for inspecting extremes...")
    df_coarse_extreme_circles = load_circles(INPUT_COARSE_EXTREME_CIRCLES)

    print("Loading qc predicted result...")
    df_qc_many = load_qc_result(INPUT_FILE_MORE,qc_type='false positive')
    df_qc_few = load_qc_rec(INPUT_FILE_FEW, df_qc_many)
    df_qc_single = load_qc_rec(INPUT_FILE_SINGLE, df_qc_many)

    print("Connecting to database...")
    db_config = load_db_config(CONFIG_FILE, DB_SECTION)
    engine = create_db_engine(db_config)

    total_records = len(df_qc_single) 
    print("Opening persistent database connection...")
    with engine.connect() as conn:

        print(f"Plotting {total_records} false qc cases @ many circles...")
        plot_qc_result(conn, total_records, all_stations, df_stations, 'many',
                       df_qc_many, df_fine_extreme_circles, raster_dem)

        print(f"Plotting {total_records} same qc cases @ few circles...")        
        plot_qc_result(conn, total_records, all_stations, df_stations, 'few',
                       df_qc_few, df_coarse_extreme_circles, raster_dem)
        
        print(f"Plotting {total_records} same qc cases @ many circles...")        
        plot_qc_result(conn, total_records, all_stations, df_stations, 'single',
                       df_qc_single, df_fine_extreme_circles, raster_dem)
        
    engine.dispose()

    print(f"\nFinished at {datetime.now()}")


if __name__ == '__main__':
    main()
