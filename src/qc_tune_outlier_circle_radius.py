# -*- coding: utf-8 -*-
"""
Test outlier detection for precipitation spatial consistency quality control.

Applies IQR-based outlier circle checks to validation records and saves
outlier percentage results for different circle parameters.
"""

from __future__ import annotations
from qc_algorithms import perform_outlier_detection
from qc_data_loader import (
    load_db_config,
    create_db_engine,
    load_station_info,
    load_circles,
    fetch_hourly_precip,
    fetch_all_station_precip,
)
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import time

SRC_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SRC_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_DIR = SRC_DIR.parent / 'data' 
CIRCLE_DIR = SRC_DIR.parent / 'data' / 'neighbor_circles_outlier'

INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'
INPUT_VALIDATION_TRAIN = DATA_DIR / 'gd_validation_data_train.csv'
OUTPUT_FILE = DATA_DIR / 'qc_compare_outlier_circles.csv'
INPUT_CIRCLES_BASENAME = 'neighbor_circles_outlier'

CIRCLE_RADIUS_KM = [10,20,30,40,50,60,70,80,90,100]

CLIMATE_LIMIT = 184.4
QC_THRESHOLD = 10  # only inspect precipitation values above this threshold


def load_validation_data(file_path: Path) -> pd.DataFrame:
    """Load validation data from CSV file."""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    # df = df.iloc[0:100]
    df['ddatetime'] = pd.to_datetime(df['ddatetime'])
    # Initialize qc_label column
    df['qc_label'] = None
    return df


def initialize_compare_results() -> pd.DataFrame:
    """Initialize a DataFrame to store comparison results."""
    df_compare_results = pd.DataFrame(index=CIRCLE_RADIUS_KM) 

    df_compare_results['circle count'] = 0
    df_compare_results['outlier count'] = 0
    df_compare_results['outlier pct'] = 0.0
    df_compare_results['total time'] = 0.0
    df_compare_results['mean time'] = 0.0

    df_compare_results['true positive'] = 0
    df_compare_results['true negative'] = 0
    df_compare_results['false positive'] = 0
    df_compare_results['false negative'] = 0

    return df_compare_results


def main() -> None:
    """Main quality control processing function."""
    print(f"Starting precipitation quality control at {datetime.now()}")

    print("Loading station location info...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    all_stations = df_stations['stacode']

    print("Loading outlier circle for detecting outliers...")
    df_circles = []
    df_compare_results = initialize_compare_results()

    for circle_radius in CIRCLE_RADIUS_KM:
        circle_file = f"{INPUT_CIRCLES_BASENAME}_{str(circle_radius)}km.csv"
        df_outlier_circles = load_circles(CIRCLE_DIR/ circle_file)
        df_circles.append(df_outlier_circles)
        df_compare_results.loc[circle_radius, 'circle count'] = len(df_outlier_circles)

    print(f"Loading validation data...")
    df_validations = load_validation_data(INPUT_VALIDATION_TRAIN)
    df_validations['qc_label'] = None  # Initialize qc_label column
    df_validations['validation_sample_size'] = None  # Initialize validation_sample_size column

    # Database setup
    print("Connecting to database...")
    db_config = load_db_config(CONFIG_FILE, DB_SECTION)
    engine = create_db_engine(db_config)
    
    total_records = len(df_validations)

    print("Opening persistent database connection...")
    with engine.connect() as conn:

        print(f"Processing {total_records} records...")
        for idx, record in df_validations.iterrows():
            precip = record['r']
            stacode = str(record['stacode'])
            ddatetime = record['ddatetime']

            print(f"QC {idx+1}/{total_records} hourly precipitation {precip} at {stacode} at {ddatetime}")
            
            valid_label_bool = bool(df_validations.at[idx, 'validation'])

            df_all_original = fetch_all_station_precip(conn, ddatetime, ddatetime, all_stations)

            for df_outlier_circles, circle_radius in zip(df_circles, CIRCLE_RADIUS_KM):
                df_all = df_all_original.copy()
                if df_all.empty:
                    continue
                
                df_all['qc_label'] = None
                df_all['validation_sample_size'] = None

                start_time = time.time()
                perform_outlier_detection(
                    df_all=df_all,
                    df_outlier_circles=df_outlier_circles
                )
                end_time = time.time()            
                elapsed_time = end_time - start_time
                df_compare_results.loc[circle_radius, 'total time'] += elapsed_time
                
                df_current = df_all[df_all['stacode'] == stacode]
                if df_current.empty:
                    continue                
                qc_label = df_current['qc_label'].iloc[0]

                if qc_label == 'OUTLIER':
                    df_compare_results.loc[circle_radius, 'outlier count'] += 1

                    if valid_label_bool:
                        df_compare_results.at[circle_radius,'false negative'] += 1
                    if not valid_label_bool:
                        df_compare_results.at[circle_radius,'true negative'] += 1

                elif qc_label != 'OUTLIER':
                    if valid_label_bool:
                        df_compare_results.at[circle_radius,'true positive'] += 1
                    if not valid_label_bool:
                        df_compare_results.at[circle_radius,'false positive'] += 1

    df_compare_results['outlier pct'] = df_compare_results['outlier count'] / total_records * 100
    df_compare_results['mean time'] = df_compare_results['total time'] / total_records

    print(f"Saving results to {OUTPUT_FILE}...")
    df_compare_results.reset_index(inplace=True)
    df_compare_results.rename(columns={'index': 'circle radius'}, inplace=True)
    df_compare_results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', float_format='%.2f')
 
    engine.dispose()
    print(f"\nFinished at {datetime.now()}")


if __name__ == '__main__':
    main()
