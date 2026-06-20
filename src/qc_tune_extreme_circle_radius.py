# -*- coding: utf-8 -*-
"""
Test extreme evaluation for precipitation spatial consistency quality control.

Applies outlier detection and extreme circle checks with temporal adjacency
to classify events and save false positive/negative results for analysis.
"""

from __future__ import annotations
from qc_algorithms import perform_outlier_detection, perform_extreme_inspection
from qc_data_loader import (
    load_db_config,
    create_db_engine,
    load_station_info,
    load_circles,
    fetch_all_station_precip,
)
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SRC_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_DIR = SRC_DIR.parent / 'data'
CIRCLE_OUTLIER_DIR = SRC_DIR.parent / 'data' / 'neighbor_circles_outlier'
CIRCLE_EXTREME_DIR = SRC_DIR.parent / 'data' / 'neighbor_circles_extreme'

# Configuration for neighbor data time window (hours before and after target time)
NEIGHBOR_HOR = 2

INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'
INPUT_VALIDATION = DATA_DIR / 'gd_validation_data_train.csv'

OUTLIER_CIRCLE_RADIUS = [60,70,80,90,100]

INPUT_OUTLIER_CIRCLES_BASENAME = 'neighbor_circles_outlier'
INPUT_EXTREME_CIRCLES_BASENAME = 'neighbor_circles_extreme'

EXTREME_CIRCLE_RADIUS = [10,20,30,40,50,60,70,80,90,100]

OUTPUT_FILE_BASENAME = 'qc_compare_extreme_circles_outlier'

OUTPUT_FILE_FP_FN_SWITCH = False  # whether to save false positive and false negative records to a separate file
OUTPUT_FILE_FP_FN_BASENAME = 'qc_compare_extreme_circles_outlier_fp_fn'

CLIMATE_LIMIT = 184.4
QC_THRESHOLD = 10  # only inspect precipitation values above this threshold


def initialize_test_results() -> pd.DataFrame:
    """Initialize a DataFrame to store test results."""
    df_test_results = pd.DataFrame(index=EXTREME_CIRCLE_RADIUS) 
    df_test_results['false positive'] = 0
    df_test_results['false negative'] = 0
    df_test_results['true positive'] = 0
    df_test_results['true negative'] = 0

    df_test_results['accuracy'] = 0
    df_test_results['precision'] = 0
    df_test_results['sensitivity'] = 0
    df_test_results['F1 score'] = 0  

    return df_test_results


def load_validation_data(file_path: Path) -> pd.DataFrame:
    """Load validation data from CSV file."""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df['ddatetime'] = pd.to_datetime(df['ddatetime'])

    # Initialize new columns for results
    df['qc_label'] = None
    df['validation_sample_size'] = None
    df['extreme_circle_count'] = None
    df['validation_circle_count'] = None
    df['validation_circle_locs'] = None

    return df  #.iloc[0:20]


def load_all_extreme_circles() -> list[pd.DataFrame]:
    lst_df_extreme_circles = [] 
    for extreme_circle in EXTREME_CIRCLE_RADIUS:
        circle_file = f"{INPUT_EXTREME_CIRCLES_BASENAME}_{str(extreme_circle)}km.csv"
        df_extreme_circles = load_circles(CIRCLE_EXTREME_DIR/ circle_file)
        lst_df_extreme_circles.append(df_extreme_circles)

    return lst_df_extreme_circles

def main() -> None:
    """Main quality control processing function."""
    print(f"Starting precipitation quality control at {datetime.now()}")

    print("Loading station location info...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    all_stations = df_stations['stacode']

    # Step 1: Load circle neighbor information
    print("Loading extreme circle neighbors for inspecting extremes...")
    lst_df_extreme_circles = load_all_extreme_circles()

    for outlier_circle in OUTLIER_CIRCLE_RADIUS:

        # Initialize list to store false positive and false negative records
        lst_fp_fn = []

        # Initialize test results
        df_test_results = initialize_test_results() 

        print(f"Loading {outlier_circle}km-radius outlier circles for detecting outliers...")
        circle_file = f"{INPUT_OUTLIER_CIRCLES_BASENAME}_{str(outlier_circle)}km.csv"
        df_outlier_circles = load_circles(CIRCLE_OUTLIER_DIR / circle_file)

        output_file = DATA_DIR / f"{OUTPUT_FILE_BASENAME}{outlier_circle}km.csv"
        output_file_fp_fn = DATA_DIR / f"{OUTPUT_FILE_FP_FN_BASENAME}{outlier_circle}km.csv"

        # Step 2: Load validation data and sort by datetime descending
        print(f"Loading validation data...")
        df_validations = load_validation_data(INPUT_VALIDATION)

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
                statype = record['statype']
                ddatetime = record['ddatetime']

                print(f"QC {idx+1}/{total_records} hourly precipitation {precip} at {stacode} at {ddatetime}")

                # Fetch adjacent hours precipitation data
                df_all_adj = fetch_all_station_precip(conn, ddatetime - timedelta(hours=NEIGHBOR_HOR), ddatetime + timedelta(hours=NEIGHBOR_HOR), all_stations)

                # Fetch current hour precipitation data for all stations
                df_all = df_all_adj[df_all_adj['ddatetime'] == ddatetime].copy()
                
                if df_all.empty:
                    continue
                
                df_all['qc_label'] = None
                df_all['validation_sample_size'] = None

                # Perform outlier circle inspection on all station data
                perform_outlier_detection(
                    df_all=df_all,
                    df_outlier_circles=df_outlier_circles
                )
                
                df_current = df_all[df_all['stacode'] == stacode]
                if df_current.empty:
                    continue                
                qc_label = df_current['qc_label'].iloc[0]
                
                if qc_label == 'NORMAL':

                    for extreme_circle in EXTREME_CIRCLE_RADIUS:              
                        valid_label_bool = bool(df_validations.at[idx, 'validation'])
                        if valid_label_bool:
                            df_test_results.at[extreme_circle,'true positive'] += 1
                        if not valid_label_bool:
                            df_test_results.at[extreme_circle,'false positive'] += 1      

                elif qc_label == 'OUTLIER':
                    for df_extreme_circles, extreme_circle in zip(lst_df_extreme_circles, EXTREME_CIRCLE_RADIUS):    

                        # Find extreme circles containing this outlier station
                        extreme_circles_mask = df_extreme_circles['neighbors'].apply(lambda neighbors: stacode in neighbors)
                        filtered_extreme_circles = df_extreme_circles[extreme_circles_mask]

                        # Perform extreme circle evaluation for OUTLIERS
                        qc_label, validation_sample_size, \
                        extreme_circle_count, validation_circle_count, validation_Circle_locs \
                        = perform_extreme_inspection(
                            target_stacode=stacode, 
                            target_statype=statype,
                            target_precip=precip,
                            filtered_extreme_circles=filtered_extreme_circles,
                            df_all_adjacent=df_all_adj,
                        )

                        valid_label_bool = bool(df_validations.at[idx, 'validation'])
                        if qc_label == 'FALSE' and valid_label_bool:
                            df_test_results.at[extreme_circle,'false negative'] += 1
                        if qc_label == 'FALSE' and not valid_label_bool:
                            df_test_results.at[extreme_circle,'true negative'] += 1

                        if qc_label != 'FALSE' and valid_label_bool:
                            df_test_results.at[extreme_circle,'true positive'] += 1
                        if qc_label != 'FALSE' and not valid_label_bool:
                            df_test_results.at[extreme_circle,'false positive'] += 1

                        # Collect False Positive & False Negative records
                        if (qc_label == 'FALSE' and valid_label_bool) | (qc_label != 'FALSE' and not valid_label_bool):
                            df_validations.at[idx, 'qc_label'] = qc_label
                            df_validations.at[idx, 'validation_sample_size'] = validation_sample_size
                            df_tmp = df_validations.iloc[idx].to_dict()
                            df_tmp['radius'] = extreme_circle
                            lst_fp_fn.append(df_tmp)

        engine.dispose()

        df_test_results['accuracy'] = (df_test_results['true positive'] + df_test_results['true negative']) / total_records
        df_test_results['precision'] = df_test_results['true positive'] / (df_test_results['true positive'] + df_test_results['false positive'])
        df_test_results['sensitivity'] = df_test_results['true positive'] / (df_test_results['true positive'] + df_test_results['false negative'])
        df_test_results['F1 score'] = 2 * df_test_results['precision'] * df_test_results['sensitivity'] / (df_test_results['precision'] + df_test_results['sensitivity'])

        print(f"Saving results to {output_file}...")
        df_test_results.reset_index(inplace=True)
        df_test_results.rename(columns={'index': 'extreme_circles_radius'}, inplace=True)
        df_test_results.to_csv(output_file, index=False, encoding='utf-8-sig', float_format='%.2f')

        if OUTPUT_FILE_FP_FN_SWITCH:
            print(f"Saving false positive and false negative records to {output_file_fp_fn}...")
            df_fp_fn = pd.DataFrame(lst_fp_fn)
            df_fp_fn.to_csv(output_file_fp_fn,index=False, encoding='utf-8-sig', float_format='%.2f')

    print(f"\nFinished at {datetime.now()}")


if __name__ == '__main__':
    main()
