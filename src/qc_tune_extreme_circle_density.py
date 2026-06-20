# -*- coding: utf-8 -*-
"""
Evaluate the impact of extreme circle density on QC performance.

Tests strategies using single circle, fewer circles (outlier substitutes),
and all available extreme circles with varying density levels.
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
from qc_circle_process_utils import find_single_extreme_circle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SRC_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_DIR = SRC_DIR.parent / 'data'

# Configuration for neighbor data time window (hours before and after target time)
NEIGHBOR_HOR = 2

CLIMATE_LIMIT = 184.4
QC_THRESHOLD = 10  # only inspect precipitation values above this threshold
OUTLIER_CIRCLES_RADIUS, EXTREME_CIRCLES_RADIUS  = 60, 80

# Test mode configuration - choose one: 'single', 'few', or 'many'
TEST_MODE = 'many' 

LOOP_ALL_CIRCLE = True  # to loop through all extreme circles (or the single one) that enclose the outlier station

DATASET = 'train'  # run this script on "train" data or "test" data.
QC_LABELS = ['EXTREME_TYPE1', 'EXTREME_TYPE2', 'EXTREME_TYPE3', 'EXTREME_TYPE4', 'EXTREME_TYPE5', 'FALSE', 'NORMAL']

INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'
INPUT_VALIDATION_DATA = DATA_DIR / f"gd_validation_data_{DATASET}.csv"

INPUT_OUTLIER_CIRCLES = DATA_DIR / 'neighbor_circles_outlier' / f"neighbor_circles_outlier_{OUTLIER_CIRCLES_RADIUS}km.csv"

# Configuration based on test mode
if TEST_MODE == 'single':
    INPUT_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{EXTREME_CIRCLES_RADIUS}km.csv"
    OUTPUT_FILE = DATA_DIR / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_single.csv"
    OUTPUT_COUNT = DATA_DIR / f"qc_count_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_single.csv"
    OUTPUT_STATISTICS = DATA_DIR / f"qc_stats_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_single.csv"

elif TEST_MODE == 'few':
    # adopt the outlier circles to substitute extreme circles because outlier circles have larger gaps, thereby few circles
    INPUT_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_outlier' / f"neighbor_circles_outlier_{EXTREME_CIRCLES_RADIUS}km.csv"
    OUTPUT_FILE = DATA_DIR / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_few.csv"
    OUTPUT_COUNT = DATA_DIR / f"qc_count_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_few.csv"
    OUTPUT_STATISTICS = DATA_DIR / f"qc_stats_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_few.csv"

elif TEST_MODE == 'many':
    INPUT_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{EXTREME_CIRCLES_RADIUS}km.csv"
    OUTPUT_FILE = DATA_DIR / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_many.csv"
    OUTPUT_COUNT = DATA_DIR / f"qc_count_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_many.csv"
    OUTPUT_STATISTICS = DATA_DIR / f"qc_stats_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km_many.csv"

else:
    raise ValueError(f"Invalid TEST_MODE: '{TEST_MODE}'. Must be 'single', 'few', or 'many'")

def initialize_test_result() -> pd.DataFrame:
    """Initialize a DataFrame to store test results."""
    
    series_test_result = pd.Series()
    series_test_result['outlier_circles_radius'] = OUTLIER_CIRCLES_RADIUS
    series_test_result['extreme_circles_radius'] = EXTREME_CIRCLES_RADIUS

    series_test_result['false positive'] = 0
    series_test_result['false negative'] = 0
    series_test_result['true positive'] = 0
    series_test_result['true negative'] = 0

    series_test_result['accuracy'] = 0.0
    series_test_result['precision'] = 0.0
    series_test_result['sensitivity'] = 0.0
    series_test_result['F1 score'] = 0.0

    return series_test_result


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

    return df


def compute_confusion_metrics(series_test_result: pd.Series, total_records: int) -> pd.Series:
    series_tmp = series_test_result.copy()
    series_tmp['accuracy'] = (series_tmp['true positive'] + series_tmp['true negative']) / total_records
    series_tmp['precision'] = series_tmp['true positive'] / (series_tmp['true positive'] + series_tmp['false positive'])
    series_tmp['sensitivity'] = series_tmp['true positive'] / (series_tmp['true positive'] + series_tmp['false negative'])
    series_tmp['F1 score'] = 2 * series_tmp['precision'] * series_tmp['sensitivity'] / (series_tmp['precision'] + series_tmp['sensitivity'])

    return series_tmp


def compute_qc_label_count(df_qc_result: pd.DataFrame) -> pd.DataFrame:
    """
    Compute counts of each qc_label for TRUE/FALSE validation status.
    """
    # Initialize result structure
    columns = ['Dataset', 'Actual'] + QC_LABELS
    result_rows = []

    # Process each validation status
    for validation_status in [True, False]:
        df_subset = df_qc_result[df_qc_result['validation'] == validation_status]
        
        # Initialize counts
        counts = {label: 0 for label in QC_LABELS}
        
        # Count each qc_label type
        for qc_label in df_subset['qc_label'].unique():
            count = len(df_subset[df_subset['qc_label'] == qc_label])

            if qc_label in QC_LABELS:
                counts[qc_label] = count
        # Create row
        row = {
            'Dataset': DATASET,
            'Actual': 'TRUE' if validation_status else 'FALSE',
            **counts
        }
        result_rows.append(row)
    
    # Create result DataFrame
    df_label_count = pd.DataFrame(result_rows, columns=columns)
    
    return df_label_count


def main() -> None:
    """Main quality control processing function."""
    print(f"Starting precipitation quality control at {datetime.now()}")
    
    print("Loading station location info...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    all_stations = df_stations['stacode']

    print(f"Loading {OUTLIER_CIRCLES_RADIUS}km outlier circles for detecting outliers...")
    df_outlier_circles = load_circles(INPUT_OUTLIER_CIRCLES)
    
    print(f"Loading {EXTREME_CIRCLES_RADIUS}km extreme circles for inspecting extremes...")
    df_extreme_circles = load_circles(INPUT_EXTREME_CIRCLES)

    series_test_result = initialize_test_result() 
    
    print(f"Loading validation data...")
    df_validations = load_validation_data(INPUT_VALIDATION_DATA)

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

            print(f"QC {idx+1}/{total_records} hourly precipitation {precip} at {stacode}, {ddatetime}")

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
                df_validations.at[idx, 'qc_label'] = 'NORMAL'
                df_validations.at[idx, 'validation_sample_size'] = df_current['validation_sample_size'].iloc[0]

                df_validations.at[idx, 'extreme_circle_count'] = 0
                df_validations.at[idx, 'validation_circle_count'] = 0
                df_validations.at[idx, 'validation_circle_locs'] = None 

                valid_label_bool = bool(df_validations.at[idx, 'validation'])
                if valid_label_bool:
                    series_test_result['true positive'] += 1
                if not valid_label_bool:
                    series_test_result['false positive'] += 1      

            elif qc_label == 'OUTLIER':

                # Find extreme circles containing this outlier station
                extreme_circles_mask = df_extreme_circles['neighbors'].apply(lambda neighbors: stacode in neighbors)
                filtered_extreme_circles = df_extreme_circles[extreme_circles_mask]


                if TEST_MODE == 'single':
                    filtered_extreme_circles = find_single_extreme_circle(
                        extreme_circles=filtered_extreme_circles,
                        target_stacode=stacode,
                        df_station_info=df_stations,
                        )


                # Perform extreme circle evaluation for OUTLIERS
                qc_label, validation_sample_size, \
                    extreme_circle_count, validation_circle_count, validation_circle_locs \
                    = perform_extreme_inspection(
                    target_stacode=stacode,
                    target_statype=statype,
                    target_precip=precip,
                    filtered_extreme_circles=filtered_extreme_circles,
                    df_all_adjacent=df_all_adj,
                    loop_all_circle=LOOP_ALL_CIRCLE
                )

                df_validations.at[idx, 'qc_label'] = qc_label
                df_validations.at[idx, 'validation_sample_size'] = validation_sample_size
                df_validations.at[idx, 'extreme_circle_count'] = extreme_circle_count
                df_validations.at[idx, 'validation_circle_count'] = validation_circle_count
                df_validations.at[idx, 'validation_circle_locs'] = validation_circle_locs

                valid_label_bool = bool(df_validations.at[idx, 'validation'])
                if qc_label == 'FALSE' and valid_label_bool:
                    series_test_result['false negative'] += 1
                if qc_label == 'FALSE' and not valid_label_bool:
                    series_test_result['true negative'] += 1

                if qc_label != 'FALSE' and valid_label_bool:
                    series_test_result['true positive'] += 1
                if qc_label != 'FALSE' and not valid_label_bool:
                    series_test_result['false positive'] += 1

    engine.dispose()

    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    df_validations.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("Computing confusion metrics...")
    series_test_result = compute_confusion_metrics(series_test_result, total_records)

    print(f"Saving results to {OUTPUT_STATISTICS}...")
    series_test_result.to_csv(OUTPUT_STATISTICS, index=True, header=False, encoding='utf-8-sig', float_format='%.2f')

    print("Computing QC label statistics...")
    df_label_count = compute_qc_label_count(df_validations)

    print(f"Saving results to {OUTPUT_COUNT}...")
    df_label_count.to_csv(OUTPUT_COUNT, index=False, encoding='utf-8-sig')

    print(f"\nFinished at {datetime.now()}")


if __name__ == '__main__':
    main()
