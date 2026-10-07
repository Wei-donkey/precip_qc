# -*- coding: utf-8 -*-
"""
Evaluate the tuned precipitation spatial consistency QC method on a labeled dataset.

Applies outlier detection (stage 1) and the two-segment extreme inspection (stage 2: iterative
for outliers >= 20 mm, single check below) with the tuned parameters to every record of the
test set (default) or the training set, and saves the labels, confusion metrics, label counts, and
metrics per rainfall intensity bin with Wilson 95% confidence intervals.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import _bootstrap  # noqa: F401
from src.qc_data_loader import (
    load_db_config,
    create_db_engine,
    load_station_info,
    load_circles,
    fetch_all_station_precip,
    load_precip_cache,
    slice_precip_cache,
)
from src.qc_algorithms import LOW_INTENSITY_EXTREME_RADIUS
from src.qc_evaluation import (
    load_validation_data,
    qc_single_record,
    count_confusion,
    compute_confusion_metrics,
    compute_qc_label_count,
    compute_binned_metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_DIR = SCRIPT_DIR.parent / 'data'

# Configuration for neighbor data time window (hours before and after target time)
NEIGHBOR_HOR = 2

DATASET = 'test'  # evaluate on the "test" or "train" data
# Results on the training set belong to tuning; results on the test sets to validation
TABLE_DIR = SCRIPT_DIR.parent / 'outputs' / 'tables' / ('tuning' if DATASET.startswith('train') else 'validation')
TABLE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'
INPUT_VALIDATION_DATA = DATA_DIR / f"gd_validation_data_{DATASET}.csv"

OUTLIER_CIRCLES_RADIUS, EXTREME_CIRCLES_RADIUS  = 60, 80

INPUT_OUTLIER_CIRCLES = DATA_DIR / 'neighbor_circles_outlier' / f"neighbor_circles_outlier_{OUTLIER_CIRCLES_RADIUS}km.csv"
INPUT_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{EXTREME_CIRCLES_RADIUS}km.csv"
# Extreme circles of the single check for outliers below LOW_INTENSITY_THRESHOLD (two-segment method)
INPUT_LOW_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{LOW_INTENSITY_EXTREME_RADIUS}km.csv"

# Cached adjacent data (data/cache/); if missing, data are fetched from the database per record
INPUT_CACHE_CANDIDATES = [DATA_DIR / 'cache' / f"{INPUT_VALIDATION_DATA.stem}_adjacent{NEIGHBOR_HOR}h.parquet"]

OUTPUT_FILE = TABLE_DIR / f"qc_result_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km.csv"
OUTPUT_COUNT = TABLE_DIR / f"qc_count_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km.csv"
OUTPUT_STATISTICS = TABLE_DIR / f"qc_stats_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km.csv"
OUTPUT_BINNED = TABLE_DIR / f"qc_binned_{DATASET}_outlier{OUTLIER_CIRCLES_RADIUS}km_extreme{EXTREME_CIRCLES_RADIUS}km.csv"


def main() -> None:
    """Main quality control evaluation function."""
    print(f"Starting precipitation quality control evaluation at {datetime.now()}")

    print("Loading station location info...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    all_stations = df_stations['stacode']

    print(f"Loading {OUTLIER_CIRCLES_RADIUS}km outlier circles for detecting outliers...")
    df_outlier_circles = load_circles(INPUT_OUTLIER_CIRCLES)

    print(f"Loading {EXTREME_CIRCLES_RADIUS}km extreme circles for inspecting extremes...")
    df_extreme_circles = load_circles(INPUT_EXTREME_CIRCLES)
    df_low_extreme_circles = load_circles(INPUT_LOW_EXTREME_CIRCLES)

    print(f"Loading validation data from {INPUT_VALIDATION_DATA}...")
    df_validations = load_validation_data(INPUT_VALIDATION_DATA)
    total_records = len(df_validations)

    # Use cached adjacent data when available, otherwise query the database per record
    input_cache = next((f for f in INPUT_CACHE_CANDIDATES if f.exists()), None)
    if input_cache is not None:
        print(f"Loading cached adjacent precipitation from {input_cache}...")
        df_cache = load_precip_cache(input_cache)
    else:
        print("No cache found, opening persistent database connection...")
        db_config = load_db_config(CONFIG_FILE, DB_SECTION)
        engine = create_db_engine(db_config)
        conn = engine.connect()

    try:
        print(f"Processing {total_records} records...")
        for idx, record in df_validations.iterrows():
            precip = record['r']
            stacode = str(record['stacode'])
            statype = record['statype']
            ddatetime = record['ddatetime']

            print(f"QC {idx+1}/{total_records} hourly precipitation {precip} at {stacode}, {ddatetime}")

            # Fetch adjacent hours precipitation data
            time_stt = ddatetime - timedelta(hours=NEIGHBOR_HOR)
            time_end = ddatetime + timedelta(hours=NEIGHBOR_HOR)
            if input_cache is not None:
                df_all_adj = slice_precip_cache(df_cache, time_stt, time_end)
            else:
                df_all_adj = fetch_all_station_precip(conn, time_stt, time_end, all_stations)

            qc_result = qc_single_record(stacode, statype, precip, ddatetime, df_all_adj,
                                         df_outlier_circles, df_extreme_circles,
                                         df_low_extreme_circles=df_low_extreme_circles)
            for key, value in qc_result.items():
                df_validations.at[idx, key] = value

    finally:
        if input_cache is None:
            conn.close()
            engine.dispose()

    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    df_validations.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("Computing confusion metrics...")
    counts = count_confusion(df_validations)
    if counts['unprocessed']:
        print(f"Warning: {counts['unprocessed']} records have no data and were not labeled")
    series_test_result = compute_confusion_metrics(counts, total_records)
    series_test_result['outlier_circles_radius'] = OUTLIER_CIRCLES_RADIUS
    series_test_result['extreme_circles_radius'] = EXTREME_CIRCLES_RADIUS

    print(f"Saving results to {OUTPUT_STATISTICS}...")
    series_test_result.to_csv(OUTPUT_STATISTICS, index=True, header=False, encoding='utf-8-sig', float_format='%.4f')

    print("Computing QC label statistics...")
    df_label_count = compute_qc_label_count(df_validations, DATASET)

    print(f"Saving results to {OUTPUT_COUNT}...")
    df_label_count.to_csv(OUTPUT_COUNT, index=False, encoding='utf-8-sig')

    print("Computing metrics per intensity bin...")
    df_binned = compute_binned_metrics(df_validations)

    print(f"Saving results to {OUTPUT_BINNED}...")
    df_binned.to_csv(OUTPUT_BINNED, index=False, encoding='utf-8-sig', float_format='%.4f')
    print(df_binned[['subset', 'precip_bin', 'n_true', 'n_false', 'sensitivity', 'specificity']].to_string(index=False))

    print(f"\nFinished at {datetime.now()}")


if __name__ == '__main__':
    main()
