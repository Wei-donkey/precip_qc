# -*- coding: utf-8 -*-
"""
Inspect extreme hourly precipitation records to re-examine the climatological limit.

Applies the iterative extreme inspection (stage 2) directly to every record in
gd_extreme_data.csv (r > 100 mm), using a relaxed CLIMATE_LIMIT of 400 mm so that
neighboring records above 184.4 mm can still serve as evidence. Records confirmed as
EXTREME_TYPE1-5 indicate genuine extremes; their upper range informs the real limit.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.qc_algorithms import perform_extreme_inspection
from src.qc_data_loader import (
    load_db_config,
    create_db_engine,
    load_station_info,
    load_circles,
    fetch_all_station_precip,
    load_precip_cache,
    slice_precip_cache,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_DIR = SCRIPT_DIR.parent / 'data'
TABLE_DIR = SCRIPT_DIR.parent / 'outputs' / 'tables' / 'climate_limit'
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Configuration for neighbor data time window (hours before and after target time)
NEIGHBOR_HOR = 2

# Relaxed limit used ONLY in this script to re-examine the climatological limit (184.4 elsewhere)
CLIMATE_LIMIT = 400  # neighbors above this value are excluded from extreme inspection

LOOP_ALL_CIRCLE = True  # to loop through all extreme circles that enclose the target station

EXTREME_CIRCLES_RADIUS = 50
QC_LABELS = ['EXTREME_TYPE1', 'EXTREME_TYPE2', 'EXTREME_TYPE3', 'EXTREME_TYPE4', 'EXTREME_TYPE5', 'FALSE']
PRECIP_BINS = [100, 150, 184.4, 250, 400, float('inf')]
PRECIP_BIN_LABELS = ['100-150', '150-184.4', '184.4-250', '250-400', '>400']

INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'
INPUT_EXTREME_DATA = DATA_DIR / 'gd_extreme_data.csv'
INPUT_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{EXTREME_CIRCLES_RADIUS}km.csv"
# Cached neighbor rainfall (data/cache/); if missing, data are fetched from the database per record
INPUT_CACHE = DATA_DIR / 'cache' / f"{INPUT_EXTREME_DATA.stem}_adjacent{NEIGHBOR_HOR}h.parquet"

OUTPUT_FILE = TABLE_DIR / f"qc_climate_limit_result_extreme{EXTREME_CIRCLES_RADIUS}km_limit{CLIMATE_LIMIT}.csv"
OUTPUT_COUNT = TABLE_DIR / f"qc_climate_limit_count_extreme{EXTREME_CIRCLES_RADIUS}km_limit{CLIMATE_LIMIT}.csv"


def load_extreme_data(file_path: Path) -> pd.DataFrame:
    """Load extreme records from CSV file."""
    df = pd.read_csv(file_path, encoding='utf-8-sig', dtype={'stacode': str})
    df['ddatetime'] = pd.to_datetime(df['ddatetime'])

    # Initialize new columns for results
    df['qc_label'] = None
    df['validation_sample_size'] = None
    df['extreme_circle_count'] = None
    df['validation_circle_count'] = None
    df['validation_circle_locs'] = None

    return df


def compute_qc_label_count(df_qc_result: pd.DataFrame) -> pd.DataFrame:
    """
    Compute counts of each qc_label for each station type and precipitation bin.
    """
    df_tmp = df_qc_result.copy()
    df_tmp['precip_bin'] = pd.cut(df_tmp['r'], bins=PRECIP_BINS, labels=PRECIP_BIN_LABELS)

    df_label_count = (
        df_tmp.groupby(['statype', 'precip_bin', 'qc_label'], observed=False)
        .size()
        .unstack('qc_label', fill_value=0)
        .reindex(columns=QC_LABELS, fill_value=0)
    )
    df_label_count['EXTREME_TOTAL'] = df_label_count[QC_LABELS[:-1]].sum(axis=1)
    df_label_count['TOTAL'] = df_label_count['EXTREME_TOTAL'] + df_label_count['FALSE']

    return df_label_count.reset_index()


def main() -> None:
    """Main extreme inspection function."""
    print(f"Starting climatological limit inspection at {datetime.now()}")

    print("Loading station location info...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    all_stations = df_stations['stacode']

    print(f"Loading {EXTREME_CIRCLES_RADIUS}km extreme circles for inspecting extremes...")
    df_extreme_circles = load_circles(INPUT_EXTREME_CIRCLES)

    print("Loading extreme data...")
    df_extremes = load_extreme_data(INPUT_EXTREME_DATA)
    total_records = len(df_extremes)

    # Use cached adjacent data when available, otherwise query the database per record
    use_cache = INPUT_CACHE.exists()
    if use_cache:
        print(f"Loading cached adjacent precipitation from {INPUT_CACHE}...")
        df_cache = load_precip_cache(INPUT_CACHE)
    else:
        print(f"Cache {INPUT_CACHE} not found, connecting to database...")
        db_config = load_db_config(CONFIG_FILE, DB_SECTION)
        engine = create_db_engine(db_config)
        conn = engine.connect()

    try:
        print(f"Processing {total_records} records...")
        for idx, record in df_extremes.iterrows():
            precip = record['r']
            stacode = str(record['stacode'])
            statype = record['statype']
            ddatetime = record['ddatetime']

            print(f"QC {idx+1}/{total_records} hourly precipitation {precip} at {stacode}, {ddatetime}")

            # Fetch adjacent hours precipitation data
            time_stt = ddatetime - timedelta(hours=NEIGHBOR_HOR)
            time_end = ddatetime + timedelta(hours=NEIGHBOR_HOR)
            if use_cache:
                df_all_adj = slice_precip_cache(df_cache, time_stt, time_end)
            else:
                df_all_adj = fetch_all_station_precip(conn, time_stt, time_end, all_stations)

            if df_all_adj.empty:
                continue

            # Find extreme circles containing this station
            extreme_circles_mask = df_extreme_circles['neighbors'].apply(lambda neighbors: stacode in neighbors)
            filtered_extreme_circles = df_extreme_circles[extreme_circles_mask]

            # Perform extreme circle evaluation directly (stage 2 only)
            qc_label, validation_sample_size, \
                extreme_circle_count, validation_circle_count, validation_circle_locs \
                = perform_extreme_inspection(
                target_stacode=stacode,
                target_statype=statype,
                target_precip=precip,
                filtered_extreme_circles=filtered_extreme_circles,
                df_all_adjacent=df_all_adj,
                loop_all_circle=LOOP_ALL_CIRCLE,
                climate_limit=CLIMATE_LIMIT,
            )

            df_extremes.at[idx, 'qc_label'] = qc_label
            df_extremes.at[idx, 'validation_sample_size'] = validation_sample_size
            df_extremes.at[idx, 'extreme_circle_count'] = extreme_circle_count
            df_extremes.at[idx, 'validation_circle_count'] = validation_circle_count
            df_extremes.at[idx, 'validation_circle_locs'] = validation_circle_locs

    finally:
        if not use_cache:
            conn.close()
            engine.dispose()

    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    df_extremes.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("Computing QC label statistics...")
    df_label_count = compute_qc_label_count(df_extremes)

    print(f"Saving results to {OUTPUT_COUNT}...")
    df_label_count.to_csv(OUTPUT_COUNT, index=False, encoding='utf-8-sig')
    print(df_label_count.to_string(index=False))

    # Largest records confirmed as genuine extremes, per station type
    df_confirmed = df_extremes[df_extremes['qc_label'].isin(QC_LABELS[:-1])]
    print("\nTop 10 confirmed extremes:")
    print(df_confirmed.nlargest(10, 'r')[['stacode', 'statype', 'ddatetime', 'r', 'qc_label', 'validation_circle_count']].to_string(index=False))

    print(f"\nFinished at {datetime.now()}")


if __name__ == '__main__':
    main()
