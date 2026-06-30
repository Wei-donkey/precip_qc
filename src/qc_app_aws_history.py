# -*- coding: utf-8 -*-
"""
Apply full QC validation to all AWST precipitation data (2003-2025).
Processes blacklist entries and performs outlier/extreme circle checks,
saving flagged records to AWST_CLI_PRE_HOR_QC table.
"""

from __future__ import annotations

import configparser
import gc
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from sqlalchemy import create_engine, text

from qc_algorithms import perform_outlier_detection, perform_extreme_inspection
from qc_data_loader import load_db_config, load_station_info, load_circles

SRC_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SRC_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_DIR = SRC_DIR.parent / 'data'

# Time range
YEAR_START, YEAR_END = 2003, 2025

# Switch: do not set to True during testing
UPDATE_FALSE_TO_NULL = False

QC_TABLE_NAME = 'AWST_CLI_PRE_HOR_QC'
DATA_TABLE_NAME = 'AWST_CLI_MUL_HOR'

# Neighbor time window
NEIGHBOR_HOUR_OFFSET = 2

# Circle files
OUTLIER_CIRCLES_RADIUS = 60
EXTREME_CIRCLES_RADIUS = 80
INPUT_OUTLIER_CIRCLES = DATA_DIR / 'neighbor_circles_outlier' / f"neighbor_circles_outlier_{OUTLIER_CIRCLES_RADIUS}km.csv"
INPUT_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{EXTREME_CIRCLES_RADIUS}km.csv"

# Labels to save (exclude EXTREME_TYPE1)
QC_LABELS_TO_SAVE = ['EXTREME_TYPE2', 'EXTREME_TYPE3', 'EXTREME_TYPE4', 'EXTREME_TYPE5', 'FALSE']

# Label mapping for database
LABEL_MAPPING = {
    'EXTREME_TYPE2': 'X_TYPE2',
    'EXTREME_TYPE3': 'X_TYPE3',
    'EXTREME_TYPE4': 'X_TYPE4',
    'EXTREME_TYPE5': 'X_TYPE5',
    'FALSE': 'FALSE'
}

# All QC labels for statistics
ALL_QC_LABELS = ['EXTREME_TYPE1', 'EXTREME_TYPE2', 'EXTREME_TYPE3', 
                 'EXTREME_TYPE4', 'EXTREME_TYPE5', 'FALSE', 'NORMAL']

# Input files
INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'

RESUME_QC_PROCESS = False
# Output file for monthly statistics
OUTPUT_MONTHLY_STATS = DATA_DIR / f"qc_monthly_statistics_{str(YEAR_START)}-{str(YEAR_END)}.csv"
if not RESUME_QC_PROCESS:
    df_stats_header = pd.DataFrame(columns=['year', 'month'] + ALL_QC_LABELS)
    df_stats_header.to_csv(OUTPUT_MONTHLY_STATS, index=False)


def create_db_engine(db_config: dict[str, str]):
    """Create SQLAlchemy engine for Oracle database connection."""
    password = quote(db_config['password'])
    conn_string = (f"oracle+oracledb://{db_config['user']}:{password}@{db_config['host']}"
                   f":{db_config['port']}/{db_config['service']}")
    return create_engine(conn_string, echo=False)


def check_record_in_qc_table(conn, stacode: str, ddatetime: datetime) -> bool:
    """Check if record exists in AWST_CLI_PRE_HOR_QC table."""
    sql = f"SELECT COUNT(*) AS cnt FROM {QC_TABLE_NAME} WHERE STACODE = '{stacode}' AND DDATETIME = TO_DATE('{ddatetime.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS')"
    df = pd.read_sql(sql, conn)
    return df['cnt'].iloc[0] > 0


def fetch_monthly_data(conn, year: int, month: int, valid_stations: list) -> pd.DataFrame:
    """Fetch AWST and SURF data for current month ±2 hours."""
    # Calculate time window
    time_start = datetime(year, month, 1, 0) - timedelta(hours=2)
    
    if month == 12:
        time_end = datetime(year + 1, 1, 1, 1) 
    else:
        time_end = datetime(year, month+1, 1, 1)
    
    all_data = []
    
    # Fetch AWST data
    for yr in range(time_start.year, time_end.year + 1):
        table_name = f"{DATA_TABLE_NAME}_{yr}"
        sql = (
            f"SELECT STACODE, DDATETIME, R FROM {table_name} "
            f"WHERE DDATETIME >= TO_DATE('{time_start.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS') "
            f"AND DDATETIME <= TO_DATE('{time_end.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS')"
        )
        df = pd.read_sql(sql, conn)
        if not df.empty:
            df['STATYPE'] = 'AWST'
            all_data.append(df)

    # Fetch SURF data
    for yr in range(time_start.year, time_end.year + 1):
        table_name = f"SURF_CLI_MUL_HOR_{yr}"
        sql = (
            f"SELECT STACODE, DDATETIME, R FROM {table_name} "
            f"WHERE DDATETIME >= TO_DATE('{time_start.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS') "
            f"AND DDATETIME <= TO_DATE('{time_end.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS')"
        )
        df = pd.read_sql(sql, conn)
        if not df.empty:
            df['STATYPE'] = 'SURF'
            all_data.append(df)
    
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Convert column names to lowercase for consistency
    df_combined.columns = [col.lower() for col in df_combined.columns]
    df_combined['ddatetime'] = pd.to_datetime(df_combined['ddatetime'])
    
    # Filter to valid stations and non-null precipitation
    df_filtered = df_combined[
        (df_combined['stacode'].isin(valid_stations)) & 
        (df_combined['r'].notna())
    ].copy()
    
    return df_filtered


def perform_hourly_qc(hour: pd.Timestamp, df_monthly: pd.DataFrame,
                     df_outlier_circles: pd.DataFrame, 
                     df_extreme_circles: pd.DataFrame) -> list[dict]:
    """Perform QC on a single hour's data."""
    # Extract current hour and adjacent data
    time_start = hour - timedelta(hours=NEIGHBOR_HOUR_OFFSET)
    time_end = hour + timedelta(hours=NEIGHBOR_HOUR_OFFSET)
    
    df_current = df_monthly[df_monthly['ddatetime'] == hour].copy()
    df_adjacent = df_monthly[
        (df_monthly['ddatetime'] >= time_start) & 
        (df_monthly['ddatetime'] <= time_end)
    ].copy()
    
    # Initialize qc_label column
    df_current['qc_label'] = None
    df_current['validation_sample_size'] = None
    
    # Perform outlier detection
    perform_outlier_detection(df_current, df_outlier_circles)
    
    results = []
    
    # Process each record - ONLY process AWST stations
    for _, record in df_current.iterrows():
        # Skip SURF stations entirely - they're only used as neighbors
        if record['statype'] != 'AWST':
            continue
        
        qc_label = record['qc_label']
        
        # Perform extreme inspection for AWST outliers
        if qc_label == 'OUTLIER':
            # Find extreme circles containing this station
            stacode = record['stacode']
            extreme_circles_mask = df_extreme_circles['neighbors'].apply(
                lambda neighbors: stacode in neighbors
            )
            filtered_extreme_circles = df_extreme_circles[extreme_circles_mask]
            
            # Perform extreme inspection regardless of whether circles exist
            # If no circles contain this station, it will be labeled as FALSE
            qc_label, _, _, _, _ = perform_extreme_inspection(
                target_stacode=stacode,
                target_statype='AWST',
                target_precip=record['r'],
                filtered_extreme_circles=filtered_extreme_circles,
                df_all_adjacent=df_adjacent,
                loop_all_circle=False  # Production setting
            )
        
        results.append({
            'stacode': record['stacode'],
            'ddatetime': record['ddatetime'],
            'r': record['r'],
            'qc_label': qc_label
        })
    
    return results


def save_qc_results_to_db(conn, df_saved: list[dict]):
    """Save QC results to AWST_CLI_PRE_HOR_QC table."""
    saved_count = 0
    
    # Convert back to list of dicts for existing DB functions
    results = df_saved.to_dict('records')
    
    for record in results:
        # Map label
        qc_label_mapped = LABEL_MAPPING.get(record['qc_label'])
        if qc_label_mapped is None:
            continue
        
        dt_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dt_ddatetime = record['ddatetime'].strftime('%Y-%m-%d %H:%M:%S')
        # Check for duplicates
        record_exist = check_record_in_qc_table(conn, record['stacode'], record['ddatetime'])

        if record_exist:
            # Update to NULL
            sql_update = text(
                f"UPDATE {QC_TABLE_NAME} SET R = {record['r']}, QC='{qc_label_mapped}', D_IYMDHM = TO_DATE('{dt_now}', 'YYYY-MM-DD HH24:MI:SS') "
                f"WHERE STACODE = '{record['stacode']}' AND DDATETIME = TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS')"
            )
            conn.execute(sql_update)
        else:
            # Insert record
            sql_insert = text(
                f"INSERT INTO {QC_TABLE_NAME} "
                f"(STACODE, DDATETIME, R, QC, D_IYMDHM) "
                f"VALUES ('{record['stacode']}', TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS'), "
                f"{record['r']}, '{qc_label_mapped}', TO_DATE('{dt_now}', 'YYYY-MM-DD HH24:MI:SS'))"
            )
            conn.execute(sql_insert)
        
        saved_count += 1
    
    return saved_count


def update_false_to_null(conn, df_false: list[dict]):
    """Update original tables to set R=NULL for false records."""
    updated_count = 0
 
    results = df_false.to_dict('records')   

    for record in results:
        year = record['ddatetime'].year
        table_name = f"{DATA_TABLE_NAME}_{year}"
        
        # Update to NULL
        dt_ddatetime = record['ddatetime'].strftime('%Y-%m-%d %H:%M:%S')
        sql_update = text(
            f"UPDATE {table_name} SET R = NULL "
            f"WHERE STACODE = '{record['stacode']}' AND DDATETIME = TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS')"
        )
        conn.execute(sql_update)
        
        updated_count += 1
    
    return updated_count


def compute_monthly_statistics(df_results: pd.DataFrame, year, month) -> dict:
    """Compute count of each QC label type."""
    stats = {label: 0 for label in ALL_QC_LABELS}
    counts = df_results['qc_label'].value_counts()
    for label in ALL_QC_LABELS:
        stats[label] = counts.get(label, 0)
    
    stats['year'] = year
    stats['month'] = month
    
    # Reorder columns
    monthly_stats = {
        'year': stats['year'],
        'month': stats['month'],
        **{label: stats[label] for label in ALL_QC_LABELS}
    }

    return monthly_stats


def append_monthly_stats_to_csv(monthly_stats: dict, output_file: Path):
    """Append a single month's statistics to CSV file (creates file if not exists)."""
    df_stats = pd.DataFrame([monthly_stats])
    
    # Check if file exists to determine if we need headers
    file_exists = output_file.exists()
    
    # Append to CSV (mode='a' for append, header=False if file exists)
    df_stats.to_csv(
        output_file, 
        mode='a', 
        header=not file_exists, 
        index=False, 
        encoding='utf-8-sig'
    )


def main():
    """Main QC processing function."""
    print(f"Starting AWST precipitation QC at {datetime.now()}")
    
    # Load configuration
    print("Loading station locations...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    valid_stations = df_stations['stacode'].tolist()
    
    print(f"Loading {OUTLIER_CIRCLES_RADIUS}km outlier circles...")
    df_outlier_circles = load_circles(INPUT_OUTLIER_CIRCLES)
    
    print(f"Loading {EXTREME_CIRCLES_RADIUS}km extreme circles...")
    df_extreme_circles = load_circles(INPUT_EXTREME_CIRCLES)
    
    print("Connecting to database...")
    db_config = load_db_config(CONFIG_FILE, DB_SECTION)
    engine = create_db_engine(db_config)
    
    # Main QC loop
    for year in range(YEAR_START, YEAR_END + 1):
        for month in range(1, 13):

            print(f"Processing year {year}, month {month}...")

            # Step 1: Fetch monthly data
            with engine.connect() as conn:
                df_monthly = fetch_monthly_data(conn, year, month, valid_stations)
            
            if df_monthly.empty:
                print(f"No data for year {year}, month {month}")
                continue
            
            # Step 2: Process each hour in memory (disconnected)
            all_results = []
            unique_hours = df_monthly['ddatetime'].drop_duplicates()
            
            for hour in unique_hours:
                # Only process hours in the current month
                if hour.month != month:
                    continue
                print(f"Processing time {hour.strftime('%Y-%m-%d %H')}...")
                
                results = perform_hourly_qc(
                    hour, df_monthly, 
                    df_outlier_circles, df_extreme_circles
                )
                all_results.extend(results)
            
            # Step 3: Save results and update originals
            if all_results:
                # Convert to DataFrame for efficient filtering and statistics
                df_results = pd.DataFrame(all_results)
                
                monthly_stats = compute_monthly_statistics(df_results, year, month)            
                
                # IMMEDIATELY append monthly statistics to CSV
                append_monthly_stats_to_csv(monthly_stats, OUTPUT_MONTHLY_STATS)
                print(f"  ✓ Monthly statistics saved for {year}-{month:02d}")

                # Filter results: exclude NORMAL and EXTREME_TYPE1 (vectorized boolean indexing)
                mask_save = ~df_results['qc_label'].isin(['NORMAL', 'EXTREME_TYPE1'])
                df_saved = df_results[mask_save]
                
                if not df_saved.empty:
                    print(f"Updating the database year {year}, month {month}...")
                    with engine.connect() as conn:
                        save_qc_results_to_db(conn, df_saved)
                        conn.commit()

                    # Filter FALSE records for nullifying originals (vectorized)
                    mask_false = df_saved['qc_label'].isin(['FALSE'])
                    df_false = df_saved[mask_false]
                    
                    if UPDATE_FALSE_TO_NULL and not df_false.empty:
                        with engine.connect() as conn:
                            update_false_to_null(conn, df_false)
                            conn.commit()
            print(f"Completed year {year}, month {month}")
            
            # Cleanup
            del df_monthly, all_results
            gc.collect()

    engine.dispose()
    print(f"\nFinished at {datetime.now()}")
    print(f"Monthly statistics saved incrementally to {OUTPUT_MONTHLY_STATS}")


if __name__ == '__main__':
    main()
