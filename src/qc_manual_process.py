# -*- coding: utf-8 -*-
"""
Manual list Processing and Data Restoration Script.

This script provides two main functionalities:
1. Process manual list entries: Move precipitation data from original tables to QC table (AWST_CLI_PRE_HOR_QC)
   and set R=NULL in original tables.
2. Restore precipitation data: Write precipitation data back to original tables from QC table,
   and remove the records from AWST_CLI_PRE_HOR_QC.
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from sqlalchemy import create_engine, text

from qc_data_loader import load_db_config

SRC_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SRC_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_DIR = SRC_DIR.parent / 'data'

# Input files
MANUAL_LIST_FILE = SRC_DIR / 'qc_manual_list.ini'

# Switch: do not set to True during testing
UPDATE_FALSE_TO_NULL = False

QC_TABLE_NAME = 'AWST_CLI_PRE_HOR_QC'
DATA_TABLE_NAME = 'AWST_CLI_MUL_HOR'


def create_db_engine(db_config: dict[str, str]):
    """Create SQLAlchemy engine for Oracle database connection."""
    password = quote(db_config['password'])
    conn_string = (f"oracle+oracledb://{db_config['user']}:{password}@{db_config['host']}"
                   f":{db_config['port']}/{db_config['service']}")
    return create_engine(conn_string, echo=False)


def load_manual_list(file_path: Path) -> list[dict]:
    """Load manual list entries from INI file.
    
    Note: This function is kept for compatibility but process_manual_list_data() 
    now auto-generates manual list via generate_manual_list().
    """
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    manual_list = []
    for _, row in df.iterrows():
        stacode = str(row['stacode']).strip()
        # Parse datetimes in YYYY-MM-DD HH format
        time_stt = datetime.strptime(str(row['time_stt']), '%Y-%m-%d %H')
        time_end = datetime.strptime(str(row['time_end']), '%Y-%m-%d %H')
        manual_list.append({
            'stacode': stacode,
            'time_stt': time_stt,
            'time_end': time_end
        })
    return manual_list


def check_record_in_qc_table(conn, stacode: str, ddatetime: datetime) -> bool:
    """Check if record exists in AWST_CLI_PRE_HOR_QC table."""
    sql = f"SELECT COUNT(*) AS cnt FROM {QC_TABLE_NAME} WHERE STACODE = '{stacode}' AND DDATETIME = TO_DATE('{ddatetime.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS')"
    df = pd.read_sql(sql, conn)
    return df['cnt'].iloc[0] > 0


def generate_manual_list(engine):
    """Generate manual_list based on FALSE records analysis from QC table.
    
    Rule: 
    1. Identify frequent-false station-month combinations (monthly FALSE of over 30mm count > 5)
    2. For each FALSE record in these specific station-month periods, create a manual_list period of ±12 hours
    3. Merge overlapping time periods for each station
    4. Each station can have multiple non-overlapping manual_list periods
    Note: No precipitation threshold - all FALSE records are considered regardless of R value
    """
    print("Generating manual_list from QC FALSE records...")
    
    conn = engine.connect()
    try:
        # Step 1: Identify frequent-false station-month combinations (no R threshold)
        sql_frequent_periods = f"""
            SELECT STACODE, 
                   EXTRACT(YEAR FROM DDATETIME) AS year,
                   EXTRACT(MONTH FROM DDATETIME) AS month
            FROM {QC_TABLE_NAME}
            WHERE QC = 'FALSE'
            AND R > 30
            GROUP BY STACODE, EXTRACT(YEAR FROM DDATETIME), EXTRACT(MONTH FROM DDATETIME)
            HAVING COUNT(*) > 5
            ORDER BY STACODE, year, month
        """
        
        df_frequent_periods = pd.read_sql(sql_frequent_periods, conn)
    
        if df_frequent_periods.empty:
            print("No frequent-false station-months found (criteria: monthly FALSE count > 5)")
            return []
        
        print(f"Found {len(df_frequent_periods)} frequent-false station-month combinations")
        
        # Step 2: Build query to fetch FALSE records only from these specific station-month periods
        conditions = []
        for _, row in df_frequent_periods.iterrows():
            stacode = str(row['stacode']).strip()
            year = int(row['year'])
            month = int(row['month'])
            conditions.append(
                f"(STACODE = '{stacode}' AND "
                f"EXTRACT(YEAR FROM DDATETIME) = {year} AND "
                f"EXTRACT(MONTH FROM DDATETIME) = {month})"
            )
        
        where_clause = " OR ".join(conditions)
        
        sql_false_records = f"""
            SELECT STACODE, DDATETIME, R
            FROM {QC_TABLE_NAME}
            WHERE QC = 'FALSE'
              AND ({where_clause})
            ORDER BY STACODE, DDATETIME
        """
        
        df_false_records = pd.read_sql(sql_false_records, conn)
        
        if df_false_records.empty:
            print("No FALSE records found for frequent-false station-months")
            return []
        
        print(f"Found {len(df_false_records)} FALSE records from frequent-false station-months")
        
        # Step 3: Generate initial manual_list periods (±12 hours for each FALSE record)
        raw_periods = []
        for _, record in df_false_records.iterrows():
            stacode = str(record['stacode']).strip()
            false_time = record['ddatetime']
            
            # Calculate ±12 hour window
            time_start = false_time - timedelta(hours=12)
            time_end = false_time + timedelta(hours=12)
            
            raw_periods.append({
                'stacode': stacode,
                'time_start': time_start,
                'time_end': time_end
            })
        
        # Step 4: Merge overlapping periods for each station
        df_raw = pd.DataFrame(raw_periods)
        merged_periods = []
        
        for stacode in df_raw['stacode'].unique():
            # Get all periods for this station and sort by start time
            station_periods = df_raw[df_raw['stacode'] == stacode].sort_values('time_start').reset_index(drop=True)
            
            if len(station_periods) == 0:
                continue
            
            # Initialize with first period
            current_start = station_periods.loc[0, 'time_start']
            current_end = station_periods.loc[0, 'time_end']
            
            # Iterate through remaining periods
            for i in range(1, len(station_periods)):
                next_start = station_periods.loc[i, 'time_start']
                next_end = station_periods.loc[i, 'time_end']
                
                # Check if periods overlap or are adjacent
                if next_start <= current_end:
                    # Merge: extend current_end if next_end is later
                    current_end = max(current_end, next_end)
                else:
                    # No overlap: save current period and start new one
                    merged_periods.append({
                        'stacode': stacode,
                        'time_stt': current_start.strftime('%Y-%m-%d %H'),
                        'time_end': current_end.strftime('%Y-%m-%d %H')
                    })
                    current_start = next_start
                    current_end = next_end
            
            # Don't forget the last period
            merged_periods.append({
                'stacode': stacode,
                'time_stt': current_start.strftime('%Y-%m-%d %H'),
                'time_end': current_end.strftime('%Y-%m-%d %H')
            })
        
        # Convert to DataFrame and sort
        df_manual_list = pd.DataFrame(merged_periods)
        df_manual_list = df_manual_list.sort_values(['stacode', 'time_stt']).reset_index(drop=True)
        
        # Save to qc_manual_list.ini
        df_manual_list.to_csv(MANUAL_LIST_FILE, index=False, encoding='utf-8-sig')
        
        print(f"manual_list saved to {MANUAL_LIST_FILE}")

    finally:
        conn.close()


def process_manual_list_data(engine):
    """Process manual_list entries: insert into QC table and nullify originals.
    
    Only processes records with R >= 30 mm during manual_list periods.
    Records original QC status in QC_ORIGINAL column (TRUE if record didn't exist before).
    """
    print("Processing manual_list data...")
    
    # First, load manual_list
    manual_list = load_manual_list(MANUAL_LIST_FILE)
    
    processed_count = 0
    skipped_count = 0
    total_entries = len(manual_list)
    
    print(f"\nProcessing {total_entries} manual_list entries...\n")
    
    conn = engine.connect()
    try:
        for entry in manual_list:
            stacode = entry['stacode']
            # Parse datetimes in YYYY-MM-DD HH format
            time_stt = entry['time_stt'] if isinstance(entry['time_stt'], datetime) else datetime.strptime(entry['time_stt'], '%Y-%m-%d %H')
            time_end = entry['time_end'] if isinstance(entry['time_end'], datetime) else datetime.strptime(entry['time_end'], '%Y-%m-%d %H')
            
            # Calculate total hours for this entry
            total_hours = int((time_end - time_stt).total_seconds() / 3600)
            hour_count = 0
            
            # Loop through each hour in the time range
            current_time = time_stt
            while current_time < time_end:  # Use < since end is exclusive
                year = current_time.year
                hour_count += 1
                
                # Fetch the R value to check if >= 10mm
                table_name = f"{DATA_TABLE_NAME}_{year}"
                sql_fetch = text(
                    f"SELECT R FROM {table_name} WHERE STACODE = '{stacode}'"
                    f" AND DDATETIME = TO_DATE('{current_time.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS')"
                    f" AND R >= 30"
                )
                df_r = pd.read_sql(sql_fetch, conn)
                r_value = df_r['r'].iloc[0] if not df_r.empty else None
                
                if r_value is not None:
                    dt_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    dt_ddatetime = current_time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    record_exists = check_record_in_qc_table(conn, stacode, current_time)
                    # Check if already in QC table to get original QC value
                    if record_exists:
                        # Record exists - fetch original QC value
                        sql_get_qc = f"SELECT QC FROM {QC_TABLE_NAME} WHERE STACODE = '{stacode}' AND DDATETIME = TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS')"
                        df_qc = pd.read_sql(sql_get_qc, conn)
                        original_qc = df_qc['qc'].iloc[0]
                        
                        # Only update if original QC is not 'MANUAL'
                        if original_qc != 'MANUAL':
                            # Update existing record
                            sql_update = text(
                                f"UPDATE {QC_TABLE_NAME} "
                                f"SET QC = 'MANUAL', QC_ORIGINAL = '{original_qc}', D_IYMDHM = TO_DATE('{dt_now}', 'YYYY-MM-DD HH24:MI:SS') "
                                f"WHERE STACODE = '{stacode}' AND DDATETIME = TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS')"
                            )
                            conn.execute(sql_update)
                    else:
                        # Record doesn't exist - original QC is TRUE
                        original_qc = 'TRUE'
                        
                        # Insert new record into QC table
                        sql_insert = text(
                            f"INSERT INTO {QC_TABLE_NAME} "
                            f"(STACODE, DDATETIME, R, QC, QC_ORIGINAL, D_IYMDHM) "
                            f"VALUES ('{stacode}', TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS'), "
                            f"{r_value}, 'MANUAL', '{original_qc}', TO_DATE('{dt_now}', 'YYYY-MM-DD HH24:MI:SS'))"
                        )
                        conn.execute(sql_insert)
                    
                    # Update original table to NULL
                    if UPDATE_FALSE_TO_NULL:
                        sql_update_original = text(
                            f"UPDATE {table_name} SET R = NULL "
                            f"WHERE STACODE = '{stacode}' AND DDATETIME = TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS')"
                        )
                        conn.execute(sql_update_original)
                
                processed_count += 1
                
                # Print progress every 100 hours or at completion
                if hour_count % 100 == 0 or hour_count == total_hours:
                    print(f"  Progress: {hour_count}/{total_hours} hours checked for {stacode}")

                current_time += timedelta(hours=1)
            
            print(f"Completed station {stacode}: {processed_count} processed, {skipped_count} skipped so far\n")
        
        conn.commit()
    finally:
        conn.close()
    
    print(f"manual_list processing complete: {processed_count} processed, {skipped_count} skipped")


def restore_manual_data(engine):
    """Restore precipitation data from QC table back to original tables and remove from QC table."""
    print("Restoring manual_list data from QC table...")
    
    restored_count = 0
    skipped_count = 0
    
    conn = engine.connect()
    try:
        # Fetch all MANUAL records from QC table
        sql_fetch_manual_data = f"SELECT STACODE, DDATETIME, R, QC_ORIGINAL FROM {QC_TABLE_NAME} WHERE QC = 'MANUAL'"
        df_manual_data = pd.read_sql(sql_fetch_manual_data, conn)
        
        total_records = len(df_manual_data)
        print(f"Found {total_records} MANUAL records to restore.\n")
        
        for idx, record in df_manual_data.iterrows():
            stacode = record['stacode']
            ddatetime = record['ddatetime']
            r_value = record['r']
            year = ddatetime.year
            qc_original = record['qc_original']
            
            # Update original table to restore R value
            table_name = f"{DATA_TABLE_NAME}_{year}"
            dt_ddatetime = ddatetime.strftime('%Y-%m-%d %H:%M:%S')
            sql_update_original = text(
                f"UPDATE {table_name} SET R = {r_value} "
                f"WHERE STACODE = '{stacode}' AND DDATETIME = TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS')"
            )
            conn.execute(sql_update_original)
            
            if qc_original == 'TRUE':
                # Delete record from QC table
                sql_delete_qc = text(
                    f"DELETE FROM {QC_TABLE_NAME} "
                    f"WHERE STACODE = '{stacode}' AND DDATETIME = TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS')"
                )
                conn.execute(sql_delete_qc)
            
            else:
                # Update QC table to restore original QC value
                sql_update_qc = text(
                    f"UPDATE {QC_TABLE_NAME} "
                    f"SET QC = '{qc_original}', QC_ORIGINAL = NULL "
                    f"WHERE STACODE = '{stacode}' AND DDATETIME = TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS')"
                )
                conn.execute(sql_update_qc)
            
            restored_count += 1
            
            # Print progress every 100 records
            if (idx + 1) % 100 == 0 or (idx + 1) == total_records:
                print(f"Progress: {idx + 1}/{total_records} records processed")
        
        conn.commit()
    finally:
        conn.close()
    
    print(f"\nData restoration complete: {restored_count} restored, {skipped_count} skipped")


def restore_false_data(engine):
    """Restore FALSE precipitation data from QC table back to original tables and remove from QC table."""
    print("Restoring false data from QC table...")
    
    restored_count = 0
    skipped_count = 0
    
    conn = engine.connect()
    try:
        # Fetch all FALSE records from QC table
        sql_fetch_fasle_data = f"SELECT STACODE, DDATETIME, R FROM {QC_TABLE_NAME} WHERE QC = 'FALSE'"
        df_false_data = pd.read_sql(sql_fetch_fasle_data, conn)
        
        if df_false_data.empty:
            print("No FALSE records found in QC table.")
            return
        
        total_records = len(df_false_data)
        print(f"Found {total_records} FALSE records to restore.\n")
        
        for idx, record in df_false_data.iterrows():
            stacode = record['stacode']
            ddatetime = record['ddatetime']
            r_value = record['r']
            year = ddatetime.year

            # Update original table to restore R value
            table_name = f"{DATA_TABLE_NAME}_{year}"
            dt_ddatetime = ddatetime.strftime('%Y-%m-%d %H:%M:%S')
            sql_update_original = text(
                f"UPDATE {table_name} SET R = {r_value} "
                f"WHERE STACODE = '{stacode}' AND DDATETIME = TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS')"
            )
            conn.execute(sql_update_original)
            
            # Delete record from QC table
            sql_delete_qc = text(
                f"DELETE FROM {QC_TABLE_NAME} "
                f"WHERE STACODE = '{stacode}' AND DDATETIME = TO_DATE('{dt_ddatetime}', 'YYYY-MM-DD HH24:MI:SS')"
            )
            conn.execute(sql_delete_qc)
            
            restored_count += 1
            
            # Print progress every 100 records
            if (idx + 1) % 100 == 0 or (idx + 1) == total_records:
                print(f"Progress: {idx + 1}/{total_records} records restored")
        
        conn.commit()
    finally:
        conn.close()
    
    print(f"\nData restoration complete: {restored_count} restored, {skipped_count} skipped")

def main():
    """Main function with user choice for operation mode."""
    print("=" * 70)
    print("manual_list Processing and Data Restoration Tool")
    print("=" * 70)
    print("\nSelect operation:")
    print("1. Generate manual_list only (analyze QC FALSE records and create qc_manual_list.ini)")
    print("2. Process manual_list (auto-generate + move data to QC table, nullify originals)")
    print("3. Restore manual data (write data back to originals, remove MANUAL from QC table)")
    print("4. Restore fasle data (write data back to originals, remove FALSE from QC table)")
    print("=" * 70)
    
    choice = input("\nEnter your choice (1, 2, 3 or 4): ").strip()
    
    if choice not in ['1', '2', '3', '4']:
        print("Invalid choice. Exiting.")
        return
    
    # Load database configuration
    print("\nLoading database configuration...")
    db_config = load_db_config(CONFIG_FILE, DB_SECTION)
    engine = create_db_engine(db_config)
    
    try:
        if choice == '1':
            generate_manual_list(engine)
        elif choice == '2':
            process_manual_list_data(engine)
        elif choice == '3':
            restore_manual_data(engine)
        elif choice == '4':
            restore_false_data(engine)

    finally:
        engine.dispose()
        print(f"\nFinished at {datetime.now()}")


if __name__ == '__main__':
    main()
