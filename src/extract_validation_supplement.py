# -*- coding: utf-8 -*-
"""
Extract supplementary validation data for awst-type events from database.

Queries automatic weather station events (100 < r <= 184.4mm) during periods
with multiple FALSE events, based on gd_validation_precip_supplement_info.csv.
"""

from __future__ import annotations

import configparser
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote
import pandas as pd
from sqlalchemy import create_engine

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / 'data'
CONFIG_FILE = SRC_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
DATA_TB_BASE = 'cli_mul_hor'
INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'
INPUT_SUPPLEMENT_TIME = SRC_DIR.parent / 'data' / "gd_validation_precip_supplement_info.csv"

EVENT_SHRESHOLD = 30
CLIMATE_LIMIT = 184.4

OUTPUT = SRC_DIR.parent / 'data' / "gd_validation_precip_supplement.csv"
# Ensure output directory exists
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def load_db_config(config_path: Path, section: str):
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8-sig')

    db = config[section]
    return {
        'user': db['user'],
        'password': db['password'],
        'host': db['host'],
        'port': db['port'],
        'service': db['service'],
    }


def create_db_engine(db_config: dict[str, str]):
    password = quote(db_config['password'])
    conn_string = (f"oracle+oracledb://{db_config['user']}:{password}@{db_config['host']}"
                   f":{db_config['port']}/{db_config['service']}")
    return create_engine(conn_string, echo=False)


def load_supplement_time(file_path: Path) -> pd.DataFrame:
    """Load supplementation info from CSV file."""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return df


def fetch_validation_precip(engine, statype: str, time_stt: pd.Timestamp, time_end: pd.Timestamp, stacode: str):
    """Load target events with r >= 100 and r<=184.4 from a specific station and period."""
    years = range(time_stt.year, time_end.year + 1)
    
    all_data = []
    for year in years:
        table_name = f"{statype}_CLI_MUL_HOR_{year}"
        sql = (
            f"SELECT stacode, ddatetime, r FROM {table_name} "
            f"WHERE ddatetime >= TO_DATE('{time_stt}', 'YYYY-MM-DD HH24:MI:SS') "
            f"AND ddatetime <= TO_DATE('{time_end}', 'YYYY-MM-DD HH24:MI:SS') "
            f"AND stacode='{stacode}' "
            f"AND r >= {EVENT_SHRESHOLD} AND r <= {CLIMATE_LIMIT}"
        )
        
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        if not df.empty:
            df['statype'] = statype
            df['validation'] = 'FALSE'
            all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def main():
    print(f"Starting precipitation validation data supplementation {datetime.now()}")
    
    print("Loading supplementation info...")
    df_supplement_time = load_supplement_time(INPUT_SUPPLEMENT_TIME)

    db_config = load_db_config(CONFIG_FILE, DB_SECTION)
    engine = create_db_engine(db_config)

    all_data = []
    for _, row in df_supplement_time.iterrows():
        stacode = row['stacode']
        time_stt = pd.to_datetime(row['date_stt'])
        time_end = pd.to_datetime(row['date_end']) + timedelta(hours=23) 
        print(f"Supplementing precipitation data from {stacode}...")

        # Load target events from awst stations
        precip_stacode = fetch_validation_precip(engine, 'AWST', time_stt, time_end, stacode)

        all_data.append(precip_stacode)

    df_out = pd.concat(all_data, ignore_index=True)
    df_out.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df_out)} rows to {OUTPUT}:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    engine.dispose()


if __name__ == '__main__':
    main()