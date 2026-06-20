# -*- coding: utf-8 -*-
"""
Extract validation data for surf-type and awst-type precipitation events from database.

Queries both surface stations (r >= 30mm) and automatic weather stations (r > 184.4mm)
based on gd_station_locations.csv.
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
# DATA_TB_AWST = 'awst_cli_mul_hor'
INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'

EVENT_SHRESHOLD = 30
CLIMATE_LIMIT = 184.4

YEAR_STT, YEAR_END = 2003, 2025
YEARS = range(YEAR_END, YEAR_STT-1, -1)

OUTPUT = SRC_DIR.parent / 'data' / f"gd_validation_precip_{YEAR_STT}-{YEAR_END}.csv"
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


def load_station_info(file_path: Path) -> pd.DataFrame:
    """Load station information from CSV file."""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return df


def fetch_validation_precip(engine, statype: str, year:int, stations_code: pd.DataFrame):
    """Load surf-type target events with r >= 30 from a specific year's datatable."""
    sql = f"select stacode, ddatetime, r from {statype}_{DATA_TB_BASE}_{year}"
    if statype == 'SURF':
        sql += f" where r>={EVENT_SHRESHOLD}"
    elif statype == 'AWST':
        sql += f" where r>{CLIMATE_LIMIT}"
    # sql += " order by ddatetime, stacode"

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    
    df['stacode'] = df['stacode'].astype(str).str.strip()
    df = df.set_index('stacode')
    df_filter = df[df.index.isin(stations_code)].reset_index()
    df_filter['statype'] = statype
    if statype == 'SURF':
        df_filter['validation'] = 'TRUE'
    elif statype == 'AWST':
        df_filter['validation'] = 'FALSE'

    return df_filter


def main():
    print(f"Starting precipitation validation data collection {datetime.now()}")
    
    print("Loading station location info...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    all_stations = df_stations['stacode']

    db_config = load_db_config(CONFIG_FILE, DB_SECTION)
    engine = create_db_engine(db_config)

    all_data = []
    for year in YEARS:
        print(f"extracting precipitation data from {year}...")

        # Load target events from surf stations
        precip_surf = fetch_validation_precip(engine, 'SURF', year, all_stations)
        # Fetch awst-type target events with r > climate limit from the same year's datatable
        precip_awst = fetch_validation_precip(engine, 'AWST', year, all_stations)

        year_sample = pd.concat([precip_surf, precip_awst], ignore_index=True)
        year_sample.drop_duplicates(subset=['stacode', 'ddatetime'], inplace=True)
        all_data.append(year_sample)

    df_out = pd.concat(all_data, ignore_index=True)
    df_out.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df_out)} rows to {OUTPUT}:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    engine.dispose()


if __name__ == '__main__':
    main()