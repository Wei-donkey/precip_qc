# -*- coding: utf-8 -*-
"""
Extract station locations for Guangdong province from database.

This script queries the Oracle database to retrieve longitude, latitude, 
and starting date for meteorological stations in Guangdong (广东) that have 
station type markers A/B. 
"""

from __future__ import annotations

import configparser
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from sqlalchemy import create_engine

SRC_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SRC_DIR / 'config_db.ini'
DB_SECTION = 'CROSS_WEATHER'
TABLE_NAME = 't_othe_station_meta_basic_tab'
OUTPUT = SRC_DIR.parent / 'data' / 'gd_stations_locations.csv'


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


def fetch_station_locations(engine, table_name: str):
    sql = (
        f"select v01301 stacode, v05001 lat, v06001 lon, to_char(stt_date,'yyyy-mm-dd') date_stt, "
        f"case "
        f"when v02301 like '%A%' then 'surf' "
        f"when v02301 like '%B%' then 'awst' "
        f"end as statype "
        f"from {table_name} "
        f"where v_prcode = '广东' "
        f"and (v02301 like '%A%' or v02301 like '%B%') "
        f"and extract(year from stt_date)>1899 "
        f"and extract(year from stt_date)<2026 "
        f"and inland=1 "
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    return df


def main() -> None:
    db_config = load_db_config(CONFIG_FILE, DB_SECTION)
    engine = create_db_engine(db_config)

    print(f"Reading station metadata from table: {TABLE_NAME}")
    locations = fetch_station_locations(engine, TABLE_NAME)

    # Ensure output directory exists
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    locations.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f"Wrote {len(locations)} rows to {OUTPUT}")

    engine.dispose()


if __name__ == '__main__':
    main()
