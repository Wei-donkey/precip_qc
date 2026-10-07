# -*- coding: utf-8 -*-
"""
Verify false labels with national observatories.

False AWS records were labeled by exceeding the climatological limit or by their occurrence within
station malfunction periods, which may also contain genuine rainfall. A record above the limit is
false by definition. A malfunction-period record is verified as false only if at least one national
meteorological observatory (NMO, staffed and manually checked) lies within RADIUS_KM and every NMO
within RADIUS_KM recorded less than DRY_MM in the same hour and the WINDOW_HOURS adjacent hours.
Every false record of the labeled pool (data/gd_validation_false_pool.csv) is checked; only the
verified ones were kept in the training and test sets. Runs offline on the cached neighbor rainfall in
data/cache/ (INPUT_CACHES), which covers the hours around every false record.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.qc_data_loader import load_precip_cache

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
VALIDATION_DIR = SCRIPT_DIR.parent / 'outputs' / 'tables' / 'validation'
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

CLIMATE_LIMIT = 184.4  # mm
RADIUS_KM = 30
DRY_MM = 1
WINDOW_HOURS = 1
EARTH_RADIUS_KM = 6371.0

INPUT_STATIONS = DATA_DIR / 'gd_stations_locations.csv'
INPUT_RECORDS = DATA_DIR / 'gd_validation_false_pool.csv'
INPUT_CACHES = [DATA_DIR / 'cache' / f"{INPUT_RECORDS.stem}_adjacent2h.parquet"]
OUTPUT_FILE = VALIDATION_DIR / 'qc_false_label_nmo_support.csv'


def main() -> None:
    print(f"Checking false labels against NMOs at {datetime.now()}")
    df_stations = pd.read_csv(INPUT_STATIONS, encoding='utf-8-sig', dtype={'stacode': str}).set_index('stacode')
    df_nmo = df_stations[df_stations['statype'].str.upper() == 'SURF']
    nmo_lat, nmo_lon = np.radians(df_nmo['lat'].to_numpy()), np.radians(df_nmo['lon'].to_numpy())

    df = pd.read_csv(INPUT_RECORDS, encoding='utf-8-sig', dtype={'stacode': str})
    df['ddatetime'] = pd.to_datetime(df['ddatetime'])
    df_false = df[df['validation'].astype(str).str.upper() == 'FALSE']

    df_cache = pd.concat([load_precip_cache(f) for f in INPUT_CACHES])
    rows = []
    for _, record in df_false.iterrows():
        if record['ddatetime'] not in df_cache.index:
            raise ValueError(f"No cached data at {record['ddatetime']} ({record['stacode']})")
        lat, lon = np.radians(df_stations.loc[record['stacode'], ['lat', 'lon']].to_numpy(dtype=float))
        a = np.sin((nmo_lat - lat) / 2) ** 2 + np.cos(lat) * np.cos(nmo_lat) * np.sin((nmo_lon - lon) / 2) ** 2
        nearby = df_nmo.index[2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)) <= RADIUS_KM]
        hours = pd.Timedelta(hours=WINDOW_HOURS)
        df_window = df_cache.loc[record['ddatetime'] - hours:record['ddatetime'] + hours]
        values = df_window.loc[df_window['stacode'].astype(str).isin(nearby), 'r']
        n_nmo = df_window.loc[df_window['stacode'].astype(str).isin(nearby), 'stacode'].nunique()
        nmo_max = values.max() if len(values) else np.nan
        exceedance = record['r'] > CLIMATE_LIMIT
        rows.append({'stacode': record['stacode'], 'ddatetime': record['ddatetime'], 'r': record['r'],
                     'n_nmo': n_nmo, 'nmo_max': nmo_max, 'exceedance': exceedance,
                     'verified': bool(exceedance or (n_nmo > 0 and nmo_max < DRY_MM))})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    bins = pd.cut(df_out['r'], [1, 5, 10, 20, 30, 50, 100, np.inf], right=False)
    print(df_out.groupby(bins, observed=True)['verified'].agg(['size', 'sum']).to_string())
    print(f"Verified: {int(df_out['verified'].sum())} of {len(df_out)} false records")
    print(f"Saved {OUTPUT_FILE.name}; finished at {datetime.now()}")


if __name__ == '__main__':
    main()
