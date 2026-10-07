# -*- coding: utf-8 -*-
"""
Section 4b(1): outlier circle radius and IQR multiplier k (manuscript Table 6).

Stage 1 accepts a record as NORMAL if r <= Q3 + k * IQR in any outlier circle containing the station.
A false record accepted by stage 1 escapes the stage 2 inspection, so stage 1 should accept as few
false records as possible. For every training record, outlier circle radius (RADII) and multiplier
(MULTIPLIERS), this script stores whether stage 1 accepts the record, with the critical multiplier
(the smallest k at which it is accepted). The table counts the false records of MIN_R mm or more
accepted by stage 1. The per-record file is also read by the other tuning scripts (05-11).
Runs offline on the cached data; the per-record file is reused unless RECOMPUTE is True or a
radius or multiplier is added.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import pandas as pd

import _bootstrap  # noqa: F401
from src.qc_data_loader import load_circles, load_precip_cache
from src.qc_evaluation import load_validation_data, stage1_critical_multiplier, stage1_normal_by_multiplier
from src.qc_tuning import CACHE_FILE, N_WORKERS, RECORDS_FILE, STAGE1_FILE, outlier_circles_file, save_table

RADII = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # km
MULTIPLIERS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 50, 100]
MIN_R = 10  # mm; below, a false value within widespread light rain cannot be separated by any fence
RECOMPUTE = False


def stage1_by_radius(radius: int) -> pd.DataFrame:
    """Critical multiplier and stage 1 acceptance per k of every record for one outlier circle radius."""
    df_records = load_validation_data(RECORDS_FILE)
    df_cache = load_precip_cache(CACHE_FILE)
    circles = load_circles(outlier_circles_file(radius))
    rows = []
    for ddatetime, df_group in df_records.groupby('ddatetime'):
        df_hour = df_cache.loc[[ddatetime]].reset_index()
        df_hour['stacode'] = df_hour['stacode'].astype(str)
        for idx, record in df_group.iterrows():
            stacode = str(record['stacode'])
            row = {'index': idx, f"critical_{radius}km": stage1_critical_multiplier(stacode, df_hour, circles)}
            for k, normal in stage1_normal_by_multiplier(stacode, df_hour, circles, MULTIPLIERS).items():
                row[f"normal_{radius}km_k{k}"] = normal
            rows.append(row)
    print(f"Finished {radius} km", flush=True)
    return pd.DataFrame(rows).set_index('index')


def stage1_file_complete() -> bool:
    if not STAGE1_FILE.exists():
        return False
    columns = set(pd.read_csv(STAGE1_FILE, encoding='utf-8-sig', nrows=0).columns)
    return all(f"normal_{r}km_k{k}" in columns for r in RADII for k in MULTIPLIERS)


def main() -> None:
    if RECOMPUTE or not stage1_file_complete():
        print(f"Stage 1 of every record for {len(RADII)} radii ({N_WORKERS} processes), {datetime.now()}", flush=True)
        df_records = load_validation_data(RECORDS_FILE)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            parts = list(executor.map(stage1_by_radius, RADII))
        df_records.join(pd.concat(parts, axis=1)).to_csv(STAGE1_FILE, index=False, encoding='utf-8-sig')

    df = pd.read_csv(STAGE1_FILE, encoding='utf-8-sig', dtype={'stacode': str})
    false = ~df['validation'].astype(str).str.upper().eq('TRUE') & (df['r'] >= MIN_R)
    print(f"Training records of {MIN_R} mm or more: {int((df['r'] >= MIN_R).sum())} ({int(false.sum())} false)")
    rows = []
    for radius in RADII:
        row = {'outlier_radius_km': radius}
        row.update({f"k={k}": int((false & df[f"normal_{radius}km_k{k}"].astype(str).str.upper().eq('TRUE')).sum())
                    for k in MULTIPLIERS})
        row['first_false_accepted_k'] = round(float(df.loc[false, f"critical_{radius}km"].min()), 2)
        rows.append(row)
    save_table(pd.DataFrame(rows), 'table6_stage1_false_accepted')


if __name__ == '__main__':
    main()
