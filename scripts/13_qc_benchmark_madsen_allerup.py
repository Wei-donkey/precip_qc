# -*- coding: utf-8 -*-
"""
Benchmark the proposed QC method against the Madsen-Allerup spatial consistency check.

The Madsen-Allerup check (Vejen et al., 2002) is a single-pass test on the 12 nearest reporting
stations at the same hour: T = (x - median) / (q75 - q25), flagged when |T| > k (see
src/qc_benchmarks.py). The multiplier k is tuned on the training set (gd_validation_data_train.csv) and
selected by the Youden index (sensitivity + specificity - 1); the values used in the literature
(k = 2, Vejen et al. 2002; k = 4, Jiang et al. 2015) are also reported. The checks and the
proposed method are then compared on the test set (gd_validation_data_test.csv), per rainfall
intensity bin with Wilson 95% confidence intervals. The proposed method's labels are read from the
results of 12_qc_evaluate.py (DATASET = 'test'). Runs offline
on the cached adjacent data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.qc_benchmarks import nearest_neighbor_order, madsen_allerup_statistic, madsen_allerup_accept
from src.qc_data_loader import load_precip_cache
from src.qc_evaluation import load_validation_data, compute_binned_metrics, compute_group_metrics, PRECIP_BIN_LABELS

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
TUNING_DIR = SCRIPT_DIR.parent / 'outputs' / 'tables' / 'tuning'
VALIDATION_DIR = SCRIPT_DIR.parent / 'outputs' / 'tables' / 'validation'
TUNING_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

NEIGHBOR_HOR = 2
MA_MULTIPLIERS = list(range(1, 21)) + list(range(25, 101, 5)) + [np.inf]  # k values tried on the training set;
# k = inf leaves only the ratio test for neighborhoods with q75 = q25
MA_LITERATURE = {2: 'Vejen et al. 2002', 4: 'Jiang et al. 2015'}

INPUT_STATIONS = DATA_DIR / 'gd_stations_locations.csv'
# (records file, cache covering its hours, subset name)
TRAIN_SETS = [(DATA_DIR / 'gd_validation_data_train.csv',
               DATA_DIR / 'cache' / f"gd_validation_data_train_adjacent{NEIGHBOR_HOR}h.parquet", 'all')]
TEST_SETS = [(DATA_DIR / 'gd_validation_data_test.csv',
              DATA_DIR / 'cache' / f"gd_validation_data_test_adjacent{NEIGHBOR_HOR}h.parquet", 'all')]
INPUT_PROPOSED = [VALIDATION_DIR / 'qc_result_test_outlier60km_extreme80km.csv']

OUTPUT_SWEEP = TUNING_DIR / 'qc_benchmark_madsen_allerup_train_sweep.csv'
OUTPUT_RECORDS = VALIDATION_DIR / 'qc_benchmark_records_test.csv'
OUTPUT_BINNED = VALIDATION_DIR / 'qc_benchmark_binned.csv'
OUTPUT_TABLE = VALIDATION_DIR / 'qc_benchmark_table_by_bin.csv'


def compute_statistics(df_records: pd.DataFrame, cache_file: Path, neighbor_order: dict) -> pd.DataFrame:
    """Madsen-Allerup statistics of every record from the cached values of all stations at its hour."""
    df_cache = load_precip_cache(cache_file)
    rows = []
    for ddatetime, df_group in df_records.groupby('ddatetime'):
        df_hour = df_cache.loc[[ddatetime]]
        precip_by_station = pd.Series(df_hour['r'].to_numpy(), index=df_hour['stacode'].astype(str).to_numpy())
        for idx, record in df_group.iterrows():
            stats = madsen_allerup_statistic(str(record['stacode']), record['r'],
                                             precip_by_station.drop(str(record['stacode']), errors='ignore'),
                                             neighbor_order)
            rows.append({'index': idx, **stats})
    df_stats = pd.DataFrame(rows).set_index('index').sort_index()
    return df_records.join(df_stats)


def load_sets(sets: list[tuple], neighbor_order: dict) -> pd.DataFrame:
    """Records of the labeled sets with their Madsen-Allerup statistics."""
    frames = []
    for records_file, cache_file, _ in sets:
        df = load_validation_data(records_file).drop(columns=['subset'], errors='ignore')
        frames.append(compute_statistics(df, cache_file, neighbor_order))
    return pd.concat(frames, ignore_index=True)


def labels_from_accept(accepted: pd.Series) -> pd.Series:
    return np.where(accepted, 'NORMAL', 'FALSE')


def build_bin_table(df_binned: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """One row per rainfall-intensity bin plus the total; FN, FP, sensitivity and specificity per method."""
    groups = [('all', b) for b in PRECIP_BIN_LABELS] + [('all', 'all')]
    names = {'all': 'all test records'}
    rows = []
    for subset, precip_bin in groups:
        df_group = df_binned[(df_binned['subset'] == subset) & (df_binned['precip_bin'] == precip_bin)]
        first = df_group.iloc[0]
        row = {'rows': names[subset] if precip_bin == 'all' else f"{precip_bin} mm",
               'n_true': int(first['n_true']), 'n_false': int(first['n_false'])}
        for method in methods:
            metrics = df_group[df_group['method'] == method].iloc[0]
            row[f"{method}: FN"] = int(metrics['false negative'])
            row[f"{method}: FP"] = int(metrics['false positive'])
            row[f"{method}: sensitivity"] = metrics['sensitivity']
            row[f"{method}: specificity"] = metrics['specificity']
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Starting Madsen-Allerup benchmark at {datetime.now()}")
    df_stations = pd.read_csv(INPUT_STATIONS, encoding='utf-8-sig', dtype={'stacode': str})
    neighbor_order = nearest_neighbor_order(df_stations)

    # Tune k on the training data
    df_train = load_sets(TRAIN_SETS, neighbor_order)
    sweep_rows = []
    for k in MA_MULTIPLIERS:
        df_train['qc_label'] = labels_from_accept(madsen_allerup_accept(df_train, k))
        metrics = compute_group_metrics(df_train)
        sweep_rows.append({'k': k, 'source': MA_LITERATURE.get(k, ''), **metrics,
                           'youden': metrics['sensitivity'] + metrics['specificity'] - 1})
    df_sweep = pd.DataFrame(sweep_rows)
    k_tuned = float(df_sweep.loc[df_sweep['youden'].idxmax(), 'k'])
    df_sweep['selected'] = df_sweep['k'] == k_tuned
    df_sweep.to_csv(OUTPUT_SWEEP, index=False, encoding='utf-8-sig', float_format='%.4f')
    print(f"Training sweep saved to {OUTPUT_SWEEP.name}; tuned k = {k_tuned:g}")
    print(f"Median / max distance of the 12th neighbor on the training data: "
          f"{df_train['ma_max_dist_km'].median():.1f} / {df_train['ma_max_dist_km'].max():.1f} km")

    # Evaluate on the test data
    df_test = load_sets(TEST_SETS, neighbor_order)
    df_proposed = pd.concat([pd.read_csv(f, encoding='utf-8-sig', dtype={'stacode': str}) for f in INPUT_PROPOSED],
                            ignore_index=True)
    df_proposed['ddatetime'] = pd.to_datetime(df_proposed['ddatetime'])
    df_test = df_test.drop(columns=['qc_label']).merge(
        df_proposed[['stacode', 'ddatetime', 'qc_label']], on=['stacode', 'ddatetime'], how='left', validate='one_to_one')

    methods = {}
    for k, source in MA_LITERATURE.items():
        if k != k_tuned:
            methods[f"Madsen-Allerup (k={k}, {source})"] = labels_from_accept(madsen_allerup_accept(df_test, k))
    methods[f"Madsen-Allerup (k={k_tuned:g}, tuned)"] = labels_from_accept(madsen_allerup_accept(df_test, k_tuned))
    methods['Proposed method'] = df_test['qc_label'].to_numpy()

    binned_frames = []
    for method, labels in methods.items():
        df_binned = compute_binned_metrics(df_test.assign(qc_label=labels))
        df_binned.insert(0, 'method', method)
        binned_frames.append(df_binned)
        df_test[f"label_{method}"] = labels
    df_binned = pd.concat(binned_frames, ignore_index=True)
    df_table = build_bin_table(df_binned, list(methods))

    df_test.to_csv(OUTPUT_RECORDS, index=False, encoding='utf-8-sig')
    df_binned.to_csv(OUTPUT_BINNED, index=False, encoding='utf-8-sig', float_format='%.4f')
    df_table.to_csv(OUTPUT_TABLE, index=False, encoding='utf-8-sig', float_format='%.3f')

    print(df_sweep[['k', 'source', 'true positive', 'false negative', 'true negative', 'false positive',
                    'sensitivity', 'specificity', 'youden']].round(4).to_string(index=False))
    for kind in ['FN', 'FP', 'sensitivity', 'specificity']:
        columns = ['rows', 'n_true', 'n_false'] + [c for c in df_table.columns if c.endswith(f": {kind}")]
        print(f"\n{kind}\n{df_table[columns].round(3).to_string(index=False)}")
    print(f"\nSaved {OUTPUT_RECORDS.name}, {OUTPUT_BINNED.name}, {OUTPUT_TABLE.name}; finished at {datetime.now()}")


if __name__ == '__main__':
    main()
