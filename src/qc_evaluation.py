# -*- coding: utf-8 -*-
"""
Evaluation utilities for the precipitation spatial consistency QC method.

Provides the per-record QC (stage 1 outlier detection followed by stage 2 extreme
inspection), confusion counts and metrics, metrics per rainfall intensity bin with
Wilson confidence intervals, and the evaluation of all labeled records on cached data.
Designed to be imported by the evaluation and tuning scripts.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.qc_algorithms import (
    calculate_p_max,
    perform_outlier_detection,
    perform_extreme_inspection,
    DEFAULT_CLIMATE_LIMIT,
    DEFAULT_QC_THRESHOLD,
    DEFAULT_IQR_MULTIPLIER,
    LOW_INTENSITY_THRESHOLD,
    LOW_INTENSITY_TCS_THRESHOLD,
)
from src.qc_circle_process_utils import find_single_extreme_circle
from src.qc_data_loader import load_circles, load_precip_cache, load_station_info, slice_precip_cache

QC_LABELS = ['EXTREME_TYPE1', 'EXTREME_TYPE2', 'EXTREME_TYPE3', 'EXTREME_TYPE4', 'EXTREME_TYPE5', 'FALSE', 'NORMAL']
RESULT_COLUMNS = ['qc_label', 'validation_sample_size', 'extreme_circle_count',
                  'validation_circle_count', 'validation_circle_locs']

# Intensity bins (mm), left-closed: [1, 5), [5, 10), ..., [100, inf)
PRECIP_BINS = [1, 5, 10, 20, 30, 50, 100, np.inf]
PRECIP_BIN_LABELS = ['1-5', '5-10', '10-20', '20-30', '30-50', '50-100', '>100']
CI_Z = 1.96  # z value of the 95% Wilson confidence interval


def load_validation_data(file_path: Path) -> pd.DataFrame:
    """Load labeled records from CSV file and initialize the result columns."""
    df = pd.read_csv(file_path, encoding='utf-8-sig', dtype={'stacode': str})
    df['ddatetime'] = pd.to_datetime(df['ddatetime'])
    df['validation'] = df['validation'].astype(str).str.upper().map({'TRUE': True, 'FALSE': False})
    for column in RESULT_COLUMNS:
        df[column] = None
    return df


def qc_single_record(stacode: str, statype: str, precip: float, ddatetime: pd.Timestamp,
                     df_all_adj: pd.DataFrame, df_outlier_circles: pd.DataFrame, df_extreme_circles: pd.DataFrame,
                     qc_threshold: float = DEFAULT_QC_THRESHOLD, climate_limit: float = DEFAULT_CLIMATE_LIMIT,
                     loop_all_circle: bool = True, tcs_thresholds: list[float] | None = None,
                     confidence_coeffs: list[float] | None = None,
                     iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
                     df_low_extreme_circles: pd.DataFrame | None = None,
                     low_intensity_threshold: float = LOW_INTENSITY_THRESHOLD,
                     df_station_info: pd.DataFrame | None = None) -> dict:
    """
    Apply outlier detection (stage 1) and, for outliers, extreme inspection (stage 2) to one record.
    df_all_adj holds all stations' data within the neighbor time window around ddatetime.
    If df_low_extreme_circles is given (two-segment method), outliers below low_intensity_threshold
    get the single check: those extreme circles, same-hour neighbors, and the first round only.
    If df_station_info is given, the iterative inspection uses only the extreme circle whose center
    is closest to the station (one fixed circle, as in the step-size test).
    Returns the result columns to store; empty if the record has no data.
    """
    if df_all_adj.empty:
        return {}

    # Current hour precipitation data for all stations
    df_all = df_all_adj[df_all_adj['ddatetime'] == ddatetime].copy()
    if df_all.empty:
        return {}

    df_all['qc_label'] = None
    df_all['validation_sample_size'] = None

    # Perform outlier circle inspection on all station data
    perform_outlier_detection(df_all=df_all, df_outlier_circles=df_outlier_circles, qc_threshold=qc_threshold,
                              iqr_multiplier=iqr_multiplier)

    df_current = df_all[df_all['stacode'] == stacode]
    if df_current.empty:
        return {}
    qc_label = df_current['qc_label'].iloc[0]

    if qc_label == 'NORMAL':
        return {
            'qc_label': 'NORMAL',
            'validation_sample_size': df_current['validation_sample_size'].iloc[0],
            'extreme_circle_count': 0,
            'validation_circle_count': 0,
            'validation_circle_locs': None,
        }

    # Two-segment method: single check for low-intensity outliers
    single_check = df_low_extreme_circles is not None and precip < low_intensity_threshold
    if single_check:
        df_extreme_circles = df_low_extreme_circles
        df_all_adj = df_all_adj[df_all_adj['ddatetime'] == ddatetime]
        tcs_thresholds = LOW_INTENSITY_TCS_THRESHOLD
        confidence_coeffs = None

    if df_station_info is not None and not single_check:
        filtered_extreme_circles = find_single_extreme_circle(df_extreme_circles, stacode, df_station_info)
    else:
        # Find extreme circles containing this outlier station
        extreme_circles_mask = df_extreme_circles['neighbors'].apply(lambda neighbors: stacode in neighbors)
        filtered_extreme_circles = df_extreme_circles[extreme_circles_mask]

    # Perform extreme circle evaluation for OUTLIERS
    qc_label, validation_sample_size, \
        extreme_circle_count, validation_circle_count, validation_circle_locs \
        = perform_extreme_inspection(
        target_stacode=stacode,
        target_statype=statype,
        target_precip=precip,
        filtered_extreme_circles=filtered_extreme_circles,
        df_all_adjacent=df_all_adj,
        loop_all_circle=loop_all_circle,
        climate_limit=climate_limit,
        tcs_thresholds=tcs_thresholds,
        confidence_coeffs=confidence_coeffs,
    )

    return {
        'qc_label': qc_label,
        'validation_sample_size': validation_sample_size,
        'extreme_circle_count': extreme_circle_count,
        'validation_circle_count': validation_circle_count,
        'validation_circle_locs': validation_circle_locs,
    }


def stage1_critical_multiplier(stacode: str, df_all: pd.DataFrame, df_outlier_circles: pd.DataFrame,
                               qc_threshold: float = DEFAULT_QC_THRESHOLD) -> float:
    """
    Smallest IQR multiplier k at which stage 1 labels the record NORMAL, i.e. the minimum over the
    outlier circles containing the station of (r - Q3) / IQR (same circle rules as perform_outlier_detection).
    -inf for values below qc_threshold; inf if no circle can accept it (always OUTLIER).
    Because stage 2 does not depend on k, a record's final label at any k is NORMAL if k >= this value
    and its stage 2 label otherwise.
    """
    r_target = df_all.loc[df_all['stacode'] == stacode, 'r']
    if r_target.empty:
        return np.nan
    r_target = r_target.iloc[0]
    if r_target < qc_threshold:
        return -np.inf

    critical = np.inf
    for neighbor_stations in df_outlier_circles['neighbors']:
        if stacode not in neighbor_stations:
            continue
        values = df_all.loc[df_all['stacode'].isin(neighbor_stations), 'r']
        if values.empty or values.max() < qc_threshold:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1 if q3 > q1 else 0.1
        critical = min(critical, (r_target - q3) / iqr)
    return critical


def stage1_normal_by_multiplier(stacode: str, df_all: pd.DataFrame, df_outlier_circles: pd.DataFrame,
                                multipliers: list[float], qc_threshold: float = DEFAULT_QC_THRESHOLD) -> dict:
    """
    Whether stage 1 labels the record NORMAL for each multiplier k, using exactly the comparison of
    perform_outlier_detection (r <= Q3 + k * IQR in any circle containing the station), so that
    values lying on the fence are classified as in the method itself.
    """
    r_target = df_all.loc[df_all['stacode'] == stacode, 'r']
    if r_target.empty:
        return {k: None for k in multipliers}
    r_target = r_target.iloc[0]
    if r_target < qc_threshold:
        return {k: True for k in multipliers}

    normal = {k: False for k in multipliers}
    for neighbor_stations in df_outlier_circles['neighbors']:
        if stacode not in neighbor_stations:
            continue
        values = df_all.loc[df_all['stacode'].isin(neighbor_stations), 'r']
        if values.empty or values.max() < qc_threshold:
            continue
        for k in multipliers:
            if not normal[k] and r_target <= calculate_p_max(values, k):
                normal[k] = True
    return normal


def count_confusion(df_qc_result: pd.DataFrame) -> dict:
    """
    Count confusion outcomes. Positive = the record is accepted as TRUE (NORMAL or EXTREME_TYPE1-5);
    negative = labeled FALSE. Records without a qc_label (no data) are counted as unprocessed.
    """
    processed = df_qc_result['qc_label'].notna()
    accepted = processed & (df_qc_result['qc_label'] != 'FALSE')
    rejected = processed & (df_qc_result['qc_label'] == 'FALSE')
    actual_true = df_qc_result['validation'].astype(bool)

    return {
        'false positive': int((accepted & ~actual_true).sum()),
        'false negative': int((rejected & actual_true).sum()),
        'true positive': int((accepted & actual_true).sum()),
        'true negative': int((rejected & ~actual_true).sum()),
        'unprocessed': int((~processed).sum()),
    }


def compute_confusion_metrics(counts: dict, total_records: int) -> pd.Series:
    """Overall metrics (accuracy over all records, as in the original stats files)."""
    tp, tn, fp, fn = counts['true positive'], counts['true negative'], counts['false positive'], counts['false negative']
    series = pd.Series(dtype=float)
    for key in ['false positive', 'false negative', 'true positive', 'true negative']:
        series[key] = counts[key]

    series['accuracy'] = (tp + tn) / total_records
    series['precision'] = tp / (tp + fp) if tp + fp else np.nan
    series['sensitivity'] = tp / (tp + fn) if tp + fn else np.nan
    series['specificity'] = tn / (tn + fp) if tn + fp else np.nan
    precision, sensitivity = series['precision'], series['sensitivity']
    series['F1 score'] = 2 * precision * sensitivity / (precision + sensitivity)

    return series


def wilson_interval(successes: int, trials: int, z: float = CI_Z) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if trials == 0:
        return np.nan, np.nan
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return center - half_width, center + half_width


def compute_group_metrics(df_group: pd.DataFrame) -> dict:
    """Confusion counts, sensitivity (TRUE retained) and specificity (FALSE detected) with 95% CIs."""
    counts = count_confusion(df_group)
    tp, tn, fp, fn = counts['true positive'], counts['true negative'], counts['false positive'], counts['false negative']

    sens_low, sens_high = wilson_interval(tp, tp + fn)
    spec_low, spec_high = wilson_interval(tn, tn + fp)
    return {
        'n_true': int(df_group['validation'].sum()),
        'n_false': int((~df_group['validation'].astype(bool)).sum()),
        **counts,
        'sensitivity': tp / (tp + fn) if tp + fn else np.nan,
        'sensitivity_ci_low': sens_low,
        'sensitivity_ci_high': sens_high,
        'specificity': tn / (tn + fp) if tn + fp else np.nan,
        'specificity_ci_low': spec_low,
        'specificity_ci_high': spec_high,
        'precision': tp / (tp + fp) if tp + fp else np.nan,
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else np.nan,
    }


def compute_binned_metrics(df_qc_result: pd.DataFrame) -> pd.DataFrame:
    """Metrics per intensity bin (plus 'all'), for every subset present (plus 'all')."""
    df_tmp = df_qc_result.copy()
    df_tmp['precip_bin'] = pd.cut(df_tmp['r'], bins=PRECIP_BINS, labels=PRECIP_BIN_LABELS, right=False)
    subsets = ['all'] + (sorted(df_tmp['subset'].dropna().unique()) if 'subset' in df_tmp.columns else [])

    rows = []
    for subset in subsets:
        df_subset = df_tmp if subset == 'all' else df_tmp[df_tmp['subset'] == subset]
        for precip_bin in ['all'] + PRECIP_BIN_LABELS:
            df_group = df_subset if precip_bin == 'all' else df_subset[df_subset['precip_bin'] == precip_bin]
            if df_group.empty:
                continue
            rows.append({'subset': subset, 'precip_bin': precip_bin, **compute_group_metrics(df_group)})

    return pd.DataFrame(rows)


def compute_qc_label_count(df_qc_result: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Counts of each qc_label for actual TRUE and FALSE records."""
    rows = []
    for validation_status in [True, False]:
        labels = df_qc_result.loc[df_qc_result['validation'] == validation_status, 'qc_label']
        rows.append({'Dataset': dataset, 'Actual': 'TRUE' if validation_status else 'FALSE',
                     **{label: int((labels == label).sum()) for label in QC_LABELS}})
    return pd.DataFrame(rows, columns=['Dataset', 'Actual'] + QC_LABELS)


def evaluate_on_cache(input_records: Path, input_cache: Path, input_outlier_circles: Path,
                      input_extreme_circles: Path, neighbor_hor: int = 2,
                      tcs_thresholds: list[float] | None = None,
                      confidence_coeffs: list[float] | None = None,
                      iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
                      input_low_extreme_circles: Path | None = None,
                      low_intensity_threshold: float = LOW_INTENSITY_THRESHOLD,
                      input_station_info: Path | None = None) -> pd.DataFrame:
    """
    Run the QC method on all labeled records using cached adjacent data (no database access).
    input_low_extreme_circles switches on the two-segment method (see qc_single_record);
    input_station_info switches the iterative inspection to one fixed circle per station.
    iqr_multiplier = -inf sends every record of qc_threshold or more to stage 2.
    """
    df_outlier_circles = load_circles(input_outlier_circles)
    df_extreme_circles = load_circles(input_extreme_circles)
    df_low_extreme_circles = load_circles(input_low_extreme_circles) if input_low_extreme_circles else None
    df_station_info = load_station_info(input_station_info).astype({'stacode': str}) if input_station_info else None
    df_records = load_validation_data(input_records)
    df_cache = load_precip_cache(input_cache)

    for idx, record in df_records.iterrows():
        ddatetime = record['ddatetime']
        df_all_adj = slice_precip_cache(df_cache, ddatetime - timedelta(hours=neighbor_hor),
                                        ddatetime + timedelta(hours=neighbor_hor))
        qc_result = qc_single_record(str(record['stacode']), record['statype'], record['r'], ddatetime,
                                     df_all_adj, df_outlier_circles, df_extreme_circles,
                                     tcs_thresholds=tcs_thresholds, confidence_coeffs=confidence_coeffs,
                                     iqr_multiplier=iqr_multiplier, df_low_extreme_circles=df_low_extreme_circles,
                                     low_intensity_threshold=low_intensity_threshold,
                                     df_station_info=df_station_info)
        for key, value in qc_result.items():
            df_records.at[idx, key] = value

    return df_records

