# -*- coding: utf-8 -*-
"""
Benchmark spatial consistency checks for comparison with the proposed QC method.

Implements the Madsen-Allerup check (Vejen et al. 2002, Section 4.2): for a target value x at
hour t, the concurrent values of the nearest neighbor stations give the median M and the
quartiles q25, q75, and the test statistic is T = (x - M) / (q75 - q25). The value is flagged
when |T| > k. When q75 = q25 (mostly dry neighbors) the statistic is undefined and the report's
second test is used instead: flag when x / sum(x, neighbors) > 0.6. As in the proposed method,
values below the QC threshold (1 mm) are accepted without testing (the report uses 4 mm for
daily sums). Designed to be imported by the benchmark script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.qc_algorithms import DEFAULT_QC_THRESHOLD

EARTH_RADIUS_KM = 6371.0
MA_NEIGHBORS = 12  # number of neighbors recommended by Vejen et al. (2002)
MA_RATIO_LIMIT = 0.6  # test for q75 = q25: x / sum of all values


def nearest_neighbor_order(df_stations: pd.DataFrame, n_candidates: int = 60) -> dict:
    """For every station, the codes and distances (km) of its n_candidates nearest other stations."""
    codes = df_stations['stacode'].astype(str).to_numpy()
    lat = np.radians(df_stations['lat'].to_numpy(dtype=float))
    lon = np.radians(df_stations['lon'].to_numpy(dtype=float))

    order = {}
    for i, code in enumerate(codes):
        dlat, dlon = lat - lat[i], lon - lon[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2
        dist = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
        dist[i] = np.inf  # exclude the station itself
        nearest = np.argsort(dist)[:n_candidates]
        order[code] = (codes[nearest], dist[nearest])
    return order


def madsen_allerup_statistic(stacode: str, target_r: float, precip_by_station: pd.Series,
                             neighbor_order: dict, n_neighbors: int = MA_NEIGHBORS) -> dict:
    """
    Test statistic of one record from the concurrent values of its n_neighbors nearest reporting stations.
    precip_by_station holds the values of all stations at the target hour, indexed by station code.
    """
    candidates, distances = neighbor_order[stacode]
    reporting = pd.Index(candidates).isin(precip_by_station.index)
    neighbor_codes = candidates[reporting][:n_neighbors]
    neighbor_dist = distances[reporting][:n_neighbors]
    values = precip_by_station.loc[neighbor_codes].to_numpy(dtype=float)

    if len(values) == 0:
        return {'ma_n_neighbors': 0}

    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    iqr = q75 - q25
    total = target_r + values.sum()
    return {
        'ma_n_neighbors': len(values),
        'ma_max_dist_km': float(neighbor_dist.max()),
        'ma_median': median,
        'ma_iqr': iqr,
        'ma_T': (target_r - median) / iqr if iqr > 0 else np.nan,
        'ma_ratio': target_r / total if total > 0 else np.nan,
    }


def madsen_allerup_accept(df_stats: pd.DataFrame, k: float, qc_threshold: float = DEFAULT_QC_THRESHOLD,
                          ratio_limit: float = MA_RATIO_LIMIT) -> pd.Series:
    """True where the record is accepted (not flagged) by the Madsen-Allerup check with multiplier k."""
    below_threshold = df_stats['r'] < qc_threshold
    iqr_positive = df_stats['ma_iqr'] > 0
    flagged_T = iqr_positive & (df_stats['ma_T'].abs() > k)
    flagged_ratio = ~iqr_positive & (df_stats['ma_ratio'] > ratio_limit)
    no_neighbors = df_stats['ma_n_neighbors'].fillna(0) == 0
    return below_threshold | no_neighbors | ~(flagged_T | flagged_ratio)


if __name__ == '__main__':
    print("This module is intended to be imported, not run directly.")
