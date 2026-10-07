# -*- coding: utf-8 -*-
"""
Shared machinery of the parameter-tuning scripts (04-11, one per subsection of manuscript Section 4b).

Stage 2 judges an outlier from the raw values of its neighbors and never uses the stage 1 result of
another station. The final label of a record is therefore NORMAL if stage 1 accepts it, and its
stage 2 label otherwise. This allows every stage 2 setting to be run once, with every training record
sent to stage 2, and combined with any outlier circle radius and IQR multiplier k afterwards:

- stage 1: per-record acceptance for every outlier circle radius and k
  (04_qc_tune_outlier_radius_multiplier.py, file STAGE1_FILE);
- stage 2: one cached run per setting in STAGE2_DIR, either an iterative inspection
  (IterativeSetting) or a single check (single_check_file); a missing run is computed on request.

The method's stage 2 label is the single check below the intensity boundary and the iterative
inspection from the boundary upward (method_labels). The adopted parameter set is ADOPTED.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.qc_algorithms import (DEFAULT_IQR_MULTIPLIER, EXTREME_CONFIDENCE_COEFF, EXTREME_ITERATION_LIMIT,
                               EXTREME_TCS_THRESHOLD, LOW_INTENSITY_EXTREME_RADIUS, LOW_INTENSITY_THRESHOLD)
from src.qc_evaluation import evaluate_on_cache, load_validation_data

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
TABLE_DIR = ROOT / 'outputs' / 'tables' / 'tuning'
STAGE2_DIR = TABLE_DIR / 'stage2'

DATASET = 'train'
RECORDS_FILE = DATA_DIR / f"gd_validation_data_{DATASET}.csv"
CACHE_FILE = DATA_DIR / 'cache' / f"gd_validation_data_{DATASET}_adjacent2h.parquet"
STATIONS_FILE = DATA_DIR / 'gd_stations_locations.csv'
STAGE1_FILE = TABLE_DIR / f"qc_critical_multiplier_by_radius_{DATASET}.csv"

N_WORKERS = 2  # parallel processes for missing stage 2 runs (each run takes about 30-40 min)

LABELS = ['EXTREME_TYPE1', 'EXTREME_TYPE2', 'EXTREME_TYPE3', 'EXTREME_TYPE4', 'EXTREME_TYPE5', 'FALSE', 'NORMAL']
EXTREME_LABELS = LABELS[:EXTREME_ITERATION_LIMIT]
RESULT_COLUMNS = ['qc_label', 'validation_sample_size', 'extreme_circle_count', 'validation_circle_count',
                  'validation_circle_locs']


@dataclass(frozen=True)
class IterativeSetting:
    """Iterative inspection of stage 2 (applied to the outliers at or above the intensity boundary).

    circles: 'extreme' = sliding extreme circles (step = radius / 200 degrees, e.g. 0.4° for 80 km),
             'outlier' = coarse sliding circles (step = radius / 100 degrees, e.g. 0.8° for 80 km),
             'fixed'   = one circle centered nearest to the target station.
    """
    extreme_radius: int = 80
    circles: str = 'extreme'
    cc0: float = EXTREME_CONFIDENCE_COEFF[0]
    tcs0: int = EXTREME_TCS_THRESHOLD[0]
    window: int = 2

    @property
    def step(self) -> str:
        if self.circles == 'fixed':
            return 'fixed'
        return f"step{self.extreme_radius / (200 if self.circles == 'extreme' else 100):g}"

    @property
    def name(self) -> str:
        return f"iterative_ext{self.extreme_radius}km_{self.step}_cc{self.cc0:g}_tcs{self.tcs0:g}_win{self.window}h"

    @property
    def file(self) -> Path:
        return STAGE2_DIR / f"{self.name}.csv"


@dataclass(frozen=True)
class MethodSetting:
    """Complete parameter set of the method."""
    outlier_radius: int = 60
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER
    boundary: float = LOW_INTENSITY_THRESHOLD
    single_radius: int = LOW_INTENSITY_EXTREME_RADIUS
    iterative: IterativeSetting = IterativeSetting()


ADOPTED = MethodSetting()  # the tuned parameter set (manuscript Table 11)


def single_check_file(radius: int) -> Path:
    """Single check (first-round criteria, same-hour neighbors, one round) with circles of this radius."""
    return STAGE2_DIR / f"single_{radius}km.csv"


def outlier_circles_file(radius: int) -> Path:
    return DATA_DIR / 'neighbor_circles_outlier' / f"neighbor_circles_outlier_{radius}km.csv"


def extreme_circles_file(radius: int) -> Path:
    return DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{radius}km.csv"


# ---------------------------------------------------------------- records and stage 1
def load_records() -> pd.DataFrame:
    """Training records indexed by (stacode, ddatetime) with r and the boolean label 'true'."""
    df = load_validation_data(RECORDS_FILE)
    df = df.set_index(['stacode', 'ddatetime'])
    return pd.DataFrame({'r': df['r'], 'true': df['validation'].astype(bool)})


RECORDS = load_records()
R, TRUE = RECORDS['r'], RECORDS['true']


def stage1_normal(outlier_radius: int, k: float) -> pd.Series:
    """Whether stage 1 accepts each record (column of 04_qc_tune_outlier_radius_multiplier.py)."""
    column = f"normal_{outlier_radius}km_k{k:g}"
    if column not in pd.read_csv(STAGE1_FILE, encoding='utf-8-sig', nrows=0).columns:
        raise KeyError(f"{column} not in {STAGE1_FILE.name}: add the radius/multiplier to "
                       f"04_qc_tune_outlier_radius_multiplier.py and run it")
    df = pd.read_csv(STAGE1_FILE, encoding='utf-8-sig', dtype={'stacode': str}, usecols=['stacode', 'ddatetime', column])
    df['ddatetime'] = pd.to_datetime(df['ddatetime'])
    return df.set_index(['stacode', 'ddatetime'])[column].astype(str).str.upper().eq('TRUE').reindex(RECORDS.index)


# ---------------------------------------------------------------- stage 2 runs
def _run_job(job: dict) -> str:
    """Worker: one stage 2 run with every record sent to stage 2; writes job['file']."""
    start = perf_counter()
    df = evaluate_on_cache(RECORDS_FILE, CACHE_FILE, outlier_circles_file(60), job['extreme_circles'],
                           neighbor_hor=job['window'], tcs_thresholds=job['tcs'], confidence_coeffs=job['cc'],
                           iqr_multiplier=-np.inf, input_low_extreme_circles=job.get('low_circles'),
                           low_intensity_threshold=job.get('low_threshold', 0),
                           input_station_info=job.get('station_info'))
    df['inspected'] = True
    df.to_csv(job['file'], index=False, encoding='utf-8-sig')
    return f"Finished {job['file'].name} in {(perf_counter() - start) / 60:.1f} min"


def _iterative_job(setting: IterativeSetting) -> dict:
    n = EXTREME_ITERATION_LIMIT
    circles = (outlier_circles_file(setting.extreme_radius) if setting.circles == 'outlier'
               else extreme_circles_file(setting.extreme_radius))
    return {'file': setting.file, 'extreme_circles': circles, 'window': setting.window,
            'cc': [round(setting.cc0 - 0.1 * i, 1) for i in range(n)], 'tcs': [setting.tcs0 + i for i in range(n)],
            'station_info': STATIONS_FILE if setting.circles == 'fixed' else None}


def _single_job(radius: int) -> dict:
    return {'file': single_check_file(radius), 'extreme_circles': extreme_circles_file(80), 'window': 0,
            'cc': None, 'tcs': None, 'low_circles': extreme_circles_file(radius), 'low_threshold': np.inf}


def _covers(file: Path, min_r: float) -> bool:
    """Whether a cached run exists and inspected every record of min_r mm or more."""
    if not file.exists():
        return False
    df = pd.read_csv(file, encoding='utf-8-sig', usecols=['r', 'inspected'])
    return bool(df.loc[df['r'] >= min_r, 'inspected'].astype(str).str.upper().eq('TRUE').all())


def prepare(iterative: list[IterativeSetting] = (), single_radii: list[int] = (), min_r: float = 0) -> None:
    """Run the stage 2 settings that are not cached yet (iterative runs must cover records >= min_r)."""
    jobs = [_iterative_job(s) for s in dict.fromkeys(iterative) if not _covers(s.file, min_r)]
    jobs += [_single_job(r) for r in dict.fromkeys(single_radii) if not single_check_file(r).exists()]
    if not jobs:
        return
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(jobs)} stage 2 setting(s) with {N_WORKERS} process(es): "
          f"{[job['file'].name for job in jobs]}", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(N_WORKERS, len(jobs)))) as executor:
        for message in executor.map(_run_job, jobs):
            print(message, flush=True)


def stage2_frame(file: Path) -> pd.DataFrame:
    """All result columns of a cached stage 2 run, aligned with RECORDS."""
    df = pd.read_csv(file, encoding='utf-8-sig', dtype={'stacode': str})
    df['ddatetime'] = pd.to_datetime(df['ddatetime'])
    for column in ['validation_sample_size', 'extreme_circle_count', 'validation_circle_count']:
        df[column] = df[column].astype('Int64')  # counts stay integers where some runs have empty rows
    return df.set_index(['stacode', 'ddatetime']).reindex(RECORDS.index)


def method_frame(setting: MethodSetting) -> pd.DataFrame:
    """Stage 2 result columns of the method: single check below the boundary, iterative inspection above."""
    df = stage2_frame(setting.iterative.file)
    low = R < setting.boundary
    if low.any():
        df.loc[low] = stage2_frame(single_check_file(setting.single_radius)).loc[low]
    return df


def final_labels(setting: MethodSetting, stage2: pd.Series | None = None) -> pd.Series:
    """Final label of every record: NORMAL if stage 1 accepts it, else its stage 2 label."""
    if stage2 is None:
        stage2 = method_frame(setting)['qc_label']
    return stage2.where(~stage1_normal(setting.outlier_radius, setting.iqr_multiplier), 'NORMAL')


# ---------------------------------------------------------------- counts
def fn_fp(labels: pd.Series, mask: pd.Series, stage2_only: bool = True) -> dict:
    """Genuine records rejected (FN) and false records accepted (FP) among mask.
    With stage2_only, FP counts only the false records accepted by stage 2 (not those accepted by stage 1)."""
    accepted = labels != 'FALSE'
    fp_accepted = accepted & (labels != 'NORMAL') if stage2_only else accepted
    return {'FN': int((mask & TRUE & ~accepted).sum()), 'FP': int((mask & ~TRUE & fp_accepted).sum())}


def label_counts(labels: pd.Series, mask: pd.Series) -> dict:
    """Records of mask that reached stage 2, by actual label and QC label (TYPE1-5 and FALSE)."""
    out = {}
    for actual, part in [('TRUE', mask & TRUE), ('FALSE', mask & ~TRUE)]:
        counts = labels[part & (labels != 'NORMAL')].value_counts()
        out.update({f"{actual}:{label}": int(counts.get(label, 0)) for label in EXTREME_LABELS + ['FALSE']})
    return out


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Write a tuning table to outputs/tables/tuning/ and print it."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    file = TABLE_DIR / f"{name}.csv"
    df.to_csv(file, index=False, encoding='utf-8-sig')
    print(f"\n== {name} ==\n{df.to_string(index=False)}\nSaved {file}")
    return file


def iterative_experiment(candidates: dict[str, IterativeSetting], min_r: float = LOW_INTENSITY_THRESHOLD,
                         base: MethodSetting = ADOPTED) -> pd.DataFrame:
    """One parameter of the iterative inspection varied with the others at their adopted values: FN, FP
    (stage 2 only) and the labels of the records of min_r mm or more that reached stage 2 (Tables 9-10)."""
    prepare(iterative=list(candidates.values()), min_r=min_r)
    heavy = R >= min_r
    rows = []
    for name, iterative in candidates.items():
        setting = MethodSetting(base.outlier_radius, base.iqr_multiplier, base.boundary, base.single_radius, iterative)
        labels = final_labels(setting)
        rows.append({'setting': name, **fn_fp(labels, heavy), **label_counts(labels, heavy)})
    return pd.DataFrame(rows)
