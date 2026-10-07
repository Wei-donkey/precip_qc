# -*- coding: utf-8 -*-
"""
Section 4b(5): step size of the sliding extreme circles (manuscript Table 10, circle step rows).

The iterative inspection with one fixed circle centered nearest to the target, with coarse sliding
circles (step = radius / 100 degrees, 0.8° for 80 km) and with fine sliding circles (step = radius /
200 degrees, 0.4°), the other parameters at their adopted values. For the training records of MIN_R mm
or more: FN, FP (stage 2 only) and the labels of the records that reached stage 2.

The per-record results of the three settings are also written for 16_plot_cases_circle_gap_comparison.py
(single = fixed circle, few = coarse step, many = fine step). Runs offline; missing stage 2 runs are
computed first.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from src.qc_tuning import (ADOPTED, TABLE_DIR, IterativeSetting, MethodSetting, iterative_experiment,
                           method_frame, save_table, stage1_normal)

MIN_R = ADOPTED.boundary  # mm
CANDIDATES = {'one fixed circle': IterativeSetting(circles='fixed'),
              'coarse step (0.8°)': IterativeSetting(circles='outlier'),
              'fine step (0.4°)': IterativeSetting(circles='extreme')}
CASE_FILES = {'single': 'one fixed circle', 'few': 'coarse step (0.8°)', 'many': 'fine step (0.4°)'}


def export_case_results() -> None:
    """Per-record results of each setting; stage 2 columns cleared for the records accepted by stage 1."""
    normal = stage1_normal(ADOPTED.outlier_radius, ADOPTED.iqr_multiplier)
    for density, candidate in CASE_FILES.items():
        setting = MethodSetting(ADOPTED.outlier_radius, ADOPTED.iqr_multiplier, ADOPTED.boundary,
                                ADOPTED.single_radius, CANDIDATES[candidate])
        df = method_frame(setting).drop(columns='inspected')
        df.loc[normal, ['qc_label', 'extreme_circle_count', 'validation_circle_count', 'validation_circle_locs']] = \
            ['NORMAL', 0, 0, None]
        file = TABLE_DIR / (f"qc_result_train_outlier{ADOPTED.outlier_radius}km_"
                            f"extreme{ADOPTED.iterative.extreme_radius}km_{density}.csv")
        df.reset_index().to_csv(file, index=False, encoding='utf-8-sig')
        print(f"Saved {file}")


def main() -> None:
    save_table(iterative_experiment(CANDIDATES, min_r=MIN_R), 'table10_circle_step')
    export_case_results()


if __name__ == '__main__':
    main()
