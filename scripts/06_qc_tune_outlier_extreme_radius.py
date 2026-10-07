# -*- coding: utf-8 -*-
"""
Section 4b(3): outlier circle radius and extreme circle radius of the iterative inspection
(manuscript Table 8).

The candidate outlier circle radii of Section 4b(1) are combined with extreme circle radii of
10-100 km; the other parameters keep their adopted values. Evaluated on the training records of
MIN_R mm or more, the records the iterative inspection applies to. Each cell gives FN / FP: genuine
records rejected and false records accepted by stage 2 (the false records accepted by stage 1, listed
in the second table, are not counted). Runs offline; missing stage 2 runs are computed first.
"""

from __future__ import annotations

import pandas as pd

import _bootstrap  # noqa: F401
from src.qc_tuning import (ADOPTED, R, TRUE, IterativeSetting, MethodSetting, final_labels, fn_fp, prepare, save_table,
                           stage1_normal)

OUTLIER_RADII = [60, 70, 80, 90, 100]  # km
EXTREME_RADII = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # km
MIN_R = ADOPTED.boundary  # mm


def main() -> None:
    settings = {e: IterativeSetting(extreme_radius=e) for e in EXTREME_RADII}
    prepare(iterative=list(settings.values()), min_r=MIN_R)
    heavy = R >= MIN_R
    rows = []
    for extreme, iterative in settings.items():
        row = {'extreme_radius_km': extreme}
        for outlier in OUTLIER_RADII:
            setting = MethodSetting(outlier, ADOPTED.iqr_multiplier, ADOPTED.boundary, ADOPTED.single_radius, iterative)
            counts = fn_fp(final_labels(setting), heavy)
            row[f"outlier_{outlier}km"] = f"{counts['FN']} / {counts['FP']}"
        rows.append(row)
    save_table(pd.DataFrame(rows), 'table8_outlier_extreme_radius')
    stage1 = [{'outlier_radius_km': o, 'false_accepted_by_stage1':
               int((heavy & ~TRUE & stage1_normal(o, ADOPTED.iqr_multiplier)).sum())} for o in OUTLIER_RADII]
    save_table(pd.DataFrame(stage1), 'table8_stage1_false_accepted')


if __name__ == '__main__':
    main()
