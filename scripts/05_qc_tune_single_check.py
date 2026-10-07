# -*- coding: utf-8 -*-
"""
Section 4b(2): circle radius and intensity boundary of the single check (manuscript Table 7).

Below the intensity boundary, outliers get the single check: first-round criteria, same-hour
neighbors, one round. Its radius and the boundary are decided from the single check alone: every
outlier of the training set is inspected with the single check, with outlier circles of
OUTLIER_RADIUS km (the largest candidate of Section 4b(1), which forwards the most records to
stage 2). Per intensity range, the table gives the false records that reach stage 2, those missed
(accepted) by the single check with circles of each radius in SINGLE_RADII, and the genuine records
rejected with the radii in REJECTED_RADII (share of the genuine records of the range in brackets).
Runs offline; missing single-check runs are computed first (about 30-40 min each).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.qc_tuning import ADOPTED, R, TRUE, MethodSetting, final_labels, prepare, save_table, stage1_normal

OUTLIER_RADIUS = 100  # km
IQR_MULTIPLIER = ADOPTED.iqr_multiplier
SINGLE_RADII = [10, 20, 30, 50, 80]  # km, false records missed
REJECTED_RADII = [10, 20]  # km, genuine records rejected
RANGES = [1, 5, 10, 15, 20, 30, 40, 50, 100, np.inf]  # mm, left-closed
TOTAL_BELOW = 20  # mm, a total row for the ranges below this value


def main() -> None:
    radii = sorted(set(SINGLE_RADII) | set(REJECTED_RADII))
    prepare(single_radii=radii)
    normal = stage1_normal(OUTLIER_RADIUS, IQR_MULTIPLIER)
    single = {r: final_labels(MethodSetting(outlier_radius=OUTLIER_RADIUS, iqr_multiplier=IQR_MULTIPLIER,
                                            boundary=np.inf, single_radius=r)) for r in radii}

    def row(name: str, rng: pd.Series) -> dict:
        reach = ~TRUE & rng & ~normal
        out = {'intensity_mm': name, 'false_reaching_stage2': int(reach.sum())}
        out.update({f"false_missed_{r}km": int((reach & (single[r] != 'FALSE')).sum()) for r in SINGLE_RADII})
        n_true = int((TRUE & rng).sum())
        for r in REJECTED_RADII:
            rejected = int((TRUE & rng & (single[r] == 'FALSE')).sum())
            out[f"genuine_rejected_{r}km"] = f"{rejected} ({100 * rejected / n_true:.1f}%)" if n_true else '0'
        return out

    rows = []
    for low, high in zip(RANGES[:-1], RANGES[1:]):
        name = f"> {low:g}" if np.isinf(high) else f"{low:g}-{high:g}"
        rows.append(row(name, (R >= low) & (R < high)))
        if high == TOTAL_BELOW:
            rows.append(row(f"{RANGES[0]:g}-{TOTAL_BELOW:g} (total)", (R >= RANGES[0]) & (R < TOTAL_BELOW)))
    save_table(pd.DataFrame(rows), 'table7_single_check')


if __name__ == '__main__':
    main()
