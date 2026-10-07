# -*- coding: utf-8 -*-
"""
Section 4b(4): number of iterations of the iterative inspection (manuscript Table 9).

With the adopted parameters, the training records of MIN_R mm or more confirmed by stage 2 are counted
by the iteration in which they were accepted (EXTREME_TYPE1-5). The second table gives FN and FP
(stage 2 only) for each number of iterations in ITERATION_CANDIDATES. Stopping after n iterations
leaves the earlier iterations unchanged and rejects the records confirmed later, so every candidate
is obtained from the run with the full schedule. The schedule has at most EXTREME_ITERATION_LIMIT (5)
iterations, since the confidence coefficient falls by 0.1 per iteration from 0.5. Runs offline; a
missing stage 2 run is computed first.
"""

from __future__ import annotations

import pandas as pd

import _bootstrap  # noqa: F401
from src.qc_tuning import ADOPTED, EXTREME_LABELS, R, TRUE, final_labels, fn_fp, prepare, save_table

ITERATION_CANDIDATES = [1, 2, 3, 4, 5]
MIN_R = ADOPTED.boundary  # mm


def main() -> None:
    if not all(1 <= n <= len(EXTREME_LABELS) for n in ITERATION_CANDIDATES):
        raise ValueError(f"ITERATION_CANDIDATES must lie between 1 and {len(EXTREME_LABELS)}")
    prepare(iterative=[ADOPTED.iterative], min_r=MIN_R)
    labels = final_labels(ADOPTED)
    heavy = R >= MIN_R
    rows = []
    for actual, part in [('TRUE', heavy & TRUE), ('FALSE', heavy & ~TRUE)]:
        counts = labels[part].value_counts()
        row = {'actual': actual, **{label.replace('EXTREME_', ''): int(counts.get(label, 0)) for label in EXTREME_LABELS}}
        row['total'] = sum(row[label.replace('EXTREME_', '')] for label in EXTREME_LABELS)
        rows.append(row)
    save_table(pd.DataFrame(rows), 'table9_iterations')

    rows = []
    for n in ITERATION_CANDIDATES:
        kept = labels.where(~labels.isin(EXTREME_LABELS[n:]), 'FALSE')
        rows.append({'iterations': n, **fn_fp(kept, heavy)})
    save_table(pd.DataFrame(rows), 'table9_fn_fp_by_iterations')


if __name__ == '__main__':
    main()
