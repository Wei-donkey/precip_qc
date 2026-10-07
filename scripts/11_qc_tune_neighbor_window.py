# -*- coding: utf-8 -*-
"""
Section 4b(8): Neighbor time window of the iterative inspection: neighbors' records within ±window
hours of the target hour (manuscript Table 10, neighbor time window rows).

Each candidate is run with the other parameters at their adopted values. For the training records of
MIN_R mm or more: FN, FP (stage 2 only) and the labels of the records that reached stage 2. Runs
offline; missing stage 2 runs are computed first (about 30-40 min each).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from src.qc_tuning import ADOPTED, IterativeSetting, iterative_experiment, save_table

MIN_R = ADOPTED.boundary  # mm
WINDOW_CANDIDATES = [0, 1, 2]  # hours


def main() -> None:
    candidates = {f"±{hours} h": IterativeSetting(window=hours) for hours in WINDOW_CANDIDATES}
    save_table(iterative_experiment(candidates, min_r=MIN_R), 'table10_neighbor_window')


if __name__ == '__main__':
    main()
