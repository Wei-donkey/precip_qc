# -*- coding: utf-8 -*-
"""
Section 4b(7): Initial total confidence score (TCS) threshold of the iterative inspection; the
threshold increases by 1 per iteration (for NMO targets it is 1 lower) (manuscript Table 10, initial
TCS rows).

Each candidate is run with the other parameters at their adopted values. For the training records of
MIN_R mm or more: FN, FP (stage 2 only) and the labels of the records that reached stage 2. Runs
offline; missing stage 2 runs are computed first (about 30-40 min each).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from src.qc_tuning import ADOPTED, IterativeSetting, iterative_experiment, save_table

MIN_R = ADOPTED.boundary  # mm
TCS0_CANDIDATES = [1, 2, 3]


def main() -> None:
    candidates = {f"TCS {tcs0} -> {tcs0 + 4}": IterativeSetting(tcs0=tcs0) for tcs0 in TCS0_CANDIDATES}
    save_table(iterative_experiment(candidates, min_r=MIN_R), 'table10_tcs_threshold')


if __name__ == '__main__':
    main()
