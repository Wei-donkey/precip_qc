# -*- coding: utf-8 -*-
"""
Section 4b(6): Initial confidence coefficient CC0 of the iterative inspection; the coefficient
decreases by 0.1 per iteration, so the ending CC is CC0 - 0.4 (manuscript Table 10, confidence
coefficient rows).

Each candidate is run with the other parameters at their adopted values. For the training records of
MIN_R mm or more: FN, FP (stage 2 only) and the labels of the records that reached stage 2. Runs
offline; missing stage 2 runs are computed first (about 30-40 min each).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from src.qc_tuning import ADOPTED, IterativeSetting, iterative_experiment, save_table

MIN_R = ADOPTED.boundary  # mm
CC0_CANDIDATES = [0.5, 0.6, 0.7]


def main() -> None:
    candidates = {f"CC {cc0:g} -> {cc0 - 0.4:.1f}": IterativeSetting(cc0=cc0) for cc0 in CC0_CANDIDATES}
    save_table(iterative_experiment(candidates, min_r=MIN_R), 'table10_confidence_coefficient')


if __name__ == '__main__':
    main()
