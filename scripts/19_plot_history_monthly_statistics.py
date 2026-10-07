# -*- coding: utf-8 -*-
"""
Plot QC monthly statistics with a broken y axis.

Visualizes monthly QC statistics from 2003-2025, showing:
1. Total records per month (area chart on the primary y axis)
2. FALSE records of 20 mm or more (area chart on the primary y axis)
3. Their proportion of the records of 20 mm or more over the preceding 12 months (area chart on a secondary y axis)
Uses a broken y axis (three parts) to handle the large range (tens to millions).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.qc_plot_utils import set_plot_style, TICK_SIZE

SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR.parent / 'outputs' / 'tables' / 'application'
OUTPUT_DIR = SCRIPT_DIR.parent / 'outputs' / 'figures' / 'qc_result_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = TABLE_DIR / 'qc_monthly_statistics_2003-2025_5mm.csv'  # QC of the full AWS archive (values below 5 mm accepted without inspection)
OUTPUT_FILE = OUTPUT_DIR / 'qc_monthly_statistics.png'

FIGSIZE = (12, 6)  # full-page width, same as the other manuscript figures
DPI = 300

QC_COLUMNS = ['EXTREME_TYPE1', 'EXTREME_TYPE2', 'EXTREME_TYPE3',
              'EXTREME_TYPE4', 'EXTREME_TYPE5', 'FALSE', 'SUSPECT', 'NORMAL']
# FALSE: records of 20 mm or more rejected by the iterative inspection. The single-check rejections below 20 mm
# (SUSPECT) are kept in the archive and not plotted. The FALSE rate is relative to the records of 20 mm or more
# (RECORDS_GE20).

# Break points of the y axis
BREAK1 = None  # upper limit of the lower part (FALSE counts); None = from the data (rounded up)
BREAK2 = 100000  # upper limit of the middle part (transition)
HEIGHT_RATIOS = [2, 0.5, 3]  # upper, middle, lower parts
FALSE_PCT_YLIM = None  # secondary y axis (%); None = from the data (rounded up)
RATE_WINDOW_MONTHS = 12  # months summed for the FALSE proportion


def nice_ceiling(value: float) -> float:
    """Smallest of 1, 2, 2.5, 5 x 10^n that is at least value."""
    exponent = 10 ** np.floor(np.log10(value))
    return next(m * exponent for m in (1, 2, 2.5, 5, 10) if m * exponent >= value)


def load_qc_statistics(csv_path: Path) -> pd.DataFrame:
    """Load QC monthly statistics and add total records and a year-month date."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    df['TOTAL'] = df[[c for c in QC_COLUMNS if c in df]].sum(axis=1)
    df['year_month'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2))
    df = df.sort_values('year_month').reset_index(drop=True)

    print(f"Loaded {len(df)} monthly records")
    print(f"Total records range: {df['TOTAL'].min():,.0f} to {df['TOTAL'].max():,.0f}")
    print(f"FALSE records range: {df['FALSE'].min():,.0f} to {df['FALSE'].max():,.0f}")
    return df


def format_large_numbers(x, pos):
    """Tick labels in K / M units."""
    if x >= 1000000:
        return f'{x/1000000:.1f}M'
    elif x >= 1000:
        return f'{x/1000:.0f}K'
    return f'{int(x)}'


def subplot_monthly_statistics(fig, gs, df: pd.DataFrame):
    """Total records, FALSE counts and FALSE percentage per month on a three-part broken y axis."""
    x_positions = np.arange(len(df))
    total_counts = df['TOTAL'].values
    false_counts = df['FALSE'].values
    # proportion over the preceding 12 months: a dry month has too few records of 20 mm or more for a monthly proportion
    false_12m = df['FALSE'].rolling(RATE_WINDOW_MONTHS, min_periods=RATE_WINDOW_MONTHS).sum().values
    records_12m = df['RECORDS_GE20'].rolling(RATE_WINDOW_MONTHS, min_periods=RATE_WINDOW_MONTHS).sum().values
    false_percentages = np.divide(false_12m * 100.0, records_12m, out=np.full(len(df), np.nan), where=records_12m > 0)

    gs_inner = gs.subgridspec(3, 1, height_ratios=HEIGHT_RATIOS, hspace=0.01)
    ax_upper = fig.add_subplot(gs_inner[0])  # total counts > BREAK2
    ax_middle = fig.add_subplot(gs_inner[1])  # transition BREAK1-BREAK2
    ax_lower = fig.add_subplot(gs_inner[2])  # FALSE counts 0-BREAK1

    # === Upper part: total records and FALSE counts ===
    ax_upper.fill_between(x_positions, total_counts, alpha=0.3,
                          color='steelblue', label='Total Records', zorder=1)
    ax_upper.plot(x_positions, total_counts, color='steelblue', linewidth=1.5, alpha=0.8, zorder=2)
    ax_upper.fill_between(x_positions, false_counts, alpha=0.5,
                          color='#FF6B6B', label='FALSE (≥ 20 mm)', zorder=3)
    ax_upper.plot(x_positions, false_counts, color='#CC0000', linewidth=2, alpha=0.9, zorder=4)
    ax_upper.set_ylim(BREAK2,)  # auto-scale upper limit
    ax_upper.grid(True, zorder=0)

    # === Middle part: transition zone (total records only) ===
    ax_middle.fill_between(x_positions, total_counts, alpha=0.3, color='steelblue', zorder=1)
    ax_middle.plot(x_positions, total_counts, color='steelblue', linewidth=1.5, alpha=0.8, zorder=2)
    ax_middle.set_ylim(BREAK1 or nice_ceiling(false_counts.max() * 1.05), BREAK2)
    ax_middle.grid(True, zorder=0)

    # === Lower part: FALSE counts, with the FALSE percentage on a secondary y axis ===
    ax_lower.fill_between(x_positions, false_counts, alpha=0.5,
                          color='#FF6B6B', label='FALSE (≥ 20 mm)', zorder=5)
    ax_lower.plot(x_positions, false_counts, color='#CC0000', linewidth=1.5, alpha=0.9, zorder=5)
    break1 = BREAK1 or nice_ceiling(false_counts.max() * 1.05)
    ax_lower.set_ylim(0, break1)
    ax_lower.grid(True, zorder=0)

    ax_lower_secondary = ax_lower.twinx()
    ax_lower_secondary.plot(x_positions, false_percentages, color='green', linewidth=1.5,
                            marker='o', markersize=0, zorder=5)
    ax_lower_secondary.fill_between(x_positions, false_percentages, alpha=0.5,
                                    color='green', label='FALSE proportion of records ≥ 20 mm, 12 months (%)', zorder=5)
    ax_lower_secondary.set_ylabel('FALSE proportion (%)', fontsize=TICK_SIZE, color='green')
    ax_lower_secondary.tick_params(axis='y', labelcolor='green', labelsize=TICK_SIZE)
    ax_lower_secondary.set_ylim(*(FALSE_PCT_YLIM or (0, nice_ceiling(np.nanmax(false_percentages) * 1.05))))
    ax_lower_secondary.grid(False)  # avoid a second set of grid lines

    # === Broken-axis styling and break marks ===
    ax_upper.spines.bottom.set_visible(False)
    ax_middle.spines.top.set_visible(False)
    ax_middle.spines.bottom.set_visible(False)
    ax_lower.spines.top.set_visible(False)
    ax_upper.xaxis.tick_top()
    ax_upper.tick_params(labeltop=False)
    ax_middle.tick_params(labelbottom=False, labeltop=False)
    ax_lower.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the break marks
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                  linestyle="none", color='k', mec='k', mew=0.8, clip_on=False)
    ax_upper.plot([0, 1], [0, 0], transform=ax_upper.transAxes, **kwargs)
    ax_middle.plot([0, 1], [1, 1], transform=ax_middle.transAxes, **kwargs)
    ax_middle.plot([0, 1], [0, 0], transform=ax_middle.transAxes, **kwargs)
    ax_lower.plot([0, 1], [1, 1], transform=ax_lower.transAxes, **kwargs)

    # === X axis: year labels at January, same ticks on all parts so grid lines align ===
    january = df['month'] == 1
    year_positions = df.index[january].tolist()
    year_labels = df.loc[january, 'year'].astype(int).astype(str).tolist()
    for ax in [ax_upper, ax_middle, ax_lower]:
        ax.set_xticks(year_positions)
    ax_lower.set_xticklabels(year_labels, rotation=45, ha='right', fontsize=TICK_SIZE)
    ax_lower.set_xlabel('Year', fontsize=TICK_SIZE)

    # === Y axes ===
    ax_lower.set_ylabel('Record Count', fontsize=TICK_SIZE)
    for ax in [ax_upper, ax_lower]:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_large_numbers))
    ax_lower.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax_upper.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    for ax in [ax_upper, ax_middle, ax_lower]:
        ax.tick_params(axis='y', labelsize=TICK_SIZE)

    # === Legend combining the primary and secondary axes ===
    handles_upper, labels_upper = ax_upper.get_legend_handles_labels()
    handles_lower, labels_lower = ax_lower_secondary.get_legend_handles_labels()
    ax_upper.legend(handles_upper + handles_lower, labels_upper + labels_lower, loc='upper left',
                    fontsize=TICK_SIZE, framealpha=0.8, edgecolor='gray', frameon=True)


def main() -> None:
    print(f"Starting QC monthly statistics figure at {datetime.now()}")
    set_plot_style()

    df = load_qc_statistics(INPUT_FILE)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(1, 1, left=0.06, right=0.92, top=0.97, bottom=0.12)

    subplot_monthly_statistics(fig, gs[0], df)

    fig.savefig(OUTPUT_FILE, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Finished: {datetime.now()}")


if __name__ == '__main__':
    main()
