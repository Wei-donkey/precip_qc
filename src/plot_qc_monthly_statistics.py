# -*- coding: utf-8 -*-
"""
Plot QC Monthly Statistics with Broken Y-Axis.

Visualizes monthly QC statistics from 2003-2025, showing:
1. Total records per month (bar chart on primary y-axis)
2. FALSE count overlay (area chart on primary y-axis)
3. FALSE percentage trend (line chart on secondary y-axis)

Uses broken y-axis to handle large range (50 to 4 million).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime

# Import style settings from qc_plot_utils
import sys
sys.path.insert(0, str(Path(__file__).parent))
from qc_plot_utils import set_plot_style, STYLE_SETTINGS

# Constants
FIGSIZE = (14, 7)
DPI = 300
DATA_FILE = Path(__file__).parent.parent / 'data' / 'qc_monthly_statistics_2003-2025.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qc_result_analysis'

OUTPUT_FILE = OUTPUT_DIR / f'qc_monthly_statistics.png'

# Break points for y-axis
BREAK1 = 80  # First break to show FALSE counts
BREAK2 = 100000  # Second break to show monthly variations

# Margin settings (adjustable)
MARGIN_PARAMS = {
    'left': 0.08,
    'right': 0.92,
    'bottom': 0.08,
    'top': 0.99,  # Further reduced to minimize gap between title and upper subplot
    'wspace': 0.01,
    'hspace': 0.01
}


def load_qc_statistics(csv_path: Path) -> pd.DataFrame:
    """Load and process QC monthly statistics."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # Calculate total records
    qc_columns = ['EXTREME_TYPE1', 'EXTREME_TYPE2', 'EXTREME_TYPE3', 
                  'EXTREME_TYPE4', 'EXTREME_TYPE5', 'FALSE', 'NORMAL']
    df['TOTAL'] = df[qc_columns].sum(axis=1)
    
    # Create a combined year-month identifier for x-axis
    df['year_month'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2))
    
    # Sort by date
    df = df.sort_values('year_month').reset_index(drop=True)
    
    print(f"Loaded {len(df)} monthly records")
    print(f"Total records range: {df['TOTAL'].min():,.0f} to {df['TOTAL'].max():,.0f}")
    print(f"FALSE records range: {df['FALSE'].min():,.0f} to {df['FALSE'].max():,.0f}")
    
    return df


def plot_monthly_statistics(df: pd.DataFrame, output_path: Path):
    """Create monthly statistics plot with broken y-axis."""
    set_plot_style()
    
    # Prepare data
    x_positions = np.arange(len(df))
    total_counts = df['TOTAL'].values
    false_counts = df['FALSE'].values
    false_percentages = (false_counts / total_counts * 100) if total_counts.sum() > 0 else np.zeros_like(total_counts)
    
    # Create figure with three subplots for broken y-axis
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    
    # Define height ratios for three sections
    # Lower: 0-50, Middle: 50-100000, Upper: 100000-max
    height_ratios = [2, 0.5, 3]  # Adjust based on visual balance
    gs = fig.add_gridspec(3, 1, height_ratios=height_ratios, hspace=0.05)
    
    # Create three axes
    ax_upper = fig.add_subplot(gs[0])  # Upper section (>100,000)
    ax_middle = fig.add_subplot(gs[1])  # Middle section (50-100,000)
    ax_lower = fig.add_subplot(gs[2])  # Lower section (0-50)

    # === Plot on Upper Axis (Total counts > 100,000) ===
    # Total records as area chart
    ax_upper.fill_between(x_positions, total_counts, alpha=0.3, 
                          color='steelblue', label='Total Records', zorder=1)
    ax_upper.plot(x_positions, total_counts, color='steelblue', 
                  linewidth=1.5, alpha=0.8, zorder=2)
    
    # FALSE counts as area chart on top
    ax_upper.fill_between(x_positions, false_counts, alpha=0.5, 
                          color='#FF6B6B', label='FALSE Count', zorder=3)
    ax_upper.plot(x_positions, false_counts, color='#CC0000', 
                  linewidth=2, alpha=0.9, zorder=4)
    
    ax_upper.set_ylim(BREAK2,)  # Auto-scale upper limit
    
    # Set grid to background AFTER all plotting (zorder=0)
    ax_upper.grid(True, zorder=0)
    
    # === Plot on Middle Axis (Transition zone - mostly empty) ===
    # This section shows the gap between 50 and 100,000
    # Total records as area chart
    ax_middle.fill_between(x_positions, total_counts, alpha=0.3, 
                           color='steelblue', zorder=1)
    ax_middle.plot(x_positions, total_counts, color='steelblue', 
                   linewidth=1.5, alpha=0.8, zorder=2)
    ax_middle.set_ylim(BREAK1, BREAK2)
    
    # Set grid to background AFTER all plotting (zorder=0)
    ax_middle.grid(True, zorder=0)
    
    # === Plot on Lower Axis (FALSE counts 0-50) ===
    # FALSE area chart in lower section
    ax_lower.fill_between(x_positions, false_counts, alpha=0.5, 
                          color='#FF6B6B', label='FALSE Count', zorder=5)
    ax_lower.plot(x_positions, false_counts, color='#CC0000', 
                  linewidth=1.5, alpha=0.9, zorder=5)
    ax_lower.set_ylim(0, BREAK1)
    
    # === Add Secondary Y-Axis for FALSE Percentage ===
    # Add to lower axis for better visibility
    ax_lower_secondary = ax_lower.twinx()
    ax_lower_secondary.plot(x_positions, false_percentages, 
                                              color='green', linewidth=1.5,
                                              marker='o', markersize=0,
                                              zorder=5)
    ax_lower_secondary.fill_between(x_positions, false_percentages, alpha=0.5, 
                          color='green', label='FALSE %', zorder=5)

    # Configure secondary y-axis with fixed range 0-2%
    ax_lower_secondary.set_ylabel('FALSE Percentage (%)', fontsize=8, color='green')
    ax_lower_secondary.tick_params(axis='y', labelcolor='green', labelsize=8)
    ax_lower_secondary.set_ylim(0, 0.02)  # Fixed range 0-2%
    
    # Set grid to background AFTER all plotting (zorder=0)
    ax_lower.grid(True, zorder=0)
    ax_lower_secondary.grid(False)  # Disable grid on secondary axis to avoid conflicts
    
    # === Format Axes ===
    # Hide spines between axes
    ax_upper.spines.bottom.set_visible(False)
    ax_middle.spines.top.set_visible(False)
    ax_middle.spines.bottom.set_visible(False)
    ax_lower.spines.top.set_visible(False)
    
    # Hide x-axis ticks on upper and middle
    ax_upper.xaxis.tick_top()
    ax_upper.tick_params(labeltop=False)
    ax_middle.tick_params(labelbottom=False, labeltop=False)
    ax_lower.xaxis.tick_bottom()
    
    # Add break marks using diagonal lines
    d = .5  # proportion of vertical to horizontal extent
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                  linestyle="none", color='k', mec='k', mew=0.8, clip_on=False)
    
    # Break between upper and middle
    ax_upper.plot([0, 1], [0, 0], transform=ax_upper.transAxes, **kwargs)
    ax_middle.plot([0, 1], [1, 1], transform=ax_middle.transAxes, **kwargs)
    
    # Break between middle and lower
    ax_middle.plot([0, 1], [0, 0], transform=ax_middle.transAxes, **kwargs)
    ax_lower.plot([0, 1], [1, 1], transform=ax_lower.transAxes, **kwargs)
    
    # === X-Axis Configuration (Year labels only) ===
    # Show only year labels at the first month of each year
    year_labels = []
    year_positions = []
    for i, row in df.iterrows():
        if row['month'] == 1:  # January
            year_labels.append(str(int(row['year'])))
            year_positions.append(i)
    
    # Set synchronized x-ticks across ALL subplots to align grid lines
    ax_upper.set_xticks(year_positions)
    ax_middle.set_xticks(year_positions)
    ax_lower.set_xticks(year_positions)
    
    # Only show labels on lower axis
    ax_lower.set_xticklabels(year_labels, rotation=45, ha='right', fontsize=8)
    ax_lower.set_xlabel('Year', fontsize=8)
    
    # === Y-Axis Labels ===
    ax_lower.set_ylabel('Record Count', fontsize=9)
    
    # Format y-axis tick labels
    def format_large_numbers(x, pos):
        if x >= 1000000:
            return f'{x/1000000:.1f}M'
        elif x >= 1000:
            return f'{x/1000:.0f}K'
        else:
            return f'{int(x)}'
    
    ax_lower.yaxis.set_major_formatter(mticker.FuncFormatter(format_large_numbers))
    ax_upper.yaxis.set_major_formatter(mticker.FuncFormatter(format_large_numbers))
    
    # Set tick locations
    ax_lower.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax_upper.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    
    # Set tick label sizes to match secondary y-axis (fontsize=8)
    ax_upper.tick_params(axis='y', labelsize=9)
    ax_middle.tick_params(axis='y', labelsize=9)
    ax_lower.tick_params(axis='y', labelsize=9)
    
    # === Legend ===
    # Combine handles from different axes
    handles_upper, labels_upper = ax_upper.get_legend_handles_labels()
    handles_lower, labels_lower = ax_lower_secondary.get_legend_handles_labels()
    
    all_handles = handles_upper + handles_lower
    all_labels = labels_upper + labels_lower
    
    ax_upper.legend(all_handles, all_labels, loc='upper left', 
                    fontsize=8, framealpha=0.8, edgecolor='gray',
                    frameon=True)
    
    # === Title ===
    fig.suptitle('Monthly QC Statistics (2003-2025)', 
                 fontsize=12, y=0.98)
        
    # Adjust margins AFTER creating subplots
    fig.subplots_adjust(
        left=MARGIN_PARAMS['left'],
        right=MARGIN_PARAMS['right'],
        bottom=MARGIN_PARAMS['bottom'],
        top=MARGIN_PARAMS['top'],
        wspace=MARGIN_PARAMS['wspace'],
        hspace=MARGIN_PARAMS['hspace']
    )
    

    # === Save Figure ===
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    """Main function to generate QC monthly statistics plot."""
    print("=" * 70)
    print("QC Monthly Statistics Visualization")
    print("=" * 70)
    
    # Load data
    if not DATA_FILE.exists():
        print(f"Error: Data file not found: {DATA_FILE}")
        return
    
    df = load_qc_statistics(DATA_FILE)
    
    # Create plot
    plot_monthly_statistics(df, OUTPUT_FILE)
    
    print("\nVisualization complete!")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
