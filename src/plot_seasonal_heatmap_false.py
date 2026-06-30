# -*- coding: utf-8 -*-
"""
Plot Seasonal Heatmap for FALSE Records.

Generates a heatmap showing the seasonal pattern of FALSE QC records
from 2003-2025, with years on x-axis and months on y-axis.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not installed. Using matplotlib fallback for heatmap.")

# Import style settings from qc_plot_utils
import sys
sys.path.insert(0, str(Path(__file__).parent))
from qc_plot_utils import set_plot_style, STYLE_SETTINGS

# Constants
FIGSIZE = (14, 7)
DPI = 300
DATA_FILE = Path(__file__).parent.parent / 'data' / 'qc_monthly_statistics_2003-2025.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qc_result_analysis'
OUTPUT_FILE = OUTPUT_DIR / 'seasonal_heatmap_false.png'

# Margin settings (adjustable)
MARGIN_PARAMS = {
    'left': 0.10,
    'right': 0.96,
    'bottom': 0.08,
    'top': 0.95,
    'wspace': 0.01,
    'hspace': 0.01
}


def load_qc_statistics(csv_path: Path) -> pd.DataFrame:
    """Load and process QC monthly statistics."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # Verify FALSE column exists
    if 'FALSE' not in df.columns:
        raise ValueError("FALSE column not found in data file")
    
    print(f"Loaded {len(df)} monthly records")
    print(f"FALSE records range: {df['FALSE'].min():,.0f} to {df['FALSE'].max():,.0f}")
    
    return df


def plot_seasonal_heatmap_false(df: pd.DataFrame, output_path: Path):
    """Create seasonal heatmap for FALSE records (year × month)."""
    set_plot_style()
    
    print("Generating seasonal heatmap for FALSE records...")
    
    # Create pivot table: month as rows (y-axis), year as columns (x-axis)
    pivot_data = df.pivot_table(values='FALSE', index='month', columns='year')
    
    # Reverse month order so Jan is at bottom and Dec is at top
    pivot_data = pivot_data.iloc[::-1]
    
    # Create figure
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    
    # Create main axes for heatmap with precise positioning
    # Leave space on right for colorbar
    ax_position = [0.10, 0.08, 0.78, 0.84]  # [left, bottom, width, height]
    ax = fig.add_axes(ax_position)
    
    # Plot heatmap
    if HAS_SEABORN:
        # Use seaborn heatmap without default colorbar
        heatmap = sns.heatmap(
            pivot_data, 
            cmap='YlOrRd', 
            annot=True, 
            fmt='.0f',
            linewidths=0.5,
            linecolor='white',
            cbar=False,  # Disable default colorbar
            ax=ax,
            square=False,
            xticklabels=True,
            yticklabels=True
        )
        
        # Create custom colorbar axes with tight spacing
        cbar_position = [0.89, 0.08, 0.02, 0.84]  # [left, bottom, width, height]
        cbar_ax = fig.add_axes(cbar_position)
        
        # Add colorbar to custom axes
        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=pivot_data.values.min(), vmax=pivot_data.values.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('FALSE Count', fontsize=10, rotation=90, labelpad=10)
        cbar.ax.tick_params(labelsize=8)
    else:
        # Matplotlib fallback
        im = ax.imshow(pivot_data.values, aspect='auto', cmap='YlOrRd')
        
        # Create custom colorbar axes
        cbar_position = [0.89, 0.08, 0.02, 0.84]
        cbar_ax = fig.add_axes(cbar_position)
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('FALSE Count', fontsize=10, rotation=90, labelpad=10)
        cbar.ax.tick_params(labelsize=8)
        
        # Set tick labels
        ax.set_xticks(range(pivot_data.shape[1]))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticks(range(pivot_data.shape[0]))
        ax.set_yticklabels(pivot_data.index)

        # Add annotations
        max_val = pivot_data.values.max()
        for i in range(pivot_data.shape[0]):
            for j in range(pivot_data.shape[1]):
                val = pivot_data.values[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val > max_val * 0.6 else 'black'
                    ax.text(j, i, f'{val:.0f}',
                           ha='center', va='center',
                           fontsize=8, color=text_color, fontweight='bold')
    
    # Configure axes
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Month', fontsize=12)
    ax.set_title('Seasonal Pattern: FALSE Records', 
                fontsize=14, pad=10)
    
    ax.grid(False)
    
    # Format month labels on y-axis (reversed order: Jan at bottom, Dec at top)
    month_labels = ['Dec', 'Nov', 'Oct', 'Sep', 'Aug', 'Jul',
                   'Jun', 'May', 'Apr', 'Mar', 'Feb', 'Jan']
    ax.set_yticklabels(month_labels, rotation=0, ha='right', fontsize=9)
    
    # Ensure all years are shown on x-axis
    years = pivot_data.columns.tolist()
    ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
    
    # Adjust margins using configurable parameters
    fig.subplots_adjust(
        left=MARGIN_PARAMS['left'],
        right=MARGIN_PARAMS['right'],
        bottom=MARGIN_PARAMS['bottom'],
        top=MARGIN_PARAMS['top'],
        wspace=MARGIN_PARAMS['wspace'],
        hspace=MARGIN_PARAMS['hspace']
    )

    # Save figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"Heatmap saved to {output_path}")
    plt.close()


def main():
    """Main function to generate FALSE seasonal heatmap."""
    print("=" * 70)
    print("FALSE Seasonal Heatmap Visualization")
    print("=" * 70)
    
    # Load data
    if not DATA_FILE.exists():
        print(f"Error: Data file not found: {DATA_FILE}")
        return
    
    df = load_qc_statistics(DATA_FILE)
    
    # Create plot
    plot_seasonal_heatmap_false(df, OUTPUT_FILE)
    
    print("\nVisualization complete!")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
