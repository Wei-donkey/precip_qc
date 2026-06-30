# -*- coding: utf-8 -*-
"""
Plot Neighbor Circles Heatmap with Contours.

Visualizes the number of nearby stations within 60-km-radius circles
as a heatmap with contour lines showing different count ranges.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata

# Import style settings from qc_plot_utils
import sys
sys.path.insert(0, str(Path(__file__).parent))
from qc_plot_utils import set_plot_style

# Constants
FIGSIZE = (12, 8)
DPI = 300
DATA_FILE = Path(__file__).parent.parent / 'data' / 'neighbor_circles_extreme' / 'neighbor_circles_extreme_60km.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qc_result_analysis'
OUTPUT_FILE = OUTPUT_DIR / 'neighbor_circles_heatmap.png'

# Margin settings (adjustable)
MARGIN_PARAMS = {
    'left': 0.04,
    'right': 0.93,
    'bottom': 0.06,
    'top': 0.96,
    'wspace': 0.01,
    'hspace': 0.01
}


def load_neighbor_data(csv_path: Path) -> pd.DataFrame:
    """Load neighbor circles data."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # Verify required columns exist
    required_cols = ['lon', 'lat', 'count']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data file")
    
    print(f"Loaded {len(df)} grid points")
    print(f"Count range: {df['count'].min():.0f} to {df['count'].max():.0f}")
    
    return df


def plot_neighbor_heatmap_with_contours(df: pd.DataFrame, output_path: Path):
    """Create heatmap with contour lines showing station count ranges."""
    set_plot_style()
    
    print("Generating neighbor circles heatmap with contours...")
    
    # Get actual data range for dynamic level calculation
    min_count = df['count'].min()
    max_count = df['count'].max()
    print(f"Data range: {min_count:.0f} to {max_count:.0f}")
    print(f"Total grid points: {len(df)}")
    
    # Dynamically calculate discrete colorbar levels
    # Use meaningful thresholds that match the reference image style
    bounds = [67, 100, 200, 300, 400, 500, 600, 800, 1000, 1118]
    # Filter out levels beyond data range
    bounds = [b for b in bounds if b <= max_count * 1.05]
    if bounds[-1] < max_count:
        bounds.append(int(max_count))
    
    # Contour levels (excluding 100 and 200 as requested)
    contour_levels = [300, 400, 500, 600, 800, 1000]
    contour_levels = [l for l in contour_levels if l < max_count * 0.98]
    
    print(f"Colorbar bounds: {bounds}")
    print(f"Contour levels: {contour_levels}")
    
    # Calculate cell boundaries from irregular grid points
    # Sort by longitude and latitude to find regular spacing
    lon_sorted = np.sort(df['lon'].unique())
    lat_sorted = np.sort(df['lat'].unique())
    
    # Calculate average spacing to determine cell size
    if len(lon_sorted) > 1:
        lon_spacing = np.mean(np.diff(lon_sorted))
    else:
        lon_spacing = 0.3  # Default
    
    if len(lat_sorted) > 1:
        lat_spacing = np.mean(np.diff(lat_sorted))
    else:
        lat_spacing = 0.3  # Default
    
    print(f"Average grid spacing: lon={lon_spacing:.2f}, lat={lat_spacing:.2f}")
    
    # Create cell boundaries (half-spacing around each point)
    lon_edges = np.concatenate([
        [lon_sorted[0] - lon_spacing/2],
        lon_sorted[:-1] + lon_spacing/2,
        [lon_sorted[-1] + lon_spacing/2]
    ])
    
    lat_edges = np.concatenate([
        [lat_sorted[0] - lat_spacing/2],
        lat_sorted[:-1] + lat_spacing/2,
        [lat_sorted[-1] + lat_spacing/2]
    ])
    
    # Create 2D grid matrix filled with NaN
    grid_matrix = np.full((len(lat_sorted), len(lon_sorted)), np.nan)
    
    # Fill in values at corresponding positions
    for _, row in df.iterrows():
        lon_idx = np.argmin(np.abs(lon_sorted - row['lon']))
        lat_idx = np.argmin(np.abs(lat_sorted - row['lat']))
        grid_matrix[lat_idx, lon_idx] = row['count']
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    
    # Create discrete colormap
    cmap_discrete = mpl.colors.ListedColormap([
        '#C0C0C0',  # 67-100 (gray)
        '#00FFFF',  # 100-200 (cyan)
        '#00CED1',  # 200-300 (dark cyan)
        '#7FFFD4',  # 300-400 (aquamarine)
        '#90EE90',  # 400-500 (light green)
        '#ADFF2F',  # 500-600 (green yellow)
        '#FFFF00',  # 600-800 (yellow)
        '#FFA500',  # 800-1000 (orange)
        '#FF4500',  # 1000+ (orange red)
    ])
    
    norm = mpl.colors.BoundaryNorm(bounds, cmap_discrete.N)
    
    # Plot filled cells using pcolormesh with discrete colors
    im = ax.pcolormesh(lon_edges, lat_edges, grid_matrix, 
                      cmap=cmap_discrete, norm=norm,
                      shading='flat', edgecolors='none', linewidth=0)
    
    # Add black contour lines (excluding 100 and 200)
    contours = ax.tricontour(mpl.tri.Triangulation(df['lon'], df['lat']), 
                            df['count'], levels=contour_levels, 
                            colors='black', linewidths=1.8, linestyles='solid')
    
    # Label contours with black color
    clabels = ax.clabel(contours, inline=True, fontsize=10, fmt='%d', colors='black')
    # Make labels bold by setting font properties
    for label in clabels:
        label.set_fontweight('bold')
    
    # Configure axes
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title('Number of Nearby Stations', fontsize=14, pad=5)
    
    # Set axis limits with padding
    lon_margin = (df['lon'].max() - df['lon'].min()) * 0.05
    lat_margin = (df['lat'].max() - df['lat'].min()) * 0.05
    ax.set_xlim(df['lon'].min() - lon_margin, df['lon'].max() + lon_margin)
    ax.set_ylim(df['lat'].min() - lat_margin, df['lat'].max() + lat_margin)
    
    # Add grid lines at EVERY cell boundary (not just 1-degree intervals)
    # Draw vertical grid lines at each longitude edge
    for lon_edge in lon_edges:
        ax.axvline(x=lon_edge, color='black', linestyle='-', linewidth=0.8, alpha=0.7, zorder=10)
    
    # Draw horizontal grid lines at each latitude edge
    for lat_edge in lat_edges:
        ax.axhline(y=lat_edge, color='black', linestyle='-', linewidth=0.8, alpha=0.7, zorder=10)
    
    # Disable default matplotlib grid since we're drawing custom grid lines
    ax.grid(False)
    
    # Set tick label font size
    ax.tick_params(axis='both', labelsize=10)
    
    # Create discrete colorbar
    cbar_position = [0.94, 0.06, 0.02, 0.90]
    cbar_ax = fig.add_axes(cbar_position)
    cbar = plt.colorbar(im, cax=cbar_ax, boundaries=bounds, ticks=bounds[::1])
    cbar.set_label('Station Count', fontsize=10, rotation=90, labelpad=2)
    cbar.ax.tick_params(labelsize=8)

    # Adjust margins
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
    plt.savefig(output_path, dpi=DPI,  facecolor='white')
    print(f"Heatmap saved to {output_path}")
    plt.close()


def main():
    """Main function to generate neighbor circles heatmap."""
    print("=" * 70)
    print("Neighbor Circles Heatmap Visualization")
    print("=" * 70)
    
    # Load data
    if not DATA_FILE.exists():
        print(f"Error: Data file not found: {DATA_FILE}")
        return
    
    df = load_neighbor_data(DATA_FILE)
    
    # Create plot
    plot_neighbor_heatmap_with_contours(df, OUTPUT_FILE)
    
    print("\nVisualization complete!")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
