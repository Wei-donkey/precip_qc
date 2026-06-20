# -*- coding: utf-8 -*-
"""
Plot meteorological station distribution and network in Guangdong Province.

Visualizes surf-type and awst-type stations with 50km-radius grid circles,
including time series of station count evolution.
"""

from __future__ import annotations

import matplotlib as mpl

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from datetime import datetime

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / 'data'
OUTPUT_DIR = SRC_DIR.parent / 'figures' / 'station_maps'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATION_FILE = DATA_DIR / 'gd_stations_locations.csv'
SHP_FILE = DATA_DIR / 'external' / 'chn_province_border.shp'
OUTPUT_FILE = OUTPUT_DIR / 'gd_station_network.png'

FIGSIZE = (14, 10)
DPI = 300

# period of calculating station count
YEAR_STT, YEAR_END = 2003, 2025

# map extent for Guangdong Province
LAT_MIN, LAT_MAX = 19.0, 26.0
LON_MIN, LON_MAX = 109.0, 118.0

RADIUS_KM = 60  # radius for grid node circles
NEIGHBOR_CIRCLES_FILE = DATA_DIR / 'neighbor_circles_outlier' / f"neighbor_circles_outlier_{RADIUS_KM}km.csv"

STYLE_SETTINGS = {
    'style': 'seaborn-v0_8-darkgrid',
    'grid.linewidth': 0.5,
    'font.size': 6,
    'lines.linewidth': 1.0,
    'figure.titlesize': 8,
    'figure.dpi': 300,
    'legend.fontsize': 6,
    'legend.frameon': True,
    'legend.framealpha': 0.5,
    'legend.facecolor': 'inherit',
    'legend.edgecolor': 'white',
}

LAND = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='face', facecolor="#ffffff")
OCEAN = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                    edgecolor='face', facecolor="#78a2e0")
COASTLINE = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                        edgecolor='black', facecolor='none')
BORDER = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m', 
                                        edgecolor='black', facecolor='none')


def set_plot_style():
    """Apply consistent plot styling."""
    for key, value in STYLE_SETTINGS.items():
        if key == 'style':
            mpl.style.use(STYLE_SETTINGS['style'])
        else:
            mpl.rcParams[key] = value


def load_station_info(station_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load station data and separate by station type."""
    df = pd.read_csv(station_file, encoding='utf-8-sig')
    
    # Separate by station type
    surf_stations = df[df['statype'] == 'surf'].copy()
    awst_stations = df[df['statype'] == 'awst'].copy()
    
    return surf_stations, awst_stations


def load_neighbor_circles(grid_file: Path) -> pd.DataFrame:
    """Load grid node data from CSV file."""
    df = pd.read_csv(grid_file, encoding='utf-8-sig')
    return df


def count_stations_by_year(surf_stations: pd.DataFrame, awst_stations: pd.DataFrame, 
                                     start_year: int, 
                                     end_year: int) -> pd.DataFrame:
    """Calculate the number of active stations for each year."""

    surf_stations['date_stt'] = pd.to_datetime(surf_stations['date_stt'])
    awst_stations['date_stt'] = pd.to_datetime(awst_stations['date_stt'])
    
    # Generate year range
    years = list(range(start_year, end_year + 1))
    
    results = []
    for year in years:
        year_end = pd.Timestamp(f'{year}-12-31')
        
        # Count stations that were active during this year
        # A station is active if it started before or during the year
        active_stations = surf_stations[surf_stations['date_stt'] <= year_end]        
        surf_count = len(active_stations[active_stations['statype'] == 'surf'])
        
        active_stations = awst_stations[awst_stations['date_stt'] <= year_end] 
        awst_count = len(active_stations[active_stations['statype'] == 'awst'])

        total_count = surf_count + awst_count
        
        results.append({
            'year': year,
            'surf_count': surf_count,
            'awst_count': awst_count,
            'total_count': total_count,
        })
    
    return pd.DataFrame(results)


def plot_station_timeline(fig, surf_stations: pd.DataFrame, awst_stations: pd.DataFrame,):
    """Plot station count time series."""
    # --- Add Station Count Time Series Plot ---
    print("Calculating yearly station counts...")
    yearly_counts = count_stations_by_year(surf_stations, awst_stations, YEAR_STT, YEAR_END)
    
    # Create inset axes in lower right corner
    ax_inset = fig.add_axes([0.5, 0.09, 0.49, 0.24])   
  
    # Plot surf-type stations
    ax_inset.plot(yearly_counts['year'], yearly_counts['surf_count'], 
                    'b-o', markersize=6, linewidth=1.5, label='National Observatory Station', zorder=2)
    
    # Plot awst-type stations
    ax_inset.plot(yearly_counts['year'], yearly_counts['awst_count'], 
                    'r-o', markersize=6, linewidth=1.5, label='Automatic Weather Station', zorder=2)
    
    # Customize inset plot
    ax_inset.set_xlabel('Year', fontsize=10) #, fontweight='bold'
    ax_inset.set_ylabel('Number of Stations', fontsize=10) #, fontweight='bold'
    ax_inset.grid(True, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_inset.legend(loc='upper left', fontsize=10, framealpha=0.9, 
                    frameon=True, edgecolor='black')
    ax_inset.set_xlim(YEAR_STT-0.5, YEAR_END+0.5)
    
    # Set x-axis ticks to show every 2 years
    ax_inset.set_xticks(range(YEAR_STT, YEAR_END+1, 2))
    ax_inset.tick_params(axis='both', labelsize=9)
    
    # Add background color to distinguish from map
    ax_inset.set_facecolor('#fafafa')


def plot_north_arrow(ax_map):
    # --- Add North Arrow ---
    # Position in axes coordinates (0-1)
    arrow_x, arrow_y = 0.95, 0.95 
    arrow_length = 0.04
    
    # Draw arrow using annotate and 'N' label
    ax_map.annotate('', xy=(arrow_x, arrow_y), xycoords='axes fraction',
                    xytext=(arrow_x, arrow_y - arrow_length), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))   
    ax_map.text(arrow_x, arrow_y + 0.01, 'N', transform=ax_map.transAxes,
                ha='center', va='bottom', fontsize=12, fontweight='bold')


def plot_neighbor_circles(ax_map, grid_nodes):
    # Plot RADIUS_KM circles around grid nodes
    if grid_nodes is not None and not grid_nodes.empty:
        print(f"Adding {len(grid_nodes)} grid node circles...")
        
        # Create circle patches for each grid node
        for _, row in grid_nodes.iterrows():
            lon = row['lon']
            lat = row['lat']
            count = row['count']
            radius_lat = row['radius_lat']
            radius_lon = row['radius_lon']
            
            # Create ellipse to account for projection distortion
            circle = mpatches.Ellipse(
                (lon, lat),
                width=2 * radius_lon,
                height=2 * radius_lat,
                transform=ccrs.PlateCarree(),
                facecolor='none',
                edgecolor='green',
                linewidth=1.5,
                alpha=0.9,
                zorder=7,
            )
            ax_map.add_patch(circle)
        
        # Add a dummy patch for legend
        circle_legend = mpatches.Patch(
            facecolor='none',
            edgecolor='green',
            linewidth=1,
            alpha=0.9,
        )
        
        # Update legend to include grid circles
        handles, labels = ax_map.get_legend_handles_labels()
        handles.append(circle_legend)
        labels.append(f"{RADIUS_KM}-km-radius Circle")  # ({len(grid_nodes)} nodes)
        
        legend = ax_map.legend(
            handles=handles,
            labels=labels,
            loc='lower left',
            frameon=True,
            framealpha=0.9,
            fontsize=11,
        )
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.0)
    else:
        # Original legend without grid circles
        legend = ax_map.legend(
            loc='lower left',
            frameon=True,
            framealpha=0.9,
            fontsize=11,
        )
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.0)


def plot_station_markers(ax_map, stations: pd.DataFrame, scatter_color: str, scatter_size: int, label: str,zorder: int):
    
    # Plot awst-type stations (red circles)
    if not stations.empty:
        ax_map.scatter(
            stations['lon'],
            stations['lat'],
            transform=ccrs.PlateCarree(),
            c=scatter_color,
            edgecolor='white',
            linewidths=0.2,
            marker='o',
            s=scatter_size,
            alpha=0.9,
            label=label,
            zorder=zorder,
        )


def plot_gridlines(ax_map):
    # Add gridlines
    gl = ax_map.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.7,
        linestyle='-',
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'rotation': 0}
    gl.ylabel_style = {'size': 10, 'rotation': 0}

    gl.xlocator = mticker.FixedLocator(np.arange(LON_MIN, LON_MAX + 1, 1))
    gl.ylocator = mticker.FixedLocator(np.arange(LAT_MIN, LAT_MAX + 1, 1))


def plot_geo_feature(ax_map):
    # Add geographic features    
    # ax_map.add_feature(LAND, zorder=0)
    ax_map.add_feature(OCEAN, zorder=0)
    ax_map.add_feature(COASTLINE, linewidth=0.5, zorder=0)

def plot_provincial_boundaries(ax_map):
    # load China's provincial boundary including Guangdong from a shapefile
    chn_boundary = gpd.read_file(SHP_FILE)
    ax_map.add_geometries(chn_boundary.geometry, crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=0.5, alpha=1)

def plot_stations_map(surf_stations: pd.DataFrame, awst_stations: pd.DataFrame, 
                      grid_nodes: pd.DataFrame = None):
    """
    Plot station distribution map using cartopy.
    Optionally adds RADIUS_KM circles around grid nodes.
    Includes a time series showing station count evolution from YEAR_STT to YEAR_END.
    """
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

    # Main map axes (takes most of the figure)
    ax_map = fig.add_axes([0.08, 0.05, 0.95, 0.95], 
                          projection=ccrs.PlateCarree())
    
    # Set map extent to Guangdong Province
    ax_map.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    plot_geo_feature(ax_map)

    plot_provincial_boundaries(ax_map)

    plot_station_markers(ax_map, surf_stations, 'blue', 80, 'National Observatory Station',6)
    plot_station_markers(ax_map, awst_stations, 'red', 40, 'Automatic Weather Station',5)
    
    plot_gridlines(ax_map)
    
    plot_neighbor_circles(ax_map, grid_nodes)

    plot_north_arrow(ax_map)

    plot_station_timeline(fig, surf_stations, awst_stations)

    print(f"Saving map to: {OUTPUT_FILE}")
    plt.savefig(OUTPUT_FILE,  bbox_inches='tight',  facecolor='white',  )


def main() -> None:
    print(f"Loading station locations from: {STATION_FILE}")
    surf_stations, awst_stations = load_station_info(STATION_FILE)
    
    print(f"Loading grid node locations from: {NEIGHBOR_CIRCLES_FILE}")
    grid_nodes = load_neighbor_circles(NEIGHBOR_CIRCLES_FILE)
    
    print("Plotting map...")
    set_plot_style()
    plot_stations_map(surf_stations, awst_stations, grid_nodes)

    print(f"Finished: {datetime.now()}")


if __name__ == '__main__':
    main()