# -*- coding: utf-8 -*-
"""
Plot the distribution of meteorological stations in Guangdong Province.
Stations are categorized by type: surf-type and awst-type.
Grid nodes with 50km-radius circles are also plotted.
Includes a time series showing station count evolution from 2003 to 2025.
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

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / 'data'
OUTPUT_DIR = SRC_DIR.parent / 'figures' / 'station_maps'

STATION_FILE = DATA_DIR / 'gd_stations_locations.csv'
GRID_NEIGHBORS_FILE = DATA_DIR / 'gd_grids_coarse_neighbors.csv'
SHP_FILE = DATA_DIR / 'external' / 'CHN_Province_Border.shp'
OUTPUT_FILE = OUTPUT_DIR / 'gd_stations_distribution_map.png'

# Geographic bounds for Guangdong Province
LON_MIN, LON_MAX = 109.0, 119.0
LAT_MIN, LAT_MAX = 19.0, 26.0

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


def set_plot_style():
    """Apply consistent plot styling."""
    for key, value in STYLE_SETTINGS.items():
        if key == 'style':
            mpl.style.use(STYLE_SETTINGS['style'])
        else:
            mpl.rcParams[key] = value


def load_station_data(station_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load station data and separate by station type.
    
    Returns:
        Tuple of (surf_stations, awst_stations) DataFrames
    """
    df = pd.read_csv(station_file, encoding='utf-8-sig')
    
    # Separate by station type
    surf_stations = df[df['statype'] == 'surf'].copy()
    awst_stations = df[df['statype'] == 'awst'].copy()
    
    return surf_stations, awst_stations


def load_grid_nodes(grid_file: Path) -> pd.DataFrame:
    """
    Load grid node data from CSV file.
    
    Returns:
        DataFrame with columns: lon, lat, neighbors, count
    """
    df = pd.read_csv(grid_file, encoding='utf-8-sig')
    return df


def calculate_yearly_station_counts(all_stations: pd.DataFrame, 
                                     start_year: int = 2003, 
                                     end_year: int = 2025) -> pd.DataFrame:
    """
    Calculate the number of active stations for each year.
    
    Args:
        all_stations: DataFrame with columns including 'date_stt' and 'statype'
        start_year: Start year for analysis
        end_year: End year for analysis
    
    Returns:
        DataFrame with columns: year, surf_count, awst_count, total_count
    """
    # Convert date_stt to datetime
    all_stations = all_stations.copy()
    all_stations['date_stt'] = pd.to_datetime(all_stations['date_stt'])
    
    # Define analysis period
    analysis_start = pd.Timestamp(f'{start_year}-01-01')
    
    # Generate year range
    years = list(range(start_year, end_year + 1))
    
    results = []
    for year in years:
        year_end = pd.Timestamp(f'{year}-12-31')
        
        # Count stations that were active during this year
        # A station is active if it started before or during the year
        active_stations = all_stations[all_stations['date_stt'] <= year_end]
        
        surf_count = len(active_stations[active_stations['statype'] == 'surf'])
        awst_count = len(active_stations[active_stations['statype'] == 'awst'])
        total_count = surf_count + awst_count
        
        results.append({
            'year': year,
            'surf_count': surf_count,
            'awst_count': awst_count,
            'total_count': total_count,
        })
    
    return pd.DataFrame(results)


def plot_stations_map(surf_stations: pd.DataFrame, awst_stations: pd.DataFrame, 
                      grid_nodes: pd.DataFrame = None,
                      all_stations: pd.DataFrame = None):
    """
    Plot station distribution map using cartopy for professional geographic visualization.
    Optionally adds 50km-radius circles around grid nodes and station count time series.
    """
    # Create figure with cartopy projection
    fig = plt.figure(figsize=(14, 10))
    
    # Main map axes (takes most of the figure)
    ax_map = fig.add_axes([0.08, 0.05, 0.95, 0.95], 
                          projection=ccrs.PlateCarree())
    
    # Set map extent to Guangdong Province
    ax_map.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    # Add geographic features
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', 
                                        edgecolor='face', facecolor='#f0f0f0')
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', 
                                         edgecolor='face', facecolor="#73a1e7")
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m', 
                                             edgecolor='black', facecolor='none')
    borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m', 
                                           edgecolor='black', facecolor='none')
    
    # ax_map.add_feature(land, zorder=0)
    ax_map.add_feature(ocean, zorder=0)
    # ax_map.add_feature(coastline, linewidth=0.5, zorder=1)
    # ax_map.add_feature(borders, linewidth=0.5, alpha=0.5, zorder=1)

     # load Guangdong boundary from a shapefile
    gd_boundary = gpd.read_file(SHP_FILE)  # Replace with your file path
    ax_map.add_geometries(gd_boundary.geometry, crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=0.5, alpha=1)


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

    # Plot surf-type stations (blue circles)
    if not surf_stations.empty:
        ax_map.scatter(
            surf_stations['lon'],
            surf_stations['lat'],
            transform=ccrs.PlateCarree(),
            c='blue',
            marker='o',
            s=50,
            alpha=0.8,
            label='National Observatory Station',
            zorder=6,
        )
    
    # Plot awst-type stations (red triangles)
    if not awst_stations.empty:
        ax_map.scatter(
            awst_stations['lon'],
            awst_stations['lat'],
            transform=ccrs.PlateCarree(),
            c='red',
            edgecolor='white',
            linewidths=0.2,
            marker='^',
            s=40,
            alpha=0.9,
            label='Automatic Weather Station',
            zorder=5,
        )
    
    # Plot 50km-radius circles around grid nodes
    if grid_nodes is not None and not grid_nodes.empty:
        print(f"Adding {len(grid_nodes)} grid node circles...")
        
        # Create circle patches for each grid node
        for idx, row in grid_nodes.iterrows():
            lon = row['lon']
            lat = row['lat']
            count = row['count']
            
            # Create a circle with 50km radius
            # Convert 50km to degrees (approximate at mid-latitude ~22.5°N)
            # 1 degree latitude ≈ 111 km
            # 1 degree longitude ≈ 111 * cos(latitude) km
            radius_lat = 50.0 / 111.0  # ~0.45 degrees
            radius_lon = 50.0 / (111.0 * np.cos(np.radians(lat)))  # ~0.49 degrees at 22.5°N
            
            # Create ellipse to account for projection distortion
            circle = mpatches.Ellipse(
                (lon, lat),
                width=2 * radius_lon,
                height=2 * radius_lat,
                transform=ccrs.PlateCarree(),
                facecolor='none',
                edgecolor='green',
                linewidth=1,
                alpha=0.8,
                zorder=7,
            )
            ax_map.add_patch(circle)
        
        # Add a dummy patch for legend
        circle_legend = mpatches.Patch(
            facecolor='none',
            edgecolor='green',
            linewidth=0.8,
            alpha=0.6,
            label=f'50km Grid Circle ({len(grid_nodes)} nodes)'
        )
        
        # Update legend to include grid circles
        handles, labels = ax_map.get_legend_handles_labels()
        handles.append(circle_legend)
        labels.append(f'50km Grid Circle ({len(grid_nodes)} nodes)')
        
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

    # --- Add North Arrow ---
    # Position in axes coordinates (0-1)
    arrow_x, arrow_y = 0.95, 0.95 
    arrow_length = 0.04
    
    # Draw arrow using annotate
    ax_map.annotate('', xy=(arrow_x, arrow_y), xycoords='axes fraction',
                    xytext=(arrow_x, arrow_y - arrow_length), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    
    # Add 'N' label
    ax_map.text(arrow_x, arrow_y + 0.01, 'N', transform=ax_map.transAxes,
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # --- Add Station Count Time Series Plot ---
    print("Calculating yearly station counts...")
    yearly_counts = calculate_yearly_station_counts(all_stations, 2003, 2025)
    
    # Create inset axes in lower right corner
    ax_inset = fig.add_axes([0.53, 0.105, 0.5, 0.3])   
  
    # Plot surf-type stations
    ax_inset.plot(yearly_counts['year'], yearly_counts['surf_count'], 
                    'b-o', markersize=6, linewidth=1.5, label='National Observatory Station', zorder=2)
    
    # Plot awst-type stations
    ax_inset.plot(yearly_counts['year'], yearly_counts['awst_count'], 
                    'r-^', markersize=6, linewidth=1.5, label='Automatic Weather Station', zorder=2)
    
    # Customize inset plot
    ax_inset.set_xlabel('Year', fontsize=10) #, fontweight='bold'
    ax_inset.set_ylabel('Number of Stations', fontsize=10) #, fontweight='bold'
    # ax_inset.set_title('Station Count Evolution (2003-2025)', 
    #                     fontsize=11, fontweight='bold', pad=8)
    ax_inset.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    ax_inset.legend(loc='upper left', fontsize=10, framealpha=0.9, 
                    frameon=True, edgecolor='black')
    ax_inset.set_xlim(2002.5, 2025.5)
    
    # Set x-axis ticks to show every 2 years
    ax_inset.set_xticks(range(2003, 2026, 2))
    ax_inset.tick_params(axis='both', labelsize=9)
    
    # Add background color to distinguish from map
    ax_inset.set_facecolor('#fafafa')
    
    print(f"Station counts calculated:")
    print(f"  2003: {yearly_counts.iloc[0]['total_count']} stations")
    print(f"  2025: {yearly_counts.iloc[-1]['total_count']} stations")

    # Save figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
    )
    print(f"Map saved to: {OUTPUT_FILE}")
    
    # Display
    # plt.show()


def main() -> None:
    surf_stations, awst_stations = load_station_data(STATION_FILE)
    
    print(f"Surf-type stations: {len(surf_stations)}")
    print(f"AWST-type stations: {len(awst_stations)}")
    
    # Load all stations for time series calculation
    all_stations = pd.read_csv(STATION_FILE, encoding='utf-8-sig')
    
    # Load grid nodes if file exists
    grid_nodes = None
    grid_nodes = load_grid_nodes(GRID_NEIGHBORS_FILE)
    print(f"Grid nodes loaded: {len(grid_nodes)}")
    
    plot_stations_map(surf_stations, awst_stations, grid_nodes, all_stations)


if __name__ == '__main__':
    main()