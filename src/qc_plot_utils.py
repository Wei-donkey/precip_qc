# -*- coding: utf-8 -*-
"""
Utility functions for QC result plotting.

Contains common visualization and helper functions used by QC plotting scripts,
designed to be imported to avoid code duplication.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rioxarray

# Constants for plotting
FIGSIZE = (4, 5)
DPI = 300
HIST_BINS = 15

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


def set_plot_style():
    """Apply consistent plot styling."""
    for key, value in STYLE_SETTINGS.items():
        if key == 'style':
            mpl.style.use(STYLE_SETTINGS['style'])
        else:
            mpl.rcParams[key] = value


def clip_dem_to_extent(dem_data, extent: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Clip and reproject DEM to the specified extent. """
    
    dem_clipped = dem_data.rio.clip_box(
        minx=extent[0],
        miny=extent[2],
        maxx=extent[1],
        maxy=extent[3],
        crs="EPSG:4326"
    )
    
    # Extract coordinates
    x_coords = dem_clipped.x.values
    y_coords = dem_clipped.y.values
    dem_data = dem_clipped.values[0]  # Get first band
    
    return dem_data, x_coords, y_coords


def clip_qpe_to_extent(qpe_data: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, 
                       extent: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_min, lon_max, lat_min, lat_max = extent
    
    # Find indices within the extent
    lon_mask = (x_coords >= lon_min) & (x_coords <= lon_max)
    lat_mask = (y_coords >= lat_min) & (y_coords <= lat_max)
    
    # Extract clipped coordinates
    x_clipped = x_coords[lon_mask]
    y_clipped = y_coords[lat_mask]
    
    # Extract the corresponding data subset
    # Need to find the actual index ranges
    lon_indices = np.where(lon_mask)[0]
    lat_indices = np.where(lat_mask)[0]

    # Get min/max indices for slicing
    lon_start, lon_end = lon_indices[0], lon_indices[-1] + 1
    lat_start, lat_end = lat_indices[0], lat_indices[-1] + 1
    
    # Clip the data
    qpe_clipped = qpe_data[lat_start:lat_end, lon_start:lon_end]
    
    return qpe_clipped, x_clipped, y_clipped


def plot_dem_layer(fig, ax_map, map_extent, dem_data):
    # === Plot DEM Background ===
    dem_mesh = None

    dem_data, x_coords, y_coords = clip_dem_to_extent(dem_data, map_extent)
    
    # Check if clipped data is valid
    if x_coords.size == 0 or y_coords.size == 0:
        print(f"Warning: DEM clipping resulted in empty coordinates")
        return None
    
    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Plot DEM with Greys colormap
    dem_mesh = ax_map.pcolormesh(X, Y, dem_data, cmap='Greys', alpha=1, zorder=2)
    
    # === Add DEM Colorbar Legend (Overlay on lower right) ===
    # Use fig.add_axes for precise control over colorbar position
    cbar_ax = fig.add_axes([0.85, 0.27, 0.06, 0.12])
    
    cbar = plt.colorbar(dem_mesh, cax=cbar_ax, orientation='vertical')
            
    # Add white background rectangle behind the colorbar using axes coordinates
    white_box = mpatches.Rectangle((-0.25, -0.06), 2.2, 1.25, transform=cbar_ax.transAxes,
                                    facecolor='white', edgecolor='black', 
                                    linewidth=0.1, zorder=6)
    ax_map.add_patch(white_box)
    
    cbar.ax.set_title('Elev. (m)', fontsize=6, pad=2, loc= 'center')
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.yaxis.set_ticks_position('right')


def plot_qpe_layer(fig, ax_map, map_extent, qpe_data: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray):
    # === Clip QPE to extent ===
    qpe_clipped, x_clipped, y_clipped = clip_qpe_to_extent(qpe_data, x_coords, y_coords, map_extent)

    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x_clipped, y_clipped)
    
    # Mask zero QPE values to make them transparent
    qpe_masked = np.ma.masked_where(qpe_clipped == 0, qpe_clipped)
    
    # Plot QPE with jet (rainbow) colormap, capped at 100mm maximum
    qpe_mesh = ax_map.pcolormesh(X, Y, qpe_masked, cmap='jet', vmin=3, vmax=80, alpha=0.8, zorder=3)
    
    # === Add QPE Colorbar Legend (Overlay on lower right) ===
    cbar_ax = fig.add_axes([0.72, 0.27, 0.06, 0.12])
    
    cbar = plt.colorbar(qpe_mesh, cax=cbar_ax, orientation='vertical')
    
    # Set custom tick positions and labels
    cbar.set_ticks([5, 25, 45, 65])
    cbar.set_ticklabels(['5', '25', '45', '65'])
    
    # Add white background rectangle behind the colorbar
    white_box = mpatches.Rectangle((-0.30, -0.06), 2.2, 1.25, transform=cbar_ax.transAxes,
                                    facecolor='white', edgecolor='black', 
                                    linewidth=0.1, zorder=6)
    ax_map.add_patch(white_box)
    
    cbar.ax.set_title('QPE (mm)', fontsize=6, pad=2, loc= 'center')
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.yaxis.set_ticks_position('right')
    
    return qpe_mesh


def plot_gridlines(ax_map):    
    # Add gridlines
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                                alpha=0.7, linestyle='--',
                                xlocs=np.arange(-180, 180, 0.4),
                                ylocs=np.arange(-90, 90, 0.4))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 5}
    gl.ylabel_style = {'size': 5}


def plot_scatter_precip(ax_map, df_circles_precip, target_stacode, target_precip):
    # Plot precipitation values as text labels (exclude target station)
    df_circles_neighbor_precip = df_circles_precip[df_circles_precip['stacode'] != target_stacode].copy()

    groupby_df_circle_precip = df_circles_neighbor_precip.groupby(['stacode','lat','lon']).agg({'r': 'max'})
    df_circles_precip_max = groupby_df_circle_precip.reset_index()

    if not df_circles_precip_max.empty and len(df_circles_precip_max) > 0:
        for _, record in df_circles_precip_max.iterrows():
            lat_val = record.get('lat')
            lon_val = record.get('lon')
            precip_val = record['r']
            if pd.notna(lat_val) and pd.notna(lon_val):
                if precip_val < 3.0:
                    # Show small dot for low precipitation
                    ax_map.scatter(lon_val, lat_val,
                                         s=2, c='gray', alpha=1, edgecolors='none',
                                         transform=ccrs.PlateCarree(),
                                         zorder=3)
                else:
                    # Show text label for higher precipitation
                    if precip_val >= 3.0 and precip_val < 10.0: color = 'black'
                    if precip_val >= 10.0 and precip_val < 20.0: color = 'blue'
                    if precip_val >= 20.0 and precip_val < 50.0: color = 'purple'
                    if precip_val >= 50.0 : color = 'red'                    
                    ax_map.text(lon_val, lat_val,
                                      f"{precip_val:.1f}",
                                      fontsize=6, color=color, ha='center', va='center',
                                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.3),
                                      transform=ccrs.PlateCarree())

    # Plot target precipitation in bold red
    df_target_station = df_circles_precip[df_circles_precip['stacode'] == target_stacode]
    if not df_target_station.empty:
        lat_val = df_target_station.iloc[0].get('lat')
        lon_val = df_target_station.iloc[0].get('lon')
        if pd.notna(lat_val) and pd.notna(lon_val):
            ax_map.text(lon_val, lat_val,
                                f"{target_precip:.1f}",
                                fontsize=8, fontweight='bold', color='red',
                                ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.3),
                                transform=ccrs.PlateCarree(),
                                zorder=4,)


def plot_circles(ax_map, df_validation_circles, flag):
    """Plot validation circles with red curves."""
    if df_validation_circles is None or df_validation_circles.empty:
        return None
    
    linewidth = 0.6
    alpha = 0.8
    if flag == 'validation_circles': 
        edgecolor = 'red'
        zorder = 3
    elif flag == 'all_circles':
        edgecolor = 'green'
        zorder = 2

    # Create a single dummy patch for legend (to avoid duplicate legend entries)
    dummy_circle = mpatches.Ellipse(
        (0, 0),
        width=0,
        height=0,
        facecolor='none',
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        label=f"{flag}"
    )
    
    for _, record in df_validation_circles.iterrows():
        circle = mpatches.Ellipse(
            (record['lon'], record['lat']),
            width=2 * record['radius_lon'],
            height=2 * record['radius_lat'],
            transform=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        ax_map.add_patch(circle)
    
    return dummy_circle


def get_qpe_at_location(qpe_data: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray,
                        target_lon: float, target_lat: float) -> float:
    """Return the QPE value at the nearest grid point to (target_lon, target_lat)."""
    lon_idx = int(np.argmin(np.abs(x_coords - target_lon)))
    lat_idx = int(np.argmin(np.abs(y_coords - target_lat)))
    return float(qpe_data[lat_idx, lon_idx])


def plot_geo_feature(ax_map):
    # Add geographic features    
    ax_map.add_feature(LAND, zorder=0)
    ax_map.add_feature(OCEAN, zorder=2)
    ax_map.add_feature(COASTLINE, linewidth=0.5, zorder=2)


def plot_circle_legend(ax_map, df_circles, df_validation_circles, handle_all_circles=None, handle_validation_circles=None):
    # Add legend for circle types
    legend_handles = []
    legend_labels = []
    
    if handle_all_circles is not None:
        n_all = len(df_circles) if df_circles is not None and not df_circles.empty else 0
        legend_handles.append(handle_all_circles)
        legend_labels.append(f"Neighborhood circle ({n_all})")
    
    if handle_validation_circles is not None:
        n_valid = len(df_validation_circles) if df_validation_circles is not None and not df_validation_circles.empty else 0
        legend_handles.append(handle_validation_circles)
        legend_labels.append(f"Effective circle ({n_valid})")
    
    if legend_handles:
        legend = ax_map.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='upper left',
            fontsize=6,
            framealpha=1,
            edgecolor='black',
            facecolor='white'
        )
        # Set the linewidth of the legend frame edge
        legend.get_frame().set_linewidth(0.1)

        # Adjust legend position manually (you can fine-tune these values)
        legend.set_bbox_to_anchor((0, 1))


def subplot_histogram(fig, gs, df_circles_precip, target_stacode, 
                     df_validation_circles=None, df_all_extreme_circles=None, 
                     break_y_at=20, is_qpe=False, qpe_data=None, x_coords=None, y_coords=None):
    """
    df_circles_precip : pd.DataFrame or None
        Circles precipitation DataFrame. If None and is_qpe=True, uses qpe_data instead.
    is_qpe : bool, optional
        If True, plot histogram of QPE grid values within circles instead of station data
    """
    # Get precipitation data based on mode
    if is_qpe:
        # Extract QPE values within circles
        from qc_data_loader import extract_qpe_in_circles

        qpe_in_circles = extract_qpe_in_circles(qpe_data, x_coords, y_coords, df_all_extreme_circles)
        
        if len(qpe_in_circles) == 0:
            print("Warning: No QPE grids found within circles")
            return
        
        # Convert to DataFrame for consistent handling
        df_data = pd.DataFrame({'r': qpe_in_circles})
        data_values = qpe_in_circles
        
    else:
        # Original QC mode - filter by validation circles if provided
        df_filtered_precip = df_circles_precip.copy()
        
        if not df_validation_circles.empty and not df_all_extreme_circles.empty:
            # Use the first validation circle
            first_val_circle = df_validation_circles.iloc[0]
            
            # Find the matching extreme circle in df_all_extreme_circles
            match_mask = (
                (df_all_extreme_circles['lon'].round(4) == first_val_circle['lon'].round(4)) &
                (df_all_extreme_circles['lat'].round(4) == first_val_circle['lat'].round(4))
            )
            
            if match_mask.any():
                # Get the neighbors list for this validation circle
                matched_circle = df_all_extreme_circles[match_mask].iloc[0]
                neighbors_list = matched_circle['neighbors']
                
                # Filter df_circles_precip to only include stations in this circle's neighbors
                df_filtered_precip = df_circles_precip[df_circles_precip['stacode'].isin(neighbors_list)].copy()
        
        # Exclude target station
        df_data = df_filtered_precip[df_filtered_precip['stacode'] != target_stacode].copy()
        data_values = df_data['r'].values
    
    # Create two sub-axes for broken y-axis (break at 20)
    gs_inner = gs.subgridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
    
    # Upper axis (shows high frequencies - outliers)
    ax_upper = fig.add_subplot(gs_inner[0])
    ax_upper.hist(data_values, bins=HIST_BINS, color='steelblue',
                 edgecolor='black', linewidth=0.5, alpha=0.7)
    ax_upper.set_ylim(break_y_at,)  # Auto-scale upper limit
    
    # Set only 2 tick labels on upper axis
    ax_upper.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    
    # Lower axis (shows low frequencies - details)
    ax_lower = fig.add_subplot(gs_inner[1])
    counts, bin_edges, patches = ax_lower.hist(data_values, bins=HIST_BINS, color='steelblue',
                                               edgecolor='black', linewidth=0.5, alpha=0.7)
    ax_lower.set_ylim(0, break_y_at)
    
    # Add labels on top of each bar starting from the second bin
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if i >= 1 and count > 0 and count <= break_y_at:  # Start from second bin (index 1) and only label non-zero bars
            x_pos = patch.get_x() + patch.get_width() / 2
            y_pos = count + 1
            ax_lower.text(x_pos, y_pos, f'{int(count)}',
                         ha='center', va='bottom', fontsize=6, color='black')
    
    # Hide spines between axes
    ax_upper.spines.bottom.set_visible(False)
    ax_lower.spines.top.set_visible(False)
    ax_upper.xaxis.tick_top()
    ax_upper.tick_params(labeltop=False)  # don't put tick labels at the top
    ax_lower.xaxis.tick_bottom()
    
    # Add break marks using marker-based diagonal lines
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=6,
                  linestyle="none", color='k', mec='k', mew=0.5, clip_on=False)
    ax_upper.plot([0, 1], [0, 0], transform=ax_upper.transAxes, **kwargs)
    ax_lower.plot([0, 1], [1, 1], transform=ax_lower.transAxes, **kwargs)
    
    # Set labels only on lower axis
    ax_lower.set_xlabel('Precipitation (mm)', fontsize=6)
    ax_lower.set_ylabel('Frequency', fontsize=6)
    ax_lower.tick_params(labelsize=5)
    ax_upper.tick_params(labelsize=5)


def plot_rainfall_event(target_stacode: str, ddatetime: pd.Timestamp, target_precip: float,
                             df_circles_precip: pd.DataFrame,
                             df_circles: pd.DataFrame,
                             df_validation_circles: pd.DataFrame,
                             confusion_type: str, 
                             dem_data: rioxarray.DataArray,
                             df_all_extreme_circles: pd.DataFrame,
                             map_extent,
                             output_file: Path):
    set_plot_style()

    # Create figure
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.08, wspace=0.1)

    # === Right Upper: Extreme circle Map ===
    subplot_map( fig, gs[0], target_stacode, target_precip, df_circles_precip, 
                df_circles, df_validation_circles, dem_data, map_extent=map_extent)

    # === Right Lower: Extreme circle data Histogram ===
    subplot_histogram(fig, gs[1], df_circles_precip, target_stacode,
                     df_validation_circles=df_validation_circles,
                     df_all_extreme_circles=df_all_extreme_circles)
    
    # Set title
    datetime_str = ddatetime.strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f"{confusion_type}: Station {target_stacode} at {datetime_str}\n"
                f"Target Precipitation: {target_precip:.1f} mm",
                fontsize=9, )

    plt.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.06)
    plt.savefig(output_file)
    plt.close(fig)
    
    print(f"Saved: {output_file.name}")


def subplot_map(fig, gs, target_stacode: str,  target_precip: float,
                df_circles_precip: pd.DataFrame,  df_circles: pd.DataFrame,
                df_validation_circles: pd.DataFrame,
                dem_data: rioxarray.DataArray,  
                map_extent,              
                ):

    ax_map = fig.add_subplot(gs, projection=ccrs.PlateCarree())

    ax_map.set_extent(map_extent, crs=ccrs.PlateCarree())

    plot_dem_layer(fig, ax_map, map_extent, dem_data)

    plot_geo_feature(ax_map)

    plot_scatter_precip(ax_map, df_circles_precip, target_stacode, target_precip)

    # Plot circles and collect legend handles
    handle_all_circles = plot_circles(ax_map, df_circles, 'all_circles')
    
    # Plot validation circles with red curves if provided
    handle_validation_circles = plot_circles(ax_map, df_validation_circles, 'validation_circles')

    plot_circle_legend(ax_map, df_circles, df_validation_circles, handle_all_circles, handle_validation_circles)

    plot_gridlines(ax_map)


def subplot_qpe_map(fig, gs, target_stacode: str, ddatetime_utc: pd.Timestamp, target_precip: float,
                    df_validation_circles: pd.DataFrame,
                    qpe_data: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray,
                    df_circles: pd.DataFrame, target_lon: float, target_lat: float,
                    map_extent, dem_data=None):

    ax_map = fig.add_subplot(gs, projection=ccrs.PlateCarree())

    ax_map.set_extent(map_extent, crs=ccrs.PlateCarree())
    
    # === Plot DEM Background (if provided) ===
    if dem_data is not None:
        plot_dem_layer(fig, ax_map, map_extent, dem_data)
    
    # === Plot Geographic Features ===
    plot_geo_feature(ax_map)
    
    # === Plot QPE Data as Colormap Overlay ===
    plot_qpe_layer(fig, ax_map, map_extent, qpe_data, x_coords, y_coords)

    # Plot circles and collect legend handles
    handle_all_circles = plot_circles(ax_map, df_circles, 'all_circles')
    
    # Plot validation circles with red curves if provided
    handle_validation_circles = plot_circles(ax_map, df_validation_circles, 'validation_circles')

    plot_circle_legend(ax_map, df_circles, df_validation_circles, handle_all_circles, handle_validation_circles)

    # === Add Gridlines ===
    plot_gridlines(ax_map)

    # === Target station marker + QPE label to the north ===
    ax_map.scatter(target_lon, target_lat,
                   s=20, c='white', marker='o', edgecolors='black', linewidths=0.8, alpha=0.4,
                   transform=ccrs.PlateCarree(), zorder=5)

    qpe_at_target = get_qpe_at_location(qpe_data, x_coords, y_coords, target_lon, target_lat)
    label_offset = (map_extent[3] - map_extent[2]) * 0.10  # 10% of map height north

    geo_transform = ccrs.PlateCarree()._as_mpl_transform(ax_map)
    ax_map.annotate(
        f"QPE: {qpe_at_target:.1f} mm",
        xy=(target_lon, target_lat),
        xytext=(target_lon, target_lat + label_offset),
        xycoords=geo_transform,
        textcoords=geo_transform,
        fontsize=6, fontweight='bold', color='black', ha='center', va='bottom',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5, pad=2),
        arrowprops=dict(arrowstyle='-', color='black', linewidth=0.7),
        zorder=6,
    )
    

def plot_qpe(target_stacode: str, ddatetime_utc: pd.Timestamp, target_precip: float,
             qpe_data: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray,
             df_circles: pd.DataFrame, target_lon: float, target_lat: float,
             map_extent, df_extreme_circles: pd.DataFrame, output_file: Path,
             dem_data=None, confusion_type: str = ''):
    set_plot_style()
    
    # Create figure with 2x1 layout (map + histogram)
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.08, wspace=0.1)
    
    # === Upper: QPE Map ===
    subplot_qpe_map(fig, gs[0], target_stacode, ddatetime_utc, target_precip,
                    None,  # df_validation_circles not available in QPE context
                    qpe_data, x_coords, y_coords, df_circles, target_lon, target_lat,
                    map_extent=map_extent, dem_data=dem_data)
    
    # === Lower: Histogram of QPE values within circles ===
    subplot_histogram(fig, gs[1], None, target_stacode,
                     df_validation_circles=None,
                     df_all_extreme_circles=df_extreme_circles, break_y_at=50,
                     is_qpe=True, qpe_data=qpe_data, x_coords=x_coords, y_coords=y_coords)
 
    # Set title
    datetime_str = (ddatetime_utc + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')
    qpe_at_target = get_qpe_at_location(qpe_data, x_coords, y_coords, target_lon, target_lat)
    fig.suptitle(f"{confusion_type}: Station {target_stacode} at {datetime_str}\n"
                 f"Target QPE: {qpe_at_target:.1f} mm",
                 fontsize=9)
    
    plt.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.06)
    plt.savefig(output_file)
    plt.close(fig)
    
    print(f"Saved: {output_file.name}")
