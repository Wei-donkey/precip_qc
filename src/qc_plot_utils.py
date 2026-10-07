# -*- coding: utf-8 -*-
"""
Utility functions for QC result plotting.

Contains common visualization and helper functions used by QC plotting scripts,
designed to be imported to avoid code duplication. Case figures are 4 x 5 in, one third
of the 12-in full-page figures, so three of them form one row at the same font size.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rioxarray

from src.qc_data_loader import extract_qpe_in_circles

# Constants for plotting
FIGSIZE = (4, 5)  # case figures: one third of the 12-in full-page width
DPI = 300
HIST_BINS = 15

# Font sizes shared by all manuscript figures (full-page figures are 12 in wide)
FONT_SIZE = 14  # titles and axis labels
TICK_SIZE = 12  # tick labels and legends
GRID_LABEL_SIZE = 11  # map gridline labels
ANNOTATION_SIZE = 10  # value labels

STYLE_SETTINGS = {
    'style': 'seaborn-v0_8-darkgrid',
    'grid.linewidth': 0.5,
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'lines.linewidth': 1.0,
    'figure.titlesize': FONT_SIZE,
    'figure.dpi': 300,
    'legend.fontsize': TICK_SIZE,
    'legend.frameon': True,
    'legend.framealpha': 0.5,
    'legend.facecolor': 'inherit',
    'legend.edgecolor': 'white',
}

LAND = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='face', facecolor="#ffffff")
# Grey-blue ocean: low saturation, so the (saturated) rainfall blues stand out over the sea
OCEAN = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                    edgecolor='face', facecolor="#cde2f8")
COASTLINE = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                        edgecolor='black', facecolor='none')

# DEM shading shared by all maps (legend shown once, in the study-area figure)
DEM_CMAP = 'Greys'
DEM_VMIN, DEM_VMAX = 0, 1200
DEM_ALPHA = 0.8

# Rainfall classes shared by all maps: one color per class between consecutive bounds (mm),
# values above the last bound in RAIN_COLOR_OVER (the colorbar's arrow)
RAIN_BOUNDS = [0.1, 5, 10, 30, 50, 100, 200]
RAIN_COLORS = ['#9bf28f', '#3dba3d', '#61b8ff', '#0000ff', '#fa00fa', '#800040']
RAIN_COLOR_OVER = '#4d0026'

# Extreme circles on case maps: all circles enclosing the target vs the effective ones that confirmed it
CIRCLE_STYLES = {
    'all_circles': {'color': '#666666', 'linestyle': '--', 'linewidth': 0.6, 'zorder': 2},
    'validation_circles': {'color': 'black', 'linestyle': '-', 'linewidth': 0.7, 'zorder': 3},
}

CONFUSION_ABBREVIATIONS = {
    'False negative': 'FN', 'True positive': 'TP', 'False positive': 'FP', 'True negative': 'TN',
}


def set_plot_style():
    """Apply consistent plot styling."""
    for key, value in STYLE_SETTINGS.items():
        if key == 'style':
            mpl.style.use(STYLE_SETTINGS['style'])
        else:
            mpl.rcParams[key] = value


def rain_colormap():
    """Discrete rainfall colormap: RAIN_COLORS for the classes between RAIN_BOUNDS."""
    cmap = mcolors.ListedColormap(RAIN_COLORS)
    cmap.set_over(RAIN_COLOR_OVER)
    norm = mcolors.BoundaryNorm(RAIN_BOUNDS, len(RAIN_COLORS))
    return cmap, norm


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

    # Get min/max indices for slicing
    lon_indices = np.where(lon_mask)[0]
    lat_indices = np.where(lat_mask)[0]
    lon_start, lon_end = lon_indices[0], lon_indices[-1] + 1
    lat_start, lat_end = lat_indices[0], lat_indices[-1] + 1

    # Clip the data
    qpe_clipped = qpe_data[lat_start:lat_end, lon_start:lon_end]

    return qpe_clipped, x_clipped, y_clipped


def plot_dem_layer(fig, ax_map, map_extent, dem_data):
    """DEM background in light grey shading (no colorbar; the legend is shown in the study-area figure)."""
    dem_values, x_coords, y_coords = clip_dem_to_extent(dem_data, map_extent)

    # Check if clipped data is valid
    if x_coords.size == 0 or y_coords.size == 0:
        print(f"Warning: DEM clipping resulted in empty coordinates")
        return None

    X, Y = np.meshgrid(x_coords, y_coords)
    return ax_map.pcolormesh(X, Y, dem_values, cmap=DEM_CMAP, vmin=DEM_VMIN, vmax=DEM_VMAX,
                             alpha=DEM_ALPHA, zorder=1)


def plot_qpe_layer(fig, ax_map, map_extent, qpe_data: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray):
    # === Clip QPE to extent ===
    qpe_clipped, x_clipped, y_clipped = clip_qpe_to_extent(qpe_data, x_coords, y_coords, map_extent)

    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x_clipped, y_clipped)

    # Mask zero QPE values to make them transparent
    qpe_masked = np.ma.masked_where(qpe_clipped == 0, qpe_clipped)

    # Plot QPE with the rainfall classes shared by all maps
    cmap, norm = rain_colormap()
    return ax_map.pcolormesh(X, Y, qpe_masked, cmap=cmap, norm=norm, alpha=1, zorder=3)


def plot_gridlines(ax_map, map_extent=None, n_ticks: int = 4):
    """Gridlines with about n_ticks labels per axis, so labels do not collide in small maps."""
    if map_extent is not None:
        xlocs = mticker.MaxNLocator(n_ticks, steps=[1, 2, 2.5, 5, 10]).tick_values(map_extent[0], map_extent[1])
        ylocs = mticker.MaxNLocator(n_ticks, steps=[1, 2, 2.5, 5, 10]).tick_values(map_extent[2], map_extent[3])
    else:
        xlocs, ylocs = np.arange(-180, 180, 0.4), np.arange(-90, 90, 0.4)
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                          alpha=0.7, linestyle='--', xlocs=xlocs, ylocs=ylocs)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': GRID_LABEL_SIZE}
    gl.ylabel_style = {'size': GRID_LABEL_SIZE}


def plot_scatter_precip(ax_map, df_circles_precip, target_stacode, target_precip):
    """
    Maximum rainfall of each neighboring station within the time window as colored markers
    (rainfall classes shared by all maps); the target value as a bold red label.
    Returns the scatter artist for a colorbar.
    """
    cmap, norm = rain_colormap()

    df_neighbor_precip = df_circles_precip[df_circles_precip['stacode'] != target_stacode]
    df_max = df_neighbor_precip.groupby(['stacode', 'lat', 'lon'], as_index=False)['r'].max().sort_values('r')
    df_max = df_max.dropna(subset=['lat', 'lon'])

    # Stations with no rain as small dots, the rest as colored markers
    dry = df_max['r'] < RAIN_BOUNDS[0]
    ax_map.scatter(df_max.loc[dry, 'lon'], df_max.loc[dry, 'lat'], s=8, c='#9a9a9a',
                   edgecolors='none', transform=ccrs.PlateCarree(), zorder=4)
    wet = df_max[~dry]
    sc = ax_map.scatter(wet['lon'], wet['lat'], c=wet['r'], cmap=cmap, norm=norm, alpha=1, s=24,
                        edgecolors='#333333', linewidths=0.2, transform=ccrs.PlateCarree(), zorder=5)

    # Target value in bold red on a white background
    df_target_station = df_circles_precip[df_circles_precip['stacode'] == target_stacode]
    if not df_target_station.empty:
        lat_val = df_target_station.iloc[0].get('lat')
        lon_val = df_target_station.iloc[0].get('lon')
        if pd.notna(lat_val) and pd.notna(lon_val):
            ax_map.text(lon_val, lat_val, f"{target_precip:.1f}",
                        fontsize=TICK_SIZE, fontweight='bold', color='red', ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.3),
                        transform=ccrs.PlateCarree(), zorder=7)
    return sc


def plot_circles(ax_map, df_validation_circles, flag):
    """
    Plot extreme circles: all circles enclosing the target ('all_circles', grey dashed) or the
    effective ones that confirmed it ('validation_circles', black solid). Line style, not only
    color, separates the two, so they stay distinguishable in greyscale and for color-blind readers.
    """
    if df_validation_circles is None or df_validation_circles.empty:
        return None

    style = CIRCLE_STYLES[flag]
    edgecolor, linestyle, linewidth, zorder = style['color'], style['linestyle'], style['linewidth'], style['zorder']
    alpha = 1.0

    # Create a single dummy patch for legend (to avoid duplicate legend entries)
    dummy_circle = mpatches.Ellipse(
        (0, 0),
        width=0,
        height=0,
        facecolor='none',
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
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
            linestyle=linestyle,
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
        legend_labels.append(f"Extreme circle ({n_all})")

    if handle_validation_circles is not None:
        n_valid = len(df_validation_circles) if df_validation_circles is not None and not df_validation_circles.empty else 0
        legend_handles.append(handle_validation_circles)
        legend_labels.append(f"Effective extreme circle ({n_valid})")

    if legend_handles:
        legend = ax_map.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='upper left',
            fontsize=ANNOTATION_SIZE,
            framealpha=1,
            edgecolor='black',
            facecolor='white',
            handlelength=1.2,
            borderpad=0.3,
            labelspacing=0.2,
        )
        # Set the linewidth of the legend frame edge
        legend.get_frame().set_linewidth(0.1)
        legend.set_bbox_to_anchor((0, 1))
        legend.set_zorder(8)


def plot_rain_colorbar(fig, gs, mappable):
    """Slim horizontal colorbar of the rainfall classes (mm; unit given in the caption to save space)."""
    ax_cbar = fig.add_subplot(gs)
    cbar = fig.colorbar(mappable, cax=ax_cbar, orientation='horizontal', extend='max')
    cbar.set_ticks(RAIN_BOUNDS)
    cbar.set_ticklabels([f"{b:g}" for b in RAIN_BOUNDS])
    cbar.ax.tick_params(labelsize=ANNOTATION_SIZE, length=2, pad=1)
    return cbar


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
        qpe_in_circles = extract_qpe_in_circles(qpe_data, x_coords, y_coords, df_all_extreme_circles)

        if len(qpe_in_circles) == 0:
            print("Warning: No QPE grids found within circles")
            return

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

    # Create two sub-axes for broken y-axis
    gs_inner = gs.subgridspec(2, 1, height_ratios=[1, 3], hspace=0.08)

    # Upper axis (shows high frequencies - outliers)
    ax_upper = fig.add_subplot(gs_inner[0])
    ax_upper.hist(data_values, bins=HIST_BINS, color='steelblue',
                  edgecolor='black', linewidth=0.5, alpha=0.7)
    ax_upper.set_ylim(break_y_at,)  # Auto-scale upper limit

    # A single tick at the highest count on the upper axis (little room at this font size)
    ax_upper.set_yticks([int(np.histogram(data_values, bins=HIST_BINS)[0].max())])

    # Lower axis (shows low frequencies - details)
    ax_lower = fig.add_subplot(gs_inner[1])
    counts, bin_edges, patches = ax_lower.hist(data_values, bins=HIST_BINS, color='steelblue',
                                               edgecolor='black', linewidth=0.5, alpha=0.7)
    ax_lower.set_ylim(0, break_y_at * 1.25)
    ax_lower.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, integer=True))

    # Add labels on top of each bar starting from the second bin
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if i >= 1 and count > 0 and count <= break_y_at:
            x_pos = patch.get_x() + patch.get_width() / 2
            ax_lower.text(x_pos, count + 0.5, f'{int(count)}',
                          ha='center', va='bottom', fontsize=ANNOTATION_SIZE - 2, color='black')

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
    ax_lower.set_xlabel('Rainfall (mm)', fontsize=ANNOTATION_SIZE + 1, labelpad=1)
    ax_lower.set_ylabel('Count', fontsize=ANNOTATION_SIZE + 1, labelpad=1)
    ax_lower.yaxis.set_label_coords(-0.09, 0.7)
    ax_lower.tick_params(labelsize=ANNOTATION_SIZE, pad=1)
    ax_upper.tick_params(labelsize=ANNOTATION_SIZE, pad=1)


def case_title(confusion_type: str, value: float, ddatetime: pd.Timestamp) -> str:
    """One-line case title, e.g. 'FN, 57.6 mm, 2019-08-22 01:00'."""
    label = CONFUSION_ABBREVIATIONS.get(confusion_type, confusion_type)
    return f"{label}, {value:.1f} mm, {ddatetime:%Y-%m-%d %H}:00"


def case_figure_layout(fig):
    """
    Map, rainfall colorbar and histogram stacked in a 4 x 5 in case figure.
    Rows 1 and 3 are spacers for the map's longitude labels and the colorbar tick labels;
    use rows 0 (map), 2 (colorbar) and 4 (histogram).
    """
    return fig.add_gridspec(5, 1, height_ratios=[4.0, 0.30, 0.08, 0.26, 1.05], hspace=0.03,
                            left=0.14, right=0.97, top=0.94, bottom=0.08)


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
    gs = case_figure_layout(fig)

    # === Upper: Extreme circle Map ===
    sc = subplot_map(fig, gs[0], target_stacode, target_precip, df_circles_precip,
                     df_circles, df_validation_circles, dem_data, map_extent=map_extent)

    # === Middle: rainfall classes ===
    plot_rain_colorbar(fig, gs[2], sc)

    # === Lower: Extreme circle data Histogram ===
    subplot_histogram(fig, gs[4], df_circles_precip, target_stacode,
                      df_validation_circles=df_validation_circles,
                      df_all_extreme_circles=df_all_extreme_circles)

    fig.suptitle(case_title(confusion_type, target_precip, ddatetime), fontsize=FONT_SIZE)

    plt.savefig(output_file, dpi=DPI, facecolor='white')
    plt.close(fig)

    print(f"Saved: {output_file.name}")


def fit_extent_to_cell(fig, gs, map_extent) -> list[float]:
    """
    Widen the longitude range of map_extent (about its center) so the map fills the width of its
    gridspec cell at full cell height; the map then aligns with the full-width panels below it.
    The latitude range, and hence the map height, is unchanged.
    """
    cell = gs.get_position(fig)
    cell_aspect = (cell.width * fig.get_figwidth()) / (cell.height * fig.get_figheight())
    lon_min, lon_max, lat_min, lat_max = map_extent
    lon_span = max(lon_max - lon_min, cell_aspect * (lat_max - lat_min))  # PlateCarree: 1 deg = 1 unit
    lon_center = (lon_min + lon_max) / 2
    return [lon_center - lon_span / 2, lon_center + lon_span / 2, lat_min, lat_max]


def subplot_map(fig, gs, target_stacode: str,  target_precip: float,
                df_circles_precip: pd.DataFrame,  df_circles: pd.DataFrame,
                df_validation_circles: pd.DataFrame,
                dem_data: rioxarray.DataArray,
                map_extent,
                ):

    map_extent = fit_extent_to_cell(fig, gs, map_extent)
    ax_map = fig.add_subplot(gs, projection=ccrs.PlateCarree())

    ax_map.set_extent(map_extent, crs=ccrs.PlateCarree())

    plot_dem_layer(fig, ax_map, map_extent, dem_data)

    plot_geo_feature(ax_map)

    sc = plot_scatter_precip(ax_map, df_circles_precip, target_stacode, target_precip)

    # Plot circles and collect legend handles
    handle_all_circles = plot_circles(ax_map, df_circles, 'all_circles')

    # Plot validation circles with red curves if provided
    handle_validation_circles = plot_circles(ax_map, df_validation_circles, 'validation_circles')

    plot_circle_legend(ax_map, df_circles, df_validation_circles, handle_all_circles, handle_validation_circles)

    plot_gridlines(ax_map, map_extent)

    return sc


def subplot_qpe_map(fig, gs, target_stacode: str, ddatetime_utc: pd.Timestamp, target_precip: float,
                    df_validation_circles: pd.DataFrame,
                    qpe_data: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray,
                    df_circles: pd.DataFrame, target_lon: float, target_lat: float,
                    map_extent, dem_data=None):

    map_extent = fit_extent_to_cell(fig, gs, map_extent)
    ax_map = fig.add_subplot(gs, projection=ccrs.PlateCarree())

    ax_map.set_extent(map_extent, crs=ccrs.PlateCarree())

    # === Plot DEM Background (if provided) ===
    if dem_data is not None:
        plot_dem_layer(fig, ax_map, map_extent, dem_data)

    # === Plot Geographic Features ===
    plot_geo_feature(ax_map)

    # === Plot QPE Data as Colormap Overlay ===
    qpe_mesh = plot_qpe_layer(fig, ax_map, map_extent, qpe_data, x_coords, y_coords)

    # Plot circles and collect legend handles
    handle_all_circles = plot_circles(ax_map, df_circles, 'all_circles')

    # Plot validation circles with red curves if provided
    handle_validation_circles = plot_circles(ax_map, df_validation_circles, 'validation_circles')

    plot_circle_legend(ax_map, df_circles, df_validation_circles, handle_all_circles, handle_validation_circles)

    # === Add Gridlines ===
    plot_gridlines(ax_map, map_extent)

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
        fontsize=ANNOTATION_SIZE, fontweight='bold', color='black', ha='center', va='bottom',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5, pad=2),
        arrowprops=dict(arrowstyle='-', color='black', linewidth=0.7),
        zorder=6,
    )
    return qpe_mesh


def plot_qpe(target_stacode: str, ddatetime_utc: pd.Timestamp, target_precip: float,
             qpe_data: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray,
             df_circles: pd.DataFrame, target_lon: float, target_lat: float,
             map_extent, df_extreme_circles: pd.DataFrame, output_file: Path,
             dem_data=None, confusion_type: str = ''):
    set_plot_style()

    # Map, rainfall colorbar and histogram, as in the station case figures
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = case_figure_layout(fig)

    # === Upper: QPE Map ===
    qpe_mesh = subplot_qpe_map(fig, gs[0], target_stacode, ddatetime_utc, target_precip,
                               None,  # df_validation_circles not available in QPE context
                               qpe_data, x_coords, y_coords, df_circles, target_lon, target_lat,
                               map_extent=map_extent, dem_data=dem_data)

    # === Middle: rainfall classes ===
    plot_rain_colorbar(fig, gs[2], qpe_mesh)

    # === Lower: Histogram of QPE values within circles ===
    subplot_histogram(fig, gs[4], None, target_stacode,
                      df_validation_circles=None,
                      df_all_extreme_circles=df_extreme_circles, break_y_at=50,
                      is_qpe=True, qpe_data=qpe_data, x_coords=x_coords, y_coords=y_coords)

    # Title in Beijing time with the QPE value at the target
    qpe_at_target = get_qpe_at_location(qpe_data, x_coords, y_coords, target_lon, target_lat)
    fig.suptitle(case_title('QPE', qpe_at_target, ddatetime_utc + timedelta(hours=8)),
                 fontsize=FONT_SIZE)

    plt.savefig(output_file, dpi=DPI, facecolor='white')
    plt.close(fig)

    print(f"Saved: {output_file.name}")
