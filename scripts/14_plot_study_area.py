# -*- coding: utf-8 -*-
"""
Plot the study area and station network (manuscript Fig. 1, 2 x 2 panels).

(a) Terrain (DEM; the elevation legend is shared by all case maps), the 60-km-radius
    neighborhood circles, and an inset locating the study area in East Asia;
(b) locations of national meteorological observatories (NMOs) and automatic weather stations (AWSs);
(c) number of stations within 60-km-radius circles, with contour lines;
(d) number of stations in operation per year, 2003-2025, as bars: AWSs and NMOs in separate subplots.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rioxarray

import _bootstrap  # noqa: F401
from src.qc_plot_utils import (
    set_plot_style,
    LAND, OCEAN, COASTLINE,
    DEM_CMAP, DEM_VMIN, DEM_VMAX, DEM_ALPHA,
    FONT_SIZE, TICK_SIZE, GRID_LABEL_SIZE, ANNOTATION_SIZE,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUTPUT_DIR = SCRIPT_DIR.parent / 'outputs' / 'figures' / 'station_maps'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATION_FILE = DATA_DIR / 'gd_stations_locations.csv'
SHP_FILE = DATA_DIR / 'external' / 'basemap' / 'chn_province_border.shp'
DEM_FILE = DATA_DIR / 'external' / 'basemap' / 'gd_dem_1km.tif'
RADIUS_KM = 60
NEIGHBOR_CIRCLES_FILE = DATA_DIR / 'neighbor_circles_outlier' / f"neighbor_circles_outlier_{RADIUS_KM}km.csv"
DENSITY_FILE = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{RADIUS_KM}km.csv"
OUTPUT_FILE = OUTPUT_DIR / 'study_area.png'

FIGSIZE = (12, 9)  # full-page width, same as the other manuscript figures
DPI = 300

MAP_EXTENT = [109.0, 118.0, 19.5, 26.0]  # lon min, lon max, lat min, lat max
INSET_EXTENT = [108.0, 130.0, 18.0, 40.0]
YEAR_STT, YEAR_END = 2003, 2025

CIRCLE_COLOR = 'green'
STUDY_AREA_COLOR = 'red'

# Station types: validated categorical slots 1 (blue) and 2 (orange); NMOs drawn on top
STATION_TYPES = [
    ('awst', 'AWS', '#eb6834', 3),
    ('surf', 'NMO', '#2a78d6', 28),
]

# Station density: multi-hue sequential scale (yellow-orange-red), distinct from the blue ocean
DENSITY_CMAP = 'YlOrRd'
DENSITY_BOUNDS = [0, 100, 200, 300, 400, 500, 600, 800, 1000, 1200]
DENSITY_CONTOURS = [300, 600]

# Broken y axis of the station counts (bars start at zero, so the lower part starts at zero)
AWS_YLIM = (0, 6000)
NMO_YLIM = (84, 88)
NMO_YTICKS = [84, 86]  # 90 omitted so the lower subplot can sit close to the upper one
BAR_WIDTH = 0.7


def plot_base_map(ax, lines_zorder: float = 4):
    """
    Land, ocean, coastline, provincial boundaries and gridlines.
    lines_zorder below that of a data layer (e.g. 2.5) draws coastline and boundaries underneath it.
    """
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(LAND, zorder=0)
    ax.add_feature(OCEAN, zorder=2)
    ax.add_feature(COASTLINE, linewidth=0.5, zorder=lines_zorder)
    boundaries = gpd.read_file(SHP_FILE)
    ax.add_geometries(boundaries.geometry, crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=0.5, zorder=lines_zorder)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.7, linestyle='--',
                      xlocs=np.arange(108, 120, 2), ylocs=np.arange(18, 28, 2))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': GRID_LABEL_SIZE}
    gl.ylabel_style = {'size': GRID_LABEL_SIZE}


def plot_north_arrow(ax):
    """North arrow in the upper right of a map."""
    ax.annotate('', xy=(0.94, 0.93), xycoords='axes fraction', xytext=(0.94, 0.85), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8), zorder=9)
    ax.text(0.94, 0.94, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=TICK_SIZE,
            fontweight='bold', zorder=9)


def white_box(ax, x0, y0, width, height):
    """White rounded box (axes coordinates) behind legends and colorbars on a map."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), width, height, boxstyle='round,pad=0.005', transform=ax.transAxes,
        facecolor='white', edgecolor='black', linewidth=0.5, alpha=0.9, zorder=8))


def map_colorbar(fig, ax, mappable, x0, y0, width, ticks, label, extend='neither'):
    """Horizontal colorbar inside a map, positioned in axes coordinates."""
    box = ax.get_position()
    cax = fig.add_axes([box.x0 + x0 * box.width, box.y0 + y0 * box.height, width * box.width, 0.035 * box.height])
    cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal', ticks=ticks, extend=extend)
    cbar.ax.tick_params(labelsize=ANNOTATION_SIZE, length=2, pad=1)
    cbar.set_label(label, fontsize=ANNOTATION_SIZE, labelpad=2)
    cax.set_zorder(10)
    return cbar


def subplot_study_area(fig, ax, df_circles: pd.DataFrame):
    """(a) Terrain, neighborhood circles, location inset (lower right), legends (lower left)."""
    plot_base_map(ax)

    dem = rioxarray.open_rasterio(DEM_FILE).squeeze()
    values = np.where(dem.values < -1000, np.nan, dem.values)  # mask no-data (-3.4e38)
    X, Y = np.meshgrid(dem.x.values, dem.y.values)
    dem_mesh = ax.pcolormesh(X, Y, values, cmap=DEM_CMAP, vmin=DEM_VMIN, vmax=DEM_VMAX,
                             alpha=DEM_ALPHA, transform=ccrs.PlateCarree(), zorder=1)

    for _, row in df_circles.iterrows():
        ax.add_patch(mpatches.Ellipse(
            (row['lon'], row['lat']), width=2 * row['radius_lon'], height=2 * row['radius_lat'],
            transform=ccrs.PlateCarree(), facecolor='none', edgecolor=CIRCLE_COLOR,
            linewidth=0.7, alpha=0.7, zorder=5))

    # Legends in the lower left: circle legend above the elevation colorbar, in one white box
    white_box(ax, 0.01, 0.92, 0.35, 0.07)
    legend = ax.legend(handles=[mpatches.Patch(facecolor='none', edgecolor=CIRCLE_COLOR, linewidth=1.2)],
                       labels=[f"{RADIUS_KM}-km-radius circle"], loc='lower left', bbox_to_anchor=(0.01, 0.90),
                       fontsize=ANNOTATION_SIZE, frameon=False, handlelength=1.2)
    legend.set_zorder(9)
    white_box(ax, 0.32, 0.03, 0.36, 0.14)
    map_colorbar(fig, ax, dem_mesh, 0.35, 0.12, 0.30, [0, 400, 800, 1200], 'Elevation (m)', extend='max')

    # Location inset in the lower right
    box = ax.get_position()
    inset_width = 0.3 * box.width
    inset_height = inset_width * (fig.get_figwidth() / fig.get_figheight())  # square extent (22 x 22 deg)
    ax_inset = fig.add_axes([box.x1 - inset_width, box.y0, inset_width, inset_height], projection=ccrs.PlateCarree())
    ax_inset.set_extent(INSET_EXTENT, crs=ccrs.PlateCarree())
    ax_inset.add_feature(LAND, zorder=0)
    ax_inset.add_feature(OCEAN, zorder=1)
    ax_inset.add_feature(COASTLINE, linewidth=0.3, zorder=2)
    ax_inset.add_geometries(gpd.read_file(SHP_FILE).geometry, crs=ccrs.PlateCarree(),
                            facecolor='none', edgecolor='black', linewidth=0.3, zorder=3)
    ax_inset.add_patch(mpatches.Rectangle((MAP_EXTENT[0], MAP_EXTENT[2]), MAP_EXTENT[1] - MAP_EXTENT[0],
                                          MAP_EXTENT[3] - MAP_EXTENT[2], transform=ccrs.PlateCarree(),
                                          facecolor='none', edgecolor=STUDY_AREA_COLOR, linewidth=1.5, zorder=4))
    ax_inset.spines['geo'].set_linewidth(1.0)
    ax_inset.set_zorder(11)

    plot_north_arrow(ax)
    ax.set_title('(a) Study area', loc='left', fontsize=FONT_SIZE)


def subplot_stations(ax, df_stations: pd.DataFrame):
    """(b) Station locations by type."""
    plot_base_map(ax)
    for statype, label, color, size in STATION_TYPES:
        df = df_stations[df_stations['statype'] == statype]
        ax.scatter(df['lon'], df['lat'], s=size, c=color, edgecolors='white', linewidths=0.2 if size < 10 else 0.6,
                   transform=ccrs.PlateCarree(), zorder=5 if statype == 'awst' else 6,
                   label=f"{label} ({len(df):,})")
    legend = ax.legend(loc='lower right', fontsize=TICK_SIZE, markerscale=1.5, frameon=True,
                       framealpha=0.9, facecolor='white', edgecolor='black')
    legend.get_frame().set_linewidth(0.5)  # same frame width as white_box()
    legend.set_zorder(8)
    plot_north_arrow(ax)
    ax.set_title('(b) Station locations', loc='left', fontsize=FONT_SIZE)


def subplot_density(fig, ax, df_density: pd.DataFrame):
    """(c) Number of stations within 60-km-radius circles, with contours; colorbar in the lower right."""
    # Coastline and boundaries underneath the (slightly translucent) density cells, so they do not
    # cut through the densest area; they remain visible outside the colored area
    plot_base_map(ax, lines_zorder=2.5)

    lon_sorted = np.sort(df_density['lon'].unique())
    lat_sorted = np.sort(df_density['lat'].unique())
    lon_spacing = np.mean(np.diff(lon_sorted))
    lat_spacing = np.mean(np.diff(lat_sorted))
    lon_edges = np.concatenate([[lon_sorted[0] - lon_spacing / 2], lon_sorted + lon_spacing / 2])
    lat_edges = np.concatenate([[lat_sorted[0] - lat_spacing / 2], lat_sorted + lat_spacing / 2])

    grid_matrix = np.full((len(lat_sorted), len(lon_sorted)), np.nan)
    for _, row in df_density.iterrows():
        grid_matrix[np.argmin(np.abs(lat_sorted - row['lat'])), np.argmin(np.abs(lon_sorted - row['lon']))] = row['count']

    n_classes = len(DENSITY_BOUNDS) - 1
    cmap = mpl.colors.ListedColormap(plt.get_cmap(DENSITY_CMAP)(np.linspace(0.1, 1.0, n_classes)))
    norm = mpl.colors.BoundaryNorm(DENSITY_BOUNDS, n_classes)
    mesh = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(grid_matrix), cmap=cmap, norm=norm,
                         alpha=0.85, edgecolors='face', linewidth=0, transform=ccrs.PlateCarree(), zorder=3)

    contours = ax.tricontour(mpl.tri.Triangulation(df_density['lon'], df_density['lat']), df_density['count'],
                             levels=DENSITY_CONTOURS, colors='black', linewidths=1.2,
                             transform=ccrs.PlateCarree(), zorder=6)
    labels = ax.clabel(contours, inline=True, fontsize=ANNOTATION_SIZE, fmt='%d', colors='black')
    for label in labels:
        label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1))

    white_box(ax, 0.50, 0.03, 0.48, 0.14)
    map_colorbar(fig, ax, mesh, 0.53, 0.12, 0.42, DENSITY_BOUNDS[::2], 'Number of stations')
    plot_north_arrow(ax)
    ax.set_title(f"(c) Station density within {RADIUS_KM} km", loc='left', fontsize=FONT_SIZE)


def count_stations_by_year(df_stations: pd.DataFrame) -> pd.DataFrame:
    """Number of stations in operation at the end of each year, by station type."""
    df = df_stations.copy()
    df['date_stt'] = pd.to_datetime(df['date_stt'])
    rows = []
    for year in range(YEAR_STT, YEAR_END + 1):
        active = df[df['date_stt'] <= pd.Timestamp(f'{year}-12-31')]
        rows.append({'year': year,
                     'surf': int((active['statype'] == 'surf').sum()),
                     'awst': int((active['statype'] == 'awst').sum())})
    return pd.DataFrame(rows)


def subplot_station_counts(fig, gs, df_counts: pd.DataFrame):
    """(d) Stations in operation per year: AWSs (upper subplot) and NMOs (lower subplot), each as bars."""
    # Third row: empty spacer for the rotated year labels and the x-axis title
    gs_inner = gs.subgridspec(3, 1, height_ratios=[2.4, 1.2, 0.32], hspace=0.06)
    ax_upper = fig.add_subplot(gs_inner[0])
    ax_lower = fig.add_subplot(gs_inner[1], sharex=ax_upper)

    types = {statype: (label, color) for statype, label, color, _ in STATION_TYPES}
    for ax, statype, ylim in [(ax_upper, 'awst', AWS_YLIM), (ax_lower, 'surf', NMO_YLIM)]:
        label, color = types[statype]
        ax.bar(df_counts['year'], df_counts[statype], width=BAR_WIDTH, color=color,
               edgecolor='white', linewidth=0.5)
        ax.set_ylim(*ylim)
        ax.text(0.02, 0.95, label, transform=ax.transAxes, ha='left', va='top', fontsize=TICK_SIZE,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2))
    ax_lower.set_yticks(NMO_YTICKS)
    ax_upper.tick_params(labelbottom=False)

    ax_lower.set_xticks(range(YEAR_STT, YEAR_END + 1, 2))
    ax_lower.tick_params(axis='x', labelrotation=45)
    for label in ax_lower.get_xticklabels():
        label.set_horizontalalignment('right')
        label.set_rotation_mode('anchor')
    ax_lower.set_xlim(YEAR_STT - 0.7, YEAR_END + 0.7)
    ax_lower.set_xlabel('Year')
    ax_upper.set_ylabel('Number of stations')
    ax_upper.yaxis.set_label_coords(-0.10, 0.3)
    ax_upper.set_title('(d) Stations in operation', loc='left', fontsize=FONT_SIZE)


def main() -> None:
    print(f"Starting study area figure at {datetime.now()}")
    set_plot_style()

    df_stations = pd.read_csv(STATION_FILE, encoding='utf-8-sig')
    df_circles = pd.read_csv(NEIGHBOR_CIRCLES_FILE, encoding='utf-8-sig')
    df_density = pd.read_csv(DENSITY_FILE, encoding='utf-8-sig')
    df_counts = count_stations_by_year(df_stations)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(2, 2, left=0.04, right=0.99, top=0.96, bottom=0.04, hspace=0.15, wspace=0.16)
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_c = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    for ax in [ax_a, ax_b, ax_c]:
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    fig.canvas.draw()  # fix map positions (aspect) before placing insets and colorbars relative to them

    subplot_study_area(fig, ax_a, df_circles)
    subplot_stations(ax_b, df_stations)
    subplot_density(fig, ax_c, df_density)
    subplot_station_counts(fig, gs[1, 1], df_counts)

    fig.savefig(OUTPUT_FILE, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Finished: {datetime.now()}")


if __name__ == '__main__':
    main()
