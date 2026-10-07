# -*- coding: utf-8 -*-
"""
Plot the climatological hourly rainfall limit analysis (3x2 figure).

Each row is one target record. Left column: map of the maximum hourly rainfall
of every station within the ±2 h window around the target, on a DEM background.
Right column: hourly rainfall of selected stations over the same 5-hour window.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray

import _bootstrap  # noqa: F401
from src.qc_plot_utils import (set_plot_style, clip_dem_to_extent, LAND, OCEAN, COASTLINE,
                               DEM_CMAP, DEM_VMIN, DEM_VMAX, DEM_ALPHA, RAIN_BOUNDS, rain_colormap)
from src.qc_data_loader import load_station_info, load_circles, load_precip_cache, slice_precip_cache
from src.qc_circle_process_utils import get_circles_for_station, construct_validation_circles  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
TABLE_DIR = SCRIPT_DIR.parent / 'outputs' / 'tables' / 'climate_limit'
OUTPUT_DIR = SCRIPT_DIR.parent / 'outputs' / 'figures' / 'climate_limit'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration for neighbor data time window (hours before and after target time)
NEIGHBOR_HOR = 2

# Must match the run of 03_qc_check_climate_limit.py being plotted
EXTREME_CIRCLES_RADIUS = 50
CLIMATE_LIMIT = 400

# Target records, one per row (Beijing time): bars for 'bar_stations';
# the map is centered on 'map_center' if given, otherwise on the target 'stacode'
CASES = [
    {'stacode': 'G2109', 'ddatetime': '2008-01-17 10:00', 'map_center': 'G2171',
     'bar_stations': ['G2171', 'G2140', 'G2109', 'G2164', 'G2135', 'G2158']},
    {'stacode': 'G2109', 'ddatetime': '2003-06-06 05:00', 'bar_stations': ['G2109']},
    {'stacode': 'G3322', 'ddatetime': '2017-05-07 06:00', 'bar_stations': ['G3322']},
]

INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'
INPUT_EXTREME_DATA = DATA_DIR / 'gd_extreme_data.csv'
INPUT_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{EXTREME_CIRCLES_RADIUS}km.csv"
INPUT_CACHE = DATA_DIR / 'cache' / f"{INPUT_EXTREME_DATA.stem}_adjacent{NEIGHBOR_HOR}h.parquet"
INPUT_QC_RESULT = TABLE_DIR / f"qc_climate_limit_result_extreme{EXTREME_CIRCLES_RADIUS}km_limit{CLIMATE_LIMIT}.csv"
DEM_FILE = DATA_DIR / 'external' / 'basemap' / 'gd_dem_1km.tif'

OUTPUT_FILE = OUTPUT_DIR / f"climate_limit_cases_extreme{EXTREME_CIRCLES_RADIUS}km.png"

# Figure settings (sized for legibility after reduction to journal page width)
FIGSIZE = (12, 15)
DPI = 300
FONT_SIZE = 14  # base font size; tick and annotation sizes derive from it

# Maps: half-width of the map around the target, so every enclosing circle fits
MAP_HALF_SPAN_KM = 100
EXTREME_LABEL_THRESHOLD = 100  # neighbors with a window maximum at or above this value are labeled (mm)
# Candidate label offsets (points), tried in order until a label does not overlap those already placed
LABEL_OFFSETS = [((0, 0), 'center'), ((-8, 10), 'right'), ((8, 10), 'left'), ((8, -10), 'left'), ((-8, -10), 'right')]

# Bars: validated categorical slots in fixed order (one color per station)
BAR_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
BAR_LABEL_THRESHOLD = 100  # in multi-station panels, only bars at or above this value are labeled (mm)


def set_font_sizes():
    """Override the small default font sizes of set_plot_style()."""
    mpl.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 2,
        'ytick.labelsize': FONT_SIZE - 2,
        'legend.fontsize': FONT_SIZE - 3,
    })



def compute_map_extent(lon: float, lat: float) -> list[float]:
    """Square map extent (in km) centered on the target station."""
    half_lat = MAP_HALF_SPAN_KM / 111
    half_lon = MAP_HALF_SPAN_KM / (111 * np.cos(np.radians(lat)))
    return [lon - half_lon, lon + half_lon, lat - half_lat, lat + half_lat]


def plot_circle_outlines(ax_map, df_circles: pd.DataFrame, **kwargs):
    """Draw extreme circles as ellipses in geographic coordinates."""
    for _, circle in df_circles.iterrows():
        ax_map.add_patch(mpatches.Ellipse(
            (circle['lon'], circle['lat']),
            width=2 * circle['radius_lon'], height=2 * circle['radius_lat'],
            transform=ccrs.PlateCarree(), facecolor='none', **kwargs))


def place_label(ax, renderer, placed_boxes: list, text: str, lon: float, lat: float, **text_kwargs):
    """Annotate text at a station, trying LABEL_OFFSETS until it does not overlap labels already placed."""
    for offset, ha in LABEL_OFFSETS:
        label = ax.annotate(text, (lon, lat), xytext=offset, textcoords='offset points',
                            ha=ha, va='center', **text_kwargs)
        box = label.get_window_extent(renderer)
        if not any(box.overlaps(b) for b in placed_boxes) or offset == LABEL_OFFSETS[-1][0]:
            placed_boxes.append(box)
            return
        label.remove()


def subplot_case_map(fig, gs, panel_letter: str, target_stacode: str, ddatetime: pd.Timestamp,
                     df_adj: pd.DataFrame, df_stations: pd.DataFrame, df_extreme_circles: pd.DataFrame,
                     df_qc_result: pd.DataFrame, dem_data, cmap, norm, map_center: str | None = None):
    """Map of each station's maximum hourly rainfall within ±NEIGHBOR_HOR h of the target."""
    target_station = df_stations[df_stations['stacode'] == target_stacode].iloc[0]
    target_lon, target_lat = target_station['lon'], target_station['lat']
    center_station = df_stations[df_stations['stacode'] == (map_center or target_stacode)].iloc[0]
    map_extent = compute_map_extent(center_station['lon'], center_station['lat'])

    # QC result of the target record
    qc_mask = (df_qc_result['stacode'] == target_stacode) & (df_qc_result['ddatetime'] == ddatetime)
    qc_record = df_qc_result[qc_mask].iloc[0]
    target_precip = qc_record['r']

    # Maximum hourly rainfall per station within the ±2 h window
    df_max = df_adj.groupby('stacode', as_index=False)['r'].max()
    df_max = df_max.merge(df_stations[['stacode', 'lon', 'lat']], on='stacode')
    in_extent = (df_max['lon'].between(map_extent[0], map_extent[1]) &
                 df_max['lat'].between(map_extent[2], map_extent[3]))
    df_neighbors = df_max[in_extent & (df_max['stacode'] != target_stacode)].sort_values('r')

    ax_map = fig.add_subplot(gs, projection=ccrs.PlateCarree())
    ax_map.set_extent(map_extent, crs=ccrs.PlateCarree())

    # DEM background (light, so rainfall classes stay readable)
    dem_values, x_coords, y_coords = clip_dem_to_extent(dem_data, map_extent)
    X, Y = np.meshgrid(x_coords, y_coords)
    ax_map.pcolormesh(X, Y, dem_values, cmap=DEM_CMAP, vmin=DEM_VMIN, vmax=DEM_VMAX, alpha=DEM_ALPHA, zorder=1)
    ax_map.add_feature(LAND, zorder=0)
    ax_map.add_feature(OCEAN, zorder=2)
    ax_map.add_feature(COASTLINE, linewidth=0.5, zorder=2)

    # Enclosing extreme circles (dashed grey) and confirming circles (solid black); uncomment to show
    # df_enclosing = get_circles_for_station(target_stacode, df_extreme_circles)
    # df_confirming = construct_validation_circles(qc_record['validation_circle_locs'], df_extreme_circles)
    # plot_circle_outlines(ax_map, df_enclosing, edgecolor='#8a8a8a', linewidth=0.8, linestyle='--', zorder=3)
    # if not df_confirming.empty:
    #     plot_circle_outlines(ax_map, df_confirming, edgecolor='black', linewidth=1.0, zorder=3)

    # Stations with no rain as small dots, the rest as colored markers
    dry = df_neighbors['r'] < RAIN_BOUNDS[0]
    ax_map.scatter(df_neighbors.loc[dry, 'lon'], df_neighbors.loc[dry, 'lat'], s=4, c='#9a9a9a',
                   edgecolors='none', transform=ccrs.PlateCarree(), zorder=4)
    wet = df_neighbors[~dry]
    sc = ax_map.scatter(wet['lon'], wet['lat'], c=wet['r'], cmap=cmap, norm=norm, s=36,
                        edgecolors='#333333', linewidths=0.3, transform=ccrs.PlateCarree(), zorder=5)

    # Label the target (bold) and extreme neighbors as red text on a white background
    ax_map.apply_aspect()
    renderer = fig.canvas.get_renderer()
    placed_boxes = []
    place_label(ax_map, renderer, placed_boxes, f"{target_precip:.1f}", target_lon, target_lat,
                fontsize=FONT_SIZE - 2, fontweight='bold', color='red', zorder=7,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.3))
    df_extreme = wet[wet['r'] >= EXTREME_LABEL_THRESHOLD].sort_values('r', ascending=False)
    for _, record in df_extreme.iterrows():
        place_label(ax_map, renderer, placed_boxes, f"{record['r']:.1f}", record['lon'], record['lat'],
                    fontsize=FONT_SIZE - 2, fontweight='bold', color='red', zorder=6,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.3)) 

    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--',
                          xlocs=np.arange(100, 130, 0.5), ylocs=np.arange(10, 40, 0.5))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': FONT_SIZE - 3}
    gl.ylabel_style = {'size': FONT_SIZE - 3}

    qc_label = str(qc_record['qc_label']).replace('EXTREME_', '')
    time_stt = ddatetime - timedelta(hours=NEIGHBOR_HOR)
    time_end = ddatetime + timedelta(hours=NEIGHBOR_HOR)
    ax_map.set_title(f"({panel_letter}) {ddatetime:%Y-%m-%d} {time_stt:%H}:00–{time_end:%H}:00", loc='left')

    return sc


def subplot_station_bars(ax, panel_letter: str, target_stacode: str, ddatetime: pd.Timestamp,
                         df_adj: pd.DataFrame, bar_stations: list[str], show_xlabel: bool = True):
    """Grouped bars of hourly rainfall at the selected stations over the ±NEIGHBOR_HOR h window."""
    hours = pd.date_range(ddatetime - timedelta(hours=NEIGHBOR_HOR),
                          ddatetime + timedelta(hours=NEIGHBOR_HOR), freq='h')
    x = np.arange(len(hours))
    n_stations = len(bar_stations)
    bar_width = 0.8 / n_stations
    single_station = n_stations == 1

    # # Highlight the target hour
    # target_idx = int(np.where(hours == ddatetime)[0][0])
    # ax.axvspan(target_idx - 0.5, target_idx + 0.5, color='#d9d9d9', alpha=0.6, zorder=0)

    for i, (stacode, color) in enumerate(zip(bar_stations, BAR_COLORS)):
        series = df_adj[df_adj['stacode'] == stacode].set_index('ddatetime')['r'].reindex(hours)
        x_pos = x - 0.4 + bar_width * (i + 0.5)
        bars = ax.bar(x_pos, series.fillna(0).values, width=bar_width, color=color,
                      edgecolor='white', linewidth=1.0, label=stacode, zorder=2)

        # Direct value labels: every hour for a single station, only large values otherwise
        for bar, value in zip(bars, series.values):
            if pd.isna(value):
                continue
            if single_station or value >= BAR_LABEL_THRESHOLD:
                # Vertical labels on narrow grouped bars so neighboring labels cannot collide
                ax.text(bar.get_x() + bar.get_width() / 2, value, f" {value:.1f}",
                        ha='center', va='bottom', rotation=0 if single_station else 90,
                        fontsize=FONT_SIZE - 4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:%H}00" for h in hours])
    if show_xlabel:
        ax.set_xlabel('Time (BJT)')
    ax.set_ylabel('Hourly rainfall (mm)')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)  # headroom for value labels

    period = f"{ddatetime:%Y-%m-%d} {hours[0]:%H}:00–{hours[-1]:%H}:00"
    if single_station:
        ax.set_title(f"({panel_letter}) {target_stacode}, {period}", loc='left')
    else:
        ax.set_title(f"({panel_letter}) {n_stations} stations, {period}", loc='left')
        ax.legend(loc='upper left', ncol=2)


def main() -> None:
    print(f"Starting climatological limit case plotting at {datetime.now()}")

    set_plot_style()
    set_font_sizes()
    cmap, norm = rain_colormap()

    print("Loading station location info...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    df_stations['stacode'] = df_stations['stacode'].astype(str)

    print(f"Loading {EXTREME_CIRCLES_RADIUS}km extreme circles...")
    df_extreme_circles = load_circles(INPUT_EXTREME_CIRCLES)

    print(f"Loading QC result from {INPUT_QC_RESULT}...")
    df_qc_result = pd.read_csv(INPUT_QC_RESULT, encoding='utf-8-sig', dtype={'stacode': str})
    df_qc_result['ddatetime'] = pd.to_datetime(df_qc_result['ddatetime'])

    print(f"Loading cached adjacent rainfall from {INPUT_CACHE}...")
    df_cache = load_precip_cache(INPUT_CACHE)

    print("Loading DEM...")
    dem_data = rioxarray.open_rasterio(DEM_FILE)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(len(CASES), 2, left=0.02, right=0.98, top=0.98, bottom=0.08,
                          hspace=0.18, wspace=0.10)

    sc = None
    for row, case in enumerate(CASES):
        stacode, ddatetime = case['stacode'], pd.Timestamp(case['ddatetime'])
        print(f"Plotting case {stacode} at {ddatetime}...")
        df_adj = slice_precip_cache(df_cache, ddatetime - timedelta(hours=NEIGHBOR_HOR),
                                    ddatetime + timedelta(hours=NEIGHBOR_HOR))
        letter_map, letter_bar = 'abcdef'[2 * row], 'abcdef'[2 * row + 1]

        sc = subplot_case_map(fig, gs[row, 0], letter_map, stacode, ddatetime, df_adj,
                              df_stations, df_extreme_circles, df_qc_result, dem_data, cmap, norm,
                              map_center=case.get('map_center'))
        subplot_station_bars(fig.add_subplot(gs[row, 1]), letter_bar, stacode, ddatetime,
                             df_adj, case['bar_stations'],
                             show_xlabel=(row == len(CASES) - 1))  # shared x-axis title on the last row only

    # Colorbar for the maps, below the map column
    cbar_ax = fig.add_axes([0.08, 0.035, 0.35, 0.012])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal', extend='max')
    cbar.set_ticks(RAIN_BOUNDS)
    cbar.set_ticklabels([f"{b:g}" for b in RAIN_BOUNDS])
    cbar.set_label(f"Maximum hourly rainfall within ±{NEIGHBOR_HOR} h (mm)")

    fig.savefig(OUTPUT_FILE, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")
    print(f"\nFinished at {datetime.now()}")


if __name__ == '__main__':
    main()
