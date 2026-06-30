# -*- coding: utf-8 -*-
"""
Plot QPE data for rainfall events from NetCDF files.

Visualizes quantitative precipitation estimation at event time and ±1 hour,
displayed as colormap overlay on DEM background.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray

# Import common utilities
from qc_plot_utils import (
    plot_qpe,
)

# Import circle processing utilities
from qc_circle_process_utils import (
    get_circles_for_station,
    compute_many_circles_extent,
)

# Import data loading utilities
from qc_data_loader import (
    load_station_info,
    load_circles,
    extract_qpe_data,
)

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / 'data'

DATASET = 'test'  # plot false case from "train" dataset or "test" dataset

# Configuration for neighbor data time window (hours before and after target time)
NEIGHBOR_HOR = 2

# Configuration constants
EXTREME_CIRCLES_RADIUS = 80
MODE = 'many'  # Use 'many' mode extent calculation

INPUT_EVENT_CSV = DATA_DIR / f"qc_result_{DATASET}_qpe_plotting.csv"
INPUT_EXTREME_CIRCLES = DATA_DIR / 'neighbor_circles_extreme' / f"neighbor_circles_extreme_{EXTREME_CIRCLES_RADIUS}km.csv"
INPUT_STATION_LOCATIONS = DATA_DIR / 'gd_stations_locations.csv'
DEM_FILE = DATA_DIR / 'external' / 'gd_dem_1km.tif'


def plot_qpe_event(df_event: pd.DataFrame, df_extreme_circles: pd.DataFrame, 
                   df_stations: pd.DataFrame, dem_data, qpe_dataset):
    # Extract event information
    target_stacode = str(df_event.iloc[0]['stacode'])
    ddatetime_beijing = pd.to_datetime(df_event.iloc[0]['ddatetime'])
    target_precip = float(df_event.iloc[0]['r'])

    confusion_type = df_event.iloc[0]['qc_type']
    output_dir = SRC_DIR.parent / 'figures' / f"qc_{confusion_type}@{MODE}_circle_{DATASET}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing event: Station {target_stacode} at {ddatetime_beijing}")
    print(f"Target precipitation: {target_precip:.1f} mm")
    
    # Get target station location
    target_station = df_stations[df_stations['stacode'] == target_stacode]
    if target_station.empty:
        print(f"Warning: Station {target_stacode} not found in station info, skipping")
        return
    
    target_lon = target_station.iloc[0]['lon']
    target_lat = target_station.iloc[0]['lat']
    
    # Get extreme circles for this station
    df_circles = get_circles_for_station(target_stacode, df_extreme_circles)
    
    if df_circles.empty:
        print(f"Warning: No extreme circles found for station {target_stacode}, skipping")
        return
    
    # Compute map extent using 'many' mode
    map_extent = compute_many_circles_extent(df_circles)
    
    # Generate 3 time steps: current, -1 hour, +1 hour (Beijing time)
    time_offsets = [i for i in range(-NEIGHBOR_HOR,NEIGHBOR_HOR+1)]
    
    for offset in time_offsets:
        # Calculate Beijing time for this step
        ddatetime_beijing_step = ddatetime_beijing + timedelta(hours=offset)

        datetime_str = ddatetime_beijing_step.strftime('%Y%m%d%H')
        output_file = output_dir / f"qc_{confusion_type}_{target_stacode}_{datetime_str}_qpe.png"

        # Convert to UTC (Beijing is UTC+8)
        ddatetime_utc = ddatetime_beijing_step - timedelta(hours=8)
        
        print(f"  Extracting QPE for UTC time: {ddatetime_utc}")

        # Extract QPE data
        qpe_values, lon_coords, lat_coords = extract_qpe_data(qpe_dataset, ddatetime_utc)
                
        # Plot QPE
        plot_qpe(
            target_stacode=target_stacode,
            ddatetime_utc=ddatetime_utc,
            target_precip=target_precip,
            qpe_data=qpe_values,
            x_coords=lon_coords,
            y_coords=lat_coords,
            df_circles=df_circles,
            target_lon=target_lon,
            target_lat=target_lat,
            map_extent=map_extent,
            df_extreme_circles=df_circles,
            output_file=output_file,
            dem_data=dem_data,
            confusion_type=confusion_type,
        )


def main():
    """Main function to process all rainfall events."""

    print("Loading DEM data...")
    raster_dem = rioxarray.open_rasterio(DEM_FILE)

    print("Loading station location info...")
    df_stations = load_station_info(INPUT_STATION_LOCATIONS)
    
    print("Loading extreme circles...")
    df_extreme_circles = load_circles(INPUT_EXTREME_CIRCLES)
    
    
    print("Loading rainfall events...")
    df_events = pd.read_csv(INPUT_EVENT_CSV, encoding='utf-8-sig')
    df_events['ddatetime'] = pd.to_datetime(df_events['ddatetime'])
    df_events['stacode'] = df_events['stacode'].astype(str)
    total_events = len(df_events)
    
    print(f"\nProcessing {total_events} rainfall event(s)...")
    
    for idx, (_, event_row) in enumerate(df_events.iterrows()):
        print(f"\nEvent {idx + 1}/{total_events}")
        df_event = pd.DataFrame([event_row])
        yyyymm = df_event.iloc[0]['ddatetime'].strftime('%Y%m')
        
        print("Loading QPE dataset...")
        qpe_file = DATA_DIR / 'external' / f"qpehour{yyyymm}.nc"
        qpe_dataset = xr.open_dataset(qpe_file)

        plot_qpe_event(
            df_event=df_event,
            df_extreme_circles=df_extreme_circles,
            df_stations=df_stations,
            dem_data=raster_dem,
            qpe_dataset=qpe_dataset
        )
    
    # Close QPE dataset
    qpe_dataset.close()
    
    print(f"\nFinished at {datetime.now()}")


if __name__ == '__main__':
    main()
