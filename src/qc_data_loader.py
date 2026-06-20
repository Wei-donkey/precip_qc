# -*- coding: utf-8 -*-
"""
Data loading and fetching utilities for QC validation.

Provides functions for database connections, precipitation data retrieval,
station information loading, and QC validation result processing.
"""

from __future__ import annotations

import configparser
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy import create_engine


def load_db_config(config_path: Path, section: str) -> dict[str, str]:
    """Load database configuration from INI file."""
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8-sig')

    db = config[section]
    return {
        'user': db['user'],
        'password': db['password'],
        'host': db['host'],
        'port': db['port'],
        'service': db['service'],
    }


def create_db_engine(db_config: dict[str, str]):
    """Create SQLAlchemy engine for Oracle database connection."""
    password = quote(db_config['password'])
    conn_string = (f"oracle+oracledb://{db_config['user']}:{password}@{db_config['host']}"
                   f":{db_config['port']}/{db_config['service']}")
    return create_engine(conn_string, echo=False)


def load_station_info(file_path: Path) -> pd.DataFrame:
    """Load station information from CSV file."""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return df


def load_circles(file_path: Path) -> pd.DataFrame:
    """Load circle neighbor information from CSV file."""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df['neighbors'] = df['neighbors'].apply(
        lambda x: [s.strip() for s in str(x).split(',')] if pd.notna(x) else []
    )
    return df


def fetch_hourly_precip(conn, station_type: str, time_stt: pd.Timestamp, time_end: pd.Timestamp) -> pd.DataFrame:
    """Fetch precipitation data between time_stt and time_end using a persistent connection."""
    years = range(time_stt.year, time_end.year + 1)
    
    all_data = []
    for year in years:
        table_name = f"{station_type}_CLI_MUL_HOR_{year}"
        sql = (
            f"SELECT stacode, ddatetime, r FROM {table_name} "
            f"WHERE ddatetime >= TO_DATE('{time_stt}', 'YYYY-MM-DD HH24:MI:SS') "
            f"AND ddatetime <= TO_DATE('{time_end}', 'YYYY-MM-DD HH24:MI:SS')"
        )
        
        try:
            # Use the passed connection instead of creating a new one
            df = pd.read_sql(sql, conn)

            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"Error fetching adjacent data from {table_name}: {e}")

    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all['statype'] = station_type
        return df_all
    else:
        return pd.DataFrame()
    

def fetch_all_station_precip(conn, time_stt: datetime, time_end: datetime, all_stations) -> pd.DataFrame:
    """Fetch current hour precipitation data for both station types and combine."""
    df_surf = fetch_hourly_precip(conn, 'SURF', time_stt, time_end)
    df_awst = fetch_hourly_precip(conn, 'AWST', time_stt, time_end)
    
    if df_surf.empty and df_awst.empty:
        return pd.DataFrame()

    df_all = pd.concat([df_surf, df_awst], ignore_index=True)
    # Filter stations early to reduce memory usage
    df_all = df_all[df_all['stacode'].isin(all_stations) & df_all['r'].notna()]
    df_all['ddatetime'] = pd.to_datetime(df_all['ddatetime'])
    return df_all


def load_qc_result(input_file: Path, qc_type: str = 'false') -> pd.DataFrame:
    """ Filter records based on QC type. """
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    df['ddatetime'] = pd.to_datetime(df['ddatetime'])
    
    if qc_type.startswith('type'):
        qc_label = 'EXTREME_' + qc_type.upper()

        # Filter false positives
        mask = (df['validation'] == False) & (df['qc_label'] == qc_label)
        df_fp = df[mask].reset_index().copy()
        df_fp['confusion_type'] = 'False positive'

        # Filter true positives    
        mask = (df['validation'] == True) & (df['qc_label'] == qc_label)
        df_fn = df[mask].reset_index().copy()
        df_fn['confusion_type'] = 'True positive'

        df_qc_result = pd.concat([df_fp, df_fn], ignore_index=True)

    elif qc_type == 'false':
        # Filter false positives (actual FALSE but predicted as TRUE)
        mask = (df['validation'] == False) & (df['qc_label'] != 'FALSE')
        df_fp = df[mask].reset_index().copy()
        df_fp['confusion_type'] = 'False positive'

        # Filter false negatives (actual TRUE but predicted as FALSE)
        mask = (df['validation'] == True) & (df['qc_label'] == 'FALSE')
        df_fn = df[mask].reset_index().copy()
        df_fn['confusion_type'] = 'False negative'

        df_qc_result = pd.concat([df_fp, df_fn], ignore_index=True)

    elif qc_type == 'false negative': 
        mask = (df['validation'] == True) & (df['qc_label'] == 'FALSE')
        df_qc_result = df[mask].reset_index().copy()
        df_qc_result['confusion_type'] = 'False negative'

    elif qc_type == 'false positive':  
        mask = (df['validation'] == False) & (df['qc_label'] != 'FALSE')
        df_qc_result = df[mask].reset_index().copy()
        df_qc_result['confusion_type'] = 'False positive'
    
    return df_qc_result


def extract_qpe_data(qpe_dataset: xr.Dataset, target_time_utc: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract QPE data for a specific UTC time from the NetCDF dataset.
    
    Parameters:
    -----------
    qpe_dataset : xr.Dataset
        Xarray dataset containing QPE data
    target_time_utc : pd.Timestamp
        Target UTC time to extract
        
    Returns:
    --------
    tuple : (qpe_values, lon_coords, lat_coords)
        QPE values in mm, longitude coordinates, latitude coordinates
    """
    # Check if 'time' dimension exists
    if 'time' not in qpe_dataset.dims:
        raise ValueError("No 'time' dimension found in QPE dataset")
    
    # Get time coordinate values
    times = qpe_dataset['time'].values
    
    # Convert target time to numpy datetime64 for comparison
    target_np = np.datetime64(target_time_utc)
    
    # Find closest time index
    time_diffs = np.abs(times - target_np)
    closest_idx = np.argmin(time_diffs)
    
    # Check if the time difference is acceptable (within 1 hour)
    if time_diffs[closest_idx] > np.timedelta64(1, 'h'):
        print(f"Warning: Closest available time differs by more than 1 hour from {target_time_utc}")
    
    # Extract QPE data for this time step
    qpe_variable = 'qpehour000'
    if qpe_variable not in qpe_dataset.data_vars:
        raise ValueError(f"Variable '{qpe_variable}' not found in QPE dataset")
    
    qpe_slice = qpe_dataset[qpe_variable].isel({'time': closest_idx})
    
    # Convert from 0.1mm to mm and squeeze extra dimensions
    qpe_values = np.squeeze(qpe_slice.values * 0.1)
    
    # Get coordinates
    lon_coords = qpe_slice.lon.values
    lat_coords = qpe_slice.lat.values
    
    return qpe_values, lon_coords, lat_coords


def extract_qpe_in_circles(qpe_values: np.ndarray, lon_coords: np.ndarray, lat_coords: np.ndarray,
                           df_circles: pd.DataFrame) -> np.ndarray:
    """
    Extract QPE grid values that fall within any of the extreme circles.
    
    Parameters:
    -----------
    qpe_values : np.ndarray
        2D array of QPE values in mm
    lon_coords : np.ndarray
        1D array of longitude coordinates
    lat_coords : np.ndarray
        1D array of latitude coordinates
    df_circles : pd.DataFrame
        DataFrame containing circle information with columns: 'lon', 'lat', 'radius_lon', 'radius_lat'
        
    Returns:
    --------
    np.ndarray
        1D array of QPE values (in mm) for all grid points within circles
    """
    if df_circles.empty:
        print("Warning: No circles provided, returning empty array")
        return np.array([])
    
    # Create meshgrid for coordinate comparison
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
    
    # Initialize mask for grids within any circle
    inside_mask = np.zeros_like(qpe_values, dtype=bool)
    
    # Check each circle
    for _, circle in df_circles.iterrows():
        center_lon = circle['lon']
        center_lat = circle['lat']
        radius_lon = circle['radius_lon']
        radius_lat = circle['radius_lat']
        
        # Calculate distance from center using Manhattan distance approximation
        # This is simpler and faster than Euclidean for grid checking
        lon_diff = np.abs(lon_grid - center_lon)
        lat_diff = np.abs(lat_grid - center_lat)
        
        # Check if grid point is within ellipse/circle
        in_circle = (lon_diff <= radius_lon) & (lat_diff <= radius_lat)
        
        # Update mask
        inside_mask |= in_circle
    
    # Extract QPE values for grids within circles
    qpe_in_circles = qpe_values[inside_mask]
    
    return qpe_in_circles
