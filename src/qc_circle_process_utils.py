# -*- coding: utf-8 -*-
"""
Circle processing utilities for QC validation.

Provides functions for filtering circles, computing map extents,
constructing validation DataFrames, and spatial operations on circle data.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# Constants for extent computation
FEW_CIRCLES_GAP = 0.8
MORE_CIRCLES_GAP = 0.4


def get_circles_for_station(target_stacode: str, df_circles: pd.DataFrame) -> pd.DataFrame:
    """Filter coarse circles that contain the given station."""
    # Check which circles contain the station in their neighbors list
    mask = df_circles['neighbors'].apply(lambda neighbors: target_stacode in neighbors)
    return df_circles[mask].reset_index()


def compute_single_circle_extent(df_circles: pd.DataFrame) -> list[float]:
    """Compute map extent from circle centers and radii."""

    lon_min = df_circles.loc[0,'lon']-df_circles.loc[0,'radius_lon']
    lon_max = df_circles.loc[0,'lon']+df_circles.loc[0,'radius_lon']
    lat_min = df_circles.loc[0,'lat']-df_circles.loc[0,'radius_lat']
    lat_max = df_circles.loc[0,'lat']+df_circles.loc[0,'radius_lat']
    
    # fix the ratio of latitude extent to longitude extent
    lon_max = lon_min + 1.05*(lat_max-lat_min)
    # Add small gap
    gap_lon = (lon_max - lon_min) * 0.2
    gap_lat = (lat_max - lat_min) * 0.2
    
    return [lon_min - gap_lon, lon_max + gap_lon, lat_min - gap_lat, lat_max + gap_lat]


def compute_few_circles_extent(df_circles: pd.DataFrame) -> list[float]:
    """Compute map extent from circle centers and radii."""

    df_lon_tmp = df_circles.groupby(['lon']).agg({'lat': 'min', 'radius_lat': 'mean', 'radius_lon': 'mean'}).reset_index()
    num_circles_lon = df_lon_tmp.shape[0]
    df_lat_tmp = df_circles.groupby(['lat']).agg({'lon': 'min', 'radius_lon': 'mean', 'radius_lat': 'mean'}).reset_index()
    num_circles_lat = df_lat_tmp.shape[0]

    lon_min = df_lon_tmp['lon'].min() - df_lon_tmp['radius_lon'].mean()
    lon_max = df_lon_tmp['lon'].max() + df_lon_tmp['radius_lon'].mean()

    if num_circles_lon == 1:
        lon_min -= 1*FEW_CIRCLES_GAP/2
        lon_max += 1*FEW_CIRCLES_GAP/2

    lat_min = df_lat_tmp['lat'].min() - df_lat_tmp['radius_lat'].mean()
    lat_max = df_lat_tmp['lat'].max() + df_lat_tmp['radius_lat'].mean()

    if num_circles_lat == 1:
        lat_min -= 1*FEW_CIRCLES_GAP/2
        lat_max += 1*FEW_CIRCLES_GAP/2
    
    # fix the ratio of latitude extent to longitude extent
    lon_max = lon_min + 1.05*(lat_max-lat_min)
    # Add small gap
    gap_lon = (lon_max - lon_min) * 0.1
    gap_lat = (lat_max - lat_min) * 0.1
    
    return [lon_min - gap_lon, lon_max + gap_lon, lat_min - gap_lat, lat_max + gap_lat]


def compute_many_circles_extent(df_circles: pd.DataFrame) -> list[float]:
    """Compute map extent from circle centers and radii."""

    df_lon_tmp = df_circles.groupby(['lon']).agg({'lat': 'min', 'radius_lat': 'mean', 'radius_lon': 'mean'}).reset_index()
    num_circles_lon = df_lon_tmp.shape[0]
    df_lat_tmp = df_circles.groupby(['lat']).agg({'lon': 'min', 'radius_lon': 'mean', 'radius_lat': 'mean'}).reset_index()
    num_circles_lat = df_lat_tmp.shape[0]

    lon_min = df_lon_tmp['lon'].min() - df_lon_tmp['radius_lon'].mean()
    lon_max = df_lon_tmp['lon'].max() + df_lon_tmp['radius_lon'].mean()

    if num_circles_lon == 3:
        lon_min -= 1*MORE_CIRCLES_GAP/2
        lon_max += 1*MORE_CIRCLES_GAP/2
    elif num_circles_lon == 2:
        lon_min -= 2*MORE_CIRCLES_GAP/2
        lon_max += 2*MORE_CIRCLES_GAP/2
    elif num_circles_lon == 1:
        lon_min -= 3*MORE_CIRCLES_GAP/2
        lon_max += 3*MORE_CIRCLES_GAP/2

    lat_min = df_lat_tmp['lat'].min() - df_lat_tmp['radius_lat'].mean()
    lat_max = df_lat_tmp['lat'].max() + df_lat_tmp['radius_lat'].mean()

    if num_circles_lat == 3:
        lat_min -= 1*MORE_CIRCLES_GAP/2
        lat_max += 1*MORE_CIRCLES_GAP/2
    elif num_circles_lat == 2:
        lat_min -= 2*MORE_CIRCLES_GAP/2
        lat_max += 2*MORE_CIRCLES_GAP/2
    elif num_circles_lat == 1:
        lat_min -= 3*MORE_CIRCLES_GAP/2
        lat_max += 3*MORE_CIRCLES_GAP/2
     
    # fix the ratio of latitude extent to longitude extent
    lon_max = lon_min + 1.05*(lat_max-lat_min)   
    # Add small gap
    gap_lon = (lon_max - lon_min) * 0.05
    gap_lat = (lat_max - lat_min) * 0.05

    return [lon_min - gap_lon, lon_max + gap_lon, lat_min - gap_lat, lat_max + gap_lat]


def find_single_extreme_circle(extreme_circles: pd.DataFrame, target_stacode: str, df_station_info: pd.DataFrame) -> pd.DataFrame:
    """
    Find the extreme circle whose center is closest to the target station's location.
    """
    # Get target station location
    target_station = df_station_info[df_station_info['stacode'] == target_stacode]
    
    target_lon = target_station.iloc[0]['lon']
    target_lat = target_station.iloc[0]['lat']
    
    # Calculate Euclidean distance from each circle center to target station
    # Using squared distance to avoid sqrt computation (preserves ordering)
    extreme_circles = extreme_circles.copy()
    extreme_circles['dist_sq'] = (
        (extreme_circles['lon'] - target_lon) ** 2 + 
        (extreme_circles['lat'] - target_lat) ** 2
    )
    
    # Find the circle with minimum distance
    min_dist_idx = extreme_circles['dist_sq'].idxmin()
    closest_extreme_circle = extreme_circles.loc[[min_dist_idx]].drop(columns=['dist_sq']).reset_index(drop=True)
    
    return closest_extreme_circle


def construct_validation_circles(validation_circle_locs_str: str, df_circles: pd.DataFrame) -> pd.DataFrame:
    """
    Construct DataFrame of validation circles by filtering df_circles based on validation locations.
    """
    # Parse validation circle locations from string
    lst_validation_circle_locs = []
    if pd.notna(validation_circle_locs_str):
        locs = validation_circle_locs_str.split(';')
        for loc in locs:
            str_lon, str_lat = loc.strip("()").split(',')  # Remove parentheses and split by comma
            lst_validation_circle_locs.append((float(str_lon), float(str_lat)))
    
    # Filter df_circles to find matching validation circles
    if lst_validation_circle_locs:
        # Create a mask to filter circles whose (lon, lat) match validation locations
        validation_mask = df_circles.apply(
            lambda row: (round(row['lon'], 4), round(row['lat'], 4)) in 
            [(round(lon, 4), round(lat, 4)) for lon, lat in lst_validation_circle_locs],
            axis=1
        )
        df_validation_circles = df_circles[validation_mask].reset_index(drop=True)
    else:
        df_validation_circles = pd.DataFrame()  # Empty DataFrame if no validation circles
    
    return df_validation_circles
