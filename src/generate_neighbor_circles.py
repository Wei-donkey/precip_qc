# -*- coding: utf-8 -*-
"""
Generate neighbor circles for outlier and extreme detection.

Creates ten sets of circles to include neighboring stations,
defined by radius, grid gap, and neighbor count parameters.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

CIRCLE_TYPE = ['outlier','extreme']  # 'outlier' for outlier circles, 'extreme' for extreme circles

SRC_DIR = Path(__file__).resolve().parent
INPUT_STATIONS = SRC_DIR.parent / 'data' / 'gd_stations_locations.csv'
OUTPUT_DIR_BASENAME = SRC_DIR.parent / 'data' / f"neighbor_circles"
OUTPUT_FILE_BASENAME = f"neighbor_circles"

EARTH_RADIUS_KM = 6371.0

# Grid of Guangdong province
LAT_MIN, LAT_MAX = 20.0, 26.0
LON_MIN, LON_MAX = 109.0, 118.0

LAT_DEGREE_KM = 111  # 1 degree latitude ≈ 111 km

CIRCLE_RADIUS_KM = [10,20,30,40,50,60,70,80,90,100]
CIRCLE_GAP_DEGREE_OUTLIER = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
CIRCLE_GAP_DEGREE_EXTREME = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
CIRCLE_NEIGHBORS_COUNT = [10,20,30,40,50,60,70,80,90,100]


def create_spatial_idx(df_stations: pd.DataFrame):
    """
    Create spatial index for station locations using BallTree.
    """
    coords = df_stations[['lat', 'lon']].values
    return BallTree(np.radians(coords), metric='haversine')


def find_neighboring_stations(df_stations: pd.DataFrame, tree, lats: np.ndarray, lons: np.ndarray, RADIUS_KM, NEIGHBORS_COUNT):
    """  Iterate through circle points [lat, lon] and find neighboring stations within SEARCH_RADIUS_KM.  """
    
    search_radius_rad = RADIUS_KM / EARTH_RADIUS_KM
    total_points = len(lats) * len(lons)
    
    results = []
    
    # Outer loop: Longitude
    for lon in lons:
        # Inner loop: Latitude
        for lat in lats:
            
            # Convert current circle point to radians 
            rad_point = np.radians([[lat, lon]])
            
            # Query stations within radius
            ind, dist = tree.query_radius(
                rad_point,
                r=search_radius_rad,
                return_distance=True,
                sort_results=True
            )
            
            indices = ind[0]
            
            # If number of neighbors is less than or equal to threshold, skip this point
            if len(indices) <= NEIGHBORS_COUNT:
                continue
            
            # Create a circle with RADIUS_KM
            # Convert RADIUS_KM to degrees (approximate at mid-latitude ~22.5°N)            
            LON_DEGREE_KM = LAT_DEGREE_KM * np.cos(np.radians(lat))
            radius_lat = RADIUS_KM / LAT_DEGREE_KM
            radius_lon = RADIUS_KM / LON_DEGREE_KM

            # Get neighbor codes for these indices
            lst_neighbors = df_stations.iloc[indices]['stacode'].astype(str).tolist()
            neighbors_str = ','.join(lst_neighbors)
            
            results.append({
                'lon': lon,
                'lat': lat,
                'radius_lon': radius_lon,
                'radius_lat': radius_lat,
                'count': len(lst_neighbors),
                f"neighbors": neighbors_str,
            })
                    
    return pd.DataFrame(results)


def main() -> None:
    # Input: Read station locations
    print(f"Reading station locations from: {INPUT_STATIONS}")
    df_stations = pd.read_csv(INPUT_STATIONS, encoding='utf-8-sig')
    
    print("Building BallTree for station locations...")
    tree = create_spatial_idx(df_stations)
    
    for circle_type in CIRCLE_TYPE:
        output_dir = OUTPUT_DIR_BASENAME.parent / f"{OUTPUT_DIR_BASENAME.name}_{circle_type}"
        output_dir.mkdir(parents=True, exist_ok=True)
        circle_step_degree = CIRCLE_GAP_DEGREE_OUTLIER if circle_type == 'outlier' else CIRCLE_GAP_DEGREE_EXTREME

        for radius, step, count in zip(CIRCLE_RADIUS_KM, circle_step_degree, CIRCLE_NEIGHBORS_COUNT):
            # Generate circle coordinates
            num_lons = int(np.round((LON_MAX - LON_MIN) / step)) + 1
            num_lats = int(np.round((LAT_MAX - LAT_MIN) / step)) + 1

            lons = np.linspace(LON_MIN, LON_MAX, num_lons)
            lats = np.linspace(LAT_MIN, LAT_MAX, num_lats)

            print(f"Finding neighbors for {circle_type} circle with {step} degrees step and {radius} km search radius...")
            df_neighbors = find_neighboring_stations(df_stations, tree, lats, lons, radius, count)
            
            print(f"Writing results to {OUTPUT_FILE_BASENAME}_{radius}km.csv...")
            df_neighbors.to_csv(output_dir / f"{OUTPUT_FILE_BASENAME}_{circle_type}_{radius}km.csv", index=False, encoding='utf-8-sig',float_format='%.2f')

    print(f"Finished: {datetime.now()}")


if __name__ == '__main__':
    main()