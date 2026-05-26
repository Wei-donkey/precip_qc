# -*- coding: utf-8 -*-
"""
Generate neighboring stations for a fixed spatial grid covering Guangdong.
Grid points are spaced by 0.5 degrees.
For each grid point, find all weather stations within 50km.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

SRC_DIR = Path(__file__).resolve().parent
INPUT_STATIONS = SRC_DIR.parent / 'data' / 'gd_stations_locations.csv'
OUTPUT_GRID_NEIGHBORS = [SRC_DIR.parent / 'data' / 'gd_grids_coarse_neighbors.csv',
                         SRC_DIR.parent / 'data' / 'gd_grids_fine_neighbors.csv']

# Parameters
EARTH_RADIUS_KM = 6371.0
SEARCH_RADIUS_KM = [50.0, 20.0]

# Grid Definition
LAT_MIN, LAT_MAX = 20.0, 25.5
LON_MIN, LON_MAX = 109.5, 117.5
STEPS = [0.5, 0.1]

# Only include grid nodes with at least 50 neighboring stations
MIN_NEIGHBORS = [50, 10]

def build_balltree(df_stations: pd.DataFrame):
    """
    Build BallTree using haversine distance from station locations.
    Input coordinates must be in radians: [lat, lon]
    """
    # Convert to radians for Haversine metric
    coords_rad = np.radians(df_stations[['lat', 'lon']].values)
    tree = BallTree(coords_rad, metric='haversine')
    return tree


def find_neighbors(df_stations: pd.DataFrame, tree, lats: np.ndarray, lons: np.ndarray, search_radius_rad, neighbors):
    """  Iterate through grid points [lat, lon] and find neighboring stations within SEARCH_RADIUS_KM.  """
    
    total_points = len(lats) * len(lons)
    
    results = []
    processed_count = 0
    
    # Outer loop: Longitude
    for lon in lons:
        # Inner loop: Latitude
        for lat in lats:
            
            # Convert current grid point to radians 
            rad_point = np.radians([[lat, lon]])
            
            # Query neighbors within radius
            ind, dist = tree.query_radius(
                rad_point,
                r=search_radius_rad,
                return_distance=True,
                sort_results=True
            )
            
            indices = ind[0]
            
            # If station number le ten, remove this point (skip)
            if len(indices) <= neighbors:
                continue
            
            # Get station codes for these indices
            lst_neighbors = df_stations.iloc[indices]['stacode'].astype(str).tolist()
            neighbors_str = ','.join(lst_neighbors)
            
            results.append({
                'lon': lon,
                'lat': lat,
                f"neighbors": neighbors_str,
                'count': len(lst_neighbors)
            })
                    
            processed_count += 1
            print(f"{processed_count}/{total_points}: Processed {lat},{lon}: {len(lst_neighbors)} neighbors")
                    
    return pd.DataFrame(results)


def main() -> None:
    # Input: Read station locations
    print(f"Reading station locations from: {INPUT_STATIONS}")
    df_stations = pd.read_csv(INPUT_STATIONS, encoding='utf-8-sig')
    
    # Build BallTree from actual stations
    tree = build_balltree(df_stations)
    
    for STEP, OUTPUT, RADIUS_KM, NEIGHBORS in zip(STEPS, OUTPUT_GRID_NEIGHBORS, SEARCH_RADIUS_KM, MIN_NEIGHBORS):
        # Generate grid coordinates
        lats = np.arange(LAT_MIN, LAT_MAX + STEP, STEP)
        lons = np.arange(LON_MIN, LON_MAX + STEP, STEP)
        
        search_radius_rad = RADIUS_KM / EARTH_RADIUS_KM
        # Processing: Generate grid neighbors
        df_neighbors = find_neighbors(df_stations, tree, lats, lons, search_radius_rad, NEIGHBORS)
        
        # Output: Save to CSV
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        df_neighbors.to_csv(OUTPUT, index=False, encoding='utf-8-sig',float_format='%.1f')

        print(f"Wrote {len(df_neighbors)} rows to {OUTPUT}")


if __name__ == '__main__':
    main()