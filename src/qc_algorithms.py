# -*- coding: utf-8 -*-
"""
Core algorithms for precipitation spatial consistency quality control.

Provides standalone functions for outlier and extreme circle evaluations,
designed to be imported by other scripts rather than executed directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Constants used in the evaluation logic
CLIMATE_LIMIT = 184.4
QC_THRESHOLD = 10
EXTREME_ITERATION_LIMIT = 5  # Number of iterative extreme inspection to evaluate for each outlier
EXTREME_QC_LABELS = ['EXTREME_TYPE1', 'EXTREME_TYPE2', 'EXTREME_TYPE3', 'EXTREME_TYPE4', 'EXTREME_TYPE5'] # extreme type corresponding to confidence score , 'EXTREME_TYPE5'
EXTREME_QC_LABELS = EXTREME_QC_LABELS[:EXTREME_ITERATION_LIMIT]

EXTREME_TCS_THRESHOLD = [2, 3, 4, 5, 6]  # confidence score threshold denoting number of neighbors to validate outliers , 6
EXTREME_CONFIDENCE_COEFF = [0.5, 0.4, 0.3, 0.2, 0.1]  # confidence coefficients for establishing thresholds (coef*outlier) for extreme check , 0.1


def calculate_p_max(precip_values: pd.Series) -> float:
    """ Calculate P_max threshold using IQR method. """
    if len(precip_values) == 0:
        return np.inf
        
    q1 = precip_values.quantile(0.25)
    q3 = precip_values.quantile(0.75)
    iqr = q3 - q1
    
    # Manually set iqr to 0.1 if it is zero to avoid p_max being equal to q3
    if iqr == 0:
        iqr = 0.1
        
    p_max = q3 + 3 * iqr
    return p_max


def perform_outlier_detection(df_all: pd.DataFrame, df_outlier_circles: pd.DataFrame) -> pd.DataFrame:
    """
    Perform outlier circle spatial consistency check and label records as NORMAL or OUTLIER.
    
    For each outlier circle, calculates P_max using IQR method from neighbor stations
    marking records with precipitation <= P_max as NORMAL. 
    Records with precipitation > P_max are labeled as OUTLIER.
    """
    
    # Automatically label low precipitation as NORMAL
    df_all.loc[df_all['r'] < QC_THRESHOLD, 'qc_label'] = 'NORMAL'

    for _, single_circle in df_outlier_circles.iterrows():
        neighbor_stations = single_circle['neighbors']

        # Filter precipitation data for neighbor stations
        neighbor_mask = df_all['stacode'].isin(neighbor_stations)
        df_neighbors = df_all[neighbor_mask]
        
        if len(df_neighbors) == 0:
            continue

        # Skip if all neighbor stations have precipitation < QC_THRESHOLD
        # The column "validation_sample_size" of these skipped stations will remain None
        if df_neighbors['r'].max() < QC_THRESHOLD:
            continue
        
        p_max = calculate_p_max(df_neighbors['r'])
        
        df_all.loc[(neighbor_mask) & (df_all['r'] <= p_max), 'qc_label'] = 'NORMAL'
        df_all.loc[(neighbor_mask) & (df_all['r'] <= p_max), 'validation_sample_size'] = df_neighbors.shape[0]
    
    # Label remaining unlabeled records as OUTLIER
    df_all.loc[df_all['qc_label'] != 'NORMAL', 'qc_label'] = 'OUTLIER'


def compute_total_confidence_score(df_neighbors, r_threshold):

    # Sort neighbors by precipitation in descending order
    df_neighbors = df_neighbors.sort_values(by='r', ascending=False)
    df_neighbors.reset_index(drop=True, inplace=True)

    # Extract values of SURF-type stations
    r_surf = df_neighbors[df_neighbors['statype']=='SURF']['r'].values
    r_awst = df_neighbors[df_neighbors['statype']=='AWST']['r'].values

    r_surf_confidence = [r for r in r_surf if r >= r_threshold]
    r_awst_confidence = [r for r in r_awst if r >= r_threshold]

    total_confidence_score = 2*len(r_surf_confidence) + len(r_awst_confidence)

    return total_confidence_score


def perform_extreme_inspection(target_stacode: str, target_statype:str, target_precip: float, 
                               filtered_extreme_circles: pd.DataFrame, df_all_adjacent: pd.DataFrame, loop_all_circle: bool=False) -> None:
    """
    Perform extreme circle evaluation for OUTLIER records in-place.
    
    For each outlier, checks if certain amount of neighboring stations within extreme circles 
    have precipitation values that satisfy specific thresholds based on the OUTLIER record.
    Labels as EXTREME_TYPE1-5 if conditions are met, otherwise FALSE.
    If loop_all_circle is True, the extreme_inspection() will loop all extreme circles containing the outlier station;
    If loop_all_circle is False, the extreme_inspection() will break once an extreme circle validates the outlier station as extreme.
    """

    is_extreme, qc_label, validation_sample_size = False, None, None

    for extreme_type in EXTREME_QC_LABELS:
        extreme_type_idx = EXTREME_QC_LABELS.index(extreme_type)
        extreme_tcs_threshold = EXTREME_TCS_THRESHOLD[extreme_type_idx]
        
        # If the target station is SURF, we give it extra confidence and reduce the threshold by 1
        if target_statype == 'SURF':
            extreme_tcs_threshold -= 1
        confidence_coeff = EXTREME_CONFIDENCE_COEFF[extreme_type_idx]
        r_threshold = confidence_coeff * target_precip

        # initialize counts and locations for extreme circles which validate the outlier station as extreme
        extreme_circle_count = filtered_extreme_circles.shape[0]
        validation_circle_count = 0
        validation_circle_locs = []

        # Loop through each filtered extreme circle
        for _, single_circle in filtered_extreme_circles.iterrows():
            neighbors = single_circle['neighbors']
            
            # Exclude the outlier station itself
            neighbors = [s for s in neighbors if s != target_stacode]

            if len(neighbors) == 0:
                continue
            
            # Filter adjacent data for these neighbors
            df_neighbors = df_all_adjacent[df_all_adjacent['stacode'].isin(neighbors)]
            # Exclude neighbors outside of climate limit
            df_neighbors = df_neighbors[df_neighbors['r']<=CLIMATE_LIMIT]           
            
            if df_neighbors.empty:
                continue

            validation_sample_size = df_neighbors.shape[0]

            total_confidence_score = compute_total_confidence_score(df_neighbors, r_threshold)

            if total_confidence_score >= extreme_tcs_threshold:
                is_extreme = True
                validation_circle_count += 1
                validation_circle_locs.append((single_circle['lon'], single_circle['lat']))
                if not loop_all_circle:
                    break
            # else:
                # is_extreme = False

        if is_extreme:
            qc_label, validation_sample_size, extreme_circle_count, validation_circle_count, validation_circle_locs \
            = extreme_type, validation_sample_size, extreme_circle_count, validation_circle_count, validation_circle_locs
            break
        else:
            validation_circle_count = 0
            validation_circle_locs = []

    validation_circle_locs = ';'.join([f'{loc}' for loc in validation_circle_locs])

    # Label based on evaluation result
    if not is_extreme:
        qc_label = 'FALSE'

    if validation_sample_size is None:
        validation_sample_size = 0
    
    return qc_label, validation_sample_size, extreme_circle_count, validation_circle_count, validation_circle_locs


if __name__ == '__main__':
    print("This module is intended to be imported, not run directly.")
