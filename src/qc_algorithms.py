# -*- coding: utf-8 -*-
"""
Core algorithms for precipitation spatial consistency quality control.

This module provides standalone functions for outlier and extreme circle evaluations.
It is designed to be imported by other scripts rather than executed directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Constants used in the evaluation logic
CLIMATE_LIMIT = 184.4
QC_THRESHOLD = 10
EXTREME_TYPES = ['EXTREME_TYPE1', 'EXTREME_TYPE2', 'EXTREME_TYPE3', 'EXTREME_TYPE4', 'EXTREME_TYPE5'] # extreme type corresponding to determined score
EXTREME_DETERMINE_SCORE = [2, 3, 4, 5, 6]  # score threshold denoting number of neighbors to validate outliers
EXTREME_DETERMINE_COEFF = [0.5, 0.4, 0.3, 0.2, 0.1]  # Coefficients for establishing thresholds (coef*outlier) for extreme check
EXTREME_FINAL_TYPE = 'EXTREME_TYPE6'  # Extreme type assigned to outliers that pass the final check but not the initial check

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


def determine_extreme_type(r_surf, r_awst, target_statype, target_precip, extreme_type_idx):
    """ Determine whether a given event is an extreme event and its type. """

    r_threshold = EXTREME_DETERMINE_COEFF[extreme_type_idx] * target_precip
    r_surf_confidence = [r for r in r_surf if r >= r_threshold]
    r_awst_confidence = [r for r in r_awst if r >= r_threshold]
    confidence_score = 2*len(r_surf_confidence) + len(r_awst_confidence)    
    score_threshold = EXTREME_DETERMINE_SCORE[extreme_type_idx]
    
    if target_statype == 'SURF': 
        score_threshold -=1  # If the target station is SURF, we give it extra confidence and reduce the threshold by 1

    if confidence_score >= score_threshold:
        is_extreme = True
        extreme_type = EXTREME_TYPES[extreme_type_idx]
    else:
        is_extreme = False
        extreme_type = None

    return is_extreme, extreme_type


def extreme_inspection(filtered_extreme_circles, df_all_adjacent, target_stacode, target_statype, target_precip):
    is_extreme, qc_label, validation_sample_size = False, None, None

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

        # Sort neighbors by precipitation in descending order
        df_neighbors = df_neighbors.sort_values(by='r', ascending=False)
        df_neighbors.reset_index(drop=True, inplace=True)
       
        # Extract values of SURF-type stations
        r_surf = df_neighbors[df_neighbors['statype']=='SURF']['r'].values
        r_awst = df_neighbors[df_neighbors['statype']=='AWST']['r'].values
        
        for extreme_type in EXTREME_TYPES:
            extreme_type_idx = EXTREME_TYPES.index(extreme_type)
            
            is_extreme, extreme_type = determine_extreme_type(r_surf, r_awst, target_statype, target_precip, extreme_type_idx)
            if is_extreme:
                qc_label, validation_sample_size = extreme_type, validation_sample_size
                break
        
        if is_extreme:
            break  # quit the loop of extreme circles

    return is_extreme, qc_label, validation_sample_size


def extra_extreme_inspection(filtered_extreme_circles, df_all_adjacent, target_stacode, target_statype, target_precip):
    is_extreme, qc_label, validation_sample_size = False, None, None

    # Final check using all extreme neighbors (excluding the outlier station itself)
    neighbors = filtered_extreme_circles['neighbors'].explode().unique()
    neighbors = [s for s in neighbors if s != target_stacode]

    # Filter adjacent data for these neighbors
    df_neighbors = df_all_adjacent[df_all_adjacent['stacode'].isin(neighbors)]
    # Exclude neighbors outside of climate limit
    df_neighbors = df_neighbors[df_neighbors['r']<=CLIMATE_LIMIT]
    
    if not df_neighbors.empty:

        validation_sample_size = df_neighbors.shape[0]

        # Sort neighbors by precipitation in descending order
        df_neighbors = df_neighbors.sort_values(by='r', ascending=False)
        df_neighbors.reset_index(drop=True, inplace=True)

        # Extract values of SURF-type stations
        r_surf = df_neighbors[df_neighbors['statype']=='SURF']['r'].values
        r_awst = df_neighbors[df_neighbors['statype']=='AWST']['r'].values
        
        for extreme_type in EXTREME_TYPES:
            extreme_type_idx = EXTREME_TYPES.index(extreme_type)

            is_extreme, extreme_type = determine_extreme_type(r_surf, r_awst, target_statype, target_precip, extreme_type_idx)
            if is_extreme:
                qc_label, validation_sample_size = extreme_type, validation_sample_size
                break

        if is_extreme is not None:
            extreme_type = EXTREME_FINAL_TYPE  # manually set to "EXTREME_TYPE6"

    return is_extreme, qc_label, validation_sample_size


def perform_extreme_inspection(target_stacode: str, target_statype:str, target_precip: float, df_extreme_circles: pd.DataFrame, df_all_adjacent: pd.DataFrame,) -> None:
    """
    Perform extreme circle evaluation for OUTLIER records in-place.
    
    For each outlier, checks if certain amount of neighboring stations within extreme circles 
    have precipitation values that satisfy specific thresholds based on the OUTLIER record.
    Labels as EXTREME_TYPE1-6 if conditions are met, otherwise FALSE.
    """
    
    # Find extreme circles containing this outlier station
    extreme_circles_mask = df_extreme_circles['neighbors'].apply(lambda neighbors: target_stacode in neighbors)
    filtered_extreme_circles = df_extreme_circles[extreme_circles_mask]
    
    is_extreme, qc_label, validation_sample_size = extreme_inspection(filtered_extreme_circles, df_all_adjacent, target_stacode, target_statype, target_precip)
    
    if not is_extreme:
        is_extreme, qc_label, validation_sample_size = extra_extreme_inspection(filtered_extreme_circles, df_all_adjacent, target_stacode, target_statype, target_precip)
    
    # Label based on evaluation result
    if not is_extreme:
        qc_label = 'FALSE'
        if validation_sample_size is None:
            validation_sample_size = 0
    
    return qc_label, validation_sample_size


if __name__ == '__main__':
    print("This module is intended to be imported, not run directly.")
