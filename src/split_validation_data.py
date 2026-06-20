# -*- coding: utf-8 -*-
"""
Split validation data into training and testing sets.

Performs stratified random split to maintain label proportions
for parameter tuning and final evaluation.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / 'data'

INPUT_VALIDATION_MAIN = DATA_DIR / 'gd_validation_precip_2003-2025.csv'
INPUT_VALIDATION_SUPPLEMENT = DATA_DIR / 'gd_validation_precip_supplement.csv'

OUTPUT_TRAIN_SET = DATA_DIR / f"gd_validation_data_train.csv"
OUTPUT_TEST_SET = DATA_DIR / f"gd_validation_data_test.csv"

# Configuration
TARGET_TOTAL_SIZE = 2500
TEST_SPLIT_RATIO = 0.4  # 40% for testing
RANDOM_STATE = 42


def load_validation_data(file_path: Path) -> pd.DataFrame:
    """Load validation data from CSV file."""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return df


def downsample_validation_data(df_main, df_supp, target_total=TARGET_TOTAL_SIZE, random_state=RANDOM_STATE):
    """ Merges data and downsamples the 30-50mm range to manage dataset size. """
    # 1. Merge Data
    df_combined = pd.concat([df_main, df_supp], ignore_index=True)
    # Remove duplicates based on station and time
    df_combined.drop_duplicates(subset=['stacode', 'ddatetime'], inplace=True)
    
    # Ensure 'validation' is boolean
    df_combined['validation'] = df_combined['validation'].astype(bool)
    
    # 2. Split into High Precip (>50) and Moderate Precip (30-50)
    df_mask = (df_combined['r'] > 50) | (df_combined['validation']==False)
    df_rare = df_combined[df_mask].copy()
    df_mod = df_combined[~df_mask].copy()
    
    print(f"Original counts: High (>50): {len(df_rare)}, Moderate (30-50): {len(df_mod)}")
    
    # 3. Determine how many moderate records to keep
    n_rare = len(df_rare)
    if n_rare >= target_total:
        # If rare precip records already exceed target, sample from them too
        # (Though usually we want to keep all rare precip cases)
        df_final = df_rare.sample(n=target_total, random_state=random_state)
    else:
        n_needed = target_total - n_rare
        if n_needed < len(df_mod):
            # Randomly sample the moderate records
            df_sampled = df_mod.sample(n=n_needed, random_state=random_state)
        else:
            # If we need more than available, keep all moderate records
            df_sampled = df_mod
            
        df_final = pd.concat([df_rare, df_sampled], ignore_index=True)
        
    # 4. Shuffle the final dataset
    df_final = df_final.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Final dataset size: {len(df_final)}")
    print(f"  - High Precip (>50): {len(df_rare)}")
    print(f"  - Moderate Precip (30-50): {len(df_sampled)}")
    
    return df_final


def split_validation_sets(df_validation: pd.DataFrame, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE):
    """ Merges main and supplement validation data, then splits into train/test sets. """
    
    # 4. Stratified Random Split
    df_train, df_test = train_test_split(
        df_validation, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df_validation['validation']  # ensures the ratio of TRUE/FALSE is preserved
    )

    df_train.sort_values(by=['ddatetime','stacode'], ascending=False, inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    df_test.sort_values(by=['ddatetime','stacode'], ascending=False, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def main():
    # Load Data
    df_main = load_validation_data(INPUT_VALIDATION_MAIN)
    df_supp = load_validation_data(INPUT_VALIDATION_SUPPLEMENT)

    print('downsampling validation data...')
    df_validation_sampled = downsample_validation_data(df_main, df_supp)

    print('splitting validation data...')
    df_train, df_test = split_validation_sets(df_validation_sampled)

    df_train.to_csv(OUTPUT_TRAIN_SET, index=False, encoding='utf-8-sig')
    df_test.to_csv(OUTPUT_TEST_SET, index=False, encoding='utf-8-sig')
    
    print(f"Training set: {len(df_train)} (True: {df_train['validation'].sum()}, False: {(~df_train['validation']).sum()})")
    print(f"Testing set: {len(df_test)} (True: {df_test['validation'].sum()}, False: {(~df_test['validation']).sum()})")


if __name__ == '__main__':
    main()