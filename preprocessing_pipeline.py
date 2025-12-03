#!/usr/bin/env python3
"""
Preprocessing Pipeline for LSTM-based Workload Prediction
Master Thesis: Predictive LSTM and Neuro-Fuzzy Hybrid Autoscaling

This script prepares Borg traces data for LSTM training:
1. Load and parse complex columns
2. Filter and sort temporally
3. Normalize features
4. Create sliding window sequences
5. Save processed data
"""

import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = '/root/dhotok/borg_traces_data.csv'
OUTPUT_DIR = '/root/dhotok/processed_data'
SAMPLE_SIZE = 200_000  # Set to None to use all data
WINDOW_SIZE = 20       # Look-back: 20 timesteps
HORIZON = 1            # Predict 1 step ahead
TRAIN_RATIO = 0.8      # 80% train, 20% validation

# Features for input and target
INPUT_FEATURES = ['avg_cpu', 'avg_mem', 'req_cpu', 'req_mem']
TARGET_FEATURES = ['avg_cpu', 'avg_mem']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_dict_column(value):
    """
    Parse string dictionary (JSON-like) to actual dict.
    Handles various edge cases in the Borg dataset.
    """
    if pd.isna(value) or value == '' or value == '{}':
        return None
    
    try:
        # Try ast.literal_eval first (handles Python dict format)
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            return parsed
        elif isinstance(value, dict):
            return value
    except (ValueError, SyntaxError):
        pass
    
    return None


def extract_cpu_mem(dict_value, prefix):
    """
    Extract CPU and memory values from parsed dictionary.
    Returns tuple (cpu, mem) or (None, None) if invalid.
    """
    if dict_value is None or not isinstance(dict_value, dict):
        return None, None
    
    cpu = dict_value.get('cpus', None)
    mem = dict_value.get('memory', None)
    
    # Handle None values in the dict itself
    if cpu is None or mem is None:
        return None, None
    
    try:
        return float(cpu), float(mem)
    except (ValueError, TypeError):
        return None, None


def create_sequences(data, window_size, horizon, input_cols, target_cols):
    """
    Create sliding window sequences for LSTM.
    
    Args:
        data: DataFrame with features
        window_size: Number of past timesteps to use
        horizon: Number of future timesteps to predict
        input_cols: List of input feature column names
        target_cols: List of target column names
    
    Returns:
        X: Input sequences (samples, window_size, features)
        y: Target values (samples, target_features)
    """
    X, y = [], []
    
    input_data = data[input_cols].values
    target_data = data[target_cols].values
    
    for i in range(len(data) - window_size - horizon + 1):
        # Input: window_size timesteps
        X.append(input_data[i:i + window_size])
        # Target: value at (window_size + horizon - 1) position
        y.append(target_data[i + window_size + horizon - 1])
    
    return np.array(X), np.array(y)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 60)
    print("PREPROCESSING PIPELINE - Borg Traces for LSTM")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading data...")
    
    if SAMPLE_SIZE:
        df = pd.read_csv(DATA_PATH, nrows=SAMPLE_SIZE)
        print(f"  Loaded {len(df):,} rows (sampled)")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"  Loaded {len(df):,} rows (full dataset)")
    
    print(f"  Original columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    # -------------------------------------------------------------------------
    # STEP 2: Parse Complex Columns
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Parsing complex columns (resource_request, average_usage, maximum_usage)...")
    
    # Parse resource_request
    print("  Parsing 'resource_request'...")
    df['_resource_request_parsed'] = df['resource_request'].apply(parse_dict_column)
    df[['req_cpu', 'req_mem']] = df['_resource_request_parsed'].apply(
        lambda x: pd.Series(extract_cpu_mem(x, 'req'))
    )
    
    # Parse average_usage (PRIMARY TARGET)
    print("  Parsing 'average_usage'...")
    df['_average_usage_parsed'] = df['average_usage'].apply(parse_dict_column)
    df[['avg_cpu', 'avg_mem']] = df['_average_usage_parsed'].apply(
        lambda x: pd.Series(extract_cpu_mem(x, 'avg'))
    )
    
    # Parse maximum_usage
    print("  Parsing 'maximum_usage'...")
    df['_maximum_usage_parsed'] = df['maximum_usage'].apply(parse_dict_column)
    df[['max_cpu', 'max_mem']] = df['_maximum_usage_parsed'].apply(
        lambda x: pd.Series(extract_cpu_mem(x, 'max'))
    )
    
    # Drop temporary columns
    df.drop(columns=['_resource_request_parsed', '_average_usage_parsed', '_maximum_usage_parsed'], 
            inplace=True)
    
    # Check parsed results
    print(f"\n  Parsed column stats:")
    for col in ['req_cpu', 'req_mem', 'avg_cpu', 'avg_mem', 'max_cpu', 'max_mem']:
        valid_count = df[col].notna().sum()
        print(f"    {col}: {valid_count:,} valid values ({100*valid_count/len(df):.1f}%)")
    
    # -------------------------------------------------------------------------
    # STEP 3: Filtering & Cleaning
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Filtering and cleaning data...")
    
    initial_count = len(df)
    
    # Drop rows with missing target values (avg_cpu, avg_mem)
    df = df.dropna(subset=['avg_cpu', 'avg_mem', 'req_cpu', 'req_mem'])
    print(f"  After dropping NaN in target columns: {len(df):,} rows")
    
    # Filter out FAIL events if 'event' column exists
    if 'event' in df.columns:
        event_counts = df['event'].value_counts()
        print(f"\n  Event distribution before filtering:")
        for event, count in event_counts.items():
            print(f"    {event}: {count:,}")
        
        # Keep valid events (exclude FAIL for cleaner usage patterns)
        valid_events = ['SCHEDULE', 'FINISH', 'ENABLE', 'EVICT', 'LOST']
        df = df[df['event'].isin(valid_events)]
        print(f"  After filtering FAIL events: {len(df):,} rows")
    
    # Remove outliers (values > 1.0 for normalized metrics, or < 0)
    # Borg data is typically normalized 0-1
    for col in ['avg_cpu', 'avg_mem', 'req_cpu', 'req_mem']:
        before = len(df)
        df = df[(df[col] >= 0) & (df[col] <= 1.0)]
        removed = before - len(df)
        if removed > 0:
            print(f"  Removed {removed:,} outliers from {col}")
    
    print(f"\n  Final dataset: {len(df):,} rows (kept {100*len(df)/initial_count:.1f}%)")
    
    # -------------------------------------------------------------------------
    # STEP 4: Sorting by Time
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Sorting by time (ascending)...")
    
    df = df.sort_values(by='time', ascending=True).reset_index(drop=True)
    print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Feature Selection & Normalization
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Normalizing features with MinMaxScaler...")
    
    # Select features for model
    feature_cols = INPUT_FEATURES.copy()
    df_features = df[feature_cols].copy()
    
    print(f"  Input features: {INPUT_FEATURES}")
    print(f"  Target features: {TARGET_FEATURES}")
    
    # Apply MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_features_scaled = pd.DataFrame(
        scaler.fit_transform(df_features),
        columns=feature_cols
    )
    
    # Save scaler for later use
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to: {scaler_path}")
    
    # Stats after scaling
    print(f"\n  Scaled data statistics:")
    print(df_features_scaled.describe().round(4))
    
    # -------------------------------------------------------------------------
    # STEP 6: Create Sliding Window Sequences
    # -------------------------------------------------------------------------
    print(f"\n[STEP 6] Creating sliding window sequences...")
    print(f"  Window size (look-back): {WINDOW_SIZE}")
    print(f"  Horizon (predict ahead): {HORIZON}")
    
    X, y = create_sequences(
        df_features_scaled, 
        window_size=WINDOW_SIZE, 
        horizon=HORIZON,
        input_cols=INPUT_FEATURES,
        target_cols=TARGET_FEATURES
    )
    
    print(f"  Total sequences created: {len(X):,}")
    print(f"  X shape: {X.shape} (samples, timesteps, features)")
    print(f"  y shape: {y.shape} (samples, targets)")
    
    # -------------------------------------------------------------------------
    # STEP 7: Train/Validation Split (Temporal)
    # -------------------------------------------------------------------------
    print(f"\n[STEP 7] Splitting data temporally ({int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% val)...")
    
    split_idx = int(len(X) * TRAIN_RATIO)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"  Training set:   X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  Validation set: X_val {X_val.shape}, y_val {y_val.shape}")
    
    # -------------------------------------------------------------------------
    # STEP 8: Save Processed Data
    # -------------------------------------------------------------------------
    print(f"\n[STEP 8] Saving processed data to {OUTPUT_DIR}/...")
    
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
    
    # Save metadata
    metadata = {
        'window_size': WINDOW_SIZE,
        'horizon': HORIZON,
        'input_features': INPUT_FEATURES,
        'target_features': TARGET_FEATURES,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'original_data_rows': initial_count,
        'filtered_data_rows': len(df)
    }
    joblib.dump(metadata, os.path.join(OUTPUT_DIR, 'metadata.joblib'))
    
    print(f"  ✓ X_train.npy: {X_train.shape}")
    print(f"  ✓ y_train.npy: {y_train.shape}")
    print(f"  ✓ X_val.npy: {X_val.shape}")
    print(f"  ✓ y_val.npy: {y_val.shape}")
    print(f"  ✓ scaler.joblib")
    print(f"  ✓ metadata.joblib")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"""
Summary:
  - Original rows: {initial_count:,}
  - After filtering: {len(df):,}
  - Total sequences: {len(X):,}
  - Training samples: {len(X_train):,}
  - Validation samples: {len(X_val):,}
  - Input shape per sample: ({WINDOW_SIZE}, {len(INPUT_FEATURES)})
  - Output shape per sample: ({len(TARGET_FEATURES)},)

Next Step: Run training script with LSTM model.
    """)
    
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = main()
