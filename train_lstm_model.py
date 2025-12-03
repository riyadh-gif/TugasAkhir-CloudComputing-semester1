#!/usr/bin/env python3
"""
High-Performance LSTM Training Script
Optimized for NVIDIA A40 (48GB VRAM)

Master Thesis: Predictive LSTM and Neuro-Fuzzy Hybrid Autoscaling
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# GPU CONFIGURATION (Must be done before importing TensorFlow)
# =============================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF logging noise

import tensorflow as tf

print("=" * 70)
print("GPU CONFIGURATION CHECK")
print("=" * 70)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU(s) detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.name}")
    
    # Enable memory growth to avoid OOM (optional for large VRAM)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  Memory growth enabled for {gpu.name}")
        except RuntimeError as e:
            print(f"  Warning: {e}")
    
    # Verify GPU is being used
    print(f"\n✓ TensorFlow version: {tf.__version__}")
    print(f"✓ Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"✓ GPU available for TF: {tf.config.list_physical_devices('GPU')}")
else:
    print("✗ NO GPU DETECTED! Training will use CPU (much slower).")
    print("  Ensure CUDA and cuDNN are properly installed.")

print("=" * 70)

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib

# =============================================================================
# CONFIGURATION - OPTIMIZED FOR NVIDIA A40 (48GB VRAM)
# =============================================================================
DATA_DIR = '/root/dhotok/processed_data'
MODEL_PATH = '/root/dhotok/best_lstm_model.keras'
PLOT_PATH = '/root/dhotok/training_performance.png'

# High-performance settings for A40
BATCH_SIZE = 2048       # Large batch for A40's massive parallelism
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10           # EarlyStopping patience

# Model architecture
LSTM_UNITS_1 = 128      # First LSTM layer
LSTM_UNITS_2 = 64       # Second LSTM layer
DROPOUT_RATE = 0.2
OUTPUT_UNITS = 2        # Predict avg_cpu and avg_mem

# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load preprocessed numpy arrays."""
    print("\n[1/4] LOADING DATA")
    print("-" * 50)
    
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    
    # Load metadata
    metadata = joblib.load(os.path.join(DATA_DIR, 'metadata.joblib'))
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_val shape:   {X_val.shape}")
    print(f"  y_val shape:   {y_val.shape}")
    print(f"  Window size:   {metadata['window_size']}")
    print(f"  Features:      {metadata['input_features']}")
    print(f"  Targets:       {metadata['target_features']}")
    
    # Calculate memory footprint
    total_mb = (X_train.nbytes + y_train.nbytes + X_val.nbytes + y_val.nbytes) / 1e6
    print(f"  Total data in memory: {total_mb:.2f} MB")
    
    return X_train, y_train, X_val, y_val, metadata


# =============================================================================
# MODEL ARCHITECTURE (CuDNN Optimized)
# =============================================================================
def build_model(input_shape):
    """
    Build CuDNN-optimized LSTM model.
    
    IMPORTANT: To use CuDNN acceleration, LSTM layers must have:
    - activation='tanh' (default)
    - recurrent_activation='sigmoid' (default)  
    - use_bias=True (default)
    - unroll=False (default)
    - No manual dropout on recurrent connections
    
    We keep all defaults and use separate Dropout layers.
    """
    print("\n[2/4] BUILDING MODEL (CuDNN Optimized)")
    print("-" * 50)
    
    model = Sequential([
        # Input layer
        Input(shape=input_shape, name='input_sequences'),
        
        # First LSTM layer - returns sequences for stacking
        # Default params ensure CuDNN kernel is used on GPU
        LSTM(
            units=LSTM_UNITS_1,
            return_sequences=True,
            name='lstm_1'
        ),
        Dropout(DROPOUT_RATE, name='dropout_1'),
        
        # Second LSTM layer - returns final state only
        LSTM(
            units=LSTM_UNITS_2,
            return_sequences=False,
            name='lstm_2'
        ),
        Dropout(DROPOUT_RATE, name='dropout_2'),
        
        # Output layer: predict [avg_cpu, avg_mem]
        Dense(
            units=OUTPUT_UNITS,
            activation='linear',  # Regression output
            name='output'
        )
    ])
    
    # Compile with Huber loss (robust to outliers)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  Loss: Huber (delta=1.0)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Architecture: Input({input_shape}) → LSTM({LSTM_UNITS_1}) → "
          f"Dropout({DROPOUT_RATE}) → LSTM({LSTM_UNITS_2}) → "
          f"Dropout({DROPOUT_RATE}) → Dense({OUTPUT_UNITS})")
    
    return model


# =============================================================================
# TRAINING
# =============================================================================
def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with callbacks."""
    print("\n[3/4] TRAINING MODEL")
    print("-" * 50)
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Steps per epoch: {len(X_train) // BATCH_SIZE}")
    print()
    
    # Callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Reduce LR on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train with large batch size optimized for A40
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_training_history(history):
    """Plot training and validation metrics."""
    print("\n[4/4] GENERATING TRAINING PLOTS")
    print("-" * 50)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    ax1 = axes[0]
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Model Loss (Huber)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MAE
    ax2 = axes[1]
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    ax2.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
    print(f"  ✓ Plot saved to: {PLOT_PATH}")
    
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("LSTM TRAINING - HIGH PERFORMANCE MODE (NVIDIA A40)")
    print("=" * 70)
    
    # Load data
    X_train, y_train, X_val, y_val, metadata = load_data()
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    model = build_model(input_shape)
    
    # Print model summary
    print("\nMODEL SUMMARY:")
    print("-" * 50)
    model.summary()
    
    # Train
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot results
    plot_training_history(history)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    # Evaluate on validation set
    val_loss, val_mae, val_mse = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"""
Results:
  - Best model saved to: {MODEL_PATH}
  - Training plot saved to: {PLOT_PATH}
  - Epochs trained: {len(history.history['loss'])}
  
Validation Metrics:
  - Loss (Huber): {val_loss:.6f}
  - MAE: {val_mae:.6f}
  - MSE: {val_mse:.6f}
  - RMSE: {np.sqrt(val_mse):.6f}

Next Step: Add Neuro-Fuzzy layer for hybrid autoscaling decisions.
    """)
    
    return model, history


if __name__ == "__main__":
    model, history = main()
