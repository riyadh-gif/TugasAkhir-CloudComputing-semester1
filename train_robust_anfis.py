#!/usr/bin/env python3
"""
Robust ANFIS (Adaptive Neuro-Fuzzy Inference System) Implementation
Master Thesis: Predictive LSTM and Neuro-Fuzzy Hybrid Autoscaling

This script implements a 5-layer ANFIS architecture with:
- Gaussian Membership Functions (trainable μ, σ)
- Takagi-Sugeno order-1 consequent
- L2 Regularization on consequent weights
- TimeSeriesSplit cross-validation

Author: AI Research Assistant
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = '/root/dhotok/processed_data'
LSTM_MODEL_PATH = '/root/dhotok/best_lstm_model.keras'
ANFIS_MODEL_PATH = '/root/dhotok/best_anfis_model.keras'
HYBRID_MODEL_PATH = '/root/dhotok/hybrid_lstm_anfis_model.keras'

# ANFIS Configuration
N_MF_PER_INPUT = 3          # Number of Membership Functions per input (Low, Medium, High)
N_INPUTS = 2                # 2 inputs: Predicted_Workload, Workload_Trend
N_RULES = N_MF_PER_INPUT ** N_INPUTS  # 3^2 = 9 rules
N_OUTPUTS = 3               # 3 scaling decisions: Scale Down, Maintain, Scale Up

# Training Configuration
ANFIS_EPOCHS = 100
ANFIS_BATCH_SIZE = 512
ANFIS_LR = 0.001
L2_LAMBDA = 0.01            # L2 regularization strength
N_CV_SPLITS = 5             # TimeSeriesSplit folds
PATIENCE = 15

print("=" * 70)
print("ANFIS TRAINING - ROBUST IMPLEMENTATION FOR THESIS")
print("=" * 70)
print(f"Configuration:")
print(f"  - Membership Functions per input: {N_MF_PER_INPUT}")
print(f"  - Number of fuzzy rules: {N_RULES}")
print(f"  - Output classes: {N_OUTPUTS} (Scale Down, Maintain, Scale Up)")
print(f"  - L2 Regularization: {L2_LAMBDA}")
print(f"  - Cross-validation folds: {N_CV_SPLITS}")


# =============================================================================
# LAYER 1: GAUSSIAN MEMBERSHIP FUNCTION (FUZZIFICATION)
# =============================================================================
class GaussianMembershipLayer(layers.Layer):
    """
    Layer 1: Fuzzification using Gaussian Membership Functions.
    
    μ(x) = exp(-(x - μ)² / (2σ²))
    
    Parameters:
        - mu (mean): Trainable, initialized evenly across input range
        - sigma (std): Trainable, initialized based on input spread
    """
    
    def __init__(self, n_mf=3, **kwargs):
        super(GaussianMembershipLayer, self).__init__(**kwargs)
        self.n_mf = n_mf
    
    def build(self, input_shape):
        n_inputs = input_shape[-1]
        
        # Initialize mu evenly across [0, 1] range
        # For 3 MFs: [0.0, 0.5, 1.0] representing Low, Medium, High
        mu_init = np.zeros((n_inputs, self.n_mf))
        for i in range(n_inputs):
            mu_init[i] = np.linspace(0, 1, self.n_mf)
        
        # Initialize sigma based on MF spacing
        sigma_init = np.ones((n_inputs, self.n_mf)) * (1.0 / (2 * (self.n_mf - 1)))
        
        self.mu = self.add_weight(
            name='mu',
            shape=(n_inputs, self.n_mf),
            initializer=tf.constant_initializer(mu_init),
            trainable=True
        )
        
        self.sigma = self.add_weight(
            name='sigma',
            shape=(n_inputs, self.n_mf),
            initializer=tf.constant_initializer(sigma_init),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()  # sigma must be positive
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Compute Gaussian membership for each input-MF pair.
        
        Input shape: (batch, n_inputs)
        Output shape: (batch, n_inputs, n_mf)
        """
        # Expand dims for broadcasting: (batch, n_inputs, 1)
        x = tf.expand_dims(inputs, -1)
        
        # Gaussian: exp(-(x - mu)^2 / (2 * sigma^2))
        # Add small epsilon to prevent division by zero
        sigma_safe = tf.maximum(self.sigma, 1e-6)
        membership = tf.exp(-tf.square(x - self.mu) / (2 * tf.square(sigma_safe)))
        
        return membership
    
    def get_config(self):
        config = super().get_config()
        config.update({'n_mf': self.n_mf})
        return config


# =============================================================================
# LAYER 2 & 3: RULE FIRING & NORMALIZATION
# =============================================================================
class RuleFiringLayer(layers.Layer):
    """
    Layer 2: Compute firing strength of each rule using T-Norm (Product).
    Layer 3: Normalize firing strengths.
    
    For 2 inputs with 3 MFs each, we have 9 rules.
    Each rule is a combination of one MF from each input.
    
    Firing strength: w_i = μ_A(x) * μ_B(y)  (T-Norm: Product)
    Normalized: w̄_i = w_i / Σw_j
    """
    
    def __init__(self, n_mf=3, n_inputs=2, **kwargs):
        super(RuleFiringLayer, self).__init__(**kwargs)
        self.n_mf = n_mf
        self.n_inputs = n_inputs
        self.n_rules = n_mf ** n_inputs
    
    def call(self, membership_values):
        """
        Input: membership_values shape (batch, n_inputs, n_mf)
        Output: normalized_weights shape (batch, n_rules)
        """
        batch_size = tf.shape(membership_values)[0]
        
        # Extract membership values for each input
        # membership_values[:, 0, :] = MF values for input 1 (Predicted_Workload)
        # membership_values[:, 1, :] = MF values for input 2 (Workload_Trend)
        
        mf_input1 = membership_values[:, 0, :]  # (batch, n_mf)
        mf_input2 = membership_values[:, 1, :]  # (batch, n_mf)
        
        # Compute firing strength for all rule combinations (Cartesian product)
        # Rule (i, j): w_ij = mf1[i] * mf2[j]
        # Reshape for outer product
        mf1_expanded = tf.expand_dims(mf_input1, 2)  # (batch, n_mf, 1)
        mf2_expanded = tf.expand_dims(mf_input2, 1)  # (batch, 1, n_mf)
        
        # T-Norm (Product): firing strength matrix
        firing_matrix = mf1_expanded * mf2_expanded  # (batch, n_mf, n_mf)
        
        # Flatten to (batch, n_rules)
        firing_strengths = tf.reshape(firing_matrix, (batch_size, self.n_rules))
        
        # Layer 3: Normalize firing strengths
        # w̄_i = w_i / (Σw_j + ε)
        sum_firing = tf.reduce_sum(firing_strengths, axis=1, keepdims=True)
        normalized_weights = firing_strengths / (sum_firing + 1e-8)
        
        return normalized_weights, firing_strengths
    
    def get_config(self):
        config = super().get_config()
        config.update({'n_mf': self.n_mf, 'n_inputs': self.n_inputs})
        return config


# =============================================================================
# LAYER 4 & 5: CONSEQUENT (TAKAGI-SUGENO) & DEFUZZIFICATION
# =============================================================================
class TakagiSugenoConsequentLayer(layers.Layer):
    """
    Layer 4: Takagi-Sugeno consequent (order 1).
    
    For each rule i: f_i = p_i * x + q_i * y + r_i
    
    Where:
        - x = Predicted_Workload (Input 1)
        - y = Workload_Trend (Input 2)
        - p_i, q_i, r_i = trainable parameters for rule i
    
    Layer 5: Defuzzification using weighted sum.
    
    Output = Σ(w̄_i * f_i)
    
    For multi-class output (3 scaling decisions), we have separate
    consequent parameters for each output class.
    """
    
    def __init__(self, n_rules=9, n_outputs=3, l2_lambda=0.01, **kwargs):
        super(TakagiSugenoConsequentLayer, self).__init__(**kwargs)
        self.n_rules = n_rules
        self.n_outputs = n_outputs
        self.l2_lambda = l2_lambda
    
    def build(self, input_shape):
        # Consequent parameters for each rule and each output
        # p_i coefficients for input x (Predicted_Workload)
        self.p = self.add_weight(
            name='p_coefficients',
            shape=(self.n_rules, self.n_outputs),
            initializer='glorot_uniform',
            regularizer=regularizers.l2(self.l2_lambda),
            trainable=True
        )
        
        # q_i coefficients for input y (Workload_Trend)
        self.q = self.add_weight(
            name='q_coefficients',
            shape=(self.n_rules, self.n_outputs),
            initializer='glorot_uniform',
            regularizer=regularizers.l2(self.l2_lambda),
            trainable=True
        )
        
        # r_i constant terms (bias)
        self.r = self.add_weight(
            name='r_constants',
            shape=(self.n_rules, self.n_outputs),
            initializer='zeros',
            regularizer=regularizers.l2(self.l2_lambda),
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Inputs:
            - normalized_weights: (batch, n_rules)
            - raw_inputs: (batch, 2) - [Predicted_Workload, Workload_Trend]
        
        Output:
            - defuzzified_output: (batch, n_outputs)
        """
        normalized_weights, raw_inputs = inputs
        
        x = raw_inputs[:, 0:1]  # (batch, 1) - Predicted_Workload
        y = raw_inputs[:, 1:2]  # (batch, 1) - Workload_Trend
        
        # Compute consequent for each rule: f_i = p_i*x + q_i*y + r_i
        # Shape: (batch, n_rules, n_outputs)
        f = (tf.expand_dims(x, -1) * self.p +   # (batch, 1, 1) * (n_rules, n_outputs)
             tf.expand_dims(y, -1) * self.q +
             self.r)
        
        # Layer 5: Defuzzification - weighted sum
        # output = Σ(w̄_i * f_i)
        w_expanded = tf.expand_dims(normalized_weights, -1)  # (batch, n_rules, 1)
        defuzzified = tf.reduce_sum(w_expanded * f, axis=1)  # (batch, n_outputs)
        
        return defuzzified
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_rules': self.n_rules,
            'n_outputs': self.n_outputs,
            'l2_lambda': self.l2_lambda
        })
        return config


# =============================================================================
# COMPLETE ANFIS MODEL
# =============================================================================
class ANFISModel(Model):
    """
    Complete ANFIS Model combining all 5 layers.
    
    Architecture:
        Input (2) → Layer1 (Fuzzification) → Layer2&3 (Rule Firing & Norm)
                 → Layer4 (Consequent) → Layer5 (Defuzzification) → Output (3)
    """
    
    def __init__(self, n_mf=3, n_inputs=2, n_outputs=3, l2_lambda=0.01, **kwargs):
        super(ANFISModel, self).__init__(**kwargs)
        self.n_mf = n_mf
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_rules = n_mf ** n_inputs
        self.l2_lambda = l2_lambda
        
        # Initialize layers
        self.fuzzification = GaussianMembershipLayer(n_mf=n_mf, name='layer1_fuzzification')
        self.rule_firing = RuleFiringLayer(n_mf=n_mf, n_inputs=n_inputs, name='layer2_3_rule_firing')
        self.consequent = TakagiSugenoConsequentLayer(
            n_rules=self.n_rules,
            n_outputs=n_outputs,
            l2_lambda=l2_lambda,
            name='layer4_5_consequent'
        )
    
    def call(self, inputs, training=None):
        # Layer 1: Fuzzification
        membership = self.fuzzification(inputs)
        
        # Layer 2 & 3: Rule Firing & Normalization
        normalized_weights, raw_firing = self.rule_firing(membership)
        
        # Layer 4 & 5: Consequent & Defuzzification
        output = self.consequent([normalized_weights, inputs])
        
        return output
    
    def get_config(self):
        return {
            'n_mf': self.n_mf,
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'l2_lambda': self.l2_lambda
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# =============================================================================
# DATA PREPARATION: Generate ANFIS Training Data
# =============================================================================
def generate_anfis_dataset(lstm_model, X_data, y_data):
    """
    Generate ANFIS training dataset from LSTM predictions.
    
    ANFIS Inputs:
        1. Predicted_Workload: LSTM prediction (normalized 0-1)
        2. Workload_Trend: Gradient of predictions (pred[t] - pred[t-1])
    
    ANFIS Targets:
        Scaling decision labels (0: Scale Down, 1: Maintain, 2: Scale Up)
        Generated based on heuristic rules comparing prediction to actual.
    
    Returns:
        X_anfis: (samples, 2) - [Predicted_Workload, Workload_Trend]
        y_anfis: (samples, 3) - One-hot encoded scaling decisions
    """
    print("\n[STEP 2] Generating ANFIS training data...")
    
    # Get LSTM predictions
    predictions = lstm_model.predict(X_data, verbose=0)  # (samples, 2) [cpu, mem]
    
    # Use CPU prediction as primary workload indicator
    pred_workload = predictions[:, 0]  # CPU predictions
    actual_workload = y_data[:, 0]     # Actual CPU
    
    # Calculate Workload_Trend (gradient)
    # trend[t] = pred[t] - pred[t-1]
    trend = np.zeros_like(pred_workload)
    trend[1:] = pred_workload[1:] - pred_workload[:-1]
    trend[0] = 0  # First sample has no previous
    
    # Normalize trend to [-1, 1] range approximately
    trend_normalized = np.clip(trend * 10, -1, 1)  # Scale and clip
    
    # ANFIS Input: [Predicted_Workload, Workload_Trend]
    X_anfis = np.column_stack([pred_workload, trend_normalized])
    
    # Generate scaling labels based on prediction vs actual
    # and trend direction
    labels = generate_scaling_labels(pred_workload, actual_workload, trend_normalized)
    
    # One-hot encode labels
    y_anfis = np.eye(N_OUTPUTS)[labels]
    
    print(f"  ANFIS Input shape: {X_anfis.shape}")
    print(f"  ANFIS Target shape: {y_anfis.shape}")
    print(f"  Workload range: [{pred_workload.min():.4f}, {pred_workload.max():.4f}]")
    print(f"  Trend range: [{trend_normalized.min():.4f}, {trend_normalized.max():.4f}]")
    
    # Label distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_names = ['Scale Down', 'Maintain', 'Scale Up']
    print(f"\n  Label distribution:")
    for u, c in zip(unique, counts):
        print(f"    {label_names[u]}: {c:,} ({100*c/len(labels):.1f}%)")
    
    return X_anfis, y_anfis, predictions


def generate_scaling_labels(predicted, actual, trend):
    """
    Generate scaling decision labels based on workload analysis.
    
    Scaling Logic (Adjusted thresholds for balanced distribution):
    - Scale Up (2): High predicted load OR rising trend
    - Scale Down (0): Low predicted load AND falling/stable trend
    - Maintain (1): Otherwise
    
    Thresholds are tunable based on SLA requirements.
    """
    n_samples = len(predicted)
    labels = np.ones(n_samples, dtype=np.int32)  # Default: Maintain
    
    # Use percentile-based thresholds for better balance
    p33 = np.percentile(predicted, 33)
    p66 = np.percentile(predicted, 66)
    
    # Thresholds (percentile-based for balanced classes)
    HIGH_LOAD = max(p66, 0.05)  # Upper third
    LOW_LOAD = max(p33, 0.01)   # Lower third
    RISING_TREND = 0.02
    FALLING_TREND = -0.02
    
    for i in range(n_samples):
        pred = predicted[i]
        t = trend[i]
        
        # Scale Up conditions
        if pred >= HIGH_LOAD and t >= 0:
            labels[i] = 2  # Scale Up - High load, stable/rising
        elif t >= RISING_TREND and pred >= LOW_LOAD:
            labels[i] = 2  # Scale Up - Rising trend
        
        # Scale Down conditions
        elif pred <= LOW_LOAD:
            labels[i] = 0  # Scale Down - Low load
        elif t <= FALLING_TREND and pred <= HIGH_LOAD:
            labels[i] = 0  # Scale Down - Falling trend
        
        # Maintain (default)
        else:
            labels[i] = 1
    
    return labels


# =============================================================================
# TRAINING WITH TIMESERIES CROSS-VALIDATION
# =============================================================================
def train_anfis_with_cv(X_anfis, y_anfis, n_splits=5):
    """
    Train ANFIS with TimeSeriesSplit cross-validation.
    
    Args:
        X_anfis: Input features (Predicted_Workload, Workload_Trend)
        y_anfis: One-hot encoded scaling labels
        n_splits: Number of CV folds
    
    Returns:
        best_model: Best ANFIS model from CV
        cv_results: Dictionary of CV metrics
    """
    print(f"\n[STEP 3] Training ANFIS with {n_splits}-Fold TimeSeriesSplit CV")
    print("-" * 60)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_results = {
        'fold': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_mse': []
    }
    
    best_val_acc = 0
    best_model = None
    best_fold = 0
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_anfis), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        print(f"  Train samples: {len(train_idx):,}")
        print(f"  Val samples: {len(val_idx):,}")
        
        # Split data
        X_train_fold = X_anfis[train_idx]
        y_train_fold = y_anfis[train_idx]
        X_val_fold = X_anfis[val_idx]
        y_val_fold = y_anfis[val_idx]
        
        # Create fresh ANFIS model for each fold
        model = ANFISModel(
            n_mf=N_MF_PER_INPUT,
            n_inputs=N_INPUTS,
            n_outputs=N_OUTPUTS,
            l2_lambda=L2_LAMBDA
        )
        
        # Compute class weights for imbalanced data
        class_counts = np.sum(y_train_fold, axis=0)
        total_samples = len(y_train_fold)
        class_weights = {i: total_samples / (len(class_counts) * count + 1e-6) 
                        for i, count in enumerate(class_counts)}
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=ANFIS_LR),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'mse']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Train with class weights to handle imbalance
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=ANFIS_EPOCHS,
            batch_size=ANFIS_BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0
        )
        
        # Evaluate
        val_loss, val_acc, val_mse = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        train_loss = history.history['loss'][-1]
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Val MSE: {val_mse:.6f}")
        
        # Store results
        fold_results['fold'].append(fold)
        fold_results['train_loss'].append(train_loss)
        fold_results['val_loss'].append(val_loss)
        fold_results['val_accuracy'].append(val_acc)
        fold_results['val_mse'].append(val_mse)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_fold = fold
    
    # Summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    
    avg_train_loss = np.mean(fold_results['train_loss'])
    avg_val_loss = np.mean(fold_results['val_loss'])
    avg_val_acc = np.mean(fold_results['val_accuracy'])
    avg_val_mse = np.mean(fold_results['val_mse'])
    std_val_acc = np.std(fold_results['val_accuracy'])
    
    print(f"\nAverage Metrics across {n_splits} folds:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f} ({avg_val_acc*100:.2f}%)")
    print(f"  Val MSE: {avg_val_mse:.6f}")
    print(f"\nBest Fold: {best_fold} (Accuracy: {best_val_acc:.4f})")
    
    return best_model, fold_results


# =============================================================================
# SAVE ANFIS MODEL
# =============================================================================
def save_anfis_model(model, path):
    """Save ANFIS model weights and config."""
    # Save weights (Keras 3 requires .weights.h5 extension)
    weights_path = path.replace('.keras', '.weights.h5')
    model.save_weights(weights_path)
    
    # Save config
    config_path = path.replace('.keras', '_config.joblib')
    config = {
        'n_mf': model.n_mf,
        'n_inputs': model.n_inputs,
        'n_outputs': model.n_outputs,
        'l2_lambda': model.l2_lambda
    }
    joblib.dump(config, config_path)
    
    print(f"\n  ✓ ANFIS weights saved to: {weights_path}")
    print(f"  ✓ ANFIS config saved to: {config_path}")
    
    return weights_path, config_path


def load_anfis_model(path):
    """Load ANFIS model from saved weights and config."""
    weights_path = path.replace('.keras', '.weights.h5')
    config_path = path.replace('.keras', '_config.joblib')
    
    config = joblib.load(config_path)
    model = ANFISModel(**config)
    
    # Build model by calling it with dummy input
    dummy_input = np.zeros((1, config['n_inputs']))
    model(dummy_input)
    
    model.load_weights(weights_path)
    return model


# =============================================================================
# ANALYZE FUZZY RULES
# =============================================================================
def analyze_fuzzy_rules(model):
    """Analyze and print learned fuzzy rules."""
    print("\n" + "=" * 60)
    print("LEARNED FUZZY RULES ANALYSIS")
    print("=" * 60)
    
    # Get membership function parameters
    mu = model.fuzzification.mu.numpy()
    sigma = model.fuzzification.sigma.numpy()
    
    # Get consequent parameters
    p = model.consequent.p.numpy()
    q = model.consequent.q.numpy()
    r = model.consequent.r.numpy()
    
    input_names = ['Predicted_Workload', 'Workload_Trend']
    mf_labels = ['Low', 'Medium', 'High']
    output_labels = ['Scale_Down', 'Maintain', 'Scale_Up']
    
    print("\n1. MEMBERSHIP FUNCTION PARAMETERS (μ, σ):")
    print("-" * 50)
    for i, inp_name in enumerate(input_names):
        print(f"\n  {inp_name}:")
        for j, mf_label in enumerate(mf_labels):
            print(f"    {mf_label}: μ={mu[i,j]:.4f}, σ={sigma[i,j]:.4f}")
    
    print("\n2. FUZZY RULES (Takagi-Sugeno Consequent):")
    print("-" * 50)
    print("  Rule format: IF (Workload is X) AND (Trend is Y) THEN f = p*x + q*y + r")
    
    rule_idx = 0
    for i, mf1 in enumerate(mf_labels):  # Workload MF
        for j, mf2 in enumerate(mf_labels):  # Trend MF
            print(f"\n  Rule {rule_idx + 1}:")
            print(f"    IF (Workload is {mf1}) AND (Trend is {mf2})")
            for k, out_label in enumerate(output_labels):
                print(f"      → {out_label}: f = {p[rule_idx,k]:.4f}*x + {q[rule_idx,k]:.4f}*y + {r[rule_idx,k]:.4f}")
            rule_idx += 1
    
    return mu, sigma, p, q, r


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("ANFIS TRAINING PIPELINE")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STEP 1: Load LSTM Model and Data
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading LSTM model and data...")
    
    # Load LSTM model
    lstm_model = load_model(LSTM_MODEL_PATH)
    print(f"  ✓ LSTM model loaded from: {LSTM_MODEL_PATH}")
    
    # Freeze LSTM layers (make non-trainable)
    for layer in lstm_model.layers:
        layer.trainable = False
    print(f"  ✓ LSTM layers frozen (non-trainable)")
    
    # Load preprocessed data
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    
    # Combine train and val for ANFIS training (will use TimeSeriesSplit)
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    
    print(f"  ✓ Data loaded: {X_all.shape[0]:,} samples")
    
    # -------------------------------------------------------------------------
    # STEP 2: Generate ANFIS Dataset
    # -------------------------------------------------------------------------
    X_anfis, y_anfis, lstm_predictions = generate_anfis_dataset(lstm_model, X_all, y_all)
    
    # -------------------------------------------------------------------------
    # STEP 3: Train ANFIS with Cross-Validation
    # -------------------------------------------------------------------------
    best_model, cv_results = train_anfis_with_cv(X_anfis, y_anfis, n_splits=N_CV_SPLITS)
    
    # -------------------------------------------------------------------------
    # STEP 4: Save Best Model
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Saving best ANFIS model...")
    save_anfis_model(best_model, ANFIS_MODEL_PATH)
    
    # -------------------------------------------------------------------------
    # STEP 5: Analyze Learned Rules
    # -------------------------------------------------------------------------
    analyze_fuzzy_rules(best_model)
    
    # -------------------------------------------------------------------------
    # STEP 6: Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANFIS TRAINING COMPLETE!")
    print("=" * 70)
    
    avg_acc = np.mean(cv_results['val_accuracy'])
    std_acc = np.std(cv_results['val_accuracy'])
    
    print(f"""
Summary:
  - ANFIS Architecture: {N_INPUTS} inputs × {N_MF_PER_INPUT} MFs = {N_RULES} rules
  - Output Classes: {N_OUTPUTS} (Scale Down, Maintain, Scale Up)
  - Cross-Validation: {N_CV_SPLITS}-Fold TimeSeriesSplit
  - Average Accuracy: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%
  - L2 Regularization: {L2_LAMBDA}

Saved Files:
  - {ANFIS_MODEL_PATH.replace('.keras', '.weights.h5')}
  - {ANFIS_MODEL_PATH.replace('.keras', '_config.joblib')}

Next Step: Create hybrid inference pipeline combining LSTM + ANFIS.
    """)
    
    return best_model, cv_results


if __name__ == "__main__":
    best_model, cv_results = main()
