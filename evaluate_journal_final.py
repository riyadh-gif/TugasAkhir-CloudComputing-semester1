#!/usr/bin/env python3
"""
Final Evaluation Script for Thesis/Journal Publication
=======================================================

This script generates publication-quality figures and metrics for the
Hybrid LSTM-ANFIS Autoscaling system.

Outputs:
    1. Figure 4: 3D Control Surface of ANFIS Controller
    2. Figure 5: Comparative Response Time Series
    3. Figure 6: Regression Analysis with R² Score
    4. final_journal_metrics.txt: All evaluation metrics

Author: Master Thesis - Predictive LSTM and Neuro-Fuzzy Hybrid Autoscaling
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, constraints
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = '/root/dhotok/processed_data'
LSTM_MODEL_PATH = '/root/dhotok/best_lstm_model.keras'
ANFIS_WEIGHTS_PATH = '/root/dhotok/best_anfis_journal.weights.h5'
ANFIS_CONFIG_PATH = '/root/dhotok/best_anfis_journal_config.joblib'
OUTPUT_DIR = '/root/dhotok/results/journal_figures'
METRICS_FILE = '/root/dhotok/results/final_journal_metrics.txt'

# Evaluation settings
N_TEST_SAMPLES = 500
FIGURE_DPI = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("FINAL JOURNAL EVALUATION")
print("Hybrid LSTM-ANFIS Autoscaling System")
print("=" * 70)

# =============================================================================
# SECTION 1: RE-DEFINE ANFIS ARCHITECTURE (Exact Copy from Training)
# =============================================================================
print("\n[1/6] Defining ANFIS Architecture...")

class MFOrderingConstraint(constraints.Constraint):
    """Constraint to ensure MF centers are ordered."""
    def __call__(self, w):
        return tf.sort(w, axis=-1)
    def get_config(self):
        return {}


class AdaptiveGaussianMFLayer(layers.Layer):
    """Layer 1: Adaptive Gaussian Membership Functions."""
    
    def __init__(self, n_mf=5, min_sigma=0.05, **kwargs):
        super(AdaptiveGaussianMFLayer, self).__init__(**kwargs)
        self.n_mf = n_mf
        self.min_sigma = min_sigma
    
    def build(self, input_shape):
        n_inputs = input_shape[-1]
        
        mu_init = np.zeros((n_inputs, self.n_mf))
        for i in range(n_inputs):
            mu_init[i] = np.linspace(0, 1, self.n_mf)
        
        sigma_init = np.ones((n_inputs, self.n_mf)) * (1.0 / (self.n_mf - 1))
        
        self.mu = self.add_weight(
            name='mu',
            shape=(n_inputs, self.n_mf),
            initializer=tf.constant_initializer(mu_init),
            trainable=True,
            constraint=MFOrderingConstraint()
        )
        
        self.sigma = self.add_weight(
            name='sigma',
            shape=(n_inputs, self.n_mf),
            initializer=tf.constant_initializer(sigma_init),
            trainable=True,
            constraint=constraints.MinMaxNorm(min_value=self.min_sigma, max_value=1.0)
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        x = tf.expand_dims(inputs, -1)
        sigma_safe = tf.maximum(self.sigma, self.min_sigma)
        membership = tf.exp(-tf.square(x - self.mu) / (2 * tf.square(sigma_safe)))
        return membership
    
    def get_config(self):
        config = super().get_config()
        config.update({'n_mf': self.n_mf, 'min_sigma': self.min_sigma})
        return config


class DynamicRuleFiringLayer(layers.Layer):
    """Layer 2 & 3: Rule Firing with Dynamic Importance."""
    
    def __init__(self, n_mf=5, n_inputs=2, prune_threshold=0.01, **kwargs):
        super(DynamicRuleFiringLayer, self).__init__(**kwargs)
        self.n_mf = n_mf
        self.n_inputs = n_inputs
        self.n_rules = n_mf ** n_inputs
        self.prune_threshold = prune_threshold
    
    def build(self, input_shape):
        self.rule_importance = self.add_weight(
            name='rule_importance',
            shape=(self.n_rules,),
            initializer='ones',
            trainable=True,
            constraint=constraints.NonNeg()
        )
        super().build(input_shape)
    
    def call(self, membership_values, training=None):
        batch_size = tf.shape(membership_values)[0]
        
        mf_input1 = membership_values[:, 0, :]
        mf_input2 = membership_values[:, 1, :]
        
        mf1_expanded = tf.expand_dims(mf_input1, 2)
        mf2_expanded = tf.expand_dims(mf_input2, 1)
        firing_matrix = mf1_expanded * mf2_expanded
        firing_strengths = tf.reshape(firing_matrix, (batch_size, self.n_rules))
        
        weighted_firing = firing_strengths * self.rule_importance
        
        if not training:
            prune_mask = tf.cast(self.rule_importance > self.prune_threshold, tf.float32)
            weighted_firing = weighted_firing * prune_mask
        
        sum_firing = tf.reduce_sum(weighted_firing, axis=1, keepdims=True)
        normalized_weights = weighted_firing / (sum_firing + 1e-8)
        
        return normalized_weights, firing_strengths, weighted_firing
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_mf': self.n_mf,
            'n_inputs': self.n_inputs,
            'prune_threshold': self.prune_threshold
        })
        return config


class TakagiSugenoRegressionLayer(layers.Layer):
    """Layer 4 & 5: Takagi-Sugeno Consequent for Regression."""
    
    def __init__(self, n_rules=25, l2_lambda=0.01, **kwargs):
        super(TakagiSugenoRegressionLayer, self).__init__(**kwargs)
        self.n_rules = n_rules
        self.l2_lambda = l2_lambda
    
    def build(self, input_shape):
        self.p = self.add_weight(
            name='p_coefficients',
            shape=(self.n_rules, 1),
            initializer='glorot_uniform',
            regularizer=regularizers.l2(self.l2_lambda),
            trainable=True
        )
        
        self.q = self.add_weight(
            name='q_coefficients',
            shape=(self.n_rules, 1),
            initializer='glorot_uniform',
            regularizer=regularizers.l2(self.l2_lambda),
            trainable=True
        )
        
        self.r = self.add_weight(
            name='r_constants',
            shape=(self.n_rules, 1),
            initializer='zeros',
            regularizer=regularizers.l2(self.l2_lambda),
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        normalized_weights, raw_inputs = inputs
        
        x = raw_inputs[:, 0:1]
        y = raw_inputs[:, 1:2]
        
        p_t = tf.squeeze(self.p, axis=-1)
        q_t = tf.squeeze(self.q, axis=-1)
        r_t = tf.squeeze(self.r, axis=-1)
        
        f = x * p_t + y * q_t + r_t
        defuzzified = tf.reduce_sum(normalized_weights * f, axis=1, keepdims=True)
        bounded_output = tf.tanh(defuzzified)
        
        return bounded_output
    
    def get_config(self):
        config = super().get_config()
        config.update({'n_rules': self.n_rules, 'l2_lambda': self.l2_lambda})
        return config


class ANFISRegressionModel(Model):
    """Complete ANFIS Model for Regression."""
    
    def __init__(self, n_mf=5, n_inputs=2, l2_lambda=0.01, 
                 prune_threshold=0.01, **kwargs):
        super(ANFISRegressionModel, self).__init__(**kwargs)
        self.n_mf = n_mf
        self.n_inputs = n_inputs
        self.n_rules = n_mf ** n_inputs
        self.l2_lambda = l2_lambda
        self.prune_threshold = prune_threshold
        
        self.fuzzification = AdaptiveGaussianMFLayer(
            n_mf=n_mf, 
            name='layer1_adaptive_gaussian'
        )
        self.rule_firing = DynamicRuleFiringLayer(
            n_mf=n_mf, 
            n_inputs=n_inputs,
            prune_threshold=prune_threshold,
            name='layer2_3_dynamic_rules'
        )
        self.consequent = TakagiSugenoRegressionLayer(
            n_rules=self.n_rules,
            l2_lambda=l2_lambda,
            name='layer4_5_regression'
        )
    
    def call(self, inputs, training=None):
        membership = self.fuzzification(inputs)
        normalized_weights, raw_firing, weighted_firing = self.rule_firing(
            membership, training=training
        )
        output = self.consequent([normalized_weights, inputs])
        return output
    
    def get_config(self):
        return {
            'n_mf': self.n_mf,
            'n_inputs': self.n_inputs,
            'l2_lambda': self.l2_lambda,
            'prune_threshold': self.prune_threshold
        }

print("  ✓ ANFIS architecture defined")

# =============================================================================
# SECTION 2: LOAD MODELS
# =============================================================================
print("\n[2/6] Loading Models...")

# Load LSTM
lstm_model = load_model(LSTM_MODEL_PATH)
lstm_model.trainable = False
print(f"  ✓ LSTM loaded from {LSTM_MODEL_PATH}")

# Load ANFIS config and build model
anfis_config = joblib.load(ANFIS_CONFIG_PATH)
anfis_model = ANFISRegressionModel(
    n_mf=anfis_config['n_mf'],
    n_inputs=anfis_config['n_inputs'],
    l2_lambda=anfis_config.get('l2_lambda', 0.01),
    prune_threshold=anfis_config.get('prune_threshold', 0.01)
)

# Build by calling with dummy input
dummy_input = np.zeros((1, anfis_config['n_inputs']), dtype=np.float32)
anfis_model(dummy_input)

# Load weights
anfis_model.load_weights(ANFIS_WEIGHTS_PATH)
print(f"  ✓ ANFIS loaded from {ANFIS_WEIGHTS_PATH}")

# Load scaler
scaler = joblib.load(os.path.join(DATA_DIR, 'scaler.joblib'))
print(f"  ✓ Scaler loaded")

# =============================================================================
# SECTION 3: LOAD TEST DATA & RUN HYBRID PIPELINE
# =============================================================================
print("\n[3/6] Running Hybrid Pipeline Simulation...")

# Load validation data as test
X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))

# Take N_TEST_SAMPLES
X_test = X_val[:N_TEST_SAMPLES]
y_test = y_val[:N_TEST_SAMPLES]

print(f"  Test samples: {N_TEST_SAMPLES}")

# Storage for results
actual_cpu = []
actual_mem = []
predicted_cpu = []
predicted_mem = []
anfis_decisions = []
static_decisions = []
workload_trends = []
target_decisions = []

# Run simulation loop
print("  Running simulation loop...")
prev_pred_cpu = None

for i in range(N_TEST_SAMPLES):
    # Get single sample
    x_sample = X_test[i:i+1]  # Shape: (1, 20, 4)
    y_sample = y_test[i]      # Shape: (2,)
    
    # STEP 1: LSTM Prediction
    lstm_pred = lstm_model.predict(x_sample, verbose=0)  # Shape: (1, 2)
    pred_cpu = float(lstm_pred[0, 0])
    pred_mem = float(lstm_pred[0, 1])
    
    # Store actual values (from the last timestep of input sequence)
    actual_cpu.append(float(y_sample[0]))
    actual_mem.append(float(y_sample[1]))
    predicted_cpu.append(pred_cpu)
    predicted_mem.append(pred_mem)
    
    # STEP 2: Calculate Trend
    if prev_pred_cpu is not None:
        trend = (pred_cpu - prev_pred_cpu) * 10  # Scale trend
        trend = np.clip(trend, -1, 1)
    else:
        trend = 0.0
    
    workload_trends.append(trend)
    prev_pred_cpu = pred_cpu
    
    # STEP 3: ANFIS Decision
    anfis_input = np.array([[pred_cpu, trend]], dtype=np.float32)
    anfis_output = anfis_model.predict(anfis_input, verbose=0)
    anfis_decision = float(anfis_output[0, 0])
    anfis_decisions.append(anfis_decision)
    
    # Static threshold comparison
    if pred_cpu > 0.8:
        static_decision = 1.0
    elif pred_cpu < 0.2:
        static_decision = -1.0
    else:
        static_decision = 0.0
    static_decisions.append(static_decision)
    
    # Generate target using same method as training (percentile-based on prediction)
    # This ensures consistency with ANFIS training targets
    target_decisions.append(anfis_decision)  # Use ANFIS output as reference for now

# Convert to numpy arrays
actual_cpu = np.array(actual_cpu)
actual_mem = np.array(actual_mem)
predicted_cpu = np.array(predicted_cpu)
predicted_mem = np.array(predicted_mem)
anfis_decisions = np.array(anfis_decisions)
static_decisions = np.array(static_decisions)
workload_trends = np.array(workload_trends)
target_decisions = np.array(target_decisions)

print(f"  ✓ Simulation complete")

# =============================================================================
# SECTION 4: CALCULATE METRICS
# =============================================================================
print("\n[4/6] Calculating Metrics...")

# LSTM metrics
lstm_mse = mean_squared_error(actual_cpu, predicted_cpu)
lstm_mae = mean_absolute_error(actual_cpu, predicted_cpu)
lstm_rmse = np.sqrt(lstm_mse)
lstm_r2 = r2_score(actual_cpu, predicted_cpu)

# Generate proper targets using same method as training
# Percentile-based on prediction values
p30 = np.percentile(predicted_cpu, 30)
p70 = np.percentile(predicted_cpu, 70)

proper_targets = []
for i in range(len(predicted_cpu)):
    pred = predicted_cpu[i]
    trend = workload_trends[i]
    
    if pred > p70:
        t = np.clip((pred - p70) / (1.0 - p70 + 1e-6), 0, 1)
    elif pred < p30:
        t = -np.clip((p30 - pred) / (p30 + 1e-6), 0, 1)
    else:
        t = 0.0
    
    t = t + 0.4 * trend
    t = np.clip(t, -1, 1)
    proper_targets.append(t)

proper_targets = np.array(proper_targets)
target_decisions = proper_targets  # Use consistent targets

# ANFIS metrics
anfis_mse = mean_squared_error(target_decisions, anfis_decisions)
anfis_mae = mean_absolute_error(target_decisions, anfis_decisions)
anfis_rmse = np.sqrt(anfis_mse)
anfis_r2 = r2_score(target_decisions, anfis_decisions)

# Static threshold metrics
static_mse = mean_squared_error(target_decisions, static_decisions)
static_mae = mean_absolute_error(target_decisions, static_decisions)
static_rmse = np.sqrt(static_mse)
static_r2 = r2_score(target_decisions, static_decisions)

# Decision accuracy
def decision_accuracy(pred, target, threshold=0.2):
    pred_class = np.where(pred > threshold, 1, np.where(pred < -threshold, -1, 0))
    target_class = np.where(target > threshold, 1, np.where(target < -threshold, -1, 0))
    return np.mean(pred_class == target_class)

anfis_decision_acc = decision_accuracy(anfis_decisions, target_decisions)
static_decision_acc = decision_accuracy(static_decisions, target_decisions)

print(f"  LSTM Performance:")
print(f"    MSE: {lstm_mse:.6f}")
print(f"    MAE: {lstm_mae:.6f}")
print(f"    RMSE: {lstm_rmse:.6f}")
print(f"    R²: {lstm_r2:.4f}")

print(f"\n  ANFIS Performance:")
print(f"    MSE: {anfis_mse:.6f}")
print(f"    MAE: {anfis_mae:.6f}")
print(f"    RMSE: {anfis_rmse:.6f}")
print(f"    R²: {anfis_r2:.4f}")
print(f"    Decision Accuracy: {anfis_decision_acc:.2%}")

print(f"\n  Static Threshold Baseline:")
print(f"    MSE: {static_mse:.6f}")
print(f"    R²: {static_r2:.4f}")
print(f"    Decision Accuracy: {static_decision_acc:.2%}")

# =============================================================================
# SECTION 5: GENERATE PUBLICATION-QUALITY FIGURES
# =============================================================================
print("\n[5/6] Generating Publication Figures...")

# Set matplotlib style for publication
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# -------------------------------------------------------------------------
# FIGURE 1: 3D Control Surface
# -------------------------------------------------------------------------
print("  Generating Figure 4: Control Surface (3D)...")

# Create mesh grid for control surface
load_range = np.linspace(0, 1, 50)
trend_range = np.linspace(-1, 1, 50)
Load, Trend = np.meshgrid(load_range, trend_range)

# Flatten for prediction
grid_points = np.column_stack([Load.ravel(), Trend.ravel()]).astype(np.float32)

# Get ANFIS predictions for entire surface
surface_predictions = anfis_model.predict(grid_points, verbose=0)
Decision = surface_predictions.reshape(Load.shape)

# Create 3D plot
fig1 = plt.figure(figsize=(12, 9))
ax1 = fig1.add_subplot(111, projection='3d')

# Plot surface
surf = ax1.plot_surface(Load, Trend, Decision, cmap='viridis', 
                        edgecolor='none', alpha=0.9, antialiased=True)

# Add contour projections
ax1.contour(Load, Trend, Decision, zdir='z', offset=-1.2, cmap='viridis', alpha=0.5)

# Labels and title
ax1.set_xlabel('Predicted Workload', fontsize=14, labelpad=10)
ax1.set_ylabel('Workload Trend', fontsize=14, labelpad=10)
ax1.set_zlabel('Scaling Decision', fontsize=14, labelpad=10)
ax1.set_title('Figure 4: Learned Control Surface of the ANFIS Controller', 
              fontsize=16, fontweight='bold', pad=20)

# Set limits
ax1.set_xlim(0, 1)
ax1.set_ylim(-1, 1)
ax1.set_zlim(-1.2, 1.2)

# Add colorbar
cbar = fig1.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('Scaling Intensity', fontsize=12)

# Annotations
ax1.text2D(0.02, 0.98, "Scale Out (+1)", transform=ax1.transAxes, fontsize=10,
           verticalalignment='top', color='green', fontweight='bold')
ax1.text2D(0.02, 0.02, "Scale In (-1)", transform=ax1.transAxes, fontsize=10,
           verticalalignment='bottom', color='red', fontweight='bold')

# Adjust view angle
ax1.view_init(elev=25, azim=45)

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, 'figure4_control_surface_3d.png')
plt.savefig(fig1_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
print(f"    ✓ Saved: {fig1_path}")
plt.close()

# -------------------------------------------------------------------------
# FIGURE 2: Comparative Response Time Series
# -------------------------------------------------------------------------
print("  Generating Figure 5: Comparative Response...")

fig2, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

time_axis = np.arange(N_TEST_SAMPLES)

# Subplot 1: Workload (Actual vs Predicted)
ax_work = axes[0]
ax_work.plot(time_axis, actual_cpu, 'k-', linewidth=1.5, label='Actual CPU', alpha=0.8)
ax_work.plot(time_axis, predicted_cpu, 'b-', linewidth=1.5, label='Predicted CPU', alpha=0.8)
ax_work.fill_between(time_axis, actual_cpu, predicted_cpu, alpha=0.2, color='blue')
ax_work.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='High Threshold (0.8)')
ax_work.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Low Threshold (0.2)')
ax_work.set_ylabel('CPU Utilization', fontsize=12)
ax_work.set_title('(a) Workload Prediction: LSTM Performance', fontsize=14, fontweight='bold')
ax_work.legend(loc='upper right', ncol=2)
ax_work.set_ylim(-0.1, 1.1)

# Add RMSE annotation
ax_work.text(0.02, 0.95, f'RMSE: {lstm_rmse:.4f}\nR²: {lstm_r2:.4f}', 
             transform=ax_work.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Subplot 2: Scaling Decisions
ax_scale = axes[1]
ax_scale.plot(time_axis, static_decisions, 'r--', linewidth=2, 
              label='Static Threshold', alpha=0.7)
ax_scale.plot(time_axis, anfis_decisions, 'g-', linewidth=2, 
              label='Hybrid ANFIS', alpha=0.9)
ax_scale.fill_between(time_axis, anfis_decisions, static_decisions, 
                       where=anfis_decisions > static_decisions,
                       alpha=0.3, color='green', label='ANFIS Advantage')
ax_scale.fill_between(time_axis, anfis_decisions, static_decisions, 
                       where=anfis_decisions < static_decisions,
                       alpha=0.3, color='red')
ax_scale.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax_scale.axhline(y=0.2, color='gray', linestyle=':', alpha=0.3)
ax_scale.axhline(y=-0.2, color='gray', linestyle=':', alpha=0.3)
ax_scale.set_ylabel('Scaling Decision', fontsize=12)
ax_scale.set_title('(b) Scaling Decision Comparison: Static vs Hybrid ANFIS', 
                   fontsize=14, fontweight='bold')
ax_scale.legend(loc='upper right', ncol=2)
ax_scale.set_ylim(-1.2, 1.2)

# Add accuracy annotation
ax_scale.text(0.02, 0.95, 
              f'ANFIS Acc: {anfis_decision_acc:.1%}\nStatic Acc: {static_decision_acc:.1%}', 
              transform=ax_scale.transAxes, fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Subplot 3: Error Analysis
ax_error = axes[2]
anfis_error = anfis_decisions - target_decisions
static_error = static_decisions - target_decisions

ax_error.plot(time_axis, static_error, 'r--', linewidth=1.5, 
              label=f'Static Error (MAE: {np.mean(np.abs(static_error)):.4f})', alpha=0.7)
ax_error.plot(time_axis, anfis_error, 'g-', linewidth=1.5, 
              label=f'ANFIS Error (MAE: {np.mean(np.abs(anfis_error)):.4f})', alpha=0.9)
ax_error.fill_between(time_axis, anfis_error, 0, alpha=0.2, color='green')
ax_error.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax_error.set_xlabel('Time Step', fontsize=12)
ax_error.set_ylabel('Decision Error', fontsize=12)
ax_error.set_title('(c) Decision Error: Gap from Ideal Target', fontsize=14, fontweight='bold')
ax_error.legend(loc='upper right')
ax_error.set_ylim(-1.5, 1.5)

plt.suptitle('Figure 5: Comparative Response of Autoscaling Methods', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, 'figure5_comparative_response.png')
plt.savefig(fig2_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
print(f"    ✓ Saved: {fig2_path}")
plt.close()

# -------------------------------------------------------------------------
# FIGURE 3: Regression Analysis
# -------------------------------------------------------------------------
print("  Generating Figure 6: Regression Analysis...")

fig3, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: ANFIS Regression
ax_anfis = axes[0]
ax_anfis.scatter(target_decisions, anfis_decisions, alpha=0.5, s=20, 
                  c='green', edgecolors='none', label='ANFIS Output')
ax_anfis.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect Fit')

# Regression line
z = np.polyfit(target_decisions, anfis_decisions, 1)
p = np.poly1d(z)
x_line = np.linspace(-1, 1, 100)
ax_anfis.plot(x_line, p(x_line), 'b-', linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

ax_anfis.set_xlabel('Target (Ideal Decision)', fontsize=12)
ax_anfis.set_ylabel('Predicted (ANFIS Output)', fontsize=12)
ax_anfis.set_title('(a) Hybrid ANFIS Controller', fontsize=14, fontweight='bold')
ax_anfis.legend(loc='lower right')
ax_anfis.set_xlim(-1.1, 1.1)
ax_anfis.set_ylim(-1.1, 1.1)
ax_anfis.set_aspect('equal')

# R² annotation
ax_anfis.text(0.05, 0.95, f'$R^2$ = {anfis_r2:.4f}\nRMSE = {anfis_rmse:.4f}\nMAE = {anfis_mae:.4f}', 
              transform=ax_anfis.transAxes, fontsize=12, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Right: Static Threshold Regression
ax_static = axes[1]
ax_static.scatter(target_decisions, static_decisions, alpha=0.5, s=20, 
                   c='red', edgecolors='none', label='Static Output')
ax_static.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect Fit')

ax_static.set_xlabel('Target (Ideal Decision)', fontsize=12)
ax_static.set_ylabel('Predicted (Static Threshold)', fontsize=12)
ax_static.set_title('(b) Static Threshold Baseline', fontsize=14, fontweight='bold')
ax_static.legend(loc='lower right')
ax_static.set_xlim(-1.1, 1.1)
ax_static.set_ylim(-1.1, 1.1)
ax_static.set_aspect('equal')

# R² annotation
ax_static.text(0.05, 0.95, f'$R^2$ = {static_r2:.4f}\nRMSE = {static_rmse:.4f}\nMAE = {static_mae:.4f}', 
               transform=ax_static.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

plt.suptitle('Figure 6: Regression Analysis - Predicted vs Target Scaling Decisions', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, 'figure6_regression_analysis.png')
plt.savefig(fig3_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
print(f"    ✓ Saved: {fig3_path}")
plt.close()

# =============================================================================
# SECTION 6: SAVE METRICS TO FILE
# =============================================================================
print("\n[6/6] Saving Final Metrics...")

metrics_content = f"""
================================================================================
FINAL JOURNAL METRICS - HYBRID LSTM-ANFIS AUTOSCALING SYSTEM
================================================================================
Date: {np.datetime64('now')}
Test Samples: {N_TEST_SAMPLES}

================================================================================
LSTM WORKLOAD PREDICTION PERFORMANCE
================================================================================
  Mean Squared Error (MSE):     {lstm_mse:.6f}
  Mean Absolute Error (MAE):    {lstm_mae:.6f}
  Root Mean Squared Error:      {lstm_rmse:.6f}
  R² Score:                     {lstm_r2:.4f}

================================================================================
ANFIS SCALING DECISION PERFORMANCE
================================================================================
  Mean Squared Error (MSE):     {anfis_mse:.6f}
  Mean Absolute Error (MAE):    {anfis_mae:.6f}
  Root Mean Squared Error:      {anfis_rmse:.6f}
  R² Score:                     {anfis_r2:.4f}
  Decision Accuracy:            {anfis_decision_acc:.2%}

================================================================================
STATIC THRESHOLD BASELINE (Comparison)
================================================================================
  Mean Squared Error (MSE):     {static_mse:.6f}
  Mean Absolute Error (MAE):    {static_mae:.6f}
  Root Mean Squared Error:      {static_rmse:.6f}
  R² Score:                     {static_r2:.4f}
  Decision Accuracy:            {static_decision_acc:.2%}

================================================================================
PERFORMANCE IMPROVEMENT (ANFIS vs Static)
================================================================================
  MSE Reduction:                {((static_mse - anfis_mse) / static_mse * 100):.1f}%
  MAE Reduction:                {((static_mae - anfis_mae) / static_mae * 100):.1f}%
  R² Improvement:               {(anfis_r2 - static_r2):.4f}
  Decision Accuracy Gain:       {((anfis_decision_acc - static_decision_acc) * 100):.1f}%

================================================================================
GENERATED FIGURES
================================================================================
  Figure 4: {fig1_path}
  Figure 5: {fig2_path}
  Figure 6: {fig3_path}

================================================================================
"""

with open(METRICS_FILE, 'w') as f:
    f.write(metrics_content)

print(f"  ✓ Metrics saved to: {METRICS_FILE}")

# Print summary
print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
print(metrics_content)

print(f"""
Output Files:
  - {fig1_path}
  - {fig2_path}
  - {fig3_path}
  - {METRICS_FILE}

Ready for Thesis/Journal submission!
""")
