#!/usr/bin/env python3
"""
Real-time Hybrid Autoscaler (LSTM + ANFIS) - Simplified Version
==============================================================
Final implementation for Master Thesis

Uses subprocess for Docker commands instead of docker library
for maximum compatibility.

Author: AI Research Assistant
Thesis: Predictive LSTM and Neuro-Fuzzy Hybrid Autoscaling
"""

import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import requests

# Check and handle TensorFlow/ML libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, constraints
    from tensorflow.keras.models import Model, load_model
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available. Please install: pip install tensorflow")
    TF_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    print("⚠️  joblib not available. Please install: pip install joblib")
    JOBLIB_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-learn not available. Please install: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # File paths
    LSTM_MODEL_PATH = '/srv/cloud-computing/ai/best_lstm_model.keras'
    ANFIS_WEIGHTS_PATH = '/srv/cloud-computing/ai/best_anfis_journal.weights.h5'
    ANFIS_CONFIG_PATH = '/srv/cloud-computing/ai/best_anfis_journal_config.joblib'
    SCALER_PATH = '/srv/cloud-computing/ai/processed_data/scaler.joblib'
    
    # Prometheus configuration
    PROMETHEUS_URL = 'http://72.61.209.157:9090'
    CPU_QUERY = 'avg(rate(container_cpu_usage_seconds_total{container_label_com_docker_swarm_service_name="microservices_frontend-service"}[1m]))'
    
    # Service configuration
    TARGET_SERVICE = 'microservices_frontend-service'
    SWARM_STACK = 'microservices'
    
    # Autoscaler parameters
    LOOP_INTERVAL = 5  # seconds
    WINDOW_SIZE = 20   # sliding window for LSTM input
    COOLDOWN_PERIOD = 60  # seconds after scaling action
    MIN_REPLICAS = 1
    MAX_REPLICAS = 10
    
    # Scaling thresholds
    SCALE_OUT_THRESHOLD = 0.4  # ANFIS score > 0.4
    SCALE_IN_THRESHOLD = -0.4  # ANFIS score < -0.4
    
    # ANFIS configuration (must match training)
    N_MF_PER_INPUT = 5
    N_INPUTS = 2
    N_RULES = N_MF_PER_INPUT ** N_INPUTS
    L2_LAMBDA = 0.01
    PRUNE_THRESHOLD = 0.01

# =============================================================================
# ANFIS LAYER DEFINITIONS (ONLY IF TENSORFLOW AVAILABLE)
# =============================================================================

if TF_AVAILABLE:
    class MFOrderingConstraint(constraints.Constraint):
        """Constraint to ensure Membership Function centers are ordered."""
        
        def __call__(self, w):
            return tf.sort(w, axis=-1)
        
        def get_config(self):
            return {}

    class AdaptiveGaussianMFLayer(layers.Layer):
        """Layer 1: Fuzzification using Adaptive Gaussian Membership Functions."""
        
        def __init__(self, n_mf=5, min_sigma=0.05, **kwargs):
            super(AdaptiveGaussianMFLayer, self).__init__(**kwargs)
            self.n_mf = n_mf
            self.min_sigma = min_sigma
        
        def build(self, input_shape):
            n_inputs = input_shape[-1]
            
            # Initialize mu evenly across [0, 1] range with ordering constraint
            mu_init = np.zeros((n_inputs, self.n_mf))
            for i in range(n_inputs):
                mu_init[i] = np.linspace(0, 1, self.n_mf)
            
            # Initialize sigma based on MF spacing
            sigma_init = np.ones((n_inputs, self.n_mf)) * (1.0 / (self.n_mf - 1))
            
            # Mu with ordering constraint
            self.mu = self.add_weight(
                name='mu',
                shape=(n_inputs, self.n_mf),
                initializer=tf.constant_initializer(mu_init),
                trainable=True,
                constraint=MFOrderingConstraint()
            )
            
            # Sigma with minimum width constraint
            self.sigma = self.add_weight(
                name='sigma',
                shape=(n_inputs, self.n_mf),
                initializer=tf.constant_initializer(sigma_init),
                trainable=True,
                constraint=constraints.MinMaxNorm(min_value=self.min_sigma, max_value=1.0)
            )
            
            super().build(input_shape)
        
        def call(self, inputs):
            x = tf.expand_dims(inputs, -1)  # (batch, n_inputs, 1)
            
            # Gaussian MF with minimum sigma
            sigma_safe = tf.maximum(self.sigma, self.min_sigma)
            membership = tf.exp(-tf.square(x - self.mu) / (2 * tf.square(sigma_safe)))
            
            return membership
        
        def get_config(self):
            config = super().get_config()
            config.update({'n_mf': self.n_mf, 'min_sigma': self.min_sigma})
            return config

    class DynamicRuleFiringLayer(layers.Layer):
        """Layer 2 & 3: Rule Firing with Dynamic Importance Weighting."""
        
        def __init__(self, n_mf=5, n_inputs=2, prune_threshold=0.01, **kwargs):
            super(DynamicRuleFiringLayer, self).__init__(**kwargs)
            self.n_mf = n_mf
            self.n_inputs = n_inputs
            self.n_rules = n_mf ** n_inputs
            self.prune_threshold = prune_threshold
        
        def build(self, input_shape):
            # Learnable rule importance weights
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
            
            # Extract membership values for each input
            mf_input1 = membership_values[:, 0, :]  # (batch, n_mf)
            mf_input2 = membership_values[:, 1, :]  # (batch, n_mf)
            
            # T-Norm (Product) for firing strength
            mf1_expanded = tf.expand_dims(mf_input1, 2)
            mf2_expanded = tf.expand_dims(mf_input2, 1)
            firing_matrix = mf1_expanded * mf2_expanded
            firing_strengths = tf.reshape(firing_matrix, (batch_size, self.n_rules))
            
            # Apply dynamic rule importance
            weighted_firing = firing_strengths * self.rule_importance
            
            # Soft pruning during inference
            if not training:
                prune_mask = tf.cast(self.rule_importance > self.prune_threshold, tf.float32)
                weighted_firing = weighted_firing * prune_mask
            
            # Normalize
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
        """Layer 4 & 5: Takagi-Sugeno Consequent for REGRESSION output."""
        
        def __init__(self, n_rules=25, l2_lambda=0.01, **kwargs):
            super(TakagiSugenoRegressionLayer, self).__init__(**kwargs)
            self.n_rules = n_rules
            self.l2_lambda = l2_lambda
        
        def build(self, input_shape):
            # Consequent: f_i = p_i * x + q_i * y + r_i (single output)
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
            
            x = raw_inputs[:, 0:1]  # Predicted_Workload (batch, 1)
            y = raw_inputs[:, 1:2]  # Workload_Trend (batch, 1)
            
            # Compute consequent: f_i = p_i*x + q_i*y + r_i
            p_t = tf.squeeze(self.p, axis=-1)  # (n_rules,)
            q_t = tf.squeeze(self.q, axis=-1)  # (n_rules,)
            r_t = tf.squeeze(self.r, axis=-1)  # (n_rules,)
            
            f = x * p_t + y * q_t + r_t  # (batch, n_rules)
            
            # Defuzzification: weighted sum
            defuzzified = tf.reduce_sum(normalized_weights * f, axis=1, keepdims=True)  # (batch, 1)
            
            # Bound output to [-1, 1] using tanh
            bounded_output = tf.tanh(defuzzified)
            
            return bounded_output
        
        def get_config(self):
            config = super().get_config()
            config.update({'n_rules': self.n_rules, 'l2_lambda': self.l2_lambda})
            return config

    class ANFISRegressionModel(Model):
        """Complete ANFIS Model for Regression-based Autoscaling."""
        
        def __init__(self, n_mf=5, n_inputs=2, l2_lambda=0.01, 
                     prune_threshold=0.01, **kwargs):
            super(ANFISRegressionModel, self).__init__(**kwargs)
            self.n_mf = n_mf
            self.n_inputs = n_inputs
            self.n_rules = n_mf ** n_inputs
            self.l2_lambda = l2_lambda
            self.prune_threshold = prune_threshold
            
            # Initialize layers
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

# =============================================================================
# REAL-TIME AUTOSCALER CLASS
# =============================================================================

class RealTimeHybridAutoscaler:
    """Real-time Hybrid Autoscaler combining LSTM prediction and ANFIS decision-making."""
    
    def __init__(self):
        self.config = Config()
        self.logger = self._setup_logging()
        
        # Check dependencies
        if not self._check_dependencies():
            sys.exit(1)
        
        # State management
        self.sliding_window = deque(maxlen=self.config.WINDOW_SIZE)
        self.last_scaling_time = 0
        self.last_prediction = 0.0
        self.iteration_count = 0
        
        # Load models and scaler
        self._load_models()
        
        self.logger.info("Real-time Hybrid Autoscaler initialized successfully!")
        self.logger.info(f"Target Service: {self.config.TARGET_SERVICE}")
        self.logger.info(f"Prometheus URL: {self.config.PROMETHEUS_URL}")
        self.logger.info(f"Loop Interval: {self.config.LOOP_INTERVAL}s")
    
    def _check_dependencies(self):
        """Check if all required dependencies are available."""
        missing = []
        if not TF_AVAILABLE:
            missing.append("tensorflow")
        if not JOBLIB_AVAILABLE:
            missing.append("joblib")
        if not SKLEARN_AVAILABLE:
            missing.append("scikit-learn")
        
        if missing:
            self.logger.error(f"❌ Missing dependencies: {', '.join(missing)}")
            self.logger.error("Please install with: pip install " + " ".join(missing))
            return False
        return True
        
    def _setup_logging(self):
        """Setup beautiful console logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger(__name__)
    
    def _load_models(self):
        """Load LSTM, ANFIS models and scaler."""
        try:
            # Load scaler
            self.logger.info("Loading scaler...")
            self.scaler = joblib.load(self.config.SCALER_PATH)
            
            # Load LSTM model
            self.logger.info("Loading LSTM model...")
            self.lstm_model = load_model(self.config.LSTM_MODEL_PATH)
            
            # Load ANFIS model
            self.logger.info("Loading ANFIS model...")
            
            # First, create ANFIS model with same config as training
            anfis_config = joblib.load(self.config.ANFIS_CONFIG_PATH)
            
            self.anfis_model = ANFISRegressionModel(
                n_mf=self.config.N_MF_PER_INPUT,
                n_inputs=self.config.N_INPUTS,
                l2_lambda=self.config.L2_LAMBDA,
                prune_threshold=self.config.PRUNE_THRESHOLD
            )
            
            # Build model with dummy input
            dummy_input = tf.zeros((1, self.config.N_INPUTS))
            _ = self.anfis_model(dummy_input, training=False)
            
            # Load weights
            self.anfis_model.load_weights(self.config.ANFIS_WEIGHTS_PATH)
            
            self.logger.info("✅ All models loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading models: {str(e)}")
            raise
    
    def _query_prometheus(self):
        """Query Prometheus for CPU metrics."""
        try:
            url = f"{self.config.PROMETHEUS_URL}/api/v1/query"
            params = {'query': self.config.CPU_QUERY}
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                # Extract value
                cpu_usage = float(data['data']['result'][0]['value'][1])
                return cpu_usage
            else:
                # No data available (service idle)
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"⚠️  Prometheus query failed: {str(e)}, assuming 0.0")
            return 0.0
    
    def _get_current_replicas(self):
        """Get current replica count for target service using Docker CLI."""
        try:
            # Use docker service ls to get service info
            result = subprocess.run(
                ['docker', 'service', 'ls', '--format', 'json'],
                capture_output=True, text=True, check=True
            )
            
            services = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    services.append(json.loads(line))
            
            for service in services:
                if self.config.TARGET_SERVICE in service.get('Name', ''):
                    # Parse replicas (format: "2/2" or "1/1")
                    replicas_str = service.get('Replicas', '1/1')
                    current_replicas = int(replicas_str.split('/')[1])
                    return current_replicas
            
            self.logger.warning(f"⚠️  Service {self.config.TARGET_SERVICE} not found, assuming 1 replica")
            return 1
            
        except Exception as e:
            self.logger.error(f"❌ Error getting current replicas: {str(e)}")
            return 1
    
    def _scale_service(self, new_replicas):
        """Scale service to new replica count using Docker CLI."""
        try:
            # Use docker service scale command
            result = subprocess.run(
                ['docker', 'service', 'scale', f'{self.config.TARGET_SERVICE}={new_replicas}'],
                capture_output=True, text=True, check=True
            )
            
            self.last_scaling_time = time.time()
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Docker scaling failed: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error scaling service: {str(e)}")
            return False
    
    def _format_log_entry(self, cpu_raw, cpu_norm, lstm_pred, trend, anfis_score, 
                         current_replicas, action="MAINTAIN"):
        """Format beautiful log entry for live demonstration."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        cpu_pct = cpu_raw * 100
        
        # Status icons
        if action.startswith("SCALE OUT"):
            icon = "📈"
            color_action = f"🔺 {action}"
        elif action.startswith("SCALE IN"):
            icon = "📉"
            color_action = f"🔻 {action}"
        elif action == "COOLDOWN":
            icon = "⏳"
            color_action = f"⏳ {action}"
        else:
            icon = "🟢"
            color_action = f"⚪ {action}"
        
        # Format scores
        lstm_str = f"{lstm_pred:+.3f}"
        trend_str = f"{trend:+.3f}" 
        anfis_str = f"{anfis_score:+.3f}"
        
        log_msg = (f"{icon} [{timestamp}] "
                  f"CPU: {cpu_pct:5.1f}% | "
                  f"LSTM: {lstm_str:>7s} | "
                  f"Trend: {trend_str:>7s} | "
                  f"ANFIS: {anfis_str:>7s} | "
                  f"Replicas: {current_replicas:2d} | "
                  f"{color_action}")
        
        self.logger.info(log_msg)
    
    def _make_scaling_decision(self, anfis_score, current_replicas):
        """Make scaling decision based on ANFIS score and constraints."""
        
        # Check cooldown period
        time_since_last_scaling = time.time() - self.last_scaling_time
        if time_since_last_scaling < self.config.COOLDOWN_PERIOD:
            return current_replicas, "COOLDOWN"
        
        # Scaling logic
        if anfis_score > self.config.SCALE_OUT_THRESHOLD and current_replicas < self.config.MAX_REPLICAS:
            new_replicas = min(current_replicas + 1, self.config.MAX_REPLICAS)
            return new_replicas, "SCALE OUT"
        
        elif anfis_score < self.config.SCALE_IN_THRESHOLD and current_replicas > self.config.MIN_REPLICAS:
            new_replicas = max(current_replicas - 1, self.config.MIN_REPLICAS)
            return new_replicas, "SCALE IN"
        
        else:
            return current_replicas, "MAINTAIN"
    
    def run_autoscaling_loop(self):
        """Main autoscaling loop."""
        
        print("\n" + "="*80)
        print("🚀 REAL-TIME HYBRID AUTOSCALER STARTED")
        print("   LSTM Prediction + ANFIS Decision Making")
        print("="*80)
        print(f"📊 Target: {self.config.TARGET_SERVICE}")
        print(f"⏱️  Interval: {self.config.LOOP_INTERVAL}s | Window: {self.config.WINDOW_SIZE}")
        print(f"📈 Scale Out: ANFIS > {self.config.SCALE_OUT_THRESHOLD}")
        print(f"📉 Scale In: ANFIS < {self.config.SCALE_IN_THRESHOLD}")
        print(f"⏳ Cooldown: {self.config.COOLDOWN_PERIOD}s")
        print("="*80)
        
        try:
            while True:
                self.iteration_count += 1
                
                # 1. Query Prometheus for CPU data
                cpu_raw = self._query_prometheus()
                
                # 2. Normalize CPU data using loaded scaler
                cpu_normalized = self.scaler.transform([[cpu_raw]])[0][0]
                
                # 3. Update sliding window
                self.sliding_window.append(cpu_normalized)
                
                # 4. Check if we have enough data for prediction
                if len(self.sliding_window) < self.config.WINDOW_SIZE:
                    remaining = self.config.WINDOW_SIZE - len(self.sliding_window)
                    self.logger.info(f"🔄 Initializing buffer... {len(self.sliding_window)}/{self.config.WINDOW_SIZE} "
                                   f"(need {remaining} more)")
                    time.sleep(self.config.LOOP_INTERVAL)
                    continue
                
                # 5. LSTM Prediction
                lstm_input = np.array(self.sliding_window).reshape(1, self.config.WINDOW_SIZE, 1)
                lstm_prediction = self.lstm_model.predict(lstm_input, verbose=0)[0][0]
                
                # 6. Calculate trend
                trend = lstm_prediction - self.last_prediction
                self.last_prediction = lstm_prediction
                
                # 7. ANFIS Decision
                anfis_input = np.array([[lstm_prediction, trend]], dtype=np.float32)
                anfis_score = self.anfis_model(anfis_input, training=False).numpy()[0][0]
                
                # 8. Get current replicas
                current_replicas = self._get_current_replicas()
                
                # 9. Make scaling decision
                new_replicas, action = self._make_scaling_decision(anfis_score, current_replicas)
                
                # 10. Execute scaling if needed
                if action in ["SCALE OUT", "SCALE IN"]:
                    if self._scale_service(new_replicas):
                        old_replicas = current_replicas
                        current_replicas = new_replicas
                        action_detail = f"{action} ({old_replicas}->{current_replicas})"
                    else:
                        action_detail = f"{action} FAILED"
                else:
                    action_detail = action
                
                # 11. Log beautiful entry for live demo
                self._format_log_entry(
                    cpu_raw, cpu_normalized, lstm_prediction, trend, anfis_score,
                    current_replicas, action_detail
                )
                
                # 12. Wait for next iteration
                time.sleep(self.config.LOOP_INTERVAL)
                
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Autoscaler stopped by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"\n❌ Fatal error in autoscaling loop: {str(e)}")
            raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point."""
    
    print("🎓 MASTER THESIS: Real-time Hybrid Autoscaler")
    print("   LSTM + ANFIS for Docker Swarm Autoscaling")
    print(f"   Working Directory: {os.getcwd()}")
    
    # Verify required files exist
    config = Config()
    required_files = [
        config.LSTM_MODEL_PATH,
        config.ANFIS_WEIGHTS_PATH,
        config.ANFIS_CONFIG_PATH,
        config.SCALER_PATH
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ ERROR: Required files not found:")
        for f in missing_files:
            print(f"   - {f}")
        sys.exit(1)
    
    print("✅ All required model files found!")
    
    # Initialize and run autoscaler
    try:
        autoscaler = RealTimeHybridAutoscaler()
        autoscaler.run_autoscaling_loop()
    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
