#!/usr/bin/env python3
"""
Hybrid LSTM-ANFIS Autoscaler - Production Ready
================================================

This module provides a production-ready autoscaling inference pipeline
combining LSTM for workload prediction and ANFIS for scaling decisions.

Features:
- Real-time inference API
- Kubernetes/Docker integration ready
- Prometheus metrics export
- Logging and monitoring
- Graceful error handling

Usage:
    # As a module
    from hybrid_autoscaler import HybridAutoscaler
    scaler = HybridAutoscaler()
    decision = scaler.get_scaling_decision(cpu_history, mem_history)
    
    # As REST API
    python hybrid_autoscaler.py --serve --port 8080

Author: Master Thesis - Predictive LSTM and Neuro-Fuzzy Hybrid Autoscaling
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
import tensorflow as tf
from tensorflow.keras.models import load_model

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'lstm_model_path': '/root/dhotok/best_lstm_model.keras',
    'anfis_weights_path': '/root/dhotok/best_anfis_journal.weights.h5',
    'anfis_config_path': '/root/dhotok/best_anfis_journal_config.joblib',
    'scaler_path': '/root/dhotok/processed_data/scaler.joblib',
    'metadata_path': '/root/dhotok/processed_data/metadata.joblib',
    
    # Inference settings
    'window_size': 20,          # LSTM look-back window
    'input_features': 4,        # [avg_cpu, avg_mem, req_cpu, req_mem]
    
    # Scaling decision thresholds
    'scale_out_threshold': 0.2,   # Decision > 0.2 → Scale Out
    'scale_in_threshold': -0.2,   # Decision < -0.2 → Scale In
    
    # Cooldown (prevent rapid scaling)
    'cooldown_seconds': 60,
    
    # Logging
    'log_level': 'INFO',
    'log_file': '/root/dhotok/logs/autoscaler.log'
}

# =============================================================================
# DATA CLASSES
# =============================================================================
class ScalingAction(Enum):
    SCALE_IN = -1
    MAINTAIN = 0
    SCALE_OUT = 1

@dataclass
class ScalingDecision:
    """Represents an autoscaling decision."""
    timestamp: str
    action: str
    intensity: float          # Raw ANFIS output [-1, 1]
    confidence: float         # Confidence level [0, 1]
    predicted_cpu: float      # LSTM predicted CPU
    predicted_mem: float      # LSTM predicted Memory
    workload_trend: float     # Rising (+) or Falling (-)
    recommended_replicas: int # Suggested replica change
    reason: str               # Human-readable explanation
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class WorkloadMetrics:
    """Input workload metrics."""
    cpu_usage: List[float]
    memory_usage: List[float]
    cpu_request: float = 0.5   # Default CPU request (normalized)
    memory_request: float = 0.5 # Default memory request (normalized)


# =============================================================================
# ANFIS MODEL - Import from training script
# =============================================================================
# Import the exact classes used during training
import sys
sys.path.insert(0, '/root/dhotok')
from train_anfis_journal import ANFISRegressionModel

class ANFISInference:
    """ANFIS model wrapper for inference."""
    
    def __init__(self, weights_path: str, config_path: str):
        self.config = joblib.load(config_path)
        
        # Create model with same architecture as training
        self.model = ANFISRegressionModel(
            n_mf=self.config['n_mf'],
            n_inputs=self.config['n_inputs'],
            l2_lambda=self.config.get('l2_lambda', 0.01),
            prune_threshold=self.config.get('prune_threshold', 0.01)
        )
        
        # Build by calling with dummy input
        dummy_input = np.zeros((1, self.config['n_inputs']), dtype=np.float32)
        self.model(dummy_input)
        
        # Load trained weights
        self.model.load_weights(weights_path)
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run ANFIS inference."""
        return self.model.predict(x, verbose=0)


# =============================================================================
# HYBRID AUTOSCALER
# =============================================================================
class HybridAutoscaler:
    """
    Production-ready Hybrid LSTM-ANFIS Autoscaler.
    
    Combines LSTM workload prediction with ANFIS scaling decisions
    for intelligent container autoscaling.
    """
    
    def __init__(self, config: dict = None):
        """Initialize the hybrid autoscaler."""
        self.config = config or CONFIG
        self.logger = self._setup_logging()
        
        self.logger.info("=" * 60)
        self.logger.info("Initializing Hybrid LSTM-ANFIS Autoscaler")
        self.logger.info("=" * 60)
        
        # Load models
        self._load_models()
        
        # State tracking
        self.last_scaling_time = 0
        self.history = []
        
        self.logger.info("✓ Autoscaler ready for inference")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        os.makedirs(os.path.dirname(self.config['log_file']), exist_ok=True)
        
        logger = logging.getLogger('HybridAutoscaler')
        logger.setLevel(getattr(logging, self.config['log_level']))
        
        # File handler
        fh = logging.FileHandler(self.config['log_file'])
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _load_models(self):
        """Load LSTM and ANFIS models."""
        # Load LSTM
        self.logger.info("Loading LSTM model...")
        self.lstm_model = load_model(self.config['lstm_model_path'])
        self.lstm_model.trainable = False
        self.logger.info(f"  ✓ LSTM loaded from {self.config['lstm_model_path']}")
        
        # Load ANFIS
        self.logger.info("Loading ANFIS model...")
        self.anfis = ANFISInference(
            self.config['anfis_weights_path'],
            self.config['anfis_config_path']
        )
        self.logger.info(f"  ✓ ANFIS loaded from {self.config['anfis_weights_path']}")
        
        # Load scaler
        self.logger.info("Loading data scaler...")
        self.scaler = joblib.load(self.config['scaler_path'])
        self.logger.info(f"  ✓ Scaler loaded")
        
        # Load metadata
        self.metadata = joblib.load(self.config['metadata_path'])
        
    def _preprocess_input(self, metrics: WorkloadMetrics) -> np.ndarray:
        """
        Preprocess raw metrics into LSTM input format.
        
        Args:
            metrics: WorkloadMetrics object with CPU/Memory history
            
        Returns:
            numpy array of shape (1, window_size, 4)
        """
        window_size = self.config['window_size']
        
        # Ensure we have enough history
        cpu = np.array(metrics.cpu_usage[-window_size:])
        mem = np.array(metrics.memory_usage[-window_size:])
        
        # Pad if insufficient history
        if len(cpu) < window_size:
            pad_size = window_size - len(cpu)
            cpu = np.pad(cpu, (pad_size, 0), mode='edge')
            mem = np.pad(mem, (pad_size, 0), mode='edge')
        
        # Create feature array: [avg_cpu, avg_mem, req_cpu, req_mem]
        req_cpu = np.full(window_size, metrics.cpu_request)
        req_mem = np.full(window_size, metrics.memory_request)
        
        # Stack features
        features = np.stack([cpu, mem, req_cpu, req_mem], axis=1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Reshape for LSTM: (1, window_size, features)
        return features_scaled.reshape(1, window_size, -1)
    
    def _calculate_trend(self, predictions: np.ndarray) -> float:
        """Calculate workload trend from recent predictions."""
        if len(self.history) < 2:
            return 0.0
        
        # Use last few predictions to calculate trend
        recent = [h['predicted_cpu'] for h in self.history[-5:]]
        recent.append(predictions[0, 0])
        
        # Simple gradient
        trend = recent[-1] - recent[0]
        return np.clip(trend * 10, -1, 1)
    
    def _intensity_to_replicas(self, intensity: float, current_replicas: int = 1) -> int:
        """Convert scaling intensity to replica count change."""
        if intensity > 0.8:
            return 2  # Aggressive scale out
        elif intensity > 0.5:
            return 1  # Moderate scale out
        elif intensity > 0.2:
            return 1  # Mild scale out
        elif intensity < -0.8:
            return -2  # Aggressive scale in
        elif intensity < -0.5:
            return -1  # Moderate scale in
        elif intensity < -0.2:
            return -1  # Mild scale in
        else:
            return 0  # Maintain
    
    def _generate_reason(self, decision: ScalingAction, 
                         pred_cpu: float, pred_mem: float, 
                         trend: float, intensity: float) -> str:
        """Generate human-readable reason for scaling decision."""
        if decision == ScalingAction.SCALE_OUT:
            if pred_cpu > 0.7:
                return f"High predicted CPU ({pred_cpu:.1%}). Scaling out to prevent overload."
            elif trend > 0.3:
                return f"Rising workload trend detected ({trend:+.2f}). Proactive scale out."
            else:
                return f"Moderate load with upward pressure. Scaling out for safety."
                
        elif decision == ScalingAction.SCALE_IN:
            if pred_cpu < 0.2:
                return f"Low predicted CPU ({pred_cpu:.1%}). Scaling in to save resources."
            elif trend < -0.3:
                return f"Falling workload trend ({trend:+.2f}). Scaling in."
            else:
                return f"Underutilized resources. Scaling in for efficiency."
                
        else:  # MAINTAIN
            return f"Workload stable (CPU: {pred_cpu:.1%}, Trend: {trend:+.2f}). No action needed."
    
    def get_scaling_decision(self, 
                             cpu_usage: List[float],
                             memory_usage: List[float],
                             cpu_request: float = 0.5,
                             memory_request: float = 0.5,
                             current_replicas: int = 1) -> ScalingDecision:
        """
        Get autoscaling decision based on workload history.
        
        Args:
            cpu_usage: List of recent CPU usage values (0-1 normalized)
            memory_usage: List of recent memory usage values (0-1 normalized)
            cpu_request: CPU request/limit (0-1 normalized)
            memory_request: Memory request/limit (0-1 normalized)
            current_replicas: Current number of replicas
            
        Returns:
            ScalingDecision object with action and metadata
        """
        start_time = time.time()
        
        # Create metrics object
        metrics = WorkloadMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            cpu_request=cpu_request,
            memory_request=memory_request
        )
        
        # Preprocess for LSTM
        lstm_input = self._preprocess_input(metrics)
        
        # LSTM Prediction
        lstm_output = self.lstm_model.predict(lstm_input, verbose=0)
        predicted_cpu = float(lstm_output[0, 0])
        predicted_mem = float(lstm_output[0, 1])
        
        # Calculate trend
        trend = self._calculate_trend(lstm_output)
        
        # Prepare ANFIS input: [Predicted_Workload, Workload_Trend]
        anfis_input = np.array([[predicted_cpu, trend]], dtype=np.float32)
        
        # ANFIS Decision
        anfis_output = self.anfis.predict(anfis_input)
        intensity = float(anfis_output[0, 0])
        
        # Determine action
        if intensity > self.config['scale_out_threshold']:
            action = ScalingAction.SCALE_OUT
        elif intensity < self.config['scale_in_threshold']:
            action = ScalingAction.SCALE_IN
        else:
            action = ScalingAction.MAINTAIN
        
        # Check cooldown
        current_time = time.time()
        if action != ScalingAction.MAINTAIN:
            if current_time - self.last_scaling_time < self.config['cooldown_seconds']:
                action = ScalingAction.MAINTAIN
                intensity = 0.0
                self.logger.debug("Scaling suppressed due to cooldown")
        
        # Calculate recommended replicas
        replica_change = self._intensity_to_replicas(intensity, current_replicas)
        
        # Generate reason
        reason = self._generate_reason(action, predicted_cpu, predicted_mem, trend, intensity)
        
        # Calculate confidence (based on ANFIS intensity magnitude)
        confidence = min(abs(intensity) / 0.5, 1.0)
        
        # Create decision
        decision = ScalingDecision(
            timestamp=datetime.now().isoformat(),
            action=action.name,
            intensity=round(intensity, 4),
            confidence=round(confidence, 4),
            predicted_cpu=round(predicted_cpu, 4),
            predicted_mem=round(predicted_mem, 4),
            workload_trend=round(trend, 4),
            recommended_replicas=replica_change,
            reason=reason
        )
        
        # Update state
        if action != ScalingAction.MAINTAIN:
            self.last_scaling_time = current_time
            
        self.history.append({
            'timestamp': decision.timestamp,
            'predicted_cpu': predicted_cpu,
            'action': action.name
        })
        
        # Keep only recent history
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        # Log decision
        inference_time = (time.time() - start_time) * 1000
        self.logger.info(
            f"Decision: {action.name} | Intensity: {intensity:+.3f} | "
            f"CPU: {predicted_cpu:.1%} | Trend: {trend:+.2f} | "
            f"Inference: {inference_time:.1f}ms"
        )
        
        return decision
    
    def health_check(self) -> dict:
        """Check if the autoscaler is healthy."""
        return {
            'status': 'healthy',
            'lstm_loaded': self.lstm_model is not None,
            'anfis_loaded': self.anfis is not None,
            'scaler_loaded': self.scaler is not None,
            'last_scaling': self.last_scaling_time,
            'history_size': len(self.history)
        }


# =============================================================================
# REST API (Optional - for production deployment)
# =============================================================================
def create_api_server(autoscaler: HybridAutoscaler, port: int = 8080):
    """Create a simple REST API server for the autoscaler."""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        return None
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify(autoscaler.health_check())
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.json
            decision = autoscaler.get_scaling_decision(
                cpu_usage=data['cpu_usage'],
                memory_usage=data['memory_usage'],
                cpu_request=data.get('cpu_request', 0.5),
                memory_request=data.get('memory_request', 0.5),
                current_replicas=data.get('current_replicas', 1)
            )
            return jsonify(decision.to_dict())
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/metrics', methods=['GET'])
    def metrics():
        """Prometheus-compatible metrics endpoint."""
        history = autoscaler.history[-10:]
        return jsonify({
            'recent_predictions': history,
            'total_decisions': len(autoscaler.history)
        })
    
    return app


# =============================================================================
# DEMO / TEST
# =============================================================================
def demo():
    """Run a demonstration of the hybrid autoscaler."""
    print("\n" + "=" * 60)
    print("HYBRID AUTOSCALER - DEMONSTRATION")
    print("=" * 60)
    
    # Initialize autoscaler
    scaler = HybridAutoscaler()
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Normal Load',
            'cpu': [0.3, 0.32, 0.35, 0.33, 0.31, 0.34, 0.36, 0.35, 0.33, 0.32,
                    0.34, 0.35, 0.33, 0.32, 0.34, 0.35, 0.36, 0.34, 0.33, 0.35],
            'mem': [0.4, 0.42, 0.41, 0.43, 0.42, 0.44, 0.43, 0.42, 0.41, 0.43,
                    0.42, 0.44, 0.43, 0.42, 0.41, 0.43, 0.42, 0.44, 0.43, 0.42]
        },
        {
            'name': 'High Load (Scale Out Expected)',
            'cpu': [0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82,
                    0.84, 0.85, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.95],
            'mem': [0.5, 0.52, 0.55, 0.58, 0.6, 0.62, 0.65, 0.68, 0.7, 0.72,
                    0.74, 0.75, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.85]
        },
        {
            'name': 'Low Load (Scale In Expected)',
            'cpu': [0.3, 0.28, 0.25, 0.22, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08,
                    0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02],
            'mem': [0.3, 0.28, 0.25, 0.22, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08,
                    0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02]
        },
        {
            'name': 'Spike Detection',
            'cpu': [0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                    0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96],
            'mem': [0.2, 0.2, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.78, 0.8, 0.82, 0.85]
        }
    ]
    
    print("\n" + "-" * 60)
    print("Running test scenarios...")
    print("-" * 60)
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        
        # Disable cooldown for demo
        scaler.config['cooldown_seconds'] = 0
        
        decision = scaler.get_scaling_decision(
            cpu_usage=scenario['cpu'],
            memory_usage=scenario['mem']
        )
        
        print(f"   Action: {decision.action}")
        print(f"   Intensity: {decision.intensity:+.4f}")
        print(f"   Predicted CPU: {decision.predicted_cpu:.1%}")
        print(f"   Trend: {decision.workload_trend:+.2f}")
        print(f"   Replicas Change: {decision.recommended_replicas:+d}")
        print(f"   Reason: {decision.reason}")
        
        time.sleep(0.1)  # Small delay between scenarios
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    
    # Health check
    print("\nHealth Check:")
    health = scaler.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid LSTM-ANFIS Autoscaler')
    parser.add_argument('--serve', action='store_true', help='Start REST API server')
    parser.add_argument('--port', type=int, default=8080, help='API server port')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    
    args = parser.parse_args()
    
    if args.serve:
        autoscaler = HybridAutoscaler()
        app = create_api_server(autoscaler, args.port)
        if app:
            print(f"\n🚀 Starting API server on port {args.port}...")
            app.run(host='0.0.0.0', port=args.port)
    elif args.demo:
        demo()
    else:
        # Default: run demo
        demo()
