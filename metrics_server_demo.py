#!/usr/bin/env python3
"""
Simple AI Metrics Server for Thesis Defense Dashboard Demo
Simulates LSTM + ANFIS metrics without requiring TensorFlow
"""

import time
import random
import math
import json
import logging
from prometheus_client import start_http_server, Gauge, Counter
import threading

# AI Metrics
ai_lstm_prediction_cpu = Gauge('ai_lstm_prediction_cpu', 'CPU predicted by LSTM', ['service'])
ai_anfis_score = Gauge('ai_anfis_score', 'ANFIS fuzzy decision score', ['service']) 
ai_scaling_decision = Gauge('ai_scaling_decision', '1=Scale Out, -1=Scale In, 0=Maintain', ['service'])
ai_current_replicas = Gauge('ai_current_replicas', 'Current replicas', ['service'])
ai_cpu_raw = Gauge('ai_cpu_raw', 'Raw CPU from Prometheus', ['service'])
ai_cpu_normalized = Gauge('ai_cpu_normalized', 'Normalized CPU', ['service'])
ai_workload_trend = Gauge('ai_workload_trend', 'Workload trend', ['service'])
ai_iteration_counter = Counter('ai_iteration_total', 'AI iterations')

def simulate_ai_autoscaler():
    """Simulate realistic AI autoscaler behavior for demo."""
    service_label = "microservices_frontend-service"
    
    # Simulation state
    base_cpu = 0.3
    trend = 0.0
    replicas = 2
    time_counter = 0
    
    print("🎓 THESIS DEFENSE - AI Metrics Demo Started")
    print(f"📊 Metrics Server: http://localhost:8000/metrics")
    print("=" * 60)
    
    while True:
        time_counter += 1
        
        # Simulate realistic CPU patterns
        # Create sine wave with noise for realistic workload simulation
        cpu_raw = base_cpu + 0.2 * math.sin(time_counter * 0.1) + random.uniform(-0.05, 0.05)
        cpu_raw = max(0.1, min(0.9, cpu_raw))  # Keep in bounds
        
        # Normalize CPU (simulate scaler)
        cpu_normalized = (cpu_raw - 0.1) / 0.8
        
        # Simulate LSTM prediction (slightly ahead of actual)
        lstm_prediction = cpu_normalized + 0.1 * math.sin(time_counter * 0.08) + random.uniform(-0.02, 0.02)
        lstm_prediction = max(0.0, min(1.0, lstm_prediction))
        
        # Calculate trend
        if time_counter > 1:
            trend = (lstm_prediction - prev_prediction) * 5  # Amplify for visibility
        trend = max(-1.0, min(1.0, trend))
        
        # Simulate ANFIS decision
        # Higher prediction + positive trend = scale out
        # Lower prediction + negative trend = scale in
        anfis_input = lstm_prediction + 0.3 * trend
        
        if anfis_input > 0.7:
            anfis_score = 0.6 + 0.3 * random.random()  # Scale out
            decision = 1
            if replicas < 5 and random.random() > 0.7:  # Sometimes scale
                replicas += 1
        elif anfis_input < 0.3:
            anfis_score = -0.6 - 0.3 * random.random()  # Scale in
            decision = -1
            if replicas > 1 and random.random() > 0.7:  # Sometimes scale
                replicas -= 1
        else:
            anfis_score = -0.2 + 0.4 * random.random()  # Maintain
            decision = 0
        
        # Update metrics
        ai_cpu_raw.labels(service=service_label).set(cpu_raw)
        ai_cpu_normalized.labels(service=service_label).set(cpu_normalized)
        ai_lstm_prediction_cpu.labels(service=service_label).set(lstm_prediction)
        ai_workload_trend.labels(service=service_label).set(trend)
        ai_anfis_score.labels(service=service_label).set(anfis_score)
        ai_current_replicas.labels(service=service_label).set(replicas)
        ai_scaling_decision.labels(service=service_label).set(decision)
        ai_iteration_counter.inc()
        
        # Console output (formatted)
        decision_text = "SCALE OUT" if decision == 1 else "SCALE IN" if decision == -1 else "MAINTAIN"
        
        # Structured JSON logging for Loki
        log_data = {
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "service": "ai-autoscaler",
            "level": "INFO",
            "message": "AI Decision Made",
            "data": {
                "cpu_raw": round(cpu_raw, 4),
                "cpu_normalized": round(cpu_normalized, 4),
                "lstm_prediction": round(lstm_prediction, 4),
                "workload_trend": round(trend, 4),
                "anfis_score": round(anfis_score, 4),
                "scaling_decision": decision_text,
                "current_replicas": replicas,
                "confidence": round(abs(anfis_score), 3)
            }
        }
        print(f"🤖 [{time.strftime('%H:%M:%S')}] "
              f"CPU: {cpu_raw*100:4.1f}% | "
              f"LSTM: {lstm_prediction:+.3f} | "
              f"ANFIS: {anfis_score:+.3f} | "
              f"Replicas: {replicas} | "
              f"{decision_text}")
        
        # JSON log for Loki (structured)
        print(json.dumps(log_data, separators=(',', ':')))
        
        prev_prediction = lstm_prediction
        time.sleep(3)  # Update every 3 seconds

def main():
    # Start metrics server
    start_http_server(8000)
    print("🎯 Prometheus metrics server started on port 8000")
    
    # Start AI simulation
    simulate_ai_autoscaler()

if __name__ == "__main__":
    main()
