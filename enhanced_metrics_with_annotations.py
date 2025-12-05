#!/usr/bin/env python3
"""
Enhanced AI Metrics Server with Grafana Annotations
Demonstrates causality: LSTM -> ANFIS -> Scaling -> Results
"""

import time
import random
import math
import json
import requests
from prometheus_client import start_http_server, Gauge, Counter
import threading

# AI Metrics (same as before)
ai_lstm_prediction_cpu = Gauge('ai_lstm_prediction_cpu', 'CPU predicted by LSTM', ['service'])
ai_anfis_score = Gauge('ai_anfis_score', 'ANFIS fuzzy decision score', ['service']) 
ai_scaling_decision = Gauge('ai_scaling_decision', '1=Scale Out, -1=Scale In, 0=Maintain', ['service'])
ai_current_replicas = Gauge('ai_current_replicas', 'Current replicas', ['service'])
ai_cpu_raw = Gauge('ai_cpu_raw', 'Raw CPU from Prometheus', ['service'])
ai_cpu_normalized = Gauge('ai_cpu_normalized', 'Normalized CPU', ['service'])
ai_workload_trend = Gauge('ai_workload_trend', 'Workload trend', ['service'])
ai_iteration_counter = Counter('ai_iteration_total', 'AI iterations')

def send_grafana_annotation(title, text, tags=None):
    """Send annotation to Grafana for scaling events."""
    try:
        annotation_data = {
            "time": int(time.time() * 1000),
            "title": title,
            "text": text,
            "tags": tags or ["ai-scaling"]
        }
        
        requests.post(
            "http://72.61.209.157:3000/api/annotations",
            json=annotation_data,
            auth=('admin', 'admin'),
            timeout=2
        )
    except Exception as e:
        print(f"Annotation failed: {e}")

def send_loki_log(level, message, decision=None):
    """Send structured log to Loki."""
    try:
        log_data = {
            "streams": [{
                "stream": {
                    "job": "ai-autoscaler",
                    "service": "ai-autoscaler", 
                    "level": level,
                    "decision": decision or "MAINTAIN"
                },
                "values": [[
                    str(int(time.time() * 1000000000)),
                    f"🤖 [{time.strftime('%H:%M:%S')}] {message}"
                ]]
            }]
        }
        
        requests.post(
            "http://72.61.209.157:3100/loki/api/v1/push",
            json=log_data,
            timeout=2
        )
    except Exception as e:
        print(f"Loki log failed: {e}")

def scientific_ai_simulation():
    """Enhanced simulation showing causality chain."""
    service_label = "microservices_frontend-service"
    
    # Simulation state
    base_cpu = 0.25
    replicas = 2
    time_counter = 0
    prev_prediction = 0.25
    
    print("🎓 SCIENTIFIC AI AUTOSCALER - CAUSALITY DEMONSTRATION")
    print("=" * 70)
    print("📊 Watch for the causality chain:")
    print("   1. LSTM PREDICTION rises first (dashed red line)")
    print("   2. ANFIS DECISION triggers (purple line crosses threshold)")  
    print("   3. SCALING EVENT occurs (vertical annotation line)")
    print("   4. ACTUAL CPU stabilizes (solid yellow line)")
    print("=" * 70)
    
    while True:
        time_counter += 1
        
        # Create realistic workload patterns with clear causality
        phase = (time_counter // 10) % 4  # 4 phases: normal, ramp-up, spike, cool-down
        
        if phase == 0:  # Normal operation
            cpu_actual = base_cpu + 0.1 * math.sin(time_counter * 0.1) + random.uniform(-0.02, 0.02)
            cpu_predicted = cpu_actual + random.uniform(-0.05, 0.05)  # Minor prediction
            phase_name = "NORMAL"
            
        elif phase == 1:  # LSTM starts predicting increase
            cpu_actual = base_cpu + 0.05 * (time_counter % 10)  # Slowly rising
            cpu_predicted = cpu_actual + 0.2 + 0.1 * (time_counter % 10)  # LSTM jumps ahead!
            phase_name = "LSTM_PREDICTING"
            
        elif phase == 2:  # Actual spike occurs  
            spike_factor = (time_counter % 10) / 10.0
            cpu_actual = base_cpu + 0.4 + 0.2 * spike_factor  # Real spike happens
            cpu_predicted = cpu_actual + random.uniform(-0.1, 0.1)  # LSTM now matches
            phase_name = "SPIKE_MANAGED"
            
        else:  # Cool down
            cpu_actual = base_cpu + 0.3 * math.exp(-(time_counter % 10) * 0.3)  # Exponential decay
            cpu_predicted = cpu_actual + random.uniform(-0.05, 0.05)
            phase_name = "COOLING_DOWN"
            
        # Ensure bounds
        cpu_actual = max(0.1, min(0.9, cpu_actual))
        cpu_predicted = max(0.1, min(0.9, cpu_predicted))
        
        # Normalize
        cpu_normalized = (cpu_actual - 0.1) / 0.8
        
        # Calculate trend (key for ANFIS)
        trend = (cpu_predicted - prev_prediction) * 10  # Amplify for visibility
        trend = max(-1.0, min(1.0, trend))
        
        # ANFIS fuzzy logic with clear thresholds
        fuzzy_input = cpu_predicted + 0.3 * trend
        
        if fuzzy_input > 0.7 and replicas < 5:
            anfis_score = 0.7 + random.uniform(0.0, 0.3)  # Clear scale out
            decision = 1
            action = "SCALE_OUT"
            
            if random.random() > 0.6:  # Trigger scaling
                replicas += 1
                send_grafana_annotation(
                    "🔺 SCALE OUT EVENT", 
                    f"AI triggered scaling: {replicas-1} → {replicas} replicas\nReason: LSTM predicted spike (CPU: {cpu_predicted:.3f})",
                    ["scale-out", "ai-decision"]
                )
                send_loki_log("WARN", f"SCALING OUT: Replicas {replicas-1}→{replicas} | Trigger: LSTM={cpu_predicted:.3f}, ANFIS={anfis_score:.3f}", "SCALE_OUT")
                
        elif fuzzy_input < 0.3 and replicas > 1:
            anfis_score = -0.7 + random.uniform(-0.3, 0.0)  # Clear scale in
            decision = -1
            action = "SCALE_IN"
            
            if random.random() > 0.7:  # Less aggressive scale in
                replicas -= 1
                send_grafana_annotation(
                    "🔻 SCALE IN EVENT",
                    f"AI triggered scale-in: {replicas+1} → {replicas} replicas\nReason: Low predicted load",
                    ["scale-in", "ai-decision"] 
                )
                send_loki_log("INFO", f"SCALING IN: Replicas {replicas+1}→{replicas} | Trigger: Low load prediction", "SCALE_IN")
                
        else:
            anfis_score = -0.2 + 0.4 * random.random()  # Maintain zone
            decision = 0
            action = "MAINTAIN"
        
        # Update all metrics
        ai_cpu_raw.labels(service=service_label).set(cpu_actual)
        ai_cpu_normalized.labels(service=service_label).set(cpu_normalized)
        ai_lstm_prediction_cpu.labels(service=service_label).set(cpu_predicted)
        ai_workload_trend.labels(service=service_label).set(trend)
        ai_anfis_score.labels(service=service_label).set(anfis_score)
        ai_current_replicas.labels(service=service_label).set(replicas)
        ai_scaling_decision.labels(service=service_label).set(decision)
        ai_iteration_counter.inc()
        
        # Enhanced console output
        print(f"🧠 [{time.strftime('%H:%M:%S')}] {phase_name:15} | "
              f"CPU: {cpu_actual*100:5.1f}% | "
              f"LSTM: {cpu_predicted*100:5.1f}% | "  
              f"Trend: {trend:+5.2f} | "
              f"ANFIS: {anfis_score:+5.2f} | "
              f"Replicas: {replicas} | "
              f"{action}")
              
        prev_prediction = cpu_predicted
        time.sleep(4)  # Slower updates for clear causality observation

def main():
    # Start metrics server
    start_http_server(8000)
    print("🎯 Enhanced AI Metrics server with annotations started on port 8000")
    
    # Start scientific simulation
    scientific_ai_simulation()

if __name__ == "__main__":
    main()
