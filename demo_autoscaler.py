#!/usr/bin/env python3
"""
Demo Autoscaler - Simulates the real autoscaler without ML dependencies
This shows the monitoring and scaling logic working.
"""

import os
import sys
import time
import json
import logging
import subprocess
import random
from datetime import datetime
from collections import deque
import requests

class DemoConfig:
    # Prometheus configuration  
    PROMETHEUS_URL = 'http://72.61.209.157:9090'
    CPU_QUERY = 'avg(rate(container_cpu_usage_seconds_total{container_label_com_docker_swarm_service_name="microservices_frontend-service"}[1m]))'
    
    # Service configuration
    TARGET_SERVICE = 'microservices_frontend-service'
    
    # Demo parameters
    LOOP_INTERVAL = 10  # seconds (slower for demo)
    WINDOW_SIZE = 5     # smaller window for demo
    COOLDOWN_PERIOD = 30  # shorter cooldown for demo
    MIN_REPLICAS = 1
    MAX_REPLICAS = 5    # smaller for demo
    
    # Scaling thresholds
    SCALE_OUT_THRESHOLD = 0.4
    SCALE_IN_THRESHOLD = -0.4

class DemoAutoscaler:
    """Demo autoscaler showing the monitoring and scaling logic."""
    
    def __init__(self):
        self.config = DemoConfig()
        self.logger = self._setup_logging()
        
        # State management
        self.sliding_window = deque(maxlen=self.config.WINDOW_SIZE)
        self.last_scaling_time = 0
        self.last_prediction = 0.0
        self.iteration_count = 0
        
        self.logger.info("Demo Autoscaler initialized!")
        self.logger.info(f"Target Service: {self.config.TARGET_SERVICE}")
        self.logger.info(f"Prometheus URL: {self.config.PROMETHEUS_URL}")
        
    def _setup_logging(self):
        """Setup console logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger(__name__)
    
    def _query_prometheus(self):
        """Query Prometheus for CPU metrics."""
        try:
            url = f"{self.config.PROMETHEUS_URL}/api/v1/query"
            params = {'query': self.config.CPU_QUERY}
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                cpu_usage = float(data['data']['result'][0]['value'][1])
                return cpu_usage
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"⚠️  Prometheus query failed: {str(e)}, assuming 0.0")
            return 0.0
    
    def _get_current_replicas(self):
        """Get current replica count using Docker CLI."""
        try:
            result = subprocess.run(
                ['docker', 'service', 'ls', '--format', 'json'],
                capture_output=True, text=True, check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    service = json.loads(line)
                    if self.config.TARGET_SERVICE in service.get('Name', ''):
                        replicas_str = service.get('Replicas', '1/1')
                        current_replicas = int(replicas_str.split('/')[1])
                        return current_replicas
            
            return 1
            
        except Exception as e:
            self.logger.error(f"❌ Error getting replicas: {str(e)}")
            return 1
    
    def _scale_service(self, new_replicas):
        """Scale service using Docker CLI."""
        try:
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
            self.logger.error(f"❌ Error scaling: {str(e)}")
            return False
    
    def _simulate_ml_prediction(self, cpu_value):
        """Simulate LSTM + ANFIS predictions for demo."""
        # Add to sliding window
        self.sliding_window.append(cpu_value)
        
        if len(self.sliding_window) < self.config.WINDOW_SIZE:
            return None, None, None
        
        # Simulate LSTM prediction (with some realistic logic)
        recent_avg = sum(list(self.sliding_window)[-3:]) / 3
        lstm_prediction = recent_avg + random.uniform(-0.1, 0.1)  # Add some noise
        
        # Calculate trend
        trend = lstm_prediction - self.last_prediction
        self.last_prediction = lstm_prediction
        
        # Simulate ANFIS score (simple heuristic based on prediction and trend)
        if lstm_prediction > 0.6 and trend > 0.05:
            anfis_score = 0.7  # Strong scale out signal
        elif lstm_prediction > 0.4 and trend > 0.0:
            anfis_score = 0.3  # Moderate scale out
        elif lstm_prediction < 0.2 and trend < -0.05:
            anfis_score = -0.7  # Strong scale in
        elif lstm_prediction < 0.3 and trend < 0.0:
            anfis_score = -0.3  # Moderate scale in
        else:
            anfis_score = random.uniform(-0.2, 0.2)  # Maintain zone
        
        return lstm_prediction, trend, anfis_score
    
    def _make_scaling_decision(self, anfis_score, current_replicas):
        """Make scaling decision."""
        # Check cooldown
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
    
    def _format_log_entry(self, cpu_raw, lstm_pred, trend, anfis_score, 
                         current_replicas, action="MAINTAIN"):
        """Format log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        cpu_pct = cpu_raw * 100 if cpu_raw else 0
        
        # Icons
        if action.startswith("SCALE OUT"):
            icon = "📈"
        elif action.startswith("SCALE IN"):
            icon = "📉"
        elif action == "COOLDOWN":
            icon = "⏳"
        else:
            icon = "🟢"
        
        # Format values
        lstm_str = f"{lstm_pred:+.3f}" if lstm_pred is not None else "  N/A  "
        trend_str = f"{trend:+.3f}" if trend is not None else "  N/A  "
        anfis_str = f"{anfis_score:+.3f}" if anfis_score is not None else "  N/A  "
        
        log_msg = (f"{icon} [{timestamp}] "
                  f"CPU: {cpu_pct:5.1f}% | "
                  f"LSTM: {lstm_str:>7s} | "
                  f"Trend: {trend_str:>7s} | "
                  f"ANFIS: {anfis_str:>7s} | "
                  f"Replicas: {current_replicas:2d} | "
                  f"{action}")
        
        self.logger.info(log_msg)
    
    def run_demo(self):
        """Run the demo autoscaling loop."""
        
        print("\n" + "="*80)
        print("🎭 DEMO AUTOSCALER - SHOWING LOGIC WITHOUT ML")
        print("   Prometheus → Simulated LSTM/ANFIS → Docker Scaling")
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
                
                # 1. Query real Prometheus data
                cpu_raw = self._query_prometheus()
                
                # 2. Simulate ML predictions
                lstm_pred, trend, anfis_score = self._simulate_ml_prediction(cpu_raw)
                
                # 3. Get current replicas
                current_replicas = self._get_current_replicas()
                
                # 4. Handle initialization phase
                if lstm_pred is None:
                    remaining = self.config.WINDOW_SIZE - len(self.sliding_window)
                    self.logger.info(f"🔄 Initializing buffer... {len(self.sliding_window)}/{self.config.WINDOW_SIZE} "
                                   f"(need {remaining} more)")
                    time.sleep(self.config.LOOP_INTERVAL)
                    continue
                
                # 5. Make scaling decision
                new_replicas, action = self._make_scaling_decision(anfis_score, current_replicas)
                
                # 6. Execute scaling if needed
                if action in ["SCALE OUT", "SCALE IN"]:
                    if self._scale_service(new_replicas):
                        old_replicas = current_replicas
                        current_replicas = new_replicas
                        action_detail = f"{action} ({old_replicas}->{current_replicas})"
                    else:
                        action_detail = f"{action} FAILED"
                else:
                    action_detail = action
                
                # 7. Log entry
                self._format_log_entry(cpu_raw, lstm_pred, trend, anfis_score,
                                     current_replicas, action_detail)
                
                # 8. Wait
                time.sleep(self.config.LOOP_INTERVAL)
                
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Demo stopped by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"\n❌ Fatal error: {str(e)}")
            raise

def main():
    """Main entry point."""
    print("🎭 DEMO: Real-time Hybrid Autoscaler")
    print("   Showing monitoring and scaling logic")
    
    try:
        demo = DemoAutoscaler()
        demo.run_demo()
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
