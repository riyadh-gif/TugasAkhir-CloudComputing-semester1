#!/usr/bin/env python3
"""
Test script to verify autoscaler components work correctly
"""

import os
import sys
import json
import subprocess

def test_model_files():
    """Test if all model files exist."""
    print("🔍 Testing model files...")
    
    files = {
        'LSTM Model': '/srv/cloud-computing/ai/best_lstm_model.keras',
        'ANFIS Weights': '/srv/cloud-computing/ai/best_anfis_journal.weights.h5',
        'ANFIS Config': '/srv/cloud-computing/ai/best_anfis_journal_config.joblib',
        'Scaler': '/srv/cloud-computing/ai/processed_data/scaler.joblib'
    }
    
    all_exist = True
    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✅ {name}: {path} ({size:,} bytes)")
        else:
            print(f"  ❌ {name}: {path} (NOT FOUND)")
            all_exist = False
    
    return all_exist

def test_dependencies():
    """Test Python dependencies."""
    print("\n🔍 Testing Python dependencies...")
    
    deps = ['numpy', 'requests']
    all_available = True
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep}")
            all_available = False
    
    # Test optional ML dependencies
    ml_deps = ['tensorflow', 'joblib', 'sklearn']
    for dep in ml_deps:
        try:
            if dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ⚠️  {dep} (will need to install)")
    
    return all_available

def test_docker_access():
    """Test Docker CLI access."""
    print("\n🔍 Testing Docker access...")
    
    try:
        result = subprocess.run(['docker', 'service', 'ls'], 
                              capture_output=True, text=True, check=True)
        services = result.stdout.strip().split('\n')[1:]  # Skip header
        
        print(f"  ✅ Docker CLI accessible ({len(services)} services found)")
        
        # Look for our target service
        for line in services:
            if 'microservices_frontend-service' in line:
                print(f"  ✅ Target service found: {line.split()[1]}")
                return True
        
        print("  ⚠️  Target service 'microservices_frontend-service' not found")
        print("     Available services:")
        for line in services:
            parts = line.split()
            if len(parts) >= 2:
                print(f"       - {parts[1]}")
        
        return True  # Docker works, just service not found
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Docker CLI failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Docker test failed: {e}")
        return False

def test_prometheus():
    """Test Prometheus connection."""
    print("\n🔍 Testing Prometheus connection...")
    
    try:
        import requests
        
        url = 'http://localhost:9090/api/v1/query'
        params = {'query': 'up'}
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        if data['status'] == 'success':
            targets = len(data['data']['result'])
            print(f"  ✅ Prometheus accessible ({targets} targets)")
            return True
        else:
            print(f"  ❌ Prometheus query failed: {data}")
            return False
            
    except ImportError:
        print("  ❌ requests library not available")
        return False
    except Exception as e:
        print(f"  ⚠️  Prometheus connection failed: {e}")
        print("      (This is OK if Prometheus isn't running yet)")
        return False

def main():
    """Run all tests."""
    print("🎓 AUTOSCALER COMPONENT TEST")
    print("="*50)
    
    tests = [
        ("Model Files", test_model_files),
        ("Dependencies", test_dependencies), 
        ("Docker Access", test_docker_access),
        ("Prometheus", test_prometheus)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    print("\n" + "="*50)
    print("📋 SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready to run autoscaler.")
    else:
        print("⚠️  Some tests failed. Check issues above.")
        print("💡 You may still be able to run with missing ML libraries.")
    
    print("\n🚀 To run the autoscaler:")
    print("   python3 realtime_autoscaler_simple.py")

if __name__ == "__main__":
    main()
