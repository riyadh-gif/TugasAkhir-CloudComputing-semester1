# 🚀 Hybrid LSTM-ANFIS Autoscaling System

**Predictive LSTM and Neuro-Fuzzy Hybrid Autoscaling for Microservices in Lightweight PaaS Environments**

> Master Thesis Project - Sistem autoscaling cerdas yang menggabungkan LSTM untuk prediksi workload dan ANFIS untuk keputusan scaling.

---

## 📋 Daftar Isi

- [Overview](#overview)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Struktur Folder](#struktur-folder)
- [Requirements](#requirements)
- [Cara Penggunaan](#cara-penggunaan)
- [Hasil Eksperimen](#hasil-eksperimen)
- [API Production](#api-production)

---

## Overview

Sistem ini terdiri dari 2 komponen utama:

| Komponen | Fungsi | Output |
|----------|--------|--------|
| **LSTM** | Memprediksi workload (CPU/Memory) masa depan | Nilai prediksi 0-1 |
| **ANFIS** | Membuat keputusan scaling berdasarkan prediksi | -1 (Scale In) → 0 (Maintain) → +1 (Scale Out) |

### Alur Kerja

```
[Data Historis] → [LSTM Prediction] → [ANFIS Decision] → [Scaling Action]
     ↓                   ↓                   ↓                  ↓
  20 timesteps      CPU: 75%           +0.8 (Scale Out)    Add 1 Pod
```

---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                   HYBRID AUTOSCALER                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌───────────────┐     ┌───────────────┐                  │
│   │     LSTM      │     │     ANFIS     │                  │
│   │   (Keras)     │ ──► │  (5-Layer)    │                  │
│   │               │     │               │                  │
│   │ Input: (20,4) │     │ Input: (2,)   │                  │
│   │ Output: (2,)  │     │ Output: (1,)  │                  │
│   └───────────────┘     └───────────────┘                  │
│         │                      │                            │
│         ▼                      ▼                            │
│   [CPU, Memory]         [Scaling Decision]                 │
│    Prediction            -1 to +1                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Struktur Folder

```
dhotok/
│
├── 📊 DATA
│   ├── borg_traces_data.csv          # Dataset utama (Google Borg Traces)
│   └── processed_data/               # Data yang sudah diproses
│       ├── X_train.npy               # Training input (122K samples)
│       ├── y_train.npy               # Training target
│       ├── X_val.npy                 # Validation input (30K samples)
│       ├── y_val.npy                 # Validation target
│       ├── scaler.joblib             # MinMaxScaler untuk denormalisasi
│       └── metadata.joblib           # Konfigurasi preprocessing
│
├── 🧠 MODELS
│   ├── best_lstm_model.keras         # Model LSTM terlatih
│   ├── best_anfis_journal.weights.h5 # Bobot ANFIS terlatih
│   └── best_anfis_journal_config.joblib
│
├── 🔧 SCRIPTS (Jalankan secara berurutan)
│   ├── 1_preprocessing_pipeline.py   # Step 1: Preprocessing data
│   ├── 2_train_lstm_model.py         # Step 2: Training LSTM
│   ├── 3_train_anfis_journal.py      # Step 3: Training ANFIS
│   ├── 4_evaluate_journal_final.py   # Step 4: Evaluasi & grafik
│   └── 5_hybrid_autoscaler.py        # Step 5: Production inference
│
├── 📈 RESULTS
│   └── journal_figures/              # Grafik untuk publikasi
│       ├── figure4_control_surface_3d.png
│       ├── figure5_comparative_response.png
│       └── figure6_regression_analysis.png
│
├── 📝 LOGS
│   └── autoscaler.log                # Log inference
│
└── README.md                         # File ini
```

---

## Requirements

### Hardware (Recommended)
- GPU: NVIDIA dengan CUDA support (tested on A40 48GB)
- RAM: Minimal 16GB
- Storage: 5GB free space

### Software
```bash
# Python 3.10+
pip install numpy pandas scikit-learn joblib matplotlib

# Deep Learning
pip install tensorflow  # atau tensorflow-gpu

# Optional (untuk REST API)
pip install flask
```

### Quick Install
```bash
pip install numpy pandas scikit-learn joblib matplotlib tensorflow
```

---

## Cara Penggunaan

### 🔵 Option A: Gunakan Model yang Sudah Terlatih (Recommended)

Jika ingin langsung menggunakan model tanpa training ulang:

```python
from hybrid_autoscaler import HybridAutoscaler

# Inisialisasi
scaler = HybridAutoscaler()

# Siapkan data (20 data points terakhir, normalized 0-1)
cpu_history = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.72,
               0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92]
mem_history = [0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58,
               0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78]

# Dapatkan keputusan scaling
decision = scaler.get_scaling_decision(cpu_history, mem_history)

print(f"Action: {decision.action}")           # SCALE_OUT / MAINTAIN / SCALE_IN
print(f"Intensity: {decision.intensity}")     # -1.0 to +1.0
print(f"Reason: {decision.reason}")           # Penjelasan
```

### 🟢 Option B: Training dari Awal

Jalankan script secara berurutan:

```bash
cd /root/dhotok

# Step 1: Preprocessing (5 menit)
python preprocessing_pipeline.py

# Step 2: Training LSTM (10-15 menit dengan GPU)
python train_lstm_model.py

# Step 3: Training ANFIS (5-10 menit)
python train_anfis_journal.py

# Step 4: Evaluasi dan generate grafik
python evaluate_journal_final.py

# Step 5: Test inference
python hybrid_autoscaler.py --demo
```

### 🟡 Option C: REST API untuk Production

```bash
# Install Flask
pip install flask

# Jalankan server
python hybrid_autoscaler.py --serve --port 8080
```

```bash
# Test dengan curl
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_usage": [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.86,
                  0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98],
    "memory_usage": [0.4, 0.45, 0.5, 0.55, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,
                     0.72, 0.74, 0.76, 0.78, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85]
  }'
```

Response:
```json
{
  "timestamp": "2025-12-03T21:30:00",
  "action": "SCALE_OUT",
  "intensity": 0.85,
  "confidence": 0.95,
  "predicted_cpu": 0.98,
  "recommended_replicas": 2,
  "reason": "High predicted CPU (98.0%). Scaling out to prevent overload."
}
```

---

## Hasil Eksperimen

### Performance Metrics

| Model | MSE | MAE | R² | Accuracy |
|-------|-----|-----|-----|----------|
| **LSTM** | 0.0012 | 0.0120 | 0.70 | - |
| **ANFIS** | 0.0418 | 0.0999 | -0.14 | **80.0%** |
| Static Threshold | 0.8629 | 0.9089 | -22.48 | 19.4% |

### Improvement vs Static Baseline

- **MSE Reduction:** 95.2%
- **MAE Reduction:** 89.0%
- **Decision Accuracy:** +60.6%

### Grafik Hasil

Lihat folder `results/journal_figures/`:

1. **figure4_control_surface_3d.png** - 3D control surface ANFIS
2. **figure5_comparative_response.png** - Perbandingan time series
3. **figure6_regression_analysis.png** - Analisis regresi

---

## Troubleshooting

### Error: Module not found
```bash
pip install numpy pandas scikit-learn joblib matplotlib tensorflow
```

### Error: CUDA not available
Model akan tetap berjalan di CPU, tapi lebih lambat. Untuk GPU:
```bash
pip install tensorflow[and-cuda]
```

### Error: Memory limit
Kurangi `BATCH_SIZE` di script training atau gunakan sampel data yang lebih kecil.

---

## Citation

Jika menggunakan kode ini untuk penelitian, mohon cite:

```bibtex
@thesis{hybrid_autoscaling_2025,
  title={Predictive LSTM and Neuro-Fuzzy Hybrid Autoscaling for Microservices},
  author={[Your Name]},
  year={2025},
  school={[Your University]}
}
```

---

## License

MIT License - Bebas digunakan untuk keperluan akademik dan penelitian.

---

## Contact

Untuk pertanyaan atau kolaborasi, silakan buat Issue atau hubungi author.

---

**Happy Scaling! 🚀**
