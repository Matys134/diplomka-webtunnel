# WebTunnel Traffic Analysis & Resilience Research
> **Master's Thesis Research Project** | Faculty of Science, University of South Bohemia in České Budějovice  
> **Author:** Bc. Matěj Kouba

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Accelerated-green.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose%20Testbed-2496ed.svg)](https://www.docker.com/)

---

## 📌 Abstract & Research Focus

**WebTunnel** is a modern Pluggable Transport for the Tor network designed to bypass aggressive Deep Packet Inspection (DPI) and state-level firewalls (such as the Great Firewall of China or Russian TSPU). It encapsulates Tor traffic into standardized HTTP/2 and WebSocket streams over TLS 1.3, making it appear as normal HTTPS traffic to an external web server.

This thesis investigates the **traffic analysis resilience** of WebTunnel against advanced Machine Learning (ML) and Deep Learning (DL) classifiers under realistic network conditions (Broadband, 4G/LTE, and Lossy WAN). We demonstrate the fundamental protocol vulnerabilities (514-byte Tor cell quantization and circuit negotiation burst patterns), evaluate base rate fallacy under ISP-scale deployment, and formulate/benchmark novel protocol-level countermeasures (**Cell Coalescing & Cover Mimicry**).

---

## 🔬 Experimental Results Summary

### 1. Model Comparison (5-Fold Stratified Cross-Validation)
Evaluated across **1,301 verified TLS 1.3 network flows** against realistic *Hard Negatives* (WebSocket Tickers, Interactive WebSocket Chat, DASH Video Streaming, and HTTPS Web Assets):

| Model | Architecture / Hardware | Accuracy | PR-AUC | ROC-AUC | Latency / Flow | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Baseline)** | 48 Tabular Stats (Ryzen 9800X3D) | **$99.5 \pm 0.3\%$** | **$1.000 \pm 0.000$** | **$1.000 \pm 0.000$** | **0.0003 ms** | **~3,660,000 flows/s** |
| **1D-CNN (Deep Packet)** | 1D ConvNet (RTX 5070 Ti CUDA) | **$99.3 \pm 0.6\%$** | **$0.993 \pm 0.013$** | **$0.999 \pm 0.002$** | **0.1201 ms** | **~8,330 flows/s** |
| **Flow-Transformer** | Multi-Head Self-Attention (CUDA) | **$98.5\%$** | **$0.998$** | **$0.999$** | **0.1760 ms** | **~5,680 flows/s** |

### 2. Pre- vs. Post-Handshake Analysis
By stripping all initial TLS handshakes (`post_handshake_only=True`), we empirically prove that model detection is **NOT dependent on TLS parameters**, but stems purely from the **514-byte cell payload quantization**:
* **Full Flow:** XGBoost Acc: **99.2%**, 1D-CNN Acc: **99.6%**
* **Post-Handshake Only:** XGBoost Acc: **99.2%**, 1D-CNN Acc: **99.6%**

### 3. Before vs. After Defense Evaluation
We propose and evaluate **Dynamic Cell Coalescing & Cover-Protocol Mimicry**:
* **1D-CNN Recall (Detection Rate):** Drops from **93.3%** down to **71.7%** (-21.6%).
* **1D-CNN F1-Score:** Drops from **96.6%** down to **83.5%** (-13.1%).
* **Spectral Quantization:** Erases the sharp 624 B and 1138 B peaks to match benign traffic.

---

## 🏗️ Repository Architecture

```text
├── 0_thesis_text/                 # LaTeX tables and upcoming thesis chapters
│   └── tables/                    # Auto-generated LaTeX tables (\input ready)
├── 1_testbed/                     # Isolated Dockerized testbed
│   ├── client/                    # WebTunnel client + traffic generator
│   ├── tor_bridge/                # Official Tor WebTunnel bridge server
│   ├── webtunnel_server/          # Nginx TLS 1.3 reverse proxy & decoy site
│   ├── legitimate_servers/        # TLS 1.3 FastAPI mock server (Hard Negatives)
│   ├── router/                    # Linux tc-netem network profile emulation
│   └── capture/                   # Multithreaded PCAP capture orchestrator
├── 2_data_pipeline/               # Anti-leakage data processing
│   ├── sanitizer.py               # L2/L3/L4 header stripping & feature extraction
│   ├── build_dataset.py           # Parallel dataset builder (Train/Val/Test)
│   └── inspect_dataset.py         # Spectral and IAT distribution analyzer
├── 3_models/                      # Machine Learning and Deep Learning models
│   ├── train_xgboost.py           # XGBoost classifier + dynamic scale_pos_weight
│   ├── train_1d_cnn.py            # PyTorch 1D-CNN (CUDA) with Focal Loss
│   ├── train_transformer.py       # PyTorch Flow-Transformer (CUDA)
│   ├── cross_validate.py          # 5-Fold Stratified Cross-Validation
│   └── explain_models.py          # XAI: Feature Importance & Gradient Saliency
├── 4_evaluation/                  # Experimental evaluation & figures
│   ├── evaluate_before_after_defenses.py # Full before-vs-after defense benchmark
│   ├── evaluate_post_handshake.py        # Pre- vs post-handshake evaluation
│   ├── evaluate_base_rate_fallacy.py    # Base rate fallacy & Bayes aggregation
│   ├── export_latex_tables.py           # Generates all .tex tables for the thesis
│   └── plots/                           # High-resolution (300 DPI) publication figures
└── run_full_benchmark.py          # Master orchestrator executing entire pipeline
```

---

## 🚀 Quick Start

### 1. Prerequisites
* Linux OS (Ubuntu / Debian / Arch / Fedora)
* Python 3.10+ with `virtualenv`
* Docker and Docker Compose
* NVIDIA GPU with CUDA support (optional, CPU fallback supported)

### 2. Setup Environment
```bash
# Clone the repository
git clone git@github.com:Matys134/diplomka-webtunnel.git
cd diplomka-webtunnel

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install xgboost scikit-learn dpkt matplotlib seaborn numpy
```

### 3. Run the Full Autonomous Benchmark
```bash
# Runs full pipeline: Capture -> Sanitize -> Train -> CV -> XAI -> Defenses -> Export
python3 run_full_benchmark.py --samples-per-profile 100
```

---

## 📜 License & Citation
This project is developed as part of academic research at the Faculty of Science, University of South Bohemia.
