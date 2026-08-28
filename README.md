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

This thesis investigates the **traffic analysis resilience** of WebTunnel against advanced Machine Learning (ML) and Deep Learning (DL) classifiers under realistic network conditions (Broadband, 4G/LTE, and Lossy WAN). We demonstrate the fundamental protocol vulnerabilities (514-byte Tor cell quantization and circuit negotiation burst patterns), evaluate base rate fallacy under ISP-scale deployment, and formulate/benchmark novel protocol-level countermeasures (**Cell Coalescing, Adaptive Padding & Cover Mimicry**).

---

## 🔬 Experimental Results Summary

### 1. Model Comparison (5-Fold Stratified Cross-Validation)
Evaluated across **1,800 PCAPs (1,595 verified TLS 1.3 network flows)** against 5 realistic *Hard Negative* classes (*Direct Web Browsing*, *WebSocket Tickers*, *Interactive WebSocket Chat*, *DASH Video Streaming*, and *HTTPS Web Assets*):

| Model | Architecture / Hardware | Accuracy ($\mu \pm \sigma$) | PR-AUC ($\mu \pm \sigma$) | ROC-AUC ($\mu \pm \sigma$) | Latency / Flow | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Baseline)** | 48 Tabular Stats (Ryzen 9800X3D) | **$99.7 \pm 0.2\%$** | **$1.000 \pm 0.001$** | **$1.000 \pm 0.000$** | **0.0004 ms** | **~2,750,000 flows/s** |
| **1D-CNN (Deep Packet)** | 1D ConvNet (RTX 5070 Ti CUDA) | **$99.6 \pm 0.3\%$** | **$0.988 \pm 0.025$** | **$0.999 \pm 0.002$** | **0.1203 ms** | **~8,310 flows/s** |
| **Flow-Transformer** | Multi-Head Self-Attention (CUDA) | **$99.2 \pm 0.8\%$** | **$0.998 \pm 0.003$** | **$1.000 \pm 0.001$** | **0.1761 ms** | **~5,680 flows/s** |

### 2. 2-Tier Cascaded Classification Architecture (L1 CPU $\rightarrow$ L2 GPU)
To achieve line-rate inspection on ISP backbone networks, we design and benchmark a hybrid pipeline:
* **L1 CPU Filter (XGBoost):** Resolves **99.1%** of traffic with sub-microsecond latency.
* **L2 GPU Inspection (1D-CNN):** Inspects only ambiguous flows ($0.05 \le p \le 0.95$, representing **0.9%** of traffic).
* **Overall Hybrid Throughput:** **2,417,606 flows/second** with **99.69% accuracy**.

### 3. Pre- vs. Post-Handshake Analysis
By stripping all initial TLS handshakes (`post_handshake_only=True`), we empirically prove that model detection is **NOT dependent on TLS metadata**, but stems purely from the **514-byte Tor cell payload quantization**:
* **Full Flow:** XGBoost Acc: **99.4%**, 1D-CNN Acc: **99.7%**
* **Post-Handshake Only:** XGBoost Acc: **98.7%**, 1D-CNN Acc: **98.4%**

### 4. Countermeasure Evaluation (Before vs. After Defense)
We evaluate two tiers of protocol-level defenses against ML/DL surveillance:
1. **Adaptive Intra-frame Padding (1–128 B):** Bandwidth overhead **9.7%**; reduces 1D-CNN detection recall to **85.0%**.
2. **Dynamic Cell Coalescing & Cover Mimicry:** Bandwidth overhead **70.8%**; completely erases the 624 B and 1138 B spectral quantization peaks.

---

## 🏗️ Repository Architecture

```text
├── 0_thesis_text/                 # LaTeX tables and upcoming thesis chapters
│   └── tables/                    # Auto-generated LaTeX tables (\input ready)
├── 1_testbed/                     # Isolated Dockerized testbed
│   ├── client/                    # WebTunnel client + traffic generator (6 classes)
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
│   ├── cross_validate.py          # 5-Fold Stratified Cross-Validation (All 3 models)
│   └── explain_models.py          # XAI: SHAP Beeswarm Summary & Gradient Saliency
├── 4_evaluation/                  # Experimental evaluation & figures
│   ├── evaluate_cascaded_pipeline.py    # 2-Tier L1 CPU -> L2 GPU cascaded benchmark
│   ├── evaluate_det_curve.py            # Logarithmic DET curve (Low-FPR regime)
│   ├── evaluate_confusion_matrix.py     # Multi-class confusion matrix breakdown
│   ├── evaluate_cross_profile.py        # Cross-profile domain generalization
│   ├── evaluate_before_after_defenses.py # Multi-level defense benchmark
│   ├── evaluate_post_handshake.py       # Pre- vs post-handshake evaluation
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
pip install xgboost scikit-learn dpkt matplotlib seaborn numpy shap pandas
```

### 3. Run the Full Autonomous Benchmark
```bash
# Runs full pipeline: Capture -> Sanitize -> Train -> CV -> XAI -> Defenses -> Export
python3 run_full_benchmark.py --samples-per-profile 100
```

---

## 📜 License & Citation
This project is developed as part of academic research at the Faculty of Science, University of South Bohemia.
