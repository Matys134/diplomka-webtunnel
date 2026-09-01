"""
Central configuration module for WebTunnel Resilience and Traffic Analysis Benchmark.
Defines unified paths, random seeds, class labels, network profiles, model hyperparameters,
and publication-ready visualization styles.
"""
import os
import sys
import random
import numpy as np

# ==============================================================================
# 1. DIRECTORY & FILE PATHS
# ==============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ensure sub-packages are importable across directories
for sub_dir in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "3_models"), os.path.join(PROJECT_ROOT, "2_data_pipeline")]:
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_PCAP_DIR = os.path.join(DATA_DIR, "raw_pcap")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

TABULAR_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "tabular_dataset.npz")
SEQUENCE_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "sequence_dataset.npz")
DATASET_SUMMARY_PATH = os.path.join(PROCESSED_DATA_DIR, "dataset_summary.json")

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "3_models", "saved_models")
XGBOOST_MODEL_JSON = os.path.join(SAVED_MODELS_DIR, "xgboost_baseline.json")
XGBOOST_MODEL_JOBLIB = os.path.join(SAVED_MODELS_DIR, "xgboost_baseline.joblib")
CNN_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "1d_cnn_best.pt")
TRANSFORMER_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "transformer_best.pt")

EVALUATION_DIR = os.path.join(PROJECT_ROOT, "4_evaluation")
PLOTS_DIR = os.path.join(EVALUATION_DIR, "plots")
LATEX_TABLES_DIR = os.path.join(PROJECT_ROOT, "0_thesis_text", "tables")

# Ensure required directories exist
for d in [PROCESSED_DATA_DIR, SAVED_MODELS_DIR, PLOTS_DIR, LATEX_TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# ==============================================================================
# 2. TRAFFIC CLASSES & LABELS
# ==============================================================================
CLASSES = [
    "webtunnel",
    "direct_web_browsing",
    "websocket_ticker",
    "websocket_chat",
    "video_streaming",
    "web_assets"
]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}

CLASS_DISPLAY_NAMES = {
    "webtunnel": "WebTunnel (Tor Transport)",
    "direct_web_browsing": "Direct Web Browsing (HTTP/2)",
    "websocket_ticker": "WebSocket Ticker (Live Orderbook)",
    "websocket_chat": "WebSocket Chat (Rich Collaboration)",
    "video_streaming": "DASH Video Streaming (ABR)",
    "web_assets": "HTTPS Web Assets (Bundles & Chunks)"
}

CLASS_SHORT_NAMES = [
    "WebTunnel",
    "Direct Browsing",
    "WS Ticker",
    "WS Chat",
    "Video Stream",
    "Web Assets"
]

# ==============================================================================
# 3. NETWORK PROFILES (NETEM)
# ==============================================================================
NETEM_PROFILES = ["broadband", "lte", "lossy"]

# F-07: these strings are DERIVED from the parameters netem actually applies
# (1_testbed/router/netem_profiles.sh and the collector's netem_params()).  v1/v2.0 hand-wrote
# "2ms RTT" / "30ms RTT" / "80ms RTT" while the script applied 20/45/90 ms, and those strings
# went straight into the thesis plots and tables.
NETEM_APPLIED = {
    "broadband": {"delay": "20ms", "jitter": "4ms",  "loss": "0.05%", "rate": "200mbit"},
    "lte":       {"delay": "45ms", "jitter": "15ms", "loss": "0.2%",  "rate": "40mbit"},
    "lossy":     {"delay": "90ms", "jitter": "25ms", "loss": "Gilbert-Elliot ~2%", "rate": "8mbit"},
}
PROFILE_LABELS = {"broadband": "Broadband", "lte": "4G/LTE", "lossy": "Lossy WAN"}
PROFILE_DISPLAY_NAMES = {
    k: (f"{PROFILE_LABELS[k]} ({v['rate']}, RTT {v['delay']} +/- {v['jitter']}, loss {v['loss']}, "
        f"ingress+egress)")
    for k, v in NETEM_APPLIED.items()
}

# ==============================================================================
# 4. GLOBAL MODEL & PIPELINE CONSTANTS
# ==============================================================================
RANDOM_SEED = 42
MAX_SEQUENCE_LENGTH = 200
SEQUENCE_CHANNELS = 2  # (Direction, PacketLength)
TABULAR_FEATURE_COUNT = 48

# Session-stratified split ranges (out of 100 sessions per profile per class)
TRAIN_SESSION_RANGE = (1, 70)
VAL_SESSION_RANGE = (71, 85)
TEST_SESSION_RANGE = (86, 100)


def set_global_seed(seed: int = RANDOM_SEED):
    """Sets random seeds across Python, NumPy and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# ==============================================================================
# 5. PUBLICATION PLOT STYLING (MATPLOTLIB)
# ==============================================================================
def setup_matplotlib_style():
    """Configures unified, publication-quality styling for all Matplotlib figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

