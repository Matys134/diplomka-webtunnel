#!/usr/bin/env python3
"""
Logarithmic Detection Error Tradeoff (DET) Curve:
Plots False Negative Rate (FNR) vs False Positive Rate (FPR) on a logarithmic scale,
providing deep visualization of low-FPR operating points critical for ISP-scale deployment.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import xgboost as xgb
from sklearn.metrics import roc_curve

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import (
    TABULAR_DATASET_PATH,
    SEQUENCE_DATASET_PATH,
    XGBOOST_MODEL_JSON,
    CNN_MODEL_PATH,
    TRANSFORMER_MODEL_PATH,
    PLOTS_DIR,
    setup_matplotlib_style
)
from architectures import WebTunnel1DCNN, WebTunnelTransformer
from utils import load_tabular_data, load_sequence_data, get_device


def generate_logarithmic_det_curve():
    print("=== Generating Logarithmic DET (Detection Error Tradeoff) Curve ===")
    setup_matplotlib_style()
    device = get_device()

    tab_data = load_tabular_data(TABULAR_DATASET_PATH)
    seq_data = load_sequence_data(SEQUENCE_DATASET_PATH)

    X_test_tab, y_test = tab_data["X_test"], tab_data["y_test"]
    X_test_seq = seq_data["X_test"]

    # 1. Predictions
    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model(XGBOOST_MODEL_JSON)
    p_xgb = clf_xgb.predict_proba(X_test_tab)[:, 1]

    model_cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model_cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model_cnn.eval()
    with torch.no_grad():
        p_cnn = model_cnn(torch.tensor(X_test_seq, dtype=torch.float32).permute(0, 2, 1).to(device)).squeeze(-1).cpu().numpy()

    model_tf = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
    model_tf.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=device))
    model_tf.eval()
    with torch.no_grad():
        p_tf = model_tf(torch.tensor(X_test_seq, dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()

    # 2. FPR & FNR calculation
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, p_xgb)
    fnr_xgb = 1.0 - tpr_xgb

    fpr_cnn, tpr_cnn, _ = roc_curve(y_test, p_cnn)
    fnr_cnn = 1.0 - tpr_cnn

    fpr_tf, tpr_tf, _ = roc_curve(y_test, p_tf)
    fnr_tf = 1.0 - tpr_tf

    # 3. Logarithmic DET Plot
    plt.figure(figsize=(9, 7))

    plt.plot(np.maximum(fpr_xgb, 1e-4) * 100, np.maximum(fnr_xgb, 1e-4) * 100,
             label="XGBoost Baseline (Tabulární)", color="#2ca02c", linewidth=2.5)
    plt.plot(np.maximum(fpr_cnn, 1e-4) * 100, np.maximum(fnr_cnn, 1e-4) * 100,
             label="1D-CNN (Deep Packet)", color="#1f77b4", linewidth=2.5, linestyle="--")
    plt.plot(np.maximum(fpr_tf, 1e-4) * 100, np.maximum(fnr_tf, 1e-4) * 100,
             label="Flow-Transformer ([CLS] Attention)", color="#ff7f0e", linewidth=2.5, linestyle="-.")

    # Reference lines for operational targets
    plt.axvline(x=0.1, color="red", linestyle=":", alpha=0.7, label=r"Cenzurní limit ($FPR = 0.1\%$)")
    plt.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5, label=r"Limit pro Edge Node ($FPR = 1\%$)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0.01, 100)
    plt.ylim(0.01, 100)

    plt.xlabel("False Positive Rate (FPR %) - Log Scale")
    plt.ylabel("False Negative Rate / Miss Rate (FNR %) - Log Scale")
    plt.title("Logarithmic Detection Error Tradeoff (DET) Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_plot = os.path.join(PLOTS_DIR, "det_curve_logarithmic.png")
    plt.savefig(out_plot)
    plt.close()
    print(f"[OK] Saved {out_plot}")


if __name__ == "__main__":
    generate_logarithmic_det_curve()
