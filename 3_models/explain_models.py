#!/usr/bin/env python3
"""
Explainable AI (XAI) and Interpretability Analysis for WebTunnel Detection Models:
- XGBoost: Tree SHAP (Feature Importance Ranking & Summary Beeswarm Plot).
- 1D-CNN: Input Gradient Saliency Attribution Map across packet sequence positions.
- Flow-Transformer: Sequence-level Gradient Saliency Attribution Map across packet positions.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import torch

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


def explain_xgboost():
    print("--- Running XGBoost Feature Attribution & SHAP Analysis ---")
    setup_matplotlib_style()

    data = load_tabular_data(TABULAR_DATASET_PATH)
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = data["feature_names"]

    clf = xgb.XGBClassifier()
    clf.load_model(XGBOOST_MODEL_JSON)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    # 1. Feature Importance Bar Plot (Top 15)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:15]
    top_names = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top_names)), top_values[::-1], color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Output)")
    ax.set_title("Top 15 Most Discriminative Features (XGBoost SHAP Attribution)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "xgboost_feature_importance.png"))
    plt.close()
    print(f"[OK] Saved {os.path.join(PLOTS_DIR, 'xgboost_feature_importance.png')}")

    # 2. SHAP Beeswarm Summary Plot
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, max_display=15, show=False)
    plt.title("XGBoost SHAP Summary (Directional Feature Impact)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "xgboost_shap_summary.png"))
    plt.close()
    print(f"[OK] Saved {os.path.join(PLOTS_DIR, 'xgboost_shap_summary.png')}")


def explain_1d_cnn():
    print("--- Running 1D-CNN Saliency & Gradient Attribution ---")
    setup_matplotlib_style()
    device = get_device()

    data = load_sequence_data(SEQUENCE_DATASET_PATH)
    X_test, y_test = data["X_test"], data["y_test"]

    # Select WebTunnel instances
    wt_indices = np.where(y_test == 1)[0]
    if len(wt_indices) == 0:
        wt_indices = range(min(50, len(X_test)))
    X_wt = X_test[wt_indices]

    model = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model.eval()

    inputs = torch.tensor(X_wt, dtype=torch.float32).permute(0, 2, 1).to(device)
    inputs.requires_grad = True

    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()

    # Saliency: Magnitude of gradients w.r.t input features across sequence positions
    saliency = inputs.grad.abs().cpu().numpy()  # [N, 2, 200]
    mean_saliency = np.mean(saliency, axis=0)   # [2, 200]

    fig, ax = plt.subplots(figsize=(12, 5))
    seq_len = mean_saliency.shape[1]
    pkt_indices = np.arange(1, seq_len + 1)

    ax.plot(pkt_indices, mean_saliency[0], label="Direction Gradient Saliency", color="#2ca02c", lw=2)
    ax.plot(pkt_indices, mean_saliency[1], label="Packet Length Gradient Saliency", color="#d62728", lw=2)
    ax.set_xlabel("Packet Position in Flow Sequence (1 - 200)")
    ax.set_ylabel("Mean Gradient Saliency Magnitude")
    ax.set_title("1D-CNN Temporal Feature Attribution (Packet-Level Saliency Map)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "1d_cnn_saliency_map.png"))
    plt.close()
    print(f"[OK] Saved {os.path.join(PLOTS_DIR, '1d_cnn_saliency_map.png')}")


def explain_transformer():
    print("--- Running Flow-Transformer Input Gradient Saliency Attribution ---")
    setup_matplotlib_style()
    device = get_device()

    data = load_sequence_data(SEQUENCE_DATASET_PATH)
    X_test, y_test = data["X_test"], data["y_test"]

    wt_indices = np.where(y_test == 1)[0]
    if len(wt_indices) == 0:
        wt_indices = range(min(50, len(X_test)))
    X_wt = X_test[wt_indices]

    model = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
    model.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=device))
    model.eval()

    inputs = torch.tensor(X_wt, dtype=torch.float32).to(device)
    inputs.requires_grad = True

    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()

    # Saliency: Gradient magnitude of sequence
    saliency = inputs.grad.abs().cpu().numpy()  # [N, 200, 2]
    mean_saliency = np.mean(saliency, axis=0)   # [200, 2]

    fig, ax = plt.subplots(figsize=(12, 5))
    seq_len = mean_saliency.shape[0]
    pkt_indices = np.arange(1, seq_len + 1)

    ax.plot(pkt_indices, mean_saliency[:, 0], label="Direction Gradient Saliency", color="#1f77b4", lw=2)
    ax.plot(pkt_indices, mean_saliency[:, 1], label="Packet Length Gradient Saliency", color="#ff7f0e", lw=2)
    ax.set_xlabel("Packet Position in Flow Sequence (1 - 200)")
    ax.set_ylabel("Mean Gradient Saliency Magnitude")
    ax.set_title("Flow-Transformer Input Gradient Saliency Attribution across Sequence")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "transformer_attention_map.png"))
    plt.close()
    print(f"[OK] Saved {os.path.join(PLOTS_DIR, 'transformer_attention_map.png')}")


def main():
    explain_xgboost()
    explain_1d_cnn()
    explain_transformer()


if __name__ == "__main__":
    main()
