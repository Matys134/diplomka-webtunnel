#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import torch
from train_1d_cnn import WebTunnel1DCNN

PROCESSED_DIR = "data/processed"
MODEL_DIR = "3_models/saved_models"
PLOT_DIR = "4_evaluation/plots"

def explain_xgboost():
    print("--- Running XGBoost Feature Attribution ---")
    data = np.load(os.path.join(PROCESSED_DIR, "tabular_dataset.npz"), allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = list(data["feature_names"])
    
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "xgboost_baseline.json"))
    
    # Feature Importances (Weight & Gain)
    booster = model.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    
    # Map f0, f1 to actual feature names
    mapped_scores = {}
    for k, v in score_gain.items():
        idx = int(k.replace("f", ""))
        mapped_scores[feature_names[idx]] = v
        
    sorted_feats = sorted(mapped_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    plt.figure(figsize=(10, 6))
    f_names = [x[0] for x in sorted_feats][::-1]
    f_gains = [x[1] for x in sorted_feats][::-1]
    plt.barh(f_names, f_gains, color="#1f77b4")
    plt.title("XGBoost Feature Importance (Information Gain)", fontsize=14, fontweight="bold")
    plt.xlabel("Average Gain in Purity", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "xgboost_feature_importance.png"), dpi=300)
    plt.close()
    print(f"[OK] Saved {PLOT_DIR}/xgboost_feature_importance.png")

def explain_1d_cnn_saliency():
    print("--- Running 1D-CNN Saliency & Gradient Attribution ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = np.load(os.path.join(PROCESSED_DIR, "sequence_dataset.npz"), allow_pickle=True)
    X_test = np.transpose(data["X_test"], (0, 2, 1))  # (B, 2, 200)
    y_test = data["y_test"]
    
    wt_indices = np.where(y_test == 1)[0]
    if len(wt_indices) == 0:
        return
        
    model = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "1d_cnn_best.pt"), map_location=device))
    model.eval()
    
    # Compute Vanilla Gradient Saliency for WebTunnel samples
    sample_tensor = torch.from_numpy(X_test[wt_indices]).to(device)
    sample_tensor.requires_grad = True
    
    preds = model(sample_tensor).squeeze(-1)
    loss = preds.sum()
    loss.backward()
    
    # Saliency magnitude across packet sequence
    grads = sample_tensor.grad.cpu().numpy()  # (N, 2, 200)
    length_grads = np.abs(grads[:, 0, :])     # Channel 0: packet size gradients
    mean_saliency = np.mean(length_grads, axis=0)
    
    # Plot Packet Saliency Map
    plt.figure(figsize=(12, 5))
    packet_indices = np.arange(1, len(mean_saliency) + 1)
    plt.plot(packet_indices, mean_saliency, color="#2ca02c", linewidth=2.0, label="1D-CNN Gradient Sensitivity")
    plt.fill_between(packet_indices, 0, mean_saliency, alpha=0.3, color="#2ca02c")
    plt.title("1D-CNN Saliency Map: Which Packets Trigger WebTunnel Detection?", fontsize=14, fontweight="bold")
    plt.xlabel("Packet Sequence Index in Flow (Packets 1-200)", fontsize=12)
    plt.ylabel("Gradient Attribution (Importance)", fontsize=12)
    plt.axvspan(1, 15, color="red", alpha=0.15, label="Tor Circuit Setup Burst (CREATE2/CREATED2)")
    plt.legend(fontsize=11)
    plt.xlim(1, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "1d_cnn_saliency_map.png"), dpi=300)
    plt.close()
    print(f"[OK] Saved {PLOT_DIR}/1d_cnn_saliency_map.png")

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    explain_xgboost()
    explain_1d_cnn_saliency()

if __name__ == "__main__":
    main()
