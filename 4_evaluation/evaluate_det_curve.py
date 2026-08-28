#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import roc_curve

sys.path.append("3_models")
from train_1d_cnn import WebTunnel1DCNN
from train_transformer import WebTunnelTransformer

PROCESSED_DIR = "data/processed"
PLOT_DIR = "4_evaluation/plots"

def generate_logarithmic_det_curve():
    print("=== Generating Logarithmic DET (Detection Error Tradeoff) Curve ===")
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Load Data
    tab_data = np.load(os.path.join(PROCESSED_DIR, "tabular_dataset.npz"), allow_pickle=True)
    seq_data = np.load(os.path.join(PROCESSED_DIR, "sequence_dataset.npz"), allow_pickle=True)
    
    X_test_tab, y_test = tab_data["X_test"], tab_data["y_test"]
    X_test_seq = seq_data["X_test"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Get Predictions
    # XGBoost
    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model("3_models/saved_models/xgboost_baseline.json")
    p_xgb = clf_xgb.predict_proba(X_test_tab)[:, 1]
    
    # 1D-CNN
    model_cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model_cnn.load_state_dict(torch.load("3_models/saved_models/1d_cnn_best.pt", map_location=device))
    model_cnn.eval()
    X_seq_t = torch.from_numpy(np.transpose(X_test_seq, (0, 2, 1))).to(device)
    with torch.no_grad():
        p_cnn = model_cnn(X_seq_t).squeeze(-1).cpu().numpy()
        
    # Transformer
    model_tf = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
    model_tf.load_state_dict(torch.load("3_models/saved_models/transformer_best.pt", map_location=device))
    model_tf.eval()
    X_tf_t = torch.from_numpy(X_test_seq).to(device)
    with torch.no_grad():
        p_tf = model_tf(X_tf_t).squeeze(-1).cpu().numpy()
        
    # 3. Calculate FPR and FNR (1 - TPR)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, p_xgb)
    fnr_xgb = 1.0 - tpr_xgb
    
    fpr_cnn, tpr_cnn, _ = roc_curve(y_test, p_cnn)
    fnr_cnn = 1.0 - tpr_cnn
    
    fpr_tf, tpr_tf, _ = roc_curve(y_test, p_tf)
    fnr_tf = 1.0 - tpr_tf
    
    # Clip zeros to 1e-4 for log scale rendering
    min_val = 1e-3
    fpr_xgb_c = np.clip(fpr_xgb, min_val, 1.0)
    fnr_xgb_c = np.clip(fnr_xgb, min_val, 1.0)
    
    fpr_cnn_c = np.clip(fpr_cnn, min_val, 1.0)
    fnr_cnn_c = np.clip(fnr_cnn, min_val, 1.0)
    
    fpr_tf_c = np.clip(fpr_tf, min_val, 1.0)
    fnr_tf_c = np.clip(fnr_tf, min_val, 1.0)
    
    # 4. Plot Logarithmic DET Curve
    plt.figure(figsize=(10, 7))
    
    plt.plot(fpr_xgb_c, fnr_xgb_c, color="#1f77b4", linewidth=2.5, marker="o", markersize=4, label="XGBoost (Baseline)")
    plt.plot(fpr_cnn_c, fnr_cnn_c, color="#2ca02c", linewidth=2.5, marker="s", markersize=4, label="1D-CNN (Deep Packet)")
    plt.plot(fpr_tf_c, fnr_tf_c, color="#d62728", linewidth=2.5, marker="^", markersize=4, label="Flow-Transformer")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-3, 1.0)
    plt.ylim(1e-3, 1.0)
    
    plt.xlabel("Míra falešných poplachů / False Positive Rate (FPR, log měřítko)", fontsize=12, fontweight="bold")
    plt.ylabel("Míra minutí cíle / Miss Rate (FNR = 1 - Recall, log měřítko)", fontsize=12, fontweight="bold")
    plt.title("Logaritmická DET křivka (Detection Error Tradeoff) pro Low-FPR režim", fontsize=13, fontweight="bold")
    
    # Highlight operational ISP low-FPR zone
    plt.axvspan(1e-3, 5e-3, color='gray', alpha=0.15, label=r'Kritická ISP Low-FPR zóna (FPR $\leq 5 \times 10^{-3}$)')
    
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "det_curve_logarithmic.png"), dpi=300)
    plt.close()
    print(f"[OK] Saved {PLOT_DIR}/det_curve_logarithmic.png")

if __name__ == "__main__":
    generate_logarithmic_det_curve()
