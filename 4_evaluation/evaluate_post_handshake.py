#!/usr/bin/env python3
import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

sys.path.append("2_data_pipeline")
sys.path.append("3_models")
from sanitizer import extract_raw_packets_from_pcap, compute_flow_statistics, build_sequence_tensor, FEATURE_NAMES
from train_1d_cnn import WebTunnel1DCNN, FocalLoss

RAW_PCAP_DIR = "data/raw_pcap"
PLOT_DIR = "4_evaluation/plots"
TABLE_DIR = "0_thesis_text/tables"
EVAL_DIR = "4_evaluation"

CLASS_MAPPING = {
    "webtunnel": 1,
    "websocket_ticker": 0,
    "websocket_chat": 0,
    "video_streaming": 0,
    "web_assets": 0,
}

def load_dataset_variants(post_handshake: bool = False):
    pcap_files = sorted(glob.glob(os.path.join(RAW_PCAP_DIR, "*.pcap")))
    X_tab, X_seq, y_bin = [], [], []
    
    for p in pcap_files:
        base = os.path.basename(p)
        label = None
        for k in CLASS_MAPPING.keys():
            if base.startswith(k):
                label = CLASS_MAPPING[k]
                break
        if label is None:
            continue
            
        pkts = extract_raw_packets_from_pcap(p, post_handshake_only=post_handshake)
        if len(pkts) < 3:
            continue
            
        X_tab.append(compute_flow_statistics(pkts))
        X_seq.append(build_sequence_tensor(pkts, max_seq_len=200))
        y_bin.append(label)
        
    X_tab = np.array(X_tab, dtype=np.float32)
    X_seq = np.array(X_seq, dtype=np.float32)
    y_bin = np.array(y_bin, dtype=np.int64)
    return X_tab, X_seq, y_bin

def train_and_eval_models(X_tab, X_seq, y_bin):
    # Split
    indices = np.arange(len(y_bin))
    tr_idx, te_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_bin)
    
    # 1. XGBoost
    scale_pos_weight = float(np.sum(y_bin[tr_idx] == 0) / max(np.sum(y_bin[tr_idx] == 1), 1))
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03,
        scale_pos_weight=scale_pos_weight, random_state=42
    )
    clf.fit(X_tab[tr_idx], y_bin[tr_idx], verbose=False)
    xgb_probs = clf.predict_proba(X_tab[te_idx])[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    
    xgb_metrics = {
        "acc": accuracy_score(y_bin[te_idx], xgb_preds),
        "prec": precision_score(y_bin[te_idx], xgb_preds, zero_division=0),
        "rec": recall_score(y_bin[te_idx], xgb_preds, zero_division=0),
        "f1": f1_score(y_bin[te_idx], xgb_preds, zero_division=0),
        "pr_auc": average_precision_score(y_bin[te_idx], xgb_probs),
        "roc_auc": roc_auc_score(y_bin[te_idx], xgb_probs)
    }
    
    # 2. 1D-CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_seq_t = np.transpose(X_seq, (0, 2, 1))
    X_tr_t = torch.from_numpy(X_seq_t[tr_idx])
    y_tr_t = torch.from_numpy(y_bin[tr_idx])
    X_te_t = torch.from_numpy(X_seq_t[te_idx])
    y_te_t = torch.from_numpy(y_bin[te_idx])
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=32, shuffle=False)
    
    cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    crit = FocalLoss(alpha=0.25, gamma=2.0)
    opt = torch.optim.AdamW(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(30):
        cnn.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            preds = cnn(bx).squeeze(-1)
            loss = crit(preds, by)
            loss.backward()
            opt.step()
            
    cnn.eval()
    all_probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            p = cnn(bx).squeeze(-1).cpu().numpy()
            all_probs.extend(p)
            
    cnn_probs = np.array(all_probs)
    cnn_preds = (cnn_probs >= 0.5).astype(int)
    
    cnn_metrics = {
        "acc": accuracy_score(y_bin[te_idx], cnn_preds),
        "prec": precision_score(y_bin[te_idx], cnn_preds, zero_division=0),
        "rec": recall_score(y_bin[te_idx], cnn_preds, zero_division=0),
        "f1": f1_score(y_bin[te_idx], cnn_preds, zero_division=0),
        "pr_auc": average_precision_score(y_bin[te_idx], cnn_probs),
        "roc_auc": roc_auc_score(y_bin[te_idx], cnn_probs)
    }
    
    return xgb_metrics, cnn_metrics

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    
    print("=== Loading Full-Flow (Standard) Dataset ===")
    X_tab_full, X_seq_full, y_full = load_dataset_variants(post_handshake=False)
    print(f"Full dataset: {len(y_full)} samples.")
    xgb_full, cnn_full = train_and_eval_models(X_tab_full, X_seq_full, y_full)
    
    print("\n=== Loading Post-Handshake-Only Dataset (Handshake Stripped) ===")
    X_tab_post, X_seq_post, y_post = load_dataset_variants(post_handshake=True)
    print(f"Post-handshake dataset: {len(y_post)} samples.")
    xgb_post, cnn_post = train_and_eval_models(X_tab_post, X_seq_post, y_post)
    
    print("\n--- Handshake Comparison Summary ---")
    print(f"Full Flow       - XGBoost Acc: {xgb_full['acc']*100:.2f}%, PR-AUC: {xgb_full['pr_auc']:.4f} | 1D-CNN Acc: {cnn_full['acc']*100:.2f}%, PR-AUC: {cnn_full['pr_auc']:.4f}")
    print(f"Post-Handshake  - XGBoost Acc: {xgb_post['acc']*100:.2f}%, PR-AUC: {xgb_post['pr_auc']:.4f} | 1D-CNN Acc: {cnn_post['acc']*100:.2f}%, PR-AUC: {cnn_post['pr_auc']:.4f}")
    
    # Generate LaTeX Table
    tex = r"""\begin{table}[htbp]
\centering
\caption{Vliv přítomnosti TLS handshaku na detekční schopnost klasifikátorů}
\label{tab:handshake_comparison}
\begin{tabular}{lcccc}
\hline
\textbf{Scénář inspekce} & \textbf{Model} & \textbf{Accuracy} & \textbf{PR-AUC} & \textbf{ROC-AUC} \\
\hline
Kompletní tok (včetně TLS Handshaku) & XGBoost & """ + f"{xgb_full['acc']*100:.2f}\\% & {xgb_full['pr_auc']:.4f} & {xgb_full['roc_auc']:.4f} \\\\\n" + \
r"""Kompletní tok (včetně TLS Handshaku) & 1D-CNN & """ + f"{cnn_full['acc']*100:.2f}\\% & {cnn_full['pr_auc']:.4f} & {cnn_full['roc_auc']:.4f} \\\\\n" + \
r"""\hline
Čistě Post-Handshake (bez TLS Handshaku) & XGBoost & """ + f"{xgb_post['acc']*100:.2f}\\% & {xgb_post['pr_auc']:.4f} & {xgb_post['roc_auc']:.4f} \\\\\n" + \
r"""Čistě Post-Handshake (bez TLS Handshaku) & 1D-CNN & """ + f"{cnn_post['acc']*100:.2f}\\% & {cnn_post['pr_auc']:.4f} & {cnn_post['roc_auc']:.4f} \\\\\n" + \
r"""\hline
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLE_DIR, "table_handshake_comparison.tex"), "w") as f:
        f.write(tex)
    print(f"\n[OK] Exported {TABLE_DIR}/table_handshake_comparison.tex")
    
    # Generate Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    categories = ["XGBoost Accuracy", "XGBoost PR-AUC", "1D-CNN Accuracy", "1D-CNN PR-AUC"]
    full_vals = [xgb_full['acc']*100, xgb_full['pr_auc']*100, cnn_full['acc']*100, cnn_full['pr_auc']*100]
    post_vals = [xgb_post['acc']*100, xgb_post['pr_auc']*100, cnn_post['acc']*100, cnn_post['pr_auc']*100]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, full_vals, width, label='Kompletní tok (vč. TLS handshaku)', color='#1f77b4')
    plt.bar(x + width/2, post_vals, width, label='Čistě Post-Handshake provoz', color='#2ca02c')
    
    plt.ylabel('Skóre (%)', fontsize=12)
    plt.title('Srovnání detekce: Kompletní tok vs. Čistě Post-Handshake aplikační data', fontsize=14, fontweight='bold')
    plt.xticks(x, categories, fontsize=11)
    plt.ylim(0, 115)
    for i in range(len(categories)):
        plt.text(x[i] - width/2, full_vals[i] + 1.5, f"{full_vals[i]:.1f}%", ha='center', fontsize=10, fontweight='bold')
        plt.text(x[i] + width/2, post_vals[i] + 1.5, f"{post_vals[i]:.1f}%", ha='center', fontsize=10, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pre_vs_post_handshake_comparison.png"), dpi=300)
    plt.close()
    print(f"[OK] Saved {PLOT_DIR}/pre_vs_post_handshake_comparison.png")

if __name__ == "__main__":
    main()
