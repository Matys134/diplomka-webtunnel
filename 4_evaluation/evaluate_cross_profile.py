#!/usr/bin/env python3
import os
import sys
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from concurrent.futures import ProcessPoolExecutor

sys.path.append("2_data_pipeline")
sys.path.append("3_models")
from sanitizer import extract_raw_packets_from_pcap, compute_flow_statistics, build_sequence_tensor, FEATURE_NAMES
from train_1d_cnn import WebTunnel1DCNN

RAW_PCAP_DIR = "data/raw_pcap"
TABLE_DIR = "0_thesis_text/tables"
PLOT_DIR = "4_evaluation/plots"

PROFILES = ["broadband", "lte", "lossy"]
CLASS_MAPPING = {
    "webtunnel": 1,
    "websocket_ticker": 0,
    "websocket_chat": 0,
    "video_streaming": 0,
    "web_assets": 0,
}

def parse_pcap(f):
    basename = os.path.basename(f)
    # Detect class and profile
    matched_class = None
    for c in CLASS_MAPPING.keys():
        if basename.startswith(c):
            matched_class = c
            break
            
    matched_profile = None
    for p in PROFILES:
        if f"_{p}_" in basename:
            matched_profile = p
            break
            
    if not matched_class or not matched_profile:
        return None
        
    pkts = extract_raw_packets_from_pcap(f)
    if len(pkts) < 3:
        return None
        
    tab = compute_flow_statistics(pkts)
    seq = build_sequence_tensor(pkts, max_seq_len=200)
    label = CLASS_MAPPING[matched_class]
    
    return {
        "profile": matched_profile,
        "class": matched_class,
        "label": label,
        "tab": tab,
        "seq": seq
    }

def train_and_eval_1d_cnn(X_train_seq, y_train_seq, test_sets, device, epochs=25):
    # Train 1D-CNN on X_train_seq
    model = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    # Transpose for PyTorch (N, 2, 200)
    X_train_t = torch.from_numpy(np.transpose(X_train_seq, (0, 2, 1))).float()
    y_train_t = torch.from_numpy(y_train_seq).float()
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    results = {}
    with torch.no_grad():
        for prof_name, (X_test_seq, y_test_seq) in test_sets.items():
            X_t = torch.from_numpy(np.transpose(X_test_seq, (0, 2, 1))).float().to(device)
            probs = model(X_t).squeeze(-1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            results[prof_name] = {
                "acc": accuracy_score(y_test_seq, preds),
                "prec": precision_score(y_test_seq, preds, zero_division=0),
                "rec": recall_score(y_test_seq, preds, zero_division=0),
                "f1": f1_score(y_test_seq, preds, zero_division=0),
                "pr_auc": average_precision_score(y_test_seq, probs),
                "roc_auc": roc_auc_score(y_test_seq, probs)
            }
    return results

def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pcap_files = glob.glob(os.path.join(RAW_PCAP_DIR, "*.pcap"))
    print(f"=== 1. Loading & Sanitizing PCAPs by Network Profile ({len(pcap_files)} total) ===")
    
    with ProcessPoolExecutor() as ex:
        parsed = list(ex.map(parse_pcap, pcap_files))
    parsed = [p for p in parsed if p is not None]
    
    # Group by profile
    profile_data = {p: [] for p in PROFILES}
    for item in parsed:
        profile_data[item["profile"]].append(item)
        
    for p in PROFILES:
        print(f"  Profile '{p}': {len(profile_data[p])} valid flows")
        
    # Prepare profile tensors
    prof_tensors = {}
    for p in PROFILES:
        items = profile_data[p]
        X_tab = np.array([it["tab"] for it in items], dtype=np.float32)
        X_seq = np.array([it["seq"] for it in items], dtype=np.float32)
        y = np.array([it["label"] for it in items], dtype=np.int64)
        prof_tensors[p] = {"X_tab": X_tab, "X_seq": X_seq, "y": y}
        
    print("\n=== 2. Cross-Profile Experiment: Train on Broadband -> Test on LTE & Lossy ===")
    
    # Train split on Broadband (80% train, 20% in-domain test)
    n_bb = len(prof_tensors["broadband"]["y"])
    np.random.seed(42)
    perm = np.random.permutation(n_bb)
    split_idx = int(0.8 * n_bb)
    
    train_idx = perm[:split_idx]
    bb_test_idx = perm[split_idx:]
    
    X_train_tab = prof_tensors["broadband"]["X_tab"][train_idx]
    y_train_tab = prof_tensors["broadband"]["y"][train_idx]
    X_train_seq = prof_tensors["broadband"]["X_seq"][train_idx]
    y_train_seq = prof_tensors["broadband"]["y"][train_idx]
    
    test_sets_tab = {
        "Broadband (In-Domain)": (prof_tensors["broadband"]["X_tab"][bb_test_idx], prof_tensors["broadband"]["y"][bb_test_idx]),
        "4G/LTE (Domain Shift)": (prof_tensors["lte"]["X_tab"], prof_tensors["lte"]["y"]),
        "Lossy WAN (Domain Shift)": (prof_tensors["lossy"]["X_tab"], prof_tensors["lossy"]["y"])
    }
    
    test_sets_seq = {
        "Broadband (In-Domain)": (prof_tensors["broadband"]["X_seq"][bb_test_idx], prof_tensors["broadband"]["y"][bb_test_idx]),
        "4G/LTE (Domain Shift)": (prof_tensors["lte"]["X_seq"], prof_tensors["lte"]["y"]),
        "Lossy WAN (Domain Shift)": (prof_tensors["lossy"]["X_seq"], prof_tensors["lossy"]["y"])
    }
    
    # 1. Train & Test XGBoost
    clf_xgb = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        scale_pos_weight=3.33, random_state=42, n_jobs=-1
    )
    clf_xgb.fit(X_train_tab, y_train_tab)
    
    xgb_results = {}
    for name, (Xt, yt) in test_sets_tab.items():
        probs = clf_xgb.predict_proba(Xt)[:, 1]
        preds = (probs >= 0.5).astype(int)
        xgb_results[name] = {
            "acc": accuracy_score(yt, preds),
            "prec": precision_score(yt, preds, zero_division=0),
            "rec": recall_score(yt, preds, zero_division=0),
            "f1": f1_score(yt, preds, zero_division=0),
            "pr_auc": average_precision_score(yt, probs),
            "roc_auc": roc_auc_score(yt, probs)
        }
        
    # 2. Train & Test 1D-CNN
    cnn_results = train_and_eval_1d_cnn(X_train_seq, y_train_seq, test_sets_seq, device)
    
    print("\n" + "="*80)
    print("       CROSS-PROFILE DOMAIN GENERALIZATION RESULTS (Train: Broadband)")
    print("="*80)
    print(f"{'Testovací profil':<26} | {'Model':<12} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'PR-AUC':<7}")
    print("-"*80)
    for name in test_sets_tab.keys():
        r_xgb = xgb_results[name]
        r_cnn = cnn_results[name]
        print(f"{name:<26} | {'XGBoost':<12} | {r_xgb['acc']*100:8.1f}% | {r_xgb['prec']*100:8.1f}% | {r_xgb['rec']*100:8.1f}% | {r_xgb['pr_auc']:7.4f}")
        print(f"{'':<26} | {'1D-CNN':<12} | {r_cnn['acc']*100:8.1f}% | {r_cnn['prec']*100:8.1f}% | {r_cnn['rec']*100:8.1f}% | {r_cnn['pr_auc']:7.4f}")
        print("-"*80)
        
    # Export LaTeX Table
    tex = r"""\begin{table}[htbp]
\centering
\caption{Cross-Profile Domain Generalization: Modely trénované pouze na profilu Broadband testované v odlišných síťových podmínkách}
\label{tab:cross_profile_generalization}
\begin{tabular}{lcccccc}
\hline
\textbf{Testovací prostředí} & \textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{PR-AUC} \\
\hline
"""
    for name in test_sets_tab.keys():
        rx = xgb_results[name]
        rc = cnn_results[name]
        tex += f"\\multirow{{2}}{{*}}{{{name}}} & XGBoost & {rx['acc']*100:.1f}\\% & {rx['prec']*100:.1f}\\% & {rx['rec']*100:.1f}\\% & {rx['f1']*100:.1f}\\% & {rx['pr_auc']:.4f} \\\\\n"
        tex += f" & 1D-CNN & {rc['acc']*100:.1f}\\% & {rc['prec']*100:.1f}\\% & {rc['rec']*100:.1f}\\% & {rc['f1']*100:.1f}\\% & {rc['pr_auc']:.4f} \\\\\n"
        tex += r"\hline" + "\n"
        
    tex += r"""\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLE_DIR, "table_cross_profile_generalization.tex"), "w") as f:
        f.write(tex)
    print(f"\n[OK] Exported {TABLE_DIR}/table_cross_profile_generalization.tex")

if __name__ == "__main__":
    main()
