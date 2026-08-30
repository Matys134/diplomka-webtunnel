#!/usr/bin/env python3
"""
Pre- vs. Post-Handshake Classification Benchmark (TLS 1.3 0x17 Cutoff Ablation):
Dynamically strips initial TLS handshakes at the first Application Data record (ContentType == 0x17),
empirically proving that detection is independent of TLS metadata and driven by Tor cell quantization.
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import (
    RAW_PCAP_DIR,
    PLOTS_DIR,
    LATEX_TABLES_DIR,
    CLASSES,
    RANDOM_SEED,
    set_global_seed,
    setup_matplotlib_style
)
from sanitizer import extract_raw_packets_from_pcap, compute_flow_statistics, build_sequence_tensor
from architectures import WebTunnel1DCNN
from utils import FlowSequenceDataset, BinaryFocalLoss, compute_metrics, get_device


def load_dataset_variants(post_handshake: bool = False):
    pcap_files = sorted(glob.glob(os.path.join(RAW_PCAP_DIR, "*.pcap")))
    X_tab, X_seq, y_bin, sample_ids = [], [], [], []

    for p in pcap_files:
        base = os.path.basename(p)
        label = None
        for k in CLASSES:
            if base.startswith(k):
                label = 1 if k == "webtunnel" else 0
                break
        if label is None:
            continue

        pkts = extract_raw_packets_from_pcap(p, post_handshake_only=post_handshake)
        if len(pkts) < 3:
            continue

        try:
            sid = int(os.path.splitext(base)[0].split("_")[-1])
        except Exception:
            sid = -1

        X_tab.append(compute_flow_statistics(pkts))
        X_seq.append(build_sequence_tensor(pkts, max_seq_len=200))
        y_bin.append(label)
        sample_ids.append(sid)

    return (
        np.array(X_tab, dtype=np.float32),
        np.array(X_seq, dtype=np.float32),
        np.array(y_bin, dtype=np.int64),
        np.array(sample_ids)
    )


def train_eval_variant(X_tab, X_seq, y_bin, sample_ids, desc="Full"):
    set_global_seed(RANDOM_SEED)
    device = get_device()

    train_idx = np.where(sample_ids <= 70)[0]
    val_idx = np.where((sample_ids > 70) & (sample_ids <= 85))[0]
    test_idx = np.where(sample_ids > 85)[0]

    # 1. XGBoost
    pos_c = int(np.sum(y_bin[train_idx] == 1))
    neg_c = int(np.sum(y_bin[train_idx] == 0))
    spw = float(neg_c / max(1, pos_c))

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=30
    )
    clf.fit(X_tab[train_idx], y_bin[train_idx], eval_set=[(X_tab[val_idx], y_bin[val_idx])], verbose=False)
    xgb_probs = clf.predict_proba(X_tab[test_idx])[:, 1]
    xgb_m = compute_metrics(y_bin[test_idx], xgb_probs, threshold=0.5)

    # 2. 1D-CNN
    train_ds = FlowSequenceDataset(X_seq[train_idx], y_bin[train_idx], channel_first=True)
    val_ds = FlowSequenceDataset(X_seq[val_idx], y_bin[val_idx], channel_first=True)
    test_ds = FlowSequenceDataset(X_seq[test_idx], y_bin[test_idx], channel_first=True)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, 30):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * len(by)
        val_loss /= len(val_ds)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    model.eval()
    cnn_probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            cnn_probs.append(model(bx.to(device)).cpu().numpy())
    cnn_probs = np.vstack(cnn_probs).flatten()
    cnn_m = compute_metrics(y_bin[test_idx], cnn_probs, threshold=0.5)

    return {"xgb": xgb_m, "cnn": cnn_m}


def main():
    setup_matplotlib_style()

    print("=== Loading Full-Flow (Standard) Dataset ===")
    X_tab_full, X_seq_full, y_full, sids_full = load_dataset_variants(post_handshake=False)
    print(f"Full dataset: {len(y_full)} samples.")

    print("\n=== Loading Post-Handshake-Only Dataset (Handshake Stripped) ===")
    X_tab_post, X_seq_post, y_post, sids_post = load_dataset_variants(post_handshake=True)
    print(f"Post-handshake dataset: {len(y_post)} samples.")

    res_full = train_eval_variant(X_tab_full, X_seq_full, y_full, sids_full, desc="Full")
    res_post = train_eval_variant(X_tab_post, X_seq_post, y_post, sids_post, desc="Post-Handshake")

    print("\n--- Handshake Comparison Summary ---")
    print(f"Full Flow       - XGBoost Acc: {res_full['xgb']['accuracy']*100:.2f}%, PR-AUC: {res_full['xgb']['pr_auc']:.4f} | 1D-CNN Acc: {res_full['cnn']['accuracy']*100:.2f}%, PR-AUC: {res_full['cnn']['pr_auc']:.4f}")
    print(f"Post-Handshake  - XGBoost Acc: {res_post['xgb']['accuracy']*100:.2f}%, PR-AUC: {res_post['xgb']['pr_auc']:.4f} | 1D-CNN Acc: {res_post['cnn']['accuracy']*100:.2f}%, PR-AUC: {res_post['cnn']['pr_auc']:.4f}")

    # Export LaTeX Table
    tex_path = os.path.join(LATEX_TABLES_DIR, "table_handshake_comparison.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Srovnání detekční přesnosti před a po dynamickém oříznutí TLS 1.3 handshaku}" + "\n")
        f.write(r"\label{tab:handshake_comparison}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Režim toku} & \textbf{Model} & \textbf{Přesnost (Accuracy)} & \textbf{Recall} & \textbf{PR-AUC} \\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(f"Kompletní tok (včetně TLS) & XGBoost & {res_full['xgb']['accuracy']*100:.2f}\\% & {res_full['xgb']['recall']*100:.2f}\\% & {res_full['xgb']['pr_auc']:.4f} \\\\\n")
        f.write(f"Kompletní tok (včetně TLS) & 1D-CNN  & {res_full['cnn']['accuracy']*100:.2f}\\% & {res_full['cnn']['recall']*100:.2f}\\% & {res_full['cnn']['pr_auc']:.4f} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(f"Pouze Post-Handshake data & XGBoost & {res_post['xgb']['accuracy']*100:.2f}\\% & {res_post['xgb']['recall']*100:.2f}\\% & {res_post['xgb']['pr_auc']:.4f} \\\\\n")
        f.write(f"Pouze Post-Handshake data & 1D-CNN  & {res_post['cnn']['accuracy']*100:.2f}\\% & {res_post['cnn']['recall']*100:.2f}\\% & {res_post['cnn']['pr_auc']:.4f} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"\n[OK] Exported {tex_path}")

    # Plot Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    categories = ["XGBoost (Full)", "XGBoost (Post-HS)", "1D-CNN (Full)", "1D-CNN (Post-HS)"]
    accs = [res_full['xgb']['accuracy']*100, res_post['xgb']['accuracy']*100, res_full['cnn']['accuracy']*100, res_post['cnn']['accuracy']*100]
    praucs = [res_full['xgb']['pr_auc']*100, res_post['xgb']['pr_auc']*100, res_full['cnn']['pr_auc']*100, res_post['cnn']['pr_auc']*100]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, praucs, width, label="PR-AUC (%)", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=10)
    ax.set_ylim(80, 102)
    ax.set_ylabel("Skóre (%)")
    ax.set_title("Vliv přítomnosti TLS Handshaku na klasifikaci (Ablace 0x17 Cutoff)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pre_vs_post_handshake_comparison.png"))
    plt.close()
    print(f"[OK] Saved {os.path.join(PLOTS_DIR, 'pre_vs_post_handshake_comparison.png')}")


if __name__ == "__main__":
    main()
