#!/usr/bin/env python3
"""
Pre- vs. Post-Handshake Classification Benchmark (TLS 1.3 0x17 Cutoff Ablation):
Strips the TLS 1.3 handshake at the client's first application record AFTER its Finished, and
reports how much discriminative power survives.

This is an ABLATION, not a proof. v2.0's docstring claimed it "empirically proved that detection
is independent of TLS metadata"; audit finding F-12 rejected that reading, because the operation
only modifies flows that HAVE a handshake, and the accuracy staying high says nothing about why.
The claim the thesis can make is the assertion in gate G1 plus the lattice derivation, not this
number.
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
from sanitizer import (extract_raw_packets_from_pcap, compute_flow_statistics,
                       build_sequence_tensor, load_manifest_for)
from architectures import WebTunnel1DCNN, WebTunnelTransformer
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

        # AUDIT 4.6 / F-12: v2.0 passed no manifest here, so this ablation ran with 5-tuple
        # demultiplexing switched off entirely. And the cutoff itself was `idx > 3`, a fixed
        # index that ignored TLS content types. The flow builder now cuts at the client's first
        # application record AFTER its Finished (TLS 1.3: the second client 0x17 record).
        manifest = load_manifest_for(p)
        if manifest is None:
            continue
        pkts = extract_raw_packets_from_pcap(p, manifest=manifest,
                                             post_handshake_only=post_handshake)
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

    max_sid = int(np.max(sample_ids))
    train_cutoff = int(max_sid * 0.70)
    val_cutoff = int(max_sid * 0.85)

    train_idx = np.where(sample_ids <= train_cutoff)[0]
    val_idx = np.where((sample_ids > train_cutoff) & (sample_ids <= val_cutoff))[0]
    test_idx = np.where(sample_ids > val_cutoff)[0]

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

    # 3. Flow-Transformer
    train_tf_ds = FlowSequenceDataset(X_seq[train_idx], y_bin[train_idx], channel_first=False)
    val_tf_ds = FlowSequenceDataset(X_seq[val_idx], y_bin[val_idx], channel_first=False)
    test_tf_ds = FlowSequenceDataset(X_seq[test_idx], y_bin[test_idx], channel_first=False)

    train_tf_loader = DataLoader(train_tf_ds, batch_size=32, shuffle=True)
    val_tf_loader = DataLoader(val_tf_ds, batch_size=32, shuffle=False)
    test_tf_loader = DataLoader(test_tf_ds, batch_size=32, shuffle=False)

    tf_model = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
    tf_optimizer = torch.optim.AdamW(tf_model.parameters(), lr=0.0005, weight_decay=1e-4)

    best_tf_loss = float("inf")
    best_tf_state = None
    for epoch in range(1, 30):
        tf_model.train()
        for bx, by in train_tf_loader:
            bx, by = bx.to(device), by.to(device)
            tf_optimizer.zero_grad()
            out = tf_model(bx)
            loss = criterion(out, by)
            loss.backward()
            tf_optimizer.step()

        tf_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_tf_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(tf_model(bx), by).item() * len(by)
        val_loss /= len(val_tf_ds)

        if val_loss < best_tf_loss:
            best_tf_loss = val_loss
            best_tf_state = tf_model.state_dict()

    tf_model.load_state_dict(best_tf_state)
    tf_model.eval()
    tf_probs = []
    with torch.no_grad():
        for bx, _ in test_tf_loader:
            tf_probs.append(tf_model(bx.to(device)).cpu().numpy())
    tf_probs = np.vstack(tf_probs).flatten()
    tf_m = compute_metrics(y_bin[test_idx], tf_probs, threshold=0.5)

    return {"xgb": xgb_m, "cnn": cnn_m, "tf": tf_m}


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
    print(f"Full Flow       - XGB Acc: {res_full['xgb']['accuracy']*100:.2f}% | CNN Acc: {res_full['cnn']['accuracy']*100:.2f}% | TF Acc: {res_full['tf']['accuracy']*100:.2f}%")
    print(f"Post-Handshake  - XGB Acc: {res_post['xgb']['accuracy']*100:.2f}% | CNN Acc: {res_post['cnn']['accuracy']*100:.2f}% | TF Acc: {res_post['tf']['accuracy']*100:.2f}%")

    # Export LaTeX Table
    tex_path = os.path.join(LATEX_TABLES_DIR, "table_handshake_comparison.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Srovnání detekční přesnosti před a po dynamickém oříznutí TLS 1.3 handshaku (Ablace)}" + "\n")
        f.write(r"\label{tab:handshake_comparison}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Režim toku} & \textbf{Model} & \textbf{Přesnost (Accuracy)} & \textbf{Recall} & \textbf{PR-AUC} \\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(f"Kompletní tok (včetně TLS) & XGBoost & {res_full['xgb']['accuracy']*100:.2f}\\% & {res_full['xgb']['recall']*100:.2f}\\% & {res_full['xgb']['pr_auc']:.4f} \\\\\n")
        f.write(f"Kompletní tok (včetně TLS) & 1D-CNN  & {res_full['cnn']['accuracy']*100:.2f}\\% & {res_full['cnn']['recall']*100:.2f}\\% & {res_full['cnn']['pr_auc']:.4f} \\\\\n")
        f.write(f"Kompletní tok (včetně TLS) & Transformer & {res_full['tf']['accuracy']*100:.2f}\\% & {res_full['tf']['recall']*100:.2f}\\% & {res_full['tf']['pr_auc']:.4f} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(f"Pouze Post-Handshake data & XGBoost & {res_post['xgb']['accuracy']*100:.2f}\\% & {res_post['xgb']['recall']*100:.2f}\\% & {res_post['xgb']['pr_auc']:.4f} \\\\\n")
        f.write(f"Pouze Post-Handshake data & 1D-CNN  & {res_post['cnn']['accuracy']*100:.2f}\\% & {res_post['cnn']['recall']*100:.2f}\\% & {res_post['cnn']['pr_auc']:.4f} \\\\\n")
        f.write(f"Pouze Post-Handshake data & Transformer & {res_post['tf']['accuracy']*100:.2f}\\% & {res_post['tf']['recall']*100:.2f}\\% & {res_post['tf']['pr_auc']:.4f} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"\n[OK] Exported {tex_path}")

    # Plot Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = ["XGB (Full)", "XGB (Post-HS)", "1D-CNN (Full)", "1D-CNN (Post-HS)", "Transf. (Full)", "Transf. (Post-HS)"]
    accs = [
        res_full['xgb']['accuracy']*100, res_post['xgb']['accuracy']*100,
        res_full['cnn']['accuracy']*100, res_post['cnn']['accuracy']*100,
        res_full['tf']['accuracy']*100, res_post['tf']['accuracy']*100
    ]
    praucs = [
        res_full['xgb']['pr_auc']*100, res_post['xgb']['pr_auc']*100,
        res_full['cnn']['pr_auc']*100, res_post['cnn']['pr_auc']*100,
        res_full['tf']['pr_auc']*100, res_post['tf']['pr_auc']*100
    ]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, praucs, width, label="PR-AUC (%)", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylim(80, 102)
    ax.set_ylabel("Skóre (%)")
    ax.set_title("Vliv přítomnosti TLS Handshaku na modely (Ablace 0x17 Cutoff)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pre_vs_post_handshake_comparison.png"))
    plt.close()
    print(f"[OK] Saved {os.path.join(PLOTS_DIR, 'pre_vs_post_handshake_comparison.png')}")


if __name__ == "__main__":
    main()
