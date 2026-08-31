#!/usr/bin/env python3
"""
Cross-Profile Domain Generalization Experiment:
Evaluates classifier robustness against network domain shifts by training on Gigabit Broadband
and evaluating out-of-domain on 4G/LTE (high jitter) and Lossy WAN (2% packet loss).
Applies strict Session-Stratified splitting (Sample ID <= 70 Train, > 70 Test).
"""
import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import (
    RAW_PCAP_DIR,
    LATEX_TABLES_DIR,
    NETEM_PROFILES,
    CLASSES,
    RANDOM_SEED,
    set_global_seed
)
from sanitizer import extract_raw_packets_from_pcap, compute_flow_statistics, build_sequence_tensor
from architectures import WebTunnel1DCNN
from utils import FlowSequenceDataset, BinaryFocalLoss, compute_metrics, get_device


def parse_pcap(f):
    basename = os.path.basename(f)
    matched_class = None
    for c in CLASSES:
        if basename.startswith(c):
            matched_class = c
            break

    matched_profile = None
    for p in NETEM_PROFILES:
        if f"_{p}_" in basename:
            matched_profile = p
            break

    if not matched_class or not matched_profile:
        return None

    try:
        sample_id = int(os.path.splitext(basename)[0].split("_")[-1])
    except Exception:
        sample_id = -1

    pkts = extract_raw_packets_from_pcap(f)
    if len(pkts) < 3:
        return None

    tab = compute_flow_statistics(pkts)
    seq = build_sequence_tensor(pkts, max_seq_len=200)
    label = 1 if matched_class == "webtunnel" else 0

    return {
        "profile": matched_profile,
        "class": matched_class,
        "sample_id": sample_id,
        "tab": tab,
        "seq": seq,
        "label": label
    }


def main():
    set_global_seed(RANDOM_SEED)
    device = get_device()

    all_pcaps = sorted(glob.glob(os.path.join(RAW_PCAP_DIR, "*.pcap")))
    print(f"=== 1. Loading & Sanitizing PCAPs by Network Profile ({len(all_pcaps)} total) ===")

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(parse_pcap, all_pcaps))

    valid = [r for r in results if r is not None]
    by_profile = {p: [r for r in valid if r["profile"] == p] for p in NETEM_PROFILES}

    for p in NETEM_PROFILES:
        print(f"  Profile '{p}': {len(by_profile[p])} valid flows")

    max_sid = max(s["sample_id"] for s in valid)
    train_cutoff = int(max_sid * 0.70)

    # 1. Train on Broadband (Session-Stratified: ID <= train_cutoff Train, ID > train_cutoff In-Domain Test)
    bb_data = by_profile["broadband"]
    train_samples = [s for s in bb_data if s["sample_id"] <= train_cutoff]
    val_samples = [s for s in bb_data if s["sample_id"] > train_cutoff]

    print(f"  Broadband split: Train={len(train_samples)} flows, Test={len(val_samples)} flows (Cutoff: {train_cutoff})")

    X_train_tab = np.array([s["tab"] for s in train_samples], dtype=np.float32)
    y_train = np.array([s["label"] for s in train_samples], dtype=np.int64)
    X_train_seq = np.array([s["seq"] for s in train_samples], dtype=np.float32)

    X_val_tab = np.array([s["tab"] for s in val_samples], dtype=np.float32)
    y_val = np.array([s["label"] for s in val_samples], dtype=np.int64)
    X_val_seq = np.array([s["seq"] for s in val_samples], dtype=np.float32)

    # Train XGBoost
    pos_c = int(np.sum(y_train == 1))
    neg_c = int(np.sum(y_train == 0))
    spw = float(neg_c / max(1, pos_c))

    clf_xgb = xgb.XGBClassifier(
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
    clf_xgb.fit(X_train_tab, y_train, eval_set=[(X_val_tab, y_val)], verbose=False)

    # Train 1D-CNN
    train_ds = FlowSequenceDataset(X_train_seq, y_train, channel_first=True)
    val_ds = FlowSequenceDataset(X_val_seq, y_val, channel_first=True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model_cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model_cnn.parameters(), lr=0.001, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, 30):
        model_cnn.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model_cnn(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model_cnn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model_cnn(bx), by).item() * len(by)
        val_loss /= len(val_ds)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model_cnn.state_dict()

    model_cnn.load_state_dict(best_state)
    model_cnn.eval()

    # 2. Evaluate on All Profiles
    print("\n=== 2. Cross-Profile Experiment: Train on Broadband -> Test on LTE & Lossy ===")
    results = {}
    for p in NETEM_PROFILES:
        samples = by_profile[p]
        if p == "broadband":
            samples = val_samples
        else:
            # For LTE & Lossy, evaluate on held-out test sessions (ID > train_cutoff) for strict comparability
            samples = [s for s in samples if s["sample_id"] > train_cutoff]

        X_p_tab = np.array([s["tab"] for s in samples], dtype=np.float32)
        X_p_seq = np.array([s["seq"] for s in samples], dtype=np.float32)
        y_p = np.array([s["label"] for s in samples], dtype=np.int64)

        # XGBoost Eval
        probs_xgb = clf_xgb.predict_proba(X_p_tab)[:, 1]
        m_xgb = compute_metrics(y_p, probs_xgb, threshold=0.5)

        # 1D-CNN Eval
        p_ds = FlowSequenceDataset(X_p_seq, y_p, channel_first=True)
        p_loader = DataLoader(p_ds, batch_size=32, shuffle=False)
        probs_cnn = []
        with torch.no_grad():
            for bx, _ in p_loader:
                probs_cnn.append(model_cnn(bx.to(device)).cpu().numpy())
        probs_cnn = np.vstack(probs_cnn).flatten()
        m_cnn = compute_metrics(y_p, probs_cnn, threshold=0.5)

        results[p] = {"xgb": m_xgb, "cnn": m_cnn}

    print("\n" + "="*80)
    print("       CROSS-PROFILE DOMAIN GENERALIZATION RESULTS (Train: Broadband)")
    print("="*80)
    print(f"{'Testovací profil':<26} | {'Model':<12} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'PR-AUC':<6}")
    print("-" * 80)
    for p_name, disp in [("broadband", "Broadband (In-Domain)"), ("lte", "4G/LTE (Domain Shift)"), ("lossy", "Lossy WAN (Domain Shift)")]:
        m_x = results[p_name]["xgb"]
        m_c = results[p_name]["cnn"]
        print(f"{disp:<26} | {'XGBoost':<12} | {m_x['accuracy']*100:>8.1f}% | {m_x['precision']*100:>8.1f}% | {m_x['recall']*100:>8.1f}% | {m_x['pr_auc']:>7.4f}")
        print(f"{'':<26} | {'1D-CNN':<12} | {m_c['accuracy']*100:>8.1f}% | {m_c['precision']*100:>8.1f}% | {m_c['recall']*100:>8.1f}% | {m_c['pr_auc']:>7.4f}")
        print("-" * 80)

    # Export LaTeX Table
    tex_path = os.path.join(LATEX_TABLES_DIR, "table_cross_profile_generalization.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Doménová generalizace klasifikátorů napříč síťovými profily (Trénováno na profilu Broadband)}" + "\n")
        f.write(r"\label{tab:cross_profile_generalization}" + "\n")
        f.write(r"\begin{tabular}{lccccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Testovací profil} & \textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{PR-AUC} \\" + "\n")
        f.write(r"\hline" + "\n")
        for p_name, disp in [("broadband", "Broadband (In-Domain)"), ("lte", "4G/LTE (Posun jitteru)"), ("lossy", "Lossy WAN (2\\% ztrátovost)")]:
            m_x = results[p_name]["xgb"]
            m_c = results[p_name]["cnn"]
            f.write(f"{disp} & XGBoost & {m_x['accuracy']*100:.1f}\\% & {m_x['precision']*100:.1f}\\% & {m_x['recall']*100:.1f}\\% & {m_x['pr_auc']:.4f} \\\\\n")
            f.write(f" & 1D-CNN & {m_c['accuracy']*100:.1f}\\% & {m_c['precision']*100:.1f}\\% & {m_c['recall']*100:.1f}\\% & {m_c['pr_auc']:.4f} \\\\\n")
            f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"[OK] Exported {tex_path}")


if __name__ == "__main__":
    main()
