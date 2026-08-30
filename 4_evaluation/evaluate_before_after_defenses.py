#!/usr/bin/env python3
"""
Protocol-Level Countermeasures Evaluation:
Simulates and evaluates:
1. Mode 1: Lightweight Adaptive Intra-frame Padding (1-128 Bytes random padding + micro-jitter, ~4-5% overhead).
2. Mode 2: Cell Coalescing & Cover Traffic Shaping (Coalesces 514B cells into 1448B frames + cover traffic, ~11-14% overhead).
Evaluates XGBoost, 1D-CNN, and Flow-Transformer before vs. after defense.
Exports LaTeX tables and publication figures.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb

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
    LATEX_TABLES_DIR,
    RANDOM_SEED,
    set_global_seed,
    setup_matplotlib_style
)
from sanitizer import compute_flow_statistics
from architectures import WebTunnel1DCNN, WebTunnelTransformer
from utils import load_tabular_data, load_sequence_data, compute_metrics, get_device


def simulate_lightweight_padding(X_seq, y_bin):
    """Mode 1: Lightweight Adaptive Intra-frame Padding (1-128 Bytes random padding + micro-jitter)."""
    X_def = X_seq.copy()
    total_orig = 0
    total_pad = 0

    for i in range(len(y_bin)):
        if y_bin[i] == 1:
            for s in range(X_def.shape[1]):
                norm_len = X_def[i, s, 0]
                norm_iat = X_def[i, s, 1]
                if abs(norm_len) < 1e-4:
                    continue
                orig_bytes = abs(norm_len) * 1500.0
                total_orig += orig_bytes

                pad = np.random.randint(1, 129)
                final_bytes = min(1480.0, orig_bytes + pad)
                total_pad += (final_bytes - orig_bytes)

                jitter_iat = min(1.0, norm_iat + np.random.uniform(0.001, 0.03))
                sign = 1.0 if norm_len > 0 else -1.0
                X_def[i, s, 0] = sign * (final_bytes / 1500.0)
                X_def[i, s, 1] = jitter_iat

    overhead_pct = (total_pad / max(total_orig, 1.0)) * 100.0
    return X_def, overhead_pct


def simulate_full_cell_coalescing_and_morphing(X_seq, y_bin):
    """Mode 2: Physical Cell Coalescing into MTU (1448B) Frames and Cover Traffic Shaping."""
    X_def = np.zeros_like(X_seq)
    total_orig_bytes = 0
    total_def_bytes = 0

    for i in range(len(y_bin)):
        if y_bin[i] == 0:
            X_def[i] = X_seq[i]
            continue

        raw_pkts = []
        for s in range(X_seq.shape[1]):
            val = X_seq[i, s, 0]
            if abs(val) < 1e-4:
                continue
            direction = 1 if val > 0 else -1
            length = int(round(abs(val) * 1500.0))
            iat = float(X_seq[i, s, 1])
            raw_pkts.append((direction, length, iat))
            total_orig_bytes += length

        if not raw_pkts:
            continue

        coalesced_pkts = []
        curr_dir = None
        curr_buf = 0
        curr_iat = 0.0

        for direction, length, iat in raw_pkts:
            if curr_dir is None:
                curr_dir = direction
                curr_buf = length
                curr_iat = iat
            elif curr_dir == direction:
                if curr_buf + length <= 1448:
                    curr_buf += length
                    curr_iat += iat * 0.5
                else:
                    coalesced_pkts.append((curr_dir, curr_buf, curr_iat))
                    curr_buf = length
                    curr_iat = iat
            else:
                coalesced_pkts.append((curr_dir, curr_buf, curr_iat))
                curr_dir = direction
                curr_buf = length
                curr_iat = iat

        if curr_buf > 0:
            coalesced_pkts.append((curr_dir, curr_buf, curr_iat))

        # Add dummy cover traffic in handshake
        shaped_pkts = []
        if coalesced_pkts:
            shaped_pkts.append((1, np.random.randint(450, 750), 0.0))
            shaped_pkts.append((-1, np.random.randint(600, 1100), np.random.uniform(0.01, 0.03)))

        for direction, length, iat in coalesced_pkts:
            if np.random.random() < 0.6:
                pad = np.random.randint(16, 128)
                length = min(1448, length + pad)
            shaped_pkts.append((direction, length, iat + np.random.uniform(0.005, 0.04)))

        for p_idx, (direction, length, iat) in enumerate(shaped_pkts[:X_def.shape[1]]):
            total_def_bytes += length
            X_def[i, p_idx, 0] = (direction * length) / 1500.0
            X_def[i, p_idx, 1] = min(1.0, iat)

    overhead_pct = ((total_def_bytes - total_orig_bytes) / max(total_orig_bytes, 1.0)) * 100.0
    return X_def, max(0.0, overhead_pct)


def recompute_tabular_features(X_seq_def):
    """Recomputes the 48 statistical flow features from defense-modified packet sequences."""
    X_tab_def = []
    for i in range(len(X_seq_def)):
        pkts = []
        curr_t = 0.0
        for s in range(X_seq_def.shape[1]):
            norm_len = X_seq_def[i, s, 0]
            norm_iat = X_seq_def[i, s, 1]
            delta_t = float(np.expm1(float(norm_iat) * 10.0))
            curr_t += max(0.0, delta_t)
            signed_len = int(round(norm_len * 1500.0))
            pkts.append((curr_t, signed_len))

        if len(pkts) < 3:
            pkts = [(0.0, 500), (0.05, -500), (0.1, 500)]
        X_tab_def.append(compute_flow_statistics(pkts))
    return np.array(X_tab_def, dtype=np.float32)


def main():
    set_global_seed(RANDOM_SEED)
    setup_matplotlib_style()
    device = get_device()

    tab_data = load_tabular_data(TABULAR_DATASET_PATH)
    seq_data = load_sequence_data(SEQUENCE_DATASET_PATH)

    X_tab_test, y_test = tab_data["X_test"], tab_data["y_test"]
    X_seq_test = seq_data["X_test"]

    # Load trained models
    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model(XGBOOST_MODEL_JSON)

    model_cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model_cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model_cnn.eval()

    model_tf = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
    model_tf.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=device))
    model_tf.eval()

    # 1. Evaluate Original (Before Defense)
    print("=== 1. Evaluating BEFORE Defense (Original WebTunnel) ===")
    probs_xgb_orig = clf_xgb.predict_proba(X_tab_test)[:, 1]
    m_xgb_orig = compute_metrics(y_test, probs_xgb_orig, threshold=0.5)

    with torch.no_grad():
        tensor_cnn = torch.tensor(X_seq_test, dtype=torch.float32).permute(0, 2, 1).to(device)
        probs_cnn_orig = model_cnn(tensor_cnn).squeeze(-1).cpu().numpy()
    m_cnn_orig = compute_metrics(y_test, probs_cnn_orig, threshold=0.5)

    with torch.no_grad():
        tensor_tf = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
        probs_tf_orig = model_tf(tensor_tf).squeeze(-1).cpu().numpy()
    m_tf_orig = compute_metrics(y_test, probs_tf_orig, threshold=0.5)

    # 2. Mode 1: Lightweight Adaptive Padding
    print("=== 2. Evaluating Mode 1: Lightweight Adaptive Padding (1-128B) ===")
    X_seq_m1, overhead_m1 = simulate_lightweight_padding(X_seq_test, y_test)
    X_tab_m1 = recompute_tabular_features(X_seq_m1)

    probs_xgb_m1 = clf_xgb.predict_proba(X_tab_m1)[:, 1]
    m_xgb_m1 = compute_metrics(y_test, probs_xgb_m1, threshold=0.5)

    with torch.no_grad():
        tensor_cnn_m1 = torch.tensor(X_seq_m1, dtype=torch.float32).permute(0, 2, 1).to(device)
        probs_cnn_m1 = model_cnn(tensor_cnn_m1).squeeze(-1).cpu().numpy()
    m_cnn_m1 = compute_metrics(y_test, probs_cnn_m1, threshold=0.5)

    with torch.no_grad():
        tensor_tf_m1 = torch.tensor(X_seq_m1, dtype=torch.float32).to(device)
        probs_tf_m1 = model_tf(tensor_tf_m1).squeeze(-1).cpu().numpy()
    m_tf_m1 = compute_metrics(y_test, probs_tf_m1, threshold=0.5)

    # 3. Mode 2: Full Cell Coalescing & Cover Traffic Shaping
    print("=== 3. Evaluating Mode 2: Full Cell Coalescing & Cover Traffic Shaping ===")
    X_seq_m2, overhead_m2 = simulate_full_cell_coalescing_and_morphing(X_seq_test, y_test)
    X_tab_m2 = recompute_tabular_features(X_seq_m2)

    probs_xgb_m2 = clf_xgb.predict_proba(X_tab_m2)[:, 1]
    m_xgb_m2 = compute_metrics(y_test, probs_xgb_m2, threshold=0.5)

    with torch.no_grad():
        tensor_cnn_m2 = torch.tensor(X_seq_m2, dtype=torch.float32).permute(0, 2, 1).to(device)
        probs_cnn_m2 = model_cnn(tensor_cnn_m2).squeeze(-1).cpu().numpy()
    m_cnn_m2 = compute_metrics(y_test, probs_cnn_m2, threshold=0.5)

    with torch.no_grad():
        tensor_tf_m2 = torch.tensor(X_seq_m2, dtype=torch.float32).to(device)
        probs_tf_m2 = model_tf(tensor_tf_m2).squeeze(-1).cpu().numpy()
    m_tf_m2 = compute_metrics(y_test, probs_tf_m2, threshold=0.5)

    # Summary Report
    print("\n" + "="*85)
    print("       KOMPLEXNÍ SROVNÁNÍ: PŘED OBRANOU vs. LEHKÝ PADDING vs. PLNÝ MORPHING")
    print("="*85)
    print(f"{'Model':<18} | {'Stav / Režim obrany':<35} | {'Accuracy':<9} | {'Recall':<9} | {'Overhead':<8}")
    print("-" * 85)
    for model_name, orig, m1, m2 in [
        ("1D-CNN", m_cnn_orig, m_cnn_m1, m_cnn_m2),
        ("Flow-Transformer", m_tf_orig, m_tf_m1, m_tf_m2),
        ("XGBoost", m_xgb_orig, m_xgb_m1, m_xgb_m2)
    ]:
        print(f"{model_name:<18} | {'1. Bez obrany (Původní WebTunnel)':<35} | {orig['accuracy']*100:>8.1f}% | {orig['recall']*100:>8.1f}% | {'0.0%':>8}")
        print(f"{'':<18} | {'2. Lehký padding (1-128B)':<35} | {m1['accuracy']*100:>8.1f}% | {m1['recall']*100:>8.1f}% | {overhead_m1:>7.1f}%")
        print(f"{'':<18} | {'3. Plný Cell Coalescing & Mimicry':<35} | {m2['accuracy']*100:>8.1f}% | {m2['recall']*100:>8.1f}% | {overhead_m2:>7.1f}%")
        print("-" * 85)

    # Export LaTeX Table
    tex_path = os.path.join(LATEX_TABLES_DIR, "table_before_after_defense.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Účinnost navržených protiopatření: Degradace detekčního Recallu a datová režie}" + "\n")
        f.write(r"\label{tab:before_after_defense}" + "\n")
        f.write(r"\begin{tabular}{llccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Klasifikátor} & \textbf{Režim obrany} & \textbf{Accuracy} & \textbf{Recall (Úspěšnost cenzora)} & \textbf{Datová režie} \\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(f"1D-CNN & 1. Bez obrany (WebTunnel baseline) & {m_cnn_orig['accuracy']*100:.1f}\\% & {m_cnn_orig['recall']*100:.1f}\\% & 0.0\\% \\\\\n")
        f.write(f" & 2. Adaptivní Intra-frame Padding (1--128B) & {m_cnn_m1['accuracy']*100:.1f}\\% & {m_cnn_m1['recall']*100:.1f}\\% & {overhead_m1:.1f}\\% \\\\\n")
        f.write(f" & 3. Cell Coalescing \\& Cover Shaping & {m_cnn_m2['accuracy']*100:.1f}\\% & {m_cnn_m2['recall']*100:.1f}\\% & {overhead_m2:.1f}\\% \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(f"Transformer & 1. Bez obrany (WebTunnel baseline) & {m_tf_orig['accuracy']*100:.1f}\\% & {m_tf_orig['recall']*100:.1f}\\% & 0.0\\% \\\\\n")
        f.write(f" & 2. Adaptivní Intra-frame Padding (1--128B) & {m_tf_m1['accuracy']*100:.1f}\\% & {m_tf_m1['recall']*100:.1f}\\% & {overhead_m1:.1f}\\% \\\\\n")
        f.write(f" & 3. Cell Coalescing \\& Cover Shaping & {m_tf_m2['accuracy']*100:.1f}\\% & {m_tf_m2['recall']*100:.1f}\\% & {overhead_m2:.1f}\\% \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(f"XGBoost & 1. Bez obrany (WebTunnel baseline) & {m_xgb_orig['accuracy']*100:.1f}\\% & {m_xgb_orig['recall']*100:.1f}\\% & 0.0\\% \\\\\n")
        f.write(f" & 2. Adaptivní Intra-frame Padding (1--128B) & {m_xgb_m1['accuracy']*100:.1f}\\% & {m_xgb_m1['recall']*100:.1f}\\% & {overhead_m1:.1f}\\% \\\\\n")
        f.write(f" & 3. Cell Coalescing \\& Cover Shaping & {m_xgb_m2['accuracy']*100:.1f}\\% & {m_xgb_m2['recall']*100:.1f}\\% & {overhead_m2:.1f}\\% \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"\n[OK] Exported {tex_path}")

    # Plot 1: Metrics comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["1D-CNN", "Flow-Transformer", "XGBoost"]
    orig_rec = [m_cnn_orig['recall']*100, m_tf_orig['recall']*100, m_xgb_orig['recall']*100]
    m1_rec = [m_cnn_m1['recall']*100, m_tf_m1['recall']*100, m_xgb_m1['recall']*100]
    m2_rec = [m_cnn_m2['recall']*100, m_tf_m2['recall']*100, m_xgb_m2['recall']*100]

    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, orig_rec, w, label="Před obranou (Původní WebTunnel)", color="#d62728", alpha=0.85)
    ax.bar(x, m1_rec, w, label=f"Mód 1: Adaptivní Padding (+{overhead_m1:.1f}% režie)", color="#ff7f0e", alpha=0.85)
    ax.bar(x + w, m2_rec, w, label=f"Mód 2: Cell Coalescing (+{overhead_m2:.1f}% režie)", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Detekční Recall cenzora (%)")
    ax.set_title("Vliv protokolárních obran na úspěšnost detekce WebTunnelu")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "before_vs_after_metrics.png"))
    plt.close()
    print(f"[OK] Saved {os.path.join(PLOTS_DIR, 'before_vs_after_metrics.png')}")

    # Plot 2: Spectral distribution shift
    plt.figure(figsize=(11, 5))
    wt_indices = np.where(y_test == 1)[0]
    lens_orig = (np.abs(X_seq_test[wt_indices, :, 0]) * 1500.0).flatten()
    lens_m1 = (np.abs(X_seq_m1[wt_indices, :, 0]) * 1500.0).flatten()
    lens_m2 = (np.abs(X_seq_m2[wt_indices, :, 0]) * 1500.0).flatten()

    lens_orig = lens_orig[lens_orig > 10]
    lens_m1 = lens_m1[lens_m1 > 10]
    lens_m2 = lens_m2[lens_m2 > 10]

    sns.kdeplot(lens_orig, label="Původní WebTunnel (Kvantizace 558 B)", color="red", lw=2)
    sns.kdeplot(lens_m1, label="Mód 1: Adaptivní Padding (1-128 B)", color="orange", lw=2)
    sns.kdeplot(lens_m2, label="Mód 2: Cell Coalescing do MTU 1448 B", color="green", lw=2)

    plt.axvline(x=558, color="darkred", linestyle="--", alpha=0.6, label="558 B Tor Cell")
    plt.axvline(x=1448, color="darkgreen", linestyle=":", alpha=0.6, label="1448 B MTU Frame")
    plt.xlim(0, 1600)
    plt.xlabel("L7 Velikost paketu (Bytes)")
    plt.ylabel("Hustota pravděpodobnosti")
    plt.title("Změna distribuce délek paketů vlivem Cell Coalescingu a Paddingu")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "before_vs_after_distributions.png"))
    plt.close()
    print(f"[OK] Saved {os.path.join(PLOTS_DIR, 'before_vs_after_distributions.png')}")


if __name__ == "__main__":
    main()
