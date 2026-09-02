#!/usr/bin/env python3
"""
Multi-Class Hard Negatives Breakdown & Confusion Matrix Evaluation:
- Computes detailed class-by-class accuracy for WebTunnel vs all 5 Hard Negative classes.
- Generates 3-panel comparative heatmaps across all 6 classes.
- Generates 2x2 binary confusion matrices (TN, FP, FN, TP) for all 3 models.
- Exports table_class_breakdown.tex, confusion_matrix_breakdown.png, and neural_network_confusion_comparison.png.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import confusion_matrix

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
    CLASS_SHORT_NAMES,
    setup_matplotlib_style
)
from architectures import WebTunnel1DCNN, WebTunnelTransformer
from utils import load_tabular_data, load_sequence_data, get_device


def generate_confusion_breakdown():
    print("=== Generating Class Breakdown & Confusion Matrices (All 3 Models) ===")
    setup_matplotlib_style()
    device = get_device()

    tab_data = load_tabular_data(TABULAR_DATASET_PATH)
    seq_data = load_sequence_data(SEQUENCE_DATASET_PATH)

    X_test_tab = tab_data["X_test"]
    y_test_bin = tab_data["y_test"]
    y_test_mul = tab_data["y_test_mul"]
    X_test_seq = seq_data["X_test"]

    # 1. Predictions
    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model(XGBOOST_MODEL_JSON)
    p_xgb = clf_xgb.predict_proba(X_test_tab)[:, 1]
    preds_xgb = (p_xgb >= 0.5).astype(int)

    model_cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model_cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model_cnn.eval()
    with torch.no_grad():
        p_cnn = model_cnn(torch.tensor(X_test_seq, dtype=torch.float32).permute(0, 2, 1).to(device)).squeeze(-1).cpu().numpy()
    preds_cnn = (p_cnn >= 0.5).astype(int)

    model_tf = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
    model_tf.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=device))
    model_tf.eval()
    with torch.no_grad():
        p_tf = model_tf(torch.tensor(X_test_seq, dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()
    preds_tf = (p_tf >= 0.5).astype(int)

    # 2. Per-Class Accuracy Decomposition
    print("\n" + "="*90)
    print("        PŘESNOST DETEKCE ROZPADLÁ NA JEDNOTLIVÉ TŘÍDY (HARD NEGATIVES)")
    print("="*90)
    print(f"{'Třída provozu':<20} | {'Vzorků':<8} | {'1D-CNN':<16} | {'Transformer':<16} | {'XGBoost':<16}")
    print("-" * 90)

    class_stats = []
    cm_6x2_xgb = np.zeros((6, 2), dtype=float)
    cm_6x2_cnn = np.zeros((6, 2), dtype=float)
    cm_6x2_tf  = np.zeros((6, 2), dtype=float)

    for i, c_name in enumerate(CLASS_SHORT_NAMES):
        mask = (y_test_mul == i)
        n_c = int(np.sum(mask))
        if n_c == 0:
            continue

        c_true = y_test_bin[mask]
        c_preds_xgb = preds_xgb[mask]
        c_preds_cnn = preds_cnn[mask]
        c_preds_tf  = preds_tf[mask]

        if i == 0: # WebTunnel
            corr_xgb = np.sum(c_preds_xgb == 1)
            corr_cnn = np.sum(c_preds_cnn == 1)
            corr_tf  = np.sum(c_preds_tf == 1)
        else: # Legitimate
            corr_xgb = np.sum(c_preds_xgb == 0)
            corr_cnn = np.sum(c_preds_cnn == 0)
            corr_tf  = np.sum(c_preds_tf == 0)

        acc_xgb = (corr_xgb / n_c) * 100.0
        acc_cnn = (corr_cnn / n_c) * 100.0
        acc_tf  = (corr_tf / n_c) * 100.0

        cm_6x2_xgb[i, 0] = np.sum(c_preds_xgb == 0) / n_c * 100.0
        cm_6x2_xgb[i, 1] = np.sum(c_preds_xgb == 1) / n_c * 100.0

        cm_6x2_cnn[i, 0] = np.sum(c_preds_cnn == 0) / n_c * 100.0
        cm_6x2_cnn[i, 1] = np.sum(c_preds_cnn == 1) / n_c * 100.0

        cm_6x2_tf[i, 0]  = np.sum(c_preds_tf == 0) / n_c * 100.0
        cm_6x2_tf[i, 1]  = np.sum(c_preds_tf == 1) / n_c * 100.0

        class_stats.append({
            "class": c_name,
            "n": n_c,
            "acc_xgb": acc_xgb, "corr_xgb": int(corr_xgb),
            "acc_cnn": acc_cnn, "corr_cnn": int(corr_cnn),
            "acc_tf": acc_tf,   "corr_tf": int(corr_tf)
        })

        print(f"{c_name:<20} | {n_c:<8} | {corr_cnn:>2}/{n_c} ({acc_cnn:>5.1f}%) | {corr_tf:>2}/{n_c} ({acc_tf:>5.1f}%) | {corr_xgb:>2}/{n_c} ({acc_xgb:>5.1f}%)")
    print("=" * 90)

    # 3. Export LaTeX Table
    tex_path = os.path.join(LATEX_TABLES_DIR, "table_class_breakdown.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Dekompozice přesnosti klasifikátorů napříč všemi 6 třídami provozu (WebTunnel vs. Hard Negatives)}" + "\n")
        f.write(r"\label{tab:class_breakdown}" + "\n")
        f.write(r"\begin{tabular}{lccccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Třída síťového provozu} & \textbf{Typ protokolu} & \textbf{Počet toků} & \textbf{1D-CNN} & \textbf{Transformer} & \textbf{XGBoost} \\" + "\n")
        f.write(r"\hline" + "\n")
        proto_map = {
            "WebTunnel": "Tor over HTTP/1.1 WebSocket",
            "Direct Browsing": "HTTP/2 Web (TLS 1.3)",
            "WS Ticker": "WSS Live JSON",
            "WS Chat": "WSS Interaktivní",
            "Video Stream": "HTTPS DASH/HLS",
            "Web Assets": "HTTPS Statický bundle"
        }
        for s in class_stats:
            f.write(f"{s['class']} & {proto_map.get(s['class'], 'HTTPS')} & {s['n']} & {s['acc_cnn']:.1f}\\% & {s['acc_tf']:.1f}\\% & {s['acc_xgb']:.1f}\\% \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"[OK] Exported {tex_path}")

    # 4. Multi-Class Breakdown Heatmap (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])

    panels = [
        (axes[0], cm_6x2_xgb, "XGBoost Baseline (Tabulární)"),
        (axes[1], cm_6x2_cnn, "1D-CNN (Deep Packet)"),
        (axes[2], cm_6x2_tf,  "Flow-Transformer ([CLS] Attention)")
    ]

    for idx, (ax, cm_mat, title) in enumerate(panels):
        sns.heatmap(
            cm_mat, annot=True, fmt=".1f", cmap="Blues", cbar=(idx == 2),
            cbar_ax=cbar_ax if idx == 2 else None, ax=ax,
            xticklabels=["Predikce: Benign", "Predikce: WebTunnel"],
            yticklabels=CLASS_SHORT_NAMES, vmin=0, vmax=100, annot_kws={"fontsize": 11, "fontweight": "bold"}
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        if idx == 0:
            ax.set_ylabel("Skutečná třída (Ground Truth)", fontsize=12, fontweight="bold")
        else:
            ax.set_ylabel("")

    plt.subplots_adjust(wspace=0.08, right=0.90)
    out_matrix = os.path.join(PLOTS_DIR, "confusion_matrix_breakdown.png")
    plt.savefig(out_matrix)
    plt.close()
    print(f"[OK] Saved {out_matrix}")

    # 5. 2x2 Binary Confusion Matrix Comparison Plot (All 3 Models)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    model_preds = [
        ("XGBoost Baseline", preds_xgb, p_xgb),
        ("1D-CNN Deep Packet", preds_cnn, p_cnn),
        ("Flow-Transformer", preds_tf, p_tf)
    ]

    for ax, (m_title, m_pred, m_prob) in zip(axes, model_preds):
        cm = confusion_matrix(y_test_bin, m_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn) * 100.0
        rec = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0

        cm_labels = np.array([
            [f"TN (Správně Benign)\n{tn}\n({tn/(tn+fp)*100:.1f}%)", f"FP (Falešný poplach)\n{fp}\n({fp/(tn+fp)*100:.1f}%)"],
            [f"FN (Přehlédnutý WT)\n{fn}\n({fn/(fn+tp)*100:.1f}%)", f"TP (Zachycený WT)\n{tp}\n({tp/(fn+tp)*100:.1f}%)"]
        ])

        sns.heatmap(
            cm, annot=cm_labels, fmt="", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Predikce: Benign", "Predikce: WebTunnel"],
            yticklabels=["Realita: Benign", "Realita: WebTunnel"],
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )
        ax.set_title(f"{m_title}\nAccuracy: {acc:.2f}% | Recall: {rec:.1f}%", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_nn_cm = os.path.join(PLOTS_DIR, "neural_network_confusion_comparison.png")
    plt.savefig(out_nn_cm)
    plt.close()
    print(f"[OK] Saved {out_nn_cm}")


if __name__ == "__main__":
    generate_confusion_breakdown()
