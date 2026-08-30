#!/usr/bin/env python3
"""
2-Tier Cascaded Classification Architecture Benchmark (L1 CPU XGBoost -> L2 GPU 1D-CNN):
- Evaluates line-rate inspection on ISP backbone networks.
- Measures single-flow latency (microseconds) and batch throughput (flows/second).
- Evaluates accuracy, PR-AUC, and percentage of ambiguous flows escalated to L2 GPU.
- Exports table_cascaded_pipeline.tex and cascaded_pipeline_throughput.png.
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
import xgboost as xgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import (
    TABULAR_DATASET_PATH,
    SEQUENCE_DATASET_PATH,
    XGBOOST_MODEL_JOBLIB,
    XGBOOST_MODEL_JSON,
    CNN_MODEL_PATH,
    EVALUATION_DIR,
    PLOTS_DIR,
    LATEX_TABLES_DIR,
    setup_matplotlib_style
)
from architectures import WebTunnel1DCNN
from utils import load_tabular_data, load_sequence_data, compute_metrics, get_device


def benchmark_cascaded_pipeline(tau_low=0.05, tau_high=0.95):
    print("=== Evaluating 2-Tier Cascaded Classification Pipeline (L1 CPU -> L2 GPU) ===")
    setup_matplotlib_style()
    device = get_device()

    tab_data = load_tabular_data(TABULAR_DATASET_PATH)
    seq_data = load_sequence_data(SEQUENCE_DATASET_PATH)

    X_test_tab, y_test = tab_data["X_test"], tab_data["y_test"]
    X_test_seq = np.transpose(seq_data["X_test"], (0, 2, 1))
    n_samples = len(y_test)

    # Load Models
    if os.path.exists(XGBOOST_MODEL_JOBLIB):
        clf_xgb = joblib.load(XGBOOST_MODEL_JOBLIB)
    else:
        clf_xgb = xgb.XGBClassifier()
        clf_xgb.load_model(XGBOOST_MODEL_JSON)
        clf_xgb.classes_ = np.array([0, 1])

    model_cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model_cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model_cnn.eval()

    # Benchmark L1 Latency (XGBoost CPU)
    for _ in range(5):
        _ = clf_xgb.predict_proba(X_test_tab)
    t0 = time.perf_counter_ns()
    p_xgb = clf_xgb.predict_proba(X_test_tab)[:, 1]
    t1 = time.perf_counter_ns()
    batch_lat_l1_us = ((t1 - t0) / 1000.0) / n_samples
    throughput_l1 = (n_samples / ((t1 - t0) / 1e9))

    single_l1_times = []
    for i in range(min(50, n_samples)):
        s_t0 = time.perf_counter_ns()
        _ = clf_xgb.predict_proba(X_test_tab[i:i+1])
        s_t1 = time.perf_counter_ns()
        single_l1_times.append((s_t1 - s_t0) / 1000.0)
    single_lat_l1_us = float(np.median(single_l1_times))

    # Benchmark L2 Latency (1D-CNN GPU)
    X_seq_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(5):
            _ = model_cnn(X_seq_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        p_cnn = model_cnn(X_seq_tensor).squeeze(-1).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()

    batch_lat_l2_us = ((t1 - t0) / 1000.0) / n_samples
    throughput_l2 = (n_samples / ((t1 - t0) / 1e9))

    single_l2_times = []
    with torch.no_grad():
        for i in range(min(50, n_samples)):
            sample_t = X_seq_tensor[i:i+1]
            if device.type == "cuda":
                torch.cuda.synchronize()
            s_t0 = time.perf_counter_ns()
            _ = model_cnn(sample_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            s_t1 = time.perf_counter_ns()
            single_l2_times.append((s_t1 - s_t0) / 1000.0)
    single_lat_l2_us = float(np.median(single_l2_times))

    # Cascaded Classification Logic
    is_ambiguous = (p_xgb >= tau_low) & (p_xgb <= tau_high)
    n_escalated = int(np.sum(is_ambiguous))
    pct_escalated = (n_escalated / n_samples) * 100.0

    final_probs = p_xgb.copy()
    final_probs[is_ambiguous] = p_cnn[is_ambiguous]

    metrics_cascade = compute_metrics(y_test, final_probs, threshold=0.5)

    eff_single_lat_us = single_lat_l1_us + (pct_escalated / 100.0) * single_lat_l2_us
    eff_batch_lat_us = batch_lat_l1_us + (pct_escalated / 100.0) * batch_lat_l2_us
    eff_throughput = 1e6 / eff_batch_lat_us if eff_batch_lat_us > 0 else throughput_l1

    results = {
        "n_samples": n_samples,
        "n_l1_resolved": n_samples - n_escalated,
        "n_l2_escalated": n_escalated,
        "pct_l2_escalated": pct_escalated,
        "single_flow_latency_l1_us": single_lat_l1_us,
        "single_flow_latency_l2_us": single_lat_l2_us,
        "effective_single_flow_latency_us": eff_single_lat_us,
        "effective_batch_throughput_fps": eff_throughput,
        "accuracy": metrics_cascade["accuracy"],
        "precision": metrics_cascade["precision"],
        "recall": metrics_cascade["recall"],
        "f1_score": metrics_cascade["f1_score"],
        "pr_auc": metrics_cascade["pr_auc"],
        "roc_auc": metrics_cascade["roc_auc"]
    }

    print("\n" + "="*85)
    print("       VÝSLEDKY EXPERIMENTÁLNÍHO BENCHMARKU KASKÁDOVÉ ARCHITEKTURY")
    print("="*85)
    print(f"Celkový počet testovacích toků: {n_samples}")
    print(f"Odbaveno na L1 filtru (CPU XGBoost): {n_samples - n_escalated} ({100 - pct_escalated:.1f} %)")
    print(f"Eskalováno na L2 hloubkovou inspekci (GPU 1D-CNN): {n_escalated} ({pct_escalated:.1f} %)")
    print(f"Single-flow latence L1 (XGBoost): {single_lat_l1_us:.2f} µs | L2 (1D-CNN): {single_lat_l2_us:.2f} µs")
    print(f"Efektivní single-flow latence kaskády: {eff_single_lat_us:.2f} µs")
    print(f"Efektivní batch propustnost kaskády: {eff_throughput:,.0f} flows/sec")
    print(f"Přesnost (Accuracy): {metrics_cascade['accuracy']*100:.2f} % | PR-AUC: {metrics_cascade['pr_auc']:.4f}")
    print("="*85)

    # Export LaTeX Table
    tex_path = os.path.join(LATEX_TABLES_DIR, "table_cascaded_pipeline.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Výkonnostní parametry 2-vrstvé kaskádové inspekční architektury (L1 CPU $\rightarrow$ L2 GPU)}" + "\n")
        f.write(r"\label{tab:cascaded_pipeline}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Inspekční vrstva} & \textbf{Hardware} & \textbf{Zpracovaný provoz} & \textbf{Single-Flow Latence} & \textbf{Batch Propustnost} \\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(f"L1: XGBoost Filtr & AMD Ryzen 9800X3D (CPU) & {100.0 - pct_escalated:.1f}\\% toků & {single_lat_l1_us:.2f}~$\\mu$s & {throughput_l1:,.0f}~toků/s \\\\\n")
        f.write(f"L2: 1D-CNN Hloubková & NVIDIA RTX 5070 Ti (CUDA) & {pct_escalated:.1f}\\% toků & {single_lat_l2_us:.2f}~$\\mu$s & {throughput_l2:,.0f}~toků/s \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(f"\\textbf{{Hybridní Kaskáda}} & \\textbf{{CPU + GPU}} & \\textbf{{100.0\\%}} & \\textbf{{{eff_single_lat_us:.2f}~$\\mu$s}} & \\textbf{{{eff_throughput:,.0f}~toků/s}} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"[OK] Exported {tex_path}")

    # Export Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["L1: XGBoost (CPU)", "L2: 1D-CNN (GPU)", "Hybridní Kaskáda (L1+L2)"]
    tp_vals = [throughput_l1, throughput_l2, eff_throughput]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]

    bars = ax.bar(names, tp_vals, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("Inspekční propustnost (Toků/sec - Log)")
    ax.set_title("Průchodnost inspekčních vrstev a efektivita hybridní kaskády")

    for bar, val in zip(bars, tp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.15, f"{val:,.0f} toků/s", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cascaded_pipeline_throughput.png"))
    plt.close()
    print(f"[OK] Saved {os.path.join(PLOTS_DIR, 'cascaded_pipeline_throughput.png')}")

    out_json = os.path.join(EVALUATION_DIR, "cascaded_pipeline_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    return results


if __name__ == "__main__":
    benchmark_cascaded_pipeline()
