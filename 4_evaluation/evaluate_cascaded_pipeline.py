#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

sys.path.append("3_models")
from train_1d_cnn import WebTunnel1DCNN

PROCESSED_DIR = "data/processed"
PLOT_DIR = "4_evaluation/plots"
TABLE_DIR = "0_thesis_text/tables"
EVAL_DIR = "4_evaluation"

def benchmark_cascaded_pipeline(tau_low=0.05, tau_high=0.95):
    print("=== Evaluating 2-Tier Cascaded Classification Pipeline (L1 CPU -> L2 GPU) ===")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Load Data
    tab_data = np.load(os.path.join(PROCESSED_DIR, "tabular_dataset.npz"), allow_pickle=True)
    seq_data = np.load(os.path.join(PROCESSED_DIR, "sequence_dataset.npz"), allow_pickle=True)
    
    X_test_tab, y_test = tab_data["X_test"], tab_data["y_test"]
    X_test_seq = np.transpose(seq_data["X_test"], (0, 2, 1)) # (N, 2, 200)
    n_samples = len(y_test)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Models
    import joblib
    xgb_joblib_path = "3_models/saved_models/xgboost_baseline.joblib"
    if os.path.exists(xgb_joblib_path):
        clf_xgb = joblib.load(xgb_joblib_path)
    else:
        clf_xgb = xgb.XGBClassifier()
        clf_xgb.load_model("3_models/saved_models/xgboost_baseline.json")
        clf_xgb.classes_ = np.array([0, 1])
    
    model_cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model_cnn.load_state_dict(torch.load("3_models/saved_models/1d_cnn_best.pt", map_location=device))
    model_cnn.eval()
    
    # 3. Benchmark L1 Latency (XGBoost CPU): Single-flow vs Batch
    # Batch throughput
    for _ in range(5):
        _ = clf_xgb.predict_proba(X_test_tab)
    t0 = time.perf_counter_ns()
    p_xgb = clf_xgb.predict_proba(X_test_tab)[:, 1]
    t1 = time.perf_counter_ns()
    lat_l1_batch_per_flow_ms = ((t1 - t0) / 1e6) / n_samples
    
    # Single-flow latency
    n_iters = min(250, n_samples)
    t0 = time.perf_counter_ns()
    for i in range(n_iters):
        _ = clf_xgb.predict_proba(X_test_tab[i:i+1])
    t1 = time.perf_counter_ns()
    single_lat_xgb_us = ((t1 - t0) / 1000.0) / n_iters
    
    # 4. Determine Escalation to L2
    escalate_mask = (p_xgb >= tau_low) & (p_xgb <= tau_high)
    n_escalated = int(np.sum(escalate_mask))
    escalation_rate_pct = (n_escalated / n_samples) * 100.0
    
    # 5. Benchmark L2 Latency (1D-CNN GPU): Batched DataLoader inference
    from torch.utils.data import TensorDataset, DataLoader
    X_seq_t = torch.from_numpy(X_test_seq)
    loader_l2 = DataLoader(TensorDataset(X_seq_t), batch_size=64, shuffle=False)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            for bx, in loader_l2:
                _ = model_cnn(bx.to(device))
    if device.type == "cuda": torch.cuda.synchronize()
    
    t0 = time.perf_counter_ns()
    all_p_cnn = []
    with torch.no_grad():
        for bx, in loader_l2:
            preds_batch = model_cnn(bx.to(device)).squeeze(-1).cpu().numpy()
            all_p_cnn.extend(preds_batch)
    if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    p_cnn = np.array(all_p_cnn)
    lat_l2_batch_per_flow_ms = ((t1 - t0) / 1e6) / n_samples
    
    # Single-flow GPU latency
    t0 = time.perf_counter_ns()
    with torch.no_grad():
        for i in range(n_iters):
            _ = model_cnn(X_seq_t[i:i+1].to(device))
            if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    single_lat_cnn_us = ((t1 - t0) / 1000.0) / n_iters
    
    # 6. Combined Cascade Decision
    p_cascade = p_xgb.copy()
    if n_escalated > 0:
        p_cascade[escalate_mask] = p_cnn[escalate_mask]
        
    preds_cascade = (p_cascade >= 0.5).astype(int)
    preds_xgb = (p_xgb >= 0.5).astype(int)
    preds_cnn = (p_cnn >= 0.5).astype(int)
    
    # 7. Compute Cascaded System Latency & Throughput
    lat_cascade_batch_per_flow_ms = lat_l1_batch_per_flow_ms + (escalation_rate_pct / 100.0) * lat_l2_batch_per_flow_ms
    single_lat_cascade_us = single_lat_xgb_us + (escalation_rate_pct / 100.0) * single_lat_cnn_us
    
    throughput_cascade = 1000.0 / max(lat_cascade_batch_per_flow_ms, 1e-6)
    throughput_xgb = 1000.0 / max(lat_l1_batch_per_flow_ms, 1e-6)
    throughput_cnn = 1000.0 / max(lat_l2_batch_per_flow_ms, 1e-6)
    
    # Metrics
    def get_metrics(y_true, probs, preds):
        return {
            "acc": accuracy_score(y_true, preds),
            "prec": precision_score(y_true, preds, zero_division=0),
            "rec": recall_score(y_true, preds, zero_division=0),
            "f1": f1_score(y_true, preds, zero_division=0),
            "pr_auc": average_precision_score(y_true, probs),
            "roc_auc": roc_auc_score(y_true, probs)
        }
        
    m_xgb = get_metrics(y_test, p_xgb, preds_xgb)
    m_cnn = get_metrics(y_test, p_cnn, preds_cnn)
    m_cas = get_metrics(y_test, p_cascade, preds_cascade)
    
    print("\n" + "="*85)
    print("       VÝSLEDKY EXPERIMENTÁLNÍHO BENCHMARKU KASKÁDOVÉ ARCHITEKTURY")
    print("="*85)
    print(f"Celkový počet testovacích toků: {n_samples}")
    print(f"Odbaveno na L1 filtru (CPU XGBoost): {n_samples - n_escalated} ({100.0 - escalation_rate_pct:.1f} %)")
    print(f"Eskalováno na L2 hloubkovou inspekci (GPU 1D-CNN): {n_escalated} ({escalation_rate_pct:.1f} %)")
    print(f"Single-flow latence L1 (XGBoost): {single_lat_xgb_us:.2f} µs | L2 (1D-CNN): {single_lat_cnn_us:.2f} µs")
    print(f"Efektivní single-flow latence kaskády: {single_lat_cascade_us:.2f} µs")
    print(f"Efektivní batch propustnost kaskády: {throughput_cascade:,.0f} flows/sec")
    print(f"Přesnost (Accuracy): {m_cas['acc']*100:.2f} % | PR-AUC: {m_cas['pr_auc']:.4f}")
    print("="*85)
    
    # 8. Export LaTeX Table
    tex = r"""\begin{table}[htbp]
\centering
\caption{Empirické zhodnocení dvoustupňové kaskádové architektury (L1 CPU $\rightarrow$ L2 GPU)}
\label{tab:cascaded_pipeline}
\begin{tabular}{lcccccc}
\hline
\textbf{Úroveň / Architektura} & \textbf{Hardware} & \textbf{Accuracy} & \textbf{PR-AUC} & \textbf{Single-flow Lat.} & \textbf{Batch Propustnost} \\
\hline
Samostatný L1 filtr (XGBoost) & Ryzen 9800X3D & """ + f"{m_xgb['acc']*100:.1f}\\% & {m_xgb['pr_auc']:.4f} & {single_lat_xgb_us:.1f}\\,\\mu\\text{{s}} & {throughput_xgb:,.0f} flows/s" + r""" \\
Samostatná L2 inspekce (1D-CNN) & RTX 5070 Ti & """ + f"{m_cnn['acc']*100:.1f}\\% & {m_cnn['pr_auc']:.4f} & {single_lat_cnn_us:.1f}\\,\\mu\\text{{s}} & {throughput_cnn:,.0f} flows/s" + r""" \\
\hline
\textbf{Dvoustupňová kaskáda (Hybrid)} & \textbf{CPU + GPU} & \textbf{""" + f"{m_cas['acc']*100:.1f}\\%" + r"""} & \textbf{""" + f"{m_cas['pr_auc']:.4f}" + r"""} & \textbf{""" + f"{single_lat_cascade_us:.1f}\\,\\mu\\text{{s}}" + r"""} & \textbf{""" + f"{throughput_cascade:,.0f} flows/s" + r"""} \\
\hline
\multicolumn{6}{l}{\footnotesize Pásmo nejistoty pro eskalaci na GPU: $p \in [0.05, 0.95]$; Míra odbavení na L1 (CPU): """ + f"{100.0 - escalation_rate_pct:.1f}\\%" + r"""} \\
\hline
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLE_DIR, "table_cascaded_pipeline.tex"), "w") as f:
        f.write(tex)
    print(f"[OK] Exported {TABLE_DIR}/table_cascaded_pipeline.tex")
    
    # 9. Plot Throughput & Architecture Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: Throughput Bar
    archs = ["L1 Filtr\n(XGBoost CPU)", "L2 Inspekce\n(1D-CNN GPU)", "Dvoustupňová\nKaskáda (Hybrid)"]
    tps = [throughput_xgb, throughput_cnn, throughput_cascade]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    bars = ax1.bar(archs, tps, color=colors, alpha=0.85)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e4, top=max(tps) * 5.0)
    ax1.set_ylabel("Line-Rate Propustnost (toků / s, log měřítko)", fontsize=11, fontweight="bold")
    ax1.set_title("Srovnání klasifikační propustnosti", fontsize=12, fontweight="bold", pad=12)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.25,
                 f"{height:,.0f} fl/s", ha='center', va='bottom', fontsize=9, fontweight='bold')
                 
    # Right: Escalation Pie Chart
    resolved_l1 = 100.0 - escalation_rate_pct
    escalated_l2 = escalation_rate_pct
    ax2.pie([resolved_l1, max(escalated_l2, 0.1)], 
            labels=[f"Odbaveno na L1 CPU\n({resolved_l1:.1f} %)", f"Eskalováno na L2 GPU\n({escalated_l2:.1f} %)"],
            colors=["#2ca02c", "#d62728"], autopct='%1.1f%%', startangle=90, explode=(0, 0.1),
            textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title("Míra odlehčení zátěže (L1 vs L2 offloading)", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cascaded_pipeline_throughput.png"), dpi=300)
    plt.close()
    print(f"[OK] Saved {PLOT_DIR}/cascaded_pipeline_throughput.png")
    
    res_dict = {
        "l1_xgb": {
            "single_latency_us": single_lat_xgb_us,
            "batch_latency_ms": lat_l1_batch_per_flow_ms,
            "throughput_flows_sec": throughput_xgb,
            "metrics": m_xgb
        },
        "l2_cnn": {
            "single_latency_us": single_lat_cnn_us,
            "batch_latency_ms": lat_l2_batch_per_flow_ms,
            "throughput_flows_sec": throughput_cnn,
            "metrics": m_cnn
        },
        "cascaded_pipeline": {
            "single_latency_us": single_lat_cascade_us,
            "batch_latency_ms": lat_cascade_batch_per_flow_ms,
            "throughput_flows_sec": throughput_cascade,
            "escalation_rate_pct": escalation_rate_pct,
            "metrics": m_cas
        }
    }
    with open(os.path.join(EVAL_DIR, "cascaded_pipeline_results.json"), "w") as f:
        json.dump(res_dict, f, indent=4)

if __name__ == "__main__":
    benchmark_cascaded_pipeline()
