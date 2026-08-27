#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

EVAL_DIR = "4_evaluation"
PLOT_DIR = "4_evaluation/plots"

def bayes_precision(tpr, fpr, alpha):
    """Calculates true Precision under Base Rate Fallacy: P(WT | Alarm)."""
    denom = (tpr * alpha) + (fpr * (1.0 - alpha))
    if denom == 0:
        return 0.0
    return (tpr * alpha) / denom

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    models = ["xgboost", "1d_cnn", "transformer"]
    model_labels = {"xgboost": "XGBoost (Baseline)", "1d_cnn": "1D-CNN (Deep Packet)", "transformer": "Flow-Transformer"}
    model_colors = {"xgboost": "#1f77b4", "1d_cnn": "#2ca02c", "transformer": "#d62728"}
    
    results = {}
    preds_data = {}
    
    for m in models:
        res_file = os.path.join(EVAL_DIR, f"{m}_results.json")
        pred_file = os.path.join(EVAL_DIR, f"{m}_test_preds.npz")
        
        if os.path.exists(res_file) and os.path.exists(pred_file):
            with open(res_file) as f:
                results[m] = json.load(f)
            preds_data[m] = np.load(pred_file)
            
    if not preds_data:
        print("No model predictions found! Run training scripts first.")
        return
        
    # 1. Plot Precision-Recall Curves
    plt.figure(figsize=(10, 6))
    for m, pdata in preds_data.items():
        y_true = pdata["y_test"]
        y_probs = pdata["y_probs"]
        prec, rec, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        plt.plot(rec, prec, label=f"{model_labels[m]} (PR-AUC = {pr_auc:.3f})", color=model_colors[m], linewidth=2.5)
        
    plt.title("Precision-Recall Curves for WebTunnel vs Hard Negatives", fontsize=14, fontweight="bold")
    plt.xlabel("Recall (True Positive Rate)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pr_curves_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Base Rate Fallacy Stress Test: FDR vs Prevalence Alpha
    alphas = np.logspace(-6, -1, 100)  # From 1 in 1,000,000 to 1 in 10
    tpr = 0.95  # Assumed 95% detection sensitivity
    fixed_fprs = [1e-2, 1e-3, 1e-4, 1e-5]
    
    plt.figure(figsize=(11, 6))
    for fpr in fixed_fprs:
        fdrs = []
        for a in alphas:
            prec = bayes_precision(tpr, fpr, a)
            fdr = (1.0 - prec) * 100.0  # False Discovery Rate %
            fdrs.append(fdr)
        plt.plot(alphas, fdrs, label=f"Classifier FPR = {fpr:.0e}", linewidth=2.5)
        
    plt.xscale("log")
    plt.axvline(x=1e-4, color="black", linestyle="--", alpha=0.7, label="ISP Edge Node (alpha=10^-4)")
    plt.axvline(x=1e-6, color="red", linestyle=":", alpha=0.7, label="Tier-1 Backbone (alpha=10^-6)")
    plt.title("Base Rate Fallacy: False Discovery Rate (FDR) vs WebTunnel Prevalence (alpha)", fontsize=14, fontweight="bold")
    plt.xlabel("WebTunnel Prevalence in Network (alpha)", fontsize=12)
    plt.ylabel("False Discovery Rate (FDR %)", fontsize=12)
    plt.ylim(-5, 105)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "base_rate_fallacy_fdr.png"), dpi=300)
    plt.close()
    
    # 3. Host-Based Aggregation Simulation (Jansen et al., NDSS 2024)
    # Evidence accumulation across M flows to the same host
    m_flows = np.arange(1, 21)
    # If single flow FPR is 1e-3, Bayes log-likelihood aggregation over M independent flows:
    # Host-level false positive probability: P_host_fp = FPR^M (approx)
    host_fdrs_1e4 = []
    host_fdrs_1e6 = []
    alpha_edge = 1e-4
    alpha_core = 1e-6
    
    for m in m_flows:
        # Accumulated log-likelihood reduces effective host-level FPR exponentially
        eff_host_fpr = max(1e-15, (1e-3) ** (m * 0.6))
        prec_edge = bayes_precision(tpr, eff_host_fpr, alpha_edge)
        prec_core = bayes_precision(tpr, eff_host_fpr, alpha_core)
        host_fdrs_1e4.append((1.0 - prec_edge) * 100.0)
        host_fdrs_1e6.append((1.0 - prec_core) * 100.0)
        
    plt.figure(figsize=(10, 6))
    plt.plot(m_flows, host_fdrs_1e4, marker='o', label="ISP Edge (alpha=10^-4)", color="#1f77b4", linewidth=2.5)
    plt.plot(m_flows, host_fdrs_1e6, marker='s', label="Tier-1 Backbone (alpha=10^-6)", color="#d62728", linewidth=2.5)
    plt.title("Host-Based Multi-Flow Aggregation: Defeating Base Rate Fallacy", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Aggregated Flows per Host (M)", fontsize=12)
    plt.ylabel("Host False Discovery Rate (FDR %)", fontsize=12)
    plt.ylim(-5, 105)
    plt.xticks(m_flows)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "host_based_aggregation.png"), dpi=300)
    plt.close()
    
    # 4. Latency vs Throughput Benchmark Comparison
    model_names_plot = []
    latencies = []
    throughputs = []
    
    for m in ["xgboost", "1d_cnn", "transformer"]:
        if m in results:
            model_names_plot.append(model_labels[m])
            latencies.append(results[m]["inference_latency_ms"])
            throughputs.append(results[m]["throughput_flows_sec"])
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(model_names_plot, latencies, color=["#1f77b4", "#2ca02c", "#d62728"])
    ax1.set_title("Inference Latency per Flow (Lower is better)", fontweight="bold")
    ax1.set_ylabel("Latency (ms / flow)")
    ax1.tick_params(axis='x', rotation=15)
    
    ax2.bar(model_names_plot, throughputs, color=["#1f77b4", "#2ca02c", "#d62728"])
    ax2.set_title("Inference Throughput (Higher is better)", fontweight="bold")
    ax2.set_ylabel("Throughput (flows / sec)")
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "computational_benchmark.png"), dpi=300)
    plt.close()
    
    print(f"\n[OK] All evaluation and publication figures generated in {PLOT_DIR}/")
    print(f"  1. {PLOT_DIR}/pr_curves_comparison.png")
    print(f"  2. {PLOT_DIR}/base_rate_fallacy_fdr.png")
    print(f"  3. {PLOT_DIR}/host_based_aggregation.png")
    print(f"  4. {PLOT_DIR}/computational_benchmark.png")

if __name__ == "__main__":
    main()
