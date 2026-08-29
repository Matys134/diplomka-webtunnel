#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

EVAL_DIR = "4_evaluation"
PLOT_DIR = "4_evaluation/plots"

def bayes_precision(tpr, fpr, alpha):
    """Computes Bayesian Precision P(WebTunnel | Alarm) given base rate alpha."""
    numerator = tpr * alpha
    denominator = numerator + fpr * (1.0 - alpha)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Load model test predictions & metrics
    models = ["xgboost", "1d_cnn", "transformer"]
    model_labels = {
        "xgboost": "XGBoost (Ryzen 9800X3D)",
        "1d_cnn": "1D-CNN (RTX 5070 Ti)",
        "transformer": "Flow-Transformer (RTX 5070 Ti)"
    }
    
    results = {}
    for m in models:
        res_file = os.path.join(EVAL_DIR, f"{m}_results.json")
        if os.path.exists(res_file):
            with open(res_file) as f:
                results[m] = json.load(f)
                
    # 1. Base Rate Fallacy: FDR vs Alpha Curve
    alphas = np.logspace(-6, -1, 200) # from 10^-6 (core) to 10^-1 (edge)
    fpr_scenarios = [1e-2, 1e-3, 1e-4, 1e-5]
    # Dynamically extract TPR from evaluated models (defaults to 0.99 if not yet cached)
    tpr = results.get("xgboost", {}).get("metrics", {}).get("recall", 0.99)
    
    plt.figure(figsize=(10, 6))
    for fpr in fpr_scenarios:
        fdrs = []
        for a in alphas:
            prec = bayes_precision(tpr, fpr, a)
            fdr = (1.0 - prec) * 100.0
            fdrs.append(fdr)
        plt.plot(alphas, fdrs, label=f"Classifier FPR = {fpr:.0e}", linewidth=2.5)
        
    plt.xscale("log")
    plt.axvline(x=1e-4, color="black", linestyle="--", alpha=0.7, label="ISP Edge Node (alpha=10^-4)")
    plt.axvline(x=1e-6, color="red", linestyle=":", alpha=0.7, label="Tier-1 Backbone (alpha=10^-6)")
    plt.title("Base Rate Fallacy: False Discovery Rate (FDR) vs WebTunnel Prevalence (alpha)", fontsize=13, fontweight="bold")
    plt.xlabel("WebTunnel Prevalence in Network (alpha)", fontsize=12)
    plt.ylabel("False Discovery Rate (FDR %)", fontsize=12)
    plt.ylim(-5, 105)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "base_rate_fallacy_fdr.png"), dpi=300)
    plt.close()
    
    # 2. Host-Based Multi-Flow Aggregation with Correlated Noise Modeling (Jansen et al. NDSS 2024)
    m_flows = np.arange(1, 11)
    single_fpr = 1e-3
    alpha_edge = 1e-4
    alpha_core = 1e-6
    rho_corr = 0.01  # 1% correlated / persistent application noise
    
    host_fdrs_ideal_edge = []
    host_fdrs_ideal_core = []
    host_fdrs_corr_core = []
    
    for m in m_flows:
        # Scale TPR for M-flow host confirmation (TPR^m)
        eff_host_tpr = max(1e-12, float(tpr ** m))
        
        # Ideal independent false positive rate (FPR^m)
        eff_host_fpr_ideal = max(1e-18, float(single_fpr ** m))
        prec_ideal_edge = bayes_precision(eff_host_tpr, eff_host_fpr_ideal, alpha_edge)
        prec_ideal_core = bayes_precision(eff_host_tpr, eff_host_fpr_ideal, alpha_core)
        host_fdrs_ideal_edge.append((1.0 - prec_ideal_edge) * 100.0)
        host_fdrs_ideal_core.append((1.0 - prec_ideal_core) * 100.0)
        
        # Correlated error mixture: rho * single_fpr + (1 - rho) * single_fpr^m
        eff_host_fpr_corr = rho_corr * single_fpr + (1.0 - rho_corr) * eff_host_fpr_ideal
        prec_corr_core = bayes_precision(eff_host_tpr, eff_host_fpr_corr, alpha_core)
        host_fdrs_corr_core.append((1.0 - prec_corr_core) * 100.0)
        
    plt.figure(figsize=(10, 6))
    plt.plot(m_flows, host_fdrs_ideal_edge, marker='o', label="ISP Edge (alpha=10^-4, Ideální)", color="#1f77b4", linewidth=2.5)
    plt.plot(m_flows, host_fdrs_ideal_core, marker='s', label="Tier-1 Backbone (alpha=10^-6, Ideální)", color="#d62728", linewidth=2.5)
    plt.plot(m_flows, host_fdrs_corr_core, marker='^', linestyle='--', label="Tier-1 Backbone s korelujícím šumem (rho=1%)", color="#ff7f0e", linewidth=2.0)
    plt.title("Host-Based Multi-Flow Aggregation: Eliminace Base Rate Fallacy a vliv korelujícího šumu", fontsize=13, fontweight="bold")
    plt.xlabel("Počet agregovaných toků na hostitele (M)", fontsize=12)
    plt.ylabel("Host False Discovery Rate (FDR %)", fontsize=12)
    plt.ylim(-5, 105)
    plt.xticks(m_flows)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "host_based_aggregation.png"), dpi=300)
    plt.close()
    
    # 3. Clean Computational Benchmark Comparison (Log Scale Throughput + Text Labels)
    model_names_plot = []
    latencies = []
    throughputs = []
    
    for m in ["xgboost", "1d_cnn", "transformer"]:
        if m in results:
            model_names_plot.append(model_labels[m])
            latencies.append(results[m]["inference_latency_ms"])
            throughputs.append(results[m]["throughput_flows_sec"])
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    
    # Subplot 1: Latency
    bars1 = ax1.bar(model_names_plot, latencies, color=colors, alpha=0.85)
    ax1.set_title("Inference Latency per Flow (Lower is better)", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Latency (ms / flow)", fontsize=11)
    ax1.tick_params(axis='x', rotation=15)
    ax1.set_ylim(0, max(latencies) * 1.25)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f} ms", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Subplot 2: Throughput (Log Scale)
    bars2 = ax2.bar(model_names_plot, throughputs, color=colors, alpha=0.85)
    ax2.set_yscale('log')
    ax2.set_title("Inference Throughput (Log Scale - Higher is better)", fontweight="bold", fontsize=12)
    ax2.set_ylabel("Throughput (flows / sec, log scale)", fontsize=11)
    ax2.tick_params(axis='x', rotation=15)
    ax2.set_ylim(1e3, 1e7)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval * 1.25, f"{int(yval):,} flows/s", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "computational_benchmark.png"), dpi=300)
    plt.close()
    
    # Remove uninformative flat plots
    if os.path.exists(os.path.join(PLOT_DIR, "pr_curves_comparison.png")):
        os.remove(os.path.join(PLOT_DIR, "pr_curves_comparison.png"))
    if os.path.exists(os.path.join(PLOT_DIR, "pre_vs_post_handshake_comparison.png")):
        os.remove(os.path.join(PLOT_DIR, "pre_vs_post_handshake_comparison.png"))
        
    print(f"\n[OK] Polished evaluation figures generated in {PLOT_DIR}/")

if __name__ == "__main__":
    main()
