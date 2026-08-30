#!/usr/bin/env python3
"""
Base Rate Fallacy & ISP Line-Rate Operational Feasibility Benchmark:
1. Base Rate Fallacy simulation across network core/edge prevalences (alpha = 10^-6 to 10^-1).
2. Host-Based Bayesian Multi-Flow Aggregation (Jansen et al. NDSS 2024 modeling).
3. Computational Line-Rate Inspection Throughput & Latency Benchmark.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import EVALUATION_DIR, PLOTS_DIR, setup_matplotlib_style


def bayes_precision(tpr, fpr, alpha):
    """Computes Bayesian Precision P(WebTunnel | Alarm) given base rate alpha."""
    numerator = tpr * alpha
    denominator = numerator + fpr * (1.0 - alpha)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    setup_matplotlib_style()

    models = ["xgboost", "1d_cnn", "transformer"]
    model_labels = {
        "xgboost": "XGBoost (Ryzen 9800X3D)",
        "1d_cnn": "1D-CNN (RTX 5070 Ti)",
        "transformer": "Flow-Transformer (RTX 5070 Ti)"
    }

    results = {}
    for m in models:
        res_file = os.path.join(EVALUATION_DIR, f"{m}_results.json")
        if os.path.exists(res_file):
            with open(res_file) as f:
                results[m] = json.load(f)

    # 1. Base Rate Fallacy: FDR vs Alpha Curve
    alphas = np.logspace(-6, -1, 200)
    fpr_scenarios = [1e-2, 1e-3, 1e-4, 1e-5]
    tpr = results.get("xgboost", {}).get("recall", 0.99)

    plt.figure(figsize=(10, 6))
    for fpr in fpr_scenarios:
        fdrs = []
        for a in alphas:
            prec = bayes_precision(tpr, fpr, a)
            fdr = (1.0 - prec) * 100.0
            fdrs.append(fdr)
        plt.plot(alphas, fdrs, label=f"Klasifikátor FPR = {fpr:.0e}", linewidth=2.5)

    plt.xscale("log")
    plt.axvline(x=1e-4, color="black", linestyle="--", alpha=0.7, label=r"ISP Edge Node ($\alpha=10^{-4}$)")
    plt.axvline(x=1e-6, color="red", linestyle=":", alpha=0.7, label=r"Páteřní Tier-1 síť ($\alpha=10^{-6}$)")
    plt.title("Base Rate Fallacy: Míra falešných poplachů (FDR) vs. Prevalence WebTunnelu v síti")
    plt.xlabel(r"Prevalence WebTunnelu v síťovém provozu ($\alpha$)")
    plt.ylabel("Míra falešně obviněných spojení (FDR %)")
    plt.ylim(-5, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "base_rate_fallacy_fdr.png"))
    plt.close()

    # 2. Host-Based Multi-Flow Aggregation Curve (Jansen et al. NDSS 2024)
    alpha_edge = 1e-4
    flows_per_host = np.arange(1, 16)
    fpr_single = 1e-3
    tpr_single = tpr
    rho_ambient = 0.01

    fdr_independent = []
    fdr_correlated = []

    for m_flows in flows_per_host:
        tpr_m = tpr_single ** m_flows
        fpr_m = fpr_single ** m_flows
        prec_ind = bayes_precision(tpr_m, fpr_m, alpha_edge)
        fdr_independent.append((1.0 - prec_ind) * 100.0)

        effective_fpr = max(fpr_m, rho_ambient * (fpr_single ** 0.5))
        prec_corr = bayes_precision(tpr_m, effective_fpr, alpha_edge)
        fdr_correlated.append((1.0 - prec_corr) * 100.0)

    plt.figure(figsize=(10, 6))
    plt.plot(flows_per_host, fdr_independent, "o-", label=r"Nezávislý Bayesův ideál ($FPR^M$)", color="green", linewidth=2.5)
    plt.plot(flows_per_host, fdr_correlated, "s--", label=r"Reálné síťové korelace ($\rho_{ambient}=1\%$)", color="crimson", linewidth=2.5)
    plt.title(r"Host-Based Multi-Flow agregace cenzora na úrovni IP adresy ($\alpha=10^{-4}$)")
    plt.xlabel("Počet po sobě jdoucích podezřelých toků z jedné IP adresy (M)")
    plt.ylabel("False Discovery Rate (FDR %)")
    plt.xticks(flows_per_host)
    plt.ylim(-5, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "host_based_aggregation.png"))
    plt.close()

    # 3. Computational Line-Rate Inspection Throughput & Latency
    plt.figure(figsize=(10, 5))
    model_keys = [k for k in models if k in results]
    names = [model_labels[k] for k in model_keys]
    throughputs = [results[k].get("throughput_flows_per_sec", 0) for k in model_keys]

    bars = plt.bar(names, throughputs, color=["#2ca02c", "#1f77b4", "#ff7f0e"], edgecolor="black", alpha=0.85)
    plt.yscale("log")
    plt.ylabel("Průchodnost inspekce (Toků za sekundu - Log)")
    plt.title("Výpočetní výkonnostní benchmark modelů na reálném hardware")

    for bar, val in zip(bars, throughputs):
        plt.text(bar.get_x() + bar.get_width()/2, val * 1.15, f"{val:,.0f} toků/s", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "computational_benchmark.png"))
    plt.close()

    print(f"[OK] Polished evaluation figures generated in {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
