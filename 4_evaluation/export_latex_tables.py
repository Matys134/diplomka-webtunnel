#!/usr/bin/env python3
"""
Exports synchronized, publication-ready LaTeX tables for the thesis text (0_thesis_text/tables/):
- table_model_comparison.tex (5-Fold CV metrics, latency, and throughput)
"""
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import EVALUATION_DIR, LATEX_TABLES_DIR


def export_model_comparison_table():
    cv_file = os.path.join(EVALUATION_DIR, "cross_validation_results.json")
    xgb_file = os.path.join(EVALUATION_DIR, "xgboost_results.json")
    cnn_file = os.path.join(EVALUATION_DIR, "1d_cnn_results.json")
    tf_file = os.path.join(EVALUATION_DIR, "transformer_results.json")

    xgb_res = json.load(open(xgb_file)) if os.path.exists(xgb_file) else {}
    cnn_res = json.load(open(cnn_file)) if os.path.exists(cnn_file) else {}
    tf_res = json.load(open(tf_file)) if os.path.exists(tf_file) else {}
    cv_res = json.load(open(cv_file)) if os.path.exists(cv_file) else {}

    tex = r"""\begin{table}[htbp]
\centering
\caption{Srovnání klasifikačních modelů pro detekci WebTunnelu vůči Hard Negatives (5-Fold CV)}
\label{tab:model_comparison}
\begin{tabular}{lcccccc}
\hline
\textbf{Model} & \textbf{Hardware} & \textbf{Accuracy} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{Latence/flow} & \textbf{Propustnost} \\
\hline
"""
    # 1. XGBoost
    xgb_lat = f"{xgb_res.get('latency_ms_per_flow', 0):.4f}~ms"
    xgb_tp = f"{xgb_res.get('throughput_flows_per_sec', 0):,.0f}~toků/s"
    if "XGBoost" in cv_res:
        acc = f"{cv_res['XGBoost']['acc']['mean']*100:.1f} \\pm {cv_res['XGBoost']['acc']['std']*100:.1f}\\%"
        pr = f"{cv_res['XGBoost']['pr_auc']['mean']:.3f} \\pm {cv_res['XGBoost']['pr_auc']['std']:.3f}"
        roc = f"{cv_res['XGBoost']['roc_auc']['mean']:.3f} \\pm {cv_res['XGBoost']['roc_auc']['std']:.3f}"
    else:
        acc = f"{xgb_res.get('accuracy', 0)*100:.1f}\\%"
        pr = f"{xgb_res.get('pr_auc', 0):.3f}"
        roc = f"{xgb_res.get('roc_auc', 0):.3f}"
    tex += f"XGBoost (Baseline) & AMD Ryzen 9800X3D & ${acc}$ & ${pr}$ & ${roc}$ & {xgb_lat} & {xgb_tp} \\\\\n"

    # 2. 1D-CNN
    cnn_lat = f"{cnn_res.get('latency_ms_per_flow', 0):.4f}~ms"
    cnn_tp = f"{cnn_res.get('throughput_flows_per_sec', 0):,.0f}~toků/s"
    if "1D-CNN" in cv_res:
        acc = f"{cv_res['1D-CNN']['acc']['mean']*100:.1f} \\pm {cv_res['1D-CNN']['acc']['std']*100:.1f}\\%"
        pr = f"{cv_res['1D-CNN']['pr_auc']['mean']:.3f} \\pm {cv_res['1D-CNN']['pr_auc']['std']:.3f}"
        roc = f"{cv_res['1D-CNN']['roc_auc']['mean']:.3f} \\pm {cv_res['1D-CNN']['roc_auc']['std']:.3f}"
    else:
        acc = f"{cnn_res.get('accuracy', 0)*100:.1f}\\%"
        pr = f"{cnn_res.get('pr_auc', 0):.3f}"
        roc = f"{cnn_res.get('roc_auc', 0):.3f}"
    tex += f"1D-CNN (Deep Packet) & NVIDIA RTX 5070 Ti & ${acc}$ & ${pr}$ & ${roc}$ & {cnn_lat} & {cnn_tp} \\\\\n"

    # 3. Flow-Transformer
    tf_lat = f"{tf_res.get('latency_ms_per_flow', 0):.4f}~ms"
    tf_tp = f"{tf_res.get('throughput_flows_per_sec', 0):,.0f}~toků/s"
    if "Flow-Transformer" in cv_res:
        acc = f"{cv_res['Flow-Transformer']['acc']['mean']*100:.1f} \\pm {cv_res['Flow-Transformer']['acc']['std']*100:.1f}\\%"
        pr = f"{cv_res['Flow-Transformer']['pr_auc']['mean']:.3f} \\pm {cv_res['Flow-Transformer']['pr_auc']['std']:.3f}"
        roc = f"{cv_res['Flow-Transformer']['roc_auc']['mean']:.3f} \\pm {cv_res['Flow-Transformer']['roc_auc']['std']:.3f}"
    else:
        acc = f"{tf_res.get('accuracy', 0)*100:.1f}\\%"
        pr = f"{tf_res.get('pr_auc', 0):.3f}"
        roc = f"{tf_res.get('roc_auc', 0):.3f}"
    tex += f"Flow-Transformer & NVIDIA RTX 5070 Ti & ${acc}$ & ${pr}$ & ${roc}$ & {tf_lat} & {tf_tp} \\\\\n"

    tex += r"""\hline
\end{tabular}
\end{table}
"""
    out_file = os.path.join(LATEX_TABLES_DIR, "table_model_comparison.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[OK] Exported {out_file}")


def main():
    export_model_comparison_table()


if __name__ == "__main__":
    main()
