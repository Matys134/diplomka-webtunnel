#!/usr/bin/env python3
import os
import json

EVAL_DIR = "4_evaluation"
TABLE_DIR = "0_thesis_text/tables"

def export_model_comparison_table():
    cv_file = os.path.join(EVAL_DIR, "cross_validation_results.json")
    xgb_file = os.path.join(EVAL_DIR, "xgboost_results.json")
    cnn_file = os.path.join(EVAL_DIR, "1d_cnn_results.json")
    tf_file = os.path.join(EVAL_DIR, "transformer_results.json")
    
    xgb_res = json.load(open(xgb_file)) if os.path.exists(xgb_file) else {}
    cnn_res = json.load(open(cnn_file)) if os.path.exists(cnn_file) else {}
    tf_res = json.load(open(tf_file)) if os.path.exists(tf_file) else {}
    cv_res = json.load(open(cv_file)) if os.path.exists(cv_file) else {}
    
    tex = r"""\begin{table}[htbp]
\centering
\caption{Srovnání klasifikačních modelů pro detekci WebTunnelu vůči Hard Negatives}
\label{tab:model_comparison}
\begin{tabular}{lcccccc}
\hline
\textbf{Model} & \textbf{Hardware} & \textbf{Accuracy} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{Latence/flow} & \textbf{Propustnost} \\
\hline
"""
    # XGBoost
    xgb_lat = f"{xgb_res.get('inference_latency_ms', 0):.4f} ms"
    xgb_tp = f"{xgb_res.get('throughput_flows_sec', 0):,.0f} flows/s"
    if "xgboost_cv_5fold" in cv_res:
        acc = f"{cv_res['xgboost_cv_5fold']['acc']['mean']*100:.1f} \\pm {cv_res['xgboost_cv_5fold']['acc']['std']*100:.1f}\\%"
        pr = f"{cv_res['xgboost_cv_5fold']['pr_auc']['mean']:.3f} \\pm {cv_res['xgboost_cv_5fold']['pr_auc']['std']:.3f}"
        roc = f"{cv_res['xgboost_cv_5fold']['roc_auc']['mean']:.3f} \\pm {cv_res['xgboost_cv_5fold']['roc_auc']['std']:.3f}"
    else:
        acc = f"{xgb_res.get('metrics', {}).get('accuracy', 0)*100:.1f}\\%"
        pr = f"{xgb_res.get('metrics', {}).get('pr_auc', 0):.3f}"
        roc = f"{xgb_res.get('metrics', {}).get('roc_auc', 0):.3f}"
    tex += f"XGBoost (Baseline) & Ryzen 9800X3D & ${acc}$ & ${pr}$ & ${roc}$ & {xgb_lat} & {xgb_tp} \\\\\n"
    
    # 1D-CNN
    cnn_lat = f"{cnn_res.get('inference_latency_ms', 0):.4f} ms"
    cnn_tp = f"{cnn_res.get('throughput_flows_sec', 0):,.0f} flows/s"
    if "1d_cnn_cv_5fold" in cv_res:
        acc = f"{cv_res['1d_cnn_cv_5fold']['acc']['mean']*100:.1f} \\pm {cv_res['1d_cnn_cv_5fold']['acc']['std']*100:.1f}\\%"
        pr = f"{cv_res['1d_cnn_cv_5fold']['pr_auc']['mean']:.3f} \\pm {cv_res['1d_cnn_cv_5fold']['pr_auc']['std']:.3f}"
        roc = f"{cv_res['1d_cnn_cv_5fold']['roc_auc']['mean']:.3f} \\pm {cv_res['1d_cnn_cv_5fold']['roc_auc']['std']:.3f}"
    else:
        acc = f"{cnn_res.get('metrics', {}).get('accuracy', 0)*100:.1f}\\%"
        pr = f"{cnn_res.get('metrics', {}).get('pr_auc', 0):.3f}"
        roc = f"{cnn_res.get('metrics', {}).get('roc_auc', 0):.3f}"
    tex += f"1D-CNN (Deep Packet) & RTX 5070 Ti & ${acc}$ & ${pr}$ & ${roc}$ & {cnn_lat} & {cnn_tp} \\\\\n"
    
    # Flow-Transformer
    tf_lat = f"{tf_res.get('inference_latency_ms', 0):.4f} ms"
    tf_tp = f"{tf_res.get('throughput_flows_sec', 0):,.0f} flows/s"
    acc = f"{tf_res.get('metrics', {}).get('accuracy', 0)*100:.1f}\\%"
    pr = f"{tf_res.get('metrics', {}).get('pr_auc', 0):.3f}"
    roc = f"{tf_res.get('metrics', {}).get('roc_auc', 0):.3f}"
    tex += f"Flow-Transformer & RTX 5070 Ti & ${acc}$ & ${pr}$ & ${roc}$ & {tf_lat} & {tf_tp} \\\\\n"
    
    tex += r"""\hline
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLE_DIR, "table_model_comparison.tex"), "w") as f:
        f.write(tex)
    print(f"[OK] Exported {TABLE_DIR}/table_model_comparison.tex")

def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    export_model_comparison_table()

if __name__ == "__main__":
    main()
