#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import confusion_matrix

sys.path.append("3_models")
from train_1d_cnn import WebTunnel1DCNN
from train_transformer import WebTunnelTransformer

PROCESSED_DIR = "data/processed"
PLOT_DIR = "4_evaluation/plots"
TABLE_DIR = "0_thesis_text/tables"

CLASS_NAMES = [
    "WebTunnel",
    "Direct Browsing",
    "WS Ticker",
    "WS Chat",
    "Video Stream",
    "Web Assets"
]

def generate_confusion_breakdown():
    print("=== Generating Class Breakdown & Confusion Matrices (All 3 Models) ===")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Load Data
    tab_data = np.load(os.path.join(PROCESSED_DIR, "tabular_dataset.npz"), allow_pickle=True)
    seq_data = np.load(os.path.join(PROCESSED_DIR, "sequence_dataset.npz"), allow_pickle=True)
    
    X_test_tab = tab_data["X_test"]
    y_test_bin = tab_data["y_test"]
    y_test_mul = tab_data["y_test_mul"] # 0 to 5
    X_test_seq = seq_data["X_test"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Get Predictions
    # XGBoost
    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model("3_models/saved_models/xgboost_baseline.json")
    p_xgb = clf_xgb.predict_proba(X_test_tab)[:, 1]
    preds_xgb = (p_xgb >= 0.5).astype(int)
    
    # 1D-CNN
    model_cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model_cnn.load_state_dict(torch.load("3_models/saved_models/1d_cnn_best.pt", map_location=device))
    model_cnn.eval()
    X_seq_t = torch.from_numpy(np.transpose(X_test_seq, (0, 2, 1))).to(device)
    with torch.no_grad():
        p_cnn = model_cnn(X_seq_t).squeeze(-1).cpu().numpy()
    preds_cnn = (p_cnn >= 0.5).astype(int)
    
    # Flow-Transformer
    model_tf = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
    model_tf.load_state_dict(torch.load("3_models/saved_models/transformer_best.pt", map_location=device))
    model_tf.eval()
    X_tf_t = torch.from_numpy(X_test_seq).to(device)
    with torch.no_grad():
        p_tf = model_tf(X_tf_t).squeeze(-1).cpu().numpy()
    preds_tf = (p_tf >= 0.5).astype(int)
    
    # 3. Class-by-Class Accuracy & Breakdown
    print("\n" + "="*90)
    print("        PŘESNOST DETEKCE ROZPADLÁ NA JEDNOTLIVÉ TŘÍDY (HARD NEGATIVES)")
    print("="*90)
    print(f"{'Třída provozu':<20} | {'Vzorků':<8} | {'1D-CNN':<16} | {'Transformer':<16} | {'XGBoost':<16}")
    print("-"*90)
    
    class_stats = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = (y_test_mul == cls_idx)
        count = int(np.sum(mask))
        if count == 0:
            continue
        target_label = 1 if cls_idx == 0 else 0
        cnn_correct = int(np.sum(preds_cnn[mask] == target_label))
        tf_correct = int(np.sum(preds_tf[mask] == target_label))
        xgb_correct = int(np.sum(preds_xgb[mask] == target_label))
        
        cnn_acc = (cnn_correct / count) * 100.0
        tf_acc = (tf_correct / count) * 100.0
        xgb_acc = (xgb_correct / count) * 100.0
        
        class_stats.append({
            "name": cls_name,
            "count": count,
            "cnn_acc": cnn_acc,
            "tf_acc": tf_acc,
            "xgb_acc": xgb_acc
        })
        print(f"{cls_name:<20} | {count:<8} | {cnn_correct}/{count} ({cnn_acc:5.1f}%) | {tf_correct}/{count} ({tf_acc:5.1f}%) | {xgb_correct}/{count} ({xgb_acc:5.1f}%)")
    print("="*90)
    
    # 4. Export LaTeX Table
    tex = r"""\begin{table}[htbp]
\centering
\caption{Dekompozice přesnosti klasifikátorů napříč všemi 6 třídami provozu (WebTunnel vs. Hard Negatives)}
\label{tab:class_breakdown}
\begin{tabular}{lccccc}
\hline
\textbf{Třída síťového provozu} & \textbf{Typ protokolu} & \textbf{Počet toků} & \textbf{1D-CNN} & \textbf{Transformer} & \textbf{XGBoost} \\
\hline
"""
    proto_map = {
        "WebTunnel": "Tor over HTTP/2 WSS",
        "Direct Browsing": "Čisté HTTPS (TLS 1.3)",
        "WS Ticker": "WSS Live JSON",
        "WS Chat": "WSS Interaktivní",
        "Video Stream": "HTTPS DASH/HLS",
        "Web Assets": "HTTPS Statický bundle"
    }
    for stat in class_stats:
        proto = proto_map.get(stat["name"], "HTTPS")
        tex += f"{stat['name']} & {proto} & {stat['count']} & {stat['cnn_acc']:.1f}\\% & {stat['tf_acc']:.1f}\\% & {stat['xgb_acc']:.1f}\\% \\\\\n"
    tex += r"""\hline
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLE_DIR, "table_class_breakdown.tex"), "w") as f:
        f.write(tex)
    print(f"[OK] Exported {TABLE_DIR}/table_class_breakdown.tex")
    
    # 5. Plot 3-Panel Side-by-Side Multi-Class Breakdown Heatmap
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    matrix_xgb = np.zeros((len(CLASS_NAMES), 2))
    matrix_cnn = np.zeros((len(CLASS_NAMES), 2))
    matrix_tf = np.zeros((len(CLASS_NAMES), 2))
    
    for cls_idx in range(len(CLASS_NAMES)):
        mask = (y_test_mul == cls_idx)
        if np.sum(mask) == 0:
            continue
        matrix_xgb[cls_idx, 0] = np.mean(preds_xgb[mask] == 0) * 100.0
        matrix_xgb[cls_idx, 1] = np.mean(preds_xgb[mask] == 1) * 100.0
        
        matrix_cnn[cls_idx, 0] = np.mean(preds_cnn[mask] == 0) * 100.0
        matrix_cnn[cls_idx, 1] = np.mean(preds_cnn[mask] == 1) * 100.0
        
        matrix_tf[cls_idx, 0] = np.mean(preds_tf[mask] == 0) * 100.0
        matrix_tf[cls_idx, 1] = np.mean(preds_tf[mask] == 1) * 100.0
        
    sns.heatmap(matrix_xgb, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=ax1,
                xticklabels=["Predikce: Benign", "Predikce: WebTunnel"],
                yticklabels=CLASS_NAMES, annot_kws={"size": 10, "fontweight": "bold"})
    ax1.set_title("XGBoost Baseline (Tabulární)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Skutečná třída (Ground Truth)", fontsize=11, fontweight="bold")
    
    sns.heatmap(matrix_cnn, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=ax2,
                xticklabels=["Predikce: Benign", "Predikce: WebTunnel"],
                yticklabels=[""] * len(CLASS_NAMES), annot_kws={"size": 10, "fontweight": "bold"})
    ax2.set_title("1D-CNN (Deep Packet)", fontsize=12, fontweight="bold", pad=10)
    
    sns.heatmap(matrix_tf, annot=True, fmt=".1f", cmap="Blues", cbar=True, ax=ax3,
                xticklabels=["Predikce: Benign", "Predikce: WebTunnel"],
                yticklabels=[""] * len(CLASS_NAMES), annot_kws={"size": 10, "fontweight": "bold"})
    ax3.set_title("Flow-Transformer ([CLS] Attention)", fontsize=12, fontweight="bold", pad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix_breakdown.png"), dpi=300)
    plt.close()
    print(f"[OK] Saved {PLOT_DIR}/confusion_matrix_breakdown.png")
    
    # 6. Plot Dedicated Binary Confusion Matrices Comparison (2x2 with Counts & Rates)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cm_models = [
        ("XGBoost Baseline", preds_xgb, axes[0]),
        ("1D-CNN Deep Packet", preds_cnn, axes[1]),
        ("Flow-Transformer", preds_tf, axes[2])
    ]
    
    for title, preds, ax in cm_models:
        cm = confusion_matrix(y_test_bin, preds)
        tn, fp, fn, tp = cm.ravel()
        labels = np.array([
            [f"TN (Správně Benign)\n{tn}\n({tn/(tn+fp)*100:.1f}%)", f"FP (Falešný poplach)\n{fp}\n({fp/(tn+fp)*100:.1f}%)"],
            [f"FN (Přehlédnutý WT)\n{fn}\n({fn/(fn+tp)*100:.1f}%)", f"TP (Zachycený WT)\n{tp}\n({tp/(fn+tp)*100:.1f}%)"]
        ])
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Predikce: Benign", "Predikce: WebTunnel"],
                    yticklabels=["Realita: Benign", "Realita: WebTunnel"],
                    annot_kws={"size": 10, "fontweight": "bold"})
        ax.set_title(f"{title}\nAccuracy: {(tp+tn)/(tp+tn+fp+fn)*100:.2f}% | Recall: {tp/(tp+fn)*100:.1f}%", fontsize=11, fontweight="bold", pad=8)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "neural_network_confusion_comparison.png"), dpi=300)
    plt.close()
    print(f"[OK] Saved {PLOT_DIR}/neural_network_confusion_comparison.png")

if __name__ == "__main__":
    generate_confusion_breakdown()
