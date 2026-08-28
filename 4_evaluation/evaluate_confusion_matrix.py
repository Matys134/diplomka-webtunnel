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
    print("=== Generating Class Breakdown & Confusion Matrices ===")
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
    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model("3_models/saved_models/xgboost_baseline.json")
    p_xgb = clf_xgb.predict_proba(X_test_tab)[:, 1]
    preds_xgb = (p_xgb >= 0.5).astype(int)
    
    model_cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model_cnn.load_state_dict(torch.load("3_models/saved_models/1d_cnn_best.pt", map_location=device))
    model_cnn.eval()
    X_seq_t = torch.from_numpy(np.transpose(X_test_seq, (0, 2, 1))).to(device)
    with torch.no_grad():
        p_cnn = model_cnn(X_seq_t).squeeze(-1).cpu().numpy()
    preds_cnn = (p_cnn >= 0.5).astype(int)
    
    # 3. Class-by-Class Accuracy & Breakdown
    print("\n" + "="*75)
    print("        PŘESNOST DETEKCE ROZPADLÁ NA JEDNOTLIVÉ TŘÍDY (HARD NEGATIVES)")
    print("="*75)
    print(f"{'Třída provozu':<22} | {'Vzorků':<8} | {'1D-CNN Shoda':<15} | {'XGBoost Shoda':<15}")
    print("-"*75)
    
    class_stats = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = (y_test_mul == cls_idx)
        count = int(np.sum(mask))
        if count == 0:
            continue
        # For class 0 (WebTunnel), correct is 1. For classes 1..5, correct is 0.
        target_label = 1 if cls_idx == 0 else 0
        cnn_correct = int(np.sum(preds_cnn[mask] == target_label))
        xgb_correct = int(np.sum(preds_xgb[mask] == target_label))
        
        cnn_acc = (cnn_correct / count) * 100.0
        xgb_acc = (xgb_correct / count) * 100.0
        
        class_stats.append({
            "name": cls_name,
            "count": count,
            "cnn_acc": cnn_acc,
            "xgb_acc": xgb_acc
        })
        print(f"{cls_name:<22} | {count:<8} | {cnn_correct}/{count} ({cnn_acc:5.1f}%) | {xgb_correct}/{count} ({xgb_acc:5.1f}%)")
    print("="*75)
    
    # 4. Export LaTeX Table
    tex = r"""\begin{table}[htbp]
\centering
\caption{Dekompozice přesnosti klasifikátorů napříč všemi 6 třídami provozu (WebTunnel vs. Hard Negatives)}
\label{tab:class_breakdown}
\begin{tabular}{lcccc}
\hline
\textbf{Třída síťového provozu} & \textbf{Typ protokolu} & \textbf{Počet toků} & \textbf{1D-CNN Přesnost} & \textbf{XGBoost Přesnost} \\
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
        tex += f"{stat['name']} & {proto} & {stat['count']} & {stat['cnn_acc']:.1f}\\% & {stat['xgb_acc']:.1f}\\% \\\\\n"
    tex += r"""\hline
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLE_DIR, "table_class_breakdown.tex"), "w") as f:
        f.write(tex)
    print(f"[OK] Exported {TABLE_DIR}/table_class_breakdown.tex")
    
    # 5. Plot Heatmap Breakdown
    plt.figure(figsize=(10, 6))
    matrix_data = np.zeros((len(CLASS_NAMES), 2))
    
    for cls_idx in range(len(CLASS_NAMES)):
        mask = (y_test_mul == cls_idx)
        if np.sum(mask) == 0:
            continue
        p_cnn_sub = preds_cnn[mask]
        matrix_data[cls_idx, 0] = np.mean(p_cnn_sub == 0) * 100.0 # Classified Benign
        matrix_data[cls_idx, 1] = np.mean(p_cnn_sub == 1) * 100.0 # Classified WebTunnel
        
    sns.heatmap(matrix_data, annot=True, fmt=".1f", cmap="Blues", cbar=True,
                xticklabels=["Predikce: Legitimní (0)", "Predikce: WebTunnel (1)"],
                yticklabels=CLASS_NAMES, annot_kws={"size": 11, "fontweight": "bold"})
    plt.title("1D-CNN: Distribuce klasifikačních rozhodnutí napříč třídami (%)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Skutečná třída (Ground Truth)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix_breakdown.png"), dpi=300)
    plt.close()
    print(f"[OK] Saved {PLOT_DIR}/confusion_matrix_breakdown.png")

if __name__ == "__main__":
    generate_confusion_breakdown()
