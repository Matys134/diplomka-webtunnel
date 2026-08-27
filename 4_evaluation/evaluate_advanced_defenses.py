#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

sys.path.append("2_data_pipeline")
sys.path.append("3_models")
from train_1d_cnn import WebTunnel1DCNN

PROCESSED_DIR = "data/processed"
PLOT_DIR = "4_evaluation/plots"
TABLE_DIR = "0_thesis_text/tables"
EVAL_DIR = "4_evaluation"

def apply_traffic_morphing(X_seq, y_bin, morph_level: str = "none"):
    """
    Simulates Traffic Morphing & Burst Reshaping (Huma NDSS 2026 / Wright et al.):
    - 'length_pad': Intra-frame Gaussian padding (64 B).
    - 'timing_jit': Micro-timing jitter (25 ms).
    - 'burst_reshape': Morphing circuit setup burst (first 15 packets) into cover stream.
    - 'full_morphing': Length padding + Timing jitter + Burst reshaping + Bin quantization.
    """
    X_mod = X_seq.copy()
    total_orig_bytes = 0
    total_added_bytes = 0
    
    for i in range(len(y_bin)):
        if y_bin[i] == 1:
            for seq_idx in range(X_mod.shape[1]):
                norm_len = X_mod[i, seq_idx, 0]
                norm_iat = X_mod[i, seq_idx, 1]
                if abs(norm_len) < 1e-4:
                    continue
                orig_bytes = abs(norm_len) * 1500.0
                total_orig_bytes += orig_bytes
                
                new_bytes = orig_bytes
                new_iat = norm_iat
                
                if morph_level in ["length_pad", "full_morphing"]:
                    pad = max(0.0, np.random.normal(96.0, 30.0))
                    total_added_bytes += pad
                    new_bytes = min(1500.0, orig_bytes + pad)
                    
                if morph_level in ["timing_jit", "full_morphing"]:
                    orig_dt = np.expm1(norm_iat * 10.0)
                    jitter = np.random.uniform(0.005, 0.035)
                    new_iat = np.clip(np.log1p(orig_dt + jitter) / 10.0, 0.0, 1.0)
                    
                if morph_level in ["burst_reshape", "full_morphing"]:
                    # Obfuscate initial Tor circuit handshake (packets 0 to 15)
                    if seq_idx < 15:
                        # Morph length into typical WebSocket small frames (60-200B) or merged frame
                        if np.random.rand() > 0.4:
                            new_bytes = np.random.choice([64.0, 128.0, 256.0, 512.0, 1420.0])
                        orig_dt = np.expm1(new_iat * 10.0)
                        new_iat = np.clip(np.log1p(orig_dt + np.random.uniform(0.01, 0.08)) / 10.0, 0.0, 1.0)
                        
                sign = 1.0 if norm_len > 0 else -1.0
                X_mod[i, seq_idx, 0] = sign * (new_bytes / 1500.0)
                X_mod[i, seq_idx, 1] = new_iat
                
    overhead_pct = (total_added_bytes / max(total_orig_bytes, 1.0)) * 100.0 if total_orig_bytes > 0 else 0.0
    return X_mod, overhead_pct

def extract_features_from_seq_batch(X_seq_batch):
    all_feats = []
    for i in range(len(X_seq_batch)):
        raw_lens = np.abs(X_seq_batch[i, :, 0]) * 1500.0
        raw_signs = np.sign(X_seq_batch[i, :, 0])
        valid_mask = raw_lens > 10.0
        
        lens = raw_lens[valid_mask]
        signs = raw_signs[valid_mask]
        
        if len(lens) == 0:
            lens = np.array([50.0])
            signs = np.array([1.0])
            
        up_lens = lens[signs > 0]
        if len(up_lens) == 0: up_lens = np.array([50.0])
        down_lens = lens[signs < 0]
        if len(down_lens) == 0: down_lens = np.array([50.0])
        
        iats = np.expm1(X_seq_batch[i, :, 1] * 10.0)
        
        f = [
            np.min(lens), np.max(lens), np.mean(lens), np.std(lens), 0.0,
            np.percentile(lens, 10), np.percentile(lens, 25), np.percentile(lens, 50), np.percentile(lens, 75), np.percentile(lens, 90),
            np.min(up_lens), np.max(up_lens), np.mean(up_lens), np.std(up_lens),
            np.percentile(up_lens, 10), np.percentile(up_lens, 25), np.percentile(up_lens, 50), np.percentile(up_lens, 75), np.percentile(up_lens, 90),
            np.min(down_lens), np.max(down_lens), np.mean(down_lens), np.std(down_lens),
            np.percentile(down_lens, 10), np.percentile(down_lens, 25), np.percentile(down_lens, 50), np.percentile(down_lens, 75), np.percentile(down_lens, 90),
            np.min(iats), np.max(iats), np.mean(iats), np.std(iats),
            np.percentile(iats, 10), np.percentile(iats, 25), np.percentile(iats, 50), np.percentile(iats, 75), np.percentile(iats, 90),
            5.0, 5.0, 1.0, np.mean(lens)*5, np.std(lens)*5, np.mean(iats)*5, np.std(iats)*5,
            len(up_lens)/max(len(lens),1), np.sum(up_lens)/max(np.sum(lens),1), len(lens), np.sum(lens)
        ]
        all_feats.append(f)
    return np.array(all_feats, dtype=np.float32)

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    seq_data = np.load(os.path.join(PROCESSED_DIR, "sequence_dataset.npz"), allow_pickle=True)
    X_test_seq, y_test = seq_data["X_test"], seq_data["y_test"]
    
    # 1. Load trained 1D-CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model_cnn.load_state_dict(torch.load("3_models/saved_models/1d_cnn_best.pt", map_location=device))
    model_cnn.eval()
    
    # 2. Load trained XGBoost
    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model("3_models/saved_models/xgboost_baseline.json")
    
    print("=== Evaluating Advanced Traffic Defenses & Morphing ===")
    
    defense_configs = [
        ("1. Bez obrany (Současný WebTunnel)", "none"),
        ("2. Samostatný Adaptive Padding (96 B)", "length_pad"),
        ("3. Samostatný Timing Jitter (25 ms)", "timing_jit"),
        ("4. Circuit Handshake Morphing", "burst_reshape"),
        ("5. Komplexní Traffic Morphing (Hybrid)", "full_morphing"),
    ]
    
    results = []
    
    for name, m_level in defense_configs:
        X_mod, overhead = apply_traffic_morphing(X_test_seq, y_test, morph_level=m_level)
        
        # 1D-CNN Evaluation
        X_seq_t = torch.from_numpy(np.transpose(X_mod, (0, 2, 1))).to(device)
        with torch.no_grad():
            probs_cnn = model_cnn(X_seq_t).squeeze(-1).cpu().numpy()
        preds_cnn = (probs_cnn >= 0.5).astype(int)
        acc_cnn = accuracy_score(y_test, preds_cnn)
        pr_cnn = average_precision_score(y_test, probs_cnn)
        
        # XGBoost Evaluation
        X_mod_tab = extract_features_from_seq_batch(X_mod)
        probs_xgb = clf_xgb.predict_proba(X_mod_tab)[:, 1]
        preds_xgb = (probs_xgb >= 0.5).astype(int)
        acc_xgb = accuracy_score(y_test, preds_xgb)
        pr_xgb = average_precision_score(y_test, probs_xgb)
        
        results.append({
            "name": name,
            "morph_level": m_level,
            "overhead": overhead,
            "cnn_acc": acc_cnn,
            "cnn_pr": pr_cnn,
            "xgb_acc": acc_xgb,
            "xgb_pr": pr_xgb
        })
        print(f"{name:<42} | Režie: {overhead:5.1f}% | CNN Acc: {acc_cnn*100:5.1f}% (PR: {pr_cnn:.3f}) | XGB Acc: {acc_xgb*100:5.1f}%")
        
    # Generate LaTeX Table
    tex = r"""\begin{table}[htbp]
\centering
\caption{Experimentální vyhodnocení navržených protiopatření vůči XGBoost a 1D-CNN}
\label{tab:defense_evaluation}
\begin{tabular}{lcccc}
\hline
\textbf{Konfigurace protiopatření} & \textbf{Režie šířky pásma} & \textbf{1D-CNN Accuracy} & \textbf{1D-CNN PR-AUC} & \textbf{XGBoost Acc} \\
\hline
"""
    for r in results:
        tex += f"{r['name']} & {r['overhead']:.1f}\\% & {r['cnn_acc']*100:.1f}\\% & {r['cnn_pr']:.4f} & {r['xgb_acc']*100:.1f}\\% \\\\\n"
        
    tex += r"""\hline
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLE_DIR, "table_defense_evaluation.tex"), "w") as f:
        f.write(tex)
    print(f"\n[OK] Exported {TABLE_DIR}/table_defense_evaluation.tex")
    
    # Generate Plot
    plt.figure(figsize=(12, 6))
    names = [r["name"] for r in results]
    cnn_accs = [r["cnn_acc"] * 100 for r in results]
    xgb_accs = [r["xgb_acc"] * 100 for r in results]
    overheads = [r["overhead"] for r in results]
    
    x = np.arange(len(names))
    width = 0.28
    
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.set_ylabel('Úspěšnost detekce cenzora (%)', fontsize=12, fontweight='bold')
    b1 = ax1.bar(x - width/2, cnn_accs, width, label='1D-CNN Přesnost (%)', color='#d62728', alpha=0.85)
    b2 = ax1.bar(x + width/2, xgb_accs, width, label='XGBoost Přesnost (%)', color='#ff7f0e', alpha=0.85)
    ax1.set_ylim(0, 115)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Náhodné hádání (50 %)')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Režie šířky pásma (%)', color='#1f77b4', fontsize=12, fontweight='bold')
    b3 = ax2.plot(x, overheads, color='#1f77b4', marker='o', linewidth=2.5, label='Režie pásma (%)')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim(-5, 40)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    plt.title("Účinnost navržených protiopatření: Snížení detekce vs. Režie protokolu", fontsize=14, fontweight="bold")
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "defense_comprehensive_evaluation.png"), dpi=300)
    plt.close()
    print(f"[OK] Saved {PLOT_DIR}/defense_comprehensive_evaluation.png")

if __name__ == "__main__":
    main()
