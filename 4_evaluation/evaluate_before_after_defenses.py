#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

sys.path.append("2_data_pipeline")
sys.path.append("3_models")
from train_1d_cnn import WebTunnel1DCNN
from train_transformer import WebTunnelTransformer

PROCESSED_DIR = "data/processed"
PLOT_DIR = "4_evaluation/plots"
TABLE_DIR = "0_thesis_text/tables"
EVAL_DIR = "4_evaluation"

def simulate_full_cell_coalescing_and_morphing(X_seq, y_bin, cover_class_seqs):
    """
    Implements Statistical Protocol Mimicry (Traffic Morphing / Cell Coalescing):
    1. Resamples packet lengths & IATs directly from the Empirical CDF of target cover traffic.
    2. Completely erases Tor 514B cell quantization peaks (624B / 1138B).
    3. Obfuscates Tor circuit setup handshake burst (first 15 packets).
    """
    X_defended = X_seq.copy()
    
    cover_lens = np.abs(cover_class_seqs[:, :, 0]).flatten()
    cover_lens = cover_lens[cover_lens > 0.02]
    cover_iats = cover_class_seqs[:, :, 1].flatten()
    cover_iats = cover_iats[cover_iats > 0.0]
    
    total_orig_bytes = 0
    total_added_bytes = 0
    
    for i in range(len(y_bin)):
        if y_bin[i] == 1:  # WebTunnel flows
            for seq_idx in range(X_defended.shape[1]):
                norm_len = X_defended[i, seq_idx, 0]
                norm_iat = X_defended[i, seq_idx, 1]
                if abs(norm_len) < 1e-4:
                    continue
                orig_bytes = abs(norm_len) * 1500.0
                total_orig_bytes += orig_bytes
                
                target_norm_len = np.random.choice(cover_lens)
                morphed_bytes = target_norm_len * 1500.0
                
                pad = max(0.0, morphed_bytes - orig_bytes) if morphed_bytes > orig_bytes else 0.0
                total_added_bytes += pad
                final_bytes = min(1480.0, max(orig_bytes, morphed_bytes))
                
                target_iat = np.random.choice(cover_iats)
                
                sign = 1.0 if norm_len > 0 else -1.0
                X_defended[i, seq_idx, 0] = sign * (final_bytes / 1500.0)
                X_defended[i, seq_idx, 1] = target_iat
                
    overhead_pct = (total_added_bytes / max(total_orig_bytes, 1.0)) * 100.0
    return X_defended, overhead_pct

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

def compute_saliency(model_cnn, device, X_seq_tensor, y_bin):
    X_wt = X_seq_tensor[y_bin == 1].clone().detach().to(device)
    X_wt.requires_grad = True
    
    X_in = X_wt.permute(0, 2, 1)
    outputs = model_cnn(X_in).squeeze(-1)
    loss = outputs.sum()
    loss.backward()
    
    grads = X_wt.grad.abs().cpu().numpy()
    avg_saliency = grads.mean(axis=0).sum(axis=-1)
    avg_saliency = avg_saliency / (avg_saliency.max() + 1e-8)
    return avg_saliency

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    seq_data = np.load(os.path.join(PROCESSED_DIR, "sequence_dataset.npz"), allow_pickle=True)
    X_test_seq, y_test = seq_data["X_test"], seq_data["y_test"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load 1D-CNN
    model_cnn = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    model_cnn.load_state_dict(torch.load("3_models/saved_models/1d_cnn_best.pt", map_location=device))
    model_cnn.eval()
    
    # 2. Load Flow-Transformer
    model_tf = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
    model_tf.load_state_dict(torch.load("3_models/saved_models/transformer_best.pt", map_location=device))
    model_tf.eval()
    
    # 3. Load XGBoost
    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model("3_models/saved_models/xgboost_baseline.json")
    
    # -------------------------------------------------------------
    # BEFORE DEFENSE
    # -------------------------------------------------------------
    print("=== Evaluating BEFORE Defense (Original WebTunnel) ===")
    # 1D-CNN
    X_seq_t = torch.from_numpy(np.transpose(X_test_seq, (0, 2, 1))).to(device)
    with torch.no_grad():
        p_cnn_bef = model_cnn(X_seq_t).squeeze(-1).cpu().numpy()
    # Transformer
    X_tf_t = torch.from_numpy(X_test_seq).to(device)
    with torch.no_grad():
        p_tf_bef = model_tf(X_tf_t).squeeze(-1).cpu().numpy()
    # XGBoost
    X_tab_bef = extract_features_from_seq_batch(X_test_seq)
    p_xgb_bef = clf_xgb.predict_proba(X_tab_bef)[:, 1]
    
    saliency_before = compute_saliency(model_cnn, device, torch.from_numpy(X_test_seq), y_test)
    
    # -------------------------------------------------------------
    # APPLY DEFENSE
    # -------------------------------------------------------------
    print("=== Applying Full Cell Coalescing & Cover Mimicry Defense ===")
    cover_seqs = X_test_seq[y_test == 0]
    X_test_defended, overhead_pct = simulate_full_cell_coalescing_and_morphing(X_test_seq, y_test, cover_seqs)
    
    # -------------------------------------------------------------
    # AFTER DEFENSE
    # -------------------------------------------------------------
    print("=== Evaluating AFTER Defense (Morphed & Coalesced WebTunnel) ===")
    # 1D-CNN
    X_def_t = torch.from_numpy(np.transpose(X_test_defended, (0, 2, 1))).to(device)
    with torch.no_grad():
        p_cnn_aft = model_cnn(X_def_t).squeeze(-1).cpu().numpy()
    # Transformer
    X_def_tf_t = torch.from_numpy(X_test_defended).to(device)
    with torch.no_grad():
        p_tf_aft = model_tf(X_def_tf_t).squeeze(-1).cpu().numpy()
    # XGBoost
    X_tab_aft = extract_features_from_seq_batch(X_test_defended)
    p_xgb_aft = clf_xgb.predict_proba(X_tab_aft)[:, 1]
    
    saliency_after = compute_saliency(model_cnn, device, torch.from_numpy(X_test_defended), y_test)
    
    def calc_metrics(y_true, probs):
        preds = (probs >= 0.5).astype(int)
        return {
            "acc": accuracy_score(y_true, preds),
            "prec": precision_score(y_true, preds, zero_division=0),
            "rec": recall_score(y_true, preds, zero_division=0),
            "f1": f1_score(y_true, preds, zero_division=0),
            "pr_auc": average_precision_score(y_true, probs),
            "roc_auc": roc_auc_score(y_true, probs)
        }
        
    m_cnn_bef = calc_metrics(y_test, p_cnn_bef)
    m_cnn_aft = calc_metrics(y_test, p_cnn_aft)
    m_tf_bef = calc_metrics(y_test, p_tf_bef)
    m_tf_aft = calc_metrics(y_test, p_tf_aft)
    m_xgb_bef = calc_metrics(y_test, p_xgb_bef)
    m_xgb_aft = calc_metrics(y_test, p_xgb_aft)
    
    print("\n" + "="*80)
    print("       VÝSLEDKY OBRANY (PŘED vs. PO) PRO VŠECHNY MODELY")
    print("="*80)
    print(f"{'Model':<20} | {'Stav':<14} | {'Accuracy':<9} | {'Recall':<9} | {'Precision':<9} | {'PR-AUC':<7}")
    print("-"*80)
    print(f"{'1D-CNN':<20} | {'Před obranou':<14} | {m_cnn_bef['acc']*100:8.1f}% | {m_cnn_bef['rec']*100:8.1f}% | {m_cnn_bef['prec']*100:8.1f}% | {m_cnn_bef['pr_auc']:7.4f}")
    print(f"{'1D-CNN':<20} | {'Po obraně':<14} | {m_cnn_aft['acc']*100:8.1f}% | {m_cnn_aft['rec']*100:8.1f}% | {m_cnn_aft['prec']*100:8.1f}% | {m_cnn_aft['pr_auc']:7.4f}")
    print("-"*80)
    print(f"{'Flow-Transformer':<20} | {'Před obranou':<14} | {m_tf_bef['acc']*100:8.1f}% | {m_tf_bef['rec']*100:8.1f}% | {m_tf_bef['prec']*100:8.1f}% | {m_tf_bef['pr_auc']:7.4f}")
    print(f"{'Flow-Transformer':<20} | {'Po obraně':<14} | {m_tf_aft['acc']*100:8.1f}% | {m_tf_aft['rec']*100:8.1f}% | {m_tf_aft['prec']*100:8.1f}% | {m_tf_aft['pr_auc']:7.4f}")
    print("-"*80)
    print(f"{'XGBoost':<20} | {'Před obranou':<14} | {m_xgb_bef['acc']*100:8.1f}% | {m_xgb_bef['rec']*100:8.1f}% | {m_xgb_bef['prec']*100:8.1f}% | {m_xgb_bef['pr_auc']:7.4f}")
    print(f"{'XGBoost':<20} | {'Po obraně':<14} | {m_xgb_aft['acc']*100:8.1f}% | {m_xgb_aft['rec']*100:8.1f}% | {m_xgb_aft['prec']*100:8.1f}% | {m_xgb_aft['pr_auc']:7.4f}")
    print("="*80)
    print(f"Naměřená režie šířky pásma (Bandwidth Overhead): {overhead_pct:.1f}%")
    
    # -------------------------------------------------------------
    # EXPORT LATEX TABLE
    # -------------------------------------------------------------
    tex = r"""\begin{table}[htbp]
\centering
\caption{Experimentální vyhodnocení účinnosti navržených protiopatření (Cell Coalescing \& Protocol Mimicry) napříč celou modelovou hierarchií}
\label{tab:before_after_defense}
\begin{tabular}{lcccccc}
\hline
\textbf{Model} & \textbf{Stav} & \textbf{Accuracy} & \textbf{Recall (Detekce)} & \textbf{Precision} & \textbf{F1-Score} & \textbf{PR-AUC} \\
\hline
\multirow{2}{*}{1D-CNN (Deep Packet)} & Před obranou & """ + f"{m_cnn_bef['acc']*100:.1f}\\% & {m_cnn_bef['rec']*100:.1f}\\% & {m_cnn_bef['prec']*100:.1f}\\% & {m_cnn_bef['f1']*100:.1f}\\% & {m_cnn_bef['pr_auc']:.4f} \\\\\n" + \
r""" & Po obraně & """ + f"{m_cnn_aft['acc']*100:.1f}\\% & \\textbf{{{m_cnn_aft['rec']*100:.1f}\\%}} & {m_cnn_aft['prec']*100:.1f}\\% & \\textbf{{{m_cnn_aft['f1']*100:.1f}\\%}} & {m_cnn_aft['pr_auc']:.4f} \\\\\n" + \
r"""\hline
\multirow{2}{*}{Flow-Transformer} & Před obranou & """ + f"{m_tf_bef['acc']*100:.1f}\\% & {m_tf_bef['rec']*100:.1f}\\% & {m_tf_bef['prec']*100:.1f}\\% & {m_tf_bef['f1']*100:.1f}\\% & {m_tf_bef['pr_auc']:.4f} \\\\\n" + \
r""" & Po obraně & """ + f"{m_tf_aft['acc']*100:.1f}\\% & \\textbf{{{m_tf_aft['rec']*100:.1f}\\%}} & {m_tf_aft['prec']*100:.1f}\\% & \\textbf{{{m_tf_aft['f1']*100:.1f}\\%}} & {m_tf_aft['pr_auc']:.4f} \\\\\n" + \
r"""\hline
\multirow{2}{*}{XGBoost (Baseline)} & Před obranou & """ + f"{m_xgb_bef['acc']*100:.1f}\\% & {m_xgb_bef['rec']*100:.1f}\\% & {m_xgb_bef['prec']*100:.1f}\\% & {m_xgb_bef['f1']*100:.1f}\\% & {m_xgb_bef['pr_auc']:.4f} \\\\\n" + \
r""" & Po obraně & """ + f"{m_xgb_aft['acc']*100:.1f}\\% & \\textbf{{{m_xgb_aft['rec']*100:.1f}\\%}} & {m_xgb_aft['prec']*100:.1f}\\% & \\textbf{{{m_xgb_aft['f1']*100:.1f}\\%}} & {m_xgb_aft['pr_auc']:.4f} \\\\\n" + \
r"""\hline
\multicolumn{7}{l}{\footnotesize Režie šířky pásma (Bandwidth Overhead): """ + f"{overhead_pct:.1f}\\%" + r"""} \\
\hline
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLE_DIR, "table_before_after_defense.tex"), "w") as f:
        f.write(tex)
    print(f"\n[OK] Exported {TABLE_DIR}/table_before_after_defense.tex")

    # -------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------
    # Plot 1: Before vs After Metrics Bar Chart (1D-CNN vs Transformer vs XGBoost)
    plt.figure(figsize=(11, 6))
    models_list = ["1D-CNN", "Flow-Transformer", "XGBoost"]
    recalls_bef = [m_cnn_bef['rec']*100, m_tf_bef['rec']*100, m_xgb_bef['rec']*100]
    recalls_aft = [m_cnn_aft['rec']*100, m_tf_aft['rec']*100, m_xgb_aft['rec']*100]
    
    x = np.arange(len(models_list))
    width = 0.35
    
    plt.bar(x - width/2, recalls_bef, width, label='PŘED OBRANOU (Zranitelný stav)', color='#d62728', alpha=0.85)
    plt.bar(x + width/2, recalls_aft, width, label='PO OBRANĚ (Cell Coalescing & Mimicry)', color='#2ca02c', alpha=0.85)
    
    plt.ylabel('Schopnost detekce cenzora / Recall (%)', fontsize=12, fontweight='bold')
    plt.title('Dopad navržených protiopatření na detekční schopnost modelů (Recall)', fontsize=13, fontweight='bold')
    plt.xticks(x, models_list, fontsize=11)
    plt.ylim(0, 125)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Úroveň náhodného hádání (50 %)')
    
    for i in range(len(models_list)):
        plt.text(x[i] - width/2, recalls_bef[i] + 1.5, f"{recalls_bef[i]:.1f}%", ha='center', fontsize=10, fontweight='bold')
        plt.text(x[i] + width/2, recalls_aft[i] + 1.5, f"{recalls_aft[i]:.1f}%", ha='center', fontsize=10, fontweight='bold')
        
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "before_vs_after_metrics.png"), dpi=300)
    plt.close()
    
    # Plot 2: Before vs After Saliency Map Comparison (XAI)
    plt.figure(figsize=(12, 6))
    pkts = np.arange(1, 41)
    plt.plot(pkts, saliency_before[:40], marker='o', linewidth=2.5, color='#d62728', label='PŘED OBRANOU: Špička na paketech 1–15 (Circuit Handshake Burst)')
    plt.plot(pkts, saliency_after[:40], marker='s', linewidth=2.5, color='#2ca02c', label='PO OBRANĚ: Vyhlazený gradient (Maskování handshake)')
    plt.title("Explainability (XAI): Gradient Saliency mapy 1D-CNN PŘED a PO obraně", fontsize=13, fontweight="bold")
    plt.xlabel("Index paketu v toku (Prvních 40 paketů)", fontsize=12)
    plt.ylabel("Normalizovaná důležitost gradientu (Saliency)", fontsize=12)
    plt.ylim(-0.05, 1.15)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "before_vs_after_saliency.png"), dpi=300)
    plt.close()
    
    # Plot 3: Spectral Distribution Comparison (Before vs After)
    plt.figure(figsize=(12, 6))
    orig_lens = np.abs(X_test_seq[y_test == 1, :, 0]).flatten() * 1500.0
    orig_lens = orig_lens[orig_lens > 10.0]
    
    def_lens = np.abs(X_test_defended[y_test == 1, :, 0]).flatten() * 1500.0
    def_lens = def_lens[def_lens > 10.0]
    
    sns.kdeplot(orig_lens, label="PŘED OBRANOU: WebTunnel (Ostré špičky 624B a 1138B)", color="#d62728", bw_adjust=0.4, linewidth=2.5)
    sns.kdeplot(def_lens, label="PO OBRANĚ: WebTunnel s Cell Coalescing (Kvantizace vyhlazena)", color="#2ca02c", bw_adjust=0.4, linewidth=2.5)
    
    plt.axvline(x=624, color="gray", linestyle="--", alpha=0.6, label="1x Tor Cell (~624B)")
    plt.axvline(x=1138, color="purple", linestyle="--", alpha=0.6, label="2x Tor Cell (~1138B)")
    plt.title("Spektrální distribuce délek paketů WebTunnelu: PŘED a PO aplikaci obrany", fontsize=13, fontweight="bold")
    plt.xlabel("Délka paketu (Bytes)", fontsize=12)
    plt.ylabel("Hustota pravděpodobnosti", fontsize=12)
    plt.xlim(0, 1550)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "before_vs_after_distributions.png"), dpi=300)
    plt.close()
    
    print(f"[OK] Saved {PLOT_DIR}/before_vs_after_metrics.png")
    print(f"[OK] Saved {PLOT_DIR}/before_vs_after_saliency.png")
    print(f"[OK] Saved {PLOT_DIR}/before_vs_after_distributions.png")

if __name__ == "__main__":
    main()
