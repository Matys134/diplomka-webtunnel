#!/usr/bin/env python3
"""
Order-Shuffle Control Experiment for Sequence Classifiers (1D-CNN and Flow-Transformer):

Hypothesis test recommended in docs/07-signoff.md:
  Randomly permute the temporal record order within each flow (train, val, and test alike)
  and retrain the 1D-CNN and the Flow-Transformer.

  - If models stay at ~100%: The sequence models are reading the order-invariant
    record length distribution (the 514 B Tor cell lattice).
  - If models drop significantly: The sequence models were relying on positional cues
    (element 0 ClientHello 267 B, or post-handshake 164 B preamble, or prefix n-grams).

Both outcomes are scientifically informative and publishable in Section 5.3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "3_models")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.config import (
    LATEX_TABLES_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    set_global_seed,
    setup_matplotlib_style
)
from architectures import WebTunnel1DCNN, WebTunnelTransformer
from utils import FlowSequenceDataset, BinaryFocalLoss, compute_metrics, get_device


def shuffle_flow_records(X: np.ndarray, seed: int = RANDOM_SEED) -> np.ndarray:
    """Randomly permute active (non-zero padded) records within each flow."""
    rng = np.random.RandomState(seed)
    X_shuffled = np.zeros_like(X)
    for i in range(len(X)):
        flow = X[i]
        active_mask = np.abs(flow).sum(axis=-1) > 1e-6
        active_len = int(np.sum(active_mask))
        if active_len > 1:
            perm = rng.permutation(active_len)
            X_shuffled[i, :active_len] = flow[:active_len][perm]
        else:
            X_shuffled[i] = flow
    return X_shuffled


def train_eval_cnn(X_tr, y_tr, X_val, y_val, X_te, y_te, device, epochs=25, seed=RANDOM_SEED):
    set_global_seed(seed)
    train_ds = FlowSequenceDataset(X_tr, y_tr, channel_first=True)
    val_ds = FlowSequenceDataset(X_val, y_val, channel_first=True)
    test_ds = FlowSequenceDataset(X_te, y_te, channel_first=True)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * len(by)
        val_loss /= len(val_ds)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    model.eval()
    probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            probs.append(model(bx.to(device)).cpu().numpy())
    probs = np.vstack(probs).flatten()
    return compute_metrics(y_te, probs, threshold=0.5)


def train_eval_transformer(X_tr, y_tr, X_val, y_val, X_te, y_te, device, epochs=25, seed=RANDOM_SEED):
    set_global_seed(seed)
    train_ds = FlowSequenceDataset(X_tr, y_tr, channel_first=False)
    val_ds = FlowSequenceDataset(X_val, y_val, channel_first=False)
    test_ds = FlowSequenceDataset(X_te, y_te, channel_first=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * len(by)
        val_loss /= len(val_ds)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    model.eval()
    probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            probs.append(model(bx.to(device)).cpu().numpy())
    probs = np.vstack(probs).flatten()
    return compute_metrics(y_te, probs, threshold=0.5)


def main():
    ap = argparse.ArgumentParser(description="Order-Shuffle Control Experiment for 1D-CNN and Transformer")
    ap.add_argument("--dataset", default="data/processed/sequence_dataset.npz")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    print("=" * 80)
    print("  Order-Shuffle Control Experiment (Testing Lattice vs. Positional Leakage)")
    print("=" * 80)

    d = np.load(args.dataset, allow_pickle=True)
    X_tr_orig = d["X_train"]
    y_tr = d["y_train"]
    X_val_orig = d["X_val"]
    y_val = d["y_val"]
    X_te_orig = d["X_test"]
    y_te = d["y_test"]

    print(f"Dataset loaded: Train={len(y_tr)}, Val={len(y_val)}, Test={len(y_te)} flows.")
    device = get_device()
    print(f"Training on device: {device}")

    print("\nPermuting temporal record order within each flow...")
    X_tr_shuf = shuffle_flow_records(X_tr_orig, seed=args.seed)
    X_val_shuf = shuffle_flow_records(X_val_orig, seed=args.seed + 1)
    X_te_shuf = shuffle_flow_records(X_te_orig, seed=args.seed + 2)

    results = []

    # 1. 1D-CNN Original Order
    print("\n--- Training 1D-CNN on Original Sequential Order ---")
    m_cnn_orig = train_eval_cnn(X_tr_orig, y_tr, X_val_orig, y_val, X_te_orig, y_te, device, args.epochs, args.seed)
    print(f"1D-CNN (Original): Acc={m_cnn_orig['accuracy']*100:.2f}%, Recall={m_cnn_orig['recall']*100:.2f}%, PR-AUC={m_cnn_orig['pr_auc']:.4f}")
    results.append({"model": "1D-CNN", "condition": "Original Sequence", **m_cnn_orig})

    # 2. 1D-CNN Shuffled Order
    print("\n--- Training 1D-CNN on Order-Shuffled Records ---")
    m_cnn_shuf = train_eval_cnn(X_tr_shuf, y_tr, X_val_shuf, y_val, X_te_shuf, y_te, device, args.epochs, args.seed)
    print(f"1D-CNN (Shuffled): Acc={m_cnn_shuf['accuracy']*100:.2f}%, Recall={m_cnn_shuf['recall']*100:.2f}%, PR-AUC={m_cnn_shuf['pr_auc']:.4f}")
    results.append({"model": "1D-CNN", "condition": "Order-Shuffled (Permuted)", **m_cnn_shuf})

    # 3. Flow-Transformer Original Order
    print("\n--- Training Flow-Transformer on Original Sequential Order ---")
    m_tf_orig = train_eval_transformer(X_tr_orig, y_tr, X_val_orig, y_val, X_te_orig, y_te, device, args.epochs, args.seed)
    print(f"Transformer (Original): Acc={m_tf_orig['accuracy']*100:.2f}%, Recall={m_tf_orig['recall']*100:.2f}%, PR-AUC={m_tf_orig['pr_auc']:.4f}")
    results.append({"model": "Flow-Transformer", "condition": "Original Sequence", **m_tf_orig})

    # 4. Flow-Transformer Shuffled Order
    print("\n--- Training Flow-Transformer on Order-Shuffled Records ---")
    m_tf_shuf = train_eval_transformer(X_tr_shuf, y_tr, X_val_shuf, y_val, X_te_shuf, y_te, device, args.epochs, args.seed)
    print(f"Transformer (Shuffled): Acc={m_tf_shuf['accuracy']*100:.2f}%, Recall={m_tf_shuf['recall']*100:.2f}%, PR-AUC={m_tf_shuf['pr_auc']:.4f}")
    results.append({"model": "Flow-Transformer", "condition": "Order-Shuffled (Permuted)", **m_tf_shuf})

    print("\n" + "=" * 90)
    print(f"{'Model':<18} | {'Condition':<28} | {'Accuracy':<10} | {'Recall':<10} | {'PR-AUC':<8} | {'ROC-AUC':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['model']:<18} | {r['condition']:<28} | {r['accuracy']*100:>8.2f}% | {r['recall']*100:>8.2f}% | {r['pr_auc']:>8.4f} | {r['roc_auc']:>8.4f}")
    print("=" * 90)

    json_path = os.path.join(PROJECT_ROOT, "4_evaluation", "order_shuffle_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"[OK] Saved {json_path}")

    tex_path = os.path.join(LATEX_TABLES_DIR, "table_order_shuffle.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Kontrolní experiment permutace pořadí paketů (Order-Shuffle Control): Vliv náhodného promíchání pořadí záznamů na sekvenční neuronové modely}" + "\n")
        f.write(r"\label{tab:order_shuffle}" + "\n")
        f.write(r"\begin{tabular}{llcccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Model} & \textbf{Režim toku} & \textbf{Přesnost (Acc)} & \textbf{Recall} & \textbf{PR-AUC} & \textbf{ROC-AUC} \\" + "\n")
        f.write(r"\hline" + "\n")
        for r in results:
            f.write(f"{r['model']} & {r['condition']} & {r['accuracy']*100:.2f}\\% & {r['recall']*100:.2f}\\% & {r['pr_auc']:.4f} & {r['roc_auc']:.4f} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"[OK] Exported {tex_path}")

    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = ["1D-CNN (Orig.)", "1D-CNN (Shuf.)", "Transf. (Orig.)", "Transf. (Shuf.)"]
    accs = [r["accuracy"] * 100 for r in results]
    praucs = [r["pr_auc"] * 100 for r in results]

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, accs, width=w, label="Accuracy (%)", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, praucs, width=w, label="PR-AUC (%)", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(80, 105)
    ax.set_ylabel("Skóre (%)")
    ax.set_title("Order-Shuffle Control: Vliv zničení časové posloupnosti na sekvenční sítě")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "order_shuffle_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[OK] Saved {plot_path}")


if __name__ == "__main__":
    main()
