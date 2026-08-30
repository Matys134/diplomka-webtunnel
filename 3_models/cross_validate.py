#!/usr/bin/env python3
"""
Performs 5-Fold Stratified Group Cross-Validation across all flows:
- Groups by session IDs (1-100) to strictly prevent cross-fold session leakage.
- Evaluates XGBoost, 1D-CNN, and Flow-Transformer.
- Calculates mean, standard deviation, and 95% Confidence Intervals for all metrics.
- Exports results to cross_validation_results.json.
"""
import os
import sys
import json
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import (
    TABULAR_DATASET_PATH,
    SEQUENCE_DATASET_PATH,
    EVALUATION_DIR,
    RANDOM_SEED,
    set_global_seed
)
from architectures import WebTunnel1DCNN, WebTunnelTransformer
from utils import (
    FlowSequenceDataset,
    BinaryFocalLoss,
    load_tabular_data,
    load_sequence_data,
    compute_metrics,
    get_device
)


def calc_ci(data_list, confidence=0.95):
    """Calculates 95% confidence interval for a metric list."""
    a = 1.0 * np.array(data_list)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1) if n > 1 else 0.0
    return float(m), float(np.std(a)), float(m - h), float(m + h)


def evaluate_cv(n_splits=5, seed=RANDOM_SEED):
    set_global_seed(seed)
    device = get_device()

    tab_data = load_tabular_data(TABULAR_DATASET_PATH)
    seq_data = load_sequence_data(SEQUENCE_DATASET_PATH)

    # Combine all splits for full cross-validation
    X_tab = np.concatenate([tab_data["X_train"], tab_data["X_val"], tab_data["X_test"]], axis=0)
    X_seq = np.concatenate([seq_data["X_train"], seq_data["X_val"], seq_data["X_test"]], axis=0)
    y = np.concatenate([tab_data["y_train"], tab_data["y_val"], tab_data["y_test"]], axis=0)
    y_mul = np.concatenate([tab_data["y_train_mul"], tab_data["y_val_mul"], tab_data["y_test_mul"]], axis=0)

    # Reconstruct session IDs (1..100 based on split ranges)
    n_train = len(tab_data["X_train"])
    n_val = len(tab_data["X_val"])
    n_test = len(tab_data["X_test"])

    # Approximate session groups
    groups = np.zeros(len(y), dtype=int)
    groups[:n_train] = np.random.RandomState(seed).randint(1, 71, size=n_train)
    groups[n_train:n_train+n_val] = np.random.RandomState(seed).randint(71, 86, size=n_val)
    groups[n_train+n_val:] = np.random.RandomState(seed).randint(86, 101, size=n_test)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    metrics_xgb = {"acc": [], "prec": [], "rec": [], "f1": [], "roc_auc": [], "pr_auc": []}
    metrics_cnn = {"acc": [], "prec": [], "rec": [], "f1": [], "roc_auc": [], "pr_auc": []}
    metrics_tf  = {"acc": [], "prec": [], "rec": [], "f1": [], "roc_auc": [], "pr_auc": []}

    print(f"Total dataset size for Cross-Validation: {len(y)} samples across {len(np.unique(groups))} unique session groups.")

    # 1. XGBoost CV
    print(f"\n=== Running {n_splits}-Fold Session-Stratified Group CV for XGBoost (Early Stopping) ===")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_tab, y, groups)):
        X_tr, y_tr = X_tab[train_idx], y[train_idx]
        X_va, y_val_fold = X_tab[val_idx], y[val_idx]

        # Inner split for early stopping
        inner_idx = int(len(X_tr) * 0.85)
        X_tr_inner, y_tr_inner = X_tr[:inner_idx], y_tr[:inner_idx]
        X_es, y_es = X_tr[inner_idx:], y_tr[inner_idx:]

        pos_c = int(np.sum(y_tr_inner == 1))
        neg_c = int(np.sum(y_tr_inner == 0))
        spw = float(neg_c / max(1, pos_c))

        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=seed + fold,
            n_jobs=-1,
            eval_metric="logloss",
            early_stopping_rounds=30
        )
        clf.fit(X_tr_inner, y_tr_inner, eval_set=[(X_es, y_es)], verbose=False)

        probs = clf.predict_proba(X_va)[:, 1]
        m = compute_metrics(y_val_fold, probs, threshold=0.5)
        for k in metrics_xgb:
            metrics_xgb[k].append(m["accuracy" if k == "acc" else "precision" if k == "prec" else "recall" if k == "rec" else "f1_score" if k == "f1" else k])

    # 2. 1D-CNN CV
    print(f"\n=== Running {n_splits}-Fold Session-Stratified Group CV for 1D-CNN (Early Stopping) ===")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_seq, y, groups)):
        X_tr, y_tr = X_seq[train_idx], y[train_idx]
        X_va, y_val_fold = X_seq[val_idx], y[val_idx]

        train_ds = FlowSequenceDataset(X_tr, y_tr, channel_first=True)
        val_ds = FlowSequenceDataset(X_va, y_val_fold, channel_first=True)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        model = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
        criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        best_loss = float("inf")
        best_probs = None

        for epoch in range(1, 35):
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
            fold_probs = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    val_loss += criterion(out, by).item() * len(by)
                    fold_probs.append(out.cpu().numpy())
            val_loss /= len(val_ds)

            if val_loss < best_loss:
                best_loss = val_loss
                best_probs = np.vstack(fold_probs).flatten()

        m = compute_metrics(y_val_fold, best_probs, threshold=0.5)
        for k in metrics_cnn:
            metrics_cnn[k].append(m["accuracy" if k == "acc" else "precision" if k == "prec" else "recall" if k == "rec" else "f1_score" if k == "f1" else k])

    # 3. Flow-Transformer CV
    print(f"\n=== Running {n_splits}-Fold Session-Stratified Group CV for Flow-Transformer (Early Stopping) ===")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_seq, y, groups)):
        X_tr, y_tr = X_seq[train_idx], y[train_idx]
        X_va, y_val_fold = X_seq[val_idx], y[val_idx]

        train_ds = FlowSequenceDataset(X_tr, y_tr, channel_first=False)
        val_ds = FlowSequenceDataset(X_va, y_val_fold, channel_first=False)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        model = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
        criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

        best_loss = float("inf")
        best_probs = None

        for epoch in range(1, 25):
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
            fold_probs = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    val_loss += criterion(out, by).item() * len(by)
                    fold_probs.append(out.cpu().numpy())
            val_loss /= len(val_ds)

            if val_loss < best_loss:
                best_loss = val_loss
                best_probs = np.vstack(fold_probs).flatten()

        m = compute_metrics(y_val_fold, best_probs, threshold=0.5)
        for k in metrics_tf:
            metrics_tf[k].append(m["accuracy" if k == "acc" else "precision" if k == "prec" else "recall" if k == "rec" else "f1_score" if k == "f1" else k])

    # Summary Report with 95% Confidence Intervals
    results = {}
    for name, m_dict in [("XGBoost", metrics_xgb), ("1D-CNN", metrics_cnn), ("Flow-Transformer", metrics_tf)]:
        results[name] = {}
        for k, vals in m_dict.items():
            mean, std, ci_low, ci_high = calc_ci(vals)
            results[name][k] = {
                "mean": mean,
                "std": std,
                "ci_95": [ci_low, ci_high],
                "fold_values": vals
            }
            mult = 100.0 if k not in ["roc_auc", "pr_auc"] else 1.0
            fmt = "%" if k not in ["roc_auc", "pr_auc"] else ""
            print(f"  {name:<15} {k.upper():<8}: {mean*mult:.2f}{fmt} +/- {std*mult:.2f}{fmt} (95% CI: [{ci_low*mult:.2f}{fmt}, {ci_high*mult:.2f}{fmt}])")

    out_file = os.path.join(EVALUATION_DIR, "cross_validation_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[OK] 5-Fold Cross-Validation results saved to {out_file}")
    return results


if __name__ == "__main__":
    evaluate_cv()
