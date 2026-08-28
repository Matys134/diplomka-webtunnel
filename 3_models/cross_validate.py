#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

sys.path.append("3_models")
sys.path.append("2_data_pipeline")
from train_1d_cnn import WebTunnel1DCNN, FocalLoss
from train_transformer import WebTunnelTransformer

PROCESSED_DIR = "data/processed"
EVAL_DIR = "4_evaluation"

import copy

def cv_xgboost(X_tab, y_bin, y_mul, groups, n_splits=5):
    print(f"\n=== Running {n_splits}-Fold Session-Stratified Group CV for XGBoost (Early Stopping) ===")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    metrics = {"acc": [], "prec": [], "rec": [], "f1": [], "roc_auc": [], "pr_auc": []}
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_tab, y_mul, groups=groups)):
        X_tr_full, y_tr_full, g_tr_full = X_tab[train_idx], y_bin[train_idx], groups[train_idx]
        X_te, y_te = X_tab[test_idx], y_bin[test_idx]
        
        # Inner group split for early stopping validation (80/20)
        inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + fold)
        inner_tr_idx, inner_val_idx = next(inner_sgkf.split(X_tr_full, y_tr_full, groups=g_tr_full))
        
        X_tr, y_tr = X_tr_full[inner_tr_idx], y_tr_full[inner_tr_idx]
        X_val, y_val = X_tr_full[inner_val_idx], y_tr_full[inner_val_idx]
        
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            early_stopping_rounds=15, random_state=42
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        metrics["acc"].append(accuracy_score(y_te, preds))
        metrics["prec"].append(precision_score(y_te, preds, zero_division=0))
        metrics["rec"].append(recall_score(y_te, preds, zero_division=0))
        metrics["f1"].append(f1_score(y_te, preds, zero_division=0))
        metrics["roc_auc"].append(roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5)
        metrics["pr_auc"].append(average_precision_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5)
        
    summary = {}
    for k, v in metrics.items():
        mean_val = float(np.mean(v))
        std_val = float(np.std(v))
        ci95_val = float(1.96 * std_val / np.sqrt(n_splits))
        summary[k] = {"mean": mean_val, "std": std_val, "ci95": ci95_val}
        print(f"  XGBoost {k.upper():<8}: {mean_val*100:.2f}% +/- {std_val*100:.2f}% (95% CI: [{mean_val*100 - ci95_val*100:.2f}%, {mean_val*100 + ci95_val*100:.2f}%])")
    return summary

def cv_1d_cnn(X_seq, y_bin, y_mul, groups, n_splits=5):
    print(f"\n=== Running {n_splits}-Fold Session-Stratified Group CV for 1D-CNN (Early Stopping) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Shape (N, 2, 200)
    X_seq_t = np.transpose(X_seq, (0, 2, 1))
    metrics = {"acc": [], "prec": [], "rec": [], "f1": [], "roc_auc": [], "pr_auc": []}
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_seq_t, y_mul, groups=groups)):
        X_tr_full, y_tr_full, g_tr_full = X_seq_t[train_idx], y_bin[train_idx], groups[train_idx]
        X_te, y_te = torch.from_numpy(X_seq_t[test_idx]), torch.from_numpy(y_bin[test_idx])
        
        # Inner group split for early stopping validation (80/20)
        inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + fold)
        inner_tr_idx, inner_val_idx = next(inner_sgkf.split(X_tr_full, y_tr_full, groups=g_tr_full))
        
        X_tr, y_tr = torch.from_numpy(X_tr_full[inner_tr_idx]), torch.from_numpy(y_tr_full[inner_tr_idx])
        X_val, y_val = torch.from_numpy(X_tr_full[inner_val_idx]), torch.from_numpy(y_tr_full[inner_val_idx])
        
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=16, shuffle=False)
        
        model = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
        
        best_val_loss = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        patience = 10
        patience_counter = 0
        
        for epoch in range(40):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                preds = model(bx).squeeze(-1)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    v_preds = model(bx).squeeze(-1)
                    val_loss += criterion(v_preds, by).item() * bx.size(0)
            val_loss /= len(X_val)
            
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                
        model.load_state_dict(best_weights)
        model.eval()
        all_probs = []
        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(device)
                probs = model(bx).squeeze(-1).cpu().numpy()
                all_probs.extend(probs)
                
        probs = np.array(all_probs)
        preds = (probs >= 0.5).astype(int)
        y_true = y_bin[test_idx]
        
        metrics["acc"].append(accuracy_score(y_true, preds))
        metrics["prec"].append(precision_score(y_true, preds, zero_division=0))
        metrics["rec"].append(recall_score(y_true, preds, zero_division=0))
        metrics["f1"].append(f1_score(y_true, preds, zero_division=0))
        metrics["roc_auc"].append(roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5)
        metrics["pr_auc"].append(average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5)
        
    summary = {}
    for k, v in metrics.items():
        mean_val = float(np.mean(v))
        std_val = float(np.std(v))
        ci95_val = float(1.96 * std_val / np.sqrt(n_splits))
        summary[k] = {"mean": mean_val, "std": std_val, "ci95": ci95_val}
        print(f"  1D-CNN  {k.upper():<8}: {mean_val*100:.2f}% +/- {std_val*100:.2f}% (95% CI: [{mean_val*100 - ci95_val*100:.2f}%, {mean_val*100 + ci95_val*100:.2f}%])")
    return summary

def cv_transformer(X_seq, y_bin, y_mul, groups, n_splits=5):
    print(f"\n=== Running {n_splits}-Fold Session-Stratified Group CV for Flow-Transformer (Early Stopping) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    metrics = {"acc": [], "prec": [], "rec": [], "f1": [], "roc_auc": [], "pr_auc": []}
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_seq, y_mul, groups=groups)):
        X_tr_full, y_tr_full, g_tr_full = X_seq[train_idx], y_bin[train_idx], groups[train_idx]
        X_te, y_te = torch.from_numpy(X_seq[test_idx]), torch.from_numpy(y_bin[test_idx])
        
        # Inner group split for early stopping validation (80/20)
        inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + fold)
        inner_tr_idx, inner_val_idx = next(inner_sgkf.split(X_tr_full, y_tr_full, groups=g_tr_full))
        
        X_tr, y_tr = torch.from_numpy(X_tr_full[inner_tr_idx]), torch.from_numpy(y_tr_full[inner_tr_idx])
        X_val, y_val = torch.from_numpy(X_tr_full[inner_val_idx]), torch.from_numpy(y_tr_full[inner_val_idx])
        
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=16, shuffle=False)
        
        model = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
        
        best_val_loss = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        patience = 10
        patience_counter = 0
        
        for epoch in range(40):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                preds = model(bx).squeeze(-1)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    v_preds = model(bx).squeeze(-1)
                    val_loss += criterion(v_preds, by).item() * bx.size(0)
            val_loss /= len(X_val)
            
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                
        model.load_state_dict(best_weights)
        model.eval()
        all_probs = []
        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(device)
                probs = model(bx).squeeze(-1).cpu().numpy()
                all_probs.extend(probs)
                
        probs = np.array(all_probs)
        preds = (probs >= 0.5).astype(int)
        y_true = y_bin[test_idx]
        
        metrics["acc"].append(accuracy_score(y_true, preds))
        metrics["prec"].append(precision_score(y_true, preds, zero_division=0))
        metrics["rec"].append(recall_score(y_true, preds, zero_division=0))
        metrics["f1"].append(f1_score(y_true, preds, zero_division=0))
        metrics["roc_auc"].append(roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5)
        metrics["pr_auc"].append(average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5)
        
    summary = {}
    for k, v in metrics.items():
        mean_val = float(np.mean(v))
        std_val = float(np.std(v))
        ci95_val = float(1.96 * std_val / np.sqrt(n_splits))
        summary[k] = {"mean": mean_val, "std": std_val, "ci95": ci95_val}
        print(f"  Transformer {k.upper():<8}: {mean_val*100:.2f}% +/- {std_val*100:.2f}% (95% CI: [{mean_val*100 - ci95_val*100:.2f}%, {mean_val*100 + ci95_val*100:.2f}%])")
    return summary

def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    tab_data = np.load(os.path.join(PROCESSED_DIR, "tabular_dataset.npz"), allow_pickle=True)
    seq_data = np.load(os.path.join(PROCESSED_DIR, "sequence_dataset.npz"), allow_pickle=True)
    
    X_tab = np.concatenate([tab_data["X_train"], tab_data["X_val"], tab_data["X_test"]], axis=0)
    X_seq = np.concatenate([seq_data["X_train"], seq_data["X_val"], seq_data["X_test"]], axis=0)
    y_bin = np.concatenate([tab_data["y_train"], tab_data["y_val"], tab_data["y_test"]], axis=0)
    y_mul = np.concatenate([tab_data["y_train_mul"], tab_data["y_val_mul"], tab_data["y_test_mul"]], axis=0)
    
    if "sample_ids_train" in tab_data:
        groups = np.concatenate([tab_data["sample_ids_train"], tab_data["sample_ids_val"], tab_data["sample_ids_test"]], axis=0)
    else:
        # Fallback: create group blocks of size 20
        groups = np.arange(len(y_bin)) // 20
        
    print(f"Total dataset size for Cross-Validation: {len(y_bin)} samples across {len(np.unique(groups))} unique session groups.")
    xgb_cv = cv_xgboost(X_tab, y_bin, y_mul, groups, n_splits=5)
    cnn_cv = cv_1d_cnn(X_seq, y_bin, y_mul, groups, n_splits=5)
    tf_cv = cv_transformer(X_seq, y_bin, y_mul, groups, n_splits=5)
    
    cv_results = {
        "xgboost_cv_5fold": xgb_cv,
        "1d_cnn_cv_5fold": cnn_cv,
        "transformer_cv_5fold": tf_cv
    }
    with open(os.path.join(EVAL_DIR, "cross_validation_results.json"), "w") as f:
        json.dump(cv_results, f, indent=4)
    print(f"\n[OK] 5-Fold Cross-Validation results saved to {EVAL_DIR}/cross_validation_results.json")

if __name__ == "__main__":
    main()
