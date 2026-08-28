#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)

PROCESSED_DIR = "data/processed"
MODEL_DIR = "3_models/saved_models"
EVAL_DIR = "4_evaluation"

def calculate_precision_at_fixed_fpr(y_true, y_probs, target_fprs=[0.01, 0.001, 0.0001]):
    """Calculates Precision achieved when decision threshold is calibrated for fixed low FPR."""
    results = {}
    neg_probs = y_probs[y_true == 0]
    pos_probs = y_probs[y_true == 1]
    
    if len(neg_probs) == 0 or len(pos_probs) == 0:
        return {f"precision_at_fpr_{fpr}": 0.0 for fpr in target_fprs}
        
    for target_fpr in target_fprs:
        threshold = np.percentile(neg_probs, 100.0 * (1.0 - target_fpr))
        tp = np.sum(pos_probs >= threshold)
        fp = np.sum(neg_probs >= threshold)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / len(pos_probs) if len(pos_probs) > 0 else 0.0
        results[f"fpr_{target_fpr}"] = {
            "threshold": float(threshold),
            "precision": float(prec),
            "recall": float(rec),
            "tp": int(tp),
            "fp": int(fp)
        }
    return results

def find_optimal_threshold(y_val, val_probs):
    """Finds optimal decision threshold on validation set maximizing F1-score."""
    prec, rec, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    dataset_path = os.path.join(PROCESSED_DIR, "tabular_dataset.npz")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}! Run data pipeline first.")
        return
        
    data = np.load(dataset_path, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = list(data["feature_names"])
    
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos_weight = float(neg_count / max(pos_count, 1))
    
    print(f"=== Training XGBoost Baseline Model ===")
    print(f"Train size: {X_train.shape} (Pos: {pos_count}, Neg: {neg_count}, ScalePosWeight: {scale_pos_weight:.2f})")
    print(f"Val size: {X_val.shape}, Test size: {X_test.shape}")
    
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42
    )
    
    start_train_t = time.time()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    train_time = time.time() - start_train_t
    print(f"[OK] Trained in {train_time:.3f}s (Best iteration: {clf.best_iteration})")
    
    val_probs = clf.predict_proba(X_val)[:, 1]
    opt_threshold = find_optimal_threshold(y_val, val_probs)
    print(f"Calibrated Optimal Threshold (from Val set): {opt_threshold:.4f}")
    
    # Benchmark inference latency
    n_benchmark = 1000
    X_bench = np.tile(X_test, (int(n_benchmark / max(len(X_test), 1)) + 1, 1))[:n_benchmark]
    
    start_inf_t = time.time()
    _ = clf.predict_proba(X_bench)
    inf_time_total = time.time() - start_inf_t
    latency_ms_per_flow = (inf_time_total / n_benchmark) * 1000.0
    throughput_flows_sec = n_benchmark / inf_time_total
    
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= opt_threshold).astype(int)
    
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.0
    pr_auc = average_precision_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.0
    
    print("\n--- XGBoost Test Evaluation ---")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")
    print(f"F1-Score  : {f1*100:.2f}%")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"Latency   : {latency_ms_per_flow:.5f} ms/flow ({throughput_flows_sec:,.0f} flows/sec)")
    
    low_fpr_results = calculate_precision_at_fixed_fpr(y_test, test_probs)
    
    import joblib
    model_save_path = os.path.join(MODEL_DIR, "xgboost_baseline.json")
    clf.save_model(model_save_path)
    joblib_save_path = os.path.join(MODEL_DIR, "xgboost_baseline.joblib")
    joblib.dump(clf, joblib_save_path)
    print(f"[OK] Model saved to {model_save_path} and {joblib_save_path}")
    
    results = {
        "model_name": "XGBoost",
        "training_time_sec": train_time,
        "inference_latency_ms": latency_ms_per_flow,
        "throughput_flows_sec": throughput_flows_sec,
        "optimal_threshold": float(opt_threshold),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc)
        },
        "low_fpr_calibration": low_fpr_results
    }
    
    with open(os.path.join(EVAL_DIR, "xgboost_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    np.savez_compressed(
        os.path.join(EVAL_DIR, "xgboost_test_preds.npz"),
        y_test=y_test,
        test_probs=test_probs,
        test_preds=test_preds
    )
    print(f"[OK] Evaluation results saved to {EVAL_DIR}/xgboost_results.json")

if __name__ == "__main__":
    main()
