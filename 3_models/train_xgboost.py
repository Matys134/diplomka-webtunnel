#!/usr/bin/env python3
"""
Trains and evaluates the Baseline XGBoost Classifier on 48 statistical flow features.
Includes:
- Dynamic scale_pos_weight for class imbalance
- Validation-set threshold calibration
- Full test set evaluation & microsecond inference latency profiling
- Serialization to JSON & Joblib
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import (
    TABULAR_DATASET_PATH,
    XGBOOST_MODEL_JSON,
    XGBOOST_MODEL_JOBLIB,
    EVALUATION_DIR,
    RANDOM_SEED,
    set_global_seed
)
from utils import load_tabular_data, compute_metrics


def train_xgboost(dataset_path: str = TABULAR_DATASET_PATH, seed: int = RANDOM_SEED):
    set_global_seed(seed)
    data = load_tabular_data(dataset_path)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos_weight = float(neg_count / max(1, pos_count))

    print("=== Training XGBoost Baseline Model ===")
    print(f"Train size: {X_train.shape} (Pos: {pos_count}, Neg: {neg_count}, ScalePosWeight: {scale_pos_weight:.2f})")
    print(f"Val size: {X_val.shape}, Test size: {X_test.shape}")

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=30
    )

    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - t0
    best_iter = clf.best_iteration if hasattr(clf, "best_iteration") else 300
    print(f"[OK] Trained in {train_time:.3f}s (Best iteration: {best_iter})")

    # Optimal threshold calibration on Validation set (maximizing F1)
    val_probs = clf.predict_proba(X_val)[:, 1]
    p_arr, r_arr, th_arr = precision_recall_curve(y_val, val_probs)
    f1_arr = 2 * (p_arr * r_arr) / (p_arr + r_arr + 1e-8)
    best_th_idx = int(np.argmax(f1_arr))
    optimal_th = float(th_arr[best_th_idx]) if best_th_idx < len(th_arr) else 0.5
    print(f"Calibrated Optimal Threshold (from Val set): {optimal_th:.4f}")

    # Latency & Throughput Benchmark on Test Set
    num_runs = 50
    t_start = time.perf_counter()
    for _ in range(num_runs):
        _ = clf.predict_proba(X_test)
    t_end = time.perf_counter()

    total_test_samples = len(X_test) * num_runs
    avg_latency_ms = ((t_end - t_start) / total_test_samples) * 1000.0
    throughput = total_test_samples / (t_end - t_start)

    test_probs = clf.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_probs, threshold=optimal_th)
    metrics["latency_ms_per_flow"] = avg_latency_ms
    metrics["throughput_flows_per_sec"] = throughput
    metrics["training_time_sec"] = train_time
    metrics["best_iteration"] = int(best_iter)

    print("\n--- XGBoost Test Evaluation ---")
    print(f"Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"Precision : {metrics['precision']*100:.2f}%")
    print(f"Recall    : {metrics['recall']*100:.2f}%")
    print(f"F1-Score  : {metrics['f1_score']*100:.2f}%")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC    : {metrics['pr_auc']:.4f}")
    print(f"Latency   : {avg_latency_ms:.5f} ms/flow ({throughput:,.0f} flows/sec)")

    # Save Model & Results
    clf.save_model(XGBOOST_MODEL_JSON)
    joblib.dump(clf, XGBOOST_MODEL_JOBLIB)
    print(f"[OK] Model saved to {XGBOOST_MODEL_JSON} and {XGBOOST_MODEL_JOBLIB}")

    res_path = os.path.join(EVALUATION_DIR, "xgboost_results.json")
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Evaluation results saved to {res_path}")

    # Save test predictions for cascading evaluation
    np.savez_compressed(
        os.path.join(EVALUATION_DIR, "xgboost_test_preds.npz"),
        probs=test_probs,
        y_test=y_test,
        y_test_mul=data["y_test_mul"]
    )
    return clf, metrics


if __name__ == "__main__":
    train_xgboost()
