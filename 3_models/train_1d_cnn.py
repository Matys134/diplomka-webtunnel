#!/usr/bin/env python3
"""
Trains and profiles the 1D-CNN (Deep Packet) classifier on raw packet sequences.
Includes:
- Binary Focal Loss for class imbalance
- Early stopping based on validation loss
- Latency and throughput benchmarking on CUDA/CPU
- Checkpointing best model weights
"""
import os
import sys
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import (
    SEQUENCE_DATASET_PATH,
    CNN_MODEL_PATH,
    EVALUATION_DIR,
    RANDOM_SEED,
    set_global_seed
)
from architectures import WebTunnel1DCNN
from utils import (
    FlowSequenceDataset,
    BinaryFocalLoss,
    load_sequence_data,
    compute_metrics,
    get_device
)


def train_1d_cnn(
    dataset_path: str = SEQUENCE_DATASET_PATH,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    patience: int = 12,
    seed: int = RANDOM_SEED
):
    set_global_seed(seed)
    device = get_device()
    data = load_sequence_data(dataset_path)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    train_ds = FlowSequenceDataset(X_train, y_train, channel_first=True)
    val_ds = FlowSequenceDataset(X_val, y_val, channel_first=True)
    test_ds = FlowSequenceDataset(X_test, y_test, channel_first=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"=== Training PyTorch 1D-CNN on device: {device} ===")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * len(batch_y)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CNN_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    train_time = time.time() - t0
    print(f"[OK] Training finished in {train_time:.2f}s. Best Val Loss: {best_val_loss:.4f}")

    # Load best checkpoint & Evaluate
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model.eval()

    test_probs = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            test_probs.append(preds.cpu().numpy())
    test_probs = np.vstack(test_probs).flatten()

    metrics = compute_metrics(y_test, test_probs, threshold=0.5)

    # Latency Benchmark
    test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    num_runs = 50
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(test_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_test_samples = len(X_test) * num_runs
    avg_latency_ms = ((t_end - t_start) / total_test_samples) * 1000.0
    throughput = total_test_samples / (t_end - t_start)

    metrics["latency_ms_per_flow"] = avg_latency_ms
    metrics["throughput_flows_per_sec"] = throughput
    metrics["training_time_sec"] = train_time
    metrics["best_val_loss"] = float(best_val_loss)

    print("\n--- 1D-CNN Test Set Evaluation ---")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Inference Latency ({device.type}):    {avg_latency_ms:.4f} ms/flow")
    print(f"Inference Throughput ({device.type}): {throughput:.1f} flows/sec")

    print(f"\n[OK] Model saved to {CNN_MODEL_PATH}")
    res_path = os.path.join(EVALUATION_DIR, "1d_cnn_results.json")
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Evaluation results saved to {res_path}")

    # Save test predictions for cascading evaluation
    np.savez_compressed(
        os.path.join(EVALUATION_DIR, "1d_cnn_test_preds.npz"),
        probs=test_probs,
        y_test=y_test
    )
    return model, metrics


if __name__ == "__main__":
    train_1d_cnn()
