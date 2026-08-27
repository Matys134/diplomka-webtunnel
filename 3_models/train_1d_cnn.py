#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

PROCESSED_DIR = "data/processed"
MODEL_DIR = "3_models/saved_models"
EVAL_DIR = "4_evaluation"

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = inputs.view(-1)
        t = targets.view(-1).float()
        
        eps = 1e-7
        p = torch.clamp(p, eps, 1.0 - eps)
        
        # Binary focal loss
        pt = torch.where(t == 1, p, 1.0 - p)
        alpha_t = torch.where(t == 1, self.alpha, 1.0 - self.alpha)
        
        loss = -alpha_t * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class WebTunnel1DCNN(nn.Module):
    """
    1D-CNN Architecture inspired by Deep Packet & Deep Fingerprinting (Sirinam et al.).
    Input shape: (Batch_Size, 2, 200) where channel 0 is normalized length, channel 1 is log IAT.
    """
    def __init__(self, in_channels=2, num_classes=1):
        super(WebTunnel1DCNN, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, 2, L)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training PyTorch 1D-CNN on device: {device} ===")
    
    dataset_path = os.path.join(PROCESSED_DIR, "sequence_dataset.npz")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}!")
        return
        
    data = np.load(dataset_path, allow_pickle=True)
    # PyTorch Conv1d expects (Batch, Channels, SeqLen) -> Transpose (B, 200, 2) -> (B, 2, 200)
    X_train = np.transpose(data["X_train"], (0, 2, 1))
    X_val = np.transpose(data["X_val"], (0, 2, 1))
    X_test = np.transpose(data["X_test"], (0, 2, 1))
    
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = WebTunnel1DCNN(in_channels=2, num_classes=1).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training Loop
    epochs = 40
    best_val_loss = float("inf")
    start_train_t = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx).squeeze(-1)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx).squeeze(-1)
                loss = criterion(preds, by)
                val_loss += loss.item() * len(by)
        val_loss /= max(len(val_dataset), 1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "1d_cnn_best.pt"))
            
    train_time = time.time() - start_train_t
    print(f"[OK] Training finished in {train_time:.2f}s. Best Val Loss: {best_val_loss:.4f}")
    
    # Load Best Model for Evaluation
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "1d_cnn_best.pt")))
    model.eval()
    
    # Benchmark Inference Latency on Device
    n_bench = 1000
    dummy_input = torch.randn(1, 2, 200, device=device)
    # Warmup
    for _ in range(50):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_inf = time.time()
    with torch.no_grad():
        for _ in range(n_bench):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    inf_total = time.time() - start_inf
    latency_ms = (inf_total / n_bench) * 1000.0
    throughput = n_bench / inf_total
    
    # Test Evaluation
    all_preds, all_probs = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            probs = model(bx).squeeze(-1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))
            
    test_probs = np.array(all_probs)
    test_preds = np.array(all_preds)
    
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.0
    pr_auc = average_precision_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.0
    cm = confusion_matrix(y_test, test_preds).tolist()
    
    print("\n--- 1D-CNN Test Set Evaluation ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Inference Latency ({device}):    {latency_ms:.4f} ms/flow")
    print(f"Inference Throughput ({device}): {throughput:.1f} flows/sec")
    
    results = {
        "model": "1D-CNN",
        "device": str(device),
        "train_time_sec": float(train_time),
        "inference_latency_ms": float(latency_ms),
        "throughput_flows_sec": float(throughput),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc)
        },
        "confusion_matrix": cm
    }
    
    with open(os.path.join(EVAL_DIR, "1d_cnn_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    np.savez_compressed(
        os.path.join(EVAL_DIR, "1d_cnn_test_preds.npz"),
        y_test=y_test,
        y_probs=test_probs
    )
    print(f"\n[OK] Model saved to {MODEL_DIR}/1d_cnn_best.pt")
    print(f"[OK] Evaluation results saved to {EVAL_DIR}/1d_cnn_results.json")

if __name__ == "__main__":
    main()
