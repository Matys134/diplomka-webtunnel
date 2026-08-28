#!/usr/bin/env python3
import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from train_1d_cnn import FocalLoss

PROCESSED_DIR = "data/processed"
MODEL_DIR = "3_models/saved_models"
EVAL_DIR = "4_evaluation"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=250):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class WebTunnelTransformer(nn.Module):
    def __init__(self, in_features=2, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, SeqLen, InFeatures) -> (B, 200, 2)
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)
        # Mean pooling across sequence dimension
        pooled = h.mean(dim=1)
        out = self.classifier(pooled)
        return out

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training PyTorch Flow-Transformer on device: {device} ===")
    
    dataset_path = os.path.join(PROCESSED_DIR, "sequence_dataset.npz")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}!")
        return
        
    data = np.load(dataset_path, allow_pickle=True)
    # Shape: (Batch, SeqLen, InFeatures) = (B, 200, 2)
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = WebTunnelTransformer(in_features=2, d_model=64, nhead=4, num_layers=2).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 40
    best_val_loss = float("inf")
    patience = 12
    patience_counter = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
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
        scheduler.step()
        
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
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "transformer_best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
            
    train_time = time.time() - start_train_t
    print(f"[OK] Training finished in {train_time:.2f}s. Best Val Loss: {best_val_loss:.4f}")
    
    # Evaluate
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "transformer_best.pt")))
    model.eval()
    
    # Inference Latency Benchmark
    n_bench = 1000
    dummy_input = torch.randn(1, 200, 2, device=device)
    for _ in range(50): _ = model(dummy_input)
    if device.type == "cuda": torch.cuda.synchronize()
    
    start_inf = time.time()
    with torch.no_grad():
        for _ in range(n_bench): _ = model(dummy_input)
    if device.type == "cuda": torch.cuda.synchronize()
    
    inf_total = time.time() - start_inf
    latency_ms = (inf_total / n_bench) * 1000.0
    throughput = n_bench / inf_total
    
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
    
    print("\n--- Transformer Test Set Evaluation ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Inference Latency ({device}):    {latency_ms:.4f} ms/flow")
    print(f"Inference Throughput ({device}): {throughput:.1f} flows/sec")
    
    results = {
        "model": "Flow-Transformer",
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
    
    with open(os.path.join(EVAL_DIR, "transformer_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    np.savez_compressed(
        os.path.join(EVAL_DIR, "transformer_test_preds.npz"),
        y_test=y_test,
        y_probs=test_probs
    )
    print(f"\n[OK] Model saved to {MODEL_DIR}/transformer_best.pt")
    print(f"[OK] Evaluation results saved to {EVAL_DIR}/transformer_results.json")

if __name__ == "__main__":
    main()
