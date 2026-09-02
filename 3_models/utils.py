"""
Shared utilities for machine learning and deep learning pipelines:
- Dataset loading and formatting
- PyTorch custom Dataset and Focal Loss
- Metric computation (Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC)
- Device selection and reproducibility
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from typing import Tuple, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import TABULAR_DATASET_PATH, SEQUENCE_DATASET_PATH, set_global_seed


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowSequenceDataset(Dataset):
    """PyTorch Dataset for raw packet sequences (batch, seq_len, 2)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, channel_first: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        if channel_first:
            # Transpose [N, L, C] -> [N, C, L] for 1D-CNN
            self.X = self.X.permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.clamp(inputs, min=1e-7, max=1.0 - 1e-7)
        bce_loss = - (targets * torch.log(inputs) + (1.0 - targets) * torch.log(1.0 - inputs))
        p_t = targets * inputs + (1.0 - targets) * (1.0 - inputs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def load_tabular_data(dataset_path: str = TABULAR_DATASET_PATH) -> Dict[str, Any]:
    """Loads tabular dataset returning train/val/test splits and metadata."""
    data = np.load(dataset_path)
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "y_train_mul": data["y_train_mul"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "y_val_mul": data["y_val_mul"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "y_test_mul": data["y_test_mul"],
        "sample_ids_train": data["sample_ids_train"] if "sample_ids_train" in data else None,
        "sample_ids_val": data["sample_ids_val"] if "sample_ids_val" in data else None,
        "sample_ids_test": data["sample_ids_test"] if "sample_ids_test" in data else None,
        "sample_ids_all": data["sample_ids_all"] if "sample_ids_all" in data else None,
        "feature_names": list(data["feature_names"]),
    }


def load_sequence_data(dataset_path: str = SEQUENCE_DATASET_PATH) -> Dict[str, Any]:
    """Loads sequence tensor dataset returning train/val/test splits."""
    data = np.load(dataset_path)
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
    }


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes comprehensive classification metrics."""
    y_preds = (y_probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_preds))
    prec = float(precision_score(y_true, y_preds, zero_division=0))
    rec = float(recall_score(y_true, y_preds, zero_division=0))
    f1 = float(f1_score(y_true, y_preds, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except Exception:
        pr_auc = 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
    }

