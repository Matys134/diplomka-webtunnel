"""
Deep Learning model architectures for WebTunnel Traffic Analysis and Fingerprinting.
Provides:
1. WebTunnel1DCNN - Multi-layer 1D Convolutional Neural Network with Batch Normalization.
2. WebTunnelTransformer - Flow Sequence Transformer with Learnable [CLS] Token & Multi-Head Self-Attention.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence transformer."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class WebTunnel1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for raw packet direction/length sequence classification.
    Input shape: [batch_size, in_channels=2, seq_len=200]
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 200 -> 100

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 100 -> 50

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 50 -> 1

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        out = self.sigmoid(self.fc2(x))
        return out


class WebTunnelTransformer(nn.Module):
    """
    Sequence Transformer with Learnable [CLS] Token and Multi-Head Attention.
    Input shape: [batch_size, seq_len=200, in_features=2]
    """
    def __init__(
        self,
        in_features: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        max_len: int = 250
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(in_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize [CLS] token
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        B, L, _ = x.shape
        proj = self.input_proj(x) * math.sqrt(self.d_model)  # [B, L, d_model] -- Vaswani et al. §3.4 scaling

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x_with_cls = torch.cat((cls_tokens, proj), dim=1)  # [B, L+1, d_model]

        # Construct key padding mask (True for zero-padded time steps)
        is_pad = (x.abs().sum(dim=-1) < 1e-6)  # [B, L]
        cls_pad = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_pad, is_pad], dim=1)  # [B, L+1]

        encoded = self.pos_encoder(x_with_cls)
        h = self.transformer_encoder(encoded, src_key_padding_mask=key_padding_mask)  # [B, L+1, d_model]

        # Extract [CLS] representation
        cls_repr = h[:, 0, :]  # [B, d_model]
        out = self.sigmoid(self.fc(self.dropout(cls_repr)))  # [B, 1]
        return out

