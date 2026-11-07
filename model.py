"""
Arquitectura ResNet-50 + Bi-LSTM + Atención Temporal
Proyecto MIA-203 — Detección de violencia urbana (UNI FIIS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TemporalAttention(nn.Module):
    """Mecanismo de Atención Temporal sobre los estados ocultos (h_1, ..., h_T)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, rnn_outputs):
        # rnn_outputs: (B, T, hidden_dim)
        scores = self.attn(rnn_outputs).squeeze(-1)          # (B, T)
        weights = F.softmax(scores, dim=1)                   # alpha_t
        context = torch.bmm(weights.unsqueeze(1), rnn_outputs).squeeze(1)
        return context, weights


class SpatioTemporalViolenceModel(nn.Module):
    """
    Arquitectura modular:
      - mode='baseline_avg'       -> ResNet-50 + Temporal Average Pooling
      - mode='resnet_rnn'         -> ResNet-50 + Vanilla RNN
      - mode='resnet_lstm'        -> ResNet-50 + LSTM Unidireccional
      - mode='resnet_bilstm_attn' -> ResNet-50 + Bi-LSTM + Atención Temporal (propuesto)
    """

    def __init__(
        self,
        mode="resnet_bilstm_attn",
        embed_dim=512,
        hidden_dim=256,
        dropout=0.3,
        freeze_backbone=True,
    ):
        super().__init__()
        self.mode = mode

        # 1. Extractor espacial: ResNet-50
        resnet = models.resnet50(weights=None)  # pesos se cargan del checkpoint
        self.backbone_features = nn.Sequential(*list(resnet.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_backbone:
            for param in self.backbone_features.parameters():
                param.requires_grad = False
            # Descongelar solo layer4
            for param in self.backbone_features[-1].parameters():
                param.requires_grad = True

        self.spatial_proj = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 2. Módulo temporal
        if mode == "baseline_avg":
            clf_in_dim = embed_dim
        elif mode == "resnet_rnn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=1, batch_first=True)
            clf_in_dim = hidden_dim
        elif mode == "resnet_lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
            clf_in_dim = hidden_dim
        elif mode == "resnet_bilstm_attn":
            self.rnn = nn.LSTM(
                embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True
            )
            self.attention = TemporalAttention(hidden_dim * 2)
            clf_in_dim = hidden_dim * 2
        else:
            raise ValueError(f"Modo no soportado: {mode}")

        # 3. Clasificador binario
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(clf_in_dim, 1),
        )

    def forward(self, x, return_attention=False):
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        feat_maps = self.backbone_features(x_flat)               # (B*T, 2048, 7, 7)
        pooled = self.global_pool(feat_maps).flatten(1)          # (B*T, 2048)
        embedded = self.spatial_proj(pooled).view(B, T, -1)      # (B, T, embed_dim)

        attn_weights = None
        if self.mode == "baseline_avg":
            context = embedded.mean(dim=1)
            attn_weights = torch.full((B, T), 1.0 / T, device=x.device)
        elif self.mode in ["resnet_rnn", "resnet_lstm"]:
            rnn_out, _ = self.rnn(embedded)
            context = rnn_out[:, -1, :]
            attn_weights = torch.zeros((B, T), device=x.device)
            attn_weights[:, -1] = 1.0
        elif self.mode == "resnet_bilstm_attn":
            rnn_out, _ = self.rnn(embedded)
            context, attn_weights = self.attention(rnn_out)

        logits = self.classifier(context).squeeze(-1)
        if return_attention:
            return logits, attn_weights
        return logits
