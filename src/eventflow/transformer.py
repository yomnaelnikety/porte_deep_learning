import math

import torch
from torch import nn


class EventTransformer(nn.Module):
    def __init__(self, n_classes, context_sizes, n_numeric, hidden=128,
                 layers=2, dropout=0.2, event_dim=32, context_dim=8,
                 use_context=True, heads=4, ff_multiplier=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.event_embedding = nn.Embedding(n_classes + 1, event_dim, padding_idx=0)
        self.context_embeddings = nn.ModuleList(
            nn.Embedding(size + 1, context_dim, padding_idx=0) for size in context_sizes
        ) if use_context else nn.ModuleList()
        width = event_dim + context_dim * len(self.context_embeddings) + n_numeric
        self.input_projection = nn.Linear(width, hidden)
        self.dropout = nn.Dropout(dropout)
        #layers constructed separately with random indepedent weights
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * ff_multiplier,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        ) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, event, context, numeric):
        parts = [self.event_embedding(event)]
        parts += [emb(context[..., i]) for i, emb in enumerate(self.context_embeddings)]
        parts.append(numeric)
        x = self.input_projection(torch.cat(parts, dim=-1))
        length, width = x.shape[1:]
        #sinusoids to encode the order
        positions = torch.arange(length, device=x.device, dtype=x.dtype)[:, None]
        frequencies = torch.exp(torch.arange(0, width, 2, device=x.device,
                                              dtype=x.dtype) * (-math.log(10000) / width))
        pe = torch.zeros((length, width), device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(positions * frequencies)
        pe[:, 1::2] = torch.cos(positions * frequencies[:width // 2])
        x = x + pe
        x = self.dropout(x)
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        padding = event.eq(0)
        for block in self.blocks:
            x = block(x, src_mask=causal, src_key_padding_mask=padding)
        return self.head(self.norm(x))
