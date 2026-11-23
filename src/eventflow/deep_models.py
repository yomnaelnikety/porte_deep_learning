

import torch
from torch import nn


class EventSequenceModel(nn.Module):
    def __init__(self, n_classes, context_sizes, n_numeric, cell="gru", hidden=128,
                 layers=1, dropout=0.2, event_dim=32, context_dim=8, use_context=True):
        super().__init__()
        if cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")
        self.use_context = use_context
        # Index 0 is padding, so the event vocabulary has n_classes + 1 entries.
        self.event_embedding = nn.Embedding(n_classes + 1, event_dim, padding_idx=0)
        self.context_embeddings = nn.ModuleList(
            nn.Embedding(size + 1, context_dim, padding_idx=0) for size in context_sizes
        ) if use_context else nn.ModuleList()
        width = event_dim + len(self.context_embeddings) * context_dim + n_numeric
        self.input_projection = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout))
        recurrent = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = recurrent(hidden, hidden, num_layers=layers, batch_first=True,
                             dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, event, context, numeric):
        """event [B, T] long, context [B, T, C] long, numeric [B, T, F] float -> logits [B, T, K]."""
        parts = [self.event_embedding(event)]
        parts += [emb(context[..., i]) for i, emb in enumerate(self.context_embeddings)]
        parts.append(numeric)
        x = self.input_projection(torch.cat(parts, dim=-1))
        states, _ = self.rnn(x)          # works for GRU (h) and LSTM ((h, c))
        return self.head(self.norm(states))
