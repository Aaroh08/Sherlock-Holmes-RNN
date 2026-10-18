from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

MODEL_NAMES = {"rnn": "Simple RNN", "birnn": "Bidirectional RNN", "lstm": "LSTM", "gru": "GRU"}


@dataclass
class ModelConfig:
    kind: str
    vocab_size: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    layers: int = 1
    dropout: float = .2
    context_length: int = 20

    def __post_init__(self):
        if self.kind not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {self.kind}")
        if min(self.vocab_size, self.embedding_dim, self.hidden_dim, self.layers, self.context_length) < 1:
            raise ValueError("Model dimensions must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must lie in [0, 1).")

    def to_dict(self):
        return asdict(self)


class NextWordModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        cell = {"rnn": nn.RNN, "birnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[config.kind]
        self.recurrent = cell(config.embedding_dim, config.hidden_dim, config.layers,
                              batch_first=True, bidirectional=config.kind == "birnn",
                              dropout=config.dropout if config.layers > 1 else 0)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_dim * (2 if config.kind == "birnn" else 1), config.vocab_size)

    def forward(self, ids: torch.Tensor):
        _, hidden = self.recurrent(self.embedding(ids))
        if isinstance(hidden, tuple):  # LSTM exposes (hidden state, cell state).
            hidden = hidden[0]
        # Use the complete final states, not output[:, -1] for a bidirectional RNN.
        context = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.config.kind == "birnn" else hidden[-1]
        return self.output(self.dropout(context))

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
