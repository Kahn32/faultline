from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _init_recurrent(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        if "weight_hh" in name:
            nn.init.orthogonal_(parameter)
        elif "weight_ih" in name:
            nn.init.xavier_uniform_(parameter)
        elif "bias" in name:
            nn.init.zeros_(parameter)
            # LSTM gates are [input, forget, cell, output]. A positive forget
            # bias is the same stabilizing initialization used in the notebook.
            if isinstance(module, nn.LSTM):
                hidden = parameter.numel() // 4
                parameter.data[hidden : hidden * 2].fill_(1.0)


class GRUDetector(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 128, n_layers: int = 2, dropout: float = .3):
        super().__init__()
        self.feature_extractor = nn.Sequential(nn.Conv1d(input_size, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.gru = nn.GRU(64, hidden_size, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
        _init_recurrent(self.gru)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x.transpose(1, 2)).transpose(1, 2)
        hidden, _ = self.gru(features)
        return self.classifier(hidden.max(dim=1).values)


class LSTMEstimator(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 256, n_layers: int = 3, dropout: float = .3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(input_size, 32, 11, padding=5), nn.BatchNorm1d(32), nn.ReLU(), nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU())
        self.lstm = nn.LSTM(128, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.magnitude_head = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.distance_head = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
        _init_recurrent(self.lstm)

    def forward(self, x: torch.Tensor):
        output, _ = self.lstm(self.conv(x.transpose(1, 2)).transpose(1, 2))
        weights = F.softmax(self.attention(output), dim=1)
        context = (weights * output).sum(dim=1)
        return self.magnitude_head(context).squeeze(-1), self.distance_head(context).squeeze(-1), weights.squeeze(-1)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_s, self.W_h, self.v = nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        scores = self.v(torch.tanh(self.W_s(decoder_hidden).unsqueeze(1) + self.W_h(encoder_outputs))).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * encoder_outputs).sum(dim=1), weights


class WaveformEncoder(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 256, n_layers: int = 2, dropout: float = .3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(input_size, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.lstm = nn.LSTM(64, hidden_size, n_layers, batch_first=True, dropout=dropout)
        _init_recurrent(self.lstm)

    def forward(self, x: torch.Tensor):
        features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.lstm(features)


class WaveformDecoder(nn.Module):
    def __init__(self, output_size: int = 3, hidden_size: int = 256, n_layers: int = 2, dropout: float = .3):
        super().__init__()
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(output_size + hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.output_proj = nn.Linear(hidden_size * 2, output_size)
        _init_recurrent(self.lstm)

    def forward_step(self, previous: torch.Tensor, h: torch.Tensor, c: torch.Tensor, encoder_outputs: torch.Tensor):
        context, weights = self.attention(h[-1], encoder_outputs)
        output, (h, c) = self.lstm(torch.cat((previous, context), dim=-1).unsqueeze(1), (h, c))
        prediction = self.output_proj(torch.cat((output.squeeze(1), context), dim=-1))
        return prediction, h, c, weights


class SeismicSeq2Seq(nn.Module):
    """Notebook-compatible attention forecaster with explicit autoregressive decoding."""

    def __init__(self, hidden_size: int = 256, n_layers: int = 2, dropout: float = .3):
        super().__init__()
        self.encoder = WaveformEncoder(3, hidden_size, n_layers, dropout)
        self.decoder = WaveformDecoder(3, hidden_size, n_layers, dropout)

    def forward(self, encoder_input: torch.Tensor, decoder_target: torch.Tensor, teacher_force: float = 0.0):
        encoded, (h, c) = self.encoder(encoder_input)
        previous, predictions, attentions = encoder_input[:, -1], [], []
        for step in range(decoder_target.shape[1]):
            prediction, h, c, attention = self.decoder.forward_step(previous, h, c, encoded)
            predictions.append(prediction)
            attentions.append(attention)
            previous = decoder_target[:, step] if torch.rand(1).item() < teacher_force else prediction.detach()
        return torch.stack(predictions, dim=1), torch.stack(attentions, dim=1)

    def predict(self, encoder_input: torch.Tensor, steps: int | None = None):
        encoded, (h, c) = self.encoder(encoder_input)
        previous, predictions, attentions = encoder_input[:, -1], [], []
        for _ in range(steps or encoder_input.shape[1] * 3):
            prediction, h, c, attention = self.decoder.forward_step(previous, h, c, encoded)
            predictions.append(prediction)
            attentions.append(attention)
            previous = prediction
        return torch.stack(predictions, dim=1), torch.stack(attentions, dim=1)
