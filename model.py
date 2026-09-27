import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    """LSTM model for music generation using PyTorch."""

    def __init__(self, input_size, hidden_size=256, num_layers=3, output_size=None, dropout=0.3):
        super(MusicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden_size, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take the last time step
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def create_network(n_vocab, sequence_length=100):
    """Create and return the LSTM model."""
    model = MusicLSTM(
        input_size=1,
        hidden_size=256,
        num_layers=3,
        output_size=n_vocab,
        dropout=0.3,
    )
    return model
