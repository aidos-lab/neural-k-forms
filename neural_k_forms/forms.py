"""Represent neural $k$-forms in a simple interface."""

import torch.nn as nn


class NeuralOneForm(nn.Module):
    """Simple neural $1$-form neural network.

    This represents a neural $1$-form as a simple neural network with
    a single hidden layer.
    """

    def __init__(self, input_dim, hidden_dim, num_cochains):
        super().__init__()

        output_dim = input_dim * num_cochains

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_cochains = num_cochains

    def forward(self, x):
        return self.model(x)
