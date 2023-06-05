"""Cochain learning on graph data sets.

This is the main script for cochain representation learning on graphs.
It is compatible with the `TUDataset` class from `pytorch-geometric`.
"""

import argparse

import torch
import torch.nn as nn

import pytorch_lightning as pl
import torchmetrics as tm

from cochain_representation_learning import generate_cochain_data_matrix
from cochain_representation_learning.graph_datasets import TUGraphDataset


class SimpleModel(nn.Module):
    """Simple model using linear layers."""

    # TODO (BR): need to discuss the relevance of the respective channel
    # sizes; maybe we should also permit deeper MLPs?
    def __init__(
        self,
        n,
        num_classes,
        c=5,
        m1=20,
        m2=10,
        hidden_dim=64
    ):
        super().__init__()

        self.n = n
        self.num_classes = num_classes
        self.c = c
        self.m1 = m1
        self.m2 = m2

        self.vector_field = nn.Sequential(
            nn.Linear(n, m1),
            nn.ReLU(),
            nn.Linear(m1, m2),
            nn.ReLU(),
            nn.Linear(m2, n * c),
        )

        self.classifier = nn.Sequential(
            nn.Linear(c, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    # TODO (BR): document
    def forward(self, x):
        # asses the dimensions are correct somewhere
        # Here the input is a chain, and the output is a vector of
        # probabilities

        X = generate_cochain_data_matrix(self.vector_field, x)

        # orientation invariant square L2-norm readout function
        # TODO (BR): this is something we might want to change, no? If
        # I understand the code correctly, we are getting information
        # for all edges. I wonder whether it would make sense to think
        # about some other readout functions here (potentially using
        # attention weights).
        X = torch.diag(X.T @ X)

        pred = self.classifier(X)
        pred = nn.functional.softmax(pred, -1)

        return pred


class CochainModelWrapper(pl.LightningModule):
    """Wrapper class for cochain representation learning on graphs.

    The purpose of this wrapper is to permit learning representations
    with various internal models, which we refer to as backbones. The
    wrapper provides a consistent training procedure.
    """

    def __init__(self, backbone, num_classes, class_ratios=None):
        super().__init__()

        self.backbone = backbone
        self.loss_fn = nn.CrossEntropyLoss(weight=class_ratios)

        self.train_accuracy = tm.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.validation_accuracy = tm.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_accuracy = tm.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def step(self, batch, batch_idx, prefix, accuracy):
        x, y = batch["chains"], batch["y"]

        edge_slices = batch._slice_dict["edge_index"]
        batch_size = len(edge_slices) - 1

        loss = 0.0

        y_hat = []

        for i in range(batch_size):
            chains = x[edge_slices[i]:edge_slices[i + 1], :]  # fmt: skip

            y_pred = self.backbone(chains)
            y_pred = y_pred.view(-1, y_pred.shape[-1])

            y_hat.append(y_pred)

            loss += self.loss_fn(y_pred, y[i].view(-1))

        loss /= batch_size

        y_hat = torch.cat(y_hat)
        accuracy(torch.argmax(y_hat, -1).view(-1), y)

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            f"{prefix}_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train", self.train_accuracy)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val", self.validation_accuracy)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test", self.test_accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--name", type=str, default="AIDS")

    args = parser.parse_args()

    dataset = TUGraphDataset(
        name=args.name, batch_size=args.batch_size, seed=args.seed
    )
    dataset.prepare_data()

    wandb_logger = pl.loggers.WandbLogger(
        name=args.name,
        project="cochain-representation-learning",
        log_model=False,
    )

    trainer = pl.Trainer(max_epochs=args.max_epochs, logger=wandb_logger)

    backbone = SimpleModel(
        n=dataset.num_features,
        num_classes=dataset.num_classes,
    )

    model = CochainModelWrapper(
        backbone, dataset.num_classes, dataset.class_ratios
    )
    trainer.fit(model, dataset)
