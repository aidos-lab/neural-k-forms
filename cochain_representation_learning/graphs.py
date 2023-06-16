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
    def __init__(self, input_dim, num_classes, num_steps=5, hidden_dim=32):
        super().__init__()

        output_dim = input_dim * num_steps

        self.vector_field = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_steps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch):
        # `batch` is the input batch (following `pytorch-geometric`
        # conventions), containing multiple graphs. We deconstruct
        # this into different inputs (chain sets).
        #
        # TODO (BR): extend documentation :-)
        #
        x = batch["chains"]

        edge_slices = batch._slice_dict["edge_index"]
        batch_size = len(edge_slices) - 1

        all_pred = []

        for i in range(batch_size):
            chains = x[edge_slices[i]:edge_slices[i + 1], :]  # fmt: skip

            X = generate_cochain_data_matrix(self.vector_field, chains)

            # orientation invariant square L2-norm readout function
            # TODO (BR): this is something we might want to change, no? If
            # I understand the code correctly, we are getting information
            # for all edges. I wonder whether it would make sense to think
            # about some other readout functions here (potentially using
            # attention weights).
            X = torch.diag(X.T @ X)

            pred = self.classifier(X)
            pred = nn.functional.softmax(pred, -1)

            all_pred.append(pred)

        return torch.stack(all_pred)


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

        num_features = 5
        self.batch_norm = nn.BatchNorm1d(num_features)

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
        y = batch["y"]

        y_pred = self.backbone(batch)
        loss = self.loss_fn(y_pred, y)

        accuracy(torch.argmax(y_pred, -1), y)

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            f"{prefix}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
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

        # Add a scheduler that halves the learning rate as soon as the
        # validation loss starts plateauing.
        #
        # TODO (BR): Make some of these parameters configurable.
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            ),
            "monitor": "val_loss",
            "frequency": 1,
            "interval": "epoch",
        }

        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
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

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=20,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=early_stopping,
    )

    backbone = SimpleModel(
        input_dim=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_dim=args.hidden_dim,
    )

    model = CochainModelWrapper(
        backbone, dataset.num_classes, dataset.class_ratios
    )
    trainer.fit(model, dataset)
    trainer.test(model, dataset)
