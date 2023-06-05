"""Cochain learning on graph data sets."""

import argparse

import torch
import torch.nn as nn

import pytorch_lightning as pl
import torchmetrics as tm

from cochain_representation_learning import generate_cochain_data_matrix
from cochain_representation_learning.graph_datasets import TUGraphDataset


class SimpleModel(pl.LightningModule):
    """Simple model using conv layers and linear layers."""

    # TODO (BR): need to discuss the relevance of the respective channel
    # sizes; maybe we should also permit deeper MLPs?
    def __init__(
        self, n, out, c=5, m1=20, m2=10, m3=20, m4=10, class_ratios=None
    ):
        super().__init__()

        self.n = n
        self.out = out  # TODO (BR): find a nicer variable name
        self.c = c
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        self.accuracy = tm.Accuracy(task="binary")

        # initialise vector field
        self.vf = nn.Sequential(
            nn.Linear(n, m1),
            nn.ReLU(),
            nn.Linear(m1, m2),
            nn.ReLU(),
            nn.Linear(m2, n * c),
        )

        # initialise MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(c, m3),
            nn.ReLU(),
            nn.Linear(m3, m4),
            nn.ReLU(),
            nn.Linear(m4, out),
        )

        self.loss_fn = nn.CrossEntropyLoss(weight=class_ratios)

    # TODO (BR): document
    def forward(self, x):
        # asses the dimensions are correct somewhere
        # Here the input is a chain, and the output is a vector of
        # probabilities

        # generate cochain data matrix
        X = generate_cochain_data_matrix(self.vf, x)

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

    def training_step(self, batch, batch_idx):
        x, y = batch["chains"], batch["y"]

        edge_slices = batch._slice_dict["edge_index"]
        batch_size = len(edge_slices) - 1

        loss = 0.0

        y_hat = []

        for i in range(batch_size):
            chains = x[edge_slices[i]:edge_slices[i + 1], :]  # fmt: skip

            y_pred = self(chains)
            y_pred = y_pred.view(-1, y_pred.shape[-1])

            y_hat.append(y_pred)

            loss += self.loss_fn(y_pred, y[i].view(-1))

        loss /= batch_size

        y_hat = torch.cat(y_hat)
        self.accuracy(torch.argmax(y_hat, -1).view(-1), y)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            "train_accuracy",
            self.accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--name", type=str, default="MUTAG")

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

    model = SimpleModel(
        n=dataset.num_features,
        out=dataset.num_classes,
        class_ratios=dataset.class_ratios,
    )

    trainer.fit(model, dataset)
