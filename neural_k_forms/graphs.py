"""Neural k-form learning on graph data sets.

This is the main script for cochain representation learning on graphs.
It is compatible with the `TUDataset` class from `pytorch-geometric`.
"""

import argparse

import torch
import torch.nn as nn

import torchmetrics as tm
import pytorch_lightning as pl

from neural_k_forms import generate_cochain_data_matrix

from neural_k_forms.forms import NeuralOneForm

from neural_k_forms.graph_datasets import LargeGraphDataset
from neural_k_forms.graph_datasets import SmallGraphDataset

from neural_k_forms.baselines import EGNN

from torch_geometric.nn.models import GAT
from torch_geometric.nn.models import GCN
from torch_geometric.nn.models import GIN

from torch_geometric.nn.pool import global_add_pool


class ChainModel(nn.Module):
    """Simple 1-chain model using linear layers."""

    # TODO (BR): need to discuss the relevance of the respective channel
    # sizes; maybe we should also permit deeper MLPs?
    def __init__(
        self,
        input_dim,
        num_classes,
        num_steps=5,
        hidden_dim=32,
        use_batch_norm=True,
        use_attention=False,
    ):
        super().__init__()

        # TODO: num_steps and num_cochains could be different.
        self.one_form = NeuralOneForm(
            input_dim, hidden_dim, num_cochains=num_steps
        )

        self.attention = (
            nn.MultiheadAttention(num_steps, 1) if use_attention else None
        )

        self.batch_norm = nn.BatchNorm1d(num_steps) if use_batch_norm else None

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
        x = batch["chains"]

        edge_slices = batch._slice_dict["edge_index"]
        batch_size = len(edge_slices) - 1

        # Will store all features generated by the model; this should be
        # a fixed-size representation for a graph.
        all_features = []

        for i in range(batch_size):
            chains = x[edge_slices[i]:edge_slices[i + 1], :]  # fmt: skip

            X = generate_cochain_data_matrix(self.one_form, chains)

            if self.attention is not None:
                X, _ = self.attention(X, X, X, need_weights=False)

            # This corresponds to the orientation-invariant square
            # $L^2$ norm readout function.
            X = torch.diag(X.T @ X)

            all_features.append(X)

        all_features = torch.stack(all_features)

        if self.batch_norm is not None:
            all_features = self.batch_norm(all_features)

        pred = self.classifier(all_features)
        pred = nn.functional.log_softmax(pred, -1)

        return pred


class BaselineModel(nn.Module):
    """Simple baseline model.

    The purpose of this module is to wrap a simple GNN module with
    a roughly similar number of parameters as a chain-based model.
    """

    name_to_class = {
        "EGNN": EGNN,
        "GAT": GAT,
        "GIN": GIN,
        "GCN": GCN,
    }

    def __init__(
        self, input_dim, num_layers, num_classes, hidden_dim=32, baseline="GIN"
    ):
        super().__init__()

        base_class = self.name_to_class[baseline]

        self.model = base_class(
            in_channels=input_dim,
            # This means that the model will have (roughly!) the same
            # number of parameters as ours.
            hidden_channels=2 * hidden_dim,
            num_layers=num_layers,
            out_channels=num_classes,
        )

        self.name = baseline

    def forward(self, data):
        # The other baselines have a slightly different way of handling
        # data, requiring the node features and edge information.
        if self.name != "EGNN":
            x, edge_index = data.x, data.edge_index

            x = self.model(x, edge_index)
            x = global_add_pool(x, data.batch, size=len(data))
        else:
            x = self.model(data)

        return nn.functional.log_softmax(x, dim=-1)


class ModelWrapper(pl.LightningModule):
    """Wrapper class for learning on graphs.

    The purpose of this wrapper is to permit learning representations
    with various internal models, which we refer to as backbones. The
    wrapper provides a consistent training procedure.
    """

    def __init__(self, backbone, num_classes, class_ratios=None):
        super().__init__()

        self.backbone = backbone
        self.loss_fn = nn.NLLLoss(weight=class_ratios)

        self.train_accuracy = tm.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.validation_accuracy = tm.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_accuracy = tm.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.train_average_precision = tm.AveragePrecision(
            task="multiclass", num_classes=num_classes
        )
        self.validation_average_precision = tm.AveragePrecision(
            task="multiclass", num_classes=num_classes
        )
        self.test_average_precision = tm.AveragePrecision(
            task="multiclass", num_classes=num_classes
        )

        self.train_auroc = tm.AUROC(task="multiclass", num_classes=num_classes)
        self.validation_auroc = tm.AUROC(
            task="multiclass", num_classes=num_classes
        )
        self.test_auroc = tm.AUROC(task="multiclass", num_classes=num_classes)

    def step(
        self, batch, batch_idx, prefix, accuracy, average_precision, auroc
    ):
        y = batch["y"]

        y_pred = self.backbone(batch)
        loss = self.loss_fn(y_pred, y)

        accuracy(torch.argmax(y_pred, -1), y)
        average_precision(y_pred, y)
        auroc(y_pred, y)

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
            prog_bar=prefix == "train",
        )

        self.log(
            f"{prefix}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=prefix == "train",
            batch_size=len(batch),
        )

        self.log(
            f"{prefix}_average_precision",
            average_precision,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        self.log(
            f"{prefix}_auroc",
            auroc,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(
            batch,
            batch_idx,
            "train",
            self.train_accuracy,
            self.train_average_precision,
            self.train_auroc,
        )

    def validation_step(self, batch, batch_idx):
        return self.step(
            batch,
            batch_idx,
            "val",
            self.validation_accuracy,
            self.validation_average_precision,
            self.validation_auroc,
        )

    def test_step(self, batch, batch_idx):
        return self.step(
            batch,
            batch_idx,
            "test",
            self.test_accuracy,
            self.test_average_precision,
            self.test_auroc,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=1e-3)

        # Add a scheduler that halves the learning rate as soon as the
        # validation loss starts plateauing.
        #
        # TODO (BR): Make some of these parameters configurable.
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10
            ),
            "monitor": "val_loss",
            "frequency": 1,
            "interval": "epoch",
        }

        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-B",
        "--baseline",
        type=str,
        choices=["EGNN", "GAT", "GCN", "GIN"],
        default=None,
    )
    parser.add_argument("-S", "--num-steps", type=int, default=5)
    parser.add_argument("-e", "--max-epochs", type=int, default=50)
    parser.add_argument("-f", "--fold", type=int, default=0)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-H", "--hidden-dim", type=int, default=32)
    parser.add_argument("-n", "--name", type=str, default="AIDS")
    parser.add_argument("-t", "--tag", type=str)

    args = parser.parse_args()

    if args.name in ["Peptides-func", "MNIST", "PATTERN"]:
        dataset = LargeGraphDataset(name=args.name, batch_size=args.batch_size)
    else:
        dataset = SmallGraphDataset(
            name=args.name,
            batch_size=args.batch_size,
            seed=args.seed,
            fold=args.fold,
        )

    dataset.prepare_data()

    tags = [args.tag] if args.tag is not None else []
    if args.baseline:
        tags.append("baseline")

    wandb_logger = pl.loggers.WandbLogger(
        name=args.name,
        entity="aidos-labs",
        project="cochain-representation-learning",
        log_model=False,
        tags=tags,
    )

    # Store the configuration in the logger so that we can make
    # everything searchable later on.
    config = {
        "model": args.baseline if args.baseline is not None else "DRACO",
        "num_steps": args.num_steps,
        "max_epochs": args.max_epochs,
        "fold": args.fold,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "dataset": args.name,
    }

    wandb_logger.experiment.config.update(config)

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=40,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[early_stopping, lr_monitor],
    )

    if args.baseline is not None:
        backbone = BaselineModel(
            input_dim=dataset.num_features,
            num_classes=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_steps,
            baseline=args.baseline,
        )
    else:
        backbone = ChainModel(
            input_dim=dataset.num_features,
            num_classes=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            num_steps=args.num_steps,
        )

    model = ModelWrapper(backbone, dataset.num_classes, dataset.class_ratios)
    trainer.fit(model, dataset)
    trainer.test(model, dataset)