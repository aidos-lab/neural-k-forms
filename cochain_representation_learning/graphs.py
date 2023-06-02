import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.datasets import TUDataset
import networkx as nx
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np

import pytorch_lightning as pl

from cochain_representation_learning.graph_datasets import TUGraphDataset

from cochain_representation_learning import DATA_ROOT
from cochain_representation_learning import generate_cochain_data_matrix

import argparse
import os


class SimpleModel(pl.LightningModule):
    """Simple model using conv layers and linear layers."""

    # TODO (BR): need to discuss the relevance of the respective channel
    # sizes; maybe we should also permit deeper MLPs?
    def __init__(self, n, out, c=5, m1=20, m2=10, m3=20, m4=10):
        super().__init__()

        self.n = n
        self.out = out  # TODO (BR): find a nicer variable name
        self.c = c
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

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

        # TODO (BR): incorporate ratios of classes again
        self.loss_fn = torch.nn.CrossEntropyLoss()

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

        # put output through classifier
        output = self.classifier(X)

        # softmax
        sm = nn.functional.softmax(output)

        return sm

    def training_step(self, batch, batch_idx):
        chains, label = batch["chains"], batch["y"]
        out = self(chains)

        # do a 1-hot encoding of data.y
        # TODO (BR): this can be solved in the dataset class as well
        y = torch.zeros(self.out)
        y[label] = 1

        loss = self.loss_fn(out, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # TODO (BR): default parameters?
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


## print the total number of graphs in each class in the dataset
# print("Number of graphs in each class in the dataset:")
# print("=============================================================")
# for i in range(train_dataset.num_classes):
#    print(f"Class {i}: {sum([1 for data in train_dataset if data.y == i])}")
#
## store the ratio of graphs in each class
# class_ratios = torch.tensor(
#    [
#        sum([1 for data in train_dataset if data.y == i]) / len(train_dataset)
#        for i in range(train_dataset.num_classes)
#    ]
# )
#
# weight_ratios = 1 - class_ratios
#
#
## create your optimizer
# optimizer = optim.SGD(basic_model.parameters(), lr=1e-1)
#
# criterion = torch.nn.CrossEntropyLoss(weight=weight_ratios)


def train(dataset, chainz):
    basic_model.train()

    correct = 0
    L = 0

    for i in range(
        len(dataset)
    ):  # Iterate in batches over the training dataset.
        chain = chainz[i]
        data = dataset[i]

        out = basic_model.forward(chain)  # Perform a single forward pass.

        # do a 1-hot encoding of data.y
        y = torch.zeros(dataset.num_classes)
        y[data.y] = 1

        # compute if prediction is correct
        if torch.argmax(out) == torch.argmax(y):
            correct += 1

        loss = criterion(out, y)  # Compute the loss.
        loss.backward()  # Derive gradients.

        L += loss.item()

        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    return L / len(dataset), correct / len(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--name", type=str, default="MUTAG")

    args = parser.parse_args()

    dataset = TUGraphDataset(name=args.name, batch_size=args.batch_size)
    dataset.prepare_data()

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
    )

    print(dataset.num_features, dataset.num_classes)

    model = SimpleModel(n=dataset.num_features, out=dataset.num_classes)
    trainer.fit(model, dataset)
