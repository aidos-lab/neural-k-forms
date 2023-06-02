"""Handling graph data sets with `pytorch-lightning`."""

import itertools
import os
import torch

import numpy as np

import pytorch_lightning as pl

from sklearn.model_selection import StratifiedKFold, train_test_split

from cochain_representation_learning import DATA_ROOT

from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from torch.utils.data import Subset


def _get_labels(dataset):
    labels = []

    for i in range(len(dataset)):
        labels.append(dataset[i].y)

    return labels


def _graph_to_chain(graph):
    """Convert graph into chains."""

    # get node features
    # TODO (BR): do we need to copy this?
    node_features = torch.tensor(graph["x"])

    # get edges
    # TODO (BR): do we need to copy this?
    edge_index = torch.tensor(graph["edge_index"]).T

    # number of 1-simplices
    r = edge_index.shape[0]

    # embedding dimension
    n = node_features.shape[1]

    # sort the edge indices
    edges = torch.tensor(
        [
            np.sort([edge_index[i][0], edge_index[i][1]])
            for i in range(len(edge_index))
        ]
    )

    # initialize chain
    ch = torch.zeros((r, 2, n))

    # turn edges into a 1-chain
    for i in range(r):
        ch[i, 0, :] = node_features[edges[i][0]]
        ch[i, 1, :] = node_features[edges[i][1]]

    # TODO (BR): is this the smartest type of transformation here?
    # Should we rather try to copy everything?
    return Data(
        x=graph["x"], y=graph["y"], edge_index=graph["edge_index"], chains=ch
    )


def describe(dataset):
    """Describe data set in textual form."""
    print()
    print(f"Dataset: {dataset}:")
    print("====================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")

    # Get the first graph object.
    data = dataset[0]

    print()
    print(data)
    print("=============================================================")

    # Gather some statistics about the first graph.
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"Has isolated nodes: {data.has_isolated_nodes()}")
    print(f"Has self-loops: {data.has_self_loops()}")
    print(f"Is undirected: {data.is_undirected()}")


class TUGraphDataset(pl.LightningDataModule):
    def __init__(
        self,
        name,
        batch_size,
        val_fraction=0.1,
        test_fraction=0.1,
        fold=0,
        seed=42,
        n_splits=5,
        legacy=True,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed

        # TODO (BR): We can use this to check whether the data set
        # actually has some node attributes. If not, we can assign
        # some based on the degrees, for instance.
        self.transform = None
        self.pre_transform = _graph_to_chain

        self.n_splits = n_splits
        self.fold = fold

    def prepare_data(self):
        # TODO (BR): Do we want to have cleaned versions of these data
        # sets that include only non-isomorphic graphs?
        cleaned = False

        dataset = TUDataset(
            root=os.path.join(DATA_ROOT, "TU"),
            name=self.name,
            # TODO (BR): Can we use these attributes somehow?
            use_node_attr=False,
            cleaned=cleaned,
            transform=self.transform,
            pre_transform=self.pre_transform,
        )

        describe(dataset)

        self.num_classes = dataset.num_classes

        n_instances = len(dataset)

        skf = StratifiedKFold(
            n_splits=self.n_splits, random_state=self.seed, shuffle=True
        )

        skf_iterator = skf.split(
            torch.tensor([i for i in range(n_instances)]),
            torch.tensor(_get_labels(dataset)),
        )

        train_index, test_index = next(
            itertools.islice(skf_iterator, self.fold, None)
        )
        train_index, val_index = train_test_split(
            train_index, random_state=self.seed
        )

        train_index = train_index.tolist()
        val_index = val_index.tolist()
        test_index = test_index.tolist()

        self.train = Subset(dataset, train_index)
        self.val = Subset(dataset, val_index)
        self.test = Subset(dataset, test_index)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
