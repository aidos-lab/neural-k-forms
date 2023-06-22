"""Handling graph data sets with `pytorch-lightning`."""

import itertools
import os
import torch

import numpy as np

import pytorch_lightning as pl

from sklearn.model_selection import StratifiedKFold, train_test_split

from cochain_representation_learning import DATA_ROOT

from torch_geometric.data import DataLoader

from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import LRGBDataset
from torch_geometric.datasets import TUDataset

from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms import Compose

from torch.utils.data import Subset


def _get_labels(dataset):
    """Auxiliary function for returning labels of a data set."""
    labels = []

    for i in range(len(dataset)):
        labels.append(dataset[i].y)

    return labels


def _get_class_ratios(dataset):
    """Auxiliary function for calculating the class ratios of a data set."""
    n_instances = len(dataset)

    labels = _get_labels(dataset)
    labels = [label.squeeze().tolist() for label in labels]

    ratios = np.bincount(labels).astype(float)
    ratios /= n_instances

    class_ratios = torch.tensor(ratios, dtype=torch.float32)
    return class_ratios


class OneHotDecoding(BaseTransform):
    def __call__(self, data):
        """Adjust multi-class labels (reverse one-hot encoding).

        This is necessary because some data sets use one-hot encoding
        for their labels, wreaks havoc with some multi-class tasks.
        """
        label = data["y"]

        if len(label.shape) > 1:
            label = label.squeeze().tolist()
            label = label.index(1.0)

        data["y"] = torch.as_tensor([label])
        return data


class ConvertGraphToChains(BaseTransform):
    def __call__(self, data):
        """Convert graph into chains."""
        node_features = data["x"]
        edge_index = data["edge_index"].T

        r = edge_index.shape[0]
        n = node_features.shape[1]

        # Sort edge indices and prepare chains. We will subsequently
        # fill them up.
        edges, _ = torch.sort(edge_index, dim=1)
        chains = torch.zeros((r, 2, n))

        # turn edges into a 1-chain
        for i in range(r):
            chains[i, 0, :] = node_features[edges[i][0]]
            chains[i, 1, :] = node_features[edges[i][1]]

        data["chains"] = chains
        return data


class LargeGraphDataset(pl.LightningDataModule):
    def __init__(
        self,
        name,
        batch_size,
    ):
        super().__init__()
        self.name = name
        self.batch_size = batch_size

        if name in ["MNIST", "PATTERN"]:
            self.base_class = GNNBenchmarkDataset
            self.transform = ConvertGraphToChains()
            self.root = os.path.join(DATA_ROOT, "GNNB")
        else:
            self.base_class = LRGBDataset

            # This is only required for the long-range graph benchmark
            # data sets.
            self.transform = Compose(
                [OneHotDecoding(), ConvertGraphToChains()]
            )
            self.root = os.path.join(DATA_ROOT, "LRGB")

    def prepare_data(self):
        def _load_data(split):
            dataset = self.base_class(
                root=self.root,
                name=self.name,
                split=split,
                transform=self.transform,
            )

            return dataset

        self.train = _load_data("train")
        self.val = _load_data("val")
        self.test = _load_data("test")

        self.num_classes = self.train.num_classes
        self.num_features = self.train.num_features
        self.class_ratios = _get_class_ratios(self.train)

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
        self.transform = ConvertGraphToChains()
        self.pre_transform = None

        self.n_splits = n_splits
        self.fold = fold

    def prepare_data(self):
        dataset = TUDataset(
            root=os.path.join(DATA_ROOT, "TU"),
            name=self.name,
            cleaned=False,
            transform=self.transform,
            use_node_attr=True,
            pre_transform=self.pre_transform,
        )

        self.num_classes = dataset.num_classes
        self.num_features = dataset.num_features

        n_instances = len(dataset)
        labels = _get_labels(dataset)

        self.class_ratios = _get_class_ratios(dataset)

        skf = StratifiedKFold(
            n_splits=self.n_splits, random_state=self.seed, shuffle=True
        )

        skf_iterator = skf.split(
            torch.tensor([i for i in range(n_instances)]),
            torch.tensor(labels),
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
