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

from cochain_representation_learning import DATA_ROOT
from cochain_representation_learning import generate_cochain_data_matrix

import os


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


dataset = TUDataset(root=os.path.join(DATA_ROOT, "TU"), name="PROTEINS_full")
describe(dataset)

torch.manual_seed(12345)
dataset = dataset.shuffle()

#train_dataset = dataset[:len_train_set]
#test_dataset = dataset[len_train_set:]
#
#print(f"Number of training graphs: {len(train_dataset)}")
#print(f"Number of test graphs: {len(test_dataset)}")


def graph_to_chain(graph):
    """
    A function for turning a graph into a chain
    """

    # get node features
    # TODO (BR): do we need to copy this?
    node_features = torch.tensor(graph["x"])

    # get edges
    # TODO (BR): do we need to copy this?
    edge_index = torch.tensor(data["edge_index"]).T

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

    return ch


# example
chainz = []
for i in range(len(dataset)):
    data = dataset[i]
    chainz.append(graph_to_chain(data))


class model(nn.Module):

    """Define a simple model using convolutional layers and linear layers
    to reduce the dim of the output"""

    def __init__(
        self, n, out, c=5, m1=20, m2=10, m3=20, m4=10
    ):  ## check channel sizes
        super().__init__()
        self.n = n
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

    def forward(self, x):
        ## asses the dimensions are correct somewhere
        "Here the input is a chain, and the output is a vector of probabilities"

        # generate cochain data matrix
        X = generate_cochain_data_matrix(self.vf, x)

        # orientation invariant square L2-norm readout function
        X = torch.diag(X.T @ X)

        # put output through classifier
        output = self.classifier(X)

        # softmax
        sm = nn.functional.softmax(output)

        return sm

n = dataset[0]['x'].shape[1] # cochain feature dimension
out = dataset.num_classes # output dimension

basic_model = model(n = n, out = out)

# FIXME (BR): This is wrong.
train_dataset = dataset

# print the total number of graphs in each class in the dataset
print('Number of graphs in each class in the dataset:')
print('=============================================================')
for i in range(train_dataset.num_classes):
    print(f'Class {i}: {sum([1 for data in train_dataset if data.y == i])}')

# store the ratio of graphs in each class
class_ratios = torch.tensor([sum([1 for data in train_dataset if data.y == i]) / len(train_dataset) for i in range(train_dataset.num_classes)])

weight_ratios = 1 - class_ratios


# create your optimizer
optimizer = optim.SGD(basic_model.parameters(), lr=1e-1)

criterion = torch.nn.CrossEntropyLoss(weight=weight_ratios)


def train(dataset, chainz):
    basic_model.train()

    correct = 0
    L = 0

    for i in range(len(dataset)):  # Iterate in batches over the training dataset.

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


epochs = 10

losses = torch.zeros(epochs)
accuracies = torch.zeros(epochs)

for j in range(epochs):

    L, A = train(train_dataset, chainz)

    losses[j] = L
    accuracies[j] = A
    
    print("Epoch = ", j, "Loss = ", L, "Accuracy = ", A)
