"""EGNN implementation.

Source: https://github.com/senya-ashukha/simple-equivariant-gnn/tree/main
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


def index_sum(agg_size, source, idx, cuda):
    """
    source is N x hid_dim [float]
    idx    is N           [int]

    Sums the rows source[.] with the same idx[.];
    """
    tmp = torch.zeros((agg_size, source.shape[1]))
    tmp = tmp.cuda() if cuda else tmp
    res = torch.index_add(tmp, 0, idx, source)
    return res


class ConvEGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, cuda=True):
        super().__init__()
        self.hid_dim = hid_dim
        self.cuda = cuda

        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
        )

        # preducts "soft" edges based on messages
        self.f_inf = nn.Sequential(nn.Linear(hid_dim, 1), nn.Sigmoid())

        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(in_dim + hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
        )

    def forward(self, b):
        e_st, e_end = b.edges[:, 0], b.edges[:, 1]
        dists = torch.norm(b.x[e_st] - b.x[e_end], dim=1).reshape(-1, 1)

        # compute messages
        tmp = torch.hstack([b.h[e_st], b.h[e_end], dists])
        m_ij = self.f_e(tmp)

        # predict edges
        e_ij = self.f_inf(m_ij)

        # average e_ij-weighted messages
        # m_i is num_nodes x hid_dim
        m_i = index_sum(b.h.shape[0], e_ij * m_ij, b.edges[:, 0], self.cuda)

        # update hidden representations
        b.h += self.f_h(torch.hstack([b.h, m_i]))

        return b


class EGNN(nn.Module):
    def __init__(
        self,
        in_channels=15,
        hidden_channels=128,
        out_channels=1,
        num_layers=7,
        cuda=False,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.emb = nn.Linear(in_channels, hidden_channels)

        self.gnn = [
            ConvEGNN(hidden_channels, hidden_channels, cuda=cuda)
            for _ in range(num_layers)
        ]
        self.gnn = nn.Sequential(*self.gnn)

        self.pre_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.post_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )

        if cuda:
            self.cuda()
        self.cuda = cuda

    def forward(self, b):
        b.h = self.emb(b["x"])
        b.edges = b.edge_index
        b.nG = b.num_graphs

        b = self.gnn(b)
        h_nodes = self.pre_mlp(b.h)

        # h_graph is num_graphs x hid_dim
        h_graph = index_sum(b.nG, h_nodes, b.batch, self.cuda)

        out = self.post_mlp(h_graph)
        return out
