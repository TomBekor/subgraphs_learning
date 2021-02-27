import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.nn import ModuleList
from torch.nn import functional
from torch.nn import Dropout
from torch_geometric.nn import GCNConv, GATConv


class GCN(Module):
    def __init__(self, in_features=14, hid_features=10, out_features=7, activation=functional.relu, dropout = 0):
        super(GCN, self).__init__()
        self._layer1 = GCNConv(in_features, hid_features)
        self._activation = activation
        self._layer2 = GCNConv(hid_features, out_features)
        self._dropout = Dropout(p=dropout)

    @staticmethod
    def adj_to_coo(adj, device):
        edges_tensor = np.vstack((adj.row, adj.col))
        return torch.tensor(edges_tensor, dtype=torch.long).to(device)

    def forward(self, x, adj):
        adj = self.adj_to_coo(adj, x.device)
        x = self._layer1(x, adj)
        x = self._activation(x)
        x = self._dropout(x)
        x = self._layer2(x, adj)

        adj=adj.to("cpu")
        # return x
        return torch.sigmoid(x)


# # original GAT
# class GatNet(torch.nn.Module):
#     def __init__(self, num_features, num_classes, h_layers=[8], dropout=0.6, activation="elu", heads=[8,1]):
#         super(GatNet, self).__init__()
#         self.conv1 = GATConv(num_features, h_layers[0], heads=heads[0], dropout=dropout)
#         # On the Pubmed dataset, use heads=8 in conv2.
#         self.conv2 = GATConv(h_layers[0] * heads[0], num_classes, heads=heads[1], concat=False, dropout=dropout)
#
#         self._dropout = dropout
#         if activation == 'tanh':
#             self._activation_func = F.tanh
#         elif activation == 'elu':
#             self._activation_func = F.elu
#         else:
#             self._activation_func = F.relu
#
#     def forward(self, data: torch_geometric.data, topo_edges=None):
#         x = F.dropout(data.x, p=self._dropout, training=self.training)
#         x = self._activation_func(self.conv1(x, data.edge_index))
#         x = F.dropout(x, p=self._dropout, training=self.training)
#         x = self.conv2(x, data.edge_index)
#         return F.log_softmax(x, dim=1)
#

# original GAT
class GatNet(torch.nn.Module):
    def __init__(self, in_features=14, hid_features=10, out_features=7, activation=functional.relu, dropout = 0):
        super(GatNet, self).__init__()
        self._conv1 = GATConv(in_features, hid_features)
        # On the Pubmed dataset, use heads=8 in conv2.
        self._conv2 = GATConv(hid_features, out_features)
        self._activation = activation
        self._dropout = Dropout(p=dropout)


    @staticmethod
    def adj_to_coo(adj, device):
        edges_tensor = np.vstack((adj.row, adj.col))
        return torch.tensor(edges_tensor, dtype=torch.long).to(device)

    def forward(self, x, adj):
        adj = self.adj_to_coo(adj, x.device)
        x = self._conv1(x, adj)
        x = self._activation(x)
        x = self._dropout(x)
        x = self._conv2(x, adj)

        return torch.softmax(x, dim=1)
