import json
import sys
from itertools import product
import time
import scipy
from torch.optim import Adam, SGD
from model_runner_reg import main_gcn
from torch_geometric.datasets import Planetoid, CoraFull, Coauthor, PPI
import os
import torch
import numpy as np
import networkx as nx
import pickle
import scipy.sparse as sp
# from miniBatch import Minibatch
from sklearn.preprocessing import StandardScaler
# import my_LOL
import LOL.lol_graph as lol

'''
SUBGRAPHS project. Has 3 parts:
1. small datasets (cora, citeseer, pubmed, pysics, cs. options: BOW=True / Flse (if false, 2k vector input is calculated instead 
2. HUGE data set (steam)
3. Graph Saint datasets (flickr, pubmed, yelp, ppi, amazon)

CHOOSE OPTIONS IN MAIN
'''


class GCN_subgraphs:
    def __init__(self, GS=True, nni=False):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if GS:
            self.load_data_GraphSaint()
        else:
            self._load_data_small_graphs()
        self._nni = nni

    # load the graphSaint datasets
    def load_data_GraphSaint(self):
        temp_data = self.load_m()
        train_data = self.process_graph_data(*temp_data)
        adj_full, adj_train, feat_full, class_arr, role = train_data
        adj_full = adj_full.astype(np.int32)
        # adj_train = adj_train.astype(np.int32)
        # adj_full_norm = adj_norm(adj_full)
        self._num_classes = class_arr.shape[1]
        # adj = _coo_scipy2torch(adj_full_norm.tocoo())
        print("create graph")
        t = time.time()
        graph = nx.from_scipy_sparse_matrix(adj_full)
        # convert the graph to the LOL format
        undirected_graph = lol.LolGraph(directed=False, weighted=False)
        undirected_graph.convert(list(graph.edges))
        self._g = undirected_graph
        print("took", time.time() - t)
        # nx.write_edgelist(self._g, "amazon.edgelist")
        # self._labels = torch.tensor(np.argwhere(class_arr==1).T[1])
        self._labels = torch.tensor(class_arr)
        self._X = torch.tensor(feat_full).to(dtype=torch.float)
        self.in_features = feat_full.shape[1]

    # relevant for the samll datasets
    def _load_data_small_graphs(self):
        # m here complete, fill the model runner from reg notmnt, and configurations and loss
        data_transform = None
        print("loading data")
        self._data_path = './DataSets/{}'.format(dataSetName)
        if dataSetName == "CoraFull":
            self._data_set = CoraFull(self._data_path)
        elif dataSetName in {"CS", "Physics"}:
            self._data_set = Coauthor(self._data_path, dataSetName)
        else:
            self._data_set = Planetoid(self._data_path, dataSetName, data_transform)

        self._data_set.data.to(self._device)
        self._data = self._data_set[0]
        self._labels = self._data.y
        self._num_classes = self._data_set.num_classes
        self._g = self.create_graph()

        if BOW:
            self._X = self._data.x
            self.in_features = self._data.num_features

        else:  # 2k vectors input
            self._X = None
            self.in_features = num_classes * 2
            self._num_classes = num_classes

    def create_graph(self):
        nodes = list(range(self._data.num_nodes))
        edges = list(zip(self._data.edge_index[0].cpu().numpy(), self._data.edge_index[1].cpu().numpy()))
        g = nx.DiGraph() if directed else nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        labels = self._labels
        if labels.dim() == 1:
            g.graph["node_labels"] = list(set(labels.tolist()))
            for x in nodes:
                g.nodes[x]['label'] = labels[x].item()
        else:
            g.graph["node_labels"] = list(range(labels.shape[1]))
            for x in nodes:
                g.nodes[x]['label'] = list(np.where(labels[x].cpu().numpy() == 1)[0])
        return g

    def load_m(self, normalize=True):
        adj_full = scipy.sparse.load_npz('DataSets/{}/adj_full.npz'.format(dataSetName)).astype(np.bool)
        adj_train = scipy.sparse.load_npz('DataSets/{}/adj_train.npz'.format(dataSetName)).astype(np.bool)
        t = time.time()
        role = json.load(open('DataSets/{}/role.json'.format(dataSetName)))
        print("took", time.time() - t)
        feats = np.load('DataSets/{}/feats.npy'.format(dataSetName))
        class_map = json.load(open('DataSets/{}/class_map.json'.format(dataSetName)))
        class_map = {int(k): v for k, v in class_map.items()}
        assert len(class_map) == feats.shape[0]
        # ---- normalize feats ----
        train_nodes = np.array(list(set(adj_train.nonzero()[0])))
        train_feats = feats[train_nodes]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
        # -------------------------
        return adj_full, adj_train, feats, class_map, role

    def process_graph_data(self, adj_full, adj_train, feats, class_map, role):
        """
        setup vertex property map for output classes, train/val/test masks, and feats
        """
        num_vertices = adj_full.shape[0]
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            class_arr = np.zeros((num_vertices, num_classes))
            for k, v in class_map.items():
                class_arr[k] = v
        else:
            num_classes = max(class_map.values()) - min(class_map.values()) + 1
            class_arr = np.zeros((num_vertices, num_classes))
            offset = min(class_map.values())
            for k, v in class_map.items():
                class_arr[k][v - offset] = 1
        return adj_full, adj_train, feats, class_arr, role

    def train(self, GS, BOW, input_params=None):
        if input_params is None:

            beta = 1 / 5
            gamma = 1 / (5 * 5)
            try:
                data = self._data
            except:
                data = None

            # TODO
            # _ = main_gcn(graph=self._g, X=self._X, data=data,
            #              labels=self._labels, in_features=self.in_features,
            #              hid_features=25, out_features=self._num_classes, ds_name=dataSetName,
            #              epochs=100, dropout=0.2923933209489382, lr=0.00448457, l2_pen=0.0029646435597208516,
            #              beta=beta, gamma=gamma, GS=GS, BOW=BOW,
            #              trials=1, dumping_name='',
            #              optimizer=Adam,
            #              is_nni=self._nni)

            _ = main_gcn(graph=self._g, X=self._X, data=data,
                         labels=self._labels, in_features=self.in_features,
                         hid_features=25, out_features=self._num_classes, ds_name=dataSetName,
                         epochs=1, dropout=0.2923933209489382, lr=0.00448457, l2_pen=0.0029646435597208516,
                         beta=beta, gamma=gamma, GS=GS, BOW=BOW,
                         trials=1, dumping_name='',
                         optimizer=Adam,
                         is_nni=self._nni)


        else:  # NNI
            beta = 1 / avarage_deg
            gamma = 1 / (avarage_deg * avarage_deg)
            _ = main_gcn(adj_matrices=self._adjacency_matrices,
                         labels=self._labels, in_features=num_classes * 2,
                         hid_features=int(input_params["hid_features"]), out_features=num_classes, ds_name=dataSetName,
                         epochs=input_params["epochs"], dropout=input_params["dropout"],
                         lr=input_params["lr"], l2_pen=input_params["regularization"],
                         beta=beta, gamma=gamma,
                         trials=7, dumping_name='',
                         optimizer=input_params["optimizer"],
                         is_nni=self._nni)
        return None


def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))


def adj_norm(adj, deg=None, sort_indices=True):
    """
    Normalize adj according to the method of rw normalization.
    Note that sym norm is used in the original GCN paper (kipf),
    while rw norm is used in GraphSAGE and some other variants.
    Here we don't perform sym norm since it doesn't seem to
    help with accuracy improvement.

    # Procedure:
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order
    rather than ascending order
    """
    diag_shape = (adj.shape[0], adj.shape[1])
    D = adj.sum(1).flatten() if deg is None else deg
    norm_diag = sp.dia_matrix((1 / D, 0), shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm


if __name__ == "__main__":

    GraphSaint = True  # put true if you want to use the GraphSaint model. If so, declare the dataset name in sys.argv[1] to be one of the following: amazon, yelp, ppi, flickr, ogbn-arxiv
    BOW = None

    # for graphSaint:
    if GraphSaint:
        dataSetName = sys.argv[1]
    else:
        BOW = False  # change to true if GraphSaint is False and also if you want to use the bag of words input (and not the 2k vectors).
        # choose dataset: relevant when running on the small graphs only (and not on graphsaint):
        dataSetName = "cora"
        num_classes = 7
        avarage_deg = 3.8980797636632203
        directed = True
        # dataSetName = "CiteSeer"; num_classes = 6; avarage_deg  = 2.7363991584009617; directed=True
        # dataSetName = "PubMed"; num_classes = 3; avarage_deg  = 4.496018664096972; directed=False
        # dataSetName = "Physics"; directed=False; avarage_deg  = 4.6
        # dataSetName = "CS"; directed=False; avarage_deg  = 4.46

    gg = GCN_subgraphs(GraphSaint)
    gg.train(GraphSaint, BOW)
    t = 0
