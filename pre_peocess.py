import networkx as nx
import pickle
import numpy as np
import os

# dataSetName = "PubMed"
# num_classes = 3
# avarage_deg  = 4.496018664096972

'DataSets: '
dataSetName = "cora"; num_classes = 7; avarage_deg  = 3.8980797636632203
'dataSetName = "CiteSeer"; num_classes = 6; avarage_deg  = 2.7363991584009617' \
'dataSetName = "PubMed"; num_classes = 3; avarage_deg  = 4.496018664096972'

def build_2k_vectors(gnx,labels, num_classes, train_indices):



    print("start bulding X")
    X = np.zeros((len(gnx), 2 * num_classes))
    for i in range(X.shape[0]):
        # if i%100 == 0:
        #     print("iteration number", i)
        f_neighbors = list(gnx.neighbors(i))
        s_neighbors = []
        for f_neighbor in f_neighbors:
            for s_neighbor in gnx.neighbors(f_neighbor):
                if s_neighbor not in f_neighbors and s_neighbor != i and s_neighbor not in s_neighbors:
                    s_neighbors += [s_neighbor]
        #sub = nx.ego_graph(gnx, 0, radius= 2)
        'Building the 2k vectors for  each node. "if n1 in train_indices " is for making cosideration only for nodes from train, as described in the article'
        X[i][0:num_classes] = [len([n1 for n1 in f_neighbors if train_indices[n1] and labels[n1] == cls]) for cls in range(num_classes)]
        X[i][num_classes:] = [len([n2 for n2 in s_neighbors if train_indices[n2] and labels[n2] == cls]) for cls in range(num_classes)]
    print("finish bulding X")


    return X







if __name__ == '__main__':

    b=3
