# subgraphs_learning

Given a very large graph, containing a small number of tagged vertices, we want to be able to label a given node/nodes,
with pre-training only on sub-graphs with limited size only.

To run the code, run the file subgraphs_GCN_reg.py.
This file can run the following:
1. small datasets (Cora,CiteSeer, Pubmed, CS). 
2.  Large datasets from GraphSaint git.

In order to run the large datasets, change in main Graph_Saint option to be True.
Otherwise, you will run the small datasets, there you can choose one of the following inputs:
1. BOW (change BOW = True in main)
2. 2k vectors input (Bow=False).
 This feature represents the number of first and second neighbors belonging to each class in the training set. For example, assume a classification task with 3 possible labels, and a node with 5 neighbors and 30 second neighbors. Further assume that 1 first neighbor belongs to the training set and has label A, 3 second neighbors belong to the training set and have label A and 1 second neighbor belongs to the training
set and has label B. The input to the node would be [1,0,0,3,1,0],
where the first three values represent the first neighbors and the last three values represent the second neighbors.
Code for creating this feature can be found in pre-precess.py file.
