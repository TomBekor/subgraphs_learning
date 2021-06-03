import json
import math
import time
import os
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import nni
import logging
import networkx as nx
from sklearn import metrics
from sklearn.metrics import f1_score
from loggers import EmptyLogger, CSVLogger, PrintLogger, FileLogger, multi_logger
from model_reg import GCN, GatNet
from pre_peocess import build_2k_vectors
import pickle

CUDA_Device = 0

class ModelRunner:
    def __init__(self, conf, GS,logger, data_logger=None, is_nni=False):
        self._logger = logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._conf = conf
        self.bar = 0.5
        self._lr = conf["lr"]
        self._is_nni = is_nni
        # choosing GPU device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device != "cpu":
            with torch.cuda.device("cuda:{}".format(CUDA_Device)):
                torch.cuda.empty_cache()
            if not self._is_nni:
                self._device = torch.device("cuda:{}".format(CUDA_Device))
        self._loss = self.graphSaintLoss if GS else self.regular_loss
        self.accuracy = self.accuracy_GraphSaint if GS else self.accuracy_regular
        self._ce_loss = torch.nn.CrossEntropyLoss(reduction="mean").to(self._device)
        self._ce_loss2 = torch.nn.BCELoss(reduction='mean')


    @property
    def logger(self):
        return self._logger

    @property
    def data_logger(self):
        return self._data_logger

    def graphSaintLoss(self, calcs, beta=None, gamma=None):
        if beta is None:
            beta = 1 / len(calcs["f_ns_out"])  if len(calcs["f_ns_out"])!=0 else 0
            gamma = 1 / len(calcs["s_ns_out"])  if len(calcs["s_ns_out"])!=0 else 0

        cn_loss = self._ce_loss2(calcs["cn_out"], calcs["cn_label"].float())
        f_ns_loss = self._ce_loss2(calcs["f_ns_out"], calcs["f_ns_labels"].float()) *(beta) if len(calcs["f_ns_out"])!=0 else 0
        s_ns_loss =  self._ce_loss2(calcs["s_ns_out"], calcs["s_ns_labels"].float()) * (gamma) if len(calcs["s_ns_out"])!=0 else 0
        return cn_loss+f_ns_loss+s_ns_loss

    def regular_loss(self, calcs, beta=None, gamma=None):
        if beta is None:
            beta = 1 / len(calcs["f_ns_out"])  if len(calcs["f_ns_out"])!=0 else 0
            gamma = 1 / len(calcs["s_ns_out"])  if len(calcs["s_ns_out"])!=0 else 0

        cn_loss = self._ce_loss(calcs["cn_out"], calcs["cn_label"].long())
        f_ns_loss = self._ce_loss(calcs["f_ns_out"], calcs["f_ns_labels"].long()) *(beta) if len(calcs["f_ns_out"])!=0 else 0
        s_ns_loss =  self._ce_loss(calcs["s_ns_out"], calcs["s_ns_labels"].long()) * (gamma) if len(calcs["s_ns_out"])!=0 else 0
        return cn_loss+f_ns_loss+s_ns_loss


    def _get_model(self):
        model = GCN(in_features=self._conf["in_features"],
                    hid_features=self._conf["hid_features"], out_features= self._conf["out_features"],
                    activation=self._conf["activation"], dropout= self._conf["dropout"])
        opt = self._conf["optimizer"](model.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])
        return {"model": model, "optimizer": opt,
                "beta": self._conf["beta"],"gamma": self._conf["gamma"],
                "labels": self._conf["labels"], "X": self._conf["X"], "ds_name": self._conf["ds_name"], "adj_tr":  self._conf["adj_tr"], "adj_te":  self._conf["adj_te"],
                "train_ind":  self._conf["train_ind"], "test_ind":  self._conf["test_ind"], "testt": self._conf["testt"], "traint": self._conf["traint"]
                }


    # verbose = 0 - silent
    # verbose = 1 - print test results
    # verbose = 2 - print train for each epoch and test results
    def run(self, verbose=2):

        if self._is_nni:
            verbose = 0
        model = self._get_model()
        ##
        loss_train, acc_train, intermediate_acc_test, losses_train, accs_train,  accs_cn_train, accs_f_train, accs_s_train, test_results = self.train(
            self._conf["epochs"],
            model=model,
            verbose=verbose)
        ##
        # Testing . ## result is only the last one! do not use. same as 7 in last
        result = self.test(model=model, verbose=verbose if not self._is_nni else 0, print_to_file=True)
        test_results.append(result)
        if self._is_nni:
            self._logger.debug('Final loss train: %3.4f' % loss_train)
            self._logger.debug('Final accuracy train: %3.4f' % acc_train)
            final_results = result["acc"]
            self._logger.debug('Final accuracy test: %3.4f' % final_results)
            # _nni.report_final_result(test_auc)

        if verbose != 0:
            names = ""
            vals = ()
            for name, val in result.items():
                names = names + name + ": %3.4f  "
                vals = vals + tuple([val])
                self._data_logger.info(name, val)
        parameters = { "lr": self._conf["lr"],
                      "weight_decay": self._conf["weight_decay"],
                      "dropout": self._conf["dropout"], "optimizer": self._conf["optim_name"]}
        return loss_train, acc_train, intermediate_acc_test, result, losses_train, accs_train, accs_cn_train, accs_f_train, accs_s_train, test_results, parameters




    def train(self, epochs, model=None, verbose=2):
        loss_train = 0.
        acc_train = 0.
        losses_train = []
        accs_train = []
        accs_cn_train = []
        accs_f_train = []
        accs_s_train = []


        test_results = []
        intermediate_test_acc = []
        for epoch in range(epochs):
            loss_train, acc_train, acc_train_cn , acc_train_f, acc_train_s= self._train(epoch, model, verbose)

            losses_train.append(loss_train)
            accs_train.append(acc_train)
            accs_cn_train.append(acc_train_cn)
            accs_f_train.append(acc_train_f)
            accs_s_train.append(acc_train_s)
            ##
            # /----------------------  FOR NNI  -------------------------
            if epoch % 5 == 0:
                test_res = self.test(model, verbose=verbose if not self._is_nni else 0)
                test_results.append(test_res)
                if self._is_nni:
                    test_acc = test_res["acc"]
                    intermediate_test_acc.append(test_acc)

        return loss_train, acc_train, intermediate_test_acc, losses_train, \
                accs_train, accs_cn_train, accs_f_train, accs_s_train, test_results

    ''' This function calculates the output and the labels for each node:
        for each node we take as an input the nodels' output and the labels, and return the output and label of the central node, of it's
        first neighbors and of it's second neighbors. NOTE: we take only those that are in train indices
    '''
    def calculate_labels_outputs(self,node,  outputs , labels, indices, ego_graph):
        f_neighbors = set(list(ego_graph.neighbors(node)))
        s_neighbors = set()
        #create second neighbors
        for f_neighbor in f_neighbors:
            for s_neighbor in ego_graph.neighbors(f_neighbor):
                if s_neighbor not in f_neighbors and s_neighbor != node and s_neighbor not in s_neighbors:
                    s_neighbors.add(s_neighbor)
        # notice we use the "index" in order to have correlation between the neihbors and the output's index (graph nodes are labeld with numbers from 0 to N (of the big graph) and the output's labels from 0 to n=len(ego graph). so using the "index" solves it (hopefully) ;)
        cn_out= outputs[[list(ego_graph.nodes).index(node)]]
        cn_label = labels[[node]]

        #create vectors for the first neighbors outputs and labels. NOTE: we take only those that are in train indices
        f_ns_out = outputs[[list(ego_graph.nodes).index(f_n) for f_n in f_neighbors if indices[f_n]]]
        f_ns_labels = labels[[f_n for f_n in f_neighbors if indices[f_n]]]
        #same for second neoghbors
        s_ns_out = outputs[[list(ego_graph.nodes).index(s_n) for s_n in s_neighbors if indices[s_n]]]
        s_ns_labels = labels[[s_n for s_n in s_neighbors if indices[s_n]]]
        return { "cn_out": cn_out, "cn_label":  cn_label, "f_ns_out": f_ns_out, "f_ns_labels": f_ns_labels,  "s_ns_out": s_ns_out, "s_ns_labels": s_ns_labels }


    def _train(self, epoch, model, verbose=2):
        model_ = model["model"]
        model_ = model_.to(self._device)
        optimizer = model["optimizer"]
        #train ind are the nodes to create subgraphs from. traint are nodes in train (that we can learn from)
        train_indices = model["train_ind"]
        model["labels"] = model["labels"].to(self._device)
        labels = model["labels"]
        beta = model["beta"]
        gamma = model["gamma"]
        model_.train()
        optimizer.zero_grad()

        loss_train = 0.
        loss_train1 = 0.
        calcs_batch = []
        BATCH_SIZE= 30
        # create subgraphs only for partial, but use labels of all train indices
        for idx,node in enumerate(train_indices):

            # adj = nx.ego_graph(model["adj_matrices"], node, radius=2)
            adj = model["adj_tr"][node]
            X_t = model["X"][list(adj.nodes)].to(device=self._device)
            output = model_(X_t, nx.adjacency_matrix(adj).tocoo())
            calcs = self.calculate_labels_outputs( node, output, labels, model["traint"], adj)
            #no batches:
            loss_train += self._loss(calcs, beta, gamma)

            # # if we want to use batches
            # loss_train1 += self._loss(calcs, beta, gamma)
            # loss_train += self._loss(calcs, beta, gamma).data.item()
            # if idx % BATCH_SIZE == 0 and idx > 0:
            #     loss_train1 /= BATCH_SIZE
            #     loss_train1.backward()
            #     optimizer.step()
            #     loss_train1 = 0.

            calcs_batch.append(calcs)

        acc_train, acc_train_cn, acc_train_f, acc_train_s = self.accuracy(calcs_batch)
        #
        loss_train /= len(train_indices)
        #
        loss_train.backward()
        optimizer.step()

        if verbose == 2:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self._logger.debug('Epoch: {:04d} '.format(epoch + 1) +
                               'ce_loss_train: {:.4f} '.format(loss_train) +

                               'acc_train: {:.4f} '.format(acc_train))
        return loss_train, acc_train, acc_train_cn , acc_train_f, acc_train_s


    ''' Accuracy function. For the graphSaint we use sigmoid on each index, then we use BCE loss, then we span the vectors(of each node's result) to one vector of all the centrals, 
        one vector of all the first neighbors, and one vector of all the second neghbors,
        put 1 in the indexes that have value >= 0.5 and 0 otherwise, then calculate f1 score on the vector
    '''
    @staticmethod
    def accuracy_GraphSaint(calcs):

        #create one vector that will contain all the central's nodes outputs (for each node in the train/test). same for labels
        out, labs = ([calcs[i]["cn_out"].data[0].tolist() for i in range(len(calcs))],
                     [calcs[i]["cn_label"].data[0].tolist() for i in range(len(calcs))])
        out = np.array(out)
        labs = np.array(labs)
        out[out > 0.5] = 1
        out[out <= 0.5] = 0
        acc_cn  = metrics.f1_score(labs, out, average="micro")

        # create one vector that will contain all the first neighbors outputs (of each node in the train/test. each node has vector of first neighbors- put it all to one long vector)
        out = []
        labs = []
        for i in range(len(calcs)):
            out += calcs[i]["f_ns_out"].data.tolist()
            labs += calcs[i]["f_ns_labels"].data.tolist()

        out=np.array(out)
        labs=np.array(labs)
        out[out > 0.5] = 1
        out[out <= 0.5] = 0

        if len(out) != 0:
            acc_f = metrics.f1_score(labs, out, average="micro")
        else:
            acc_f = np.nan

        # same for second neighbors (same as first)
        out = []
        labs = []
        for i in range(len(calcs)):
            out += calcs[i]["s_ns_out"].data.tolist()
            labs += calcs[i]["s_ns_labels"].data.tolist()

        out = np.array(out)
        labs = np.array(labs)
        out[out > 0.5] = 1
        out[out <= 0.5] = 0
        if len(out) != 0:
            # fpr, tpr, thresholds = metrics.roc_curve(labs2, out2)
            # acc_s = metrics.auc(fpr, tpr)
            acc_s =  metrics.f1_score(labs, out, average="micro")
        else:
            acc_s = np.nan

        return np.nanmean(np.array([acc_cn, acc_f, acc_s])), acc_cn, acc_f, acc_s


    def accuracy_regular(self,calcs):
        out, labs = ([calcs[i]["cn_out"].data[0].tolist() for i in range(len(calcs))],
                     [calcs[i]["cn_label"].data[0].tolist() for i in range(len(calcs))])
        acc_cn = sum(np.argmax(np.array(out), axis=1) == labs) / len(labs)

        out = []
        labs = []
        for i in range(len(calcs)):
            out += calcs[i]["f_ns_out"].data.tolist()
            labs += calcs[i]["f_ns_labels"].data.tolist()
        if len(out) != 0:
            acc_f = sum(np.argmax(np.array(out), axis=1) == labs) / len(labs)
        else:
            acc_f = np.nan

        out = []
        labs = []
        for i in range(len(calcs)):
            out += calcs[i]["s_ns_out"].data.tolist()
            labs += calcs[i]["s_ns_labels"].data.tolist()

        if len(out) != 0:
            acc_s = sum(np.argmax(np.array(out), axis=1) == labs) / len(labs)
        else:
            acc_s = np.nan

        return np.nanmean(np.array([acc_cn, acc_f, acc_s])), acc_cn, acc_f, acc_s




    def test(self, model=None, verbose=2, print_to_file=False):
        model_ = model["model"]
        test_indices = model["test_ind"]
        labels = model["labels"]
        beta = model["beta"]
        gamma = model["gamma"]
        model_.eval()

        test_loss = 0
        calcs_batch=[]
        with torch.no_grad():
            for node in test_indices:
                # adj = nx.ego_graph(model["adj_matrices"], node, radius=2)
                adj = model["adj_te"][node]
                X_t = model["X"][list(adj.nodes)].to(device=self._device)
                output = model_(X_t, nx.adjacency_matrix(adj).tocoo())
                calcs = self.calculate_labels_outputs(node, output, labels, model["testt"], adj)
                test_loss += self._loss(calcs, beta, gamma).data.item()
                calcs_batch.append(calcs)

            test_loss /= len(test_indices)
            test_acc, acc_test_cn, acc_test_f, acc_test_s = self.accuracy(calcs_batch)

            if verbose != 0:
                self._logger.info("Test: ce_loss= {:.4f} ".format(test_loss) + "acc= {:.4f}".format(test_acc))


            result = {"loss": test_loss, "acc": test_acc, "acc_cn": acc_test_cn, "acc_f":acc_test_f, "acc_s":acc_test_s}
            return result



def plot_graphs(train_loss_mean, train_acc_mean,train_cn_acc_mean,train_f_acc_mean, train_s_acc_mean, test_loss_mean, test_acc_mean,
                test_cn_acc_mean,test_f_acc_mean,test_s_acc_mean, parameters, plots_data):


    regulariztion = str(round(parameters["weight_decay"],3))
    lr = str(round(parameters["lr"],3))
    optimizer = str(parameters["optimizer"])
    dropout = str(round(parameters["dropout"],2))


    ds_name = plots_data["ds_name"]

    #Train

    # Share a X axis with each column of subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    plt.suptitle("DataSet: " + ds_name
                    + ", final_train_accuracies_mean: " + str(round(plots_data["final_train_accuracies_mean"],2)) + ", final_train_accuracies_ste: " + str(round(plots_data["final_train_accuracies_ste"],2))
                    + "\nfinal_test_accuracies_mean: " + str(round(plots_data["final_test_accuracies_mean"],2)) + ", final_test_accuracies_ste: " + str(round(plots_data["final_test_accuracies_ste"],2))
                 + "\nlr="+lr+" reg= "+regulariztion+ ", dropout= "+dropout+", opt= "+optimizer, fontsize=12, y=0.99)

    epoch = [e for e in range(1, len(train_loss_mean)+1)]
    axes[0, 0].set_title('Loss train')
    axes[0, 0].set_xlabel("epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].plot(epoch, train_loss_mean)

    axes[0, 1].set_title('Accuracy train')
    axes[0, 1].set_xlabel("epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].plot(epoch, train_acc_mean)


    axes[0, 2].set_title('Accuracy layers Train')
    axes[0, 2].set_xlabel("epochs")
    axes[0, 2].set_ylabel("Accuracies")
    axes[0, 2].plot(epoch, train_cn_acc_mean, label='CentralNode')
    axes[0, 2].plot(epoch, train_f_acc_mean, label='FirstNeighbors')
    axes[0, 2].plot(epoch, train_s_acc_mean, label='SecondNeighbors')
    axes[0, 2].legend(loc='best')


    #test

    epoch = [e for e in range(1, len(test_loss_mean)+1)]

    axes[1, 0].set_title('Loss test')
    axes[1, 0].set_xlabel("epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].plot(epoch, test_loss_mean)


    axes[1, 1].set_title('Accuracy test')
    axes[1, 1].set_xlabel("epochs")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].plot(epoch, test_acc_mean)


    axes[1, 2].set_title('Accuracy layers Test')
    axes[1, 2].set_xlabel("epochs")
    axes[1, 2].set_ylabel("Accuracies")
    axes[1, 2].plot(epoch, test_cn_acc_mean, label='CentralNode')
    axes[1, 2].plot(epoch, test_f_acc_mean, label='FirstNeighbors')
    axes[1, 2].plot(epoch, test_s_acc_mean, label='SecondNeighbors')

    axes[1, 2].legend(loc='best')

    fig.tight_layout()
    plt.subplots_adjust(top=0.85)

    # fig.delaxes(axes[1,0])
    plt.savefig("figures/"+plots_data["ds_name"]+"_.png")

    plt.clf()
    #plt.show()




def execute_runner(runners, plots_data, is_nni=False):
    train_losses = []
    train_accuracies = []
    train_cn_accuracies = []
    train_f_accuracies = []
    train_s_accuracies = []
    test_intermediate_results = []
    test_losses = []
    test_accuracies = []
    test_cn_accuracies = []
    test_f_accuracies = []
    test_s_accuracies = []
    results = []
    last= runners[-1]
    for i in range(len(runners)):
    #for idx_r, runner in enumerate(runners):
        with torch.cuda.device("cuda:{}".format(CUDA_Device)):
            torch.cuda.empty_cache()
            time.sleep(1)
        print("trial number",i)
        result_one_iteration = runners[0].run(verbose=2)
        train_losses.append(result_one_iteration[0])
        train_accuracies.append(result_one_iteration[1])
        test_intermediate_results.append(result_one_iteration[2])
        test_losses.append(result_one_iteration[3]["loss"])
        test_accuracies.append(result_one_iteration[3]["acc"])
        results.append(result_one_iteration)
        #todo check if can be deleted (from first check - not changing)
        if len(runners) >1:
            runners=runners[1:]
        print("len runners", len(runners))

    # for printing results on graphs. for other uses - the last result is the one should be used.
    size = len(results)
    print('results - 0 - 4 - 0:')
    print(type(results))
    print(type(results[0]))
    print(type(results[0][4]))
    print(type(results[0][4][0]))
    print('results - 0 - 5 - 0:')
    print(type(results))
    print(type(results[0]))
    print(type(results[0][5]))
    print(type(results[0][5][0]))
    print('results - 0 - 6 - 0:')
    print(type(results))
    print(type(results[0]))
    print(type(results[0][6]))
    print(type(results[0][6][0]))
    #train_loss_mean = torch.stack([torch.tensor([results[j][4][i] for i in range(len(results[j][4]))]) for j in range(size)]).mean(axis=0)
    train_loss_mean = np.mean([[results[j][4][i].cpu().detach().numpy() for i in range(len(results[j][4]))] for j in range(size)], axis=0)
    print('nice')
    #train_acc_mean = torch.stack([ torch.tensor([results[j][5][i] for i in range(len(results[j][5]))]) for j in range(size) ]).mean(axis=0)
    train_acc_mean = np.mean([ [results[j][5][i] for i in range(len(results[j][5]))] for j in range(size) ], axis=0)
    print('weird')
    train_cn_acc_mean = np.mean([[results[j][6][i] for i in range(len(results[j][6]))] for j in range(size)], axis=0)
    train_f_acc_mean = np.nanmean([[results[j][7][i] for i in range(len(results[j][7]))] for j in range(size)], axis=0)
    train_s_acc_mean = np.nanmean([[results[j][8][i] for i in range(len(results[j][8]))] for j in range(size)], axis=0)
    #test_loss_mean = torch.stack([ torch.tensor([results[j][6][i]["loss"] for i in range(len(results[j][6]))]) for j in range(size) ]).mean(axis=0)
    test_loss_mean = np.mean([ [results[j][9][i]["loss"] for i in range(len(results[j][9]))] for j in range(size) ], axis=0)
    #test_acc_mean = torch.stack([ torch.tensor([torch.tensor(results[j][6][i]["acc"]) for i in range(len(results[j][6]))]) for j in range(size) ])
    test_acc_mean = np.mean([ [results[j][9][i]["acc"] for i in range(len(results[j][9]))] for j in range(size) ], axis=0 )
    test_cn_acc_mean = np.mean([[results[j][9][i]["acc_cn"] for i in range(len(results[j][9]))] for j in range(size)], axis=0)
    test_f_acc_mean = np.mean([[results[j][9][i]["acc_f"] for i in range(len(results[j][9]))] for j in range(size)], axis=0)
    test_s_acc_mean = np.mean([[results[j][9][i]["acc_s"] for i in range(len(results[j][9]))] for j in range(size)], axis=0) #todo take care of None here too?

    final_train_accuracies_mean = np.mean(train_accuracies)
    final_train_accuracies_ste = np.std(train_accuracies) / math.sqrt(len(runners))
    final_test_accuracies_mean = np.mean(test_accuracies)
    final_test_accuracies_ste = np.std(test_accuracies) / math.sqrt(len(runners))

    plots_data["final_train_accuracies_mean"] = final_train_accuracies_mean
    plots_data["final_train_accuracies_ste"] = final_train_accuracies_ste
    plots_data["final_test_accuracies_mean"] = final_test_accuracies_mean
    plots_data["final_test_accuracies_ste"] = final_test_accuracies_ste

    #plot to graphs
    plot_graphs(train_loss_mean, train_acc_mean,train_cn_acc_mean,train_f_acc_mean, train_s_acc_mean, test_loss_mean, test_acc_mean,
                test_cn_acc_mean,test_f_acc_mean,test_s_acc_mean, results[0][10], plots_data)

    if is_nni:
        mean_intermediate_res = np.mean(test_intermediate_results, axis=0)
        for i in mean_intermediate_res:
            nni.report_intermediate_result(i)
        nni.report_final_result(np.mean(test_accuracies))



    # T takes the final of each iteration and for them mkes mean and std
    last.logger.info("*" * 15 + "Final accuracy train: %3.4f" % final_train_accuracies_mean)
    last.logger.info("*" * 15 + "Std accuracy train: %3.4f" % final_train_accuracies_ste)
    last.logger.info("*" * 15 + "Final accuracy test: %3.4f" % final_test_accuracies_mean)
    last.logger.info("*" * 15 + "Std accuracy test: %3.4f" % final_test_accuracies_ste)
    last.logger.info("Finished")
    return



def build_model(rand_test_indices, train_indices,traint,testt, labels ,X, adj_tr, adj_te, in_features,
                hid_features,out_features,ds_name, activation, optimizer, epochs, dropout, lr, l2_pen,
                beta, gamma, dumping_name, GS,is_nni=False):
    optim_name="SGD"
    if optimizer==optim.Adam:
        optim_name = "Adam"
    conf = {"in_features":in_features, "hid_features": hid_features, "out_features":out_features,"ds_name":ds_name,
            "dropout": dropout, "lr": lr, "weight_decay": l2_pen,
             "beta": beta, "gamma": gamma,
            #"training_mat": training_data, "training_labels": training_labels,
            # "test_mat": test_data, "test_labels": test_labels,
            "train_ind": train_indices, "test_ind": rand_test_indices, "traint":traint,"testt":testt, "labels":labels, "X":X,
            "adj_tr": adj_tr,"adj_te": adj_te,
            "optimizer": optimizer, "epochs": epochs, "activation": activation,"optim_name":optim_name}

    products_path = os.path.join(os.getcwd(), "logs", dumping_name, time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("MyLogger", level=logging.DEBUG),
        FileLogger("results_%s" % dumping_name, path=products_path, level=logging.INFO)], name=None)

    data_logger = CSVLogger("results_%s" % dumping_name, path=products_path)
    data_logger.info("model_name", "loss", "acc")



    runner = ModelRunner(conf, GS,logger=logger, data_logger=data_logger, is_nni=is_nni)
    return runner



def main_gcn(graph,data, X, GS, BOW, labels,in_features, hid_features, out_features, ds_name,
             optimizer=optim.Adam, epochs=200, dropout=0.3, lr=0.01, l2_pen=0.005,  beta=1/4, gamma = 1/16,
             trials=1, dumping_name='', is_nni=False):
    plot_data = {"ds_name": ds_name}
    runners = []
    print("trials")
    for it in range(trials):

        traint = [0] * len(labels)
        testt =  [0] * len(labels)

        if GS:
            #for graphsSaint:
            nodes = np.array(list(range(len(labels))))
            np.random.shuffle(nodes)
            print(len(nodes))
            with open('DataSets/{}/role.json'.format(ds_name)) as json_file:
                data = json.load(json_file)
            tr_inds = data['tr']
            te_indes = data['te']

            #arr with 1 in the indexes of the train/test nodes
            for i in tr_inds:
                traint[i]=1
            for j in te_indes:
                testt[j]=1

            #choose the nodes from which we create the subgraphs
            train_nodes_to_create_subgraphs = te_indes[0:200]
            test_nodes_to_create_subgraphs = tr_inds[0:200]

            #create subgraphs
            t = time.time()
            print("create subs train")
            adj_tr = {}
            for idx, node in enumerate(train_nodes_to_create_subgraphs):  # this may be in batches for big graphs todo
                adj = graph.create_sub(node)  #nx.ego_graph(adj_matrices,node,radius=2)
                adj_tr[node]=adj
            print("took", time.time()-t)
            t = time.time()
            print("create subs test")
            adj_te = {}
            for idx, node in enumerate(test_nodes_to_create_subgraphs):  # this may be in batches for big graphs todo
                adj =  graph.create_sub(node)
                adj_te[node]=adj
            print("took", time.time() - t)

        else:
            #for small graphs:
            # for BOW input :
            traint = data.train_mask
            tr_inds = [i for i, x in enumerate(data.train_mask) if x]
            testt = data.val_mask
            te_indes = [i for i, x in enumerate(data.val_mask) if x]
            test_nodes_to_create_subgraphs = te_indes
            train_nodes_to_create_subgraphs = tr_inds
            print("create subs train")
            adj_tr = {}
            for idx, node in enumerate(train_nodes_to_create_subgraphs):  # this may be in batches for big graphs todo
                adj = nx.ego_graph(graph,node,radius=2)
                adj_tr[node]=adj
            print("create subs test")
            adj_te = {}
            for idx, node in enumerate(test_nodes_to_create_subgraphs):  # this may be in batches for big graphs todo
                adj =   nx.ego_graph(graph,node,radius=2)
                adj_te[node]=adj
            '''create x - releveant for 2k only, so we create it for each train set. with BOW not relevant.'''
            if not BOW:
                X = torch.from_numpy(build_2k_vectors(graph, labels, out_features, traint)).to(dtype=torch.float )



        activation = torch.nn.functional.relu
        runner = build_model(test_nodes_to_create_subgraphs, train_nodes_to_create_subgraphs, traint, testt, labels,X, adj_tr, adj_te, in_features, hid_features,
                             out_features,ds_name, activation, optimizer, epochs, dropout, lr,
                             l2_pen, beta, gamma, dumping_name,GS, is_nni=is_nni)

        runners.append(runner)

    execute_runner(runners, plot_data, is_nni=is_nni)
    return














