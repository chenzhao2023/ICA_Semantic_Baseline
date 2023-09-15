import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import time
import pandas as pd
import os
import argparse
import copy

import Artery
import networkx as nx
from tree_lstm import TreeLSTM
from data_prepare import convert_graph_2_tree, extract_features_with_random
from example_usage import convert_tree_to_tensors
from tqdm import tqdm

from sklearn import metrics
# from pthflops import count_ops
from flopth import flopth

class UTD_TreeLSTM(nn.Module):
    def __init__(self, in_features, mlp_hidden, lstm_hidden, out_features):
        """
        in_features: dimension of the input features for each arterial segment
        mlp_hidden: dimension of the mlp layers
        lstm_hidden: dimension of the LSTM unit, default=30
        out_features: dimension of the output space, equals to number of classes
        """
        super(UTD_TreeLSTM, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features, mlp_hidden), nn.ReLU(),
                                     nn.Linear(mlp_hidden, mlp_hidden//2), nn.ReLU(),
                                     nn.Linear(mlp_hidden//2, mlp_hidden//4), nn.ReLU(),
                                     nn.Linear(mlp_hidden//4, mlp_hidden//8), nn.ReLU())
        self.forward_tree_lstm = TreeLSTM(mlp_hidden//8, lstm_hidden)
        self.classifier = nn.Linear(lstm_hidden, out_features)
    
    def forward(self, features, node_order, adjacency_list, edge_order):
        # MLP, FORWARD TREE LSTM, BACKWARD TREE LSTM, SOFTMAX
        features = self.encoder(features)
        features, _ = self.forward_tree_lstm(features, node_order, adjacency_list, edge_order)
        features = self.classifier(features)
        features = F.softmax(features, dim=1)
        return features


class DTU_TreeLSTM(nn.Module):
    def __init__(self, in_features, mlp_hidden, lstm_hidden, out_features):
        """
        in_features: dimension of the input features for each arterial segment
        mlp_hidden: dimension of the mlp layers
        lstm_hidden: dimension of the LSTM unit, default=30
        out_features: dimension of the output space, equals to number of classes
        """
        super(DTU_TreeLSTM, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features, mlp_hidden), nn.ReLU(),
                                     nn.Linear(mlp_hidden, mlp_hidden//2), nn.ReLU(),
                                     nn.Linear(mlp_hidden//2, mlp_hidden//4), nn.ReLU(),
                                     nn.Linear(mlp_hidden//4, mlp_hidden//8), nn.ReLU())
        self.forward_tree_lstm = TreeLSTM(mlp_hidden//8, lstm_hidden)
        self.classifier = nn.Linear(lstm_hidden, out_features)
    
    def forward(self, features, node_order, adjacency_list, edge_order):
        # MLP, FORWARD TREE LSTM, BACKWARD TREE LSTM, SOFTMAX
        everted_node_order = torch.flip(node_order, dims=[0])
        reverted_edge_order = torch.flip(edge_order, dims=[0])
        reverted_adjacency_list = torch.flip(adjacency_list, dims=[0])
        features = torch.flip(features, dims=[0])

        features = self.encoder(features)
        features, _ = self.forward_tree_lstm(features, everted_node_order, reverted_adjacency_list, reverted_edge_order)
        features = self.classifier(features)
        features = F.softmax(features, dim=1)
        return features


class BiTreeLSTM(nn.Module):

    def __init__(self, in_features=12, mlp_hidden=128, lstm_hidden=30, out_features=5):
        """
        in_features: dimension of the input features for each arterial segment
        mlp_hidden: dimension of the mlp layers
        lstm_hidden: dimension of the LSTM unit, default=30
        out_features: dimension of the output space, equals to number of classes
        """
        super(BiTreeLSTM, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features, mlp_hidden), nn.ReLU(),
                                     nn.Linear(mlp_hidden, mlp_hidden//2), nn.ReLU(),
                                     nn.Linear(mlp_hidden//2, mlp_hidden//4), nn.ReLU(),
                                     nn.Linear(mlp_hidden//4, mlp_hidden//8), nn.ReLU())
        self.forward_tree_lstm = TreeLSTM(mlp_hidden//8, lstm_hidden)
        self.backward_tree_lstm = TreeLSTM(lstm_hidden, lstm_hidden)
        self.classifier = nn.Linear(lstm_hidden*2, out_features)
    
    def forward(self, features, node_order, adjacency_list, edge_order):
        # MLP, FORWARD TREE LSTM, BACKWARD TREE LSTM, SOFTMAX
        reverted_node_order = torch.flip(node_order, dims=[0])
        reverted_edge_order = torch.flip(edge_order, dims=[0])
        reverted_adjacency_list = torch.flip(adjacency_list, dims=[0])
        features = self.encoder(features)
        features_forward, _ = self.forward_tree_lstm(features, node_order, adjacency_list, edge_order)
        features_backward, _ = self.backward_tree_lstm(features_forward, 
                                                       reverted_node_order, 
                                                       reverted_adjacency_list, 
                                                       reverted_edge_order)
        features_backward = torch.flip(features_backward, dims=[0]) # reverse feature since the feature was reversed in back procedure
        features = torch.concat([features_forward, features_backward], dim=1)
        features = self.classifier(features)
        features = F.softmax(features, dim=1)
        return features


class Trainer():
    def __init__(self, params):
        self.params = params
        # init dataset
        dataset, training_samples = Artery._load_graph_in_mem(self.params.data_file_path, "")
        training_samples, test_samples = Artery.get_split_deterministic(training_samples, self.params.cv, self.params.cv_max)
        self.dataset, self.train_samples, self.test_samples = dataset, training_samples, test_samples
        self.__init_model__()

        self.rand = np.random.RandomState(seed=params.seed)

    def __init_model__(self):
        in_features = self.params.in_features
        mlp_hidden = self.params.mlp_hidden
        lstm_hidden = self.params.lstm_hidden
        out_features = self.params.num_class

        if self.params.model == 'bi':
            self.model = BiTreeLSTM(in_features, mlp_hidden, lstm_hidden, out_features)
        elif self.params.model == 'dtu':
            self.model = DTU_TreeLSTM(in_features, mlp_hidden, lstm_hidden, out_features)
        elif self.params.model == "utd":
            self.model = UTD_TreeLSTM(in_features, mlp_hidden, lstm_hidden, out_features)
        else:
            raise ValueError("! Not a supported model : ", self.params.model)

        self.model.to(self.params.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.decay) 
        self.optimizer = optimizer

        print(f"[x] number of parameters = {sum(param.numel() for param in self.model.parameters())}")
        self.loss_function = torch.nn.BCEWithLogitsLoss()

    def train(self):
        best_acc = 0.
        exp_path = os.path.join(self.params.exp, f"{self.params.model}_cv{self.params.cv}")
        if not os.path.isdir(exp_path):
            os.makedirs(exp_path)
        with open(os.path.join(exp_path, 'config.json'), 'w') as fp:
            json.dump(self.params.__dict__, fp, indent=4)

        target = open(os.path.join(exp_path, "eval.csv"), "w")
        target.write("epoch,acc,precision,recall,f1\n")
        for epoch in tqdm(range(self.params.epoch)):
            self.model.train()
            FLOPs = []
            for mini_batch in range(len(self.train_samples)):
                tree = self.dataset[self.train_samples[mini_batch]]['tree']
                data = convert_tree_to_tensors(tree, self.params.device)
                logits = self.model(data['features'], data['node_order'], data['adjacency_list'], data['edge_order'])
                # r = count_ops(self.model, input=(data['features'], data['node_order'], data['adjacency_list'], data['edge_order']))
                flops, params = flopth(self.model, inputs=(data['features'], data['node_order'], data['adjacency_list'], data['edge_order']))
                
                print(flops)
                # print(params)
                FLOPs.append(float(flops[:-1])*1000)
                labels = data['labels']
                loss = self.loss_function(logits, labels)
                loss.backward()
                self.optimizer.step()
            print(np.mean(FLOPs))
            if epoch % self.params.validation_epoch == 0:
                self.model.eval()
                preds = []
                gts = []
                last_time = time.time()
                for mini_batch in range(len(self.test_samples)):
                    tree = self.dataset[self.test_samples[mini_batch]]['tree']
                    data = convert_tree_to_tensors(tree, self.params.device)
                    logits = self.model(data['features'], data['node_order'], data['adjacency_list'], data['edge_order'])
                    pred_cls = torch.argmax(logits, dim=1).detach().cpu().numpy()
                    gt = torch.argmax(data['labels'], dim=1).detach().cpu().numpy()
                    preds.extend(pred_cls)
                    gts.extend(gt)
                acc = metrics.accuracy_score(gts, preds)
                precision = metrics.precision_score(gts, preds, average="weighted")
                recall = metrics.recall_score(gts, preds, average="weighted")
                f1_score = metrics.f1_score(gts, preds, average="weighted")
                print(f"[x] avg prediction time = {(time.time() - last_time)/len(self.test_samples)}")
                cm = metrics.confusion_matrix(gts, preds)
                clf_report = metrics.classification_report(gts, preds, target_names=Artery.MAIN_BRANCH_CATEGORY, output_dict=True)
                np.save(os.path.join(exp_path, f"confusion_matrix_{epoch:04d}.npy"), cm)
                with open(os.path.join(exp_path, f'clf_report_{epoch:04d}.json'), 'w') as fp:
                    json.dump(clf_report, fp, indent=4)
                target.write(f"{epoch},{acc:0.4f},{precision:0.4f},{recall:0.4f},{f1_score:0.4f}\n")
                target.flush()
                print(f"test @ epoch {epoch}, acc = {acc}, precision = {precision}, recall = {recall}, f1 = {f1_score}")
                print(cm)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(self.model.state_dict(), os.path.join(exp_path, "model.pth"))
        
        print(f"training done, best acc = {best_acc}")
        target.close()

    def __restore__(self):
        exp_path = os.path.join(self.params.exp, f"{self.params.model}_cv{self.params.cv}")
        print(f"[x] restore model file from {os.path.join(exp_path, 'model.pth')}")
        self.model.load_state_dict(torch.load(os.path.join(exp_path, "model.pth")))
        self.model.eval()


    def __trim_tree__(self, g, binary_image, original_image, prob):
        removed_nodes_idx = []
        removed_nodes = []
        ban_list = []
        g = copy.deepcopy(g)
        for node in g.nodes():
            if g.nodes()[node]['data'].node1.degree == 1 or g.nodes()[node]['data'].node2.degree == 1:
                if g.nodes()[node]['data'].vessel_class not in ["LMA", "LAD1", "LCX1", "LAD2", "LCX2", "LAD3", "LCX3"]:
                    # removeable
                    if self.rand.rand() < prob: # generate random value according to pre-defined seed
                        removed_nodes.append(g.nodes()[node]['data'].vessel_class)
                        removed_nodes_idx.append(node)
            elif g.nodes()[node]['data'].vessel_class in ["LMA", "LAD2", "LAD3", "LCX2", "LCX3"]:
                if self.rand.rand() < prob:
                    ban_list.append(g.nodes()[node]['data'].vessel_class)
                    removed_nodes.append(g.nodes()[node]['data'].vessel_class)

        for idx, n in enumerate(removed_nodes_idx):
            g.remove_node(n)
        # re assign node index
        mapping = {old_label:new_label for new_label, old_label in enumerate(g.nodes())}
        g = nx.relabel_nodes(g, mapping)
        
        if "OM1" in removed_nodes:
            for n in g.nodes():
                if g.nodes()[n]['data'].vessel_class == "OM2":
                    g.nodes()[n]['data'].vessel_class = "OM1"
        
        if "D1" in removed_nodes:
            for n in g.nodes():
                if g.nodes()[n]['data'].vessel_class == "D2":
                    g.nodes()[n]['data'].vessel_class = "D1"
        
        _, tree = extract_features_with_random(g, binary_image, original_image, ban_list)
        # tree = convert_graph_2_tree(g)
        return tree, removed_nodes

    def test_with_missing(self, epoch, prob):
        exp_path = os.path.join(self.params.exp, f"{self.params.model}_cv{self.params.cv}")

        self.model.eval()
        preds = []
        gts = []
        for mini_batch in range(len(self.test_samples)):
            #tree = self.dataset[self.test_samples[mini_batch]]['tree']
            g = self.dataset[self.test_samples[mini_batch]]['g']
            binary_image = self.dataset[self.test_samples[mini_batch]]['binary_image']
            original_image = self.dataset[self.test_samples[mini_batch]]['image']
            tree, removed_nodes = self.__trim_tree__(g, binary_image, original_image, prob)
            # print(f"test attach {self.test_samples[mini_batch]}, removed nodes : {removed_nodes}")
            data = convert_tree_to_tensors(tree, self.params.device)
            logits = self.model(data['features'], data['node_order'], data['adjacency_list'], data['edge_order'])
            pred_cls = torch.argmax(logits, dim=1).detach().cpu().numpy()
            gt = torch.argmax(data['labels'], dim=1).detach().cpu().numpy()
            preds.extend(pred_cls)
            gts.extend(gt)

        acc = metrics.accuracy_score(gts, preds)
        precision = metrics.precision_score(gts, preds, average="weighted")
        recall = metrics.recall_score(gts, preds, average="weighted")
        f1_score = metrics.f1_score(gts, preds, average="weighted")

        cm = metrics.confusion_matrix(gts, preds)
        clf_report = metrics.classification_report(gts, preds, target_names=Artery.MAIN_BRANCH_CATEGORY, output_dict=True)
        np.save(os.path.join(exp_path, f"confusion_matrix_{epoch}.npy"), cm)
        with open(os.path.join(exp_path, f'clf_report_{epoch}.json'), 'w') as fp:
            json.dump(clf_report, fp, indent=4)
        print(f"test with removing @ epoch {epoch}, acc = {acc}, precision = {precision}, recall = {recall}, f1 = {f1_score}")
        print(cm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # exp
    parser.add_argument('--exp', type=str, default="exp")

    # data
    parser.add_argument('--data_file_path', type=str, default=Artery.DATA_FILE_PATH)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--cv_max', type=int, default=5)
    
    # model
    parser.add_argument('--model', type=str, default="dtu", choices=["bi", "utd", "dtu"])
    parser.add_argument('--in_features', type=int, default=12)
    parser.add_argument('--mlp_hidden', type=int, default=128)
    parser.add_argument('--lstm_hidden', type=int, default=30)
    parser.add_argument('--num_class', type=int, default=5)
    parser.add_argument('--device', type=str, default="cuda:0")

    # training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=1e-5)
    # parser.add_argument('--alpha', type=float, default=1e-4) # hyper params for L2 regularization
    parser.add_argument('--epoch', type=int, default=201)
    parser.add_argument('--validation_epoch', type=int, default=10)

    parser.add_argument('--train', type=str, default="train", choices=["train", "attack"])
    parser.add_argument('--prob', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    trainer = Trainer(args)
    if args.train == "train":
        trainer.train()
    else:
        trainer.__restore__()
        trainer.test_with_missing(f"attack_{args.prob}", args.prob)