import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse
import copy


import Artery

from tqdm import tqdm
from model.conv_lstm import ConvBLSTM
from model.gcn import Gconv
from data_utils import convert_tree_to_tensors, convert_graph_to_im_tensors
from sklearn import metrics

from flopth import flopth

class GPR_GCN(nn.Module):

    def __init__(self, pos_feat_dim, cnn_feat_dim, mlp_hidden, lstm_hidden, out_features, gcn_features):
        super(GPR_GCN, self).__init__()
        self.pos_encoder = nn.Sequential(nn.Linear(pos_feat_dim, mlp_hidden), nn.ReLU(),
                                         nn.Linear(mlp_hidden, mlp_hidden//2), nn.ReLU(),
                                         nn.Linear(mlp_hidden//2, mlp_hidden//4), nn.ReLU(),
                                         nn.Linear(mlp_hidden//4, mlp_hidden//8), nn.ReLU())
        self.pix_encoder = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=1, out_channels=cnn_feat_dim, padding='same'),
                                         nn.MaxPool2d(kernel_size=2),
                                         nn.Conv2d(kernel_size=3, in_channels=cnn_feat_dim, out_channels=cnn_feat_dim*2, padding='same'),
                                         nn.MaxPool2d(kernel_size=2),
                                         nn.Conv2d(kernel_size=3, in_channels=cnn_feat_dim*2, out_channels=cnn_feat_dim*4, padding='same'),
                                         nn.MaxPool2d(kernel_size=2))

        self.blstm = ConvBLSTM(in_channels=cnn_feat_dim*4, hidden_channels=lstm_hidden, kernel_size=[(3, 3)]*4, num_layers=4)
        self.gcn1 = Gconv(in_features=mlp_hidden//8+lstm_hidden, out_features=gcn_features)
        self.gcn1_weighted = nn.Sequential(nn.Linear(gcn_features, mlp_hidden//8), nn.ReLU())
        self.gcn2 = Gconv(in_features=mlp_hidden//8+lstm_hidden, out_features=gcn_features)
        self.gcn2_weighted = nn.Sequential(nn.Linear(gcn_features, mlp_hidden//8), nn.ReLU())
        self.classifier = nn.Linear(mlp_hidden//8, out_features)

    def forward(self, pos_features, im_patches, splitter, n_node, A):
        pos_features = self.pos_encoder(pos_features) # pos_feature: [n_node, mlp_hidden//8]
        im_features = self.pix_encoder(im_patches)
        # im_features_vec = torch.flatten(im_features, start_dim=1)
        conditions = []
        for i in range(n_node): # torch.flip(node_im_features, dims=[0])[2]
            node_im_features = im_features[splitter[i][0]:splitter[i][1]] 
            node_im_features = torch.unsqueeze(node_im_features, dim=1) # convert (b,c,h,w) to (t, b, c, h, w), batch to time steps
            node_im_features_reverse =  torch.flip(node_im_features, dims=[0])
            node_im_features = self.blstm(node_im_features, node_im_features_reverse)
            node_im_features = node_im_features[:, -1, :, :, :]
            node_im_features_flatten = torch.flatten(node_im_features)
            conditions.append(node_im_features_flatten)
        
        conditions = torch.stack(conditions)

        # GCN 1
        fused_features = torch.cat((pos_features, conditions), 1)
        features = torch.squeeze(self.gcn1(A, torch.unsqueeze(fused_features, dim=0)))
        features = self.gcn1_weighted(features)
        pos_features = features + pos_features # n_node* mlp//16

        # GCN 2
        fused_features = torch.cat((pos_features, conditions), 1)
        features = torch.squeeze(self.gcn2(A, torch.unsqueeze(fused_features, dim=0)))
        features = self.gcn2_weighted(features)
        pos_features = features + pos_features # n_node* mlp//16

        logits = self.classifier(pos_features)
        return logits

class Trainer():

    def __init__(self, params):
        self.params = params
        dataset, training_samples = Artery._load_graph_in_mem(self.params.data_file_path, "")
        training_samples, test_samples = Artery.get_split_deterministic(training_samples, self.params.cv, self.params.cv_max)
        self.dataset, self.train_samples, self.test_samples = dataset, training_samples, test_samples
        self.__init_model__()
        self.rand = np.random.RandomState(seed=params.seed)

    def __init_model__(self):
        pos_features = self.params.pos_features
        cnn_features = self.params.cnn_features
        mlp_hidden = self.params.mlp_hidden
        lstm_hidden = self.params.lstm_hidden
        out_features = self.params.num_class
        gcn_features = self.params.gcn_features

        self.model = GPR_GCN(pos_features, cnn_features, mlp_hidden, lstm_hidden, out_features, gcn_features)
        self.model.to(self.params.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.decay) 
        self.optimizer = optimizer

        self.loss_function = torch.nn.BCEWithLogitsLoss()

    def train(self):
        best_acc = 0.
        exp_path = os.path.join(self.params.exp)
        if not os.path.isdir(exp_path):
            os.makedirs(exp_path)
        with open(os.path.join(exp_path, 'config.json'), 'w') as fp:
            json.dump(self.params.__dict__, fp, indent=4)

        target = open(os.path.join(exp_path, "eval.csv"), "w")
        target.write("epoch,acc,precision,recall,f1\n")

        for epoch in tqdm(range(self.params.epoch)):
            print(f"training @ epoch {epoch}")
            self.model.train()
            FLOPs = []
            for mini_batch in tqdm(range(len(self.train_samples))):
                tree = self.dataset[self.train_samples[mini_batch]]['tree']
                g = self.dataset[self.train_samples[mini_batch]]['g']
                images, splitter = convert_graph_to_im_tensors(g, self.params.device)
                data = convert_tree_to_tensors(tree, self.params.device)
                pos_features, labels = data['features'], data['labels']
                A = torch.from_numpy(np.array(nx.adjacency_matrix(g).todense(), dtype=np.float32)).to(self.params.device)
                A = torch.unsqueeze(A, dim=0)
                n_node = len(g.nodes())

                logits = self.model(pos_features, images, splitter, n_node, A)

                flops, params = flopth(self.model, inputs=(pos_features, images, 
                                                           torch.from_numpy(np.array(splitter)), 
                                                           torch.from_numpy(np.array(n_node)), A))
                print(flops)
                FLOPs.append(float(flops[:-1])*1000)

                loss = self.loss_function(logits, labels)
                loss.backward()
                self.optimizer.step()
            print(np.mean(FLOPs))

            if epoch % self.params.validation_epoch == 0:
                print(f"test @ epoch {epoch}")
                self.model.eval()
                preds = []
                gts = []
                for mini_batch in tqdm(range(len(self.test_samples))):
                    tree = self.dataset[self.test_samples[mini_batch]]['tree']
                    g = self.dataset[self.test_samples[mini_batch]]['g']
                    images, splitter = convert_graph_to_im_tensors(g, self.params.device)
                    data = convert_tree_to_tensors(tree, self.params.device)
                    pos_features, labels = data['features'], data['labels']
                    A = torch.from_numpy(np.array(nx.adjacency_matrix(g).todense(), dtype=np.float32)).to(self.params.device)
                    A = torch.unsqueeze(A, dim=0)
                    n_node = len(g.nodes())

                    logits = self.model(pos_features, images, splitter, n_node, A)
                    
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

    def __restore__(self):
        exp_path = os.path.join(self.params.exp)
        self.model.load_state_dict(torch.load(os.path.join(exp_path, "model.pth")))
        self.model.eval()
    
    def __trim_graph__(self, g, prob):
        removed_nodes_idx = []
        removed_nodes = []

        g = copy.deepcopy(g)
        for node in g.nodes():
            if g.nodes()[node]['data'].node1.degree == 1 or g.nodes()[node]['data'].node2.degree == 1:
                # removeable 
                if self.rand.rand() < prob: # generate random value according to pre-defined seed
                    removed_nodes.append(g.nodes()[node]['data'].vessel_class)
                    removed_nodes_idx.append(node)
        
        for n in removed_nodes_idx:
            g.remove_node(n)
        # re assign node index
        mapping = {old_label:new_label for new_label, old_label in enumerate(g.nodes())}
        g = nx.relabel_nodes(g, mapping)
        return g, removed_nodes, removed_nodes_idx

    def test_with_removing(self, epoch, prob):
        exp_path = os.path.join(self.params.exp)
        print(f"test attack @ epoch {epoch}")
        self.model.eval()
        preds = []
        gts = []
        for mini_batch in tqdm(range(len(self.test_samples))):
            tree = self.dataset[self.test_samples[mini_batch]]['tree']
            g = self.dataset[self.test_samples[mini_batch]]['g']
            n_nodes_ori = len(g.nodes())
            g, removed_nodes, removed_nodes_idx = self.__trim_graph__(g, prob)
            print(f"trim graph {self.test_samples[mini_batch]}, removed nodes = {removed_nodes}")

            images, splitter = convert_graph_to_im_tensors(g, self.params.device)
            data = convert_tree_to_tensors(tree, self.params.device)
            pos_features, labels = data['features'], data['labels']
            slice = [x for x in range(n_nodes_ori) if x not in removed_nodes_idx]
            pos_features = pos_features[slice]
            labels = labels[slice]

            A = torch.from_numpy(np.array(nx.adjacency_matrix(g).todense(), dtype=np.float32)).to(self.params.device)
            A = torch.unsqueeze(A, dim=0)
            n_node = len(g.nodes())

            logits = self.model(pos_features, images, splitter, n_node, A)
            pred_cls = torch.argmax(logits, dim=1).detach().cpu().numpy()
            gt = torch.argmax(labels, dim=1).detach().cpu().numpy()
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
        print(f"test attack @ epoch {epoch}, acc = {acc}, precision = {precision}, recall = {recall}, f1 = {f1_score}")
        print(cm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # exp
    parser.add_argument('--exp', type=str, default="exp/MLP64_GCN128/CV0")

    # data
    parser.add_argument('--data_file_path', type=str, default=Artery.DATA_FILE_PATH)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--cv_max', type=int, default=5)
    
    # model
    parser.add_argument('--model', type=str, default="cpr_gcn")
    parser.add_argument('--pos_features', type=int, default=8)
    parser.add_argument('--cnn_features', type=int, default=16)
    parser.add_argument('--mlp_hidden', type=int, default=64)
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--num_class', type=int, default=5)
    parser.add_argument('--gcn_features', type=int, default=128)
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
        trainer.test_with_removing(f"attack_{args.prob}", args.prob)