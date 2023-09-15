import json
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import pickle
from torch.utils.data import Dataset


from src.loss_func import *
from src.evaluation_metric import matching_accuracy

from tqdm import tqdm

import artery.Artery as Artery
from data.artery_utils import *

from artery.dataset import build_dataloader, collate_fn
from artery.models.pca_model import PCA_GM
from train_artery_pca import ArteryDatasetPCA


def data_to_cuda(inputs, device="cuda:0"):
    """
    Call cuda() on all tensor elements in inputs
    :param inputs: input list/dictionary
    :return: identical to inputs while all its elements are on cuda
    """
    device = torch.device(device)
    if type(inputs) is list:
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is tuple:
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is dict:
        for key in inputs:
            inputs[key] = data_to_cuda(inputs[key])
    elif type(inputs) in [str, int, float, nx.Graph, np.str_]:
        inputs = inputs
    elif type(inputs) in [torch.Tensor]:
        inputs = inputs.to(device)
    else:
        raise TypeError('Unknown type of inputs: {}'.format(type(inputs)))
    return inputs


class ArteryDatasetPCASSL(ArteryDatasetPCA):
    def __init__(self, dataset, samples, rand):
        self.dataset = dataset
        self.samples = samples
        self.rand = rand

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        category_id = self.rand.randint(0, len(Artery.ARTERY_CATEGORY))
        all_sample_list = list(self.dataset.keys())
        sample_list = []
        for sample_name in all_sample_list:
            if sample_name.rfind(Artery.ARTERY_CATEGORY[category_id]) != -1:
                sample_list.append(sample_name)

        sample_idx = self.rand.randint(0, len(sample_list), size=2)
        sample_idx = self.__switch__(sample_idx, sample_list)
        
        ###################################################################
        g0 = self.dataset[sample_list[sample_idx[0]]]['g']
        n0 = g0.number_of_nodes()
        g1 = self.augmentation(g0)

        perm_mat = np.zeros((n0, n0))
        for i in range(n0):
            for j in range(n0):
                if g0.nodes()[i]['data'].vessel_class == g0.nodes()[j]['data'].vessel_class:
                    perm_mat[i, j] = 1.0

        A0 = self.__build_graphs__(g0)
        A1 = self.__build_graphs__(g0)

        ret_dict = {'ns': [torch.tensor(x) for x in [n0, n0]],
                    'gt_perm_mat': torch.tensor(np.array(perm_mat, dtype=np.float32)),
                    'As': [torch.tensor(x) for x in [A0, A1]],
                    'id_list': [sample_list[sample_idx[0]], sample_list[sample_idx[1]]],}

        feat0 = np.stack([np.array(g0.nodes()[i]['data'].features, dtype=np.float32) for i in range(n0)], axis=-1).T
        feat1 = np.stack([np.array(g0.nodes()[i]['data'].features, dtype=np.float32) for i in range(n1)], axis=-1).T
        ret_dict['pos_features'] = [torch.tensor(x) for x in [feat0, feat1]]

        # if self.cache:
        #     pickle.dump(ret_dict, open(f"{self.cache_path}/{sample_list[sample_idx[0]]}_{sample_list[sample_idx[1]]}.pkl", "wb"))

        return ret_dict
    

class PCA_SSL_Trainer(object):
    def __init__(self, params, device):
        self.params = params
        self.rand = np.random.RandomState(seed=params.seed)
        self.device = device
        self.__init_dataset__()
        self.__init_model__()

        if not os.path.isdir(self.params.exp):
            os.makedirs(self.params.exp)
        with open(os.path.join(self.params.exp, 'config.json'), 'w') as fp:
            json.dump(self.params.__dict__, fp, indent=4)
        if not os.path.isdir(f"{self.params.exp}/saved_models"):
            os.makedirs(f"{self.params.exp}/saved_models")

    def __select_templates__(self):
        df_view_angles = pd.read_csv(os.path.join(self.params.data_path, "view_angles.csv"))
        train, valid = scsplit(df_view_angles, stratify=df_view_angles['second'],
                               test_size=self.params.template_ratio,
                               train_size=1 - self.params.template_ratio,
                               random_state=self.params.seed)
        
        training_samples = train['id'].values
        template_samples = valid['id'].values
        return training_samples, template_samples

    def __init_dataset__(self):
        dataset, sample_list = Artery._load_graph_in_mem(self.params.data_path, "")
        # split dataset
        training_samples, template_samples = self.__select_templates__()
        training_samples, test_samples = get_split_deterministic(training_samples, self.params.cv, self.params.cv_max)
        print(f"training samples {len(training_samples)}, test samples {len(test_samples)}, template_samples {len(template_samples)}")
        self.sample_train = training_samples
        self.sample_test = test_samples
        self.sample_template = template_samples
        self.sample_train = np.concatenate([self.sample_train, self.sample_template])

        dataset_train, dataset_template, dataset_test = {}, {}, {}
        for k in self.sample_train:
            dataset_train[k] = dataset[k]
        for k in self.sample_template:
            dataset_template[k] = dataset[k]
        for k in self.sample_test:
            dataset_test[k] = dataset[k]
        
        if self.params.cache:
            print("PCA_Model_Trainer.__init_dataset__, set cache dir")
            if not os.path.isdir(self.params.cache_path):
                os.makedirs(self.params.cache_path)

        self.dataset = {"train": dataset_train, "test": dataset_test, "template": dataset_template, "all": dataset}
        dataloader_train = ArteryDatasetPCASSL(dataset_train, self.sample_train, self.rand, cache=self.params.cache, cache_path=self.params.cache_path)
        self.dataloaders = {}
        self.dataloaders['train'] = build_dataloader(dataloader_train, self.params.batch_size, self.params.n_workers, fix_seed=True, shuffle=True)

    def __init_model__(self):

        pca_params = {"FEATURE_CHANNEL": self.params.feature_channel, 
                      "SK_ITER_NUM": self.params.sk_iter_num, "SK_EPSILON": self.params.sk_epsilon, "SK_TAU": self.params.sk_tau,
                      "CROSS_ITER": self.params.cross_iter, "CROSS_ITER_NUM": self.params.cross_iter_num}

        gnn_params = {"GNN_FEAT": [self.params.gnn_feat]*self.params.gnn_layers, "GNN_LAYER": self.params.gnn_layers}

        node_emb_params = {}

        train_params = {"MOMENTUM": self.params.momentum, "OPTIMIZER": self.params.optimizer, "EPOCH_ITERS": self.params.n_iters,
                        "LR_DECAY": 0.1, "LR_STEP": [2, 6, 10], 'LR': self.params.lr, "LOSS_FUNC": "ce"}
        self.pca_params = pca_params
        self.gnn_params = gnn_params
        self.train_params = train_params
        
        self.model = PCA_GM(pca_params, gnn_params, node_emb_params).to(self.device)

        if train_params['OPTIMIZER'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=train_params['LR'], 
                                  momentum=train_params['MOMENTUM'], nesterov=True)
        elif train_params['OPTIMIZER'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=train_params['LR'])
        
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                   milestones=train_params['LR_STEP'], 
                                                   gamma=train_params['LR_DECAY'], 
                                                   last_epoch=-1)

        self.optimizer = optimizer
        self.scheduler = scheduler

        if train_params["LOSS_FUNC"].lower() == 'offset':
            # criterion = OffsetLoss(norm=cfg.TRAIN.RLOSS_NORM) # TODO
            pass
        elif train_params["LOSS_FUNC"].lower() == 'perm':
            criterion = PermutationLoss()
        elif train_params["LOSS_FUNC"].lower() == 'ce':
            criterion = CrossEntropyLoss()
        elif train_params["LOSS_FUNC"].lower() == 'focal':
            criterion = FocalLoss(alpha=.5, gamma=0.)
        elif train_params["LOSS_FUNC"].lower() == 'hung':
            criterion = PermutationLossHung()
        elif train_params["LOSS_FUNC"].lower() == 'hamming':
            criterion = HammingLoss()

        self.criterion = criterion

    def __restore__(self):
        print("PCA_Model_Trainer.__restore__")
        self.model.load_state_dict(torch.load(f"{self.params.exp}/saved_models/model.pt"))

    def train(self, start=0):
        print("[x] PCA_Model_Trainer.train")

        self.max_accuracy = 0.0
        pbar = tqdm(range(start, self.train_params['EPOCH_ITERS']))

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            running_loss = 0.0

            for inputs in self.dataloaders['train']:
                inputs = data_to_cuda(inputs)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # forward
                    outputs = self.model(inputs)
                    if 'ds_mat' in outputs:
                        # compute loss
                        ds_mat = torch.nan_to_num(outputs['ds_mat'], 0.)
                        ds_mat = torch.where(torch.isinf(ds_mat), torch.tensor(0.0).cuda(), ds_mat)
                        loss = self.criterion(ds_mat, outputs['gt_perm_mat'], *outputs['ns'])
                        # compute accuracy
                        acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])
                        pbar.set_description(f"epoch @ {epoch}, acc = {acc.mean().cpu().numpy():.4f}, loss = {loss.item():.4f}")

                        # backward + optimize
                        loss.backward()
                        self.optimizer.step()

                        batch_num = inputs['batch_size']

                        # statistics
                        running_loss += loss.item() * batch_num
                        epoch_loss += loss.item() * batch_num
                    else:
                        # print("[!] nan value encountered")
                        return
            
            if epoch % self.params.n_eval == 0:
                acc = self.test(epoch)
                print(f"[x] epoch @ {epoch}, acc = {acc}")
                if acc > self.max_accuracy:
                    self.max_accuracy = acc
                    torch.save(self.model.state_dict(), f"{self.params.exp}/saved_models/model.pt")


    def load_data_test(self):
        start_time = time.time()
        count = 0
        for i in tqdm(range(100)):
            for inputs in self.dataloaders['train']:
                count += 1
        print(f"load data test, load training data for {count} batches: {time.time() - start_time} s")


    def test(self, iteration):
        self.model.eval()
        save_path = os.path.join(self.params.exp, "%06d" % iteration)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        df = pd.DataFrame(columns=["test_sample", "template_sample", "category", "n", "matched", "unmatched"] + Artery.SUB_BRANCH_CATEGORY)

        accs = []
        for i in tqdm(range(len(self.sample_test))):
            for j in range(len(self.sample_template)):
                g0_category = Artery.get_category(self.sample_test[i])
                g1_category = Artery.get_category(self.sample_template[j])
                g0 = self.dataset["test"][self.sample_test[i]]['g']
                g1 = self.dataset["template"][self.sample_template[j]]['g']
                n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()

                if g0_category == g1_category and n0 <= n1:
                    inputs = ArteryDatasetPCASSL.generate_pair(g0, g1)
                    inputs = data_to_cuda(inputs)
                    inputs = collate_fn([inputs])
                    outputs = self.model(inputs)
                    acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])
                    output_perm_mat = outputs['perm_mat'].detach().cpu().numpy()[0]
                    gt_perm_mat = outputs['gt_perm_mat'].detach().cpu().numpy()[0]
                    mappings = {}
                    for k in range(output_perm_mat.shape[0]):
                        g1_idx = np.where(outputs['perm_mat'].detach().cpu().numpy()[0][k]==1)[0][0]
                        mappings[g0.nodes[k]['data'].vessel_class] = g1.nodes[g1_idx]['data'].vessel_class
                    accs.append(acc.detach().cpu().numpy()[0])
                    data_row = {"test_sample": self.sample_test[i],
                                "template_sample": self.sample_template[j],
                                "category": g0_category, "n": len(mappings.keys())}

                    matched, unmatched = 0, 0
                    for key in mappings:
                        data_row[key] = mappings[key]
                        if key == mappings[key]:
                            matched += 1
                        else:
                            unmatched += 1
                    data_row["matched"] = matched
                    data_row["unmatched"] = unmatched
                    df = df.append(data_row, ignore_index=True)

                    if self.params.plot:
                        plot_match(self.dataset['all'], [self.sample_test[i], self.sample_template[j]], output_perm_mat, gt_perm_mat, 
                                   Artery.SEMANTIC_MAPPING, save_path)
        
        acc = self.__evaluate_pandas_dataframe__(df, f"{iteration:06d}")
        return acc

    def __evaluate_pandas_dataframe__(self, df, save_path):
        # evaluate each pair for arterial branches
        df.to_csv(f"{self.params.exp}/{save_path}/matching_results_raw.csv")
        # evaluate matching results for each test sample for sub coronary artery branches
        df_post_voting = post_processing_voting(df, self.dataset['all'])
        acc = df_post_voting['matched'].sum()/df_post_voting['n'].sum()
        df_post_voting.to_csv(f"{self.params.exp}/{save_path}/matching_results_post.csv")

        print("[x] test @ {}, N {}, MATCHED {}, ACC {:.4f}".format(save_path, df_post_voting['n'].sum(), df_post_voting['matched'].sum(), acc))

        # evaluate matching results for each test sample for main coronary artery branches
        df = evaluate_main_branches(df_post_voting, self.dataset['all'], print_result=True)
        df.to_csv(f"{self.params.exp}/{save_path}/matching_results_main_branch.csv")

        cm, clf_report, acc, precision, recall, f1_score = evaluate_main_branches_sklearn(df_post_voting, self.dataset['all'])
        print("[x] test @ {}, ACC {:.4f}, PRECISION {:.4f}, RECALL {:.4f}, F1 {:.4f}".
                format(save_path, acc, precision, recall, f1_score))

        np.save(f"{self.params.exp}/{save_path}/confusion_matrix_{save_path}.npy", cm)
        with open(f"{self.params.exp}/{save_path}/clf_report_{save_path}.json", 'w') as fp:
            json.dump(clf_report, fp, indent=4)
        return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('--feature_channel', type=int, default=121)
    parser.add_argument('--sk_iter_num', type=int, default=10)
    parser.add_argument('--sk_epsilon', type=float, default=1e-10)
    parser.add_argument('--sk_tau', type=float, default=0.05)
    parser.add_argument('--cross_iter', type=bool, default=True)
    parser.add_argument('--cross_iter_num', type=int, default=3)
    parser.add_argument('--gnn_feat', type=int, default=64)
    parser.add_argument('--gnn_layers', type=int, default=3)

    # data parameters
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--data_path', type=str, default='/media/z/data21/artery_semantic_segmentation/ThinkMatch/artery/data_augment')
    parser.add_argument('--cache', type=bool, default=True)
    parser.add_argument('--cache_path', type=str, default='/media/z/data21/artery_semantic_segmentation/ThinkMatch/cache/pca')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--cv_max', type=int, default=5)
    parser.add_argument('--template_ratio', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=1)

    # training parameters
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--n_iters', type=int, default=10001)
    parser.add_argument('--n_eval', type=int, default=200)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1.0e-5)
    parser.add_argument('--loss_func', type=str, default='ce')
    parser.add_argument('--plot', type=str, default=False)
    parser.add_argument('--gpu', type=int, default=0)

    # exp
    parser.add_argument('--exp', type=str, default="exp/pca/CV0")
    parser.add_argument('--prob', type=float, default=0.05)

    # flow control
    parser.add_argument('--flow', type=str, default="train")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.flow == "train":
        trainer = PCA_Model_Trainer(args, device)

        if os.path.isdir(f"{args.exp}"):
            if len(os.listdir(f"{args.exp}")) > 1:
                if len([int(x) for x in os.listdir(f"{args.exp}") if x.startswith("0")]) == 0:
                    last_epoch = 0
                else:
                    last_epoch = sorted([int(x) for x in os.listdir(f"{args.exp}") if x.startswith("0")])[-1] + 1
                    trainer.__restore__()
            else:
                last_epoch = 0
        else:
            last_epoch = 0
        
        trainer.train(start=last_epoch)
        trainer.__restore__()
        trainer.test(999999)

    elif args.flow == "test":
        trainer = PCA_Model_Trainer(args, device)
        trainer.__restore__()
        trainer.test(999999)
    elif args.flow == "load_data_test":
        trainer = PCA_Model_Trainer(args, device)
        trainer.load_data_test()
    elif args.flow == "attack":
        trainer = PCA_Model_Trainer(args, device)
        trainer.__restore__()
        trainer.test_random_missing(f"attack_{args.prob}")