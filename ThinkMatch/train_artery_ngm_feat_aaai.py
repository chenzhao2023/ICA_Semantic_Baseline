import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import numpy as np
import copy

from torch.utils.data import Dataset
from src.loss_func import *
from src.evaluation_metric import matching_accuracy

from tqdm import tqdm

import artery.Artery as Artery
from data.artery_utils import *
from artery.dataset import worker_init_fix, worker_init_rand
from artery.models.ngm_model import GMN_Net, GMN_FEAT_Net
from itertools import product

from src.factorize_graph_matching import kronecker_sparse, kronecker_torch
from src.sparse_torch.csx_matrix import CSRMatrix3d, CSCMatrix3d


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """
    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            #pad_pattern = torch.from_numpy(np.asfortranarray(pad_pattern))
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Keys mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        # elif type(inp[0]) == pyg.data.Data:
        #     ret = pyg.data.Batch.from_data_list(inp)
        elif type(inp[0]) == str:
            ret = inp
        elif type(inp[0]) == tuple:
            ret = inp
        elif type(inp[0]) == nx.Graph:
            ret = inp
        elif type(inp[0]) == np.str_:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    # compute CPU-intensive Kronecker product here to leverage multi-processing nature of dataloader
    if 'Gs' in ret and 'Hs' in ret:
        if '2GM' and len(ret['Gs']) == 2 and len(ret['Hs']) == 2:
            G1, G2 = ret['Gs']
            H1, H2 = ret['Hs']
            sparse_dtype = np.float32
            K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2, G1)]  # 1 as source graph, 2 as target graph
            K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2, H1)]
            K1G = CSRMatrix3d(K1G)
            K1H = CSRMatrix3d(K1H).transpose()

            ret['KGHs'] = K1G, K1H

    ret['batch_size'] = len(data)

    for v in ret.values():
        if type(v) is list:
            ret['num_graphs'] = len(v)
            break

    return ret


def build_dataloader(dataset, batch_size, num_workers, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        pin_memory=False, 
        worker_init_fn=worker_init_fix if fix_seed else worker_init_rand)


def data_to_cuda(inputs):
    """
    Call cuda() on all tensor elements in inputs
    :param inputs: input list/dictionary
    :return: identical to inputs while all its elements are on cuda
    """
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
    elif type(inputs) in [torch.Tensor, CSRMatrix3d, CSCMatrix3d]:
        inputs = inputs.cuda()
    elif type(inputs) in [torch.Tensor]:
        inputs = inputs.cuda()
    else:
        raise TypeError('Unknown type of inputs: {}'.format(type(inputs)))
    return inputs


class ArteryDatasetNGM(Dataset):
    def __init__(self, dataset, samples, rand, cache, cache_path):
        self.dataset = dataset
        self.samples = samples
        self.rand = rand
        self.cache = cache
        self.cache_path = cache_path

    def __len__(self):
        return len(self.samples)

    def __switch__(self, sample_idx, sample_list):
        assert len(sample_idx) == 2
        sample_name0, sample_name1 = sample_list[sample_idx[0]], sample_list[sample_idx[1]]
        g0, g1 = self.dataset[sample_name0]['g'], self.dataset[sample_name1]['g']
        n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()

        if n0 >= n1:
            return [sample_idx[1], sample_idx[0]]
        else:
            return sample_idx

    def __build_graphs__(self, g: nx.Graph):
        A = nx.adjacency_matrix(g).todense()
        edge_num = g.number_of_edges()*2 # in network graph, adjmatrix indicates undirected graph, thus, for directed graph, the number of edges are doubled
        node_num = g.number_of_nodes()
        G = np.zeros((node_num, edge_num), dtype=np.float32)
        H = np.zeros((node_num, edge_num), dtype=np.float32)
        edge_idx = 0
        for i in range(node_num):  # iterative graph adjacency matrix
            for j in range(node_num):
                if A[i, j] == 1:
                    G[i, edge_idx] = 1
                    H[j, edge_idx] = 1
                    edge_idx += 1
        return A, G, H, edge_num

    @staticmethod
    def generate_pair(g0, g1):

        def __build_graphs__(g: nx.Graph):
            A = nx.adjacency_matrix(g).todense()
            edge_num = g.number_of_edges()*2 # in network graph, adjmatrix indicates undirected graph, thus, for directed graph, the number of edges are doubled
            node_num = g.number_of_nodes()
            G = np.zeros((node_num, edge_num), dtype=np.float32)
            H = np.zeros((node_num, edge_num), dtype=np.float32)
            edge_idx = 0
            for i in range(node_num):  # iterative graph adjacency matrix
                for j in range(node_num):
                    if A[i, j] == 1:
                        G[i, edge_idx] = 1
                        H[j, edge_idx] = 1
                        edge_idx += 1
            return A, G, H, edge_num
            
        n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()
        perm_mat = np.zeros((n0, n1))
        g0_vessel_names = [g0.nodes()[i]['data'].vessel_class for i in range(n0)]
        g1_vessel_names = [g1.nodes()[i]['data'].vessel_class for i in range(n1)]
        for i in range(n0):
            for j in range(n1):
                if g0.nodes()[i]['data'].vessel_class == g1.nodes()[j]['data'].vessel_class:
                    perm_mat[i, j] = 1.0

        A0, G0, H0, e0 = __build_graphs__(g0)
        A1, G1, H1, e1 = __build_graphs__(g1)

        ret_dict = {'ns': [torch.tensor(x) for x in [n0, n1]],
                    'es': [torch.tensor(x) for x in [e0, e1]],
                    'gs': [g0, g1],
                    'gt_perm_mat': perm_mat,
                    'Gs': [torch.Tensor(x) for x in [G0, G1]],
                    'Hs': [torch.Tensor(x) for x in [H0, H1]],
                    'As': [torch.Tensor(x) for x in [A0, A1]],
                    'labels': [g0_vessel_names, g1_vessel_names]}

        feat0 = np.stack([np.array(g0.nodes()[i]['data'].features) for i in range(n0)], axis=-1).T
        feat1 = np.stack([np.array(g1.nodes()[i]['data'].features) for i in range(n1)], axis=-1).T
        ret_dict['pos_features'] = [torch.Tensor(x) for x in [feat0, feat1]]

        return ret_dict

    def __getitem__(self, index):
        category_id = self.rand.randint(0, len(Artery.ARTERY_CATEGORY))
        all_sample_list = list(self.dataset.keys())
        sample_list = []
        for sample_name in all_sample_list:
            if sample_name.rfind(Artery.ARTERY_CATEGORY[category_id]) != -1:
                sample_list.append(sample_name)

        sample_idx = self.rand.randint(0, len(sample_list), size=2)
        sample_idx = self.__switch__(sample_idx, sample_list)

        
        if self.cache:
            if os.path.isfile(f"{self.cache_path}/{sample_list[sample_idx[0]]}_{sample_list[sample_idx[1]]}.pkl"):
                ret_dict = pickle.load(open(f"{self.cache_path}/{sample_list[sample_idx[0]]}_{sample_list[sample_idx[1]]}.pkl", "rb"))
                assert ret_dict['ns'][0]<=ret_dict['ns'][1]
                return ret_dict
        
        g0 = self.dataset[sample_list[sample_idx[0]]]['g']
        g1 = self.dataset[sample_list[sample_idx[1]]]['g']

        n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()
        g0_vessel_names = [g0.nodes()[i]['data'].vessel_class for i in range(n0)]
        g1_vessel_names = [g1.nodes()[i]['data'].vessel_class for i in range(n1)]
        assert n0 <= n1
        
        perm_mat = np.zeros((n0, n1))
        for i in range(n0):
            for j in range(n1):
                if g0.nodes()[i]['data'].vessel_class == g1.nodes()[j]['data'].vessel_class:
                    perm_mat[i, j] = 1.0

        A0, G0, H0, e0 = self.__build_graphs__(g0)
        A1, G1, H1, e1 = self.__build_graphs__(g1)

        ret_dict = {'ns': [torch.tensor(x) for x in [n0, n1]],
                    'es': [torch.tensor(x) for x in [e0, e1]],
                    'gs': [g0, g1],
                    'gt_perm_mat': perm_mat,
                    'Gs': [torch.Tensor(x) for x in [G0, G1]],
                    'Hs': [torch.Tensor(x) for x in [H0, H1]],
                    'As': [torch.Tensor(x) for x in [A0, A1]],
                    'cls': Artery.ARTERY_CATEGORY[category_id],
                    'id_list': [sample_list[sample_idx[0]], sample_list[sample_idx[1]]], 
                    'labels': [g0_vessel_names, g1_vessel_names]}

        feat0 = np.stack([np.array(g0.nodes()[i]['data'].features) for i in range(n0)], axis=-1).T
        feat1 = np.stack([np.array(g1.nodes()[i]['data'].features) for i in range(n1)], axis=-1).T
        ret_dict['pos_features'] = [torch.Tensor(x) for x in [feat0, feat1]]

        if self.cache:
            pickle.dump(ret_dict, open(f"{self.cache_path}/{sample_list[sample_idx[0]]}_{sample_list[sample_idx[1]]}.pkl", "wb"))

        return ret_dict


class NGM_Model_Trainer(object):
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
            print("NGM_Model_Trainer.__init_dataset__, set cache dir")
            if not os.path.isdir(self.params.cache_path):
                os.makedirs(self.params.cache_path)

        self.dataset = {"train": dataset_train, "test": dataset_test, "template": dataset_template, "all": dataset}
        dataloader_train = ArteryDatasetNGM(self.dataset["train"], self.sample_train, self.rand, self.params.cache, self.params.cache_path)
        self.dataloaders = {}
        self.dataloaders['train'] = build_dataloader(dataloader_train, self.params.batch_size, self.params.n_workers, fix_seed=True, shuffle=True)

    def __init_model__(self):
        feat_emb_params = {"POS_FEAT_DIM": self.params.feature_channel, 
                           "POS_FEAT_HIDDEN": self.params.feat_hidden_dim, 
                           "EMB_DIM": self.params.feat_emb_dim, 
                           "N_LAYERS": self.params.feat_emb_n_layers}

        ngm_params = {"EDGE_FEATURE": "cat", 
                      "SK_ITER_NUM": self.params.sk_iter_num, 
                      "SK_EPSILON": self.params.sk_epsilon, 
                      "SK_TAU": self.params.sk_tau, 
                      "SK_EMB": 1 if self.params.sk_emb else 0,
                      "FIRST_ORDER": True, 
                      "EDGE_EMB": False,
                      "GAUSSIAN_SIGMA": 1.0, 
                      "GUMBEL_SK": -1}

        gnn_params = {"GNN_FEAT": [self.params.gnn_feat]*self.params.gnn_layers, "GNN_LAYER": self.params.gnn_layers}

        train_params = {"MOMENTUM": self.params.momentum, "OPTIMIZER": self.params.optimizer, "EPOCH_ITERS": self.params.n_iters,
                        "LR_DECAY": 0.1, "LR_STEP": [2, 6, 10], 'LR': self.params.lr, "LOSS_FUNC": self.params.loss_func}
        
        self.feat_emb_params = feat_emb_params
        self.ngm_params = ngm_params
        self.gnn_params = gnn_params
        
        self.model = GMN_FEAT_Net(ngm_params, gnn_params, feat_emb_params).to(self.device)

        if train_params['OPTIMIZER'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=train_params['LR'], 
                                  momentum=train_params['MOMENTUM'], nesterov=True)
        elif train_params['OPTIMIZER'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=train_params['LR'])
        
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params['LR_STEP'], 
                                                   gamma=train_params['LR_DECAY'], last_epoch=-1)

        self.optimizer = optimizer
        self.scheduler = scheduler

        if train_params["LOSS_FUNC"].lower() == 'perm':
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

    def train(self, start):
        self.max_accuracy = 0.
        #for epoch in tqdm(range(self.train_params['EPOCH_ITERS'])):
        pbar = tqdm(range(start, self.params.n_iters))
        for epoch in pbar:
            self.model.train()
            
            for inputs in self.dataloaders['train']:
                inputs = data_to_cuda(inputs)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    # forward
                    outputs = self.model(inputs)
                    assert 'ds_mat' in outputs
                    assert 'perm_mat' in outputs
                    assert 'gt_perm_mat' in outputs
                    # compute loss
                    loss = self.criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
                    # compute accuracy
                    acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])
                    pbar.set_description(f"epoch @ {epoch}, acc = {acc.mean().cpu().numpy():.4f}, loss = {loss.item():.4f}")
                    # backward + optimize
                    loss.backward()
                    self.optimizer.step()

            # print(f"[x] training @ epoch {epoch}, acc = {np.mean(accs)}")
            if epoch % self.params.n_eval == 0:
                acc = self.test(epoch)
                print(f"[x] testing epoch @ {epoch}, acc = {acc}")
                if acc > self.max_accuracy:
                    self.max_accuracy = acc
                    torch.save(self.model.state_dict(), f"{self.params.exp}/saved_models/model.pt")

    def test(self, iteration):
        self.model.eval()
        save_path = os.path.join(self.params.exp, "%06d" % iteration)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        df = pd.DataFrame(columns=["test_sample", "template_sample", "category", "n", "matched", "unmatched"] + Artery.SUB_BRANCH_CATEGORY)

        for i in tqdm(range(len(self.sample_test))):
            for j in range(len(self.sample_template)):
                g0_category = Artery.get_category(self.sample_test[i])
                g1_category = Artery.get_category(self.sample_template[j])
                g0 = self.dataset["test"][self.sample_test[i]]['g']
                g1 = self.dataset["template"][self.sample_template[j]]['g']
                n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()

                if g0_category == g1_category and n0 <= n1:
                    inputs = ArteryDatasetNGM.generate_pair(g0, g1)
                    inputs = collate_fn([inputs])
                    inputs = data_to_cuda(inputs)
                    outputs = self.model(inputs)
                    acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])

                    output_perm_mat = outputs['perm_mat'].detach().cpu().numpy()[0]
                    gt_perm_mat = outputs['gt_perm_mat'].detach().cpu().numpy()[0]
                    mappings = {}
                    for k in range(output_perm_mat.shape[0]):
                        g1_idx = np.where(outputs['perm_mat'].detach().cpu().numpy()[0][k]==1)[0][0]
                        mappings[g0.nodes[k]['data'].vessel_class] = g1.nodes[g1_idx]['data'].vessel_class
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
                        plot_match(self.dataset['all'], [self.sample_test[i], self.sample_template[j]], 
                                   output_perm_mat, gt_perm_mat, Artery.SEMANTIC_MAPPING, save_path)

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

    def test_random_missing(self, save_path):
        if not os.path.isdir(f"{self.params.exp}/{save_path}"):
            os.makedirs(f"{self.params.exp}/{save_path}")
        df = pd.DataFrame(columns=["test_sample", "template_sample", "category", "n", "matched", "unmatched"] + Artery.SUB_BRANCH_CATEGORY)
        for i in tqdm(range(len(self.sample_test))):
            g0 = self.dataset['all'][self.sample_test[i]]['g']
            g0_category = Artery.get_category(self.sample_test[i])
            # trim graph
            g0, removed_nodes = Artery.__trim_graph__(g0, self.params.prob, self.rand)
            self.dataset['all'][self.sample_test[i]]['g'] = g0
            print(f"trim graph {self.sample_test[i]}, removed nodes = {removed_nodes}")
            
            for j in range(len(self.sample_template)):
                g1_category = Artery.get_category(self.sample_template[j])
                g1 = self.dataset['all'][self.sample_template[j]]['g']
                n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()

                if g0_category == g1_category and n0 <= n1:

                    inputs = ArteryDatasetNGM.generate_pair(g0, g1)
                    inputs = collate_fn([inputs])
                    inputs = data_to_cuda(inputs)
                    outputs = self.model(inputs)
                    acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])

                    output_perm_mat = outputs['perm_mat'].detach().cpu().numpy()[0]
                    gt_perm_mat = outputs['gt_perm_mat'].detach().cpu().numpy()[0]
                    mappings = {}
                    for k in range(output_perm_mat.shape[0]):
                        g1_idx = np.where(outputs['perm_mat'].detach().cpu().numpy()[0][k]==1)[0][0]
                        mappings[g0.nodes[k]['data'].vessel_class] = g1.nodes[g1_idx]['data'].vessel_class
                    data_row = {"test_sample": self.sample_test[i], 
                                "template_sample": self.sample_template[j], 
                                "category": g0_category, 
                                "n": len(mappings.keys())}

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
                        plot_match(self.dataset['all'], [self.sample_test[i], self.sample_template[j]], 
                                   output_perm_mat, gt_perm_mat, Artery.SEMANTIC_MAPPING, save_path)
        
        acc = self.__evaluate_pandas_dataframe__(df, save_path)
        return acc
    
    def __restore__(self):
        print("NGM_Model_Trainer.__restore__")
        self.model.load_state_dict(torch.load(f"{self.params.exp}/saved_models/model.pt"))

    def calculate_complexity(self):
        # FLOPs = []
        # for inputs in self.dataloaders['train']:
        #     inputs = data_to_cuda(inputs)
        #     del inputs['id_list']
        #     flops, params = flopth(self.model, inputs=inputs)
        #     print(flops)
        #     FLOPs.append(float(flops[:-1])*1000)

        # print(np.mean(FLOPs))

        from thop import profile
        for inputs in self.dataloaders['train']:
            inputs = data_to_cuda(inputs)
            flops, params = profile(self.model, inputs=(inputs,))
            break
        print(flops)
        print(params)

        total_params = sum(param.numel() for param in self.model.parameters())
        print(total_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('--feature_channel', type=int, default=121)
    parser.add_argument('--sk_iter_num', type=int, default=10)
    parser.add_argument('--sk_epsilon', type=float, default=1e-10)
    parser.add_argument('--sk_tau', type=float, default=0.05)
    parser.add_argument('--sk_emb', type=bool, default=True)

    parser.add_argument('--gnn_feat', type=int, default=64)
    parser.add_argument('--gnn_layers', type=int, default=2)

    parser.add_argument('--feat_emb_dim', type=int, default=64)
    parser.add_argument('--feat_hidden_dim', type=int, default=64)
    parser.add_argument('--feat_emb_n_layers', type=int, default=1)

    # data parameters
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--data_path', type=str,  default='/media/z/data2/artery_semantic_segmentation/hnn_hm_june/data/artery_with_feature')
    parser.add_argument('--cache', type=bool, default=False)
    parser.add_argument('--cache_path', type=str, default='.')
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--cv_max', type=int, default=5)
    parser.add_argument('--template_ratio', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=1)

    # training parameters
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--n_iters', type=int, default=3001)
    parser.add_argument('--n_eval', type=int, default=50)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1.0e-3)
    parser.add_argument('--loss_func', type=str, default='ce')
    parser.add_argument('--plot', type=str, default=False)
    parser.add_argument('--gpu', type=int, default=0)

    # exp
    parser.add_argument('--exp', type=str, default="exp_aaai/ngm/CV0")

    # flow control
    parser.add_argument('--flow', type=str, default="train")
    parser.add_argument('--prob', type=float, default=0.05)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.flow == "train":
        trainer = NGM_Model_Trainer(args, device)
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
        trainer.train(last_epoch)
        trainer.__restore__()
        trainer.test(999999)
    elif args.flow == "test":
        trainer = NGM_Model_Trainer(args, device)
        trainer.__restore__()
        trainer.test(999999)
    elif args.flow == "attack":
        trainer = NGM_Model_Trainer(args, device)
        trainer.__restore__()
        trainer.test_random_missing(f"attack_{args.prob}")
    elif args.flow == "complexity":
        trainer = NGM_Model_Trainer(args, device)
        trainer.calculate_complexity()
