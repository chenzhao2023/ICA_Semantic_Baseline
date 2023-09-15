import math
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np

from torch.utils.data import Dataset
from src.loss_func import *
from src.evaluation_metric import matching_accuracy

from tqdm import tqdm

import artery.Artery as Artery
from data.artery_utils import *
from artery.dataset import worker_init_fix, worker_init_rand
from artery.models.ngm_model import GMN_Net
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
        worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )


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
    def __init__(self, dataset, samples, rand):
        self.dataset = dataset
        self.samples = samples
        self.rand = rand

    def __len__(self):
        return len(self.samples)

    def __switch__(self, sample_idx):
        assert len(sample_idx) == 2
        sample_name0, sample_name1 = self.samples[sample_idx[0]], self.samples[sample_idx[1]]
        _, _, g0 = self.dataset[sample_name0]['image'], self.dataset[sample_name0]['binary_image'], self.dataset[sample_name0]['g']
        _, _, g1 = self.dataset[sample_name1]['image'], self.dataset[sample_name1]['binary_image'], self.dataset[sample_name1]['g']

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

    def _gather_patches(self, patches):
        r = []
        for patchset in patches:
            for i in range(patchset.shape[0]):
                r.append(patchset[i])
        return np.array(r, dtype=np.float32)

    def convert_graph_to_im_tensors(self, g):
        patches = []
        splitter = []
        start = 0
        for node in g.nodes():
            node_patch = g.nodes()[node]['data'].patches
            splitter.append((start, start+len(node_patch)))
            start = start+len(node_patch)
            patches.append(np.array(node_patch))
        patches = self._gather_patches(patches)
        patches = np.expand_dims(patches, 1)
        # patches = torch.from_numpy(patches).to(device)
        return patches, splitter

    def __getitem__(self, index):
        category_id = self.rand.randint(0, len(Artery.ARTERY_CATEGORY))
        all_sample_list = list(self.dataset.keys())
        sample_list = []
        for sample_name in all_sample_list:
            if sample_name.rfind(Artery.ARTERY_CATEGORY[category_id]) != -1:
                sample_list.append(sample_name)

        sample_idx = self.rand.randint(0, len(sample_list), size=2)
        sample_idx = self.__switch__(sample_idx)

        image0, binary_image0, g0 = \
            self.dataset[sample_list[sample_idx[0]]]['image'], \
            self.dataset[sample_list[sample_idx[0]]]['binary_image'], \
            self.dataset[sample_list[sample_idx[0]]]['g']
        image1, binary_image1, g1 = \
            self.dataset[sample_list[sample_idx[1]]]['image'], \
            self.dataset[sample_list[sample_idx[1]]]['binary_image'], \
            self.dataset[sample_list[sample_idx[1]]]['g']

        patches0, splitter0 = self.convert_graph_to_im_tensors(g0)
        patches1, splitter1 = self.convert_graph_to_im_tensors(g1)

        n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()
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
                    'ims': [image0, image1],
                    'im_bins': [binary_image0, binary_image1],
                    'gt_perm_mat': perm_mat,
                    'Gs': [torch.Tensor(x) for x in [G0, G1]],
                    'Hs': [torch.Tensor(x) for x in [H0, H1]],
                    'As': [torch.Tensor(x) for x in [A0, A1]],
                    'cls': Artery.ARTERY_CATEGORY[category_id],
                    'id_list': [sample_list[sample_idx[0]], sample_list[sample_idx[1]]],
                    'patches': [torch.Tensor(x) for x in [patches0, patches1]],
                    'splitter': [torch.Tensor(x) for x in [splitter0, splitter1]]}

        feat0 = np.stack([np.array(g0.nodes()[i]['data'].features) for i in range(n0)], axis=-1).T
        feat1 = np.stack([np.array(g1.nodes()[i]['data'].features) for i in range(n1)], axis=-1).T
        ret_dict['pos_features'] = [torch.Tensor(x) for x in [feat0, feat1]]

        return ret_dict


class Model_Trainer(object):
    def __init__(self, params, device):
        self.params = params
        self.rand = np.random.RandomState(seed=params.seed)
        self.device = device
        self.__init_dataset__()
        self.__init_model__()

    def __init_dataset__(self):
        data_parmas = {"DATA_FILE_PATH": self.params.data_path, "CV": 0, "CV_MAX": 5, "TEMP_RATIO": 0.1,
                       "BATCH_SIZE": 1, "NUM_WORKERS": 1}

        dataset, sample_list = Artery._load_graph_in_mem(data_parmas['DATA_FILE_PATH'], "")
        # split dataset
        training_samples, template_samples = get_split_deterministic(sample_list, 0, int(1/data_parmas['TEMP_RATIO']), self.params.seed)
        training_samples, test_samples = get_split_deterministic(training_samples, data_parmas['CV'], data_parmas['CV_MAX'])
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
        
        self.dataset = {"train": dataset_train, "test": dataset_test, "template": dataset_template}
        dataloader_train = ArteryDatasetNGM(self.dataset["train"], self.sample_train, self.rand)
        self.dataloaders = {}
        # dataset, batch_size, num_workers, fix_seed=True, shuffle=False
        self.dataloaders['train'] = build_dataloader(dataloader_train, data_parmas["BATCH_SIZE"], data_parmas["NUM_WORKERS"],
                     fix_seed=True, shuffle=True)

    def __init_model__(self):
        crnn_params = {"CNN_FEAT_DIM": 32, "CNN_FEAT": False, 
                       "POS_FEAT_DIM": 36, "POS_FEAT_HIDDEN": 32, "POS_FEAT": True,
                       "LSTM_HIDDEN": 32, "N_LSTM_LAYER": 3, "BLSTM": False,
                       "EMB_DIM": 36}

        ngm_params = {"EDGE_FEATURE": "cat", "FEATURE_CHANNEL": 512, 
                      "SK_ITER_NUM": 20, "SK_EPSILON": 1.0e-10, "SK_TAU": 0.05, "SK_EMB": 1,
                      "FIRST_ORDER": True, "EDGE_EMB": False,
                      "GAUSSIAN_SIGMA": 1.0}

        gnn_params = {"GNN_FEAT": [16, 16], "GNN_LAYER": 1}

        train_params = {"MOMENTUM": 0.9, "OPTIMIZER": "sgd", "EPOCH_ITERS": 2000,
                        "LR_DECAY": 0.1, "LR_STEP": [2, 6, 10], 'LR': 1.0e-5,
                        "LOSS_FUNC": "ce"}
        self.crnn_params = crnn_params
        self.ngm_params = ngm_params
        self.gnn_params = gnn_params
        self.train_params = train_params
        
        self.model = GMN_Net(ngm_params, gnn_params, crnn_params).to(self.device)

        if train_params['OPTIMIZER'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=train_params['LR'], 
                                  momentum=train_params['MOMENTUM'], nesterov=True)
        elif train_params['OPTIMIZER'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=train_params['LR'])
        
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params['LR_STEP'], 
                                                   gamma=train_params['LR_DECAY'], last_epoch=-1)

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

    def train(self):
        
        for epoch in tqdm(range(self.train_params['EPOCH_ITERS'])):
            self.model.train()
            accs = []
            for inputs in self.dataloaders['train']:
#                if self.model.module.device != torch.device('cpu'):
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
                    accs.append(acc.item())
                    # backward + optimize
                    loss.backward()
                    self.optimizer.step()


            print(f"[x] training @ epoch {epoch}, acc = {np.mean(accs)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--data_path', type=str, default="/media/z/data21/artery_semantic_segmentation/ThinkMatch/artery/data")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer = Model_Trainer(args, device)
    trainer.train()