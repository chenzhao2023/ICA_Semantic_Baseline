import torch
import torch.nn as nn

from src.lap_solvers.sinkhorn import Sinkhorn
from src.feature_align import feature_align
from src.gconv import Siamese_Gconv
from models.PCA.affinity_layer import Affinity
from src.lap_solvers.hungarian import hungarian

from src.utils.config import cfg
from models.PCA.model_config import model_cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)


class NodeFeatureEmbedding(nn.Module):

    def __init__(self, pos_feat_dim, pos_feat_hidden, emb_dim, n_layers=0):
        self.emb = nn.Sequential(nn.Linear(pos_feat_dim, pos_feat_hidden), nn.ReLU(),
                                 nn.Linear(pos_feat_dim, emb_dim), nn.ReLU())
    
    def forward(self, x):
        return self.emb(x)

class PCA_GM(nn.Module):
    def __init__(self, pca_params, gnn_params, node_emb_params):
        super(PCA_GM, self).__init__()
        self.pca_params = pca_params
        self.gnn_params = gnn_params
        self.node_emb_params = node_emb_params

        self.sinkhorn = Sinkhorn(max_iter=pca_params['SK_ITER_NUM'], epsilon=pca_params['SK_EPSILON'], tau=pca_params['SK_TAU'])
        # self.l2norm = nn.LocalResponseNorm(pca_params['FEATURE_CHANNEL'] * 2, alpha=pca_params['FEATURE_CHANNEL'] * 2, beta=0.5, k=0)
        self.gnn_layer = gnn_params['GNN_LAYER']
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(pca_params['FEATURE_CHANNEL'], gnn_params['GNN_FEAT'][i])
            else:
                gnn_layer = Siamese_Gconv(gnn_params['GNN_FEAT'][i-1], gnn_params['GNN_FEAT'][i])
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(gnn_params['GNN_FEAT'][i]))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(gnn_params['GNN_FEAT'][i] * 2, gnn_params['GNN_FEAT'][i]))

        self.cross_iter = pca_params['CROSS_ITER']
        self.cross_iter_num = pca_params['CROSS_ITER_NUM']

    def forward(self, data_dict, **kwargs):
        # synthetic data
        src, tgt = data_dict['pos_features']
        ns_src, ns_tgt = data_dict['ns']
        A_src, A_tgt = data_dict['As']

        emb1 = src # concat([batch, feat_dim, n_node], [batch, feat_dim, n_node])
        emb2 = tgt
        ss = []
        if not self.cross_iter:
            # Vanilla PCA-GM
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2]) 
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2) # s [batch, n_node, n_node]
                s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)

                ss.append(s)

                if i == self.gnn_layer - 2:
                    cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                    new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                    new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                    emb1 = new_emb1
                    emb2 = new_emb2
        else:
            # IPCA-GM
            for i in range(self.gnn_layer - 1):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])

            emb1_0, emb2_0 = emb1, emb2
            s = torch.zeros(emb1.shape[0], emb1.shape[1], emb2.shape[1], device=emb1.device)

            for x in range(self.cross_iter_num):
                i = self.gnn_layer - 2
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1 = cross_graph(torch.cat((emb1_0, torch.bmm(s, emb2_0)), dim=-1))
                emb2 = cross_graph(torch.cat((emb2_0, torch.bmm(s.transpose(1, 2), emb1_0)), dim=-1))

                i = self.gnn_layer - 1
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)
                s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                ss.append(s)

        if torch.sum(torch.isnan(ss[-1])) > 0:
            print(ss[-1])
            return data_dict
        else:
            data_dict.update({'ds_mat': ss[-1], 'perm_mat': hungarian(ss[-1], ns_src, ns_tgt)})

            return data_dict
