import math
import torch
import torch.nn as nn
import numpy as np

from src.lap_solvers.sinkhorn import Sinkhorn, GumbelSinkhorn
from src.build_graphs import reshape_edge_feature
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat
from models.NGM.gnn import GNNLayer
from models.NGM.geo_edge_feature import geo_edge_feature
from models.GMN.affinity_layer import InnerpAffinity, GaussianAffinity
from src.evaluation_metric import objective_score
from src.lap_solvers.hungarian import hungarian
from src.utils.gpu_memory import gpu_free_memory
from src.utils.config import cfg

from artery.models.conv_lstm import ConvBLSTM, ConvLSTM


class NodeFeatureExtractor(nn.Module):
    def __init__(self, cnn_feat_dim, cnn_feature=True,
                       pos_feat_dim=16, pos_feat_hidden=32, pos_feature=True,
                       lstm_hidden=32, n_lstm_layer=3, blstm=True,
                       embedding_dim=128) -> None:
        super().__init__()
        self.cnn_feature = cnn_feature
        self.pos_feature = pos_feature
        self.blstm = blstm
        self.pix_encoder = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=1, out_channels=cnn_feat_dim, padding='same'),
                                         nn.MaxPool2d(kernel_size=2),
                                         nn.Conv2d(kernel_size=3, in_channels=cnn_feat_dim, out_channels=cnn_feat_dim*2, padding='same'),
                                         nn.MaxPool2d(kernel_size=2),
                                         nn.Conv2d(kernel_size=3, in_channels=cnn_feat_dim*2, out_channels=cnn_feat_dim*4, padding='same'),
                                         nn.MaxPool2d(kernel_size=2))
        # self.pos_encoder = nn.Sequential(nn.Linear(pos_feat_dim, pos_feat_hidden), nn.ReLU(),
        #                                  nn.Linear(pos_feat_hidden, pos_feat_hidden//2), nn.ReLU(),
        #                                  nn.Linear(pos_feat_hidden//2, pos_feat_hidden//4), nn.ReLU(),
        #                                  nn.Linear(pos_feat_hidden//4, pos_feat_hidden//8), nn.ReLU())
        self.pos_encoder = nn.Sequential(nn.Linear(pos_feat_dim, pos_feat_hidden), nn.Sigmoid())

        if blstm:
            self.lstm = ConvBLSTM(in_channels=cnn_feat_dim*4,
                                  hidden_channels=lstm_hidden,
                                  kernel_size=[(3, 3)]*n_lstm_layer,
                                  num_layers=n_lstm_layer)
        else:
            self.lstm = ConvLSTM(in_channels=cnn_feat_dim*4, 
                                 hidden_channels=lstm_hidden, 
                                 kernel_size=[(3, 3)]*n_lstm_layer, 
                                 num_layers=n_lstm_layer)
        
        if cnn_feature and pos_feature:
            self.embedding = nn.Linear(pos_feat_hidden//8 + lstm_hidden, embedding_dim)
        elif cnn_feature and not pos_feature:
            self.embedding = nn.Linear(lstm_hidden, embedding_dim)
        else:
            self.embedding = nn.Linear(pos_feat_hidden, embedding_dim)
    

    def forward_batch_pixel_feature(self, im_patches, splitter, n_node):
        pixel_features = []
        splitter = np.asarray(splitter.detach().cpu().numpy(), dtype=int)
        im_features = self.pix_encoder(im_patches)
        if self.blstm:
            for i in range(n_node): # torch.flip(node_im_features, dims=[0])[2]
                node_im_features = im_features[splitter[i][0]:splitter[i][1]] 
                node_im_features = torch.unsqueeze(node_im_features, dim=1) # convert (b,c,h,w) to (t, b, c, h, w), batch to time steps
                node_im_features_reverse =  torch.flip(node_im_features, dims=[0])
                node_im_features = self.lstm(node_im_features, node_im_features_reverse)
                node_im_features = node_im_features[:, -1, :, :, :]
                node_im_features_flatten = torch.flatten(node_im_features)
                pixel_features.append(node_im_features_flatten)
        else:
            for i in range(n_node): # torch.flip(node_im_features, dims=[0])[2]
                node_im_features = im_features[splitter[i][0]:splitter[i][1]] 
                node_im_features = torch.unsqueeze(node_im_features, dim=1) # convert (b,c,h,w) to (t, b, c, h, w), batch to time steps
                node_im_features, _ = self.lstm(node_im_features)
                node_im_features = node_im_features[0]
                node_im_features = node_im_features[:, -1, :, :, :]
                node_im_features_flatten = torch.flatten(node_im_features)
                pixel_features.append(node_im_features_flatten)

        pixel_features = torch.stack(pixel_features) # n_node*feature_dim
        return pixel_features

    def forward_batch_pos_feature(self, pos_features):
        pos_features = self.pos_encoder(pos_features) # pos_feature: [n_node, mlp_hidden//8]
        return pos_features

    def forward(self, pos_features, im_patches, splitter, n_node):
        # im_features_vec = torch.flatten(im_features, start_dim=1)
        # if self.cnn_feature:
        #     pixel_features = self.forward_batch_pixel_feature(im_patches[0], splitter[0], n_node)

        # if self.pos_feature:
        #     pos_features = self.forward_batch_pos_feature(pos_features[0]) # pos_feature: [n_node, mlp_hidden//8]
            
        # if self.cnn_feature and self.pos_feature:
        #     fused_features = torch.cat((pos_features, pixel_features), 1)
        # elif self.cnn_feature and not self.pos_feature:
        #     fused_features = pixel_features
        # else:
        #     fused_features = pos_features
        
        # print("feature before embedding", fused_features)
        # extracted_features = self.embedding(torch.unsqueeze(fused_features, axis=0))
        # return extracted_features
        return pos_features


class GMN_Net(nn.Module):
    def __init__(self, ngm_params, gnn_params, crnn_params):
        self.ngm_params = ngm_params
        self.gnn_params = gnn_params
        self.crnn_params = crnn_params

        super(GMN_Net, self).__init__()
        if ngm_params["EDGE_FEATURE"] == 'cat':
            self.affinity_layer = InnerpAffinity(crnn_params["EMB_DIM"])
        # elif ngm_params["EDGE_FEATURE"] == 'geo':
        #     self.affinity_layer = GaussianAffinity(1, ngm_params["GAUSSIAN_SIGMA"])
        else:
            raise ValueError('Unknown edge feature type {}'.format(ngm_params["FEATURE_CHANNEL"]))
        self.tau = ngm_params["SK_TAU"]
        # self.rescale = cfg.PROBLEM.RESCALE #TODO
        self.sinkhorn = Sinkhorn(max_iter=ngm_params["SK_ITER_NUM"], tau=self.tau, epsilon=ngm_params["SK_EPSILON"])
        self.gumbel_sinkhorn = GumbelSinkhorn(max_iter=ngm_params["SK_ITER_NUM"], tau=self.tau * 10, epsilon=ngm_params["SK_EPSILON"], batched_operation=True)
        self.l2norm = nn.LocalResponseNorm(crnn_params["EMB_DIM"], alpha=crnn_params["EMB_DIM"], beta=0.5, k=0)

        self.node_layers = NodeFeatureExtractor(
                        cnn_feat_dim=crnn_params["CNN_FEAT_DIM"], cnn_feature=crnn_params["CNN_FEAT"],
                        pos_feat_dim=crnn_params["POS_FEAT_DIM"], pos_feat_hidden=crnn_params["POS_FEAT_HIDDEN"], pos_feature=crnn_params["POS_FEAT"],
                        lstm_hidden=crnn_params["LSTM_HIDDEN"], n_lstm_layer=crnn_params["N_LSTM_LAYER"], blstm=crnn_params["BLSTM"],
                        embedding_dim=crnn_params["EMB_DIM"])
        self.edge_layers = NodeFeatureExtractor(
                        cnn_feat_dim=crnn_params["CNN_FEAT_DIM"], cnn_feature=crnn_params["CNN_FEAT"],
                        pos_feat_dim=crnn_params["POS_FEAT_DIM"], pos_feat_hidden=crnn_params["POS_FEAT_HIDDEN"], pos_feature=crnn_params["POS_FEAT"],
                        lstm_hidden=crnn_params["LSTM_HIDDEN"], n_lstm_layer=crnn_params["N_LSTM_LAYER"], blstm=crnn_params["BLSTM"],
                        embedding_dim=crnn_params["EMB_DIM"])

        self.gnn_layer = gnn_params["GNN_LAYER"]
        for i in range(self.gnn_layer):
            if i == 0:
                #gnn_layer = Gconv(1, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(1, 1, 
                                     gnn_params["GNN_FEAT"][i] + (1 if ngm_params["SK_EMB"] else 0), 
                                     gnn_params["GNN_FEAT"][i],
                                     sk_channel=ngm_params["SK_EMB"], sk_tau=ngm_params["SK_TAU"], edge_emb=ngm_params["EDGE_EMB"])
                #gnn_layer = HyperConvLayer(1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            else:
                #gnn_layer = Gconv(cfg.NGM.GNN_FEAT, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(gnn_params["GNN_FEAT"][i - 1] + (1 if ngm_params["SK_EMB"] else 0), 
                                     gnn_params["GNN_FEAT"][i - 1],
                                     gnn_params["GNN_FEAT"][i] + (1 if ngm_params["SK_EMB"] else 0), 
                                     gnn_params["GNN_FEAT"][i],
                                     sk_channel=ngm_params["SK_EMB"],
                                     sk_tau=ngm_params["SK_TAU"], 
                                     edge_emb=ngm_params["EDGE_EMB"])
                #gnn_layer = HyperConvLayer(cfg.NGM.GNN_FEAT[i-1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i-1],
                #                           cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(gnn_params["GNN_FEAT"][-1] + (1 if ngm_params["SK_EMB"] else 0), 1)

    def forward(self, data_dict, **kwargs):
        batch_size = data_dict['batch_size']
        # src, tgt = data_dict['images']
        # P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']
        G_src, G_tgt = data_dict['Gs']
        H_src, H_tgt = data_dict['Hs']
        K_G, K_H = data_dict['KGHs']

        # extract feature
        pos_features_src, pos_features_tgt = data_dict['pos_features']
        im_patches_src, im_patches_tgt = data_dict['patches']
        splitter_src, splitter_tgt = data_dict["splitter"]
        # pos_features, im_patches, splitter, n_node
        src_node = self.node_layers(pos_features_src, im_patches_src, splitter_src, ns_src)     # batch*n_node1*crnn_params["EMB_DIM"]
        src_edge = self.edge_layers(pos_features_src, im_patches_src, splitter_src, ns_src)     # batch*n_node1*crnn_params["EMB_DIM"]
        tgt_node = self.node_layers(pos_features_tgt, im_patches_tgt, splitter_tgt, ns_tgt)     # batch*n_node2*crnn_params["EMB_DIM"]
        tgt_edge = self.edge_layers(pos_features_tgt, im_patches_tgt, splitter_tgt, ns_tgt)     # batch*n_node2*crnn_params["EMB_DIM"]

        # feature normalization
        U_src = torch.transpose(self.l2norm(src_node), 1, 2) # batch*crnn_params["EMB_DIM"]*n_node1
        F_src = torch.transpose(self.l2norm(src_edge), 1, 2) # batch*crnn_params["EMB_DIM"]*n_node1
        U_tgt = torch.transpose(self.l2norm(tgt_node), 1, 2) # batch*crnn_params["EMB_DIM"]*n_node2
        F_tgt = torch.transpose(self.l2norm(tgt_edge), 1, 2) # batch*crnn_params["EMB_DIM"]*n_node2

        # calculate affnity matrix
        if self.ngm_params['EDGE_FEATURE'] == 'cat':
            X = reshape_edge_feature(F_src, G_src, H_src) # crnn_params["EMB_DIM"]*2*edge_num
            Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt) # crnn_params["EMB_DIM"]*2*edge_num
        # elif cfg.NGM.EDGE_FEATURE == 'geo':
        #     X = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
        #     Y = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))

        # affinity layer, Ke shape = 
        Ke, Kp = self.affinity_layer(X, Y, U_src, U_tgt) # Ke=X*\omega*Y, Kp=U1*U2.T where U1 and U2 are node features, X and Y are edge features
        # ke: shape = n_edge_in_G1*n_edge_in_G2, Kp: shape = n_node_in_G1*n_node_in_G2
        K = construct_aff_mat(Ke, torch.zeros_like(Kp), K_G, K_H) #K.shape = n_node_in_G1*n_node_in_G2

        A = (K > 0).to(K.dtype)

        if self.ngm_params["FIRST_ORDER"]:
            emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1) # shape = n_node in G1*n_node in G2
        else:
            emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

        emb_K = K.unsqueeze(-1) # (n_node in G1*n_node in G2) * (n_node in G1*n_node in G2)

        # NGM qap solver
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_K, emb = gnn_layer(A, emb_K, emb, ns_src, ns_tgt) #, norm=False) # emb_K, edge features, emb: node features

        v = self.classifier(emb)  # batch*(n_node1*n_node2)*1
        s = v.view(v.shape[0], ns_tgt, -1).transpose(1, 2)

        if self.training or self.ngm_params["GUMBEL_SK"] <= 0:
        #if cfg.NGM.GUMBEL_SK <= 0:
            ss = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True) #n1*n2
            print(ss)
            x = hungarian(ss, ns_src, ns_tgt)
        else:
            gumbel_sample_num = self.ngm_params["GUMBEL_SK"]
            if self.training:
                gumbel_sample_num //= 10
            ss_gumbel = self.gumbel_sinkhorn(s, ns_src, ns_tgt, sample_num=gumbel_sample_num, dummy_row=True)

            repeat = lambda x, rep_num=gumbel_sample_num: torch.repeat_interleave(x, rep_num, dim=0)
            if not self.training:
                ss_gumbel = hungarian(ss_gumbel, repeat(ns_src), repeat(ns_tgt))
            ss_gumbel = ss_gumbel.reshape(batch_size, gumbel_sample_num, ss_gumbel.shape[-2], ss_gumbel.shape[-1])

            if ss_gumbel.device.type == 'cuda':
                dev_idx = ss_gumbel.device.index
                free_mem = gpu_free_memory(dev_idx) - 100 * 1024 ** 2 # 100MB as buffer for other computations
                K_mem_size = K.element_size() * K.nelement()
                max_repeats = free_mem // K_mem_size
                if max_repeats <= 0:
                    print('Warning: GPU may not have enough memory')
                    max_repeats = 1
            else:
                max_repeats = gumbel_sample_num

            obj_score = []
            for idx in range(0, gumbel_sample_num, max_repeats):
                if idx + max_repeats > gumbel_sample_num:
                    rep_num = gumbel_sample_num - idx
                else:
                    rep_num = max_repeats
                obj_score.append(
                    objective_score(
                        ss_gumbel[:, idx:(idx+rep_num), :, :].reshape(-1, ss_gumbel.shape[-2], ss_gumbel.shape[-1]),
                        repeat(K, rep_num)
                    ).reshape(batch_size, -1)
                )
            obj_score = torch.cat(obj_score, dim=1)
            min_obj_score = obj_score.min(dim=1)
            ss = ss_gumbel[torch.arange(batch_size), min_obj_score.indices.cpu(), :, :]
            x = hungarian(ss, repeat(ns_src), repeat(ns_tgt))

        data_dict.update({'ds_mat': ss, 'perm_mat': x, 'aff_mat': K})
        return data_dict


class NodeFeatureEmb(nn.Module):
    def __init__(self, n_layers, pos_feat_dim=16, pos_feat_hidden=32, embedding_dim=128) -> None:
        super(NodeFeatureEmb, self).__init__()
        self.n_layers = n_layers
        for i in range(n_layers):
            if i == 0:
                self.add_module(f'mlp_{i}', nn.Linear(pos_feat_dim, pos_feat_hidden))
                self.add_module(f'act_{i}', nn.ReLU())
            elif i == n_layers-1:
                self.add_module(f'mlp_{i}', nn.Linear(pos_feat_hidden, embedding_dim))
                self.add_module(f'act_{i}', nn.ReLU())
            else:
                self.add_module(f'mlp_{i}', nn.Linear(pos_feat_hidden, pos_feat_hidden))
                self.add_module(f'act_{i}', nn.ReLU())

    def forward(self, x):
        for i in range(self.n_layers):
            mlp_layer = getattr(self, f'mlp_{i}')
            act = getattr(self, f'act_{i}')
            x = act(mlp_layer(x))
        return x


class GMN_FEAT_Net(nn.Module):
    def __init__(self, ngm_params, gnn_params, feat_emb_params):
        self.ngm_params = ngm_params
        self.gnn_params = gnn_params
        self.feat_emb_params = feat_emb_params

        super(GMN_FEAT_Net, self).__init__()
        if ngm_params["EDGE_FEATURE"] == 'cat':
            self.affinity_layer = InnerpAffinity(feat_emb_params["EMB_DIM"])
        else:
            raise ValueError('Unknown edge feature type {}'.format(ngm_params["EDGE_FEATURE"]))
        self.tau = ngm_params["SK_TAU"]
        # self.rescale = cfg.PROBLEM.RESCALE # TODO
        self.sinkhorn = Sinkhorn(max_iter=ngm_params["SK_ITER_NUM"], tau=self.tau, epsilon=ngm_params["SK_EPSILON"])
        self.gumbel_sinkhorn = GumbelSinkhorn(max_iter=ngm_params["SK_ITER_NUM"], tau=self.tau * 10, epsilon=ngm_params["SK_EPSILON"], batched_operation=True)
        self.l2norm = nn.LocalResponseNorm(feat_emb_params["EMB_DIM"], alpha=feat_emb_params["EMB_DIM"], beta=0.5, k=0)

        self.node_layers = NodeFeatureEmb(feat_emb_params['N_LAYERS'], pos_feat_dim=feat_emb_params['POS_FEAT_DIM'], 
                                          pos_feat_hidden=feat_emb_params['POS_FEAT_HIDDEN'], embedding_dim=feat_emb_params['EMB_DIM'])
        self.edge_layers = NodeFeatureEmb(feat_emb_params['N_LAYERS'], pos_feat_dim=feat_emb_params['POS_FEAT_DIM'], 
                                          pos_feat_hidden=feat_emb_params['POS_FEAT_HIDDEN'], embedding_dim=feat_emb_params['EMB_DIM'])

        self.gnn_layer = gnn_params["GNN_LAYER"]
        for i in range(self.gnn_layer):
            if i == 0:
                #gnn_layer = Gconv(1, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(1, 1, 
                                     gnn_params["GNN_FEAT"][i] + (1 if ngm_params["SK_EMB"] else 0), 
                                     gnn_params["GNN_FEAT"][i],
                                     sk_channel=ngm_params["SK_EMB"], sk_tau=ngm_params["SK_TAU"], edge_emb=ngm_params["EDGE_EMB"])
                #gnn_layer = HyperConvLayer(1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            else:
                #gnn_layer = Gconv(cfg.NGM.GNN_FEAT, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(gnn_params["GNN_FEAT"][i - 1] + (1 if ngm_params["SK_EMB"] else 0), 
                                     gnn_params["GNN_FEAT"][i - 1],
                                     gnn_params["GNN_FEAT"][i] + (1 if ngm_params["SK_EMB"] else 0), 
                                     gnn_params["GNN_FEAT"][i],
                                     sk_channel=ngm_params["SK_EMB"],
                                     sk_tau=ngm_params["SK_TAU"], 
                                     edge_emb=ngm_params["EDGE_EMB"])
                #gnn_layer = HyperConvLayer(cfg.NGM.GNN_FEAT[i-1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i-1],
                #                           cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(gnn_params["GNN_FEAT"][-1] + (1 if ngm_params["SK_EMB"] else 0), 1)

    def forward(self, data_dict, **kwargs):
        batch_size = data_dict['batch_size']
        # src, tgt = data_dict['images']
        # P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']
        G_src, G_tgt = data_dict['Gs']
        H_src, H_tgt = data_dict['Hs']
        K_G, K_H = data_dict['KGHs']

        # extract feature
        pos_features_src, pos_features_tgt = data_dict['pos_features']
        # pos_features, im_patches, splitter, n_node
        src_node = self.node_layers(pos_features_src)     # batch*n_node1*feat_emb_params["EMB_DIM"]
        src_edge = self.edge_layers(pos_features_src)     # batch*n_node1*feat_emb_params["EMB_DIM"]
        tgt_node = self.node_layers(pos_features_tgt)     # batch*n_node2*feat_emb_params["EMB_DIM"]
        tgt_edge = self.edge_layers(pos_features_tgt)     # batch*n_node2*feat_emb_params["EMB_DIM"]

        # feature normalization
        # U_src = torch.transpose(self.l2norm(src_node), 1, 2) # batch*feat_emb_params["EMB_DIM"]*n_node1
        # F_src = torch.transpose(self.l2norm(src_edge), 1, 2) # batch*feat_emb_params["EMB_DIM"]*n_node1
        # U_tgt = torch.transpose(self.l2norm(tgt_node), 1, 2) # batch*feat_emb_params["EMB_DIM"]*n_node2
        # F_tgt = torch.transpose(self.l2norm(tgt_edge), 1, 2) # batch*feat_emb_params["EMB_DIM"]*n_node2
        U_src = torch.transpose(src_node, 1, 2) # [BATCH_SIZE, EMB, NS_max]
        F_src = torch.transpose(src_edge, 1, 2)
        U_tgt = torch.transpose(tgt_node, 1, 2)
        F_tgt = torch.transpose(tgt_edge, 1, 2)

        # calculate affnity matrix
        if self.ngm_params['EDGE_FEATURE'] == 'cat':
            X = reshape_edge_feature(F_src, G_src, H_src) # feat_emb_params["EMB_DIM"]*2*edge_num
            Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt) # feat_emb_params["EMB_DIM"]*2*edge_num
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))

        # affinity layer, Ke shape = 
        Ke, Kp = self.affinity_layer(X, Y, U_src, U_tgt) # Ke=X*\omega*Y, Kp=U1*U2.T where U1 and U2 are node features, X and Y are edge features
        # ke: shape = n_edge_in_G1*n_edge_in_G2, Kp: shape = n_node_in_G1*n_node_in_G2
        K = construct_aff_mat(Ke, torch.zeros_like(Kp), K_G, K_H) #K.shape = n_node_in_G1*n_node_in_G2

        A = (K > 0).to(K.dtype)

        if self.ngm_params["FIRST_ORDER"]:
            emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1) # shape = n_node in G1*n_node in G2
        else:
            emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

        emb_K = K.unsqueeze(-1) # (n_node in G1*n_node in G2) * (n_node in G1*n_node in G2)

        # NGM qap solver
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_K, emb = gnn_layer(A, emb_K, emb, ns_src, ns_tgt) #, norm=False) # emb_K, edge features, emb: node features

        v = self.classifier(emb)  # batch*(n_node1*n_node2)*1
        s = v.view(v.shape[0], Kp.shape[2], -1).transpose(1, 2)

        if self.training or self.ngm_params["GUMBEL_SK"] <= 0:
            ss = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True) #n1*n2
            # print(ss)
            x = hungarian(ss, ns_src, ns_tgt)
        else:
            gumbel_sample_num = self.ngm_params["GUMBEL_SK"]
            if self.training:
                gumbel_sample_num //= 10
            ss_gumbel = self.gumbel_sinkhorn(s, ns_src, ns_tgt, sample_num=gumbel_sample_num, dummy_row=True)

            repeat = lambda x, rep_num=gumbel_sample_num: torch.repeat_interleave(x, rep_num, dim=0)
            if not self.training:
                ss_gumbel = hungarian(ss_gumbel, repeat(ns_src), repeat(ns_tgt))
            ss_gumbel = ss_gumbel.reshape(batch_size, gumbel_sample_num, ss_gumbel.shape[-2], ss_gumbel.shape[-1])

            if ss_gumbel.device.type == 'cuda':
                dev_idx = ss_gumbel.device.index
                free_mem = gpu_free_memory(dev_idx) - 100 * 1024 ** 2 # 100MB as buffer for other computations
                K_mem_size = K.element_size() * K.nelement()
                max_repeats = free_mem // K_mem_size
                if max_repeats <= 0:
                    print('Warning: GPU may not have enough memory')
                    max_repeats = 1
            else:
                max_repeats = gumbel_sample_num

            obj_score = []
            for idx in range(0, gumbel_sample_num, max_repeats):
                if idx + max_repeats > gumbel_sample_num:
                    rep_num = gumbel_sample_num - idx
                else:
                    rep_num = max_repeats
                obj_score.append(
                    objective_score(
                        ss_gumbel[:, idx:(idx+rep_num), :, :].reshape(-1, ss_gumbel.shape[-2], ss_gumbel.shape[-1]),
                        repeat(K, rep_num)
                    ).reshape(batch_size, -1)
                )
            obj_score = torch.cat(obj_score, dim=1)
            min_obj_score = obj_score.min(dim=1)
            ss = ss_gumbel[torch.arange(batch_size), min_obj_score.indices.cpu(), :, :]
            x = hungarian(ss, repeat(ns_src), repeat(ns_tgt))

        data_dict.update({'ds_mat': ss, 'perm_mat': x, 'aff_mat': K})
        return data_dict
