import torch
import torch.nn as nn
import numpy as np
# import flopth
from flopth import flopth

from tree_lstm_artery import BiTreeLSTM, UTD_TreeLSTM, DTU_TreeLSTM


if __name__ == '__main__':

    model = BiTreeLSTM(in_features=12, mlp_hidden=128, lstm_hidden=30, out_features=5)
    
    # features, node_order, adjacency_list, edge_order
    features = torch.rand(7, 12)
    node_order = torch.from_numpy(np.array([3,2,1,0,0,0], dtype=np.int64))
    adjacency_list = torch.from_numpy(np.array([[0,1],
                                                [1,2],
                                                [1,3],
                                                [0,2],
                                                [2,3],
                                                [2,4]], dtype=np.int64))
    edge_order = torch.from_numpy(np.array([3,2,2,3,1,1], dtype=np.int64))

    flops, params = flopth(model, inputs=(features, node_order, adjacency_list, edge_order))
    print(flops)
    print(params)