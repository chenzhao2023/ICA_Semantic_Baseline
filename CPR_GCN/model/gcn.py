import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List, Union



class Gconv(nn.Module):
    r"""
    Graph Convolutional Layer which is inspired and developed based on Graph Convolutional Network (GCN).
    Inspired by `Kipf and Welling. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.
    <https://arxiv.org/abs/1609.02907>`_

    :param in_features: the dimension of input node features
    :param out_features: the dimension of output node features
    """
    def __init__(self, in_features: int, out_features: int):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, A: Tensor, x: Tensor, norm: bool=True) -> Tensor:
        r"""
        Forward computation of graph convolution network.
        A: batch*n_node*n_node;
        x: batch*n_node*node_feature_dim
        
        :param A: :math:`(b\times n\times n)` {0,1} adjacency matrix. :math:`b`: batch size, :math:`n`: number of nodes
        :param x: :math:`(b\times n\times d)` input node embedding. :math:`d`: feature dimension
        :param norm: normalize connectivity matrix or not
        :return: :math:`(b\times n\times d^\prime)` new node embedding
        """
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        x = torch.bmm(A, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)
        return x


if __name__ == '__main__':
    x = torch.rand([5, 10, 19]) # batch, n_node, n_node_feat_dim
    a = torch.rand([5, 10, 10]) # batch, n_node, n_node_feat_dim

    gcn = Gconv(19, 20)
    out = gcn(a,x)
    print(out.shape)