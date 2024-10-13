# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/6 20:32
@Auth ： He Yu
@File ：models.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import dgl
import dgl.nn.pytorch as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from tqdm import tqdm
import cf_cython
import utils
import dgl.function as fn
from dgl.nn.pytorch import GATConv

"""
残差网络特征投影
"""


class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(in_dim[i], out_dim, bias=False) for i in range(len(in_dim))])
        # 使用多层网络结构

        self.bn = nn.BatchNorm1d(out_dim)
        self.ac = nn.LeakyReLU()
        # 使用残差连接
        self.shortcuts = nn.ModuleList([nn.Linear(in_dim[i], out_dim) for i in range(len(in_dim))])
        self.reset_params()

    def forward(self, in_feats):
        outs = []
        for i in range(len(in_feats)):
            x = self.linears[i](in_feats[i])
            x = self.bn(x)
            x = self.ac(x)
            res = self.shortcuts[i](in_feats[i])
            out = torch.add(x, res)
            outs.append(out)
        return outs

    def reset_params(self):
        for linear, shortcut in zip(self.linears, self.shortcuts):
            linear.reset_parameters()
            shortcut.reset_parameters()
        self.bn.reset_parameters()


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.bn = nn.BatchNorm1d(out_size * layer_num_heads)
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(
                in_size,
                out_size,
                layer_num_heads,
                dropout,
                activation=F.leaky_relu,
                allow_zero_in_degree=True
            ))
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, hs):
        # gs传入各种子图（type：DGLGraph）
        # h传入源节点和目标节点的特征（src_feat, dst_feat）
        lnc_semantic_embs = []
        dis_semantic_embs = []
        semantic_embs = []
        # print(self.gat_layers)
        # input_h = [(l_x, d_x), (l_x, l_x), (d_x, l_x), (d_x, d_x)]
        for i, (g, h) in enumerate(zip(gs, hs)):
            # print(g, i, h[0].shape, h[1].shape)
            semantic_embs.append(self.gat_layers[i](g, h).flatten(1))

        lnc_semantic_embs = self.semantic_attention(torch.stack((semantic_embs[1], semantic_embs[2]), dim=1))
        dis_semantic_embs = self.semantic_attention(torch.stack((semantic_embs[0], semantic_embs[3]), dim=1))
        # remove semantic attention
        # lnc_semantic_embs = semantic_embs[1]
        # dis_semantic_embs = semantic_embs[0]
        # remove RNAs simi networks
        # lnc_semantic_embs = self.semantic_attention(torch.stack((semantic_embs[1], semantic_embs[1]), dim=1))
        # dis_semantic_embs = self.semantic_attention(torch.stack((semantic_embs[0], semantic_embs[0]), dim=1))
        # lnc_semantic_embs = torch.cat((semantic_embs[1], semantic_embs[2]), dim=1)
        # dis_semantic_embs = torch.cat((semantic_embs[0], semantic_embs[3]), dim=1)
        return [(lnc_semantic_embs, dis_semantic_embs),
                (lnc_semantic_embs, lnc_semantic_embs),
                (dis_semantic_embs, lnc_semantic_embs),
                (dis_semantic_embs, dis_semantic_embs)]

    def reset_params(self):
        for lay in self.gat_layers:
            try:
                lay.reset_parameters()
            except:
                continue


class HANEncoder(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, num_heads, dropout):
        super(HANEncoder, self).__init__()
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(hidden_size * num_heads[0]) for _ in range(len(num_heads))])
        self.act = nn.LeakyReLU()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(
                num_meta_paths, in_size, hidden_size, num_heads[0], dropout
            )
        )
        # multi HAN layer
        # for l in range(1, len(num_heads)):
        #     self.layers.append(
        #         HANLayer(
        #             num_meta_paths,
        #             hidden_size * num_heads[l - 1],
        #             hidden_size,
        #             num_heads[l],
        #             dropout
        #         )
        #     )

    def forward(self, g, h):
        """

        :param g: list-->[DGLGraph:]
        :param h: list-->[(src:Tensor, dst:Tensor),..,]
        :return:  tuple-->(src:Tensor, dst:Tensor)
        """
        # print(self.layers)
        for i, (gnn, bn) in enumerate(zip(self.layers, self.bns)):
            # print(i, gnn)
            h = gnn(g, h)
        return h[0]

    def rest_params(self):
        for lay in self.layers:
            lay.reset_params()


class Hadmard(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            message = fn.u_mul_v('h', 'h', 'z')
            graph.apply_edges(message, etype=etype)
            return graph.edges[etype].data['z']


class Decoder(nn.Module):
    def __init__(self, dim_z, head_size):
        super(Decoder, self).__init__()
        dim_in = head_size * dim_z + 1
        # dim_in = head_size * dim_z
        # dim_in = 2 * head_size * dim_z + 1
        self.mlp_out = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2, bias=True),
            nn.BatchNorm1d(dim_in // 2),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(dim_in // 2, dim_in // 2 // 2, bias=True),
            nn.BatchNorm1d(dim_in // 2 // 2),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(dim_in // 2 // 2, 1, bias=False)
        )

    def forward(self, z):
        h = self.mlp_out(z).squeeze()
        return h

    def reset_params(self):
        for lin in self.mlp_out:
            try:
                lin.reset_parameters()
            except:
                continue


class Model(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(Model, self).__init__()
        self.project = ResNet(in_size, hidden_size)  # input[lnc_feats, dis_feats]
        self.encoder = HANEncoder(num_meta_paths, hidden_size, out_size, num_heads, dropout)
        self.hadmard = Hadmard()
        self.decoder = Decoder(out_size, num_heads[0])
        self.init_params()

    def forward(self, g, neg_g, x, etype):
        # 在encoder中需要传入元路径子图
        meta_gs = list(reversed([g.edge_type_subgraph([etype]) for etype in g.canonical_etypes]))
        # 特征投影
        l_x, d_x = self.project(x)
        input_h = [(l_x, d_x), (l_x, l_x), (d_x, l_x), (d_x, d_x)]
        # remove RNAs simi netwotks
        # input_h = [(l_x, d_x), (d_x, l_x)]
        # embeddings
        if neg_g is None:  # 此判断开启新的预测，即案例分析
            l_embs, d_embs = self.encoder(meta_gs, input_h)
            z = self.hadmard(g, {'lnc': l_embs, 'dis': d_embs}, etype)
            return z
        l_embs, d_embs = self.encoder(meta_gs, input_h)
        pos_z = self.hadmard(g, {'lnc': l_embs, 'dis': d_embs}, etype)
        neg_z = self.hadmard(neg_g, {'lnc': l_embs, 'dis': d_embs}, etype)
        return pos_z, neg_z, l_embs, d_embs

    def init_params(self):
        self.project.reset_params()
        self.encoder.rest_params()
        self.decoder.reset_params()


if __name__ == '__main__':
    # 测试

    pass
