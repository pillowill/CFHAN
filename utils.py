# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/26 15:48
@Auth ： He Yu
@File ：utils.py
@IDE ：PyCharm
@Function ：Function of the script
"""
# from sknetwork.clustering import Louvain
from sknetwork.hierarchy import Ward, cut_straight
from sknetwork.utils import membership_matrix
from sknetwork.clustering import Louvain, KMeans, PropagationClustering
# from scipy.spatial.distance import cdist
from multiprocessing_on_dill.dummy import Pool
from itertools import product
import numpy as np
import torch
from tqdm import tqdm
from geomloss import SamplesLoss
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import math

torch.autograd.set_detect_anomaly(True)


def get_etype_adj(graph, etype, sizes):
    """
    用于返回异构图指定类型边的meta-graph的邻接矩阵
    :param graph: 传入的图，一般为异构图
    :param etype: 指定异构图的边类型
    :return: 邻接矩阵
    """
    etype_adj = np.zeros(sizes)
    for i, j in zip(graph.adj(etype=etype).indices()[0, :], graph.adj(etype=etype).indices()[1, :]):
        etype_adj[i][j] = 1
    return etype_adj


def get_edges(edges1, edges2):
    """
    dgl中edges返回是字典，因此需要对字典进行重组
    :param edges1:
    :param edges2:
    :return:
    """
    src = edges1.long().reshape(-1, 1)
    dst = edges2.long().reshape(-1, 1)
    return torch.cat((src, dst), dim=1)


def ward_T_f(adj):
    ward = Ward()
    lnc_dendrogram = ward.fit_transform(adj)
    dis_dendrogram = ward.fit_transform(adj.T)
    lnc_labels = cut_straight(lnc_dendrogram, 100)
    dis_labels = cut_straight(dis_dendrogram, 100)
    lnc_mem_mat = membership_matrix(lnc_labels)
    dis_mem_mat = membership_matrix(dis_labels)
    # print(lnc_mem_mat.shape, dis_mem_mat.shape)
    T = (lnc_mem_mat @ dis_mem_mat.T).astype(int)
    return torch.from_numpy(T.A)


def Louvain_T_f(adj):
    m, n = adj.shape
    louvain = Louvain()
    bi_adj = np.concatenate(
        (np.concatenate((np.zeros((m, m)), adj), axis=1),
         np.concatenate((adj.T, np.zeros((n, n))), axis=1)),
        axis=0
    )
    labels = louvain.fit_transform(bi_adj)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return torch.from_numpy(T.A[:m, m:])


def get_T_f(adj, method='ward'):
    """
    在异构图发现社区
    :param adj: lnc-dis的邻接矩阵
    :param method: 指定社区发现的算法
    :return: Tensor格式的矩阵
    """
    # 注：如果计算PmliPred的数据集，Louvain会导致计算很慢，使用ward
    if method == 'ward':
        ward_tf = ward_T_f(adj)
        return ward_tf
    elif method == 'louvain':
        louvain_tf = Louvain_T_f(adj)
        return louvain_tf


def get_CF_single(params):
    adj, simi_mat, lnc_node_nns, dis_node_nns, T_f, thresh, node_pairs, verbose = params
    T_cf = torch.zeros(adj.shape)
    adj_cf = torch.zeros(adj.shape)
    for a, b in tqdm(node_pairs, position=0):
        nns_a = lnc_node_nns[a]
        nns_b = dis_node_nns[b]
        i, j = 0, 0
        while i < len(nns_a) - 1 and j < len(nns_b) - 1:
            if simi_mat[a, nns_a[i]] + simi_mat[nns_b[j], b] > 2 * thresh:
                T_cf[a, b] = T_f[a, b]
                adj_cf[a, b] = adj[a, b]
                break
            if T_f[nns_b[j], nns_a[i]] != T_f[a, b]:
                T_cf[a, b] = 1 - T_f[a, b]
                adj_cf[a, b] = adj[nns_b[j], nns_a[i]]
                break
            if simi_mat[a, nns_a[i + 1]] < simi_mat[nns_b[j + 1], b]:
                i += 1
            else:
                j += 1
    return T_cf, adj_cf


def get_CF(adj, lnc_embs, dis_embs, T_f, dist='cosine', thresh=0.5, n_workers=10):
    """
    根据公式得到T_f对应的反事实连接与反事实邻接矩阵，并行运算
    :param adj: lnc-dis邻接矩阵
    :param lnc_embs:
    :param dis_embs:
    :param T_f:
    :param dist: 距离矩阵的类型
    :param thresh:
    :param n_workers:
    :return:
    """
    # Euclidean distance
    # simi_mat行索引代表基因，列索引代表疾病
    if dist == 'euclidean':
        simi_mat = torch.cdist(lnc_embs, dis_embs, p=2)
    else:
        # dist == 'cosine'
        simi_mat = \
            1 - (lnc_embs @ dis_embs.T) / (
                    torch.norm(lnc_embs, dim=1).view(-1, 1) @ torch.norm(dis_embs, dim=1).view(1, -1))

    # 对simi_mat 算一个阈值
    thresh = torch.quantile(simi_mat, thresh)

    # 对simi_mat按每行/列进行排序，即按相似度大小排序，升序
    lnc_node_nns = torch.argsort(simi_mat, dim=1)
    dis_node_nns = torch.argsort(simi_mat.T, dim=1)
    # 接下来需要找到每个基因-疾病节点对最邻近的 CF 节点对（也是基因-疾病节点对）

    node_pairs = list(product(range(adj.shape[0]), range(adj.shape[1])))
    pool = Pool(n_workers)
    batches = np.array_split(node_pairs, n_workers)
    results = pool.map(get_CF_single,
                       [(adj, simi_mat, lnc_node_nns, dis_node_nns, T_f, thresh, np_batch, True) for np_batch in
                        batches])
    results = list(zip(*results))
    T_cf = torch.sum(torch.stack(results[0]), dim=0)
    adj_cf = torch.sum(torch.stack(results[1]), dim=0)
    return T_cf, adj_cf


def sample_nodepairs(num_np, T_f, T_cf):
    # TODO: add sampling with separated treatments
    # get the factual node pairs

    nodepairs_f = np.array(np.where(T_f == 1)).T
    # get the counter factual node pairs
    nodepairs_cf = np.array(np.where(T_cf == 1)).T

    f_idx = np.random.choice(len(nodepairs_f), min(num_np, len(nodepairs_f)), replace=False)
    np_f = nodepairs_f[f_idx]

    cf_idx = np.random.choice(len(nodepairs_cf), min(num_np, len(nodepairs_f)), replace=False)
    np_cf = nodepairs_cf[cf_idx]
    return np_f, np_cf


def calc_disc(disc_func, l_embs, d_embs, nodepairs_f, nodepairs_cf):
    X_f = torch.cat((l_embs[nodepairs_f.T[0]], d_embs[nodepairs_f.T[1]]), dim=1)
    X_cf = torch.cat((l_embs[nodepairs_cf.T[0]], d_embs[nodepairs_cf.T[1]]), dim=1)
    # print([nodepairs_f.T[0].shape, nodepairs_f.T[1].shape, nodepairs_cf.T[0].shape, nodepairs_cf.T[1].shape])
    if disc_func == 'lin':
        mean_f = X_f.mean(0)
        mean_cf = X_cf.mean(0)
        loss_disc = torch.sqrt(F.mse_loss(mean_f, mean_cf) + 1e-6)
    elif disc_func == 'kl':
        # TODO: kl divergence
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        # 将X_f和X_cf都转化为概率分布
        X_cf_P = F.log_softmax(X_cf.float(), dim=1)
        X_f_P = F.softmax(X_f.float(), dim=1)
        # print(X_f_P.shape, X_cf_P.shape)
        return kl_loss(X_cf_P, X_f_P)
    elif disc_func == 'w':
        # Wasserstein distance
        dist = SamplesLoss(loss="gaussian", p=2, blur=0.05)
        loss_disc = dist(X_cf, X_f)
    else:
        raise Exception('unsupported distance function for discrepancy loss')
    return loss_disc


def calc_pre_loss(pos_score, neg_score):
    # Ensure pos_score and neg_score have the same size
    min_len = min(pos_score.size(0), neg_score.size(0))
    pos_score = pos_score[:min_len]
    neg_score = neg_score[:min_len]

    # Calculate hinge loss for link prediction
    n_edges = min_len
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


def cf_cy_type_match(cf_params):
    """
    专为cython版本的getCF函数进行数据格式转换
    :return:
    """
    adj, lnc_embs, dis_embs, T_f, dist, thresh = cf_params
    adj = adj.astype(np.double)
    lnc_embs = lnc_embs.cpu().detach().numpy().astype(np.double)
    dis_embs = dis_embs.cpu().detach().numpy().astype(np.double)
    T_f = T_f.numpy().astype(np.double)
    dist = str(dist)
    thresh = float(thresh)
    return adj, lnc_embs, dis_embs, T_f, dist, thresh


class MultipleOptimizer:
    """ a class that wraps multiple optimizers """

    def __init__(self, lr_scheduler, *op):
        self.optimizers = op
        self.steps = 0
        self.reset_count = 0
        self.next_start_step = 10
        self.multi_factor = 2
        self.total_epoch = 0
        if lr_scheduler == 'sgdr':
            self.update_lr = self.update_lr_SGDR
        elif lr_scheduler == 'cos':
            self.update_lr = self.update_lr_cosine
        elif lr_scheduler == 'zigzag':
            self.update_lr = self.update_lr_zigzag
        elif lr_scheduler == 'none':
            self.update_lr = self.no_update

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def no_update(self, base_lr):
        return base_lr

    def update_lr_SGDR(self, base_lr):
        end_lr = 1e-3  # 0.001
        total_T = self.total_epoch + 1
        if total_T >= self.next_start_step:
            self.steps = 0
            self.next_start_step *= self.multi_factor
        cur_T = self.steps + 1
        lr = end_lr + 1 / 2 * (base_lr - end_lr) * (1.0 + math.cos(math.pi * cur_T / total_T))
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        self.total_epoch += 1
        return lr

    def update_lr_zigzag(self, base_lr):
        warmup_steps = 50
        annealing_steps = 20
        end_lr = 1e-4
        if self.steps < warmup_steps:
            lr = base_lr * (self.steps + 1) / warmup_steps
        elif self.steps < warmup_steps + annealing_steps:
            step = self.steps - warmup_steps
            q = (annealing_steps - step) / annealing_steps
            lr = base_lr * q + end_lr * (1 - q)
        else:
            self.steps = self.steps - warmup_steps - annealing_steps
            lr = end_lr
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        return lr

    def update_lr_cosine(self, base_lr):
        """ update the learning rate of all params according to warmup and cosine annealing """
        # 400, 1e-3
        warmup_steps = 10
        annealing_steps = 500
        end_lr = 1e-3
        if self.steps < warmup_steps:
            lr = base_lr * (self.steps + 1) / warmup_steps
        elif self.steps < warmup_steps + annealing_steps:
            step = self.steps - warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / annealing_steps))
            lr = base_lr * q + end_lr * (1 - q)
        else:
            # lr = base_lr * 0.001
            self.steps = self.steps - warmup_steps - annealing_steps
            lr = end_lr
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        return lr
