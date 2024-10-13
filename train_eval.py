# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/10 17:19
@Auth ： He Yu
@File ：train_eval.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import os
import time
import main
import utils
import torch
import cf_cython
import numpy as np
from tqdm import tqdm
from models import Model
from ranger import Ranger
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat, loadmat
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, auc, confusion_matrix
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_labels = []


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_edges(g, neg_g, etype='link'):
    pos_edges = utils.get_edges(g.edge_type_subgraph([etype]).edges()[0],
                                g.edge_type_subgraph([etype]).edges()[1])
    neg_edges = utils.get_edges(neg_g.edge_type_subgraph([etype]).edges()[0],
                                neg_g.edge_type_subgraph([etype]).edges()[1])
    return pos_edges, neg_edges


def calc_Tf_Tcf_Acf(g, neg_g, x):
    pos_edges, neg_edges = get_edges(g, neg_g)
    T_f_list = torch.zeros(pos_edges.shape[0] + neg_edges.shape[0])
    T_cf_list = torch.zeros(pos_edges.shape[0] + neg_edges.shape[0])
    # x-->[lnc_emb, dis_emb]
    etype_adj = utils.get_etype_adj(g, 'link', sizes=(x[0].shape[0], x[1].shape[0]))
    print(f"adj:{etype_adj.shape}")
    # 1.首先由观察adj得到事实adj
    # 注：如果计算PmliPred的数据集，Louvain会导致计算很慢，推荐使用ward
    T_f = utils.get_T_f(etype_adj, method='louvain')
    cy_cf_params = [etype_adj, x[0], x[1], T_f, None, 70]
    cy_adj, cy_lnc_embs, cy_dis_embs, cy_T_f, cy_dist, cy_thresh \
        = utils.cf_cy_type_match(cy_cf_params)
    T_cf, adj_cf = cf_cython.get_CF_cython(
        adj=cy_adj,
        lnc_node_embs=cy_lnc_embs,
        dis_node_embs=cy_dis_embs,
        T_f=cy_T_f,
        thresh=cy_thresh
    )
    for idx, (a, b) in enumerate(zip(
            torch.cat((pos_edges[:, 0], neg_edges[:, 0]), dim=0),
            torch.cat((pos_edges[:, 1], neg_edges[:, 1]), dim=0))):
        try:
            T_f_list[idx] = T_f[a, b]
        except:
            pass
    print(f"T_f:{T_f.shape}")
    for idx, (a, b) in enumerate(zip(
            torch.cat((pos_edges[:, 0], neg_edges[:, 0]), dim=0),
            torch.cat((pos_edges[:, 1], neg_edges[:, 1]), dim=0))):
        try:
            T_cf_list[idx] = T_cf[a, b]
        except:
            pass
    print(f"T_cf:{T_cf.shape}")
    return T_f, T_cf, T_f_list, T_cf_list, adj_cf


def single_eval(i, k_fold, model, graphs):
    model.eval()
    with torch.no_grad():
        te_g = graphs[2][k_fold[i]].to(device)
        te_neg_g = graphs[3][k_fold[i]].to(device)
        lnc_feat = te_g.nodes['lnc'].data['features'].float().to(device)
        dis_feat = te_neg_g.nodes['dis'].data['features'].float().to(device)
        z, neg_z, l_emb, d_emb = model(te_g, te_neg_g, [lnc_feat, dis_feat], 'link')
        T_f_array, T_cf_array, T_f_1dim, T_cf_1dim, A_cf = calc_Tf_Tcf_Acf(te_g, te_neg_g, [l_emb, d_emb])
        multi_z = torch.cat((z.to(device), neg_z.to(device)), dim=0)
        z_f = torch.cat([multi_z, T_f_1dim.reshape(-1, 1).to(device)], dim=1)
        te_logit_f = model.decoder(z_f).reshape(-1, 1)
    return te_logit_f


def draw_roc(y_pre, y_label):
    fpr, tpr, thersholds = roc_curve(y_label, y_pre)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def calc_metrics(scores, labels, show_roc=False):
    preds = (scores > torch.mean(scores)).float()
    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    prec = precision_score(labels.cpu().numpy(), preds.cpu().numpy())
    rec = recall_score(labels.cpu().numpy(), preds.cpu().numpy())
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
    cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
    my_auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    aupr = average_precision_score(labels.cpu().numpy(), scores.cpu().numpy())
    if show_roc:
        draw_roc(torch.sigmoid(preds).cpu().numpy(), labels.cpu().numpy())
    return {'auc': my_auc,
            'aupr': aupr,
            'precision': prec,
            'recall': rec,
            'accuracy': acc,
            'f1': f1,
            'cm': cm
            }


def get_labels(adj_cf, pos_g, neg_g):
    labels_f = torch.cat([torch.ones(pos_g.num_edges(etype='link')),
                          torch.zeros(neg_g.num_edges(etype='link'))]).reshape(-1, 1)
    pos_edges, neg_edges = get_edges(pos_g, neg_g)
    labels_cf = torch.zeros(pos_edges.shape[0] + neg_edges.shape[0])
    for i, (a, b) in enumerate(zip(
            torch.cat((pos_edges[:, 0], neg_edges[:, 0]), dim=0),
            torch.cat((pos_edges[:, 1], neg_edges[:, 1]), dim=0))):
        labels_cf[i] = (adj_cf[a, b])
    return labels_f.reshape(-1, 1), labels_cf.reshape(-1, 1)


def model_train(in_dims, hid_dim, out_dim, num_heads, graphs, k_fold, num_epochs=100, lr=0.001, weight_decay=0.001,
                device='cpu', save_model_path=None, if_save_model=False, show_loss=True, show_eval=True):
    assert type(k_fold) == list, 'input list of the order of k fold'
    assert len(in_dims) == 2 and type(in_dims) == list, 'input dims must contains 2 elements and type of list'
    if os.path.exists('./lnc_mi/dataset/Training-validation dataset(PmliPEMG)/train_cfs.mat'):
        train_cdf_dict = loadmat('./lnc_mi/dataset/Training-validation dataset(PmliPEMG)/train_cfs.mat')
        T_f_arrays, T_cf_arrays, T_f_1dims, T_cf_1dims, A_cfs \
            = (train_cdf_dict['T_f_arrays'], train_cdf_dict['T_cf_arrays'], train_cdf_dict['T_f_1dims'],
               train_cdf_dict['T_cf_1dims'], train_cdf_dict['A_cfs'])
        print('load train_cdf_dict successfully')
    else:
        T_f_arrays, T_cf_arrays, T_f_1dims, T_cf_1dims, A_cfs = [], [], [], [], []
        for i in tqdm(k_fold, desc='computing cf of train LMI'):
            g = graphs[0][i]
            neg_g = graphs[1][i]
            lnc_feat = g.nodes['lnc'].data['features'].float()
            dis_feat = g.nodes['dis'].data['features'].float()
            T_f_array, T_cf_array, T_f_1dim, T_cf_1dim, A_cf = calc_Tf_Tcf_Acf(g, neg_g, [lnc_feat, dis_feat])
            T_f_arrays.append(T_cf_array)
            T_cf_arrays.append(T_cf_array), T_f_1dims.append(T_f_1dim)
            T_cf_1dims.append(T_cf_1dim)
            A_cfs.append(A_cf)
        cf_dict = {"T_f_arrays": T_f_arrays, "T_cf_arrays": T_cf_arrays, "T_f_1dims": T_f_1dims,
                   "T_cf_1dims": T_cf_1dims, "A_cfs": A_cfs}
        # savemat('./lnc_mi/dataset/Training-validation dataset(PmliPEMG)/train_cfs.mat', cf_dict)
        savemat('./lnc_mi/dataset/Training-validation dataset(PmliPEMG)/train_cfs.mat', cf_dict)
    trained_models = []
    for i in k_fold:
        # set_random_seed(3407-i)
        # initialize model settings
        model = Model(num_meta_paths=4,
                      in_size=in_dims,
                      hidden_size=hid_dim,
                      out_size=out_dim,
                      num_heads=num_heads,
                      dropout=0.3)
        model = model.to(device)
        model.train()
        optimizer = Ranger(model.parameters(), lr=lr, weight_decay=weight_decay)
        cache_loss = []
        # 注意：在训练miRNA与lncRNA互作模型时，由于特征维度是相同的，所以每一折数据中只需要算一次反事实链接（导致特征纬度维度不同的原因是GIP）
        # 在单独训练物种LMI或者LDA时需要注释下面代码或者用参数
        # 如果不使用GIP， 此过程较耗时
        # if not with_GIP:
        # g = graphs[0][i].to(device)
        # neg_g = graphs[1][i].to(device)
        # lnc_feat = g.nodes['lnc'].data['features'].float().to(device)
        # dis_feat = g.nodes['dis'].data['features'].float().to(device)
        # start = time.perf_counter()
        # T_f_array, T_cf_array, T_f_1dim, T_cf_1dim, A_cf = calc_Tf_Tcf_Acf(g, neg_g, [lnc_feat, dis_feat])
        # end = time.perf_counter()
        # runTime = end - start
        # print(f'calculate the counterfactual treatment and adj successfully: {runTime}s')
        global val_labels
        pbar = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            model.train()
            # input heterogeneous graph with all meta-graphs
            # data preparing
            g = graphs[0][i].to(device)
            neg_g = graphs[1][i].to(device)
            lnc_feat = g.nodes['lnc'].data['features'].float().to(device)
            dis_feat = g.nodes['dis'].data['features'].float().to(device)
            # model train
            # forward: contain using encoder features embeddings and compute counter-factual links
            z, neg_z, l_emb, d_emb = model(g, neg_g, [lnc_feat, dis_feat], 'link')
            # 需要使用GIP特征的情况
            # if with_GIP:
            #     T_f_array, T_cf_array, T_f_1dim, T_cf_1dim, A_cf = calc_Tf_Tcf_Acf(g, neg_g, [l_emb, d_emb])
            T_f_array, T_cf_array, T_f_1dim, T_cf_1dim, A_cf = (
                torch.Tensor(T_f_arrays[i]), torch.Tensor(T_cf_arrays[i]), torch.Tensor(T_f_1dims[i]),
                torch.Tensor(T_cf_1dims[i]), torch.Tensor(A_cfs[i]))
            # using decoder gets logit_f and logit_cf
            multi_z = torch.cat((z.to(device), neg_z.to(device)), dim=0)
            # multi_z = multi_z.reshape(-1, 1)
            # print(multi_z.shape)
            z_f = torch.cat([multi_z, T_f_1dim.reshape(-1, 1).to(device)], dim=1)
            z_cf = torch.cat([multi_z, T_cf_1dim.reshape(-1, 1).to(device)], dim=1)
            # # remove RNAs simi networks
            # multi_z = multi_z.reshape(-1, 1)
            # z_f = torch.cat([multi_z, T_f_1dim.reshape(-1, 1).to(device)], dim=1)
            # z_cf = torch.cat([multi_z, T_cf_1dim.reshape(-1, 1).to(device)], dim=1)
            # Ablation study
            # remove cf
            # z_f = multi_z
            # z_cf = multi_z
            logit_f = model.decoder(z_f).reshape(-1, 1)
            logit_cf = model.decoder(z_cf).reshape(-1, 1)
            # compute distance discrepancy between distribution of factual and distribution of counter-factual
            nodepairs_f, nodepairs_cf = utils.sample_nodepairs(3000, T_f_array, T_cf_array)
            # labels preparing: contain f_labels cf_labels
            f_labels, cf_labels = get_labels(adj_cf=A_cf, pos_g=g, neg_g=neg_g)
            # calculate loss
            # backward
            # increase the weight of negative samples
            weight1 = 0.4 * torch.ones_like(logit_f).to(device)
            weight2 = 0.3 * torch.ones_like(logit_cf).to(device)
            pos_w_cf = (cf_labels.shape[0] - cf_labels.sum()) / cf_labels.sum()
            pos_edges, neg_edges = get_edges(g, neg_g, 'link')
            for k in range(pos_edges.shape[0], weight1.shape[0]):
                weight1[k] = 1 - weight1[k]
            for k in range(neg_edges.shape[0], weight2.shape[0]):
                weight2[k] = 1 - weight2[k]
            loss_f = F.binary_cross_entropy(torch.sigmoid(logit_f), f_labels.to(device))
            loss_cf = F.binary_cross_entropy(torch.sigmoid(logit_cf), cf_labels.to(device))
            # loss_f = F.binary_cross_entropy_with_logits(logit_f, f_labels.to(device), pos_weight=weight1)
            # loss_cf = F.binary_cross_entropy_with_logits(logit_cf, cf_labels.to(device), pos_weight=pos_w_cf)
            # kl loss of distance discrepancy
            # loss for factual and counter-factual and distance discrepancy
            loss_disc = utils.calc_disc('kl', l_emb, d_emb, nodepairs_f, nodepairs_cf)
            # multi-loss
            loss = loss_f + 2 * loss_cf + loss_disc
            # loss = loss_f + 2 * loss_cf
            # loss = loss_f
            cache_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_description(f'cv:{i},epoch:{epoch},loss:{loss.item():.5f}')
        pbar.close()

        # calculate individual treatment effect(ITE) and
        # observed averaged treatment effect(ATE_obs) and
        # estimated averaged treatment effect(ATE_est)
        # ATE_obs is used for answer the question of Q1:
        # "Does model sufficiently learn the ATE_obs derived from the counterfactual links?"
        # Bigger ATE indicates stronger causal relationship between the treatment and outcome, and vice versa.
        """
        
        """
        if if_save_model:
            torch.save(model.state_dict(), f'{save_model_path}/CF_HAN_model_{i}.pth')
        if show_loss:
            plt.plot([i for i in range(len(cache_loss))], [i.cpu().detach().numpy() for i in cache_loss],
                     alpha=0.5, linewidth=1, label='loss')
            plt.show()
        if show_eval:
            print("*===== single model evaluating =====*")
            single_te_pre = single_eval(i, k_fold, model, graphs)
            labels = torch.cat([torch.ones(graphs[2][k_fold[i]].num_edges(etype='link')),
                                torch.zeros(graphs[3][k_fold[i]].num_edges(etype='link'))]).reshape(-1, 1)
            metrics = main.calc_metrics(single_te_pre, labels, show_roc=True)
            print(metrics)
        trained_models.append(model)
    return trained_models


def model_eval(models, k_fold, graphs):
    if os.path.exists('./lnc_mi/dataset/Training-validation dataset(PmliPEMG)/test_cfs.mat'):
        te_pres = []
        test_cdf_dict = loadmat('./lnc_mi/dataset/Training-validation dataset(PmliPEMG)/test_cfs.mat')
        T_f_arrays, T_cf_arrays, T_f_1dims, T_cf_1dims, A_cfs \
            = (test_cdf_dict['T_f_arrays'], test_cdf_dict['T_cf_arrays'], test_cdf_dict['T_f_1dims'],
               test_cdf_dict['T_cf_1dims'], test_cdf_dict['A_cfs'])
        print('load test_cdf_dict successfully')
    else:
        T_f_arrays, T_cf_arrays, T_f_1dims, T_cf_1dims, A_cfs = [], [], [], [], []
        te_pres = []
        for i in tqdm(k_fold, desc='computing cf of test LMI'):
            g = graphs[2][k_fold[i]]
            neg_g = graphs[3][k_fold[i]]
            lnc_feat = g.nodes['lnc'].data['features'].float()
            dis_feat = g.nodes['dis'].data['features'].float()
            T_f_array, T_cf_array, T_f_1dim, T_cf_1dim, A_cf = calc_Tf_Tcf_Acf(g, neg_g, [lnc_feat, dis_feat])
            T_f_arrays.append(T_cf_array)
            T_cf_arrays.append(T_cf_array), T_f_1dims.append(T_f_1dim)
            T_cf_1dims.append(T_cf_1dim)
            A_cfs.append(A_cf)
        cf_dict = {"T_f_arrays": T_f_arrays, "T_cf_arrays": T_cf_arrays, "T_f_1dims": T_f_1dims,
                   "T_cf_1dims": T_cf_1dims,
                   "A_cfs": A_cfs}
        savemat('./lnc_mi/dataset/Training-validation dataset(PmliPEMG)/test_cfs.mat', cf_dict)
    for i in range(len(models)):
        model = models[i].to(device)
        model.eval()
        with torch.no_grad():
            te_g = graphs[2][k_fold[i]].to(device)
            te_neg_g = graphs[3][k_fold[i]].to(device)
            lnc_feat = te_g.nodes['lnc'].data['features'].float().to(device)
            dis_feat = te_g.nodes['dis'].data['features'].float().to(device)
            z, neg_z, l_emb, d_emb = model(te_g, te_neg_g, [lnc_feat, dis_feat], 'link')
            # T_f_array, T_cf_array, T_f_1dim, T_cf_1dim, A_cf = calc_Tf_Tcf_Acf(te_g, te_neg_g, [l_emb, d_emb])
            T_f_array, T_cf_array, T_f_1dim, T_cf_1dim, A_cf = (
                torch.Tensor(T_f_arrays[i]), torch.Tensor(T_cf_arrays[i]), torch.Tensor(T_f_1dims[i]),
                torch.Tensor(T_cf_1dims[i]), torch.Tensor(A_cfs[i]))
            T_f_1dim = T_f_1dim.reshape(-1, 1).to(device)
            multi_z = torch.cat((z, neg_z), dim=0).to(device)
            z_f = torch.cat([multi_z, T_f_1dim], dim=1)
            # z_f = multi_z
            te_logit_f = model.decoder(z_f).reshape(-1, 1)
        te_pres.append(te_logit_f)
    return te_pres
