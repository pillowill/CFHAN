# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/7 20:38
@Auth ： He Yu
@File ：main.py
@IDE ：PyCharm
@Function ：Function of the script
"""
# -*- coding: utf-8 -*-
import torch
import random
import train_eval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclass import LDdataset
from torchsummary import summary
from sklearn.metrics import roc_curve
from dgl.data.utils import load_graphs, save_graphs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, auc, confusion_matrix, matthews_corrcoef


# torch.autograd.set_detect_anomaly(True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_g2local(path, gs):
    save = ['tr_p', 'tr_n', 'te_p', 'te_n']
    for tr_te_g in range(len(save)):
        for pos_neg_g in range(5):
            save_graphs(f'{path}/{save[tr_te_g]}_{pos_neg_g}.dgl', [gs[tr_te_g][pos_neg_g]])
    print(f"Graphs data save to {path} successfully")


def get_data_g():
    dataset = LDdataset()
    hetero_graphs = dataset[0]
    lnc_feats = hetero_graphs[0][0].nodes['lnc'].data['features']
    lnc_feats_dims = lnc_feats.shape[1]
    dis_feats = hetero_graphs[0][0].nodes['dis'].data['features']
    dis_feats_dims = dis_feats.shape[1]

    return hetero_graphs, lnc_feats_dims, dis_feats_dims


def load_cache_g(path):
    # # 读入图数据
    tr_pos_graphs = []
    tr_neg_graphs = []
    te_pos_graphs = []
    te_neg_graphs = []
    for i in ['tr_p_', 'tr_n_', 'te_p_', 'te_n_']:
        for j in range(5):
            if i == 'tr_p_':
                tr_pos_graphs.append(load_graphs(f'./{path}{i}{j}.dgl')[0][0])
            elif i == 'tr_n_':
                tr_neg_graphs.append(load_graphs(f'./{path}{i}{j}.dgl')[0][0])
            elif i == 'te_p_':
                te_pos_graphs.append(load_graphs(f'./{path}{i}{j}.dgl')[0][0])
            else:
                te_neg_graphs.append(load_graphs(f'./{path}{i}{j}.dgl')[0][0])
    hetero_graphs = [tr_pos_graphs, tr_neg_graphs, te_pos_graphs, te_neg_graphs]
    lnc_feats = hetero_graphs[0][0].nodes['lnc'].data['features']
    lnc_feats_dims = lnc_feats.shape[1]
    dis_feats = hetero_graphs[0][0].nodes['dis'].data['features']
    dis_feats_dims = dis_feats.shape[1]

    return hetero_graphs, lnc_feats_dims, dis_feats_dims


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
    mcc = matthews_corrcoef(labels.cpu().numpy(), preds.cpu().numpy())
    if show_roc:
        draw_roc(torch.sigmoid(preds).cpu().numpy(), labels.cpu().numpy())
    return {'auc': my_auc,
            'aupr': aupr,
            'precision': prec,
            'recall': rec,
            'accuracy': acc,
            'f1': f1,
            'mcc': mcc
            }


if __name__ == '__main__':
    import models
    # metrics_df.iloc[:, :6].mean()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set_random_seed(seed=3407)
    # load data from path, return positive and negative of all train or test graphs
    # graphs_saved_path = 'graphs/mi_lnc/o.sa/'
    # hetero_graphs, lnc_dim, dis_dim = load_cache_g(path=graphs_saved_path)
    # print("load heterogeneous graphs successfully")
    hetero_graphs, lnc_dim, dis_dim = get_data_g()
    # save_g2local('./graphs/mi_lnc/homo', hetero_graphs)
    num_epochs = 100
    k = 5
    k_fold_idx = [i for i in range(k)]
    num_heads = [6, 6, 6, 6]
    # remove multi head
    # num_heads = [1, 1, 1, 1]
    # remove RNAs simi networks
    # num_heads = [7, 7]
    in_dims = [lnc_dim, dis_dim]
    print(in_dims)
    hid_dim = 128
    out_dim = 64
    lr = 0.001
    w_d = 0.001
    print("*===== model training =====*\n")
    trained_models = train_eval.model_train(
        in_dims=in_dims,
        hid_dim=hid_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        lr=lr,
        weight_decay=w_d,
        device=device,
        graphs=hetero_graphs,
        k_fold=k_fold_idx,
        num_epochs=num_epochs,
        show_eval=False,
        show_loss=False
    )

    print("*===== all model evaluating results =====\n*")
    # saved_models = []
    # control_condi = [[6, 6, 6, 6], 128, 64, 11]
    # state_dict = torch.load(
    #     f'./trained_model/mi_lnc/plant/no_scaler/HAN1layer_{control_condi[0][0]}_{control_condi[1]}_{control_condi[2]}_{control_condi[3]}.pth')
    # for i in range(5):
    #     model = models.Model(num_meta_paths=4,
    #                          in_size=in_dims,
    #                          hidden_size=hid_dim,
    #                          out_size=out_dim,
    #                          num_heads=num_heads,
    #                          dropout=0.3)
    #     model.load_state_dict(state_dict[f'model{i + 1}'])
    #     model.to(device)
    #     saved_models.append(model)
    metrics_dict = {}
    te_pres = train_eval.model_eval(trained_models, k_fold_idx, hetero_graphs)
    # te_pres = train_eval.model_eval(trained_models, k_fold_idx, hetero_graphs)
    for i in range(len(te_pres)):
        labels = torch.cat([torch.ones(hetero_graphs[2][k_fold_idx[i]].num_edges(etype='link')),
                            torch.zeros(hetero_graphs[3][k_fold_idx[i]].num_edges(etype='link'))]).reshape(-1, 1)
        metrics_dict[i] = calc_metrics(te_pres[i], labels)
    metrics_df = pd.DataFrame(metrics_dict).T
    print(f'your setting as follows:\n'
          f'lr:{lr},hid_dim:{hid_dim},out_dim:{out_dim},num_heads:{num_heads[0]},train_epochs:{num_epochs}')
    print(f'your models structure as follows:\n'
          f'{summary(trained_models[0])}')
    print("PmliPEMG数据集：")
    print(f'the 5-cv results as follows:\n'
          f'{metrics_df}')
    # if you do not want to save trained models, just exegesis the following code
    # torch.save({'model1': trained_models[0].state_dict(), 'model2': trained_models[1].state_dict(),
    #             'model3': trained_models[2].state_dict(), 'model4': trained_models[3].state_dict(),
    #             'model5': trained_models[4].state_dict()}, f'./trained_model/mi_lnc/O_sa/O_sa_models.pth')
    model_dict = {}
    for i in range(k):
        model_dict[f'model{i + 1}'] = trained_models[i].state_dict()
    torch.save(model_dict,
               f'./trained_model/mi_lnc/plant/no_scaler/1layer/5filtered0.6_{num_heads[0]}_128_64_11.pth')
