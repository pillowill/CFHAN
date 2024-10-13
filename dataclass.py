import os
import dgl
import torch
import cv_main
import pandas as pd
from dgl.data import DGLDataset
from collections import defaultdict
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import seaborn as sns
from sklearn import manifold

# from calc_feats import split_data
from sklearn.feature_selection import VarianceThreshold


def feat_selection(lnc_feats):
    # 计算相关系数矩阵
    lnc_corr = lnc_feats.corr(method='pearson')
    lnc_corr = lnc_corr.abs()

    # 定义 G-gap_seq 和 G-gap_str 特征的索引范围
    g_gap_seq_range = range(84, 132)
    g_gap_str_range = range(146, 194)
    th = 0.6
    # 找出与 G-gap_seq 和 G-gap_str 特征相关系数大于 0.6 的特征
    to_remove = set()
    for i in g_gap_seq_range:
        correlated_features = lnc_corr.columns[(lnc_corr.iloc[i, :] > th) & (lnc_corr.columns != lnc_corr.index[i])]
        to_remove.update(correlated_features)

    for i in g_gap_str_range:
        correlated_features = lnc_corr.columns[(lnc_corr.iloc[i, :] > th) & (lnc_corr.columns != lnc_corr.index[i])]
        to_remove.update(correlated_features)

    # 保留 G-gap_seq 和 G-gap_str 特征
    to_keep = set(g_gap_seq_range).union(g_gap_str_range)

    # 移除与 G-gap_seq 和 G-gap_str 特征相关系数大于 0.6 的特征，但保留 G-gap_seq 和 G-gap_str 特征本身
    final_features = [feat for feat in lnc_feats.columns if feat not in to_remove or feat in to_keep]

    # 过滤特征矩阵
    filtered_lnc_feats = lnc_feats[final_features]

    # 输出保留特征的索引
    kept_indices = [i for i in range(lnc_feats.shape[1]) if lnc_feats.columns[i] in final_features]
    return kept_indices


def ANONV(X):
    selector = VarianceThreshold(threshold=0.001)
    X_selected = selector.fit_transform(X)
    # 获取被选中特征的原始索引
    selected_indices = selector.get_support(indices=True)
    return selected_indices


class LDdataset(DGLDataset):
    def __init__(self):
        super(LDdataset, self).__init__(name='lddata')

    def process_files(self, root_path):
        """

        :param root_path: feats_root_path
        :param feats_dict:
        :return:
        """
        scaler = StandardScaler()
        feats_dict = defaultdict(list)
        for root, dirs, files in os.walk(root_path):
            for feat_type in dirs:
                for _, dirs1, files1 in os.walk(os.path.join(root_path, feat_type)):
                    for data_type in dirs1:
                        for _, dirs2, files2 in os.walk(os.path.join(root_path, feat_type + '/' + data_type)):
                            for pos_neg in dirs2:
                                for _, dirs3, files3 in os.walk(
                                        os.path.join(root_path, feat_type + '/' + data_type + '/' + pos_neg)):
                                    key = feat_type + '_' + data_type + '_' + pos_neg
                                    for file in files3:
                                        feats_dict[key].append(scaler.fit_transform(
                                            pd.read_csv(
                                                root_path + '/' + feat_type + '/' + data_type + '/' + pos_neg + '/' + file,
                                                header=None
                                            ).values
                                        ))
                                        # feats_dict[key].append(
                                        #     pd.read_csv(
                                        #         root_path + '/' + feat_type + '/' + data_type + '/' + pos_neg + '/' + file,
                                        #         header=None
                                        #     ).values
                                        # )
        return feats_dict

    def construct_hetero_g(self, link_arr, lnc_feat, dis_feat, k=11):
        """

        :param link_arr:
        :param lnc_feat: Tensor
        :param dis_feat: Tensor
        :param k:
        :return:
        """

        lnc_feat = torch.tensor(lnc_feat)
        dis_feat = torch.tensor(dis_feat)
        knn_lnc_g = dgl.knn_graph(lnc_feat, k=k, dist='cosine')
        ll_src, ll_dst = knn_lnc_g.edges()

        knn_dis_g = dgl.knn_graph(dis_feat, k=k, dist='cosine')
        dd_src, dd_dst = knn_dis_g.edges()
        # link_arr = np.nonzero(link_arr)
        hetero_g = dgl.heterograph({
            ('lnc', 'link', 'dis'): (
                torch.tensor(link_arr[:, 0]), torch.tensor(link_arr[:, 1])
            ),
            ('dis', 're_link', 'lnc'): (
                torch.tensor(link_arr[:, 1]), torch.tensor(link_arr[:, 0])
            ),
            ('lnc', 'cos_ll', 'lnc'): (
                ll_src, ll_dst
            ),
            ('dis', 'cos_dd', 'dis'): (
                dd_src, dd_dst
            )
        })
        # hetero_g = dgl.heterograph({
        #     ('lnc', 'link', 'dis'): (
        #         torch.tensor(link_arr[0]), torch.tensor(link_arr[1])
        #     ),
        #     ('dis', 're_link', 'lnc'): (
        #         torch.tensor(link_arr[1]), torch.tensor(link_arr[0])
        #     ),
        #     ('lnc', 'cos_ll', 'lnc'): (
        #         ll_src, ll_dst
        #     ),
        #     ('dis', 'cos_dd', 'dis'): (
        #         dd_src, dd_dst
        #     )
        # })
        # 获取dis节点个数
        num_dis_nodes = hetero_g.number_of_nodes('dis')
        num_lnc_nodes = hetero_g.number_of_nodes('lnc')
        # 判断dis节点个数是否小于412
        if num_dis_nodes < dis_feat.shape[0]:
            # 计算需要补充的dis节点个数
            num_dis_to_add = dis_feat.shape[0] - num_dis_nodes
            # 为dis节点补充到412个
            hetero_g.add_nodes(num_dis_to_add, ntype='dis')
        if num_lnc_nodes < lnc_feat.shape[0]:
            # 计算需要补充的dis节点个数
            num_lnc_to_add = lnc_feat.shape[0] - num_lnc_nodes
            # 为dis节点补充到412个
            hetero_g.add_nodes(num_lnc_to_add, ntype='lnc')
        hetero_g.nodes['lnc'].data['features'] = lnc_feat
        hetero_g.nodes['dis'].data['features'] = dis_feat
        return hetero_g

    def process(self):
        """
        1. 从正样本得到负样本
        2. 正负样本合并, 5k划分->每一折数据层级关系（共5折）: train->pos;neg   test->pos;neg
        3. 构建共20个异构图: train_pos_g 5个, train_neg_g 5个, test_pos_g 5个, test_neg_g 5个
        4. 若训练miRNA和lncRNA时，需要传入负样本矩阵
        :return:
        """
        # 以下是按照PmliPred的规则建立植物的miRNA-lncRNA interaction预测的数据输入
        # pos和neg提前分开，且提前计算好特征
        # train_test_path = './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/train_test.mat'
        # train_test = loadmat(train_test_path)
        # pos_adj = train_test['pos_adj']

        pos_adj_path = './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/PmliPEMG_pos_adj.csv'
        neg_adj_path = './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/PmliPEMG_neg_adj.csv'
        pos_mi_complex_path = './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/mi_feats.csv'
        pos_lnc_complex_path = './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/lnc_feats.csv'
        neg_mi_complex_path = './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/mi_feats.csv'
        neg_lnc_complex_path = './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/lnc_feats.csv'
        # pos_mi_complex_path = './lnc_mi/dataset/RANBert/train_mi_bert_emb_1024.npy'
        # pos_lnc_complex_path = './lnc_mi/dataset/RANBert/train_lnc_bert_emb_1024.npy'
        # neg_mi_complex_path = './lnc_mi/dataset/RNABert/train_mi_bert_emb_1024.npy'
        # neg_lnc_complex_path = './lnc_mi/dataset/RNABert/train_lnc_bert_emb_1024.npy'
        pos_adj = pd.read_csv(pos_adj_path, header=None)
        neg_adj = pd.read_csv(neg_adj_path, header=None)
        lnc_pca = PCA(n_components=72)
        mi_pca = PCA(n_components=83)
        pos_mi_complex = pd.read_csv(pos_mi_complex_path, header=None, index_col=0)
        pos_lnc_complex = pd.read_csv(pos_lnc_complex_path, header=None, index_col=0)
        filtered_lnc_idx = feat_selection(pos_lnc_complex)
        filtered_mi_idx = feat_selection(pos_mi_complex)
        # self.train_pos_g, self.train_neg_g, self.test_pos_g, self.test_neg_g = [], [], [], []
        # self.train_pos_g.append(self.construct_hetero_g(pos_adj,
        #                                                 lnc_feat=filtered_mi,
        #                                                 dis_feat=filtered_lnc))
        # self.train_neg_g.append(self.construct_hetero_g(neg_adj,
        #                                                 lnc_feat=filtered_mi,
        #                                                 dis_feat=filtered_lnc
        #                                                 ))
        # neg_mi_complex = pd.read_csv(neg_mi_complex_path, header=None)
        # neg_lnc_complex = pd.read_csv(neg_lnc_complex_path, header=None)
        # 以下是对单个物种分开进行训练与建模
        # mi_complex_path = './lnc_mi/Homo/complex_feats/mi_complex_feats.csv'
        # lnc_complex_path = './lnc_mi/Homo/complex_feats/lnc_complex_feats.csv'
        # feats_root_path = './calc_feats/相似度矩阵/multi_feats/'
        # dissim_path = './calc_feats/相似度矩阵/my_rice_traits_sim.23.11.17.csv'
        # link_info_file_path = './lnc_mi/Homo/homo_mi_lnc_adj.csv'
        # link_array = pd.read_csv(link_info_file_path, header=None)
        # link_array:
        # train_pos_data_list, train_neg_data_list,
        # test_pos_data_list, test_neg_data_list
        k = 5
        times_neg = 1
        # cv_link_array, cv_feat_dicts = cv_main.run(adj_m=link_array, dissim_file=dissim_path, times_neg=times_neg,
        # seed=12312, k=k)
        self.cv_link_array, self.cv_feat_dicts = cv_main.run_rice_cv_main(
            adj_m=pos_adj,
            mi_complex_file=pos_mi_complex_path,
            lnc_complex_file=pos_lnc_complex_path,
            times_neg=times_neg,
            seed=12312, k=k, neg_adj=neg_adj, neg_mi_complex_file=neg_mi_complex_path,
            neg_lnc_complex_file=neg_lnc_complex_path)
        # # cv_link_array = split_data.cv_split_data(link_array, 12312, 5)
        self.train_pos_g, self.train_neg_g, self.test_pos_g, self.test_neg_g = [], [], [], []
        # tr_p_dict = self.process_files(feats_root_path)
        # tr_n_dict = self.process_files(feats_root_path)
        # te_p_dict = self.process_files(feats_root_path)
        # te_n_dict = self.process_files(feats_root_path)

        # for i in range(len(self.cv_link_array[0])):
        #     self.train_pos_g.append(
        #         self.construct_hetero_g(self.cv_link_array[0][i],
        #                                 lnc_feat=mi_pca.fit_transform(self.cv_feat_dicts['lnc_train_pos'][i]),
        #                                 dis_feat=lnc_pca.fit_transform(self.cv_feat_dicts['dis_train_pos'][i]))
        #     )
        # for i in range(len(self.cv_link_array[1])):
        #     self.train_neg_g.append(
        #         self.construct_hetero_g(self.cv_link_array[1][i],
        #                                 lnc_feat=mi_pca.fit_transform(self.cv_feat_dicts['lnc_train_neg'][i]),
        #                                 dis_feat=lnc_pca.fit_transform(self.cv_feat_dicts['dis_train_neg'][i]))
        #     )
        # for i in range(len(self.cv_link_array[2])):
        #     self.test_pos_g.append(
        #         self.construct_hetero_g(self.cv_link_array[2][i],
        #                                 lnc_feat=mi_pca.fit_transform(self.cv_feat_dicts['lnc_test_pos'][i]),
        #                                 dis_feat=lnc_pca.fit_transform(self.cv_feat_dicts['dis_test_pos'][i]))
        #     )
        # for i in range(len(self.cv_link_array[3])):
        #     self.test_neg_g.append(
        #         self.construct_hetero_g(self.cv_link_array[3][i],
        #                                 lnc_feat=mi_pca.fit_transform(self.cv_feat_dicts['lnc_test_neg'][i]),
        #                                 dis_feat=lnc_pca.fit_transform(self.cv_feat_dicts['dis_test_neg'][i]))
        #     )
        # all_feat = pd.concat([pos_lnc_complex, pos_mi_complex])
        # var_selected = ANONV(all_feat)
        for i in range(len(self.cv_link_array[0])):
            self.train_pos_g.append(
                self.construct_hetero_g(self.cv_link_array[0][i],
                                        lnc_feat=self.cv_feat_dicts['lnc_train_pos'][i][:, filtered_mi_idx],
                                        dis_feat=self.cv_feat_dicts['dis_train_pos'][i][:, filtered_lnc_idx])
            )
        for i in range(len(self.cv_link_array[1])):
            self.train_neg_g.append(
                self.construct_hetero_g(self.cv_link_array[1][i],
                                        lnc_feat=self.cv_feat_dicts['lnc_train_neg'][i][:, filtered_mi_idx],
                                        dis_feat=self.cv_feat_dicts['dis_train_neg'][i][:, filtered_lnc_idx])
            )
        for i in range(len(self.cv_link_array[2])):
            self.test_pos_g.append(
                self.construct_hetero_g(self.cv_link_array[2][i],
                                        lnc_feat=self.cv_feat_dicts['lnc_test_pos'][i][:, filtered_mi_idx],
                                        dis_feat=self.cv_feat_dicts['dis_test_pos'][i][:, filtered_lnc_idx])
            )
        for i in range(len(self.cv_link_array[3])):
            self.test_neg_g.append(
                self.construct_hetero_g(self.cv_link_array[3][i],
                                        lnc_feat=self.cv_feat_dicts['lnc_test_neg'][i][:, filtered_mi_idx],
                                        dis_feat=self.cv_feat_dicts['dis_test_neg'][i][:, filtered_lnc_idx])
            )

    def __getitem__(self, idx):
        return self.train_pos_g, self.train_neg_g, self.test_pos_g, self.test_neg_g

    def __len__(self):
        return 1


def pca4feats(feats):
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(feats)
    explained_variance_ratios = []
    components = range(1, 100)

    for n_components in components:
        pca = PCA(n_components=n_components)
        pca.fit(scaled_feats)
        explained_variance_ratios.append(np.sum(pca.explained_variance_ratio_))

    num_components_95 = np.argmax(np.array(explained_variance_ratios) >= 0.85) + 1
    return explained_variance_ratios, num_components_95


def plot_explained_variance(lnc_ratios, mi_ratios, lnc_num_components_95, mi_num_components_95):
    components = range(1, 100)
    plt.figure(figsize=(10, 6))
    plt.axhline(y=0.85, color='r', linestyle='--', label='0.85 Threshold')
    plt.axvline(x=lnc_num_components_95, color='g', linestyle='--', label=f'lncRNA Components: {lnc_num_components_95}')
    plt.axvline(x=mi_num_components_95, color='b', linestyle='--', label=f'miRNA Components: {mi_num_components_95}')
    plt.plot(components, lnc_ratios, label='lncRNA Cumulative Explained Variance Ratio')
    plt.plot(components, mi_ratios, label='miRNA Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Principal Components', fontsize=14)
    plt.ylabel('Cumulative Explained Variance Ratio', fontsize=14)
    # plt.title('Explained variance ratios for lnc and mi')
    plt.legend(fontsize=12)
    plt.xticks(list(range(0, 101, 25)) + [lnc_num_components_95, mi_num_components_95], fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'./实验图/lnc_mi_pca_85.png', dpi=600)
    plt.show()


def feat_compare():
    data = {
        'Test set': ['B. distachyon', 'M. truncatula', 'S. tuberosum'] * 4,
        'Combination': ['K-mer+G-gap'] * 3 + ['K-mer+CTD'] * 3 + ['G-gap+CTD'] * 3 + ['K-mer+G-gap+CTD'] * 3,
        'AUC': [0.99118, 0.99463, 0.99457, 0.99055, 0.9942, 0.99498, 0.99178, 0.99304, 0.9959, 0.99537, 0.99384,
                0.99532],
        'Acc': [0.97595, 0.9752, 0.9784, 0.97174, 0.9712, 0.9736, 0.97114, 0.973, 0.9724, 0.97836, 0.9782, 0.971],
    }

    df_new = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'lines.linewidth': 2})
    # Subplot for AUC
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Combination', y='AUC', data=df_new, palette='Set2')
    plt.xlabel('Combination', fontsize=14)
    plt.ylabel('AUC', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    # Subplot for Precision
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Combination', y='Acc', data=df_new, palette='Set2')
    plt.xlabel('Combination', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig('./实验图/feature_comparison.png', dpi=600)
    plt.show()


def feat_corr(feat, type):
    lnc_feats = feat
    lnc_corr = lnc_feats.corr(method='pearson')

    plt.figure(figsize=(8, 6))
    sns.heatmap(lnc_corr, cmap='coolwarm', center=0)

    # 创建掩码，掩盖上三角部分
    mask = np.triu(np.ones_like(lnc_corr, dtype=bool))

    # 设置绘图大小和热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(lnc_corr, mask=mask, cmap='coolwarm', center=0, annot=False, fmt=".2f", cbar_kws={"shrink": .5})

    # 自定义 x 轴标签
    labels = [''] * lnc_corr.shape[1]
    interval_labels = {
        (0, 84): 'K-mer_seq',
        (84, 132): 'G-gap_seq',
        (132, 146): 'K-mer_str',
        (146, 194): 'G-gap_str',
        (194, 224): 'CTD'
    }

    for (start, end), label in interval_labels.items():
        mid = (start + end) // 2
        labels[mid] = label
        plt.axvline(x=end + 0.5, color='black', linestyle='--')
        plt.axhline(y=end + 0.5, color='black', linestyle='--')

    plt.xticks(ticks=range(lnc_corr.shape[1]), labels=labels, rotation=0)
    plt.yticks(ticks=range(lnc_corr.shape[1]), labels=labels, rotation=0)
    plt.ylabel('Dimension', fontsize=14)

    # 在上三角部分添加相关系数值
    for i in range(lnc_corr.shape[0]):
        for j in range(i + 1, lnc_corr.shape[1]):
            plt.text(j + 0.5, i + 0.5, f'{lnc_corr.iloc[i, j]:.2f}',
                     ha='center', va='center', color='black', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'./实验图/{type}_corr.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    pass
    # data = LDdataset()
    pos_mi_complex_path = './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/mi_feats.csv'
    pos_lnc_complex_path = './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/lnc_feats.csv'

    lnc_feats = pd.read_csv(pos_lnc_complex_path, header=None, index_col=0)
    mi_feats = pd.read_csv(pos_mi_complex_path, header=None, index_col=0)
    # # # a = LDdataset()
    # # feat_compare()
    # lnc_ratios, lnc_num_components_95 = pca4feats(lnc_feats)
    # mi_ratios, mi_num_components_95 = pca4feats(mi_feats)
    #
    # plot_explained_variance(lnc_ratios, mi_ratios, lnc_num_components_95, mi_num_components_95)
    # # lnc_ratios, lnc_num_components_85 = pca4feats(lnc_feats)
    # # mi_ratios, mi_num_components_85 = pca4feats(mi_feats)
    # #
    # # plot_explained_variance(lnc_ratios, mi_ratios, lnc_num_components_85, mi_num_components_85)

    # kmer = pd.read_csv(
    #     './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/kmer_mi_feats.csv', header=None, index_col=0
    # )
    # ggap = pd.read_csv(
    #     './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/ggap_mi_feats.csv', header=None, index_col=0
    # )
    # ctd = pd.read_csv(
    #     './lnc_mi/dataset/Training-validation dataset(PmliPEMG)/ctd_mi_feats.csv', header=None, index_col=0
    # )
    # t_SNE(kmer, ggap, ctd)
    # feat_corr(lnc_feats, 'lnc')
    # feat_corr(mi_feats, 'mi')

    # import pandas as pd
    # import matplotlib.pyplot as plt
    #
    # # Create a dictionary to store the data
    data = {
        'Feature Combination': ['K-mer'] * 15 + ['G-gap'] * 15 + ['CTD'] * 15 + ['K-mer+G-gap'] * 15 + [
            'K-mer+CTD'] * 15 + ['G-gap+CTD'] * 15 + ['De-redundancy']*15 + ['K-mer+G-gap+CTD'] * 15,
        'Acc':
            [0.9709418837675351, 0.9739478957915831, 0.9909819639278558, 0.9769539078156313, 0.9869739478957916, 0.987,
             0.974, 0.99, 0.983, 0.987, 0.963, 0.968, 0.977, 0.975, 0.979] +
            [0.9829659318637275, 0.9719438877755511, 0.9899799599198397, 0.9729458917835672, 0.9829659318637275, 0.982,
             0.981, 0.985, 0.974, 0.988, 0.983, 0.961, 0.98, 0.98, 0.98] +
            [0.9559, 0.9649, 0.9729, 0.9679, 0.9569, 0.968, 0.966, 0.973, 0.96, 0.96, 0.961, 0.964, 0.973, 0.962,
             0.957] +
            [0.9849699398797596, 0.9559118236472945, 0.9829659318637275, 0.9739478957915831, 0.9579158316633266] + [
                0.976, 0.959, 0.978, 0.983, 0.969] + [0.98, 0.966, 0.979, 0.969, 0.968] +
            [0.9749498997995992, 0.9749498997995992, 0.9789579158316634, 0.9729458917835672, 0.9779559118236473] + [
                0.979, 0.971, 0.979, 0.971, 0.976] + [0.977, 0.975, 0.979, 0.982, 0.979] +
            [0.9729458917835672, 0.9719438877755511, 0.9679358717434869, 0.9739478957915831, 0.9719438877755511] + [
                0.968, 0.979, 0.967, 0.973, 0.969] + [0.985, 0.968, 0.972, 0.973, 0.97] +
            [0.9739478957915831, 0.9869739478957916, 0.9729458917835672, 0.9859719438877755, 0.9759519038076152, 0.987,
             0.992, 0.972, 0.987, 0.976, 0.97, 0.99, 0.979, 0.984, 0.982] +
            [0.9779559118236473, 0.966933867735471, 0.9849699398797596, 0.9819639278557114, 0.9799599198396793] + [
                0.979, 0.966, 0.979, 0.979, 0.988] + [0.981, 0.959, 0.964, 0.977, 0.974]
    }
    df_new = pd.DataFrame(data)
    plt.figure(figsize=(14, 11))
    plt.rcParams.update({'font.size': 16})
    sns.boxplot(x='Feature Combination', y='Acc', data=df_new, palette='Set2', width=0.5)
    # plt.xlabel('Combination', fontsize=14)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xticks(rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y')
    # plt.savefig('./实验图/feature_comparison.png', dpi=600)
    plt.show()
    # import matplotlib.pyplot as plt
    # import pandas as pd
    #
    # # [0.97595, 0.9752, 0.9784, 0.97174, 0.9712, 0.9736, 0.97114, 0.973, 0.9724, 0.97836, 0.9782, 0.971]
    # # Data preparation
    # data = {
    #     'Test set': ['B. distachyon', 'M. truncatula', 'S. tuberosum'],
    #     'kmer+ggap': [0.97595, 0.9752, 0.9784],
    #     'kmer+ctd': [0.97174, 0.9712, 0.9736],
    #     'ggap+ctd': [ 0.97114, 0.973, 0.9724],
    #     'Original Features': [0.97836, 0.9782, 0.971]
    # }
    #
    # df = pd.DataFrame(data)
    # df.set_index('Test set', inplace=True)
    #
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(df.index, df['kmer+ggap'], marker='o', label='kmer+ggap')
    # plt.plot(df.index, df['kmer+ctd'], marker='o', label='kmer+ctd')
    # plt.plot(df.index, df['ggap+ctd'], marker='o', label='ggap+ctd')
    # plt.plot(df.index, df['Original Features'], marker='o', label='Original Features')
    #
    # plt.title('Accuracy Comparison for Different Feature Combinations')
    # plt.xlabel('Test set')
    # plt.ylabel('Accuracy')
    # plt.ylim(0.97, 0.98)
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    #
    # # Save the plot to a file
    # # plt.savefig('/mnt/data/feature_importance_comparison.png')
    #
    # plt.show()
