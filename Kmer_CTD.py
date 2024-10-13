import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count
import time
from Bio import SeqIO
from tqdm import tqdm


def CTD(seq):
    n = float(len(seq))
    num_A, num_T, num_G, num_C = 0.0, 0.0, 0.0, 0.0
    AT_trans, AG_trans, AC_trans, TG_trans, TC_trans, GC_trans = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(seq) - 1):
        if seq[i] == "A":
            num_A = num_A + 1
        if seq[i] == "U":
            num_T = num_T + 1
        if seq[i] == "G":
            num_G = num_G + 1
        if seq[i] == "C":
            num_C = num_C + 1
        if (seq[i] == "A" and seq[i + 1] == "U") or (seq[i] == "U" and seq[i + 1] == "A"):
            AT_trans = AT_trans + 1
        if (seq[i] == "A" and seq[i + 1] == "G") or (seq[i] == "G" and seq[i + 1] == "A"):
            AG_trans = AG_trans + 1
        if (seq[i] == "A" and seq[i + 1] == "C") or (seq[i] == "C" and seq[i + 1] == "A"):
            AC_trans = AC_trans + 1
        if (seq[i] == "U" and seq[i + 1] == "G") or (seq[i] == "G" and seq[i + 1] == "U"):
            TG_trans = TG_trans + 1
        if (seq[i] == "U" and seq[i + 1] == "C") or (seq[i] == "C" and seq[i + 1] == "U"):
            TC_trans = TC_trans + 1
        if (seq[i] == "G" and seq[i + 1] == "C") or (seq[i] == "C" and seq[i + 1] == "G"):
            GC_trans = GC_trans + 1

    a, t, g, c = 0, 0, 0, 0
    A0_dis, A1_dis, A2_dis, A3_dis, A4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
    T0_dis, T1_dis, T2_dis, T3_dis, T4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
    G0_dis, G1_dis, G2_dis, G3_dis, G4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
    C0_dis, C1_dis, C2_dis, C3_dis, C4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(seq) - 1):
        if seq[i] == "A":
            a = a + 1
            if a == 1:
                A0_dis = ((i * 1.0) + 1) / n
            if a == int(round(num_A / 4.0)):
                A1_dis = ((i * 1.0) + 1) / n
            if a == int(round(num_A / 2.0)):
                A2_dis = ((i * 1.0) + 1) / n
            if a == int(round((num_A * 3 / 4.0))):
                A3_dis = ((i * 1.0) + 1) / n
            if a == num_A:
                A4_dis = ((i * 1.0) + 1) / n
        if seq[i] == "U":
            t = t + 1
            if t == 1:
                T0_dis = ((i * 1.0) + 1) / n
            if t == int(round(num_T / 4.0)):
                T1_dis = ((i * 1.0) + 1) / n
            if t == int(round((num_T / 2.0))):
                T2_dis = ((i * 1.0) + 1) / n
            if t == int(round((num_T * 3 / 4.0))):
                T3_dis = ((i * 1.0) + 1) / n
            if t == num_T:
                T4_dis = ((i * 1.0) + 1) / n
        if seq[i] == "G":
            g = g + 1
            if g == 1:
                G0_dis = ((i * 1.0) + 1) / n
            if g == int(round(num_G / 4.0)):
                G1_dis = ((i * 1.0) + 1) / n
            if g == int(round(num_G / 2.0)):
                G2_dis = ((i * 1.0) + 1) / n
            if g == int(round(num_G * 3 / 4.0)):
                G3_dis = ((i * 1.0) + 1) / n
            if g == num_G:
                G4_dis = ((i * 1.0) + 1) / n
        if seq[i] == "C":
            c = c + 1
            if c == 1:
                C0_dis = ((i * 1.0) + 1) / n
            if c == int(round(num_C / 4.0)):
                C1_dis = ((i * 1.0) + 1) / n
            if c == int(round(num_C / 2.0)):
                C2_dis = ((i * 1.0) + 1) / n
            if c == int(round(num_C * 3 / 4.0)):
                C3_dis = ((i * 1.0) + 1) / n
            if c == num_C:
                C4_dis = ((i * 1.0) + 1) / n
    return [num_A / n, num_T / n, num_G / n, num_C / n,
            AT_trans / n - 1, AG_trans / (n - 1), AC_trans / (n - 1),
            TG_trans / n - 1, TC_trans / (n - 1), GC_trans / (n - 1),
            A0_dis, A1_dis, A2_dis, A3_dis, A4_dis,
            T0_dis, T1_dis, T2_dis, T3_dis, T4_dis,
            G0_dis, G1_dis, G2_dis, G3_dis, G4_dis,
            C0_dis, C1_dis, C2_dis, C3_dis, C4_dis]


def k_mer(seq):
    def get_1mer(seq):
        A_count = seq.count("A")
        T_count = seq.count("U")
        C_count = seq.count("C")
        G_count = seq.count("G")
        return [A_count / len(seq), T_count / len(seq), C_count / len(seq), G_count / len(seq)]

    def get_2mer(seq):
        res_dict = {}
        for x in "AUCG":
            for y in "AUCG":
                k = x + y
                res_dict[k] = 0
        i = 0
        while i + 2 < len(seq):
            k = seq[i:i + 2]
            i = i + 1
            res_dict[k] = res_dict[k] + 1
        return [x / len(seq) for x in list(res_dict.values())]

    def get_3mer(seq):
        res_dict = {}
        for x in "AUCG":
            for y in "AUCG":
                for z in "AUCG":
                    k = x + y + z
                    res_dict[k] = 0
        i = 0
        while i + 3 < len(seq):
            k = seq[i:i + 3]
            i = i + 1
            res_dict[k] = res_dict[k] + 1
        return [x / len(seq) for x in list(res_dict.values())]

    def get_4mer(seq):
        res_dict = {}
        for x in "AUCG":
            for y in "AUCG":
                for z in "AUCG":
                    for p in "AUCG":
                        k = x + y + z + p
                        res_dict[k] = 0
        i = 0
        while i + 4 < len(seq):
            k = seq[i:i + 4]
            i = i + 1
            res_dict[k] = res_dict[k] + 1
        return [x / len(seq) for x in list(res_dict.values())]

    return get_1mer(seq) + get_2mer(seq) + get_3mer(seq) + get_4mer(seq)


def segment(seq):
    res = []
    i = 0
    while i + 3 < len(seq):
        tmp = seq[i:i + 3]
        res.append(tmp)
        i = i + 1
    return res


def read_fa(path):
    res = {}
    rescords = list(SeqIO.parse(path, format="fasta"))
    for x in rescords:
        id = str(x.id)
        seq = str(x.seq).replace("U", "T").replace("N", "")
        res[id] = seq
    return res


def to_dict(seq_dict, feature_list):
    res_dict = {}
    for i, k in enumerate(list(seq_dict.keys())):
        res_dict[k] = feature_list[i]
    return res_dict


def save_dict(x_dict, path):
    f = open(path, "w")
    for k, v in x_dict.items():
        tmp = k + "," + ",".join([str(x) for x in v])
        f.write(tmp + "\n")
    f.close()


def load_dict(path):
    lines = open(path, "r").readlines()
    res = {}
    for line in lines:
        x_list = line.strip().split(",")
        id = str(x_list[0])
        vec = [np.float(x) for x in x_list[1:]]
        res[id] = vec
    return res


if __name__ == '__main__':
    seq = 'ATACGTATCTGCTGACGTAGC'.replace('T', 'U')
    # mirna_dict = read_fa("./test_homo_miRNA.fa")
    # pool = Pool(cpu_count())
    # print("miRNA")
    # # mirna_ctds = pool.map(CTD, list(mirna_dict.values()))
    # # mir_ctd_dict = to_dict(mirna_dict, mirna_ctds)
    # # mirna_kmers = pool.map(k_mer, list(mirna_dict.values()))
    # # mir_kmer_dict = to_dict(mirna_dict, mirna_kmers)
    # mirna_ctds = [CTD(seq) for seq in tqdm(mirna_dict.values())]
    # mir_ctd_dict = to_dict(mirna_dict, mirna_ctds)
    #
    # mirna_kmers = [k_mer(seq) for seq in tqdm(mirna_dict.values())]
    # mir_kmer_dict = to_dict(mirna_dict, mirna_kmers)
