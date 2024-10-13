# Data process

import numpy as np
import math
import pandas as pd
from Path import PathInput, PathOutput
import RNA
from tqdm import tqdm

separator, Sequencekmertotal, SequenceGgaptotal, Structurekmertotal, StructureGgaptotal = ' ', 3, 3, 3, 3


def SequencekmerExtract(sequence, totalkmer):
    sequence = sequence.replace('U', 'T')
    character = 'ATCG'
    sequencekmer = ''
    for k in range(totalkmer):
        kk = k + 1
        sk = len(sequence) - kk + 1
        wk = 1 / (4 ** (totalkmer - kk))
        # 1-mer
        if kk == 1:
            for char11 in character:
                s1 = char11
                f1 = wk * sequence.count(s1) / sk
                string1 = str(f1) + separator
                sequencekmer = sequencekmer + string1
        # 2-mer
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    s2 = char21 + char22
                    numkmer2 = 0
                    for lkmer2 in range(len(sequence) - kk + 1):
                        if sequence[lkmer2] == s2[0] and sequence[lkmer2 + 1] == s2[1]:
                            numkmer2 = numkmer2 + 1
                    f2 = wk * numkmer2 / sk
                    string2 = str(f2) + separator
                    sequencekmer = sequencekmer + string2
        # 3-mer
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    for char33 in character:
                        s3 = char31 + char32 + char33
                        numkmer3 = 0
                        for lkmer3 in range(len(sequence) - kk + 1):
                            if sequence[lkmer3] == s3[0] and sequence[lkmer3 + 1] == s3[1] and sequence[lkmer3 + 2] == \
                                    s3[2]:
                                numkmer3 = numkmer3 + 1
                        f3 = wk * numkmer3 / sk
                        string3 = str(f3) + separator
                        sequencekmer = sequencekmer + string3
        # 4-mer
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    for char43 in character:
                        for char44 in character:
                            s4 = char41 + char42 + char43 + char44
                            numkmer4 = 0
                            for lkmer4 in range(len(sequence) - kk + 1):
                                if sequence[lkmer4] == s4[0] and sequence[lkmer4 + 1] == s4[1] and sequence[
                                    lkmer4 + 2] == s4[2] and sequence[lkmer4 + 3] == s4[3]:
                                    numkmer4 = numkmer4 + 1
                            f4 = wk * numkmer4 / sk
                            string4 = str(f4) + separator
                            sequencekmer = sequencekmer + string4
        # 5-mer
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    for char53 in character:
                        for char54 in character:
                            for char55 in character:
                                s5 = char51 + char52 + char53 + char54 + char55
                                numkmer5 = 0
                                for lkmer5 in range(len(sequence) - kk + 1):
                                    if sequence[lkmer5] == s5[0] and sequence[lkmer5 + 1] == s5[1] and sequence[
                                        lkmer5 + 2] == s5[2] and sequence[lkmer5 + 3] == s5[3] and sequence[
                                        lkmer5 + 4] == s5[4]:
                                        numkmer5 = numkmer5 + 1
                                f5 = wk * numkmer5 / sk
                                string5 = str(f5) + separator
                                sequencekmer = sequencekmer + string5
        # 6-mer
        if kk == 6:
            for char61 in character:
                for char62 in character:
                    for char63 in character:
                        for char64 in character:
                            for char65 in character:
                                for char66 in character:
                                    s6 = char61 + char62 + char63 + char64 + char65 + char66
                                    numkmer6 = 0
                                    for lkmer6 in range(len(sequence) - kk + 1):
                                        if sequence[lkmer6] == s6[0] and sequence[lkmer6 + 1] == s6[1] and sequence[
                                            lkmer6 + 2] == s6[2] and sequence[lkmer6 + 3] == s6[3] and sequence[
                                            lkmer6 + 4] == s6[4] and sequence[lkmer6 + 5] == s6[5]:
                                            numkmer6 = numkmer6 + 1
                                    f6 = wk * numkmer6 / sk
                                    string6 = str(f6) + separator
                                    sequencekmer = sequencekmer + string6
    return sequencekmer


def SequenceGgapExtract(sequence, totalGgap):
    sequence = sequence.replace('U', 'T')
    character = 'ATCG'
    sequenceGgap = ''
    for k in range(totalGgap):
        kk = k + 1
        sk = len(sequence) - kk + 1
        wk = 1 / (4 ** (totalGgap - kk))
        if kk == 1:
            for char11 in character:
                for char12 in character:
                    num1 = 0
                    for l1 in range(len(sequence) - kk - 1):
                        if sequence[l1] == char11 and sequence[l1 + kk + 1] == char12:
                            num1 = num1 + 1
                    f1 = wk * num1 / sk
                    string1 = str(f1) + separator
                    sequenceGgap = sequenceGgap + string1
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    num2 = 0
                    for l2 in range(len(sequence) - kk - 3):
                        if sequence[l2] == char21 and sequence[l2 + kk + 1] == char22:
                            num2 = num2 + 1
                    f2 = wk * num2 / sk
                    string2 = str(f2) + separator
                    sequenceGgap = sequenceGgap + string2
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    num3 = 0
                    for l3 in range(len(sequence) - kk - 3):
                        if sequence[l3] == char31 and sequence[l3 + kk + 1] == char32:
                            num3 = num3 + 1
                    f3 = wk * num3 / sk
                    string3 = str(f3) + separator
                    sequenceGgap = sequenceGgap + string3
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    num4 = 0
                    for l4 in range(len(sequence) - kk - 3):
                        if sequence[l4] == char41 and sequence[l4 + kk + 1] == char42:
                            num4 = num4 + 1
                    f4 = wk * num4 / sk
                    string4 = str(f4) + separator
                    sequenceGgap = sequenceGgap + string4
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    num5 = 0
                    for l5 in range(len(sequence) - kk - 3):
                        if sequence[l5] == char51 and sequence[l5 + kk + 1] == char52:
                            num5 = num5 + 1
                    f5 = wk * num5 / sk
                    string5 = str(f5) + separator
                    sequenceGgap = sequenceGgap + string5
    return sequenceGgap


def StructurekmerExtract(structure, totalkmer):
    character = ').'
    structurekmer = ''  # 特征

    sp = structure.split()
    ssf = sp[0]
    ssf = ssf.replace('(', ')')
    for k in range(totalkmer):
        kk = k + 1
        sk = len(ssf) - kk + 1
        wk = 1 / (2 ** (totalkmer - kk))
        # 1-mer
        if kk == 1:
            for char11 in character:
                s1 = char11
                f1 = wk * ssf.count(s1) / sk
                string1 = str(f1) + separator
                structurekmer = structurekmer + string1
        # 2-mer
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    s2 = char21 + char22
                    numkmer2 = 0
                    for lkmer2 in range(len(ssf) - kk + 1):
                        if ssf[lkmer2] == s2[0] and ssf[lkmer2 + 1] == s2[1]:
                            numkmer2 = numkmer2 + 1
                    f2 = wk * numkmer2 / sk
                    string2 = str(f2) + separator
                    structurekmer = structurekmer + string2
        # 3-mer
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    for char33 in character:
                        s3 = char31 + char32 + char33
                        numkmer3 = 0
                        for lkmer3 in range(len(ssf) - kk + 1):
                            if ssf[lkmer3] == s3[0] and ssf[lkmer3 + 1] == s3[1] and ssf[lkmer3 + 2] == s3[2]:
                                numkmer3 = numkmer3 + 1
                        f3 = wk * numkmer3 / sk
                        string3 = str(f3) + separator
                        structurekmer = structurekmer + string3
        # 4-mer
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    for char43 in character:
                        for char44 in character:
                            s4 = char41 + char42 + char43 + char44
                            numkmer4 = 0
                            for lkmer4 in range(len(ssf) - kk + 1):
                                if ssf[lkmer4] == s4[0] and ssf[lkmer4 + 1] == s4[1] and ssf[lkmer4 + 2] == s4[2] and \
                                        ssf[lkmer4 + 3] == s4[3]:
                                    numkmer4 = numkmer4 + 1
                            f4 = wk * numkmer4 / sk
                            string4 = str(f4) + separator
                            structurekmer = structurekmer + string4
        # 5-mer
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    for char53 in character:
                        for char54 in character:
                            for char55 in character:
                                s5 = char51 + char52 + char53 + char54 + char55
                                numkmer5 = 0
                                for lkmer5 in range(len(ssf) - kk + 1):
                                    if ssf[lkmer5] == s5[0] and ssf[lkmer5 + 1] == s5[1] and ssf[lkmer5 + 2] == s5[
                                        2] and ssf[lkmer5 + 3] == s5[3] and ssf[lkmer5 + 4] == s5[4]:
                                        numkmer5 = numkmer5 + 1
                                f5 = wk * numkmer5 / sk
                                string5 = str(f5) + separator
                                structurekmer = structurekmer + string5
    return structurekmer


def StructureGgapExtract(structure, totalGgap):
    character = ').'
    structureGgap = ''
    sp = structure.split()
    ssf = sp[0]
    ssf = ssf.replace('(', ')')
    for k in range(totalGgap):
        kk = k + 1
        sk = len(ssf) - kk + 1
        wk = 1 / (2 ** (totalGgap - kk))
        if kk == 1:
            for char11 in character:
                for char12 in character:
                    for char13 in character:
                        for char14 in character:
                            num1 = 0
                            for l1 in range(len(ssf) - kk - 3):
                                if ssf[l1] == char11 and ssf[l1 + 1] == char12 and ssf[l1 + kk + 2] == char13 and ssf[
                                    l1 + kk + 3] == char14:
                                    num1 = num1 + 1
                            f1 = wk * num1 / sk
                            string1 = str(f1) + separator
                            structureGgap = structureGgap + string1
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    for char23 in character:
                        for char24 in character:
                            num2 = 0
                            for l2 in range(len(ssf) - kk - 3):
                                if ssf[l2] == char21 and ssf[l2 + 1] == char22 and ssf[l2 + kk + 2] == char23 and ssf[
                                    l2 + kk + 3] == char24:
                                    num2 = num2 + 1
                            f2 = wk * num2 / sk
                            string2 = str(f2) + separator
                            structureGgap = structureGgap + string2
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    for char33 in character:
                        for char34 in character:
                            num3 = 0
                            for l3 in range(len(ssf) - kk - 3):
                                if ssf[l3] == char31 and ssf[l3 + 1] == char32 and ssf[l3 + kk + 2] == char33 and ssf[
                                    l3 + kk + 3] == char34:
                                    num3 = num3 + 1
                            f3 = wk * num3 / sk
                            string3 = str(f3) + separator
                            structureGgap = structureGgap + string3
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    for char43 in character:
                        for char44 in character:
                            num4 = 0
                            for l4 in range(len(ssf) - kk - 3):
                                if ssf[l4] == char41 and ssf[l4 + 1] == char42 and ssf[l4 + kk + 2] == char43 and ssf[
                                    l4 + kk + 3] == char44:
                                    num4 = num4 + 1
                            f4 = wk * num4 / sk
                            string4 = str(f4) + separator
                            structureGgap = structureGgap + string4
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    for char53 in character:
                        for char54 in character:
                            num5 = 0
                            for l5 in range(len(ssf) - kk - 3):
                                if ssf[l5] == char51 and ssf[l5 + 1] == char52 and ssf[l5 + kk + 2] == char53 and ssf[
                                    l5 + kk + 3] == char54:
                                    num5 = num5 + 1
                            f5 = wk * num5 / sk
                            string5 = str(f5) + separator
                            structureGgap = structureGgap + string5
    return structureGgap


def calc_single_complex_feats(rna_seq, rna_struc):
    seq_kmer = SequencekmerExtract(rna_seq, Sequencekmertotal)
    # lncRNA sequence g-gap
    seq_Ggap = SequenceGgapExtract(rna_seq, SequenceGgaptotal)
    # lncRNA structure k-mer
    struc_kmer = StructurekmerExtract(rna_struc, Structurekmertotal)
    # lncRNA structure g-gap
    struc_Ggap = StructureGgapExtract(rna_struc, StructureGgaptotal)
    # Complex Feature
    seq_ker = np.fromstring(seq_kmer, sep=' ').reshape(1, -1)
    seq_Ggap = np.fromstring(seq_Ggap, sep=' ').reshape(1, -1)
    struc_kmer = np.fromstring(struc_kmer, sep=' ').reshape(1, -1)
    struc_Ggap = np.fromstring(struc_Ggap, sep=' ').reshape(1, -1)
    # complex_feats = np.concatenate([seq_ker, seq_Ggap, struc_kmer, struc_Ggap], axis=1)
    # complex_feats = np.concatenate([seq_ker, struc_kmer], axis=1)
    complex_feats = np.concatenate([seq_Ggap, struc_Ggap], axis=1)
    return complex_feats


def FusionComplexFeature(mis, lncs):
    LncTotalComplexFeature = []
    MiTotalComplexFeature = []
    for i in tqdm(range(0, len(mis), 3), desc='computing miRNA sec-structure'):
        miRNAname, miRNAsequence, miRNAstructure = mis[i], mis[i + 1], mis[i + 2]
        mi_complex_feats = calc_single_complex_feats(miRNAsequence, miRNAstructure)
        MiTotalComplexFeature.append(mi_complex_feats)

    for i in tqdm(range(0, len(lncs), 3), desc='computing lncRNA sec-structure'):
        lncRNAname, lncRNAsequence, lncRNAstructure = lncs[i], lncs[i + 1], lncs[i + 2]
        lnc_complex_feats = calc_single_complex_feats(lncRNAsequence, lncRNAstructure)
        LncTotalComplexFeature.append(lnc_complex_feats)

    LncTotalComplexFeature = np.array(LncTotalComplexFeature).squeeze()
    MiTotalComplexFeature = np.array(MiTotalComplexFeature).squeeze()

    return MiTotalComplexFeature, LncTotalComplexFeature


def readFa(fa):
    """
    @msg: 读取一个fasta文件
    @param fa {str}  fasta 文件路径
    @return: {generator} 返回一个生成器，能迭代得到fasta文件的每一个序列名和序列
    """
    with open(fa, 'r') as FA:
        seqName, seq = '', ''
        while 1:
            line = FA.readline()
            line = line.strip('\n')
            if (line.startswith('>') or not line) and seqName:
                yield seqName, seq
            if line.startswith('>'):
                seqName = line[1:]
                seq = ''
            else:
                seq += line
            if not line: break


def write_Fa(header, sequence, filename, is_cal_struc=True):
    with open(filename, 'a') as fasta_file:
        if is_cal_struc:
            structure, energy = RNA.fold(sequence)
            fasta_file.write(f'>{header}\n{sequence}\n{structure} ({"{:.2f}".format(energy)})\n')
        else:
            fasta_file.write(f'>{header}\n{sequence}\n')


def get_LM_seq(LMI, ):
    # 由于在LMI中部分lncRNA和miRNA没有序列数据，所以需要筛选出那些有序列的数据
    O_sa_LMI = LMI
    # O_sa_LMI.to_csv('./Example/O_sa_LMI.csv', header=None, index=False)
    mi_fa = "./Example/O.sa.mi.txt"
    lnc_fa = "./Example/O.sa.lnc.txt"
    # 读取一个fasta文件，并输出其中的每一条序列名，序列长度和GC含量
    pbar = tqdm(total=1901)
    for seqName, seq in tqdm(readFa(mi_fa)):
        if np.isin(seqName, O_sa_LMI.iloc[:, 0]):
            # print(seqName)
            write_Fa(seqName, seq, './Example/O.sa.LMI.mi.fasta')
            pbar.update(1)

    pbar1 = tqdm(total=602)
    for seqName, seq in tqdm(readFa(lnc_fa)):
        if np.isin(seqName, O_sa_LMI.iloc[:, 1]):
            # print(seqName)
            write_Fa(seqName, seq, './Example/O.sa.LMI.lns.fasta')
            pbar1.update(1)


def process_rna_struc():
    # Load data
    O_sa_LMI = pd.read_csv('./Example/O.sa.LMI.csv', sep='\t', header=None)
    mi_names = np.unique(O_sa_LMI.iloc[:, 0])
    lnc_names = np.unique(O_sa_LMI.iloc[:, 1])


if __name__ == '__main__':
    # lnc_fa = "./Example/rna_seq_data/O.sa.lnc1_100.fasta"
    # # 读取一个fasta文件，并输出其中的每一条序列名，序列长度和GC含量
    # pbar = tqdm(total=100)
    # O_sa_mi = []
    # O_sa_lnc = []
    # for seqName, seq in readFa(lnc_fa):
    #     write_Fa(seqName, seq, './Example/rna_structure/O.sa.lnc1_100.fasta')
    #     O_sa_mi.append(seqName)
    #     pbar.update(1)
    # O_sa_mi = np.array(O_sa_mi).reshape(-1, 1)
    # lnc_structure = './Example/rna_structure/O.sa.lnc201_300.fasta'
    # ListlncRNA = open(lnc_structure, 'r').readlines()
    #
    # mi_structure = './Example/O.sa.mi.test1.fasta'
    # ListmiRNA = open(mi_structure, 'r').readlines()
    #
    # a, b = FusionComplexFeature(ListmiRNA, ListlncRNA)
    ListmiRNA = open('./Example/rna_structure/O.sa.mi_st.fasta', 'r').readlines()
    ListlncRNA = open('./Example/rna_structure/O.sa.lnc.st.fasta', 'r').readlines()

    mi_fests, lnc_feats = FusionComplexFeature(ListmiRNA, ListlncRNA)
    pd.DataFrame(mi_fests).to_csv('./Example/complex_feats/mi_complex_feats.csv', header=None, index=False)
    pd.DataFrame(lnc_feats).to_csv('./Example/complex_feats/lnc_complex_feats.csv', header=None, index=False)
