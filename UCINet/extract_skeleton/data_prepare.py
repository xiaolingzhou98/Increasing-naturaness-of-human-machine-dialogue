# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:54:10 2018

@author: Yanglei
"""

import codecs
import numpy as np
import pandas as pd
import jieba.posseg as pseg
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec
import re

data2label = {}
data2flag = {}
excel_path = 'data/input/skeleton.xlsx'
data = pd.read_excel(excel_path, sheetname='Sheet1')

words = list(np.array(data.loc[:, ['question']]))
words = [str(word) for word in words]


#
# def sub_str(str_ori,start_index,end_index,str_temp):
#    return str_ori[0:start_index]+str_temp+str_ori[end_index:]


def update_label_str(str_ori, que, mark_list, start='s', middle='m', end='e', short='i'):
    for mark in mark_list:
        mark = re.sub(u'[^\u4e00-\u9fa5a-zA-Z0-9-，。,.]', "", mark)
        if mark != 'nan' and mark in que:
            # print(que, mark)
            start_index = que.index(mark)
            end_index = start_index + len(mark)
            if len(mark) == 1:
                str_temp = short
                str_ori = str_ori[0:start_index] + str_temp + str_ori[end_index:]
            else:
                str_temp = start + (len(mark) - 2) * middle + end
                str_ori = str_ori[0:start_index] + str_temp + str_ori[end_index:]
            # print(mark, que)
        elif mark != 'nan' and mark not in que:
            print(que, mark)
    return str_ori


def generate_str(que, type_information, target_information):
    label_str = len(que) * 'O'
    label_str = update_label_str(label_str, que, target_information.split('，'))
    if '，' not in type_information:
        label_str = update_label_str(label_str, que, [type_information], start='x', middle='y', end='z', short='g')
    else:
        assert len(type_information.split('，')) == 2
        label_str = update_label_str(label_str, que, [type_information.split('，')[0]], start='S', middle='M', end='E',
                                     short='I')
        label_str = update_label_str(label_str, que, [type_information.split('，')[1]], start='a', middle='b', end='c',
                                     short='d')
    return label_str


def make_label_file():
    fw = codecs.open('data/sample_train.txt', 'w', 'utf-8')
    for desc_index in range(data.last_valid_index()):
        desc = list(np.array(data.loc[desc_index]))
        print(desc)
        target_information = str(desc[1])
        type_information = str(desc[2])
        que = str(desc[0])
        que = re.sub(u'[^\u4e00-\u9fa5a-zA-Z0-9-，。,.]', "", que)
        type_information = re.sub(u'[^\u4e00-\u9fa5a-zA-Z0-9-，。,.]', "", type_information)
        target_information = re.sub(u'[^\u4e00-\u9fa5a-zA-Z0-9-，。,.]', "", target_information)

        # blue = [item for item in desc[2:9] if type(item) == str]
        # red = [item for item in desc[9:14] if type(item) == str]
        # green = [item for item in desc[14:] if type(item) == str]

        type_information = type_information.replace(',', '，')
        target_information = target_information.replace(',', '，')
        label_str = generate_str(que, type_information, target_information)
        sequence = []
        sequence_flag = []
        data_flag = []
        lines = pseg.cut(que)
        for word, flag in lines:
            sequence.append(list(word))
            sequence_flag.append(flag)
        for s in range(len(sequence_flag)):
            for zi in sequence[s]:
                data_flag.append(sequence_flag[s])
        if len(que) == len(label_str):
            for i in range(len(que)):
                line = que[i] + '\t' + data_flag[i] + '\t' + label_str[i] + '\n'
                fw.writelines(line)
            fw.writelines('\n')
        else:
            raise Exception("Some thing get wrong in term of length")
    fw.close()


def embedding_sentences(sentences):
    w2vModel = Word2Vec(sentences, size=64, window=5, min_count=1)
    w2vModel.save('Model/Word2vec_model.pkl')
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(w2vModel.wv.vocab.keys(), allow_update=True)
    w2indx = {v: k for k, v in gensim_dict.items()}
    w2vec = {word: w2vModel[word] for word in w2indx.keys()}
    return w2vec


make_label_file()
embedding_sentences(words)
