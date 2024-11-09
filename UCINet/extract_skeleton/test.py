#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'jxliu.nlper@gmail.com'
"""
    标记文件
"""
import yaml
import pickle
import tensorflow as tf
from .load_data import load_vocs, init_data
from .model import SequenceLabelingModel
from .deal import writetxt

# import sys
# import os
# sys.path.append(os.getcwd())
# print(os.getcwd())


# 加载配置文件
with open('extract_skeleton/config.yml', 'rb') as file_config:
    config = yaml.load(file_config)

feature_names = config['model_params']['feature_names']

# 初始化embedding shape, dropouts, 预训练的embedding也在这里初始化)
feature_weight_shape_dict, feature_weight_dropout_dict, \
feature_init_weight_dict = dict(), dict(), dict()
for feature_name in feature_names:
    feature_weight_shape_dict[feature_name] = \
        config['model_params']['embed_params'][feature_name]['shape']
    feature_weight_dropout_dict[feature_name] = \
        config['model_params']['embed_params'][feature_name]['dropout_rate']
    path_pre_train = config['model_params']['embed_params'][feature_name]['path']
    if path_pre_train:
        with open(path_pre_train, 'rb') as file_r:
            feature_init_weight_dict[feature_name] = pickle.load(file_r)

# 加载vocs
path_vocs = []
for feature_name in feature_names:
    path_vocs.append(config['data_params']['voc_params'][feature_name]['path'])
path_vocs.append(config['data_params']['voc_params']['label']['path'])
vocs = load_vocs(path_vocs)

# 加载模型
model = SequenceLabelingModel(
    sequence_length=config['model_params']['sequence_length'],
    nb_classes=config['model_params']['nb_classes'],
    nb_hidden=config['model_params']['bilstm_params']['num_units'],
    feature_weight_shape_dict=feature_weight_shape_dict,
    feature_init_weight_dict=feature_init_weight_dict,
    feature_weight_dropout_dict=feature_weight_dropout_dict,
    dropout_rate=config['model_params']['dropout_rate'],
    nb_epoch=config['model_params']['nb_epoch'], feature_names=feature_names,
    batch_size=config['model_params']['batch_size'],
    train_max_patience=config['model_params']['max_patience'],
    use_crf=config['model_params']['use_crf'],
    l2_rate=config['model_params']['l2_rate'],
    rnn_unit=config['model_params']['rnn_unit'],
    learning_rate=config['model_params']['learning_rate'],
    path_model=config['model_params']['path_model'])

saver = tf.train.Saver()
saver.restore(model.sess, config['model_params']['path_model'])


def make_skeleton_indicator(user_question):
    lab = writetxt(user_question)
    # 加载数据
    if len(lab) == 0:
        return []
    sep_str = config['data_params']['sep']
    assert sep_str in ['table', 'space']
    sep = '\t' if sep_str == 'table' else ' '
    data_dict = init_data(path=config['data_params']['path_test'], feature_names=feature_names, sep=sep, vocs=vocs,
                          max_len=config['model_params']['sequence_length'], model='test')
    out_seqs = model.predict(data_dict)
    # 1: type -1:target 0:other
    end_seqs = []
    # print(out_seqs)
    for seq in out_seqs:
        temp_seq = []
        for label in seq:
            if label == 1:
                temp_seq.append(0)
            elif label in [2, 3, 4]:
                temp_seq.append(-1)
            else:
                temp_seq.append(1)
        end_seqs.append(temp_seq)
    return end_seqs
