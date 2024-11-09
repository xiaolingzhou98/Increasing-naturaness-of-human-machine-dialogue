import json
import os
from metric_model_utils import train
import tensorflow as tf
import sys
#from test import test
import random
import pickle
import numpy as np
from data_stream_final import DataStream
from metric_judge_model import AnswerUnderstander
from me_utils_final23 import read_file, make_instances_parallel, \
    make_corpus, generate_vocabulary, \
    split_train_dev, split_train_dev_2, drop_instances_to_excel, \
    gather_question_dict, split_template_choose, gather_samples, \
    gather_targets_for_samples
from bert_encoder import BertEncoder
from w2v_encoder import W2VEncoder

# def train_model():
if __name__ == "__main__":
    #np.random.seed(200)
    with open('metric_judge_config.json', encoding='utf-8') as infile:
        config = json.load(infile)

    int2bool = {1: True, 0: False}
    data_path = config["train_config"]["DATA_PATH"]
    data_path_x = config["train_config"]["DATA_PATH_x"]
    data_path_xp = config["train_config"]["DATA_PATH_xp"]
    data_path_xn = config["train_config"]["DATA_PATH_xn"]
    sentiment_words_path = config["train_config"]["SENTIMENT_WORDS_PATH"]
    max_sequence_len = config["train_config"]["MAX_SEQUENCE_LEN"]
    batch_size = config["train_config"]["BATCH_SIZE"]
    is_shuffle = int2bool[config["train_config"]["IS_SHUFFLE"]]
    is_loop = int2bool[config["train_config"]["Is_LOOP"]]
    is_sort = int2bool[config["train_config"]["IS_SORT"]]
    nb_epoch = config["train_config"]["NB_EPOCH"]
    dropout_rate = config["train_config"]["DROPOUT_RATE"]
    nb_classes = config["train_config"]["NB_CLASSES"]
    attention_dim = config["train_config"]["ATTENTION_DIM"]
    nb_hops = config["train_config"]["NB_HOPS"]
    use_bert = int2bool[config["train_config"]["USE_BERT"]]
    optimizer = config["train_config"]["OPTIMIZER"]
    learning_rate = config["train_config"]["LEARNING_RATE"]
    grad_clipper = config["train_config"]["GRAD_CLIPPER"]
    drop_train_path_x = config["train_config"]["DROP_TRAIN_PATH_x"]
    drop_path_x = config["train_config"]["DROP_CHOOSE_DEV_PATH_x"]
    drop_template_path_x = config["train_config"]["DROP_CHOOSE_TEMPLATE_PATH_x"]
    drop_train_path_xp = config["train_config"]["DROP_TRAIN_PATH_xp"]
    drop_path_xp = config["train_config"]["DROP_CHOOSE_DEV_PATH_xp"]
    drop_template_path_xp = config["train_config"]["DROP_CHOOSE_TEMPLATE_PATH_xp"]
    drop_train_path_xn = config["train_config"]["DROP_TRAIN_PATH_xn"]
    drop_path_xn = config["train_config"]["DROP_CHOOSE_DEV_PATH_xn"]
    drop_template_path_xn = config["train_config"]["DROP_CHOOSE_TEMPLATE_PATH_xn"]
    best_path = config["train_config"]["BEST_PATH"]
    question2targets_path = config["train_config"]["QUESTION2TARGETS_PATH"]
    use_extra_feature = config["train_config"]["USE_EXTRA_FEATURE"]
    ner_dict_size = config["train_config"]["NER_DICT_SIZE"]
    pos_dict_size = config["train_config"]["POS_DICT_SIZE"]
    extra_feature_dim = config["train_config"]["EXTRA_FEATURE_DIM"]
    ner_dict_path = config["train_config"]["NER_DICT_PATH"]
    pos_dict_path = config["train_config"]["POS_DICT_PATH"]
    rnn_dim = config["train_config"]["RNN_DIM"]
    lambda_l2 = config["train_config"]["LAMBDA_L2"]
    sentiment_polarity_multiple = config["train_config"]["POLARITY_MULTIPLE"]
    pretrained_model = config["train_config"]["PRETRAINED_MODEL"]

    use_w2v = True
    if use_bert:
        use_w2v = False

    char_w2v_path_x = config["w2v_config_x"]["CHAR_W2V_PATH_x"]
    char_voc_path_x = config["w2v_config_x"]["CHAR_VOC_PATH_x"]
    char_embedding_matrix_path_x = config["w2v_config_x"][
        "CHAR_EMBEDDING_MATRIX_PATH_x"]
    word_w2v_path_x = config["w2v_config_x"]["WORD_W2V_PATH_x"]
    word_voc_path_x = config["w2v_config_x"]["WORD_VOC_PATH_x"]
    word_embedding_matrix_path_x = config["w2v_config_x"][
        "WORD_EMBEDDING_MATRIX_PATH_x"]
    char_w2v_path_xp = config["w2v_config_xp"]["CHAR_W2V_PATH_xp"]
    char_voc_path_xp = config["w2v_config_xp"]["CHAR_VOC_PATH_xp"]
    char_embedding_matrix_path_xp = config["w2v_config_xp"][
        "CHAR_EMBEDDING_MATRIX_PATH_xp"]
    word_w2v_path_xp = config["w2v_config_xp"]["WORD_W2V_PATH_xp"]
    word_voc_path_xp = config["w2v_config_xp"]["WORD_VOC_PATH_xp"]
    word_embedding_matrix_path_xp = config["w2v_config_xp"][
        "WORD_EMBEDDING_MATRIX_PATH_xp"]
    char_w2v_path_xn = config["w2v_config_xn"]["CHAR_W2V_PATH_xn"]
    char_voc_path_xn = config["w2v_config_xn"]["CHAR_VOC_PATH_xn"]
    char_embedding_matrix_path_xn = config["w2v_config_xn"][
        "CHAR_EMBEDDING_MATRIX_PATH_xn"]
    word_w2v_path_xn = config["w2v_config_xn"]["WORD_W2V_PATH_xn"]
    word_voc_path_xn = config["w2v_config_xn"]["WORD_VOC_PATH_xn"]
    word_embedding_matrix_path_xn = config["w2v_config_xn"][
        "WORD_EMBEDDING_MATRIX_PATH_xn"]

    bert_model_path = config["bert_config"]["BERT_MODEL_PATH"]
    bert_config_file = config["bert_config"]["CONFIG_FILE"]
    bert_checkpoint_path = config["bert_config"]["INIT_CHECKPOINT"]
    bert_voc_path = config["bert_config"]["VOC_FILE"]
    sen2id_path = config["bert_config"]["SEN2ID_PATH"]
    samples, _, _ = read_file(data_path)
    samples_x, _, _ = read_file(data_path_x)
    samples_xp, _, _ = read_file(data_path_xp)
    samples_xn, _, _ = read_file(data_path_xn)
    
    
    ans_max_len = config["train_config"]["ANS_MAX_LEN"]
    que_max_len = config["train_config"]["QUE_MAX_LEN"]
    char_corpus_x, word_corpus_x = make_corpus(samples_x)
    char_corpus_xp, word_corpus_xp = make_corpus(samples_xp)
    char_corpus_xn, word_corpus_xn = make_corpus(samples_xn)
    # default sen2id from char_corpus
    sen2id_x = generate_vocabulary(sen2id_path, char_corpus_x)
    sen2id_xp = generate_vocabulary(sen2id_path, char_corpus_xp)
    sen2id_xn = generate_vocabulary(sen2id_path, char_corpus_xn)
    question2targets = gather_targets_for_samples(samples, True,
                                                  question2targets_path)
    bert_max_seq_len_x = max([len(sen) for sen in sen2id_x.keys()]) + 2
    bert_max_seq_len_xp = max([len(sen) for sen in sen2id_xp.keys()]) + 2
    bert_max_seq_len_xn = max([len(sen) for sen in sen2id_xn.keys()]) + 2
    max_sequence_len_x = max(bert_max_seq_len_x, max_sequence_len)
    max_sequence_len_xp = max(bert_max_seq_len_xp, max_sequence_len)
    max_sequence_len_xn = max(bert_max_seq_len_xn, max_sequence_len)
    bert_max_seq_len_x = max_sequence_len_x
    bert_max_seq_len_xp = max_sequence_len_xp
    bert_max_seq_len_xn = max_sequence_len_xn

    # question2index = gather_question_dict(samples)
    # question2index, question2template = split_template_choose(samples, question2index)
    # samples_template = gather_samples(samples, question2template)
    # samples = gather_samples(samples, question2index)

    w2v_encoder_x = W2VEncoder(True, char_corpus_x, word_corpus_x,
                             char_w2v_path_x, char_voc_path_x,
                             char_embedding_matrix_path_x, word_w2v_path_x,
                             word_voc_path_x, word_embedding_matrix_path_x)
    w2v_encoder_xp = W2VEncoder(True, char_corpus_xp, word_corpus_xp,
                             char_w2v_path_xp, char_voc_path_xp,
                             char_embedding_matrix_path_xp, word_w2v_path_xp,
                             word_voc_path_xp, word_embedding_matrix_path_xp)
    w2v_encoder_xn = W2VEncoder(True, char_corpus_xn, word_corpus_xn,
                             char_w2v_path_xn, char_voc_path_xn,
                             char_embedding_matrix_path_xn, word_w2v_path_xn,
                             word_voc_path_xn, word_embedding_matrix_path_xn)
    bert_encoder_x = BertEncoder(model_root=bert_model_path,
                               bert_config_file=bert_config_file,
                               init_checkpoint=bert_checkpoint_path,
                               vocab_file=bert_voc_path,
                               max_sequence_len=bert_max_seq_len_x,
                               embedding_batch=3,
                               embedding_matrix_path=None,
                               sen2id_path=sen2id_path,
                               vec_dim=768)
    bert_encoder_xp = BertEncoder(model_root=bert_model_path,
                               bert_config_file=bert_config_file,
                               init_checkpoint=bert_checkpoint_path,
                               vocab_file=bert_voc_path,
                               max_sequence_len=bert_max_seq_len_xp,
                               embedding_batch=3,
                               embedding_matrix_path=None,
                               sen2id_path=sen2id_path,
                               vec_dim=768)
    bert_encoder_xn = BertEncoder(model_root=bert_model_path,
                               bert_config_file=bert_config_file,
                               init_checkpoint=bert_checkpoint_path,
                               vocab_file=bert_voc_path,
                               max_sequence_len=bert_max_seq_len_xn,
                               embedding_batch=3,
                               embedding_matrix_path=None,
                               sen2id_path=sen2id_path,
                               vec_dim=768)

    # ins_template = make_instances(samples_template,
    #                               w2v_encoder.char_voc,
    #                               w2v_encoder.word_voc,
    #                               sentiment_words_path,
    #                               ner_dict_path=ner_dict_path,
    #                               pos_dict_path=pos_dict_path,
    #                               use_extra_feature=use_extra_feature,
    #                               question2targets=question2targets,
    #                               is_training=True,
    #                               need_augment=True)
    instances_x = make_instances_parallel(samples_x,
                               w2v_encoder_x.char_voc,
                               w2v_encoder_x.word_voc,
                               sentiment_words_path,
                               ner_dict_path=ner_dict_path,
                               pos_dict_path=pos_dict_path,
                               use_extra_feature=use_extra_feature,
                               question2targets=question2targets,
                               is_training=True,
                               need_augment=False)
    instances_xp = make_instances_parallel(samples_xp,
                               w2v_encoder_xp.char_voc,
                               w2v_encoder_xp.word_voc,
                               sentiment_words_path,
                               ner_dict_path=ner_dict_path,
                               pos_dict_path=pos_dict_path,
                               use_extra_feature=use_extra_feature,
                               question2targets=question2targets,
                               is_training=True,
                               need_augment=False)
    instances_xn = make_instances_parallel(samples_xn,
                               w2v_encoder_xn.char_voc,
                               w2v_encoder_xn.word_voc,
                               sentiment_words_path,
                               ner_dict_path=ner_dict_path,
                               pos_dict_path=pos_dict_path,
                               use_extra_feature=use_extra_feature,
                               question2targets=question2targets,
                               is_training=True,
                               need_augment=False)
    instances_train_x, instances_dev_x = split_train_dev(instances_x)
    instances_train_xp, instances_dev_xp = split_train_dev(instances_xp)
    instances_train_xn, instances_dev_xn = split_train_dev(instances_xn)
    #np.random.shuffle(instances_train)
    #np.random.shuffle(instances_dev)
    drop_instances_to_excel(instances_train_x, drop_train_path_x)
    drop_instances_to_excel(instances_train_xp, drop_train_path_xp)
    drop_instances_to_excel(instances_train_xn, drop_train_path_xn)
    drop_instances_to_excel(instances_dev_x, drop_path_x)
    drop_instances_to_excel(instances_dev_xp, drop_path_xp)
    drop_instances_to_excel(instances_dev_xn, drop_path_xn)
    drop_instances_to_excel(instances_x, 'temp_x.xlsx')
    drop_instances_to_excel(instances_xp, 'temp_xp.xlsx')
    drop_instances_to_excel(instances_xn, 'temp_xn.xlsx')
    # drop_instances_to_excel(ins_template, drop_template_path)
    instances_train_x, instances_valid_x = split_train_dev_2(instances_train_x)
    instances_train_xp, instances_valid_xp = split_train_dev_2(instances_train_xp)
    instances_train_xn, instances_valid_xn = split_train_dev_2(instances_train_xn)
    #np.random.shuffle(instances_train) 
    #np.random.shuffle(instances_valid)

    data_stream_train_x = DataStream(instances=instances_train_x,
                                   is_shuffle=False,
                                   is_loop=is_loop,
                                   batch_size=batch_size,
                                   ans_max_len=ans_max_len,
                                   que_max_len=que_max_len,
                                   use_bert=use_bert,
                                   bert_encoder=bert_encoder_x,
                                   is_sort=is_sort)
    data_stream_train_xp = DataStream(instances=instances_train_xp,
                                   is_shuffle=False,
                                   is_loop=is_loop,
                                   batch_size=batch_size,
                                   ans_max_len=ans_max_len,
                                   que_max_len=que_max_len,
                                   use_bert=use_bert,
                                   bert_encoder=bert_encoder_xp,
                                   is_sort=is_sort)
    data_stream_train_xn = DataStream(instances=instances_train_xn,
                                   is_shuffle=False,
                                   is_loop=is_loop,
                                   batch_size=batch_size,
                                   ans_max_len=ans_max_len,
                                   que_max_len=que_max_len,
                                   use_bert=use_bert,
                                   bert_encoder=bert_encoder_xn,
                                   is_sort=is_sort)
    data_stream_valid_x = DataStream(instances=instances_valid_x,
                                   is_shuffle=False,
                                   is_loop=is_loop,
                                   batch_size=batch_size,
                                   ans_max_len=ans_max_len,
                                   que_max_len=que_max_len,
                                   use_bert=use_bert,
                                   bert_encoder=bert_encoder_x,
                                   is_sort=is_sort)
    data_stream_valid_xp = DataStream(instances=instances_valid_xp,
                                   is_shuffle=False,
                                   is_loop=is_loop,
                                   batch_size=batch_size,
                                   ans_max_len=ans_max_len,
                                   que_max_len=que_max_len,
                                   use_bert=use_bert,
                                   bert_encoder=bert_encoder_xp,
                                   is_sort=is_sort)
    data_stream_valid_xn = DataStream(instances=instances_valid_xn,
                                   is_shuffle=False,
                                   is_loop=is_loop,
                                   batch_size=batch_size,
                                   ans_max_len=ans_max_len,
                                   que_max_len=que_max_len,
                                   use_bert=use_bert,
                                   bert_encoder=bert_encoder_xn,
                                   is_sort=is_sort)


    with tf.Graph().as_default():
        
        initializer = tf.glorot_uniform_initializer()
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            answer_understander_train = AnswerUnderstander(
                use_bert=use_bert,
                use_w2v=use_w2v,
                rnn_unit='lstm',
                dropout_rate=dropout_rate,
                optimizer=optimizer,
                learning_rate=learning_rate,
                grad_clipper=grad_clipper,
                global_step=global_step,
                attention_dim=attention_dim,
                nb_hops=nb_hops,
                rnn_dim=rnn_dim,
                lambda_l2=lambda_l2,
                is_training=True,
                sentiment_polarity_multiple=sentiment_polarity_multiple,
                nb_classes=nb_classes,
                use_extra_feature=use_extra_feature,
                ner_dict_size=ner_dict_size,
                pos_dict_size=pos_dict_size,
                ans_max_len=ans_max_len,
                que_max_len=que_max_len,
                extra_feature_dim=extra_feature_dim,
                char_w2v_embedding_matrix_path_x=char_embedding_matrix_path_x,
                word_w2v_embedding_matrix_path_x=word_embedding_matrix_path_x,
                char_w2v_embedding_matrix_path_xp=char_embedding_matrix_path_xp,
                word_w2v_embedding_matrix_path_xp=word_embedding_matrix_path_xp,
                char_w2v_embedding_matrix_path_xn=char_embedding_matrix_path_xn,
                word_w2v_embedding_matrix_path_xn=word_embedding_matrix_path_xn)
        
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            answer_understander_valid = AnswerUnderstander(
                use_bert=use_bert,
                use_w2v=use_w2v,
                rnn_unit='lstm',
                dropout_rate=dropout_rate,
                optimizer=optimizer,
                learning_rate=learning_rate,
                grad_clipper=grad_clipper,
                global_step=None,
                attention_dim=attention_dim,
                nb_hops=nb_hops,
                rnn_dim=rnn_dim,
                lambda_l2=lambda_l2,
                is_training=False,
                sentiment_polarity_multiple=sentiment_polarity_multiple,
                nb_classes=nb_classes,
                use_extra_feature=use_extra_feature,
                ner_dict_size=ner_dict_size,
                pos_dict_size=pos_dict_size,
                ans_max_len=ans_max_len,
                que_max_len=que_max_len,
                extra_feature_dim=extra_feature_dim,
                char_w2v_embedding_matrix_path_x=char_embedding_matrix_path_x,
                word_w2v_embedding_matrix_path_x=word_embedding_matrix_path_x,
                char_w2v_embedding_matrix_path_xp=char_embedding_matrix_path_xp,
                word_w2v_embedding_matrix_path_xp=word_embedding_matrix_path_xp,
                char_w2v_embedding_matrix_path_xn=char_embedding_matrix_path_xn,
                word_w2v_embedding_matrix_path_xn=word_embedding_matrix_path_xn)
        saver = tf.train.Saver()
        sess = tf.Session()
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        print('begin training...')
        train(nb_epoch=nb_epoch,
              sess=sess,
              saver=saver,
              data_stream_train_x=data_stream_train_x,
              data_stream_train_xp=data_stream_train_xp,
              data_stream_train_xn=data_stream_train_xn,
              data_stream_valid_x=data_stream_valid_x,
              data_stream_valid_xp=data_stream_valid_xp,
              data_stream_valid_xn=data_stream_valid_xn,
              answer_understander_train=answer_understander_train,
              answer_understander_valid=answer_understander_valid,
              best_path=best_path)

#if __name__ == "__main__":
 #   train_model()
