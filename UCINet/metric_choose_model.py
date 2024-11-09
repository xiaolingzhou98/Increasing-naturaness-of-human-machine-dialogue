import numpy as np
import tensorflow as tf
import model_utils_reuse

from data_stream_final import InstanceBatch


class AnswerUnderstander(object):
    def __init__(self, use_bert, use_w2v, rnn_unit, dropout_rate,
                 char_w2v_embedding_matrix_path_x,char_w2v_embedding_matrix_path_xp,char_w2v_embedding_matrix_path_xn, rnn_dim, nb_classes,
                 optimizer, learning_rate, grad_clipper, global_step, nb_hops,
                 attention_dim, is_training, use_extra_feature, ans_max_len,
                 que_max_len, extra_feature_dim, ner_dict_size, pos_dict_size,
                 lambda_l2, sentiment_polarity_multiple,
                 word_w2v_embedding_matrix_path_x,word_w2v_embedding_matrix_path_xp,word_w2v_embedding_matrix_path_xn):
        self.use_bert = use_bert
        self.use_w2v = use_w2v
        self.rnn_unit = rnn_unit
        self.dropout_rate = dropout_rate
        self.rnn_dim = rnn_dim
        self.nb_classes = nb_classes
        self.is_training = is_training
        self.sentiment_polarity_multiple = sentiment_polarity_multiple
        self.nb_hops = nb_hops
        self.attention_dim = attention_dim
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.grad_clipper = grad_clipper
        self.use_extra_feature = use_extra_feature
        self.ner_dict_size = ner_dict_size
        self.pos_dict_size = pos_dict_size
        self.extra_feature_dim = extra_feature_dim
        self.global_step = global_step
        self.lambda_l2 = lambda_l2
        self.ans_max_len = ans_max_len
        self.que_max_len = que_max_len

        assert self.use_w2v or self.use_bert
        self.max_option_length = 5
        # create placeholders
        self.weight = tf.placeholder(tf.float32)  # [batch_size]
        self.que_lens_x = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.ans_lens_x = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.que_lens_xp = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.ans_lens_xp = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.que_lens_xn = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.ans_lens_xn = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.in_ans_append_x = tf.placeholder(tf.int32, [None, None])
        self.in_ans_append_xp = tf.placeholder(tf.int32, [None, None])
        self.in_ans_append_xn = tf.placeholder(tf.int32, [None, None])
        self.len_option_sequence_x = tf.placeholder(tf.int32,[None])
        self.len_option_sequence_xp = tf.placeholder(tf.int32,[None])
        self.len_option_sequence_xn = tf.placeholder(tf.int32,[None])
        self.word_option_length_x = tf.placeholder(tf.int32,[None,None])
        self.word_option_length_xp = tf.placeholder(tf.int32,[None,None])
        self.word_option_length_xn = tf.placeholder(tf.int32,[None,None])
        self.label_sequence_change_x = tf.placeholder(tf.int32, [None,None]) #dim 1 reprsent the label of option 1
        self.label_sequence_change_xp = tf.placeholder(tf.int32, [None,None]) #dim 1 reprsent the label of option 1
        self.label_sequence_change_xn = tf.placeholder(tf.int32, [None,None]) #dim 1 reprsent the label of option 1
        self.que_skeleton_label_x = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        #option
        self.in_option_ww2v_index_matrix_x = tf.placeholder(  #all options' word embedding
            tf.int32,[None,None,None]) #[batch_size,max_option_length,max_sequence_len]
        
        self.in_que_cw2v_index_matrix_x = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_ww2v_index_matrix_x = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_cw2v_index_matrix_x = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_ww2v_index_matrix_x = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]

        self.in_que_sentiment_polarity_matrix_x = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_sentiment_polarity_matrix_x = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_indicate_target_matrix_x = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_indicate_target_matrix_x = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.que_skeleton_label_xp = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        #option
        self.in_option_ww2v_index_matrix_xp = tf.placeholder(  #all options' word embedding
            tf.int32,[None,None,None]) #[batch_size,max_option_length,max_sequence_len]
        
        self.in_que_cw2v_index_matrix_xp = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_ww2v_index_matrix_xp = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_cw2v_index_matrix_xp = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_ww2v_index_matrix_xp = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]

        self.in_que_sentiment_polarity_matrix_xp = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_sentiment_polarity_matrix_xp = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_indicate_target_matrix_xp = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_indicate_target_matrix_xp = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.que_skeleton_label_xn = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        #option
        self.in_option_ww2v_index_matrix_xn = tf.placeholder(  #all options' word embedding
            tf.int32,[None,None,None]) #[batch_size,max_option_length,max_sequence_len]
        
        self.in_que_cw2v_index_matrix_xn = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_ww2v_index_matrix_xn = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_cw2v_index_matrix_xn = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_ww2v_index_matrix_xn = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]

        self.in_que_sentiment_polarity_matrix_xn = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_sentiment_polarity_matrix_xn = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_indicate_target_matrix_xn = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_indicate_target_matrix_xn = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        if self.use_extra_feature:
            self.in_que_pos_index_matrix_x = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_que_ner_index_matrix_x = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_ans_pos_index_matrix_x = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_ans_ner_index_matrix_x = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_que_pos_index_matrix_xp = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_que_ner_index_matrix_xp = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_ans_pos_index_matrix_xp = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_ans_ner_index_matrix_xp = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_que_pos_index_matrix_xn = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_que_ner_index_matrix_xn = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_ans_pos_index_matrix_xn = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_ans_ner_index_matrix_xn = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]

        if self.use_bert:
            self.que_bert_matrix_x = tf.placeholder(tf.float32,
                                                  [None, None, 768])
            self.ans_bert_matrix_x = tf.placeholder(tf.float32,
                                                  [None, None, 768])
            self.que_bert_matrix_xp = tf.placeholder(tf.float32,
                                                  [None, None, 768])
            self.ans_bert_matrix_xp = tf.placeholder(tf.float32,
                                                  [None, None, 768])
            self.que_bert_matrix_xn = tf.placeholder(tf.float32,
                                                  [None, None, 768])
            self.ans_bert_matrix_xn = tf.placeholder(tf.float32,
                                                  [None, None, 768])

        # init basic embedding matrix
        char_w2v_embedding_matrix_x, word_w2v_embedding_matrix_x = model_utils_reuse.load_variable_from_file(
            char_w2v_embedding_matrix_path_x, word_w2v_embedding_matrix_path_x)
        self.char_w2v_embedding_matrix_x = tf.convert_to_tensor(
            char_w2v_embedding_matrix_x)
        self.word_w2v_embedding_matrix_x = tf.convert_to_tensor(
            word_w2v_embedding_matrix_x)
        char_w2v_embedding_matrix_xp, word_w2v_embedding_matrix_xp = model_utils_reuse.load_variable_from_file(
            char_w2v_embedding_matrix_path_xp, word_w2v_embedding_matrix_path_xp)
        self.char_w2v_embedding_matrix_xp = tf.convert_to_tensor(
            char_w2v_embedding_matrix_xp)
        self.word_w2v_embedding_matrix_xp = tf.convert_to_tensor(
            word_w2v_embedding_matrix_xp)
        char_w2v_embedding_matrix_xn, word_w2v_embedding_matrix_xn = model_utils_reuse.load_variable_from_file(
            char_w2v_embedding_matrix_path_xn, word_w2v_embedding_matrix_path_xn)
        self.char_w2v_embedding_matrix_xn = tf.convert_to_tensor(
            char_w2v_embedding_matrix_xn)
        self.word_w2v_embedding_matrix_xn = tf.convert_to_tensor(
            word_w2v_embedding_matrix_xn)
        # if self.use_bert:
        #     self.bert_embedding_matrix = tf.convert_to_tensor(bert_embedding_matrix)

        # create model
        self.create_model_graph()

    def create_feed_dict(self, cur_batch,cur_batch_p,cur_batch_n,weight):
        assert isinstance(cur_batch, InstanceBatch)
        assert isinstance(cur_batch_p, InstanceBatch)
        assert isinstance(cur_batch_n, InstanceBatch)
        feed_dict = {
            self.que_lens_x: cur_batch.que_lens,
            self.ans_lens_x: cur_batch.ans_lens,
            self.que_lens_xp: cur_batch_p.que_lens,
            self.ans_lens_xp: cur_batch_p.ans_lens,
            self.que_lens_xn: cur_batch_n.que_lens,
            self.ans_lens_xn: cur_batch_n.ans_lens,
            self.label_sequence_change_x:cur_batch.label_tag_change,
            self.label_sequence_change_xp:cur_batch_p.label_tag_change,
            self.label_sequence_change_xn:cur_batch_n.label_tag_change,
            self.len_option_sequence_x:cur_batch.len_tag,#the length of question's options
            self.len_option_sequence_xp:cur_batch_p.len_tag,#the length of question's options
            self.len_option_sequence_xn:cur_batch_n.len_tag,#the length of question's options
            self.word_option_length_x:cur_batch.word_emb_option_lens, #the legth of every option word embedding
            self.word_option_length_xp:cur_batch_p.word_emb_option_lens, #the legth of every option word embedding
            self.word_option_length_xn:cur_batch_n.word_emb_option_lens, #the legth of every option word embedding
            self.in_ans_append_x:cur_batch.que_with_ans,
            self.in_ans_append_xp:cur_batch_p.que_with_ans,
            self.in_ans_append_xn:cur_batch_n.que_with_ans,
            self.in_option_ww2v_index_matrix_x: cur_batch.option_ww2v_index_matrix,
            self.in_option_ww2v_index_matrix_xp: cur_batch_p.option_ww2v_index_matrix,
            self.in_option_ww2v_index_matrix_xn: cur_batch_n.option_ww2v_index_matrix,
            self.in_ans_cw2v_index_matrix_x: cur_batch.ans_cw2v_index_matrix,
            self.in_ans_ww2v_index_matrix_x: cur_batch.ans_ww2v_index_matrix,
            self.in_que_cw2v_index_matrix_x: cur_batch.que_cw2v_index_matrix,
            self.in_que_ww2v_index_matrix_x: cur_batch.que_ww2v_index_matrix,
            self.in_ans_cw2v_index_matrix_xp: cur_batch_p.ans_cw2v_index_matrix,
            self.in_ans_ww2v_index_matrix_xp: cur_batch_p.ans_ww2v_index_matrix,
            self.in_que_cw2v_index_matrix_xp: cur_batch_p.que_cw2v_index_matrix,
            self.in_que_ww2v_index_matrix_xp: cur_batch_p.que_ww2v_index_matrix,
            self.in_ans_cw2v_index_matrix_xn: cur_batch_n.ans_cw2v_index_matrix,
            self.in_ans_ww2v_index_matrix_xn: cur_batch_n.ans_ww2v_index_matrix,
            self.in_que_cw2v_index_matrix_xn: cur_batch_n.que_cw2v_index_matrix,
            self.in_que_ww2v_index_matrix_xn: cur_batch_n.que_ww2v_index_matrix,
            self.in_que_indicate_target_matrix_x:
                cur_batch.que_indicate_target_matrix,
            self.in_ans_indicate_target_matrix_x:
                cur_batch.ans_indicate_target_matrix,
            self.in_que_sentiment_polarity_matrix_x:
                cur_batch.que_sentiment_polarity_matrix,
            self.in_ans_sentiment_polarity_matrix_x:
                cur_batch.ans_sentiment_polarity_matrix,
            self.que_skeleton_label_x: cur_batch.que_skeleton_label_matrix,
            self.in_que_indicate_target_matrix_xp:
                cur_batch_p.que_indicate_target_matrix,
            self.in_ans_indicate_target_matrix_xp:
                cur_batch_p.ans_indicate_target_matrix,
            self.in_que_sentiment_polarity_matrix_xp:
                cur_batch_p.que_sentiment_polarity_matrix,
            self.in_ans_sentiment_polarity_matrix_xp:
                cur_batch_p.ans_sentiment_polarity_matrix,
            self.que_skeleton_label_xp: cur_batch_p.que_skeleton_label_matrix,
            self.in_que_indicate_target_matrix_xn:
                cur_batch_n.que_indicate_target_matrix,
            self.in_ans_indicate_target_matrix_xn:
                cur_batch_n.ans_indicate_target_matrix,
            self.in_que_sentiment_polarity_matrix_xn:
                cur_batch_n.que_sentiment_polarity_matrix,
            self.in_ans_sentiment_polarity_matrix_xn:
                cur_batch_n.ans_sentiment_polarity_matrix,
            self.que_skeleton_label_xn: cur_batch_n.que_skeleton_label_matrix,
            self.weight: weight
        }
        if self.use_bert:
            feed_dict.update({
                self.que_bert_matrix_x: cur_batch.que_bert_matrix,
                self.ans_bert_matrix_x: cur_batch.ans_bert_matrix,
                self.que_bert_matrix_xp: cur_batch_p.que_bert_matrix,
                self.ans_bert_matrix_xp: cur_batch_p.ans_bert_matrix,
                self.que_bert_matrix_xn: cur_batch_n.que_bert_matrix,
                self.ans_bert_matrix_xn: cur_batch_n.ans_bert_matrix
            })
        if self.use_extra_feature:
            feed_dict.update({
                self.in_que_pos_index_matrix_x:
                    cur_batch.que_pos_index_matrix,
                self.in_que_ner_index_matrix_x:
                    cur_batch.que_ner_index_matrix,
                self.in_ans_pos_index_matrix_x:
                    cur_batch.ans_pos_index_matrix,
                self.in_ans_ner_index_matrix_x:
                    cur_batch.ans_ner_index_matrix,
                self.in_que_pos_index_matrix_xp:
                    cur_batch_p.que_pos_index_matrix,
                self.in_que_ner_index_matrix_xp:
                    cur_batch_p.que_ner_index_matrix,
                self.in_ans_pos_index_matrix_xp:
                    cur_batch_p.ans_pos_index_matrix,
                self.in_ans_ner_index_matrix_xp:
                    cur_batch_p.ans_ner_index_matrix,
                self.in_que_pos_index_matrix_xn:
                    cur_batch_n.que_pos_index_matrix,
                self.in_que_ner_index_matrix_xn:
                    cur_batch_n.que_ner_index_matrix,
                self.in_ans_pos_index_matrix_xn:
                    cur_batch_n.ans_pos_index_matrix,
                self.in_ans_ner_index_matrix_xn:
                    cur_batch_n.ans_ner_index_matrix
            })
        return feed_dict
    
    def create_model_graph(self):
        # truths = tf.get_variable(self.truths, name='truths')
        for i in range(self.max_option_length):
            locals()['labels_x'+str(i+1)] = model_utils_reuse.make_label(self.label_sequence_change_x[i], self.nb_classes)
            locals()['labels_xp'+str(i+1)] = model_utils_reuse.make_label(self.label_sequence_change_xp[i], self.nb_classes)
            locals()['labels_xn'+str(i+1)] = model_utils_reuse.make_label(self.label_sequence_change_xn[i], self.nb_classes)
        que_in_features_x = []
        que_in_features_xp = []
        que_in_features_xn = []
        for i in range(self.max_option_length):
            locals()["que_in_features_x"+str(i+1)] = []
            locals()["que_in_features_xp"+str(i+1)] = []
            locals()["que_in_features_xn"+str(i+1)] = []
        ans_in_features_x = []
        ans_in_features_xp = []
        ans_in_features_xn = []
        # feature_dim = 0

        # w2v embedding
        if self.use_w2v:
            que_char_w2v_features_x = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix_x,
                ids=self.in_que_cw2v_index_matrix_x)
            que_in_features_x.append(que_char_w2v_features_x)
            que_word_w2v_features_x = tf.nn.embedding_lookup(
                params=self.word_w2v_embedding_matrix_x,
                ids=self.in_que_ww2v_index_matrix_x)
            que_in_features_x.append(que_word_w2v_features_x)
            for i in range(self.max_option_length):
                locals()["que_in_features_x" + str(i + 1)].append(que_char_w2v_features_x)
                locals()["que_in_features_x" + str(i + 1)].append(que_word_w2v_features_x)
            
            que_char_w2v_features_xp = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix_xp,
                ids=self.in_que_cw2v_index_matrix_xp)
            que_in_features_xp.append(que_char_w2v_features_xp)
            que_word_w2v_features_xp = tf.nn.embedding_lookup(
                params=self.word_w2v_embedding_matrix_xp,
                ids=self.in_que_ww2v_index_matrix_xp)
            que_in_features_xp.append(que_word_w2v_features_xp)
            for i in range(self.max_option_length):
                locals()["que_in_features_xp" + str(i + 1)].append(que_char_w2v_features_xp)
                locals()["que_in_features_xp" + str(i + 1)].append(que_word_w2v_features_xp)
            
            que_char_w2v_features_xn = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix_xn,
                ids=self.in_que_cw2v_index_matrix_xn)
            que_in_features_xn.append(que_char_w2v_features_xn)
            que_word_w2v_features_xn = tf.nn.embedding_lookup(
                params=self.word_w2v_embedding_matrix_xn,
                ids=self.in_que_ww2v_index_matrix_xn)
            que_in_features_xn.append(que_word_w2v_features_xn)
            for i in range(self.max_option_length):
                locals()["que_in_features_xn" + str(i + 1)].append(que_char_w2v_features_xn)
                locals()["que_in_features_xn" + str(i + 1)].append(que_word_w2v_features_xn)
            
            ans_char_w2v_features_x = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix_x,
                ids=self.in_ans_cw2v_index_matrix_x)
            ans_in_features_x.append(ans_char_w2v_features_x)
            ans_word_w2v_features_x = tf.nn.embedding_lookup(
                params=self.word_w2v_embedding_matrix_x,
                ids=self.in_ans_ww2v_index_matrix_x)
            ans_in_features_x.append(ans_word_w2v_features_x)
            
            ans_char_w2v_features_xp = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix_xp,
                ids=self.in_ans_cw2v_index_matrix_xp)
            ans_in_features_xp.append(ans_char_w2v_features_xp)
            ans_word_w2v_features_xp = tf.nn.embedding_lookup(
                params=self.word_w2v_embedding_matrix_xp,
                ids=self.in_ans_ww2v_index_matrix_xp)
            ans_in_features_xp.append(ans_word_w2v_features_xp)
            
            ans_char_w2v_features_xn = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix_xn,
                ids=self.in_ans_cw2v_index_matrix_xn)
            ans_in_features_xn.append(ans_char_w2v_features_xn)
            ans_word_w2v_features_xn = tf.nn.embedding_lookup(
                params=self.word_w2v_embedding_matrix_xn,
                ids=self.in_ans_ww2v_index_matrix_xn)
            ans_in_features_xn.append(ans_word_w2v_features_xn)
            
            que_and_ans_x = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix_x,
                ids=self.in_ans_append_x)
            ans_in_features_x.append(que_and_ans_x)
            
            que_and_ans_xp = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix_xp,
                ids=self.in_ans_append_xp)
            ans_in_features_xp.append(que_and_ans_xp)
            
            que_and_ans_xn = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix_xn,
                ids=self.in_ans_append_xn)
            ans_in_features_xn.append(que_and_ans_xn)

        # bert embedding
        if self.use_bert:
            que_in_features_x = [self.que_bert_matrix_x]
            for i in range(self.max_option_length):
                locals()["que_in_features_x" + str(i + 1)].append([self.que_bert_matrix_x])
            ans_in_features_x = [self.ans_bert_matrix_x]
            que_in_features_xp = [self.que_bert_matrix_xp]
            for i in range(self.max_option_length):
                locals()["que_in_features_xp" + str(i + 1)].append([self.que_bert_matrix_xp])
            ans_in_features_xp = [self.ans_bert_matrix_xp]
            que_in_features_xn = [self.que_bert_matrix_xn]
            for i in range(self.max_option_length):
                locals()["que_in_features_xn" + str(i + 1)].append([self.que_bert_matrix_xn])
            ans_in_features_xn = [self.ans_bert_matrix_xn]
        # que_bert_features = tf.nn.embedding_lookup(params=self.bert_embedding_matrix, ids=self.que_ids)
        # ans_bert_features = tf.nn.embedding_lookup(params=self.bert_embedding_matrix, ids=self.ans_ids)

        # add extra features
        if self.use_extra_feature:
            indicate_ner_matrix = tf.get_variable(
                name='indicate_ner_embedding',
                shape=[self.ner_dict_size, self.extra_feature_dim],
                trainable=True,
                dtype=tf.float32)
            indicate_pos_matrix = tf.get_variable(
                name='indicate_pos_embedding',
                shape=[self.pos_dict_size, self.extra_feature_dim],
                trainable=True,
                dtype=tf.float32)
            que_indicate_ner_features_x = tf.nn.embedding_lookup(
                params=indicate_ner_matrix, ids=self.in_que_ner_index_matrix_x)
            ans_indicate_ner_features_x = tf.nn.embedding_lookup(
                params=indicate_ner_matrix, ids=self.in_ans_ner_index_matrix_x)
            que_indicate_pos_features_x = tf.nn.embedding_lookup(
                params=indicate_pos_matrix, ids=self.in_que_pos_index_matrix_x)
            ans_indicate_pos_features_x = tf.nn.embedding_lookup(
                params=indicate_pos_matrix, ids=self.in_ans_pos_index_matrix_x)
            que_in_features_x.append(que_indicate_ner_features_x)
            que_in_features_x.append(que_indicate_pos_features_x)
            for i in range(self.max_option_length):
                locals()["que_in_features_x"+str(i+1)].append(que_indicate_ner_features_x)
                locals()["que_in_features_x" + str(i + 1)].append(que_indicate_pos_features_x)
            ans_in_features_x.append(ans_indicate_ner_features_x)
            ans_in_features_x.append(ans_indicate_pos_features_x)

            que_indicate_ner_features_xp = tf.nn.embedding_lookup(
                params=indicate_ner_matrix, ids=self.in_que_ner_index_matrix_xp)
            ans_indicate_ner_features_xp = tf.nn.embedding_lookup(
                params=indicate_ner_matrix, ids=self.in_ans_ner_index_matrix_xp)
            que_indicate_pos_features_xp = tf.nn.embedding_lookup(
                params=indicate_pos_matrix, ids=self.in_que_pos_index_matrix_xp)
            ans_indicate_pos_features_xp = tf.nn.embedding_lookup(
                params=indicate_pos_matrix, ids=self.in_ans_pos_index_matrix_xp)
            que_in_features_xp.append(que_indicate_ner_features_xp)
            que_in_features_xp.append(que_indicate_pos_features_xp)
            for i in range(self.max_option_length):
                locals()["que_in_features_xp"+str(i+1)].append(que_indicate_ner_features_xp)
                locals()["que_in_features_xp" + str(i + 1)].append(que_indicate_pos_features_xp)
            ans_in_features_xp.append(ans_indicate_ner_features_xp)
            ans_in_features_xp.append(ans_indicate_pos_features_xp)
            
        
            que_indicate_ner_features_xn = tf.nn.embedding_lookup(
                params=indicate_ner_matrix, ids=self.in_que_ner_index_matrix_xn)
            ans_indicate_ner_features_xn = tf.nn.embedding_lookup(
                params=indicate_ner_matrix, ids=self.in_ans_ner_index_matrix_xn)
            que_indicate_pos_features_xn = tf.nn.embedding_lookup(
                params=indicate_pos_matrix, ids=self.in_que_pos_index_matrix_xn)
            ans_indicate_pos_features_xn = tf.nn.embedding_lookup(
                params=indicate_pos_matrix, ids=self.in_ans_pos_index_matrix_xn)
            que_in_features_xn.append(que_indicate_ner_features_xn)
            que_in_features_xn.append(que_indicate_pos_features_xn)
            for i in range(self.max_option_length):
                locals()["que_in_features_xn"+str(i+1)].append(que_indicate_ner_features_xn)
                locals()["que_in_features_xn" + str(i + 1)].append(que_indicate_pos_features_xn)
            ans_in_features_xn.append(ans_indicate_ner_features_xn)
            ans_in_features_xn.append(ans_indicate_pos_features_xn)

        # indicate-target vectors
        indicate_target_matrix = np.concatenate(
            [np.zeros([1, 30]), 0.3 * np.ones([1, 30])], axis=0)
        indicate_target_matrix = tf.Variable(indicate_target_matrix,
                                             trainable=True,
                                             name="indicate_target_embedding",
                                             dtype=tf.float32)
        que_indicate_target_features_x = tf.nn.embedding_lookup(
            params=indicate_target_matrix,
            ids=self.in_que_indicate_target_matrix_x)
        ans_indicate_target_features_x = tf.nn.embedding_lookup(
            params=indicate_target_matrix,
            ids=self.in_ans_indicate_target_matrix_x)
        que_in_features_x.append(que_indicate_target_features_x)
        for i in range(self.max_option_length):
            locals()["que_in_features_x" + str(i + 1)].append(que_indicate_target_features_x)
        ans_in_features_x.append(ans_indicate_target_features_x)
        
        que_indicate_target_features_xp = tf.nn.embedding_lookup(
            params=indicate_target_matrix,
            ids=self.in_que_indicate_target_matrix_xp)
        ans_indicate_target_features_xp = tf.nn.embedding_lookup(
            params=indicate_target_matrix,
            ids=self.in_ans_indicate_target_matrix_xp)
        que_in_features_xp.append(que_indicate_target_features_xp)
        for i in range(self.max_option_length):
            locals()["que_in_features_xp" + str(i + 1)].append(que_indicate_target_features_xp)
        ans_in_features_xp.append(ans_indicate_target_features_xp)
        print('que_indicate_target_features shape:',
              que_indicate_target_features_xp)
        
        que_indicate_target_features_xn = tf.nn.embedding_lookup(
            params=indicate_target_matrix,
            ids=self.in_que_indicate_target_matrix_xn)
        ans_indicate_target_features_xn = tf.nn.embedding_lookup(
            params=indicate_target_matrix,
            ids=self.in_ans_indicate_target_matrix_xn)
        que_in_features_xn.append(que_indicate_target_features_xn)
        for i in range(self.max_option_length):
            locals()["que_in_features_xn" + str(i + 1)].append(que_indicate_target_features_xn)
        ans_in_features_xn.append(ans_indicate_target_features_xn)

        # sentiment-polarity vectors
        # sentiment-polarity map ,keep sentiment's location, complete polarity flip model
        sentiment_polarity_matrix = np.concatenate(
            [np.identity(3) for i in range(self.sentiment_polarity_multiple)],
            axis=1)
        sentiment_polarity_matrix = tf.Variable(
            sentiment_polarity_matrix,
            name="sentiment_polarity_matrix",
            trainable=False,
            dtype=tf.float32)
        ans_sentiment_polarity_features_x = tf.nn.embedding_lookup(
            params=sentiment_polarity_matrix,
            ids=self.in_ans_sentiment_polarity_matrix_x)
        que_sentiment_polarity_features_x = tf.nn.embedding_lookup(
            params=sentiment_polarity_matrix,
            ids=self.in_que_sentiment_polarity_matrix_x)
        que_in_features_x.append(que_sentiment_polarity_features_x)
        for i in range(self.max_option_length):
            locals()["que_in_features_x" + str(i + 1)].append(que_sentiment_polarity_features_x)
        ans_in_features_x.append(ans_sentiment_polarity_features_x)
        
        ans_sentiment_polarity_features_xp = tf.nn.embedding_lookup(
            params=sentiment_polarity_matrix,
            ids=self.in_ans_sentiment_polarity_matrix_xp)
        que_sentiment_polarity_features_xp = tf.nn.embedding_lookup(
            params=sentiment_polarity_matrix,
            ids=self.in_que_sentiment_polarity_matrix_xp)
        que_in_features_xp.append(que_sentiment_polarity_features_xp)
        for i in range(self.max_option_length):
            locals()["que_in_features_xp" + str(i + 1)].append(que_sentiment_polarity_features_xp)
        ans_in_features_xp.append(ans_sentiment_polarity_features_xp)
        
        ans_sentiment_polarity_features_xn = tf.nn.embedding_lookup(
            params=sentiment_polarity_matrix,
            ids=self.in_ans_sentiment_polarity_matrix_xn)
        que_sentiment_polarity_features_xn = tf.nn.embedding_lookup(
            params=sentiment_polarity_matrix,
            ids=self.in_que_sentiment_polarity_matrix_xn)
        que_in_features_xn.append(que_sentiment_polarity_features_xn)
        for i in range(self.max_option_length):
            locals()["que_in_features_xn" + str(i + 1)].append(que_sentiment_polarity_features_xn)
        ans_in_features_xn.append(ans_sentiment_polarity_features_xn)
        
        
        # [batch_size, question_len, dim]
        in_question_repr_x = tf.concat(axis=2, values=que_in_features_x)
        in_question_repr_xp = tf.concat(axis=2, values=que_in_features_xp)
        in_question_repr_xn = tf.concat(axis=2, values=que_in_features_xn)
        
        
        for i in range(self.max_option_length):
            in_option_x = self.in_option_ww2v_index_matrix_x[i]
            name_word = "option"+str(i+1)+"_word_w2v_features_x"
            #name_word_id = "in_option_ww2v_index_matrix"+str(i+1)
            locals()["option"+str(i+1)+"_word_w2v_features_x"] = tf.nn.embedding_lookup(
            params=self.word_w2v_embedding_matrix_x,
            ids=in_option_x)
            in_option_xp = self.in_option_ww2v_index_matrix_xp[i]
            name_word = "option"+str(i+1)+"_word_w2v_features_xp"
            #name_word_id = "in_option_ww2v_index_matrix"+str(i+1)
            locals()["option"+str(i+1)+"_word_w2v_features_xp"] = tf.nn.embedding_lookup(
            params=self.word_w2v_embedding_matrix_xp,
            ids=in_option_xp)
            in_option_xn = self.in_option_ww2v_index_matrix_xn[i]
            name_word = "option"+str(i+1)+"_word_w2v_features_xn"
            #name_word_id = "in_option_ww2v_index_matrix"+str(i+1)
            locals()["option"+str(i+1)+"_word_w2v_features_xn"] = tf.nn.embedding_lookup(
            params=self.word_w2v_embedding_matrix_xn,
            ids=in_option_xn)
            
        for i in range(self.max_option_length):
            locals()["que_in_features_x"+str(i+1)].append(locals()["option"+str(i+1)+"_word_w2v_features_x"])
            locals()["in_question_repr_x"+str(i+1)] = tf.concat(axis=2, values=locals()["que_in_features_x"+str(i+1)])
            locals()["in_question_repr_x" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["in_question_repr_x" + str(i + 1)],
                                                         self.dropout_rate,
                                                         self.is_training)
            locals()["que_in_features_xp"+str(i+1)].append(locals()["option"+str(i+1)+"_word_w2v_features_xp"])
            locals()["in_question_repr_xp"+str(i+1)] = tf.concat(axis=2, values=locals()["que_in_features_xp"+str(i+1)])
            locals()["in_question_repr_xp" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["in_question_repr_xp" + str(i + 1)],
                                                         self.dropout_rate,
                                                         self.is_training)
            locals()["que_in_features_xn"+str(i+1)].append(locals()["option"+str(i+1)+"_word_w2v_features_xn"])
            locals()["in_question_repr_xn"+str(i+1)] = tf.concat(axis=2, values=locals()["que_in_features_xn"+str(i+1)])
            locals()["in_question_repr_xn" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["in_question_repr_xn" + str(i + 1)],
                                                         self.dropout_rate,
                                                         self.is_training)
        # [batch_size, question_len, dim]
        in_answer_repr_x = tf.concat(axis=2, values=ans_in_features_x)
        in_answer_repr_xp = tf.concat(axis=2, values=ans_in_features_xp)
        in_answer_repr_xn = tf.concat(axis=2, values=ans_in_features_xn)
        print("in_question_repr shape:", in_question_repr_x.shape)
        print("in_question_repr1 shape:", locals()["in_question_repr_x" + str(1)].shape)
        print("in_answer_repr shape:", in_answer_repr_x.shape)
        in_answer_repr_x = model_utils_reuse.dropout_layer(in_answer_repr_x,
                                                   self.dropout_rate,
                                                   self.is_training)
        in_answer_repr_xp = model_utils_reuse.dropout_layer(in_answer_repr_xp,
                                                   self.dropout_rate,
                                                   self.is_training)
        in_answer_repr_xn = model_utils_reuse.dropout_layer(in_answer_repr_xn,
                                                   self.dropout_rate,
                                                   self.is_training)
        # TODO: complete skeleton information indicator
        indicate_skeleton_matrix_x = self.que_skeleton_label_x
        indicate_skeleton_matrix_xp = self.que_skeleton_label_xp
        indicate_skeleton_matrix_xn = self.que_skeleton_label_xn
        # basic encode using bi-lstm
        assert self.rnn_unit == 'lstm' or self.rnn_unit == 'gru'
        
        
        answer_bi_x = model_utils_reuse.my_rnn_layer(input_reps=in_answer_repr_x,
                                             rnn_dim=self.rnn_dim,
                                             rnn_unit=self.rnn_unit,
                                             input_lengths=self.ans_lens_x,
                                             scope_name='basic_encode',
                                             is_training=self.is_training,
                                             reuse=False)
        answer_bi_last_x = model_utils_reuse.collect_final_step_of_lstm(
            answer_bi_x, self.ans_lens_x-1)
        self.answer_repr_x = answer_bi_last_x
        
        answer_bi_xp = model_utils_reuse.my_rnn_layer(input_reps=in_answer_repr_xp,
                                             rnn_dim=self.rnn_dim,
                                             rnn_unit=self.rnn_unit,
                                             input_lengths=self.ans_lens_xp,
                                             scope_name='basic_encode',
                                             is_training=self.is_training,
                                             reuse=True)
        answer_bi_last_xp = model_utils_reuse.collect_final_step_of_lstm(
            answer_bi_xp, self.ans_lens_xp-1)
        self.answer_repr_xp = answer_bi_last_xp
        
        answer_bi_xn = model_utils_reuse.my_rnn_layer(input_reps=in_answer_repr_xn,
                                             rnn_dim=self.rnn_dim,
                                             rnn_unit=self.rnn_unit,
                                             input_lengths=self.ans_lens_xn,
                                             scope_name='basic_encode',
                                             is_training=self.is_training,
                                             reuse=True)
        answer_bi_last_xn = model_utils_reuse.collect_final_step_of_lstm(
            answer_bi_xn, self.ans_lens_xn-1)
        self.answer_repr_xn = answer_bi_last_xn
##################################################################################################
        for i in range(self.max_option_length):
            if(i==0):
                locals()["question_bi_x"+str(i+1)] = model_utils_reuse.my_rnn_layer(input_reps=locals()["in_question_repr_x"+str(i+1)],
                                                                                       rnn_dim=self.rnn_dim,
                                                                                       rnn_unit=self.rnn_unit,
                                                                                       input_lengths=self.que_lens_x,
                                                                                       scope_name='basic_encode',
                                                                                       is_training=self.is_training,
                                                                                       reuse=True)
                locals()["question_bi_x" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["question_bi_x"+str(i+1)], self.dropout_rate,
                                                                                            self.is_training)
                locals()["question_target_repr_x"+str(i+1)] = model_utils_reuse.get_target_representation(
                                                                                                            locals()["question_bi_x" + str(i + 1)], 
                                                                                                            self.in_que_indicate_target_matrix_x,
                                                                                                            "target_repr",reuse=False)
                locals()["question_bi_last_x"+str(i+1)] = model_utils_reuse.collect_final_step_of_lstm(locals()["question_bi_x" + str(i + 1)], self.que_lens_x)
                locals()["answer_bi_x"+str(i+1)] = model_utils_reuse.sentiment_polarity_flip(
                                                                                              answer_bi_x, ans_sentiment_polarity_features_x, 
                                                                                              locals()["question_target_repr_x"+str(i+1)],
                                                                                              self.sentiment_polarity_multiple, self.attention_dim,
                                                                                              "sentiment_polarity_flip",reuse=False)
                locals()["answer_bi_x" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["answer_bi_x" + str(i + 1)], self.dropout_rate,
                                                                                        self.is_training)
                locals()["question_skeleton_repr_x"+str(i+1)] = locals()["question_target_repr_x"+str(i+1)]
                locals()["question_semantic_repr_x"+str(i+1)] = model_utils_reuse.generate_semantic_representation(
                                                                                                                    locals()["question_skeleton_repr_x"+str(i+1)], 
                                                                                                                    locals()["question_bi_x" + str(i + 1)], 
                                                                                                                    self.que_lens_x,
                                                                                                                    self.attention_dim,
                                                                                                                    'get_semantic_representation_x'+str(i+1))
                locals()["question_aware_repr_x"+str(i+1)] = model_utils_reuse.get_aware_repr(
                locals()["answer_bi_x" + str(i + 1)], locals()["question_skeleton_repr_x"+str(i+1)], locals()["question_semantic_repr_x"+str(i+1)],
                self.nb_hops, self.attention_dim, self.rnn_dim, self.ans_max_len,
                self.ans_lens_x, self.lambda_l2,"get_aware",reuse=False)
                locals()["question_aware_repr_x"+str(i+1)] = model_utils_reuse.dropout_layer(
                locals()["question_aware_repr_x"+str(i+1)], self.dropout_rate, self.is_training)
                locals()["answer_aware_repr_x"+str(i+1)] = model_utils_reuse.multi_hop_match(
                                                          answer_bi_last_x, locals()["question_bi_x" + str(i + 1)], self.nb_hops, self.rnn_dim,
                                                          self.attention_dim, 'answer_aware', self.que_max_len, self.que_lens_x,
                                                          self.lambda_l2,reuse=False)
                
                locals()["question_bi_xp"+str(i+1)] = model_utils_reuse.my_rnn_layer(input_reps=locals()["in_question_repr_xp"+str(i+1)],
                                                                                       rnn_dim=self.rnn_dim,
                                                                                       rnn_unit=self.rnn_unit,
                                                                                       input_lengths=self.que_lens_xp,
                                                                                       scope_name='basic_encode',
                                                                                       is_training=self.is_training,
                                                                                       reuse=True)
                locals()["question_bi_xp" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["question_bi_xp"+str(i+1)], self.dropout_rate,
                                                                                            self.is_training)
                locals()["question_target_repr_xp"+str(i+1)] = model_utils_reuse.get_target_representation(
                                                                                                            locals()["question_bi_xp" + str(i + 1)], 
                                                                                                            self.in_que_indicate_target_matrix_xp,
                                                                                                            "target_repr",reuse=True)
                locals()["question_bi_last_xp"+str(i+1)] = model_utils_reuse.collect_final_step_of_lstm(locals()["question_bi_xp" + str(i + 1)], self.que_lens_xp)
                locals()["answer_bi_xp"+str(i+1)] = model_utils_reuse.sentiment_polarity_flip(
                                                                                              answer_bi_xp, ans_sentiment_polarity_features_xp, 
                                                                                              locals()["question_target_repr_xp"+str(i+1)],
                                                                                              self.sentiment_polarity_multiple, self.attention_dim,
                                                                                              "sentiment_polarity_flip",reuse=True)
                locals()["answer_bi_xp" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["answer_bi_xp" + str(i + 1)], self.dropout_rate,
                                                                                        self.is_training)
                locals()["question_skeleton_repr_xp"+str(i+1)] = locals()["question_target_repr_xp"+str(i+1)]
                locals()["question_semantic_repr_xp"+str(i+1)] = model_utils_reuse.generate_semantic_representation(
                                                                                                                    locals()["question_skeleton_repr_xp"+str(i+1)], 
                                                                                                                    locals()["question_bi_xp" + str(i + 1)], 
                                                                                                                    self.que_lens_xp,
                                                                                                                    self.attention_dim,
                                                                                                                    'get_semantic_representation_xp'+str(i+1))
                locals()["question_aware_repr_xp"+str(i+1)] = model_utils_reuse.get_aware_repr(
                locals()["answer_bi_xp" + str(i + 1)], locals()["question_skeleton_repr_xp"+str(i+1)], locals()["question_semantic_repr_xp"+str(i+1)],
                self.nb_hops, self.attention_dim, self.rnn_dim, self.ans_max_len,
                self.ans_lens_xp, self.lambda_l2,"get_aware",reuse=True)
                locals()["question_aware_repr_xp"+str(i+1)] = model_utils_reuse.dropout_layer(
                locals()["question_aware_repr_xp"+str(i+1)], self.dropout_rate, self.is_training)
                locals()["answer_aware_repr_xp"+str(i+1)] = model_utils_reuse.multi_hop_match(
                                                          answer_bi_last_xp, locals()["question_bi_xp" + str(i + 1)], self.nb_hops, self.rnn_dim,
                                                          self.attention_dim, 'answer_aware', self.que_max_len, self.que_lens_xp,
                                                          self.lambda_l2,reuse=True)
                                                          
                locals()["question_bi_xn"+str(i+1)] = model_utils_reuse.my_rnn_layer(input_reps=locals()["in_question_repr_xn"+str(i+1)],
                                                                                       rnn_dim=self.rnn_dim,
                                                                                       rnn_unit=self.rnn_unit,
                                                                                       input_lengths=self.que_lens_xn,
                                                                                       scope_name='basic_encode',
                                                                                       is_training=self.is_training,
                                                                                       reuse=True)
                locals()["question_bi_xn" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["question_bi_xn"+str(i+1)], self.dropout_rate,
                                                                                            self.is_training)
                locals()["question_target_repr_xn"+str(i+1)] = model_utils_reuse.get_target_representation(
                                                                                                            locals()["question_bi_xn" + str(i + 1)], 
                                                                                                            self.in_que_indicate_target_matrix_xn,
                                                                                                            "target_repr",reuse=True)
                locals()["question_bi_last_xn"+str(i+1)] = model_utils_reuse.collect_final_step_of_lstm(locals()["question_bi_xn" + str(i + 1)], self.que_lens_xn)
                locals()["answer_bi_xn"+str(i+1)] = model_utils_reuse.sentiment_polarity_flip(
                                                                                              answer_bi_xn, ans_sentiment_polarity_features_xn, 
                                                                                              locals()["question_target_repr_xn"+str(i+1)],
                                                                                              self.sentiment_polarity_multiple, self.attention_dim,
                                                                                              "sentiment_polarity_flip",reuse=True)
                locals()["answer_bi_xn" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["answer_bi_xn" + str(i + 1)], self.dropout_rate,
                                                                                        self.is_training)
                locals()["question_skeleton_repr_xn"+str(i+1)] = locals()["question_target_repr_xn"+str(i+1)]
                locals()["question_semantic_repr_xn"+str(i+1)] = model_utils_reuse.generate_semantic_representation(
                                                                                                                    locals()["question_skeleton_repr_xn"+str(i+1)], 
                                                                                                                    locals()["question_bi_xn" + str(i + 1)], 
                                                                                                                    self.que_lens_xn,
                                                                                                                    self.attention_dim,
                                                                                                                    'get_semantic_representation_xn'+str(i+1))
                locals()["question_aware_repr_xn"+str(i+1)] = model_utils_reuse.get_aware_repr(
                locals()["answer_bi_xn" + str(i + 1)], locals()["question_skeleton_repr_xn"+str(i+1)], locals()["question_semantic_repr_xn"+str(i+1)],
                self.nb_hops, self.attention_dim, self.rnn_dim, self.ans_max_len,
                self.ans_lens_xn, self.lambda_l2,"get_aware",reuse=True)
                locals()["question_aware_repr_xn"+str(i+1)] = model_utils_reuse.dropout_layer(
                locals()["question_aware_repr_xn"+str(i+1)], self.dropout_rate, self.is_training)
                locals()["answer_aware_repr_xn"+str(i+1)] = model_utils_reuse.multi_hop_match(
                                                          answer_bi_last_xn, locals()["question_bi_xn" + str(i + 1)], self.nb_hops, self.rnn_dim,
                                                          self.attention_dim, 'answer_aware', self.que_max_len, self.que_lens_xn,
                                                          self.lambda_l2,reuse=True)
            else:
                locals()["question_bi_x"+str(i+1)] = model_utils_reuse.my_rnn_layer(input_reps=locals()["in_question_repr_x"+str(i+1)],
                                                                                       rnn_dim=self.rnn_dim,
                                                                                       rnn_unit=self.rnn_unit,
                                                                                       input_lengths=self.que_lens_x,
                                                                                       scope_name='basic_encode',
                                                                                       is_training=self.is_training,
                                                                                       reuse=True)
                locals()["question_bi_x" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["question_bi_x"+str(i+1)], self.dropout_rate,
                                                                                            self.is_training)
                locals()["question_target_repr_x"+str(i+1)] = model_utils_reuse.get_target_representation(
                                                                                                            locals()["question_bi_x" + str(i + 1)], 
                                                                                                            self.in_que_indicate_target_matrix_x,
                                                                                                            "target_repr",reuse=True)
                locals()["question_bi_last_x"+str(i+1)] = model_utils_reuse.collect_final_step_of_lstm(locals()["question_bi_x" + str(i + 1)], self.que_lens_x)
                locals()["answer_bi_x"+str(i+1)] = model_utils_reuse.sentiment_polarity_flip(
                                                                                              answer_bi_x, ans_sentiment_polarity_features_x, 
                                                                                              locals()["question_target_repr_x"+str(i+1)],
                                                                                              self.sentiment_polarity_multiple, self.attention_dim,
                                                                                              "sentiment_polarity_flip",reuse=True)
                locals()["answer_bi_x" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["answer_bi_x" + str(i + 1)], self.dropout_rate,
                                                                                        self.is_training)
                locals()["question_skeleton_repr_x"+str(i+1)] = locals()["question_target_repr_x"+str(i+1)]
                locals()["question_semantic_repr_x"+str(i+1)] = model_utils_reuse.generate_semantic_representation(
                                                                                                                    locals()["question_skeleton_repr_x"+str(i+1)], 
                                                                                                                    locals()["question_bi_x" + str(i + 1)], 
                                                                                                                    self.que_lens_x,
                                                                                                                    self.attention_dim,
                                                                                                                    'get_semantic_representation_x'+str(i+1))
                locals()["question_aware_repr_x"+str(i+1)] = model_utils_reuse.get_aware_repr(
                locals()["answer_bi_x" + str(i + 1)], locals()["question_skeleton_repr_x"+str(i+1)], locals()["question_semantic_repr_x"+str(i+1)],
                self.nb_hops, self.attention_dim, self.rnn_dim, self.ans_max_len,
                self.ans_lens_x, self.lambda_l2,"get_aware",reuse=True)
                locals()["question_aware_repr_x"+str(i+1)] = model_utils_reuse.dropout_layer(
                locals()["question_aware_repr_x"+str(i+1)], self.dropout_rate, self.is_training)
                locals()["answer_aware_repr_x"+str(i+1)] = model_utils_reuse.multi_hop_match(
                                                          answer_bi_last_x, locals()["question_bi_x" + str(i + 1)], self.nb_hops, self.rnn_dim,
                                                          self.attention_dim, 'answer_aware', self.que_max_len, self.que_lens_x,
                                                          self.lambda_l2,reuse=True)
                
                locals()["question_bi_xp"+str(i+1)] = model_utils_reuse.my_rnn_layer(input_reps=locals()["in_question_repr_xp"+str(i+1)],
                                                                                       rnn_dim=self.rnn_dim,
                                                                                       rnn_unit=self.rnn_unit,
                                                                                       input_lengths=self.que_lens_xp,
                                                                                       scope_name='basic_encode',
                                                                                       is_training=self.is_training,
                                                                                       reuse=True)
                locals()["question_bi_xp" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["question_bi_xp"+str(i+1)], self.dropout_rate,
                                                                                            self.is_training)
                locals()["question_target_repr_xp"+str(i+1)] = model_utils_reuse.get_target_representation(
                                                                                                            locals()["question_bi_xp" + str(i + 1)], 
                                                                                                            self.in_que_indicate_target_matrix_xp,
                                                                                                            "target_repr",reuse=True)
                locals()["question_bi_last_xp"+str(i+1)] = model_utils_reuse.collect_final_step_of_lstm(locals()["question_bi_xp" + str(i + 1)], self.que_lens_xp)
                locals()["answer_bi_xp"+str(i+1)] = model_utils_reuse.sentiment_polarity_flip(
                                                                                              answer_bi_xp, ans_sentiment_polarity_features_xp, 
                                                                                              locals()["question_target_repr_xp"+str(i+1)],
                                                                                              self.sentiment_polarity_multiple, self.attention_dim,
                                                                                              "sentiment_polarity_flip",reuse=True)
                locals()["answer_bi_xp" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["answer_bi_xp" + str(i + 1)], self.dropout_rate,
                                                                                        self.is_training)
                locals()["question_skeleton_repr_xp"+str(i+1)] = locals()["question_target_repr_xp"+str(i+1)]
                locals()["question_semantic_repr_xp"+str(i+1)] = model_utils_reuse.generate_semantic_representation(
                                                                                                                    locals()["question_skeleton_repr_xp"+str(i+1)], 
                                                                                                                    locals()["question_bi_xp" + str(i + 1)], 
                                                                                                                    self.que_lens_xp,
                                                                                                                    self.attention_dim,
                                                                                                                    'get_semantic_representation_xp'+str(i+1))
                locals()["question_aware_repr_xp"+str(i+1)] = model_utils_reuse.get_aware_repr(
                locals()["answer_bi_xp" + str(i + 1)], locals()["question_skeleton_repr_xp"+str(i+1)], locals()["question_semantic_repr_xp"+str(i+1)],
                self.nb_hops, self.attention_dim, self.rnn_dim, self.ans_max_len,
                self.ans_lens_xp, self.lambda_l2,"get_aware",reuse=True)
                locals()["question_aware_repr_xp"+str(i+1)] = model_utils_reuse.dropout_layer(
                locals()["question_aware_repr_xp"+str(i+1)], self.dropout_rate, self.is_training)
                locals()["answer_aware_repr_xp"+str(i+1)] = model_utils_reuse.multi_hop_match(
                                                          answer_bi_last_xp, locals()["question_bi_xp" + str(i + 1)], self.nb_hops, self.rnn_dim,
                                                          self.attention_dim, 'answer_aware', self.que_max_len, self.que_lens_xp,
                                                          self.lambda_l2,reuse=True)
                                                          
                locals()["question_bi_xn"+str(i+1)] = model_utils_reuse.my_rnn_layer(input_reps=locals()["in_question_repr_xn"+str(i+1)],
                                                                                       rnn_dim=self.rnn_dim,
                                                                                       rnn_unit=self.rnn_unit,
                                                                                       input_lengths=self.que_lens_xn,
                                                                                       scope_name='basic_encode',
                                                                                       is_training=self.is_training,
                                                                                       reuse=True)
                locals()["question_bi_xn" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["question_bi_xn"+str(i+1)], self.dropout_rate,
                                                                                            self.is_training)
                locals()["question_target_repr_xn"+str(i+1)] = model_utils_reuse.get_target_representation(
                                                                                                            locals()["question_bi_xn" + str(i + 1)], 
                                                                                                            self.in_que_indicate_target_matrix_xn,
                                                                                                            "target_repr",reuse=True)
                locals()["question_bi_last_xn"+str(i+1)] = model_utils_reuse.collect_final_step_of_lstm(locals()["question_bi_xn" + str(i + 1)], self.que_lens_xn)
                locals()["answer_bi_xn"+str(i+1)] = model_utils_reuse.sentiment_polarity_flip(
                                                                                              answer_bi_xn, ans_sentiment_polarity_features_xn, 
                                                                                              locals()["question_target_repr_xn"+str(i+1)],
                                                                                              self.sentiment_polarity_multiple, self.attention_dim,
                                                                                              "sentiment_polarity_flip",reuse=True)
                locals()["answer_bi_xn" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["answer_bi_xn" + str(i + 1)], self.dropout_rate,
                                                                                        self.is_training)
                locals()["question_skeleton_repr_xn"+str(i+1)] = locals()["question_target_repr_xn"+str(i+1)]
                locals()["question_semantic_repr_xn"+str(i+1)] = model_utils_reuse.generate_semantic_representation(
                                                                                                                    locals()["question_skeleton_repr_xn"+str(i+1)], 
                                                                                                                    locals()["question_bi_xn" + str(i + 1)], 
                                                                                                                    self.que_lens_xn,
                                                                                                                    self.attention_dim,
                                                                                                                    'get_semantic_representation_xn'+str(i+1))
                locals()["question_aware_repr_xn"+str(i+1)] = model_utils_reuse.get_aware_repr(
                locals()["answer_bi_xn" + str(i + 1)], locals()["question_skeleton_repr_xn"+str(i+1)], locals()["question_semantic_repr_xn"+str(i+1)],
                self.nb_hops, self.attention_dim, self.rnn_dim, self.ans_max_len,
                self.ans_lens_xn, self.lambda_l2,"get_aware",reuse=True)
                locals()["question_aware_repr_xn"+str(i+1)] = model_utils_reuse.dropout_layer(
                locals()["question_aware_repr_xn"+str(i+1)], self.dropout_rate, self.is_training)
                locals()["answer_aware_repr_xn"+str(i+1)] = model_utils_reuse.multi_hop_match(
                                                          answer_bi_last_xn, locals()["question_bi_xn" + str(i + 1)], self.nb_hops, self.rnn_dim,
                                                          self.attention_dim, 'answer_aware', self.que_max_len, self.que_lens_xn,
                                                          self.lambda_l2,reuse=True)
        for i in range(self.max_option_length):
            if(i==0):
                locals()["end_rper_x"+str(i+1)] = tf.concat([locals()["question_aware_repr_x"+str(i+1)], locals()["answer_aware_repr_x"+str(i+1)]],
                                     axis=-1)
                locals()["logits_x"+str(i+1)] = model_utils_reuse.full_connect_layer(locals()["end_rper_x"+str(i+1)],
                                                        self.nb_classes,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        scope="full_connect",
                                                        reuse=False)
                locals()["loss_x"+str(i+1)] = model_utils_reuse.compute_cross_entropy(locals()["logits_x"+str(i+1)], locals()['labels_x'+str(i+1)])
                locals()["prob_x"+str(i+1)] = tf.nn.softmax(locals()["logits_x"+str(i+1)])
                locals()["correct_x"+str(i+1)] = tf.nn.in_top_k(tf.to_float(locals()["logits_x"+str(i+1)]), self.label_sequence_change_x[i], 1)
                locals()["cor_x"+str(i+1)] = tf.reduce_sum(tf.cast(locals()["correct_x"+str(i+1)], tf.int32))
                locals()["pre_x"+str(i+1)] = tf.argmax(locals()["prob_x"+str(i+1)], 1)
                
                locals()["end_rper_xp"+str(i+1)] = tf.concat([locals()["question_aware_repr_xp"+str(i+1)], locals()["answer_aware_repr_xp"+str(i+1)]],
                                     axis=-1)
                locals()["logits_xp"+str(i+1)] = model_utils_reuse.full_connect_layer(locals()["end_rper_xp"+str(i+1)],
                                                        self.nb_classes,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        scope="full_connect",
                                                        reuse=True)
                locals()["loss_xp"+str(i+1)] = model_utils_reuse.compute_cross_entropy(locals()["logits_xp"+str(i+1)], locals()['labels_xp'+str(i+1)])
                locals()["prob_xp"+str(i+1)] = tf.nn.softmax(locals()["logits_xp"+str(i+1)])
                locals()["correct_xp"+str(i+1)] = tf.nn.in_top_k(tf.to_float(locals()["logits_xp"+str(i+1)]), self.label_sequence_change_xp[i], 1)
                locals()["cor_xp"+str(i+1)] = tf.reduce_sum(tf.cast(locals()["correct_xp"+str(i+1)], tf.int32))
                locals()["pre_xp"+str(i+1)] = tf.argmax(locals()["prob_xp"+str(i+1)], 1)
                
                locals()["end_rper_xn"+str(i+1)] = tf.concat([locals()["question_aware_repr_xn"+str(i+1)], locals()["answer_aware_repr_xn"+str(i+1)]],
                                     axis=-1)
                locals()["logits_xn"+str(i+1)] = model_utils_reuse.full_connect_layer(locals()["end_rper_xn"+str(i+1)],
                                                        self.nb_classes,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        scope="full_connect",
                                                        reuse=True)
                locals()["loss_xn"+str(i+1)] = model_utils_reuse.compute_cross_entropy(locals()["logits_xn"+str(i+1)], locals()['labels_xn'+str(i+1)])
                locals()["prob_xn"+str(i+1)] = tf.nn.softmax(locals()["logits_xn"+str(i+1)])
                locals()["correct_xn"+str(i+1)] = tf.nn.in_top_k(tf.to_float(locals()["logits_xn"+str(i+1)]), self.label_sequence_change_xn[i], 1)
                locals()["cor_xn"+str(i+1)] = tf.reduce_sum(tf.cast(locals()["correct_xn"+str(i+1)], tf.int32))
                locals()["pre_xn"+str(i+1)] = tf.argmax(locals()["prob_xn"+str(i+1)], 1)
            else:
                locals()["end_rper_x"+str(i+1)] = tf.concat([locals()["question_aware_repr_x"+str(i+1)], locals()["answer_aware_repr_x"+str(i+1)]],
                                     axis=-1)
                locals()["logits_x"+str(i+1)] = model_utils_reuse.full_connect_layer(locals()["end_rper_x"+str(i+1)],
                                                        self.nb_classes,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        scope="full_connect",
                                                        reuse=True)
                locals()["loss_x"+str(i+1)] = model_utils_reuse.compute_cross_entropy(locals()["logits_x"+str(i+1)], locals()['labels_x'+str(i+1)])
                locals()["prob_x"+str(i+1)] = tf.nn.softmax(locals()["logits_x"+str(i+1)])
                locals()["correct_x"+str(i+1)] = tf.nn.in_top_k(tf.to_float(locals()["logits_x"+str(i+1)]), self.label_sequence_change_x[i], 1)
                locals()["cor_x"+str(i+1)] = tf.reduce_sum(tf.cast(locals()["correct_x"+str(i+1)], tf.int32))
                locals()["pre_x"+str(i+1)] = tf.argmax(locals()["prob_x"+str(i+1)], 1)
                
                locals()["end_rper_xp"+str(i+1)] = tf.concat([locals()["question_aware_repr_xp"+str(i+1)], locals()["answer_aware_repr_xp"+str(i+1)]],
                                     axis=-1)
                locals()["logits_xp"+str(i+1)] = model_utils_reuse.full_connect_layer(locals()["end_rper_xp"+str(i+1)],
                                                        self.nb_classes,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        scope="full_connect",
                                                        reuse=True)
                locals()["loss_xp"+str(i+1)] = model_utils_reuse.compute_cross_entropy(locals()["logits_xp"+str(i+1)], locals()['labels_xp'+str(i+1)])
                locals()["prob_xp"+str(i+1)] = tf.nn.softmax(locals()["logits_xp"+str(i+1)])
                locals()["correct_xp"+str(i+1)] = tf.nn.in_top_k(tf.to_float(locals()["logits_xp"+str(i+1)]), self.label_sequence_change_xp[i], 1)
                locals()["cor_xp"+str(i+1)] = tf.reduce_sum(tf.cast(locals()["correct_xp"+str(i+1)], tf.int32))
                locals()["pre_xp"+str(i+1)] = tf.argmax(locals()["prob_xp"+str(i+1)], 1)
                
                locals()["end_rper_xn"+str(i+1)] = tf.concat([locals()["question_aware_repr_xn"+str(i+1)], locals()["answer_aware_repr_xn"+str(i+1)]],
                                     axis=-1)
                locals()["logits_xn"+str(i+1)] = model_utils_reuse.full_connect_layer(locals()["end_rper_xn"+str(i+1)],
                                                        self.nb_classes,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        scope="full_connect",
                                                        reuse=True)
                locals()["loss_xn"+str(i+1)] = model_utils_reuse.compute_cross_entropy(locals()["logits_xn"+str(i+1)], locals()['labels_xn'+str(i+1)])
                locals()["prob_xn"+str(i+1)] = tf.nn.softmax(locals()["logits_xn"+str(i+1)])
                locals()["correct_xn"+str(i+1)] = tf.nn.in_top_k(tf.to_float(locals()["logits_xn"+str(i+1)]), self.label_sequence_change_xn[i], 1)
                locals()["cor_xn"+str(i+1)] = tf.reduce_sum(tf.cast(locals()["correct_xn"+str(i+1)], tf.int32))
                locals()["pre_xn"+str(i+1)] = tf.argmax(locals()["prob_xn"+str(i+1)], 1)
        
        sum_count = 2
        while(sum_count<=self.max_option_length):
            
            locals()["sum_acc_x"+str(sum_count)] = locals()["cor_x"+str(1)]
            locals()["sum_loss_x" + str(sum_count)] = locals()["loss_x" + str(1)]
            locals()["hid_emb_x"+str(sum_count)] = locals()["end_rper_x"+str(1)]
            locals()["sum_acc_xp"+str(sum_count)] = locals()["cor_xp"+str(1)]
            locals()["sum_loss_xp" + str(sum_count)] = locals()["loss_xp" + str(1)]
            locals()["hid_emb_xp"+str(sum_count)] = locals()["end_rper_xp"+str(1)]
            locals()["sum_acc_xn"+str(sum_count)] = locals()["cor_xn"+str(1)]
            locals()["sum_loss_xn" + str(sum_count)] = locals()["loss_xn" + str(1)]
            locals()["hid_emb_xn"+str(sum_count)] = locals()["end_rper_xn"+str(1)]
            for i in range(1,sum_count):
                locals()["sum_acc_x"+str(sum_count)] += locals()["cor_x"+str(i+1)]
                locals()["sum_loss_x" + str(sum_count)] += locals()["loss_x" + str(i+1)]
                locals()["hid_emb_x"+str(sum_count)] += locals()["end_rper_x"+str(i+1)]
                locals()["sum_acc_xp"+str(sum_count)] += locals()["cor_xp"+str(i+1)]
                locals()["sum_loss_xp" + str(sum_count)] += locals()["loss_xp" + str(i+1)]
                locals()["hid_emb_xp"+str(sum_count)] += locals()["end_rper_xp"+str(i+1)]
                locals()["sum_acc_xn"+str(sum_count)] += locals()["cor_xn"+str(i+1)]
                locals()["sum_loss_xn" + str(sum_count)] += locals()["loss_xn" + str(i+1)]
                locals()["hid_emb_xn"+str(sum_count)] += locals()["end_rper_xn"+str(i+1)]
            sum_count=sum_count+1
        names=locals()
        for i in range(2,self.max_option_length+1):
            names["len"+str(i)] = tf.convert_to_tensor(i, dtype=tf.int32)
            names["len_float"+str(i)] = tf.convert_to_tensor(i, dtype=tf.float32)
            names["accuracy_x"+str(i)] = names["sum_acc_x"+str(i)]/names["len"+str(i)] 
            names["model_loss_x"+str(i)] = names["sum_loss_x" + str(i)]
            names["hid_x"+str(i)] = names["hid_emb_x" + str(i)]/names["len_float"+str(i)] 
            names["accuracy_xp"+str(i)] = names["sum_acc_xp"+str(i)]/names["len"+str(i)] 
            names["model_loss_xp"+str(i)] = names["sum_loss_xp" + str(i)]
            names["hid_xp"+str(i)] = names["hid_emb_xp" + str(i)]/names["len_float"+str(i)] 
            names["accuracy_xn"+str(i)] = names["sum_acc_xn"+str(i)]/names["len"+str(i)] 
            names["model_loss_xn"+str(i)] = names["sum_loss_xn" + str(i)]
            names["hid_xn"+str(i)] = names["hid_emb_xn" + str(i)]/names["len_float"+str(i)]
        
        def triplet_loss(model_anchor, model_positive, model_negative, margin):
            distance1 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_positive, 2)))
            distance2 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_negative, 2)))
            return tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0)) + 1e-9
        tri_loss = triplet_loss(names["hid_x"+str(5)],names["hid_xp"+str(5)] , names["hid_xn"+str(5)] ,2)
        self.acc = (names["accuracy_x"+str(5)]+names["accuracy_xp"+str(5)]+names["accuracy_xn"+str(5)])/3
        self.loss = (names["model_loss_x"+str(5)]+names["model_loss_xp"+str(5)]+names["model_loss_xn"+str(5)])/3+(1-self.weight)*tri_loss
        
        prediction_list_x = []
        prediction_list_xp = []
        prediction_list_xn = []
        for i in range(self.max_option_length):
            prediction_list_x.append(names["pre_x"+str(i+1)])
            prediction_list_xp.append(names["pre_xp"+str(i+1)])
            prediction_list_xn.append(names["pre_xn"+str(i+1)])
        self.prediction_ls_x = prediction_list_x
        self.prediction_ls_xp = prediction_list_xp
        self.prediction_ls_xn = prediction_list_xn
        tvars = tf.trainable_variables()
        self.tvars = tvars

        if self.lambda_l2 > 0.0:
            l2_loss = tf.add_n(
                [tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + self.lambda_l2 * l2_loss

        if self.is_training:
            trainable_variables = tf.trainable_variables()
            if self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate=self.learning_rate)
            else:
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate)
            grads = model_utils_reuse.compute_gradients(self.loss,
                                                  trainable_variables)
            # TODO: compute grad_clipper
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clipper)
            self.train_op = optimizer.apply_gradients(
                zip(grads, trainable_variables), global_step=self.global_step)
