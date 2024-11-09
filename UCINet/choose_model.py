import numpy as np
import tensorflow as tf
import model_utils_reuse

from data_stream_final import InstanceBatch


class AnswerUnderstander(object):
    def __init__(self, use_bert, use_w2v, rnn_unit, dropout_rate,
                 char_w2v_embedding_matrix_path, rnn_dim, nb_classes,
                 optimizer, learning_rate, grad_clipper, global_step, nb_hops,
                 attention_dim, is_training, use_extra_feature, ans_max_len,
                 que_max_len, extra_feature_dim, ner_dict_size, pos_dict_size,
                 lambda_l2, sentiment_polarity_multiple,
                 word_w2v_embedding_matrix_path,max_option_length):
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
        self.max_option_length = max_option_length
        # create placeholders
        self.que_lens = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.ans_lens = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.in_ans_append = tf.placeholder(tf.int32, [None, None])
        #self.len_option_sequence = tf.placeholder(tf.int32,[None])
        #self.word_option_length = tf.placeholder(tf.int32,[None,None])
        self.label_sequence_change = tf.placeholder(tf.int32, [None,None]) #dim 1 reprsent the label of option 1
        self.que_skeleton_label = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        #option
        self.in_option_ww2v_index_matrix = tf.placeholder(  #all options' word embedding
            tf.int32,[None,None,None]) #[batch_size,max_option_length,max_sequence_len]
        
        self.in_que_cw2v_index_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_ww2v_index_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_cw2v_index_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_ww2v_index_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]

        self.in_que_sentiment_polarity_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_sentiment_polarity_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_indicate_target_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_indicate_target_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        if self.use_extra_feature:
            self.in_que_pos_index_matrix = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_que_ner_index_matrix = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_ans_pos_index_matrix = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_ans_ner_index_matrix = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]

        if self.use_bert:
            self.que_bert_matrix = tf.placeholder(tf.float32,
                                                  [None, None, 768])
            self.ans_bert_matrix = tf.placeholder(tf.float32,
                                                  [None, None, 768])

        # init basic embedding matrix
        char_w2v_embedding_matrix, word_w2v_embedding_matrix = model_utils_reuse.load_variable_from_file(
            char_w2v_embedding_matrix_path, word_w2v_embedding_matrix_path)
        self.char_w2v_embedding_matrix = tf.convert_to_tensor(
            char_w2v_embedding_matrix)
        self.word_w2v_embedding_matrix = tf.convert_to_tensor(
            word_w2v_embedding_matrix)
        # if self.use_bert:
        #     self.bert_embedding_matrix = tf.convert_to_tensor(bert_embedding_matrix)

        # create model
        self.create_model_graph()

    def create_feed_dict(self, cur_batch):
        assert isinstance(cur_batch, InstanceBatch)
        feed_dict = {
            self.que_lens: cur_batch.que_lens,
            self.ans_lens: cur_batch.ans_lens,
            self.label_sequence_change:cur_batch.label_tag_change,
            #self.len_option_sequence:cur_batch.len_tag,#the length of question's options
            #self.word_option_length:cur_batch.word_emb_option_lens, #the legth of every option word embedding
            self.in_ans_append:cur_batch.que_with_ans,
            self.in_option_ww2v_index_matrix: cur_batch.option_ww2v_index_matrix,
            self.in_ans_cw2v_index_matrix: cur_batch.ans_cw2v_index_matrix,
            self.in_ans_ww2v_index_matrix: cur_batch.ans_ww2v_index_matrix,
            self.in_que_cw2v_index_matrix: cur_batch.que_cw2v_index_matrix,
            self.in_que_ww2v_index_matrix: cur_batch.que_ww2v_index_matrix,
            self.in_que_indicate_target_matrix:
                cur_batch.que_indicate_target_matrix,
            self.in_ans_indicate_target_matrix:
                cur_batch.ans_indicate_target_matrix,
            self.in_que_sentiment_polarity_matrix:
                cur_batch.que_sentiment_polarity_matrix,
            self.in_ans_sentiment_polarity_matrix:
                cur_batch.ans_sentiment_polarity_matrix,
            self.que_skeleton_label: cur_batch.que_skeleton_label_matrix
        }
        if self.use_bert:
            feed_dict.update({
                self.que_bert_matrix: cur_batch.que_bert_matrix,
                self.ans_bert_matrix: cur_batch.ans_bert_matrix
            })
        if self.use_extra_feature:
            feed_dict.update({
                self.in_que_pos_index_matrix:
                    cur_batch.que_pos_index_matrix,
                self.in_que_ner_index_matrix:
                    cur_batch.que_ner_index_matrix,
                self.in_ans_pos_index_matrix:
                    cur_batch.ans_pos_index_matrix,
                self.in_ans_ner_index_matrix:
                    cur_batch.ans_ner_index_matrix,
            })
        return feed_dict
    
    def create_model_graph(self):
        # truths = tf.get_variable(self.truths, name='truths')
        for i in range(self.max_option_length):
            locals()['labels'+str(i+1)] = model_utils_reuse.make_label(self.label_sequence_change[i], self.nb_classes)
        #que_in_features = []
        for i in range(self.max_option_length):
            locals()["que_in_features"+str(i+1)] = []
        ans_in_features = []
        # feature_dim = 0

        # w2v embedding
        if self.use_w2v:
            que_char_w2v_features = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix,
                ids=self.in_que_cw2v_index_matrix)
            #que_in_features.append(que_char_w2v_features)
            que_word_w2v_features = tf.nn.embedding_lookup(
                params=self.word_w2v_embedding_matrix,
                ids=self.in_que_ww2v_index_matrix)
            #que_in_features.append(que_word_w2v_features)
            for i in range(self.max_option_length):
                locals()["que_in_features" + str(i + 1)].append(que_char_w2v_features)
                locals()["que_in_features" + str(i + 1)].append(que_word_w2v_features)
            ans_char_w2v_features = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix,
                ids=self.in_ans_cw2v_index_matrix)
            ans_in_features.append(ans_char_w2v_features)
            ans_word_w2v_features = tf.nn.embedding_lookup(
                params=self.word_w2v_embedding_matrix,
                ids=self.in_ans_ww2v_index_matrix)
            ans_in_features.append(ans_word_w2v_features)
            
            que_and_ans = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix,
                ids=self.in_ans_append)
            ans_in_features.append(que_and_ans)

        # bert embedding
        if self.use_bert:
            #que_in_features = [self.que_bert_matrix]
            for i in range(self.max_option_length):
                locals()["que_in_features" + str(i + 1)].append([self.que_bert_matrix])
            ans_in_features = [self.ans_bert_matrix]
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
            que_indicate_ner_features = tf.nn.embedding_lookup(
                params=indicate_ner_matrix, ids=self.in_que_ner_index_matrix)
            ans_indicate_ner_features = tf.nn.embedding_lookup(
                params=indicate_ner_matrix, ids=self.in_ans_ner_index_matrix)
            que_indicate_pos_features = tf.nn.embedding_lookup(
                params=indicate_pos_matrix, ids=self.in_que_pos_index_matrix)
            ans_indicate_pos_features = tf.nn.embedding_lookup(
                params=indicate_pos_matrix, ids=self.in_ans_pos_index_matrix)
            #que_in_features.append(que_indicate_ner_features)
            #que_in_features.append(que_indicate_pos_features)
            for i in range(self.max_option_length):
                locals()["que_in_features"+str(i+1)].append(que_indicate_ner_features)
                locals()["que_in_features" + str(i + 1)].append(que_indicate_pos_features)
            ans_in_features.append(ans_indicate_ner_features)
            ans_in_features.append(ans_indicate_pos_features)

        # indicate-target vectors
        indicate_target_matrix = np.concatenate(
            [np.zeros([1, 30]), 0.3 * np.ones([1, 30])], axis=0)
        indicate_target_matrix = tf.Variable(indicate_target_matrix,
                                             trainable=True,
                                             name="indicate_target_embedding",
                                             dtype=tf.float32)
        que_indicate_target_features = tf.nn.embedding_lookup(
            params=indicate_target_matrix,
            ids=self.in_que_indicate_target_matrix)
        ans_indicate_target_features = tf.nn.embedding_lookup(
            params=indicate_target_matrix,
            ids=self.in_ans_indicate_target_matrix)
        #que_in_features.append(que_indicate_target_features)
        for i in range(self.max_option_length):
            locals()["que_in_features" + str(i + 1)].append(que_indicate_target_features)
        ans_in_features.append(ans_indicate_target_features)
        print('que_indicate_target_features shape:',
              que_indicate_target_features)

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
        ans_sentiment_polarity_features = tf.nn.embedding_lookup(
            params=sentiment_polarity_matrix,
            ids=self.in_ans_sentiment_polarity_matrix)
        que_sentiment_polarity_features = tf.nn.embedding_lookup(
            params=sentiment_polarity_matrix,
            ids=self.in_que_sentiment_polarity_matrix)
        #que_in_features.append(que_sentiment_polarity_features)
        for i in range(self.max_option_length):
            locals()["que_in_features" + str(i + 1)].append(que_sentiment_polarity_features)
        ans_in_features.append(ans_sentiment_polarity_features)
        # [batch_size, question_len, dim]
        #in_question_repr = tf.concat(axis=2, values=que_in_features)
        
        for i in range(self.max_option_length):
            in_option = self.in_option_ww2v_index_matrix[i]
            name_word = "option"+str(i+1)+"_word_w2v_features"
            #name_word_id = "in_option_ww2v_index_matrix"+str(i+1)
            locals()["option"+str(i+1)+"_word_w2v_features"] = tf.nn.embedding_lookup(
            params=self.word_w2v_embedding_matrix,
            ids=in_option)
        for i in range(self.max_option_length):
            locals()["que_in_features"+str(i+1)].append(locals()["option"+str(i+1)+"_word_w2v_features"])
            locals()["in_question_repr"+str(i+1)] = tf.concat(axis=2, values=locals()["que_in_features"+str(i+1)])
            locals()["in_question_repr" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["in_question_repr" + str(i + 1)],
                                                         self.dropout_rate,
                                                         self.is_training)
        # [batch_size, question_len, dim]
        in_answer_repr = tf.concat(axis=2, values=ans_in_features)
        #print("in_question_repr shape:", in_question_repr.shape)
        #print("in_question_repr1 shape:", locals()["in_question_repr" + str(1)].shape)
        #print("in_answer_repr shape:", in_answer_repr.shape)
        in_answer_repr = model_utils_reuse.dropout_layer(in_answer_repr,
                                                   self.dropout_rate,
                                                   self.is_training)
        # TODO: complete skeleton information indicator
        #indicate_skeleton_matrix = self.que_skeleton_label

        # basic encode using bi-lstm
        assert self.rnn_unit == 'lstm' or self.rnn_unit == 'gru'
        
        
        answer_bi = model_utils_reuse.my_rnn_layer(input_reps=in_answer_repr,
                                             rnn_dim=self.rnn_dim,
                                             rnn_unit=self.rnn_unit,
                                             input_lengths=self.ans_lens,
                                             scope_name='basic_encode',
                                             is_training=self.is_training,
                                             reuse=False)

        answer_bi_last = model_utils_reuse.collect_final_step_of_lstm(
            answer_bi, self.ans_lens-1)
        self.answer_repr = answer_bi_last
##################################################################################################
        for i in range(self.max_option_length):
            if(i==0):
                locals()["question_bi"+str(i+1)] = model_utils_reuse.my_rnn_layer(input_reps=locals()["in_question_repr"+str(i+1)],
                                                                                       rnn_dim=self.rnn_dim,
                                                                                       rnn_unit=self.rnn_unit,
                                                                                       input_lengths=self.que_lens,
                                                                                       scope_name='basic_encode',
                                                                                       is_training=self.is_training,
                                                                                       reuse=True)
                locals()["question_bi" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["question_bi"+str(i+1)], self.dropout_rate,
                                                                                            self.is_training)
                locals()["question_target_repr"+str(i+1)] = model_utils_reuse.get_target_representation(
                                                                                                            locals()["question_bi" + str(i + 1)], 
                                                                                                            self.in_que_indicate_target_matrix,
                                                                                                            "target_repr",reuse=False)
                locals()["question_bi_last"+str(i+1)] = model_utils_reuse.collect_final_step_of_lstm(locals()["question_bi" + str(i + 1)], self.que_lens)
                locals()["answer_bi"+str(i+1)] = model_utils_reuse.sentiment_polarity_flip(
                                                                                              answer_bi, ans_sentiment_polarity_features, 
                                                                                              locals()["question_target_repr"+str(i+1)],
                                                                                              self.sentiment_polarity_multiple, self.attention_dim,
                                                                                              "sentiment_polarity_flip",reuse=False)
                locals()["answer_bi" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["answer_bi" + str(i + 1)], self.dropout_rate,
                                                                                        self.is_training)
                locals()["question_skeleton_repr"+str(i+1)] = locals()["question_target_repr"+str(i+1)]
                locals()["question_semantic_repr"+str(i+1)] = model_utils_reuse.generate_semantic_representation(
                                                                                                                    locals()["question_skeleton_repr"+str(i+1)], 
                                                                                                                    locals()["question_bi" + str(i + 1)], 
                                                                                                                    self.que_lens,
                                                                                                                    self.attention_dim,
                                                                                                                    'get_semantic_representation'+str(i+1))
                locals()["question_aware_repr"+str(i+1)] = model_utils_reuse.get_aware_repr(
                locals()["answer_bi" + str(i + 1)], locals()["question_skeleton_repr"+str(i+1)], locals()["question_semantic_repr"+str(i+1)],
                self.nb_hops, self.attention_dim, self.rnn_dim, self.ans_max_len,
                self.ans_lens, self.lambda_l2,"get_aware",reuse=False)
                locals()["question_aware_repr"+str(i+1)] = model_utils_reuse.dropout_layer(
                locals()["question_aware_repr"+str(i+1)], self.dropout_rate, self.is_training)
                #locals()["question_aware_repr"+str(i+1)] = locals()["question_semantic_repr"+str(i+1)]
                locals()["answer_aware_repr"+str(i+1)] = model_utils_reuse.multi_hop_match(
                                                          answer_bi_last, locals()["question_bi" + str(i + 1)], self.nb_hops, self.rnn_dim,
                                                          self.attention_dim, 'answer_aware', self.que_max_len, self.que_lens,
                                                          self.lambda_l2,reuse=False)
                #locals()["answer_aware_repr"+str(i+1)] = answer_bi_last
            else:
                locals()["question_bi"+str(i+1)] = model_utils_reuse.my_rnn_layer(input_reps=locals()["in_question_repr"+str(i+1)],
                                                                                       rnn_dim=self.rnn_dim,
                                                                                       rnn_unit=self.rnn_unit,
                                                                                       input_lengths=self.que_lens,
                                                                                       scope_name='basic_encode',
                                                                                       is_training=self.is_training,
                                                                                       reuse=True)
                locals()["question_bi" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["question_bi"+str(i+1)], self.dropout_rate,
                                                                                            self.is_training)
                locals()["question_target_repr"+str(i+1)] = model_utils_reuse.get_target_representation(
                                                                                                            locals()["question_bi" + str(i + 1)], 
                                                                                                            self.in_que_indicate_target_matrix,
                                                                                                            "target_repr",reuse=True)
                locals()["question_bi_last"+str(i+1)] = model_utils_reuse.collect_final_step_of_lstm(locals()["question_bi" + str(i + 1)], self.que_lens)
                locals()["answer_bi"+str(i+1)] = model_utils_reuse.sentiment_polarity_flip(
                                                                                              answer_bi, ans_sentiment_polarity_features, 
                                                                                              locals()["question_target_repr"+str(i+1)],
                                                                                              self.sentiment_polarity_multiple, self.attention_dim,
                                                                                              "sentiment_polarity_flip",reuse=True)
                locals()["answer_bi" + str(i + 1)] = model_utils_reuse.dropout_layer(locals()["answer_bi" + str(i + 1)], self.dropout_rate,
                                                                                        self.is_training)
                locals()["question_skeleton_repr"+str(i+1)] = locals()["question_target_repr"+str(i+1)]
                locals()["question_semantic_repr"+str(i+1)] = model_utils_reuse.generate_semantic_representation(
                                                                                                                    locals()["question_skeleton_repr"+str(i+1)], 
                                                                                                                    locals()["question_bi" + str(i + 1)], 
                                                                                                                    self.que_lens,
                                                                                                                    self.attention_dim,
                                                                                                                    'get_semantic_representation'+str(i+1))
                locals()["question_aware_repr"+str(i+1)] = model_utils_reuse.get_aware_repr(
                locals()["answer_bi" + str(i + 1)], locals()["question_skeleton_repr"+str(i+1)], locals()["question_semantic_repr"+str(i+1)],
                self.nb_hops, self.attention_dim, self.rnn_dim, self.ans_max_len,
                self.ans_lens, self.lambda_l2,"get_aware",reuse=True)
                locals()["question_aware_repr"+str(i+1)] = model_utils_reuse.dropout_layer(
                locals()["question_aware_repr"+str(i+1)], self.dropout_rate, self.is_training)
                #locals()["question_aware_repr"+str(i+1)] = locals()["question_semantic_repr"+str(i+1)]
                locals()["answer_aware_repr"+str(i+1)] = model_utils_reuse.multi_hop_match(
                                                          answer_bi_last, locals()["question_bi" + str(i + 1)], self.nb_hops, self.rnn_dim,
                                                          self.attention_dim, 'answer_aware', self.que_max_len, self.que_lens,
                                                          self.lambda_l2,reuse=True)
                #locals()["answer_aware_repr"+str(i+1)] = answer_bi_last
        for i in range(self.max_option_length):
            if(i==0):
                locals()["end_rper"+str(i+1)] = tf.concat([locals()["question_aware_repr"+str(i+1)], locals()["answer_aware_repr"+str(i+1)]],
                                     axis=-1)
                locals()["logits"+str(i+1)] = model_utils_reuse.full_connect_layer(locals()["end_rper"+str(i+1)],
                                                        self.nb_classes,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        scope="full_connect",
                                                        reuse=False)
                locals()["loss"+str(i+1)] = model_utils_reuse.compute_cross_entropy(locals()["logits"+str(i+1)], locals()['labels'+str(i+1)])
                locals()["prob"+str(i+1)] = tf.nn.softmax(locals()["logits"+str(i+1)])
                locals()["correct"+str(i+1)] = tf.nn.in_top_k(tf.to_float(locals()["logits"+str(i+1)]), self.label_sequence_change[i], 1)
                locals()["cor"+str(i+1)] = tf.reduce_sum(tf.cast(locals()["correct"+str(i+1)], tf.int32))
                locals()["pre"+str(i+1)] = tf.argmax(locals()["prob"+str(i+1)], 1)
            else:
                locals()["end_rper"+str(i+1)] = tf.concat([locals()["question_aware_repr"+str(i+1)], locals()["answer_aware_repr"+str(i+1)]],
                                     axis=-1)
                locals()["logits"+str(i+1)] = model_utils_reuse.full_connect_layer(locals()["end_rper"+str(i+1)],
                                                        self.nb_classes,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        scope="full_connect",
                                                        reuse=True)
                locals()["loss"+str(i+1)] = model_utils_reuse.compute_cross_entropy(locals()["logits"+str(i+1)], locals()['labels'+str(i+1)])
                locals()["prob"+str(i+1)] = tf.nn.softmax(locals()["logits"+str(i+1)])
                locals()["correct"+str(i+1)] = tf.nn.in_top_k(tf.to_float(locals()["logits"+str(i+1)]), self.label_sequence_change[i], 1)
                locals()["cor"+str(i+1)] = tf.reduce_sum(tf.cast(locals()["correct"+str(i+1)], tf.int32))
                locals()["pre"+str(i+1)] = tf.argmax(locals()["prob"+str(i+1)], 1)
        
        #sum_count = 2
        #while(sum_count<=self.max_option_length):
            
        #    locals()["sum_acc"+str(sum_count)] = locals()["cor"+str(1)]
        #    locals()["sum_loss" + str(sum_count)] = locals()["loss" + str(1)]
        #    for i in range(1,sum_count):
        #        locals()["sum_acc"+str(sum_count)] += locals()["cor"+str(i+1)]
        #        locals()["sum_loss" + str(sum_count)] += locals()["loss" + str(i+1)]
        #    sum_count=sum_count+1
        names=locals()
        #for i in range(2,self.max_option_length+1):
        #    names["len"+str(i)] = tf.convert_to_tensor(i, dtype=tf.int32)
        #    names["accuracy"+str(i)] = names["sum_acc"+str(i)]/names["len"+str(i)] 
        #    names["model_loss"+str(i)] = names["sum_loss" + str(i)]
        #self.acc = names["accuracy"+str(5)]
        #self.loss = names["model_loss"+str(5)]
        self.acc = (locals()["cor"+str(1)]+locals()["cor"+str(2)]+locals()["cor"+str(3)]+locals()["cor"+str(4)]+locals()["cor"+str(5)])/tf.convert_to_tensor(5, dtype=tf.int32)
        self.loss = locals()["loss" + str(1)]+locals()["loss" + str(2)]+locals()["loss" + str(3)]+locals()["loss" + str(4)]+locals()["loss" + str(5)]
        prediction_list = []
        for i in range(self.max_option_length):
            prediction_list.append(names["pre"+str(i+1)])
        self.prediction_ls = prediction_list
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
