import numpy as np
import tensorflow as tf
import multi_class_model_utils_
import multi_class_utils_
import math

import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

from tensorflow.python.ops import array_ops
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from multi_class_data_stream_ import InstanceBatch
from tensorflow.python.util import nest
tf.reset_default_graph()

class AnswerGenerator(object):
    def __init__(self, use_w2v, rnn_unit, dropout_rate, use_beam, use_residual,
                 char_w2v_embedding_matrix_path, rnn_dim, nb_classes, dtype,
                 num_encoder_symbols, beam_size, num_decoder_symbols, depth,
                 optimizer, learning_rate, grad_clipper, global_step, nb_hops,
                 attention_dim, is_training, use_extra_feature, batch_size,
                 extra_feature_dim, ner_dict_size, pos_dict_size, lambda_l2,
                 sentiment_polarity_multiple, word_w2v_embedding_matrix_path,
                 use_dropout, attention_type, attn_input_feeding):
        self.use_w2v = use_w2v
        self.cell_type = rnn_unit
        self.dropout_rate = dropout_rate
        self.hidden_units = rnn_dim
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
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.num_encoder_symbols = num_encoder_symbols
        self.num_decoder_symbols = num_decoder_symbols
        self.dtype = dtype
        self.depth = depth
        self.attention_type = attention_type
        self.attn_input_feeding = attn_input_feeding
        self.use_residual = use_residual
        self.keep_prob_placeholder = 1 - self.dropout_rate
        self.use_dropout = use_dropout
        self.char_w2v_embedding_matrix_path = char_w2v_embedding_matrix_path
        self.word_w2v_embedding_matrix_path = word_w2v_embedding_matrix_path

        self.use_beamsearch_decode = False
        if not self.is_training:
            self.beam_width = self.beam_size
            self.use_beamsearch_decode = True if self.beam_width > 1 else False
            self.max_decode_step = 32
        # create model
        self.build_model()

    def build_model(self):
        print("building model..")

        # Building encoder and decoder networks
        self.init_placeholders()
        self.build_encoder()
        self.build_decoder()

        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()
        if self.is_training:
            trainable_variables = tf.trainable_variables()
            if self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate=self.learning_rate)
            else:
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate)
            grads = multi_class_model_utils_.compute_gradients(self.loss,
                                                  trainable_variables)
            # TODO: compute grad_clipper
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clipper)
            self.train_op = optimizer.apply_gradients(
                zip(grads, trainable_variables), global_step=self.global_step)

    def init_placeholders(self):

        # create placeholders
        self.que_lens = tf.placeholder(tf.int32, [None])  # [batch_size]
        #self.truths = tf.placeholder(tf.int32, [None])  # [batch_size]

        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=(None, None), name='encoder_inputs')

        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='encoder_inputs_length')

        # get dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        # self.in_que_cw2v_index_matrix = tf.placeholder(
        #     tf.int32, [None, None])  # [batch_size, max_sequence_len]
        # self.in_que_ww2v_index_matrix = tf.placeholder(
        #     tf.int32, [None, None])  # [batch_size, max_sequence_len]

        self.in_que_sentiment_polarity_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_class_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_indicate_target_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]

        if self.is_training:
            self.decoder_inputs = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.decoder_inputs_length = tf.placeholder(
                tf.int32, [None])  # [batch_size]
            decoder_start_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * multi_class_utils_.start_token
            decoder_end_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * multi_class_utils_.end_token

            self.decoder_inputs_train = tf.concat(
                [decoder_start_token, self.decoder_inputs], axis=1)

            # decoder_inputs_length_train: [batch_size]
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1

            # decoder_targets_train: [batch_size, max_time_steps + 1]
            # insert EOS symbol at the end of each decoder input
            self.decoder_targets_train = tf.concat([self.decoder_inputs,
                                                    decoder_end_token], axis=1)

        if self.use_extra_feature:
            self.in_que_pos_index_matrix = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_que_ner_index_matrix = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]

        # init basic embedding matrix
        char_w2v_embedding_matrix, word_w2v_embedding_matrix = multi_class_model_utils_. \
            load_variable_from_file(
                self.char_w2v_embedding_matrix_path,
                self.word_w2v_embedding_matrix_path
            )
        self.char_w2v_embedding_matrix = tf.convert_to_tensor(
            char_w2v_embedding_matrix)
        self.word_w2v_embedding_matrix = tf.convert_to_tensor(
            word_w2v_embedding_matrix)
        self.embedding_size = self.char_w2v_embedding_matrix. \
            get_shape().as_list()[-1]

        self.sentiment_representation = tf.get_variable(
            name='sentiment_representation',
            shape=[4, self.sentiment_polarity_multiple])
        #indicate_senti_matrix = np.concatenate(
        #    [np.zeros([1, 30]), 0.3 * np.ones([1, 30])], axis=0)
        #self.indicate_senti_matrix = tf.Variable(indicate_senti_matrix,
        #                                          trainable=True,
        #                                          name="indicate_senti_matrix",
        #                                          dtype=tf.float32)
        self.class_representation = tf.get_variable(
            name='class_representation',
            shape=[43, self.sentiment_polarity_multiple],
            trainable =True)
        # indicate-target vectors
        indicate_target_matrix = np.concatenate(
            [np.zeros([1, 30]), 0.3 * np.ones([1, 30])], axis=0)
        self.indicate_target_matrix = tf.Variable(indicate_target_matrix,
                                                  trainable=True,
                                                  name="indicate_target_matrix",
                                                  dtype=tf.float32)
        #class_matrix = np.concatenate(
        #    [np.zeros([1, 30]), 0.3 * np.ones([1, 30])], axis=0)
        #self.class_matrix = tf.Variable(class_matrix,
        #                                          trainable=True,
        #                                          name="class_matrix",
        #                                          dtype=tf.float32)
        if self.use_extra_feature:
            self.indicate_ner_matrix = tf.get_variable(
                name='indicate_ner_embedding',
                shape=[self.ner_dict_size, self.extra_feature_dim],
                trainable=True,
                dtype=tf.float32)
            self.indicate_pos_matrix = tf.get_variable(
                name='indicate_pos_embedding',
                shape=[self.pos_dict_size, self.extra_feature_dim],
                trainable=True,
                dtype=tf.float32)

    def create_feed_dict(self, cur_batch):
        assert isinstance(cur_batch, InstanceBatch)
        feed_dict = {
            self.encoder_inputs_length: cur_batch.que_lens,
            #self.truths: cur_batch.truths,
            self.encoder_inputs: cur_batch.que_cw2v_index_matrix,
            # self.in_que_ww2v_index_matrix: cur_batch.que_ww2v_index_matrix,
            self.in_que_indicate_target_matrix:
                cur_batch.que_indicate_target_matrix,
            self.in_que_class_matrix:
                cur_batch.que_class_matrix,
            self.in_que_sentiment_polarity_matrix:
                cur_batch.que_sentiment_polarity_matrix
            # self.que_skeleton_label: cur_batch.que_skeleton_label_matrix
        }

        if self.is_training:
            feed_dict.update({
                self.decoder_inputs_length: cur_batch.ans_lens,
                # self.in_ans_sentiment_polarity_matrix:
                # cur_batch.ans_sentiment_polarity_matrix,
                # self.in_ans_indicate_target_matrix:
                # cur_batch.ans_indicate_target_matrix,
                self.decoder_inputs: cur_batch.ans_cw2v_index_matrix
                # self.in_ans_ww2v_index_matrix: cur_batch.ans_ww2v_index_matrix
            })

        if self.use_extra_feature:
            feed_dict.update({
                self.in_que_pos_index_matrix:
                    cur_batch.que_pos_index_matrix,
                self.in_que_ner_index_matrix:
                    cur_batch.que_ner_index_matrix
            })
        return feed_dict

    def build_encoder(self):
        print("building encoder..")
        with tf.variable_scope('encoder'):
            # Building encoder_cell
            self.encoder_cell = self.build_encoder_cell()

            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(
                -sqrt3, sqrt3, dtype=self.dtype)

            # Embedded_inputs: [batch_size, time_step, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix, ids=self.encoder_inputs)
            # concat the sentiment_polarity
            self.encoder_sentiment = tf.nn.embedding_lookup(
                params=self.sentiment_representation,
                ids=self.in_que_sentiment_polarity_matrix)

            if self.use_extra_feature:
                self.ner_features = tf.nn.embedding_lookup(
                    params=self.indicate_ner_matrix, ids=self.in_que_ner_index_matrix)
                self.pos_features = tf.nn.embedding_lookup(
                    params=self.indicate_pos_matrix, ids=self.in_que_pos_index_matrix)

            self.encoder_indicator = tf.nn.embedding_lookup(
                params=self.indicate_target_matrix,
                ids=self.in_que_indicate_target_matrix)
            self.encoder_class = tf.nn.embedding_lookup(
                params=self.class_representation,
                ids=self.in_que_class_matrix)
            if not self.use_extra_feature:
                self.encoder_input_end = tf.concat(
                    [self.encoder_inputs_embedded,
                     self.encoder_sentiment,
                     self.encoder_indicator
                     #self.encoder_class
                     ], axis=-1)
            else:
                self.encoder_input_end = tf.concat(
                    [self.encoder_inputs_embedded,
                        self.encoder_sentiment,
                        #self.encoder_class,
                        self.ner_features,
                        self.pos_features,
                        self.encoder_indicator], axis=-1)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units,
                                dtype=self.dtype, name='input_projection')
            # Embedded inputs having gone through input projection layer
            self.encoder_inputs_embedded = input_layer(
                self.encoder_input_end)
           # self.class_embed = input_layer_class(self.encoder_class)
            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            self.encoder_outputs1, self.encoder_last_state1 = tf.nn.dynamic_rnn(
                cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, dtype=self.dtype,
                time_major=False)
            output_layer = Dense(self.hidden_units,
                                dtype=self.dtype, name='output_projection',trainable=True)
            #self.en_out1 = output_layer(self.encoder_outputs1)
            self.en_out2 = output_layer(self.encoder_class)
            def a(x):
                return x+tf.reduce_mean(self.en_out2,axis=1,keep_dims=False)
            self.encoder_last_state = nest.map_structure(a,self.encoder_last_state1)
           
            #style_vec = [self.en_out2,(self.en_out2)
            #self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
            #    cell=self.encoder_cell, inputs=self.encoder_inputs_embedded+self.en_out2,
            #    sequence_length=self.encoder_inputs_length, dtype=self.dtype,
            #    time_major=False)
            #self.encoder_last_state = tf.map_fn(lambda x:tf.add(x,style_vec),self.encoder_last_state1)
            #self.encoder_cell = CopyNetWrapper(self.encoder_cell, self.encoder_last_state1,self.en_out2)
            #self.encoder_last_state = self.encoder_cell.zero_state(batch_size=self.batch_size, dtype=self.dtype)
            
            #Copy = CopyNetWrapper(self.encoder_cell, self.encoder_last_state1,self.en_out2)
            #self.encoder_outputs1, self.encoder_last_state1 = Copy(self.encoder_outputs1,self.encoder_last_state1)
            #self.encoder_last_state = Copy.zero_state(batch_size=self.batch_size, dtype=self.dtype)
            #self.encoder_outputs = tf.concat(
            #        [self.encoder_outputs1,
            #            self.encoder_class
            #            ], axis=-1)
            
            
            self.encoder_outputs = self.encoder_outputs1+self.encoder_class
            #self.encoder_outputs = self.en_out1+self.en_out2
    def build_decoder(self):
        print("building decoder and attention..")
        with tf.variable_scope('decoder'):
            # Building decoder_cell and decoder_initial_state
            self.decoder_cell, self.decoder_initial_state = \
                self.build_decoder_cell()

            # Initialize decoder embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(
                -sqrt3, sqrt3, dtype=self.dtype)

            self.decoder_embeddings = tf.get_variable(
                name='embedding',
                shape=[self.num_decoder_symbols, self.embedding_size],
                initializer=initializer, dtype=self.dtype)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units,
                                dtype=self.dtype, name='input_projection')

            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(self.num_decoder_symbols,
                                 name='output_projection')

            if self.is_training:
                # decoder_inputs_embedded:
                #  [batch_size, max_time_step + 1, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings,
                    ids=self.decoder_inputs_train)

                # Embedded inputs having gone through input projection layer
                self.decoder_inputs_embedded = input_layer(
                    self.decoder_inputs_embedded)

                # Helper to feed inputs for training: read inputs from
                # dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(
                    inputs=self.decoder_inputs_embedded,
                    sequence_length=self.decoder_inputs_length_train,
                    time_major=False,
                    name='training_helper')

                training_decoder = seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=training_helper,
                    initial_state=self.decoder_initial_state,
                    output_layer=output_layer)
                # output_layer=None)

                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(
                    self.decoder_inputs_length_train)

                # decoder_outputs_train: BasicDecoderOutput
                #                        namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output:
                # [batch_size, max_time_step + 1, num_decoder_symbols]
                # if output_time_major=False
                # [max_time_step + 1, batch_size, num_decoder_symbols]
                # if output_time_major=True
                # decoder_outputs_train.sample_id: [batch_size], tf.int32
                (self.decoder_outputs_train, self.decoder_last_state_train,
                 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                     decoder=training_decoder,
                     output_time_major=False,
                     impute_finished=True,
                     maximum_iterations=max_decoder_length))

                # More efficient to do the projection on the
                # batch-time-concatenated tensor
                # logits_train:
                # [batch_size, max_time_step + 1, num_decoder_symbols]
                # self.decoder_logits_train =
                # output_layer(self.decoder_outputs_train.rnn_output)
                self.decoder_logits_train = tf.identity(
                    self.decoder_outputs_train.rnn_output)
                # Use argmax to extract decoder symbols to emit
                self.decoder_pred_train = tf.argmax(
                    self.decoder_logits_train, axis=-1,
                    name='decoder_pred_train')

                # masks: masking for valid and padded time steps,
                # [batch_size, max_time_step + 1]
                masks = tf.sequence_mask(
                    lengths=self.decoder_inputs_length_train,
                    maxlen=max_decoder_length, dtype=self.dtype, name='masks')

                current_ts = tf.to_int32(
                    tf.minimum(tf.shape(self.decoder_targets_train)[1], tf.shape(self.decoder_logits_train)[1]))
                self.decoder_targets_train = tf.slice(self.decoder_targets_train, begin=[
                                                      0, 0], size=[-1, current_ts])

                # Computes per word average cross-entropy over a batch
                # Internally calls
                # 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default

                self.loss = seq2seq.sequence_loss(
                    logits=self.decoder_logits_train,
                    targets=self.decoder_targets_train,
                    weights=masks,
                    average_across_timesteps=True,
                    average_across_batch=True, )
                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)

            elif not self.is_training:

                # Start_tokens: [batch_size,] `int32` vector
                start_tokens = tf.ones(
                    [self.batch_size, ], tf.int32) * multi_class_utils_.start_token
                end_token = multi_class_utils_.end_token

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(
                        self.decoder_embeddings, inputs))

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding:
                    #  uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(
                        start_tokens=start_tokens,
                        end_token=end_token,
                        embedding=embed_and_input_proj)
                    # Basic decoder performs greedy decoding at each time step
                    print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(
                        cell=self.decoder_cell,
                        helper=decoding_helper,
                        initial_state=self.decoder_initial_state,
                        output_layer=output_layer)
                else:
                    # Beamsearch is used to approximately
                    # find the most likely translation
                    print("building beamsearch decoder..")
                    inference_decoder = beam_search_decoder.BeamSearchDecoder(
                        cell=self.decoder_cell,
                        embedding=embed_and_input_proj,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=self.decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=output_layer, )
                # For GreedyDecoder, return
                # decoder_outputs_decode: BasicDecoderOutput instance
                #                         namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_decode.rnn_output:
                # [batch_size, max_time_step, num_decoder_symbols]
                # 	if output_time_major=False
                # [max_time_step, batch_size, num_decoder_symbols]
                # 	if output_time_major=True
                # decoder_outputs_decode.sample_id:
                # [batch_size, max_time_step], tf.int32
                # 	if output_time_major=False
                # [max_time_step, batch_size], tf.int32
                #     if output_time_major=True

                # For BeamSearchDecoder, return
                # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
                #
                # namedtuple(predicted_ids, beam_search_decoder_output)
                # decoder_outputs_decode.predicted_ids:
                # [batch_size, max_time_step, beam_width]
                #  if output_time_major=False
                # [max_time_step, batch_size, beam_width]
                #  if output_time_major=True
                # decoder_outputs_decode.beam_search_decoder_output:
                #  BeamSearchDecoderOutput instance
                #  namedtuple(scores, predicted_ids, parent_ids)

                (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                     decoder=inference_decoder,
                     output_time_major=False,
                     # impute_finished=True,	# error occurs
                     maximum_iterations=self.max_decode_step))

                if not self.use_beamsearch_decode:
                    # decoder_outputs_decode.sample_id:
                    # [batch_size, max_time_step]
                    # Or use argmax to find decoder symbols to emit:
                    # self.decoder_pred_decode = tf.argmax(
                    # self.decoder_outputs_decode.rnn_output,
                    # axis=-1, name='decoder_pred_decode')

                    # Here, we use expand_dims to be compatible with the
                    # result of the beamsearch decoder
                    # decoder_pred_decode: [batch_size, max_time_step, 1]
                    # (output_major=False)
                    self.decoder_pred_decode = tf.expand_dims(
                        self.decoder_outputs_decode.sample_id, -1)

                else:
                    # Use beam search to approximately find the
                    # most likely translation
                    # decoder_pred_decode:
                    #  [batch_size, max_time_step, beam_width]
                    #   (output_major=False)
                    self.decoder_pred_decode = \
                        self.decoder_outputs_decode.predicted_ids

    def build_single_cell(self):
        cell_type = LSTMCell
        if (self.cell_type.lower() == 'gru'):
            cell_type = GRUCell
        cell = cell_type(self.hidden_units)

        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=self.keep_prob_placeholder, )
        if self.use_residual:
            cell = ResidualWrapper(cell)

        return cell

    # Building encoder cell
    def build_encoder_cell(self):

        return MultiRNNCell([self.build_single_cell()
                             for i in range(self.depth)])

    # Building decoder cell and attention. Also returns decoder_initial_state
    def build_decoder_cell(self):

        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length

        # To use BeamSearchDecoder, encoder_outputs,
        # encoder_last_state, encoder_inputs_length
        # needs to be tiled so that: [batch_size, .., ..]
        # -> [batch_size x beam_width, .., ..]
        if self.use_beamsearch_decode:
            print("use beamsearch decoding..")
            encoder_outputs = seq2seq.tile_batch(
                self.encoder_outputs, multiplier=self.beam_width)
            encoder_last_state = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, self.beam_width),
                self.encoder_last_state)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)

        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        self.attention_mechanism = attention_wrapper.BahdanauAttention(
            num_units=self.hidden_units, memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length, )
        # 'Luong' style attention: https://arxiv.org/abs/1508.04025
        if self.attention_type.lower() == 'luong':
            self.attention_mechanism = attention_wrapper.LuongAttention(
                num_units=self.hidden_units, memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length, )

        # Building decoder_cell
        self.decoder_cell_list = [
            self.build_single_cell() for i in range(self.depth)]
        decoder_initial_state = encoder_last_state

        def attn_decoder_input_fn(inputs, attention):
            if not self.attn_input_feeding:
                return inputs

            # Essential when use_residual=True
            _input_layer = Dense(self.hidden_units, dtype=self.dtype,
                                 name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
        self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_units,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=False,
            name='Attention_Wrapper')

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into
        # the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # Also if beamsearch decoding is used,
        # the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        batch_size = self.batch_size if not self.use_beamsearch_decode \
            else self.batch_size * self.beam_width
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
            batch_size=batch_size, dtype=self.dtype)
        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state
