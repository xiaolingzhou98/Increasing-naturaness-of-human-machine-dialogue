import json
from sag_choose_model import AnswerGenerator
from multi_class_model_utils_ import decode
import tensorflow as tf
from multi_class_data_stream_ import DataStream
from multi_class_utils_ import read_file, make_instances
import pickle


def test():
    with open('sag_choose_config.json', encoding='utf-8') as infile:
        config = json.load(infile)
    int2bool = {1: True, 0: False}

    data_path = config["train_config"]["DATA_PATH"]
    sentiment_words_path = config["train_config"]["SENTIMENT_WORDS_PATH"]
    batch_size = config["train_config"]["BATCH_SIZE"]
    is_shuffle = int2bool[config["train_config"]["IS_SHUFFLE"]]
    is_loop = int2bool[config["train_config"]["Is_LOOP"]]
    is_sort = int2bool[config["train_config"]["IS_SORT"]]
    nb_epoch = config["train_config"]["NB_EPOCH"]
    dropout_rate = config["train_config"]["DROPOUT_RATE"]
    nb_classes = config["train_config"]["NB_CLASSES"]
    attention_dim = config["train_config"]["ATTENTION_DIM"]
    nb_hops = config["train_config"]["NB_HOPS"]
    use_dropout = int2bool[config["train_config"]["USE_DROPOUT"]]
    optimizer = config["train_config"]["OPTIMIZER"]
    learning_rate = config["train_config"]["LEARNING_RATE"]
    grad_clipper = config["train_config"]["GRAD_CLIPPER"]
    drop_train_path = config["train_config"]["DROP_TRAIN_PATH"]
    drop_path = config["train_config"]["DROP_CHOOSE_DEV_PATH"]
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
    beam_size = config["train_config"]["BEAM_SIZE"]
    depth = config["train_config"]["DEPTH"]
    num_encoder_symbols = config["train_config"]["NUM_ENCODER_SYMBOLS"]
    num_decoder_symbols = config["train_config"]["NUM_DECODER_SYMBOLS"]
    use_beam = int2bool[config["train_config"]["USE_BEAM"]]
    use_residual = int2bool[config["train_config"]["USE_RESIDUAL"]]
    attention_type = config["train_config"]["ATTENTION_TYPE"]
    add_data_root = config["train_config"]["ADD_DATA_ROOT"]
    attn_input_feeding = int2bool[config["train_config"]["ATTN_INPUT_FEEDING"]]
    use_w2v = True

    char_w2v_path = config["w2v_config"]["CHAR_W2V_PATH"]
    char_voc_path = config["w2v_config"]["CHAR_VOC_PATH"]
    char_embedding_matrix_path = config["w2v_config"][
        "CHAR_EMBEDDING_MATRIX_PATH"]
    word_w2v_path = config["w2v_config"]["WORD_W2V_PATH"]
    word_voc_path = config["w2v_config"]["WORD_VOC_PATH"]
    word_embedding_matrix_path = config["w2v_config"][
        "WORD_EMBEDDING_MATRIX_PATH"]
    ans_max_len = config["train_config"]["ANS_MAX_LEN"]
    que_max_len = config["train_config"]["QUE_MAX_LEN"]

    samples = read_file([data_path, add_data_root])
    test_samples = read_file('generate_test.xlsx')
    with open(char_voc_path, 'rb') as infile:
        char_voc = pickle.load(infile)
    with open(word_voc_path, 'rb') as infile:
        word_voc = pickle.load(infile)

    instances_choose_dev = make_instances(
        test_samples,
        char_voc,
        word_voc,
        sentiment_words_path,
        ner_dict_path=ner_dict_path,
        pos_dict_path=pos_dict_path,
        use_extra_feature=use_extra_feature,
        question2targets_path=question2targets_path,
        is_training=False,
        need_augment=False)

    data_stream_choose_dev = DataStream(instances=instances_choose_dev,
                                        is_shuffle=is_shuffle,
                                        is_loop=is_loop,
                                        is_training=False,
                                        batch_size=batch_size,
                                        ans_max_len=ans_max_len,
                                        que_max_len=que_max_len,
                                        is_sort=is_sort)
    with tf.Graph().as_default():
        initializer = tf.glorot_uniform_initializer()
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            answer_generator_dev = AnswerGenerator(
                use_w2v=use_w2v,
                rnn_unit='gru',
                dropout_rate=dropout_rate,
                beam_size=beam_size,
                use_beam=use_beam,
                num_decoder_symbols=num_decoder_symbols,
                num_encoder_symbols=num_encoder_symbols,
                optimizer=optimizer,
                learning_rate=learning_rate,
                grad_clipper=grad_clipper,
                global_step=global_step,
                attention_dim=attention_dim,
                nb_hops=nb_hops,
                rnn_dim=rnn_dim,
                lambda_l2=lambda_l2,
                is_training=False,
                sentiment_polarity_multiple=sentiment_polarity_multiple,
                nb_classes=nb_classes,
                batch_size=batch_size,
                dtype=tf.float32,
                depth=depth,
                attn_input_feeding=attn_input_feeding,
                use_dropout=use_dropout,
                attention_type=attention_type,
                use_residual=use_residual,
                use_extra_feature=use_extra_feature,
                ner_dict_size=ner_dict_size,
                pos_dict_size=pos_dict_size,
                extra_feature_dim=extra_feature_dim,
                char_w2v_embedding_matrix_path=char_embedding_matrix_path,
                word_w2v_embedding_matrix_path=word_embedding_matrix_path)

        saver = tf.train.Saver()
        sess = tf.Session()
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        saver.restore(sess, best_path)
        decode(sess, answer_generator_dev,
               data_stream_choose_dev, 'result.txt')


if __name__ == "__main__":
    with open('multi_class_config_sum.json', encoding='utf-8') as infile:
        config = json.load(infile)
    int2bool = {1: True, 0: False}

    data_path = config["train_config"]["DATA_PATH"]
    sentiment_words_path = config["train_config"]["SENTIMENT_WORDS_PATH"]
    batch_size = config["train_config"]["BATCH_SIZE"]
    is_shuffle = int2bool[config["train_config"]["IS_SHUFFLE"]]
    is_loop = int2bool[config["train_config"]["Is_LOOP"]]
    is_sort = int2bool[config["train_config"]["IS_SORT"]]
    nb_epoch = config["train_config"]["NB_EPOCH"]
    dropout_rate = config["train_config"]["DROPOUT_RATE"]
    nb_classes = config["train_config"]["NB_CLASSES"]
    attention_dim = config["train_config"]["ATTENTION_DIM"]
    nb_hops = config["train_config"]["NB_HOPS"]
    use_dropout = int2bool[config["train_config"]["USE_DROPOUT"]]
    optimizer = config["train_config"]["OPTIMIZER"]
    learning_rate = config["train_config"]["LEARNING_RATE"]
    grad_clipper = config["train_config"]["GRAD_CLIPPER"]
    drop_train_path = config["train_config"]["DROP_TRAIN_PATH"]
    drop_path = config["train_config"]["DROP_CHOOSE_DEV_PATH"]
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
    beam_size = config["train_config"]["BEAM_SIZE"]
    depth = config["train_config"]["DEPTH"]
    num_encoder_symbols = config["train_config"]["NUM_ENCODER_SYMBOLS"]
    num_decoder_symbols = config["train_config"]["NUM_DECODER_SYMBOLS"]
    use_beam = int2bool[config["train_config"]["USE_BEAM"]]
    use_residual = int2bool[config["train_config"]["USE_RESIDUAL"]]
    attention_type = config["train_config"]["ATTENTION_TYPE"]
    add_data_root = config["train_config"]["ADD_DATA_ROOT"]
    attn_input_feeding = int2bool[config["train_config"]["ATTN_INPUT_FEEDING"]]
    use_w2v = True

    char_w2v_path = config["w2v_config"]["CHAR_W2V_PATH"]
    char_voc_path = config["w2v_config"]["CHAR_VOC_PATH"]
    char_embedding_matrix_path = config["w2v_config"][
        "CHAR_EMBEDDING_MATRIX_PATH"]
    word_w2v_path = config["w2v_config"]["WORD_W2V_PATH"]
    word_voc_path = config["w2v_config"]["WORD_VOC_PATH"]
    word_embedding_matrix_path = config["w2v_config"][
        "WORD_EMBEDDING_MATRIX_PATH"]
    ans_max_len = config["train_config"]["ANS_MAX_LEN"]
    que_max_len = config["train_config"]["QUE_MAX_LEN"]

    test_samples = read_file(['data/instances_dev_choose_augment_40_class_4.xlsx'])
    with open(char_voc_path, 'rb') as infile:
        char_voc = pickle.load(infile)
    with open(word_voc_path, 'rb') as infile:
        word_voc = pickle.load(infile)
    char_voc_reverse = dict(zip(char_voc.values(), char_voc.keys()))

    instances_choose_dev = make_instances(
        test_samples,
        char_voc,
        word_voc,
        sentiment_words_path,
        ner_dict_path=ner_dict_path,
        pos_dict_path=pos_dict_path,
        use_extra_feature=use_extra_feature,
        question2targets_path=question2targets_path,
        is_training=False,
        need_augment=True)

    data_stream_choose_dev = DataStream(instances=instances_choose_dev,
                                        is_shuffle=is_shuffle,
                                        is_loop=is_loop,
                                        is_training=False,
                                        batch_size=batch_size,
                                        ans_max_len=ans_max_len,
                                        que_max_len=que_max_len,
                                        is_sort=is_sort)
    with tf.Graph().as_default():
        initializer = tf.glorot_uniform_initializer()
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            answer_generator_dev = AnswerGenerator(
                use_w2v=use_w2v,
                rnn_unit='gru',
                dropout_rate=dropout_rate,
                beam_size=beam_size,
                use_beam=use_beam,
                num_decoder_symbols=num_decoder_symbols,
                num_encoder_symbols=num_encoder_symbols,
                optimizer=optimizer,
                learning_rate=learning_rate,
                grad_clipper=grad_clipper,
                global_step=global_step,
                attention_dim=attention_dim,
                nb_hops=nb_hops,
                rnn_dim=rnn_dim,
                lambda_l2=lambda_l2,
                is_training=False,
                sentiment_polarity_multiple=sentiment_polarity_multiple,
                nb_classes=nb_classes,
                batch_size=batch_size,
                dtype=tf.float32,
                depth=depth,
                attn_input_feeding=attn_input_feeding,
                use_dropout=use_dropout,
                attention_type=attention_type,
                use_residual=use_residual,
                use_extra_feature=use_extra_feature,
                ner_dict_size=ner_dict_size,
                pos_dict_size=pos_dict_size,
                extra_feature_dim=extra_feature_dim,
                char_w2v_embedding_matrix_path=char_embedding_matrix_path,
                word_w2v_embedding_matrix_path=word_embedding_matrix_path)

        saver = tf.train.Saver()
        sess = tf.Session()
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        saver.restore(sess, best_path)
        end = decode(sess, answer_generator_dev, char_voc_reverse,
                     data_stream_choose_dev, 'choose_all_40_class_4.txt')
