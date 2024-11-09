import os
import pickle
import pyltp
import re
import json
import jieba
import numpy as np
import pandas as pd
import random

# from extract_skeleton import test##

CN_NUM = {
    '〇': 0,
    '一': 1,
    '二': 2,
    '三': 3,
    '四': 4,
    '五': 5,
    '六': 6,
    '七': 7,
    '八': 8,
    '九': 9,
    '零': 0,
    '壹': 1,
    '贰': 2,
    '叁': 3,
    '肆': 4,
    '伍': 5,
    '陆': 6,
    '柒': 7,
    '捌': 8,
    '玖': 9
}
CN_UNIT = {
    '十': 10,
    '拾': 10,
    '百': 100,
    '佰': 100,
    '千': 1000,
    '仟': 1000,
    '万': 10000,
    '萬': 10000,
    '亿': 100000000,
    '億': 100000000,
    '兆': 1000000000000,
}
LTP_DATA_DIR ='/home/nlp/Desktop/answer_generation1/ltp_data_v3.4.0/'
_GO = '_GO'
EOS = '_EOS'  # also function as PAD
UNK = '_UNK'

extra_tokens = [_GO, EOS, UNK]

start_token = extra_tokens.index(_GO)  # start_token = 0
end_token = extra_tokens.index(EOS)  # end_token = 1
unk_token = extra_tokens.index(UNK)


def read_file(file_path_list):
    samples = []
    for file_path in file_path_list:
        if os.path.isfile(file_path):
            _, file = os.path.split(file_path)
            try:
                if "xlsx" in file or "xls" in file:
                    data_frame = pd.read_excel(file_path)
                if "csv" in file:
                    data_frame = pd.read_csv(file_path)
            except (ValueError, TypeError):
                print("Please check file type")
            data_frame.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
            samples += [
                data_frame.iloc[index].to_dict()
                for index in range(len(data_frame))
            ]
    processed_samples = []
    for sample in samples:
        sample.update({
            "question": pre_processed(str(sample["question"]))
        })
        if 'answer' in sample.keys():
            sample.update({"answer": pre_processed(str(sample["answer"]))})
        if 'question_type' not in sample.keys():
            sample.update({'question_type': 0, 'domain': '未归类'})
        processed_samples.append(sample)
    return samples


def seq2words(seq, inverse_target_dictionary):
    print(seq)
    words = []
    for w in seq[:, -1]:
        if w == end_token:
            break
        if w in inverse_target_dictionary:
            words.append(inverse_target_dictionary[w])
        else:
            words.append(UNK)
    return ' '.join(words)


def chinese2digit(sentence):
    unit = 0  # current
    l_dig = []  # digest
    for cn_dig in reversed(sentence):
        if cn_dig in CN_UNIT:
            unit = CN_UNIT.get(cn_dig)
            if unit == 10000 or unit == 100000000:
                l_dig.append(unit)
                unit = 1
        else:
            dig = CN_NUM.get(cn_dig)
            if unit:
                dig *= unit
                unit = 0
            l_dig.append(dig)
    if unit == 10:
        l_dig.append(10)
    val, tmp = 0, 0
    for x in reversed(l_dig):
        if x == 10000 or x == 100000000:
            val += tmp * x
            tmp = 0
        else:
            tmp += x
    val += tmp
    return val


def unify_different_number(sentence):
    number_pattern = re.compile('[一二三四五六七八九零十百千万亿]*')
    numbers = number_pattern.findall(sentence)
    numbers = [number for number in numbers if number != '']
    if numbers:
        for number in numbers:
            sentence = sentence.replace(number, str(chinese2digit(number)))
    return sentence


def pre_processed(sentence):
    # remove special symbols
    sentence = re.sub(u'[^\u4e00-\u9fa5a-zA-Z0-9-~，。,.]', "", sentence)
    # unify different type number
    if len(sentence) > 35:
        return sentence[0:35]
    return sentence
    # return unify_different_number(sentence)


def make_corpus(samples):
    que_raw_corpus = [sample["question"] for sample in samples]
    ans_raw_corpus = [sample["answer"] for sample in samples]
    char_corpus = unique_list(ans_raw_corpus + que_raw_corpus)
    word_corpus = [jieba.lcut(sentence) for sentence in char_corpus]
    return char_corpus, word_corpus


def unique_list(list_):
    assert isinstance(list_, list)
    if list_ != []:
        return list(set(list_))
    return []


def reverse_dict(dict_):
    assert isinstance(dict_, dict)
    return {v: k for k, v in dict_.items()}


def split_targets(str_):
    return re.split(r'[和，,、/]', str_)


def gather_targets_for_samples(samples, is_training, question2target_path):
    if is_training:
        que2targets = {}
        for sample in samples:
            assert isinstance(sample, dict)
            question = sample["question"]
            if int(sample["question_type"]) == 1:
                if question in que2targets.keys():
                    targets_ori = que2targets[question]
                    que2targets.update({
                        question:
                            targets_ori | set(
                                split_targets(str(sample["label"])))
                    })
                else:
                    que2targets.update(
                        {question: set(split_targets(str(sample["label"])))})
            else:
                que2targets.update({question: set()})
        with open(question2target_path, 'wb') as out_file:
            pickle.dump(que2targets, out_file)
    else:
        with open(question2target_path, 'rb') as infile:
            que2targets = pickle.load(infile)

    for question in que2targets.keys():
        targets = que2targets[question]
        if targets is not None:
            if 'all' in targets:
                targets.remove('all')
            if 'none_1' in targets:
                targets.remove('none_1')
            if 'none_2' in targets:
                targets.remove('none_2')
            if '' in targets:
                targets.remove('')
        que2targets.update({question: targets})
    return que2targets


def make_true_sen_labels_for_choose(targets, label):
    target_labels = {}
    for target in targets:
        # TODO: correct train data
        if str(label) == 'all' or target in str(label):
            target_labels.update({target: 1})
        elif str(label) == 'none_1':
            target_labels.update({target: 2})
        else:
            target_labels.update({target: 0})
    # print(target_labels)
    return target_labels


def instance_augment(instances, question2targets):
    end_instances = []
    retain_keys = [key for key in instances[0].keys() if key != 'label']
    for instance in instances:
        if int(instance["question_type"]) == 1:
            question = instance["question"]
            targets = question2targets[question]
            label = instance["label"]
            targets_sen_labels = make_true_sen_labels_for_choose(
                targets, label)
            for target_, sen_label_ in targets_sen_labels.items():
                instance_temp = {k: instance[k] for k in retain_keys}
                instance_temp.update({
                    "target": target_,
                    "sen_label": sen_label_
                })
                end_instances.append(instance_temp)
                # print(instance)
        else:
            instance.update({"target": "none", "sen_label": instance["label"]})
            end_instances.append(instance)
    return end_instances


def make_instances(samples,
                   char_voc,
                   word_voc,
                   sentiment_words_path,
                   question2targets_path,
                   need_augment,
                   is_training,
                   use_extra_feature,
                   ner_dict_path,
                   pos_dict_path,
                   dtype=np.int32):
    # TODO: build sentiment words for own data
    positive_words, negative_words = load_sentiment_words(sentiment_words_path)
    question2targets = gather_targets_for_samples(samples, is_training,
                                                  question2targets_path)
    if need_augment:
        samples = instance_augment(samples, question2targets)

    questions = unique_list([sample['question'] for sample in samples])
    # question2skeleton = gather_skeleton_indicator(questions)

    if use_extra_feature:
        with open(ner_dict_path, 'rb') as infile:
            ner_dict = pickle.load(infile)
        with open(pos_dict_path, 'rb') as infile:
            pos_dict = pickle.load(infile)

        ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
        cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
        pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
        segmentor = pyltp.Segmentor()
        segmentor.load(cws_model_path)
        postagger = pyltp.Postagger()
        postagger.load(pos_model_path)
        recognizer = pyltp.NamedEntityRecognizer()
        recognizer.load(ner_model_path)

    assert set(positive_words) & set(negative_words) == set()

    sentiment_words = positive_words + negative_words
    sentiment_words.sort(key=lambda sen_word: len(sen_word), reverse=True)
    sentiment_words_dic = {}
    for sen_word in sentiment_words:
        if sen_word in positive_words:
            sentiment_words_dic[sen_word] = 1
        else:
            sentiment_words_dic[sen_word] = 0

    for sample in samples:

        assert isinstance(sample, dict)
        if 'sen_label' not in sample.keys():
            sample['sen_label'] = sample['label']
        que_ww2v_index_sequence = []
        if "answer" in sample.keys():
            ans_ww2v_index_sequence = []
            ans = sample["answer"]
            for word in jieba.lcut(ans):
                for _ in range(len(word)):
                    ans_ww2v_index_sequence.append(word_voc[word])
            ans_ww2v_index_sequence = np.array(ans_ww2v_index_sequence,
                                               dtype=dtype)
        que = sample["question"]
        for word in jieba.lcut(que):
            for _ in range(len(word)):
                try:
                    que_ww2v_index_sequence.append(word_voc[word])
                except:
                    que_ww2v_index_sequence.append(2)

        que_ww2v_index_sequence = np.array(que_ww2v_index_sequence,
                                           dtype=dtype)

        def make_extra_index_sequence(str_):
            pos_end = []
            ner_end = []
            words = segmentor.segment(str_)
            postages = postagger.postag(words)
            netags = recognizer.recognize(words, postages)

            for word_index in range(len(list(words))):
                word = list(words)[word_index]
                for _ in range(len(word)):
                    pos_end.append(pos_dict[list(postages)[word_index]])
                    ner_temp = list(netags)[word_index]
                    if '-' in ner_temp:
                        ner_temp = ner_temp[ner_temp.index('-') + 1:]
                    ner_end.append(ner_dict[ner_temp])
            return np.array(ner_end, dtype=dtype), np.array(pos_end,
                                                            dtype=dtype)

        if use_extra_feature:
            que_ner_index_sequence, que_pos_index_sequence = make_extra_index_sequence(
                que)
            if "answer" in sample.keys():
                ans_ner_index_sequence, ans_pos_index_sequence = make_extra_index_sequence(
                    ans)
        else:
            que_ner_index_sequence = np.array([0] * len(que), dtype=dtype)
            que_pos_index_sequence = np.array([0] * len(que), dtype=dtype)
            if "answer" in sample.keys():
                ans_ner_index_sequence = np.array([0] * len(ans), dtype=dtype)
                ans_pos_index_sequence = np.array([0] * len(ans), dtype=dtype)

        que_cw2v_index_sequence = [
            char_voc[char] for char in sample['question']
        ]
        que_cw2v_index_sequence = np.array(que_cw2v_index_sequence,
                                           dtype=dtype)
        if "answer" in sample.keys():
            ans_cw2v_index_sequence = [char_voc[char] for char in sample['answer']]
            ans_cw2v_index_sequence = np.array(ans_cw2v_index_sequence,
                                               dtype=dtype)
        # que_skeleton_label = question2skeleton[que]

        assert len(que_cw2v_index_sequence) == len(que_ww2v_index_sequence)

        def make_sentiment_polarity_labels(str_):
            # 0：neutral, 1: positive 2: negative
            if sample["sen_label"] == 0:
                sentiment_polarity_labels = np.array(
                    [0] * len(str_), dtype=dtype)
            elif sample["sen_label"] == 1:
                sentiment_polarity_labels = np.array(
                    [1] * len(str_), dtype=dtype)
            else:
                sentiment_polarity_labels = np.array(
                    [2] * len(str_), dtype=dtype)
            return sentiment_polarity_labels
        def make_class_labels1():
            # 0：neutral, 1: positive 2: negative
            for i in range(40):
                if(str(sample["classes"])==str(i)):
                    class_labels = np.array(
                    [i] * len(que), dtype=dtype)
                    return class_labels
        def make_class_labels2():
            for i in range(40):
                if(str(sample["classes"])==str(i)):
                    class_labels1 = np.array(
                    [i] * len(ans), dtype=dtype)
                    return class_labels1
        def make_indicate_target_labels():
            # 0: not 1: is

            indicate_target_labels = np.array([0] * len(que), dtype=dtype)
            if "target" in sample.keys():
                specify_target = str(sample["target"])
                if specify_target in que:
                    indicate_target_labels[que.index(specify_target): que.index(specify_target) + len(specify_target)] = \
                        np.ones(len(specify_target), dtype=dtype)
            return indicate_target_labels

        if "answer" in sample.keys():
            ans_sentiment_polarity_labels = make_sentiment_polarity_labels(ans)
            ans_class_labels = make_class_labels2()
            sample.update({
                'ans_ww2v_index_sequence':
                    ans_ww2v_index_sequence,
                'ans_cw2v_index_sequence':
                    ans_cw2v_index_sequence,
                'ans_class_labels':
                    ans_class_labels,
                'ans_sentiment_polarity_labels':
                    ans_sentiment_polarity_labels,
                'ans_indicate_target_labels':
                    np.array([0] * len(ans), dtype=dtype),
                'ans_ner_index_sequence':
                    ans_ner_index_sequence,
                'ans_pos_index_sequence':
                    ans_pos_index_sequence

            })
        que_sentiment_polarity_labels = make_sentiment_polarity_labels(que)
        indicate_target_labels = make_indicate_target_labels()
        que_class_labels = make_class_labels1()
        sample.update({
            'que_ww2v_index_sequence':
                que_ww2v_index_sequence,
            'que_cw2v_index_sequence':
                que_cw2v_index_sequence,
            'que_sentiment_polarity_labels':
                que_sentiment_polarity_labels,
            'que_class_labels':
                que_class_labels,
            'que_indicate_target_labels':
                indicate_target_labels,
            'que_ner_index_sequence':
                que_ner_index_sequence,
            'que_pos_index_sequence':
                que_pos_index_sequence
        })
        # 'question_id': sen_voc[sample["question"]],
        # 'answer_id': sen_voc[sample["answer"]]

    return samples


def generate_vocabulary(voc_path, sentence_bag):
    # sentence_bag is equal to char_corpus
    voc = {}
    sentence_bag = unique_list(sentence_bag)
    for index, sentence in enumerate(sentence_bag):
        voc.update({sentence: index})
    with open(voc_path, 'wb') as outfile:
        pickle.dump(voc, outfile)
    return voc


def load_sentiment_words(sentiment_words_path):
    with open(sentiment_words_path, encoding="utf-8") as infile:
        raw_words = infile.readlines()
    all_words = []
    for word in raw_words:
        if word[-1] == '\n':
            all_words.append(word[:-1])
        else:
            all_words.append(word)
    segment_point = all_words.index('')
    positive_words = all_words[:segment_point]
    negative_words = all_words[segment_point + 1:]
    positive_words = unique_list(positive_words)
    negative_words = unique_list(negative_words)
    positive_words.sort(key=lambda word: len(word), reverse=True)
    negative_words.sort(key=lambda word: len(word), reverse=True)
    return positive_words, negative_words


def un_padding_vectors(ori_vec):
    assert len(ori_vec.shape) == 2 or len(ori_vec.shape) == 3
    dim = ori_vec.shape[-1]
    zero_vec = np.zeros(dim)
    if len(ori_vec.shape) == 2:
        for i in range(ori_vec.shape[0]):
            # print('b', i)
            if (zero_vec == ori_vec[i, :]).all():
                return ori_vec[:i, :]
    else:
        end = []
        for i in range(ori_vec.shape[0]):
            # print('a', i)
            end.append(un_padding_vectors(ori_vec[i, :, :]))
        return end


def save_dic_to_excel(dic_, excel_file):
    # used to make extract skeleton information's model
    questions = [question for question in dic_.keys()]
    targets = []
    for question in questions:
        temp = [
            target for target in dic_[question]
            if target != 'none_1' and target != 'none_2'
        ]
        if len(temp) > 0:
            targets.append(','.join(temp))
        else:
            targets.append('none')
    data_frame = pd.DataFrame({'question': questions, 'target': targets})
    data_frame.to_excel(excel_file)


def visual_skeleton(sentence, indicator):
    target_str = ''
    type_str = ''
    indicator = list(indicator)
    for index in range(len(sentence)):
        if indicator[index] == 1:
            type_str += sentence[index]
            if index + 1 < len(sentence) and indicator[index] != indicator[
                index +
                1] and index - 1 > 0 and indicator[index] == indicator[
                index - 1]:
                type_str += ','
        if indicator[index] == -1:
            target_str += sentence[index]
            if index + 1 < len(sentence) and indicator[index] != indicator[
                index +
                1] and index - 1 > 0 and indicator[index] == indicator[
                index - 1]:
                target_str += ','
    return [target_str, type_str]


def drop_instances_to_excel(instances, excel_file):
    question = [instance['question'] for instance in instances]
    answer = [instance['answer'] for instance in instances]
    target = [instance['target'] for instance in instances]
    domain = [instance['domain'] for instance in instances]

    question_type = [instance['question_type'] for instance in instances]
    sen_label = [instance['sen_label'] for instance in instances]
    data_frame = pd.DataFrame({
        'domain': domain,
        'question_type': question_type,
        'question': question,
        'answer': answer,
        'target': target,
        'sen_label': sen_label
    })
    data_frame.to_excel(excel_file)


def split_train_dev(instances, dev_size=0.2):
    assert isinstance(instances, list)
    random.shuffle(instances)
    boundary = int((1. - dev_size) * len(instances))
    instances_train = instances[:boundary]
    instances_dev = instances[boundary:]
    return instances_train, instances_dev


def get_info_from_original_data(dic_ori):
    label_dic = {
        'No': 0,
        'Yes': 1,
        'Depends': 2
    }
    samples = []
    try:
        assert len(dic_ori['answers']) == len(dic_ori['yesno_answers'])

        for answer_id in range(len(dic_ori['answers'])):
            samples.append(
                [dic_ori['question'], dic_ori['answers'][answer_id],
                 label_dic[dic_ori['yesno_answers'][answer_id]]]
            )
    except:
        pass
    return samples


def get_data_from_added_file(file_dir):
    if os.path.exists('data/dureader_judge.csv'):
        os.remove('data/dureader_judge.csv')
    samples = []
    i = 0
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            with open(os.path.join(root, file), encoding='utf-8') as infile:
                cur_line = infile.readline()
                while cur_line:
                    samples += get_info_from_original_data(eval(cur_line))
                    cur_line = infile.readline()
                    i += 1
                    if i % 1000 == 0:
                        print(i, end='-:')
                    if i % 10000 == 0:
                        print('\n')

    df = pd.DataFrame(samples, columns=['question', 'answer', 'sen_label'])

    df.to_csv('data/dureader_judge.csv', encoding='utf_8_sig')
    return samples


if __name__ == "__main__":
    with open('config.json', encoding='utf-8') as infile:
        config = json.load(infile)
    data_path = config["train_config"]["ADD_DATA_ROOT"]
    sample_temp = get_data_from_added_file(data_path)

    # data_path = config["train_config"]["DATA_PATH"]
    # samples = read_file(data_path)
    # question2targets = gather_targets_for_samples(samples)
    # save_dic_to_excel(question2targets, excel_file='skeleton.xlsx')
    # instances = make_instances()
    # test.make_skeleton_indicator('今天是个好日子')
