import numpy as np


class DataStream(object):
    def __init__(self,
                 instances,
                 is_shuffle,
                 is_loop,
                 batch_size,
                 ans_max_len,
                 que_max_len,
                 is_training,
                 is_sort=True):
        self.instances = instances
        self.is_shuffle = is_shuffle
        self.is_loop = is_loop
        self.batch_size = batch_size
        self.nb_instances = len(instances)
        self.cur_pointer = 0
        self.ans_max_len = ans_max_len
        self.que_max_len = que_max_len
        self.is_training = is_training

        # sort instances based on sentence length
        if is_sort:
            self.instances = sorted(instances,
                                    key=lambda instance: len(instance["question"]))

        # distribute into different buckets
        def make_batches():
            nb_batch = int(np.ceil(self.nb_instances / float(self.batch_size)))
            return [(i * batch_size,
                     min(self.nb_instances, (i + 1) * self.batch_size))
                    for i in range(nb_batch)]

        batch_spans = make_batches()
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instance = []
            for i in range(batch_start, batch_end):
                cur_instance.append(self.instances[i])
            # TODO: complete building batch class
            cur_batch = InstanceBatch(cur_instance, self.ans_max_len,
                                      self.que_max_len, self.is_training)
            self.batches.append(cur_batch)

        # shuffle batch index
        self.nb_batch = len(self.batches)
        self.index_array = np.arange(self.nb_batch)
        if self.is_shuffle:
            np.random.shuffle(self.index_array)

    def __len__(self):
        return len(self.instances)

    def gather_specific_question_type_instance(self, type_id):
        """
        gather sub samples by the given question type
        :param type_id: question type id, (0:judge question, 1:choose question), int
        :return: list of given type question samples, list[dict]
        """
        return [
            instance for instance in self.instances
            if instance["question_type"] == type_id
        ]

    def gather_specific_domain_instance(self, domain):
        return [
            instance for instance in self.instances
            if instance["domain"] == domain
        ]

    def next_batch(self):
        if self.cur_pointer >= self.nb_batch:
            if not self.is_loop:
                return None
            self.cur_pointer = 0
            if self.is_shuffle:
                np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def shuffle(self):
        if self.is_shuffle:
            np.random.shuffle(self.index_array)

    def reset(self):
        self.cur_pointer = 0

    def get_nb_batch(self):
        return self.nb_batch

    def get_nb_instance(self):
        return self.nb_instances

    def get_batch(self, i):
        if i >= self.nb_batch:
            return None
        return self.batches[self.index_array[i]]


class InstanceBatch(object):
    def __init__(self, instances, ans_max_len, que_max_len, is_training):
        self.instances = instances
        self.batch_size = len(instances)
        self.is_training = is_training

        que_lens = []
        ans_lens = []
        options = []
        labels = []
        #truths = []
        targets = []
        classes = []
        questions = []
        answers = []
        # que_indexs = []
        # ans_indexs = []
        for instance in instances:
            assert isinstance(instance, dict)
            assert len(np.array(instance['que_ww2v_index_sequence'])) == len(
                instance['que_ww2v_index_sequence'])

            que_lens.append(len(np.array(instance['que_ww2v_index_sequence'])))
            #truths.append(instance['sen_label'])
            questions.append(instance['question'])
            options.append(instance['options'])
            labels.append(instance['labels'])
            classes.append(instance['class'])
            targets.append(instance['target'])
            if self.is_training:
                answers.append((instance['answer']))
                ans_lens.append(len(np.array(instance['ans_ww2v_index_sequence'])))

            # que_indexs.append(instance['question_id'])
            # ans_indexs.append(instance['answer_id'])
        self.classes = classes
        self.que_lens = np.array(que_lens, dtype=np.int32)
        self.targets = targets
        self.options = options
        self.labels = labels
        #self.truths = np.array(truths, dtype=np.int32)
        self.questions = questions
        # self.ans_ids = np.array(ans_indexs, dtype=np.int32)
        # self.que_ids = np.array(que_indexs, dtype=np.int32)

        in_dim_1 = self.batch_size
        in_dim_ans = ans_max_len
        in_dim_que = que_max_len
        dtype = np.int32

        que_ww2v_index_matrix = np.ones((in_dim_1, in_dim_que), dtype=dtype)
        que_cw2v_index_matrix = np.ones((in_dim_1, in_dim_que), dtype=dtype)
        que_pos_index_matrix = np.zeros((in_dim_1, in_dim_que), dtype=dtype)
        que_ner_index_matrix = np.zeros((in_dim_1, in_dim_que), dtype=dtype)
        que_sentiment_polarity_matrix = np.zeros((in_dim_1, in_dim_que),
                                                 dtype=dtype)
        que_class_matrix = np.zeros((in_dim_1, in_dim_que),
                                                 dtype=dtype)
        que_indicate_target_matrix = np.zeros((in_dim_1, in_dim_que),
                                              dtype=dtype)

        if self.is_training:
            self.ans_lens = np.array(ans_lens, dtype=np.int32)

            ans_sentiment_polarity_matrix = np.zeros((in_dim_1, in_dim_ans),
                                                     dtype=dtype)
            ans_class_matrix = np.zeros((in_dim_1, in_dim_ans),
                                                     dtype=dtype)
            ans_indicate_target_matrix = np.zeros((in_dim_1, in_dim_ans),
                                                  dtype=dtype)
            self.answers = answers
            ans_ner_index_matrix = np.zeros((in_dim_1, in_dim_ans), dtype=dtype)
            ans_pos_index_matrix = np.zeros((in_dim_1, in_dim_ans), dtype=dtype)
            ans_cw2v_index_matrix = np.ones((in_dim_1, in_dim_ans), dtype=dtype)
            ans_ww2v_index_matrix = np.ones((in_dim_1, in_dim_ans), dtype=dtype)
        # 'ans_sentiment_polarity_labels': sentiment_polarity_labels,
        # 'que_sentiment_polarity_labels': np.array([0] * len(que), dtype=dtype),
        # 'que_indicate_target_labels': indicate_target_labels,
        # 'ans_indicate_target_labels':
        for instance_index in range(in_dim_1):
            cur_que_len = self.que_lens[instance_index]
            que_ww2v_index_matrix[instance_index, 0:cur_que_len] = np.array(
                instances[instance_index]['que_ww2v_index_sequence'],
                dtype=dtype)
            que_cw2v_index_matrix[instance_index, 0:cur_que_len] = np.array(
                instances[instance_index]['que_cw2v_index_sequence'],
                dtype=dtype)
            que_pos_index_matrix[instance_index, 0:cur_que_len] = np.array(
                instances[instance_index]['que_pos_index_sequence'],
                dtype=dtype)
            que_ner_index_matrix[instance_index, 0:cur_que_len] = np.array(
                instances[instance_index]['que_ner_index_sequence'],
                dtype=dtype)
            que_sentiment_polarity_matrix[
            instance_index, 0:cur_que_len] = np.array(
                instances[instance_index]['que_sentiment_polarity_labels'],
                dtype=dtype)
            que_class_matrix[
            instance_index, 0:cur_que_len] = np.array(
                instances[instance_index]['que_class_labels'],
                dtype=dtype)
            que_indicate_target_matrix[
            instance_index, 0:cur_que_len] = np.array(
                instances[instance_index]['que_indicate_target_labels'],
                dtype=dtype)
            self.que_cw2v_index_matrix = que_cw2v_index_matrix
            self.que_ww2v_index_matrix = que_ww2v_index_matrix
            self.que_sentiment_polarity_matrix = que_sentiment_polarity_matrix
            self.que_class_matrix = que_class_matrix
            self.que_indicate_target_matrix = que_indicate_target_matrix
            self.que_pos_index_matrix = que_pos_index_matrix
            self.que_ner_index_matrix = que_ner_index_matrix

            if self.is_training:
                cur_ans_len = self.ans_lens[instance_index]
                ans_ww2v_index_matrix[instance_index, 0:cur_ans_len] = np.array(
                    instances[instance_index]['ans_ww2v_index_sequence'],
                    dtype=dtype)
                ans_cw2v_index_matrix[instance_index, 0:cur_ans_len] = np.array(
                    instances[instance_index]['ans_cw2v_index_sequence'],
                    dtype=dtype)
                ans_pos_index_matrix[instance_index, 0:cur_ans_len] = np.array(
                    instances[instance_index]['ans_pos_index_sequence'],
                    dtype=dtype)
                ans_ner_index_matrix[instance_index, 0:cur_ans_len] = np.array(
                    instances[instance_index]['ans_ner_index_sequence'],
                    dtype=dtype)
                ans_sentiment_polarity_matrix[
                instance_index, 0:cur_ans_len] = np.array(
                    instances[instance_index]['ans_sentiment_polarity_labels'],
                    dtype=dtype)
                ans_class_matrix[
                instance_index, 0:cur_ans_len] = np.array(
                    instances[instance_index]['ans_class_labels'],
                    dtype=dtype)
                ans_indicate_target_matrix[
                instance_index, 0:cur_ans_len] = np.array(
                    instances[instance_index]['ans_indicate_target_labels'],
                    dtype=dtype)
                self.ans_cw2v_index_matrix = ans_cw2v_index_matrix
                self.ans_ww2v_index_matrix = ans_ww2v_index_matrix
                self.ans_sentiment_polarity_matrix = ans_sentiment_polarity_matrix
                self.ans_class_matrix = ans_class_matrix
                self.ans_indicate_target_matrix = ans_indicate_target_matrix
                self.ans_pos_index_matrix = ans_pos_index_matrix
                self.ans_ner_index_matrix = ans_ner_index_matrix



