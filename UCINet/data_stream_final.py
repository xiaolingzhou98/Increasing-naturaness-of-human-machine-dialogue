import numpy as np


class DataStream(object):
    def __init__(self,
                 instances,
                 is_shuffle,
                 is_loop,
                 batch_size,
                 ans_max_len,
                 que_max_len,
                 use_bert,
                 bert_encoder,
                 is_sort=True):
        self.instances = instances
        self.is_shuffle = is_shuffle
        self.is_loop = is_loop
        self.batch_size = batch_size
        self.nb_instances = len(instances)
        self.cur_pointer = 0
        self.ans_max_len = ans_max_len
        self.que_max_len = que_max_len
        self.use_bert = use_bert
        self.bert_encoder = bert_encoder

        # sort instances based on sentence length
        if is_sort:
            self.instances = sorted(instances,
                                    key=lambda instance: (len(instance[
                                        "question"]), len(instance["answer"])))

        # distribute into different buckets
        def make_batches():
            nb_batch = int(np.floor(self.nb_instances / float(self.batch_size)))
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
                                      self.que_max_len, self.use_bert,
                                      self.bert_encoder)
            self.batches.append(cur_batch)

        # shuffle batch index
        self.nb_batch = len(self.batches)
        self.index_array = np.arange(self.nb_batch)
        if self.is_shuffle:
            np.random.shuffle(self.index_array)

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
    def __init__(self, instances, ans_max_len, que_max_len, use_bert,
                 bert_encoder):
        self.instances = instances
        self.batch_size = len(instances)
        self.bert_encoder = bert_encoder
        max_option_length = 5
        que_lens = []
        ans_lens = []
        option_lens = []
        #truths = []
        questions = []
        question_byte=[]
        answers = []
        targets = []
        label_tags = []
        label_tags_change = [[],[],[],[],[]]
        len_tags = []
        options = []
        labels = []
        len_option = []
        question_embedding = []
        # que_indexs = []
        # ans_indexs = []
        count=-1
        for instance in instances:
            count=count+1
            assert isinstance(instance, dict)
            assert len(np.array(instance['que_ww2v_index_sequence'])) == len(
                instance['que_ww2v_index_sequence'])
            assert len(np.array(instance['ans_cw2v_index_sequence'])) == len(
                instance['ans_cw2v_index_sequence'])
            #print(len(np.array(instance['ans_cw2v_index_sequence'])))
            que_lens.append(len(np.array(instance['que_ww2v_index_sequence'])))
            ans_lens.append(len(np.array(instance['ans_cw2v_index_sequence'])))
            option_lens.append([])
            #print(in)
            for j in range(max_option_length):
                if(j<=len(instance['options'])-1):
                    option_lens[count].append(len(np.array(instance['option_word_embedding'][instance['options'][j]])))
                else:
                    option_lens[count].append(0)
            len_option.append(instance['len_option']) #the length of question's options
            label_tags.append(instance['label_tag'])
            for i in range(max_option_length):
                label_tags_change[i].append(instance['label_tag'][i])
            len_tags.append(instance['len_options'])
            #truths.append(instance['sen_label']) #0,1,2
            questions.append(instance['question'])
            answers.append(instance['answer'])
            targets.append(instance['target']) # option
            options.append(instance['options']) # options(list)
            labels.append(instance['labels']) #labels (dict:option:label)
            question_byte.append(instance['question_byte'])
            question_embedding.append(instance['que_ww2v_index_sequence'])#list only word embedding
            # que_indexs.append(instance['question_id'])
            # ans_indexs.append(instance['answer_id'])
            
        self.que_lens = np.array(que_lens, dtype=np.int32)
        self.ans_lens = np.array(ans_lens, dtype=np.int32)
        self.label_tag = np.array(label_tags, dtype=np.int32)
        self.label_tag_change = np.array(label_tags_change,dtype=np.int32)
        self.len_tag = np.array(len_tags,dtype=np.int32)
        #self.truths = np.array(truths, dtype=np.int32)
        self.word_emb_option_lens = np.array(option_lens,dtype=np.int32)
        self.questions = questions
        self.answers = answers
        self.targets = targets
        self.options = options
        self.labels = labels
        self.question_byte=question_byte
        self.question_embedding = question_embedding
        self.len_option = len_option
        # self.ans_ids = np.array(ans_indexs, dtype=np.int32)
        # self.que_ids = np.array(que_indexs, dtype=np.int32)

        in_dim_1 = self.batch_size
        in_dim_ans = ans_max_len
        in_dim_que = que_max_len
        dtype = np.int32
        #no more than five options,there is before set
        for i in range(max_option_length):
            option_name = 'option_ww2v_index_matrix' + str(i+1)
            locals()['option_ww2v_index_matrix' + str(i+1)] = np.zeros((in_dim_1, in_dim_que), dtype=dtype)
        option_ww2v_index_matrix =[]
        que_ww2v_index_matrix = np.zeros((in_dim_1, in_dim_que), dtype=dtype)
        ans_ww2v_index_matrix = np.zeros((in_dim_1, in_dim_ans), dtype=dtype)
        que_cw2v_index_matrix = np.zeros((in_dim_1, in_dim_que), dtype=dtype)
        ans_cw2v_index_matrix = np.zeros((in_dim_1, in_dim_ans), dtype=dtype)
        ans_appendition = np.zeros((in_dim_1, in_dim_que),dtype=dtype)
        que_pos_index_matrix = np.zeros((in_dim_1, in_dim_que), dtype=dtype)
        ans_pos_index_matrix = np.zeros((in_dim_1, in_dim_ans), dtype=dtype)
        que_ner_index_matrix = np.zeros((in_dim_1, in_dim_que), dtype=dtype)
        ans_ner_index_matrix = np.zeros((in_dim_1, in_dim_ans), dtype=dtype)
        que_indicate_target_matrix = np.zeros((in_dim_1, in_dim_que),
                                              dtype=dtype)
        ans_indicate_target_matrix = np.zeros((in_dim_1, in_dim_ans),
                                              dtype=dtype)
        ans_sentiment_polarity_matrix = np.zeros((in_dim_1, in_dim_ans),
                                                 dtype=dtype)
        que_sentiment_polarity_matrix = np.zeros((in_dim_1, in_dim_que),
                                                 dtype=dtype)
        que_skeleton_label_matrix = np.zeros((in_dim_1, in_dim_que), dtype=dtype)
        # 'ans_sentiment_polarity_labels': sentiment_polarity_labels,
        # 'que_sentiment_polarity_labels': np.array([0] * len(que), dtype=dtype),
        # 'que_indicate_target_labels': indicate_target_labels,
        # 'ans_indicate_target_labels':
        #print(self.options)
        for instance_index in range(in_dim_1):
            #print(self.options[instance_index])
            cur_que_len = self.que_lens[instance_index]
            cur_ans_len = self.ans_lens[instance_index]
            #get every option's ww2v
            #print(self.len_option[instance_index])
            for i in range(self.len_option[instance_index]):
                if(i<max_option_length):
                    cur_option_length = option_lens[instance_index][i]
                    #print(cur_option_length)
                    if(cur_option_length>=0):
                        
                        #print(self.options[instance_index][i])
                        option_name = 'option_ww2v_index_matrix' + str(i+1)
                        locals()['option_ww2v_index_matrix' + str(i+1)][instance_index, 0:cur_option_length]= np.array(#the legth need to be one2one
                        instances[instance_index]['option_word_embedding'][self.options[instance_index][i]],
                        dtype=dtype)
                        #print(locals()['option_ww2v_index_matrix' + str(i+1)])
            que_ww2v_index_matrix[instance_index, 0:cur_que_len] = np.array(
                instances[instance_index]['que_ww2v_index_sequence'],
                dtype=dtype)
            que_cw2v_index_matrix[instance_index, 0:cur_que_len] = np.array(
                instances[instance_index]['que_cw2v_index_sequence'],
                dtype=dtype)
            ans_ww2v_index_matrix[instance_index, 0:cur_ans_len] = np.array(
                instances[instance_index]['ans_ww2v_index_sequence'],
                dtype=dtype)
            ans_cw2v_index_matrix[instance_index, 0:cur_ans_len] = np.array(
                instances[instance_index]['ans_cw2v_index_sequence'],
                dtype=dtype)
            que_indicate_target_matrix[
                instance_index, 0:cur_que_len] = np.array(
                    instances[instance_index]['que_indicate_target_labels'],
                    dtype=dtype)
            ans_indicate_target_matrix[
                instance_index, 0:cur_ans_len] = np.array(
                    instances[instance_index]['ans_indicate_target_labels'],
                    dtype=dtype)
            ans_sentiment_polarity_matrix[
                instance_index, 0:cur_ans_len] = np.array(
                    instances[instance_index]['ans_sentiment_polarity_labels'],
                    dtype=dtype)
            que_sentiment_polarity_matrix[
                instance_index, 0:cur_que_len] = np.array(
                    instances[instance_index]['que_sentiment_polarity_labels'],
                    dtype=dtype)
            que_pos_index_matrix[instance_index, 0:cur_que_len] = np.array(
                instances[instance_index]['que_pos_index_sequence'],
                dtype=dtype)
            que_ner_index_matrix[instance_index, 0:cur_que_len] = np.array(
                instances[instance_index]['que_ner_index_sequence'],
                dtype=dtype)
            ans_pos_index_matrix[instance_index, 0:cur_ans_len] = np.array(
                instances[instance_index]['ans_pos_index_sequence'],
                dtype=dtype)
            ans_ner_index_matrix[instance_index, 0:cur_ans_len] = np.array(
                instances[instance_index]['ans_ner_index_sequence'],
                dtype=dtype)

            que_skeleton_label_matrix[
                instance_index, 0:cur_que_len] = np.array(
                    instances[instance_index]['que_skeleton_label'],
                    dtype=dtype)
        #1-5 has some no value,because the len(options) may smaller than 5, if is a zero atrix, then not has the i-th option
        #for i in range(5):
         #   option_name = 'option_ww2v_index_matrix' + str(i+1)
           # option_name_self = 'option_ww2v_index_matrix' + str(i+1)
          #  self.locals()['option_ww2v_index_matrix' + str(i+1)] = locals()['option_ww2v_index_matrix' + str(i+1)]
        for i in range(max_option_length):
            option_ww2v_index_matrix.append(locals()['option_ww2v_index_matrix' + str(i+1)])
        self.option_ww2v_index_matrix = option_ww2v_index_matrix
        self.que_cw2v_index_matrix = que_cw2v_index_matrix
        self.que_ww2v_index_matrix = que_ww2v_index_matrix
        self.ans_cw2v_index_matrix = ans_cw2v_index_matrix
        self.ans_ww2v_index_matrix = ans_ww2v_index_matrix
        self.que_indicate_target_matrix = que_indicate_target_matrix
        self.ans_indicate_target_matrix = ans_indicate_target_matrix
        self.que_skeleton_label_matrix = que_skeleton_label_matrix
        self.ans_sentiment_polarity_matrix = ans_sentiment_polarity_matrix
        self.que_sentiment_polarity_matrix = que_sentiment_polarity_matrix
        self.que_pos_index_matrix = que_pos_index_matrix
        self.que_ner_index_matrix = que_ner_index_matrix
        self.ans_pos_index_matrix = ans_pos_index_matrix
        self.ans_ner_index_matrix = ans_ner_index_matrix
        self.que_bert_matrix = None
        self.ans_bert_matrix = None
        self.que_with_ans = ans_appendition
        if use_bert:
            self.que_bert_matrix = self.bert_encoder.encode_by_model(self.questions)
            self.ans_bert_matrix = self.bert_encoder.encode_by_model(self.answers)

    #def update_bert_vector(self):
    #    self.que_bert_matrix = self.bert_encoder.encode_by_model(self.questions)
    #    self.ans_bert_matrix = self.bert_encoder.encode_by_model(self.answers)
