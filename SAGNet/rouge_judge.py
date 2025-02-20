import json
from utils import read_file
import jieba
from sumeval.metrics.rouge import RougeCalculator
rouge = RougeCalculator(stopwords=True, lang="zh")
# rouge should compute avg

def make_corpus(corpus):
    word_corpus = [sentence for sentence in corpus]
    return word_corpus


def train_model():
    with open('judge_config_.json', encoding='utf-8') as infile:
        config = json.load(infile)
    # data_path = config["train_config"]["DATA_PATH"]
    # add_data_root = config["train_config"]["ADD_DATA_ROOT"]

    # samples = read_file([data_path, add_data_root])
    samples = read_file(['/home/nlp/Desktop/AntNet-rverseQA-master/data/data_judge.xlsx'])
    # samples = read_file([add_data_root])
    question2answers = gather_question_dict(samples)
    return question2answers


def gather_question_dict(samples):
    question_dict = {}
    for i in range(len(samples)):
        question = samples[i]['question']
        #label = samples[i]['label']
        #print(label)
        #print(question+str(label))
        if question not in question_dict.keys():
            question_dict.update({question: tuple([samples[i]['answer']])})
        else:
            question_dict.update({
                question: question_dict[question] + tuple([samples[i]['answer']])})
    return question_dict


if __name__ == "__main__":
    fw = open("rouge_judge_15.txt","w")
    question2answers = train_model()
    f = open('best_judge_all_fromchoose.txt', 'r', encoding='utf-8')
    content = f.readlines()
    #number= len(content)/23
    #print(len(content))
    count = 0
    best_value = []
    while(count<=len(content)-23):
        #maxbleu1 = 0
        #maxbleu2 = 0
        #maxbleu3 = 0
        #maxbleu4 = 0
        m1 = dict()
        #m1=0
        for i in range(count,count+23):
            print(i)
            answer = ''.join(content[i].split('\t')[1][7:].split(' '))
            question = ''.join(content[i].split('\t')[0][9:].split(' '))
            if(question in question2answers.keys()):
                reference=make_corpus(question2answers[question])
                #rouge_value = rouge.rouge_l(answer, reference)
                rouge_value = rouge.rouge_n(answer,reference,n=1)
                #rouge_value = rouge.rouge_n(answer,reference,n=2)
                m1.update({i:rouge_value})
        m1_list=list(m1.items())
        m1_list.sort(key=lambda x:x[1],reverse=True)
        #print(len(m1_list)) 
        if(len(m1_list)>=15):
        #get the biggest five index
            for i in range(15):
                fw.write(content[m1_list[i][0]])
                best_value.append(m1_list[i][1])
                #print(str(m1_list[i][1]))
        count=count+23
    sum_total = 0
    for each in best_value:
        sum_total=sum_total+float(each)
    print(sum_total)
    print(len(best_value))
    print('Cumulative rouge: %f' %
          (sum_total/len(best_value)))
    f.close()
    fw.close()