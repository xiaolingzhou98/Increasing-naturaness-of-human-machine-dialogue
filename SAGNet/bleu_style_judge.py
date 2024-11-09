import json
from utils import read_file
import jieba
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
#from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction
#smooth = SmoothingFunction()
def make_corpus(corpus):
    word_corpus = [jieba.lcut(sentence) for sentence in corpus]
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
        if question not in question_dict.keys():
            question_dict.update({question: tuple([samples[i]['answer']])})
        else:
            question_dict.update({
                question: question_dict[question] + tuple([samples[i]['answer']])})
    return question_dict


if __name__ == "__main__":
    fw = open("judge_15.txt","w")
    question2answers = train_model()
    f = open('best_judge_all_fromchoose.txt', 'r', encoding='utf-8')
    content = f.readlines()
    #number= len(content)/23
    print(len(content))
    count = 0
    candidate_best = []
    reference_best = []
    while(count<=len(content)-16):
        #maxbleu1 = 0
        #maxbleu2 = 0
        #maxbleu3 = 0
        #maxbleu4 = 0
        m1 = dict()
        #m1=0
        for i in range(count,count+16):
            answer = ''.join(content[i].split('\t')[1][7:].split(' '))
            question = ''.join(content[i].split('\t')[0][9:].split(' '))
            candidate=[jieba.lcut(answer)]
            #print(candidate)
            reference=[make_corpus(question2answers[question])]
            #print(reference)
            #candidate = [jieba.lcut(answer)]
            #print(candidate)
            #answers = list(question2answers[question])
            #for each in answers:
            #    reference =  ''.join(each)
            #print(reference)
            bleu1 = corpus_bleu(reference, candidate, weights=(1, 0, 0, 0))
            bleu2 = corpus_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
            bleu3 = corpus_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
            bleu4 = corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
            m1.update({i:bleu4})
        m1_list=list(m1.items())
        m1_list.sort(key=lambda x:x[1],reverse=True)
        #get the biggest five index
        for i in range(15):
            fw.write(content[m1_list[i][0]])
            answer_best = ''.join(content[m1_list[i][0]].split('\t')[1][7:].split(' '))
            question_best = ''.join(content[m1_list[i][0]].split('\t')[0][9:].split(' '))
            candidate_best.append(jieba.lcut(answer_best))
            reference_best.append(make_corpus(question2answers[question_best]))
            print(str(m1_list[i][0]))
        count=count+16
    print('Cumulative 1-gram: %f' %
          corpus_bleu(reference_best, candidate_best, weights=(1, 0, 0, 0)))
    print('Cumulative 2-gram: %f' %
          corpus_bleu(reference_best, candidate_best, weights=(0.5, 0.5, 0, 0)))
    print('Cumulative 3-gram: %f' %
          corpus_bleu(reference_best, candidate_best, weights=(0.33, 0.33, 0.33, 0)))
    print('Cumulative 4-gram: %f' %
          corpus_bleu(reference_best, candidate_best, weights=(0.25, 0.25, 0.25, 0.25)))
    f.close()
    fw.close()
