import re
import jieba.posseg as pseg
import codecs
from .langconv import Converter

filtrate = re.compile(u'[^\u4E00-\u9FA5A0-9a-zA-Z-~，。？！、,.?!]')


# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line


def input1(word):
    word = word.replace(' ', '，')
    word = word.replace('~', '，').replace('…', '，')
    word = word.replace(',', '，').replace('。', '，').replace('.', '，').replace('！', '，').replace('？', '，').replace('!',
                                                                                                                  '，').replace(
        '?', '，').replace('：', '，').replace('；', '，')
    word = filtrate.sub(r'', word)
    words = word.split('，')
    string = [x for x in words if len(x) > 0]
    return string


def data_prepare(words):
    dataList = []
    flagList = []

    for i in range(len(words)):
        # datalist
        dataList.append(list(words[i]))

        # flaglist
        sequence = []
        sequence_flag = []
        data = []
        data_flag = []
        lines = pseg.cut(words[i])
        for word, flag in lines:
            sequence.append(list(word))
            sequence_flag.append(flag)
        for s in range(len(sequence_flag)):
            for zi in sequence[s]:
                data.append(zi)
                data_flag.append(sequence_flag[s])
        flagList.append(data_flag)
    return dataList, flagList


def dealText_1(string):
    string_ori = string
    string = cht_to_chs(string)
    string = filtrate.sub(r'', string)
    string = string.upper()
    # 符号    （没有过英文冒号）
    string = string.replace('。', '，').replace('？', '，').replace('！', '，')
    string = string.replace(',', '，').replace('.', '，').replace('?', '，').replace('!', '，')
    if  len(string) != len(string_ori):
        print(string)
        print(string_ori)
    # string = string.split('，')
    # string = [x for x in string if len(x)>0]
    return string


def dealText(string):
    if isinstance(string, str):
        return [dealText_1(string)]
    else:
        return [dealText_1(sen) for sen in string]


def write(word, flag):
    fw = codecs.open('extract_skeleton/data/sample_test.txt', 'w', 'utf-8')
    for i in range(len(word)):
        for j in range(len(word[i])):
            line = ''.join([word[i][j] + '\t' + flag[i][j] + '\t']) + '\n'
            fw.writelines(line)
        fw.writelines('\n')
    fw.close()


def writetxt(string):
    string = dealText(string)
    word, flag = data_prepare(string)
    write(word, flag)
    return string
