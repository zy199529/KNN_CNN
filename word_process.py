import codecs
import os
import shutil

import numpy as np


def stop_words():  # 读取停用词
    stop_words = []
    with open('./data/stopwords_en.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    return stop_words


def read_file(filename):
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                labels.append(label)
                contents.append(content)
            except:
                pass
    return labels, contents


def cleanReview(content):
    # 数据处理函数
    a = []
    stop_word = stop_words()
    content = content.replace('#', '').replace('=', '').replace("\\", "").replace("\'", "").replace('/', '').replace(
        '"', '').replace(',', '').replace(
        '.', '').replace('?', '').replace('(', '').replace(')', '')
    content = content.strip().split(" ")
    for word in content:
        word = word.lower()
        if word not in stop_word:
            a.append(word)

    # content = " ".join(content)
    return a


def readData(filePath):
    """

    :param filePath:
    :return:
    """
    labels, s = read_file(filePath)
    contents = []
    for i in range(len(labels)):
        content = cleanReview(s[i])
        # content = content.strip().split(" ")
        contents.append(content)
    return labels, contents


def compute_file(filename):
    label = []
    labels, contents = readData(filename)
    class_df_list = np.zeros(4)
    j = 0
    PATH_NAME = "./new_train"
    shutil.rmtree(PATH_NAME)  # 将整个文件夹删除
    os.makedirs(PATH_NAME)  # 创建一个文件夹
    for i in labels:
        word = []
        if '1' in labels[j]:
            label.append(1)
            class_df_list[0] += 1
            word = contents[j]
            with open('new_train/1.txt', 'a+', encoding='utf-8', errors='ignore') as f:
                f.write(str(word))
                f.write('\n')
        elif '2' in labels[j]:
            label.append(2)
            class_df_list[1] += 1
            word = contents[j]
            with open('new_train/2.txt', 'a+', encoding='utf-8', errors='ignore') as f:
                f.write(str(word))
                f.write('\n')
        elif '3' in labels[j]:
            label.append(3)
            class_df_list[2] += 1
            word = contents[j]
            with open('new_train/3.txt', 'a+', encoding='utf-8', errors='ignore') as f:
                f.write(str(word))
                f.write('\n')
        elif '4' in labels[j]:
            label.append(4)
            class_df_list[3] += 1
            word = contents[j]
            with open('new_train/4.txt', 'a+', encoding='utf-8', errors='ignore') as f:
                f.write(str(word))
                f.write('\n')
        j = j + 1
    return label, class_df_list, contents


# labels, contents = readData('./data/ag_news.txt')
# rate = 0.8
# trainIndex = int(len(labels) * rate)
# trainContents = contents[:trainIndex]
# trainLabels = labels[:trainIndex]
# evalContents = contents[trainIndex:]
# evalLabels = labels[trainIndex:]
# label, class_df_list, trainContents = compute_file(trainLabels, trainContents)
