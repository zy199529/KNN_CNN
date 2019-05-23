import operator

from IG import *


def reduction_words_large():
    IG_word = feature_selection_ig()  # 返回每个词的信息熵
    file_path = "new_train"  # 分完类的文件夹
    each_word = []  # 存放降维后的词典
    # 该文件夹下宗共有10个文件夹，分别存储10大类的新闻数据
    floder_list = os.listdir(file_path)  # 读取文件夹下的每一个文件夹
    for i in range(len(floder_list)):
        word_list = []  # 每个类别的词典
        IG_word_value = []  # 每个类别的词语信息熵
        new_floder_path = file_path + '/' + floder_list[i]  # 每个类别下面的文件夹名称命名规范
        print(new_floder_path)
        with open(new_floder_path, encoding='utf-8', errors='ignore') as f:
            list1 = f.readlines()
            for fields in list1:
                fields = fields.strip()
                fields = fields.strip("[]")
                fields = fields.split(",")
                word_list.append(fields)
            word_list = createVocabList(word_list)  # 每个类别的词典
            print("词库大小", len(word_list))
            # IG_word = np.array(IG_word)
            for word in word_list:  # 遍历每个类词典的单词
                for k in range(len(IG_word)):  # 遍历之前的信息熵
                    if IG_word[k][0] in str(word):
                        IG_word_value.append(IG_word[k])  # 加入找到该单词，则返回词语和其信息熵
            IG_word_value = list(set(IG_word_value))  # 去重
            IG_word_value.sort(key=operator.itemgetter(1), reverse=True)  # 排序
            for x in range(300):  # 提取前1500个
                each_word.append(IG_word_value[x])  # 提取前1500个单词
                # print(each_word)
    each_word = np.array(list(set(each_word)))
    each_word = each_word[np.lexsort(each_word.T)]
    word = []
    for x in range(len(each_word)):
        word.append(each_word[x][0])
    return word  # 去重词语按信息熵从大到小后返回


if __name__ == '__main__':
    reduction_word = reduction_words_large()  # 降维后的特征集
    # print(reduction_words_large())
    with open('reduction_words.txt', 'w') as f:
        for i in range(len(reduction_word)):
            f.write(str(reduction_word[i]))
            f.write('\n')
