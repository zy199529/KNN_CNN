import operator

from TF_IDF import *


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def cosineDistance(instance1, instance2):
    dist1 = 1 - np.dot(instance1, instance2) / (np.linalg.norm(instance1) * np.linalg.norm(instance2) + 1)
    return dist1


def getNeighbors(trainingSet, testInstance, k, label):
    distances = []
    # length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = cosineDistance(testInstance, trainingSet[x])
        distances.append((label[x], dist, trainingSet[x]))
    distances.sort(key=operator.itemgetter(1))
    k_text = []
    neighbors = []
    d = []
    for x in range(k):
        one_hot = np.eye(4, dtype='int64')
        neighbors.append(one_hot[int(distances[x][0]) - 1])
        d.append(distances[x][1])
        k_text.append(distances[x][2])
    return neighbors, d, k_text


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccurcy(test_label, predictions):
    correct = 0
    for x in range(len(test_label)):
        if test_label[x] == predictions[x]:
            correct += 1
    return (correct / len(test_label)) * 100.0


def testbagOfWord2Vec(vocabList, inputSet):  # 词袋模型，统计概率的
    tmp = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            tmp[vocabList.index(word)] += 1  # 当前文档有这个词条，则根据词典位置获取其位置并赋值为1
    return tmp


def euclidean(instance1, instance2):  # 计算测试文本和每个训练集文本的距离
    distance = 0
    length = len(instance2)
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def KNN_class(filename):
    train_vec_List, idf_array, label = tf_idf_train()
    test_vec_List, test_label = tf_idf(filename)
    k = 16
    k_all = []
    for x in range(len(test_vec_List)):
        consine = []
        neighbors, d, k_text = getNeighbors(train_vec_List, test_vec_List[x], k, label)
        consine.append(d)
        print(consine)
        consine = np.mat(consine)
        print(np.array(consine).shape)
        y_test = np.mat(neighbors)
        print(np.array(y_test).shape)
        print(np.array(k_text).shape)
        k_1_2 = np.matmul(consine, y_test.astype(np.float64))
        k_2_2 = np.matmul(consine, np.mat(k_text).astype(np.float32))
        outputs_all = np.concatenate([k_1_2, k_2_2], 1)
        outputs_all = np.array(outputs_all)
        k_all.extend(outputs_all)
    return k_all


if __name__ == '__main__':
    # 测试文本，使用KNN分类
    print(KNN_class('./data/k_set.txt'))
