from math import log
import operator

def calcShannonEnt(dataSet):
    """
    计算熵值
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    labelCounts = {}
    #（以下五行)为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # print(currentLabel)
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #print(labelCounts)
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        #print(prob)
        #print(prob * log(prob,2))
        # 以2为底求对数
        shannonEnt -= prob * log(prob,2)
    return shannonEnt


def createDataSet():
    """
    创建数据集
    :return:
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no1','no2','flippers']
    return dataSet, labels



dataset, labels = createDataSet()
#print(len(dataset))
#print(calcShannonEnt(dataset))

"""根据特征划分数据集"""
"""
>>> a=[1,2,3]
>>> b=[4,5,6]
>>> a.append(b)
>>> a
[1, 2, 3, [4, 5, 6]]

>>> a=[1,2,3]
>>> a.extend(b)
>>> a
[1, 2, 3, 4, 5, 6]
"""
def splitDataSet(dataSet, axis, value):
    """
      按照特征的下标和对应的值对数据进行划分
    :param dataSet:
    :param axis:
    :param value:
    :return: [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']]
    """
    #❶ 创建新的list对象
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #❷ （以下三行）抽取
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# print(splitDataSet(dataset, 1, 1))  # [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']]


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的特征向量，并返回相应的下标
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1
    # 基础熵值
    baseEntropy = calcShannonEnt(dataSet)
    # 初始化最好信息熵增和最好的特征列
    bestInfoGain = 0.0; bestFeature = -1

    for i in range(numFeatures):
        #（以下两行）创建唯一的分类标签列表
        # 提取特征列的所有值
        featList = [example[i] for example in dataSet]
        # 将特征列进行去重处理
        uniqueVals = set(featList)
        # 新的熵值
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            # 计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    # print("this is the best entropy")
    return bestFeature
# print(chooseBestFeatureToSplit(dataset))



def majorityCnt(classList):
    """
    返回类别最多的那一类
    :param classList:
    :return:
    """
    classCount={}
    # 遍历所有向量中的类别，对每个类别进行计数统计，返回一个字典类型的数据
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    # sortedClassCount = [(3, 6), (1, 3), (2, 3), (4, 3), (5, 1), (6, 1)] 将字典转化为数组，键值对用元组进行存储
    sortedClassCount=sorted(classCount.items(),
        key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
# print("majorityCnt test")

list = [1, 1, 2, 3, 4, 3,3,3,4,5, 2, 3, 6,3,4,2,1]
# <class 'tuple'>
# print(type(majorityCnt(list)[0]))

# print(majorityCnt(list))

def createTree(dataSet,labels):
    """
    生成决策树，返回决策树字典
    :param dataSet:
    :param labels:
    :return:
    """
    # 取每行数据的类别信息
    classList = [example[-1] for example in dataSet]
    #❶ （以下两行）类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #❷ （以下两行）遍历完所有特征时返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    print(bestFeat)
    print(len(labels))
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel:{}}
    #❸ 得到列表包含的所胡属性值
    # del删除的是变量，而不是数据。
    del(labels[bestFeat])
    # 得到最好的特征的值的列表
    featValues = [example[bestFeat] for example in dataSet]
    # 对数据进行去重然后遍历
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 创建一个labels的副本
        subLabels = labels[:]
        # 将最好的特征的每一个值作为一个key，放到myTree的bestFeatLabel键下
        myTree[bestFeatLabel][value] = createTree(splitDataSet
                           (dataSet, bestFeat, value),subLabels)
    return myTree

if __name__ == '__main__':
    myTree = createTree(dataset, labels)
    print(myTree)