#程序是基于电子邮件垃圾过滤的实例对朴素贝叶斯的实现

import re
import numpy as np

def spamTest():
    docList = []    #存储文本分割成的字符串列表
    classList = []  #存储每个邮件的类别
    #一共有50封邮件，25封垃圾邮件，25封正常邮件，将每封邮件转化成单次向量，添加到docList中，并将邮件类型添加到classList中
    for i in range(1,26):
        #file1 = "E:\courseware\machine leaning\code\MachineLearningInAction\Ch04\email\spam\%d.txt" % i
        stringOfText = open("E:\courseware\machine leaning\code\MachineLearningInAction\Ch04\email\spam\%d.txt" % i, 'r', encoding='gbk', errors='ignore').read()
        wordList = textParse(stringOfText)
        docList.append(wordList)
        classList.append(1)
        #file2 = "E:\courseware\machine leaning\code\MachineLearningInAction\Ch04\email\ham\%d.txt" % i
        stringOfText = open("E:\courseware\machine leaning\code\MachineLearningInAction\Ch04\email\ham\%d.txt" % i, 'r', encoding='gbk', errors='ignore' ).read()
        wordList = textParse(stringOfText)
        docList.append(wordList)
        classList.append(0)
    #利用docList获得词汇表向量
    vocabList = createVocabList(docList)
    #分割训练集和测试集，一共有50个样本，从中选出10个样本作为测试集
    trainIndex = list(range(50))
    testIndex = []
    for i in range(10):
        index = int(np.random.uniform(0, len(trainIndex))) #从0到len的均匀分布中随机选取一个整数
        testIndex.append(trainIndex[index])
        trainIndex.pop(index)
    trainSet = []
    trainClass = []
    #将训练索引中对应的文档向量添加到训练集
    for docIndex in trainIndex:
        trainSet.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    #用训练集训练分类器
    p0Vect, p1Vect, p1 = fit(np.array(trainSet), np.array(trainClass))
    #用测试集才评估训练的分类器
    errorCount = 0
    for docIndex in testIndex:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if predict(wordVector, p0Vect,p1Vect,p1) != classList[docIndex]:
            errorCount += 1
    errorRate = float(errorCount) / len(testIndex)
    print("The error rate is:%f" % errorRate)

def textParse(testString):
    '''将整个文本组成的字符串按word分割成字符串列表
    :param testString:
    :return:
    '''
    stringListOfText = re.split(r'\W*', testString)
    return [s.lower() for s in stringListOfText if len(s) > 2]

def createVocabList(dataSet):
    '''创建词汇表向量，里面包含出现在所有文本中的word
    :param dataSet:
    :return:
    '''
    vocabList = set([])
    for text in dataSet:
        vocabList = vocabList | set(text)  # | 的作用是集合求并集或者按位求或(or)的操作
    return list(vocabList)

def setOfWords2Vec(vocabList, wordSet):
    '''将一个文本的词向量转化成文档向量，向量每个元素为0或1，表示文本中的单词是否在词汇表中出现，eg[0,1,0,1,1,1]
    :param vocabList:
    :param wordSet:
    :return:
    '''
    resultVec = [0]*len(vocabList)
    for word in wordSet:
        if word in vocabList:
            resultVec[vocabList.index(word)] = 1
    return resultVec

def fit(trainSet,trainClass):
    '''朴素贝叶斯分类器的训练函数,以二分类为例
    :param trainSet:
    :param trainClass:
    :return:
    '''
    numDocs = len(trainClass)
    numWords = len(trainSet[0])
    p1 = (sum(trainClass)+1) / (float(numDocs)+2)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numDocs):
        if trainClass[i] == 1:
            p1Num += trainSet[i]
            p0Denom += sum(trainSet[i])
        else:
            p0Num += trainSet[i]
            p1Denom +=sum(trainSet[i])
    p0Vec = np.log(p0Num / p0Denom)
    p1Vec = np.log(p1Num / p1Denom)
    return p0Vec, p1Vec, p1

def predict(testVec, p0Vec, p1Vec, pClass1):
    '''
    :param testVec:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    '''
    p1 = sum(testVec * p1Vec) + np.log(pClass1)
    p0 = sum(testVec * p0Vec) + np.log(1.0 - pClass1)
    if p0 > p1:
        return 0
    else:
        return 1

if __name__ == '__main__':
    spamTest()