
import re
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split

def naive_bayesian():
    #加载数据并划分训练集和测试集
    docList = []    #存储文本分割成的字符串列表
    classList = []  #存储每个邮件的类别
    #一共有50封邮件，25封垃圾邮件，25封正常邮件，将每封邮件转化成单次向量，添加到docList中，并将邮件类型添加到classList中
    for i in range(1,26):
        #file1 = "E:\courseware\machine leaning\code\MachineLearningInAction\Ch04\email\spam\%d.txt" % i
        stringOfText = open("E:\courseware\machine leaning\code\MachineLearningInAction\Ch04\email\spam\%d.txt" % i,encoding='gbk', errors='ignore').read()
        wordList = textParse(stringOfText)
        docList.append(wordList)
        classList.append(1)
        #file2 = "E:\courseware\machine leaning\code\MachineLearningInAction\Ch04\email\ham\%d.txt" % i
        stringOfText = open("E:\courseware\machine leaning\code\MachineLearningInAction\Ch04\email\ham\%d.txt" % i, encoding='gbk', errors='ignore' ).read()
        wordList = textParse(stringOfText)
        docList.append(wordList)
        classList.append(0)
    #利用docList获得词汇表向量
    vocabList = createVocabList(docList)
    dataList = []
    for docIndex in range(50):
        dataList.append(setOfWords2Vec(vocabList, docList[docIndex]))
    x_train, x_test, y_train, y_test = train_test_split(dataList, classList)
    model = naive_bayes.GaussianNB()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test,y_test)
    print("精确度为：%f%%" % (accuracy*100))

#将整个文本组成的字符串按word分割成字符串列表
def textParse(testString):
    stringListOfText = re.split(r'\W*', testString)
    return [s.lower() for s in stringListOfText if len(s) > 2]

#创建词汇表向量，里面包含出现在所有文本中的word
def createVocabList(dataSet):
    vocabList = set([])
    for text in dataSet:
        vocabList = vocabList | set(text)  # | 的作用是集合求并集或者按位求或(or)的操作
    return list(vocabList)

#将一个文本的词向量转化成文档向量，向量每个元素为0或1，表示文本中的单词是否在词汇表中出现，eg[0,1,0,1,1,1]
def setOfWords2Vec(vocabList, wordSet):
    resultVec = [0]*len(vocabList)
    for word in wordSet:
        if word in vocabList:
            resultVec[vocabList.index(word)] = 1
    return resultVec

if __name__ == '__main__':
    naive_bayesian()
