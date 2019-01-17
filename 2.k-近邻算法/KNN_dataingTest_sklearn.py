import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
'''
def datatype(s):
    it = {"didntLike": 0, "smallDoses": 1, "largeDoses": 2}
    return it[s]
'''
def KNN():
    #导入数据
    filename = "E:\courseware\machine leaning\code\MLiA_SourceCode\machinelearninginaction\Ch02\datingTestSet.txt"
    #data = np.loadtxt(filename,dtype=float, delimiter=' ', converters={3: datatype} )
    data, labels = load_data(filename)
    #划分训练集和测试集
    x_train,x_test, y_train, y_test = train_test_split(data, labels,test_size=0.1)
    #数据归一化
    scalar = MinMaxScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.fit_transform(x_test)
    #构建模型
    list =[]
    for i in range(1,10):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(x_train, y_train)
        #accuracy = model.score(x_test,y_test)
        predict = model.predict(x_test)
        right = sum(predict == y_test)
        accuracy = right*1.0/predict.shape[0]
        list.append(accuracy)
        #predict = np.hstack((np.reshape(predict,(-1,1)), np.reshape(y_test,(-1,1))))
        #print(predict)
    #print("测试集的精确度是：%f%%" % (right*100.0/predict.shape[0]))
    min_accuracy = np.mean(list)
    print("平均精确度为: %f" % min_accuracy)
    plt.figure()
    plt.plot(list)
    plt.show()

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()  #按行读取文本中的数据，返回的是list形式
        length = len(lines)    #数据的个数
        data = np.zeros((length,3))
        labels = []
        dict = {}
        index = 0
        labelindex = 0
        for line in lines:
            line = line.strip()  #strip()移除字符串头和尾的字符，默认空格和换行
            atrlist = line.split('\t') #根据空格分割字符，得到三个特征列表
            data[index,:] = atrlist[0:3]
            key = atrlist[-1]
            if key in dict:
                labels.append(dict.get(key))
            else:
                dict[key] = labelindex
                labelindex += 1
                labels.append(dict.get(key))
            index += 1
        return data, labels


if __name__ == '__main__':
    KNN()