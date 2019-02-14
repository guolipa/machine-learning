import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
'''
实例：使用Logistic Regression来预测患有疝病的马的存活问题
数据集：数据集包含368个样本，每个样本有28个特征，分为训练集和测试集
'''
def test():
    #导入训练集和测试集
    filename_train = "data\horseColicTraining.txt"
    filename_test = "data\horseColicTest.txt"
    data_train = np.loadtxt(filename_train, dtype = float, delimiter = '\t')
    X_train = data_train[:, : 21]
    y_train = data_train[:, 21]
    data_test = np.loadtxt(filename_test, dtype = float, delimiter = '\t')
    X_test = data_test[:, : 21]
    y_test = data_test[:, 21]
    #利用sklearn中的LogisticRegression进行训练测试
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    accuracy1 = clf.score(X_test, y_test)
    print("sklearn分类精确度为：%f" % accuracy1)
    #利用自己实现的基于随机梯度下降的Logistics Regression进行训练测试
    #比较不同迭代次数的分类精度[50:500:50]
    l = []
    for i in range(0, 550, 50):
        weights = SGD(X_train, y_train, maxLoop=i)
        accuracy2 = score(X_test, y_test, weights)
        l.append(accuracy2)
        print("%d 次迭代SGD分类精确度为：%f" % (i, accuracy2))
    plt.figure()
    x = range(0, 550, 50)
    plt.plot(x, l)
    plt.show()

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def SGD(X, y, maxLoop = 150):
    X = np.array(X)
    m, n = np.shape(X)
    weights = np.ones(n)
    alpha = 0.01
    for i in range(maxLoop):
        for j in range(m):
            h = sigmoid(sum(X[j] * weights))
            error = y[j] - h
            weights = weights + alpha * error * X[j]
    return weights

def predict(X, weights):
    h = sum(X * weights)
    if h >= 0.0:
        return 1
    else:
        return 0

def score(X, y, weights):
    X = np.array(X)
    m = X.shape[0]
    count = 0.0
    for i in range(m):
        predict_label = predict(X[i], weights)
        if predict_label == int(y[i]):
            count += 1
    accuracy = count / m
    return accuracy

if __name__ == '__main__':
    test()