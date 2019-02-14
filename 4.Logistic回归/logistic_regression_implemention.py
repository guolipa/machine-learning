import numpy as np
import math
import matplotlib.pyplot as plt
def LogisticRegression():
    #导入训练数据：这里使用的数据简单，样本只有两个特征，二分类，采用线性拟合
    #D:\software\Github\machine-learning\4.逻辑回归\
    filename = "testSet.txt"
    X, y = loadData(filename)
    #weight1 = BatchGradientAscent(X, y, 500)
    #weight2 = StochasticGradientDescent(X, y)
    weight3 = StochasticGradientDescent_a1(X, y)
    weight4 = StochasticGradientDescent_a2(X, y)
    #plotBestFit(X, y, weight1)
    #plotBestFit(X, y, weight2)
    plotBestFit(X, y, weight3)
    plotBestFit(X, y, weight4)

def loadData(filename):
    '''导入数据集
    :param filename: 数据文件路径
    :return:
        X：训练样本集
        y：训练样本的标记集
    '''
    X = []; y = []
    f = open(filename)
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        X.append([1.0, float(lineArr[0]), float(lineArr[1])])
        y.append(int(lineArr[2]))
    return X, y

def sigmoid(x):
    '''sigmoid函数
    :param x: 输入数值或者向量
    :return: 经过sigmoid函数运算的结果
    '''
    return 1.0 / (1 + np.exp(-x))

def BatchGradientAscent(X, y, maxLoop):
    '''批量梯度下降法
    :param X: 训练数据
    :param y: 训练数据标记
    :param maxLoop: 梯度下降的次数
    :return: 返回最佳分类边界线/拟合曲线的参数
    '''
    X = np.mat(X)
    y = np.mat(y).transpose()
    m, n = X.shape
    weights = np.ones((n, 1))
    alpha = 0.001
    for i in range(maxLoop):
        h = sigmoid(X * weights)
        error = (y - h)
        weights = weights + alpha * X.transpose() * error
    return weights

def StochasticGradientDescent(X, y):
    '''随机梯度下降法
    :param X:
    :param y:
    :return: 返回参数矩阵
    '''
    X = np.array(X)
    m, n = np.shape(X)
    weights = np.ones(n)
    alpha = 0.009
    for i in range(m):
        h = sigmoid(sum(X[i] * weights))
        error = (y[i] - h)
        weights = weights + alpha * error * X[i]
    return weights

def StochasticGradientDescent_a1(X, y):
    '''随机梯度下降法改进1：重复随机梯度下降200次,绘制各个参数随迭代次数的变化图
    :param X:
    :param y:
    :return: 返回参数矩阵
    '''
    X = np.array(X)
    m, n = np.shape(X)
    weights = np.ones(n)
    alpha = 0.01
    x0 = [1.0]; x1 = [1.0]; x2 = [1.0]
    for i in range(200):
        for j in range(m):
            h = sigmoid(sum(X[j] * weights))
            error = (y[j] - h)
            weights = weights + alpha * error * X[j]
            x0.append(weights[0])
            x1.append(weights[1])
            x2.append(weights[2])
    print("最终参数：w0:%f, w1:%f, w2:%f" % (weights[0], weights[1], weights[2]))
    fig = plt.figure()
    ax0 = fig.add_subplot(311)
    ax1 = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)
    x = range(0, 20001)
    ax0.plot(x, x0)
    #ax0.ylabel("w0")
    ax1.plot(x, x1)
    #ax1.ylabel("w1")
    ax2.plot(x, x2)
    #ax2.ylabel("w2")
    plt.xlabel("number of iteration")
    plt.show()
    return weights

def StochasticGradientDescent_a2(X, y):
    X = np.array(X)
    m, n = X.shape
    weights = np.ones(n)
    x0 = [1.0]; x1 = [1.0]; x2 = [1.0]
    for i in range(200):
        dataIndex = list(range(m))
        for j in range(m):
            index = int(np.random.uniform(0, len(dataIndex)))
            alpha = 4 / (1.0 + i + j) + 0.01
            h = sigmoid(sum(X[index] * weights))
            error = y[index] - h
            weights = weights + alpha * error * X[index]
            dataIndex.pop(index)
            x0.append(weights[0])
            x1.append(weights[1])
            x2.append(weights[2])
    fig = plt.figure()
    ax0 = fig.add_subplot(311)
    ax1 = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)
    x = range(0, 20001)
    ax0.plot(x, x0)
    # ax0.ylabel("w0")
    ax1.plot(x, x1)
    # ax1.ylabel("w1")
    ax2.plot(x, x2)
    # ax2.ylabel("w2")
    plt.xlabel("number of iteration")
    plt.show()
    return weights

def predict(X, weights):
    p = sum(X * weights)
    if p >= 0.0:
        return 1
    else:
        return 0

def score(X, y, weights):
    count = 0.0
    m = np.shape(X)[0]
    for i in range(m):
        if predict(X[i], weights) == y[i]:
            print(X[i])
            count +=1
    accuracy = count / m
    return accuracy
def plotBestFit(X, y, weights):
    '''绘制样本点和分类边界线
    :param X:训练数据
    :param y:训练数据的标记
    :param weights:求解得到的边界线的参数
    :return:
    '''
    X = np.array(X)
    #getA()的作用是将矩阵自身作为ndarray对象返回
    #weights = weights.getA()
    n = np.shape(X)[0]
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in range(n):
        if y[i] == 1:
            x1.append(X[i, 1])
            y1.append(X[i, 2])
        else:
            x2.append(X[i, 1])
            y2.append(X[i, 2])
    plt.figure()
    plt.scatter(x1, y1, s = 30, c = 'red')
    plt.scatter(x2, y2, s = 30, c = 'green')
    X_fit = np.arange(-3.0, 3.0, 0.1)
    y_fit = (-weights[0] - weights[1] * X_fit) / weights[2]
    plt.plot(X_fit, y_fit)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("The best fit of train set")
    plt.show()

if __name__ == '__main__':
    LogisticRegression()

