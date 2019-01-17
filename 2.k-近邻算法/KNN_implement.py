
from sklearn import datasets,neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#导入数据
iris = datasets.load_iris()
#取数据的前两个特征列
X = iris.data[:,:2]
y = iris.target
#绘制样本数据的分布图
plt.figure()
plt.scatter(X[:,0], X[:,1],c = y, s = 30)
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.title("KNN")
plt.show()

list = []
for i in range(1,50):
    #划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i/100.0)
    #构建KNN模型
    clf = neighbors.KNeighborsClassifier(n_neighbors = 15, weights = 'distance')
    clf.fit(X_train, y_train)
    #用模型测试测试集
    result = clf.predict(X_test)
    sum = 0.
    for i in range(len(result)):
        if result[i] == y_test[i]:
            sum +=1
    accuracy = sum / len(result)
    list.append(accuracy)

plt.figure()
plt.plot(list)
plt.xlabel("test_size")
plt.ylabel("accuracy")
plt.show()

