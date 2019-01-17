# K-近邻算法(K Nearest Neighbors Classification)  

## 1. KNN概述  

### 1.1 KNN算法原理  

基于近邻的的分类是一种基于实例的学习或者非泛化的学习，它不会训练一个通用的模型，只是存储训练数据的实例样本,每当有新样本时，找到该样本在训练集中一些最近邻居样本，新样本的类别从它的这些最近邻居的类别投票中得到(少数服从多数)。

**算法原理**：存在训练样本集，训练集中的每个样本都有自己标记/所属的分类。当输入没有标记的测试样本时，将测试样本的每个特征与训练集中每个样本对应特征进行比较，得到测试样本和训练集中每个样本的相似程度(距离值)，我们只选择样本集中前k个最相似的样本数据，这就是k-近邻中k的出处，通常k不大于20。最后选择k个最相似数据中出现次数最多的分类，作为新样本的类别。  

- 计算测试样本和训练集中每个样本的相似距离
- 按照距离将训练样本进行排序
- 选择前k个样本(一般情况 k<=20)
- 计算k个样本所属类别出现的频率
- 将出现频率最高的分类作为新样本的类别   

### 1.2 相似距离计算  

计算新样本和测试集中样本的相似距离的公式：  

![](https://i.loli.net/2019/01/15/5c3d9d5bc9fb5.png) 

### 1.3 K值的选择

The K-neighbors classification in KNeighborsClassifier is the most commonly used technique. The optimal choice of the value K is highly data-dependent: in general a larger K suppresses the effects of noise, but makes the classification boundaries less distinct. 

### 1.4 KNN特点

k-近邻算法是分类数据最简单最有效的算法，它是基于实例的学习，不需要训练模型。k-近邻算法必须保存全部数据集，如果训练数据集过大，不光使用大量的存储空间，而且计算量大耗时。

- 优点：精度高，对异常值不敏感，无数据输入假定
- 缺点：计算复杂度高，空间复杂度高

适用数据范围：数值型和标称型

# KNN在scikit-learn中的实现  

在scikit-learn中 实现了 k-近邻算法：```KNeighborsClassifier```  

```Class：sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=None, **kwargs)```

**参数**：  
 
- n_neighbors:邻居的个数，默认值为5   
  
- weights:权重函数
   	
	- uniform:相同的权重
	- distance：邻近样本和测试样本距离的倒数，距离近的邻近点的权重会更大  
	- [callable]:自己定义的计算权重函数，函数输入距离的的矩阵，返回相同形式的权重矩阵  

- algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional 计算K近邻的算法

	- ‘ball_tree’ ball树 
	- ‘kd_tree’   K-D树
	- ‘brute’ 	  暴力法
	- ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method  

- p：计算距离的方法

	-	p=1 曼哈度距离
	-	p=2 欧氏距离，默认值  


**方法：**  

- fit(X,y)

- predict(X)

- score(X,y,sample_weight=None)

```python  
from sklearn.neighbors import KNeighborsClassifier
model = KNeighboesClassifier()
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy = model.score(x_test, y_test)```   


tile()函数
归一化,标准化
Python 
	向量矩阵切片
	tile函数，numpy
	file open，readlines
	argsort
matplotlib 
	scatter()

