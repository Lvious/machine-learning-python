
# Datasets

## 机器学习资料集/ 范例三: The iris dataset


http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

这个范例目的是介绍机器学习范例资料集中的iris 鸢尾花资料集


## (一)引入函式库及内建手写数字资料库


```python
#这行是在ipython notebook的介面里专用，如果在其他介面则可以拿掉
%matplotlib inline

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()
# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

```

![png](ex3_fig1.png)


## (二)资料集介绍
`iris = datasets.load_iris()` 将一个dict型别资料存入iris，我们可以用下面程式码来观察里面资料


```python
for key,value in iris.items() :
    try:
        print (key,value.shape)
    except:
        print (key)
print(iris['feature_names'])
```

| 显示 | 说明 |
| -- | -- |
| ('target_names', (3L,))| 共有三种鸢尾花 setosa, versicolor, virginica |
| ('data', (150L, 4L)) | 有150笔资料，共四种特徵 |
| ('target', (150L,))| 这150笔资料各是那一种鸢尾花|
| DESCR | 资料之描述 |
| feature_names| 四个特徵代表的意义，分别为 萼片(sepal)之长与宽以及花瓣(petal)之长与宽

为了用视觉化方式呈现这个资料集，下面程式码首先使用PCA演算法将资料维度降低至3


```python
X_reduced = PCA(n_components=3).fit_transform(iris.data)
```

接下来将三个维度的资料立用`mpl_toolkits.mplot3d.Axes3D` 建立三维绘图空间，并利用 `scatter`以三个特徵资料数值当成座标绘入空间，并以三种iris之数值 Y，来指定资料点的颜色。我们可以看出三种iris中，有一种明显的可以与其他两种区别，而另外两种则无法明显区别。


```python
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
```


![png](ex3_fig2.png)



```python
#接著我们尝试将这个机器学习资料之描述档显示出来
print(iris['DESCR'])
```

    Iris Plants Database

    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
        :Summary Statistics:

        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
        ============== ==== ==== ======= ===== ====================

        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988

    This is a copy of UCI ML iris datasets.
    http://archive.ics.uci.edu/ml/datasets/Iris

    The famous Iris database, first used by Sir R.A Fisher

    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.

    References
    ----------
       - Fisher,R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ...



这个描述档说明了这个资料集是在 1936年时由Fisher建立，为图形识别领域之重要经典范例。共例用四种特徵来分类三种鸢尾花

## (三)应用范例介绍
在整个scikit-learn应用范例中，有以下几个范例是利用了这组iris资料集。

* 分类法 Classification
   * [EX 3: Plot classification probability](../Classification/ex3_Plot_classification_probability.md)
* 特徵选择 Feature Selection
   * [Ex 5: Test with permutations the significance of a classification score](../Feature_Selection/ex5_test_with_permutations_the_significance_of_a__.md)
   * [Ex 6: Univariate Feature Selection](../Feature_Selection/ex6_univariate_feature_selection.md)
* 通用范例 General Examples
   * [Ex 2: Concatenating multiple feature extraction methods](../general_examples/Ex2_feature_stacker.md)
