
# 通用范例/范例二: Concatenating multiple feature extraction methods

http://scikit-learn.org/stable/auto_examples/feature_stacker.html

在许多实际应用中，会有很多方法可以从一个数据集中提取特征。也常常会组合多个方法来获得良好的特征。这个例子说明如何使用` FeatureUnion` 来结合由` PCA` 和` univariate selection` 时的特征。

这个范例的主要目的：
1. 资料集：iris 鸢尾花资料集
2. 特征：鸢尾花特征
3. 预测目标：是那一种鸢尾花
4. 机器学习方法：SVM 支持向量机
5. 探讨重点：特征结合
6. 关键函式： `sklearn.pipeline.FeatureUnion`

# (一)资料汇入及描述

* 首先先汇入iris 鸢尾花资料集，使用from sklearn.datasets import load_iris将资料存入
* 准备X (特征资料) 以及 y (目标资料)


```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

iris = load_iris()

X, y = iris.data, iris.target
```

测试资料：<br />
`iris`为一个dict型别资料。

| 显示 | 说明 |
| -- | -- |
| ('target_names', (3L,))| 共有三种鸢尾花 setosa, versicolor, virginica |
| ('data', (150L, 4L)) | 有150笔资料，共四种特征 |
| ('target', (150L,))| 这150笔资料各是那一种鸢尾花|
| DESCR | 资料之描述 |
| feature_names| 4个特征代表的意义 |

# (二)PCA与SelectKBest
* `PCA(n_components = 主要成份数量)`:Principal Component Analysis(PCA)主成份分析，是一个常用的将资料维度减少的方法。它的原理是找出一个新的座标轴，将资料投影到该轴时，数据的变异量会最大。利用这个方式减少资料维度，又希望能保留住原数据点的特性。

* `SelectKBest(score_func , k )`: `score_func`是选择特征值所依据的函式，而`K`值则是设定要选出多少特征。


```python
# This dataset is way to high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)
```

# (三)FeatureUnionc

* 使用sklearn.pipeline.FeatureUnion合併主成分分析(PCA)和综合筛选(SelectKBest)。
* 最后得到选出的特征



```python
# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)
```

# (四)找到最佳的结果
* Scikit-learn的支持向量机分类函式库利用 SVC() 建立运算物件，之后并可以用运算物件内的方法 .fit() 与 .predict() 来做训练与预测。

* 使用`GridSearchCV`交叉验证，得到由参数网格计算出的分数网格，并找到分数网格中最佳点。最后显示这个点所代表的参数


```python
svm = SVC(kernel="linear")

# Do grid search over k, n_components and C:

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print(grid_search.best_estimator_)
```
结果显示
``` Fitting 3 folds for each of 18 candidates, totalling 54 fits
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=0.1
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=0.1, score=0.960784 -   0.0s
```


## (五)完整程式码
Python source code: feature_stacker.py
http://scikit-learn.org/stable/auto_examples/feature_stacker.html

```python
# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 clause

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

iris = load_iris()

X, y = iris.data, iris.target

# This dataset is way to high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)

svm = SVC(kernel="linear")

# Do grid search over k, n_components and C:

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print(grid_search.best_estimator_)
```
