# 特徵选择/范例一: Pipeline Anova SVM

http://scikit-learn.org/stable/auto_examples/feature_selection/feature_selection_pipeline.html

此机器学习范例示范伫列的使用，依照顺序执行ANOVA挑选主要特徵，并且使用C-SVM来计算特徵的权重与预测。

1. 使用 `make_classification` 建立模拟资料
2. 使用 `SelectKBest` 设定要用哪种目标函式，以挑出可提供信息的特徵
3. 使用 `SVC` 设定支持向量机为分类计算以及其核函数
4. 用 `make_pipeline` 合併 SelectKBest物件 与 SVC物件
5. 用 `fit` 做训练，并且以 `predict` 来做预测


---
### (一)建立模拟资料

在选择特徵之前需要有整理好的特徵与目标资料。在此范例中，将以`make_classification`功能建立特徵与目标。该功能可以依照使用者想模拟的情况，建立含有不同特性的模拟资料，像是总特徵数目，其中有几项特徵含有目标资讯性、目标聚集的程度、目标分为几类等等的特性。


```python
# import some data to play with
X, y = samples_generator.make_classification(
    n_features=20, n_informative=3, n_redundant=0, n_classes=4,
    n_clusters_per_class=2)
```
在本范例，我们将X建立为一个有20个特徵的资料，其中有3种特徵具有目标资讯性，0个特徵是由目标资讯性特徵所产生的线性组合，目标分为4类，而每个分类的目标分布为2个群集。


### (二)选择最好的特徵

在机器学习的训练之前，可以藉由统计或指定评分函数，算出特徵与目标之间的关系，并挑选出最具有关系的特徵作为训练的素材，而不直接使用所有特徵做为训练的素材。

其中一种方法是统计特徵与目标之间的F-score做为评估分数，再挑选F-score最高的几个特徵作为训练素材。我们可以用 `SelectKBest()` 来建立该功能的运算物件。

```python
# ANOVA SVM-C
# 1) anova filter, take 3 best ranked features
anova_filter = SelectKBest(f_regression, k=3)
```
`SelectKBest()`的第一项参数须给定评分函数，在本范例是设定为`f_regression` 。第二项参数代表选择评估分数最高的3个特徵做为训练的素材。建立完成后，即可用物件内的方法`.fit_transform(X,y)` 来提取被选出来的特徵。

### (三)以伫列方式来设定支持向量机分类法运算物件

Scikit-lenarn的支持向量机分类涵式库提供使用简单易懂的指令，只要用 `SVC()` 建立运算物件后，便可以用运算物件内的方法 `.fit()` 与 `.predict()` 来做训练与预测。

本范例在建立运算物件后，不直接用`SelectKBest().fit_transform()` 提出训练素材。而是以 `make_pipeline()`合併先前设定好的两个运算物件。再执行`.fit()` 与 `.predict()`来完成训练与预测的动作。

```python
# 2) svm
clf = svm.SVC(kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X, y)
anova_svm.predict(X)
```
当我们以伫列建立好的运算物件，就可以直接给定所有的特徵资料与目标资料做训练与预测。在训练过程中，会依照给定的特徵素材数目从特徵资料中挑出特徵素材。预测时，也会从预测资料中挑出对应特徵素材的资料来做预测判断。

若是将`SelectKBest()`与 `SVC()`物件分开来执行，当 `SVC()`物件在做学习时给定的特徵即为被选出来的特徵素材数目。那预测的时候也必须从预测资料中，挑出被`SelectKBest()`选出来的特徵来给`SVC()`做预测。

---

## (四)原始码

### Python source code: [feature_selection_pipeline.py](http://scikit-learn.org/stable/_downloads/feature_selection_pipeline.py)

```python
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

# import some data to play with
X, y = samples_generator.make_classification(
    n_features=20, n_informative=3, n_redundant=0, n_classes=4,
    n_clusters_per_class=2)

# ANOVA SVM-C
# 1) anova filter, take 3 best ranked features
anova_filter = SelectKBest(f_regression, k=3)
# 2) svm
clf = svm.SVC(kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X, y)
anova_svm.predict(X)
```

## (五)函式用法
###[`make_classification()`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) 的参数



```Python
sklearn.datasets.make_classification(   n_samples=100,
                                        n_features=20,
                                        n_informative=2,
                                        n_redundant=2,
                                        n_repeated=0,
                                        n_classes=2,
                                        n_clusters_per_class=2,    
                                        weights=None,
                                        flip_y=0.01,
                                        class_sep=1.0,
                                        hypercube=True,
                                        shift=0.0,
                                        scale=1.0,
                                        shuffle=True,
                                        random_state=None)
```

参数:
* n_samples :
* n_fratures : 总特徵数目
* n_informative: 有意义的特徵数目
* n_redundant : 产生有意义特徵的随机线性组合
* n_repeated
* n_classes: 共分类为几类
* n_clusters_per_class: 一个类群有几个群组分布
* weights :
* flip_y :
* class_sep :
* hypercube :
* shift :
* scale :
* shuffle :
* random_state :

输出:
* X : 特徵矩阵资料
* Y : 对应目标资料

类似的功能:

[make_blobs](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs)

[make_gaussian_quantiles](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html#sklearn.datasets.make_gaussian_quantiles)


---

###[`SelectKBest()`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) 的参数


SelectKBest 的使用:
* 选择最好的特徵(目标函式, 特徵个数)
* 目标函式:  测试X与Y之间关系，须提供F score与p-value
* 特徵个数: 最好的特徵个数

f_regression 的使用：

* f_regression(X,y)
* 输入X与y
* 输出F score与p-value
