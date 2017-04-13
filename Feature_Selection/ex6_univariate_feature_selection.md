##特徵选择/范例六: Univariate Feature Selection

http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html


此范例示范单变量特徵的选择。鸢尾花资料中会加入数个杂讯特徵(不具影响力的特徵资讯)并且选择单变量特徵。选择过程会画出每个特徵的 p-value 与其在支持向量机中的权重。可以从图表中看出主要影响力特徵的选择会选出具有主要影响力的特徵，并且这些特徵会在支持向量机有相当大的权重。
在本范例的所有特徵中，只有最前面的四个特徵是对目标有意义的。我们可以看到这些特徵的单变量特徵评分很高。而支持向量机会赋予最主要的权重到这些具影响力的特徵之一，但也会挑选剩下的特徵来做判断。在支持向量机增加权重之前就确定那些特徵较具有影响力，从而增加辨识率。

1. 资料集：鸢尾花
2. 特徵：萼片(sepal)之长与宽以及花瓣(petal)之长与宽
3. 预测目标：共有三种鸢尾花 setosa, versicolor, virginica
4. 机器学习方法：线性分类
5. 探讨重点：使用单变量选择(`SelectPercentile`)挑出训练特徵，与直接将所有训练特徵输入的分类器做比较
6. 关键函式： `sklearn.feature_selection.SelectPercentile`


### (一)修改原本的鸢尾花资料

用`datasets.load_iris()`读取鸢尾花的资料做为具有影响力的特徵，并以`np.random.uniform`建立二十个随机资料做为不具影响力的特徵，并合併做为训练样本。
```###############################################################################
# import some data to play with

# The iris dataset
iris = datasets.load_iris()

# Some noisy data not correlated
E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

# Add the noisy data to the informative features
X = np.hstack((iris.data, E))
y = iris.target
```

### (二)使用f-value作为判断的基准来找主要影响力特徵

以`SelectPercentile`作单变量特徵的计算，以F-test(`f_classif`)来做为选择的统计方式，挑选函式输出结果大于百分之十的特徵。并将计算出来的单便量特徵分数结果做正规化，以便比较每特徵在使用单变量计算与未使用单变量计算的差别。
```
###############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='g')
```
### (三)找出不计算单变量特徵的分类权重

以所有特徵资料，以线性核函数丢入支持向量分类机，找出各特徵的权重。
```
###############################################################################
# Compare to the weights of an SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()

plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='r')
```
### (四)找出以单变量特徵选出的分类权重

以单变量特徵选择选出的特徵，做为分类的训练特徵，差别在于训练的特徵资料是使用`selector.transform(X)`将`SelectPercentile`选择的结果读取出来，并算出以单变量特徵选择做预先选择后，该分类器的判断权重。
```
clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(X), y)

svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()

plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
        width=.2, label='SVM weights after selection', color='b')
```

### (五)原始码出处
Python source code: [plot_feature_selection.py](http://scikit-learn.org/stable/_downloads/plot_feature_selection.py)

```Python
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

###############################################################################
# import some data to play with

# The iris dataset
iris = datasets.load_iris()

# Some noisy data not correlated
E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

# Add the noisy data to the informative features
X = np.hstack((iris.data, E))
y = iris.target

###############################################################################
plt.figure(1)
plt.clf()

X_indices = np.arange(X.shape[-1])

###############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='g')

###############################################################################
# Compare to the weights of an SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()

plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='r')

clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(X), y)

svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()

plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
        width=.2, label='SVM weights after selection', color='b')


plt.title("Comparing feature selection")
plt.xlabel('Feature number')
plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')
plt.show()
```
