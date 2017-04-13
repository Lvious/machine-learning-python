
# Datasets

## 机器学习资料集/ 范例一: The digits dataset


http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html

这个范例目的是介绍机器学习范例资料集的操作，对于初学者以及授课特别适合使用。

## (一)引入函式库及内建手写数字资料库


```python
#这行是在ipython notebook的介面里专用，如果在其他介面则可以拿掉
%matplotlib inline
from sklearn import datasets

import matplotlib.pyplot as plt

#载入数字资料集
digits = datasets.load_digits()

#画出第一个图片
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
```


![png](ex1_fig1.png)


## (二)资料集介绍
`digits = datasets.load_digits()` 将一个dict型别资料存入digits，我们可以用下面程式码来观察里面资料


```python
for key,value in digits.items() :
    try:
        print (key,value.shape)
    except:
        print (key)

```

    ('images', (1797L, 8L, 8L))
    ('data', (1797L, 64L))
    ('target_names', (10L,))
    DESCR
    ('target', (1797L,))


| 显示 | 说明 |
| -- | -- |
| ('images', (1797L, 8L, 8L))| 共有 1797 张影像，影像大小为 8x8 |
| ('data', (1797L, 64L)) | data 则是将8x8的矩阵摊平成64个元素之一维向量 |
| ('target_names', (10L,)) | 说明10种分类之对应 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |
| DESCR | 资料之描述 |
| ('target', (1797L,))| 记录1797张影像各自代表那一个数字 |

接下来我们试著以下面指令来观察资料档，每张影像所对照的实际数字存在`digits.target`变数中


```python
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
```


![png](ex1_fig2.png)



```python
#接著我们尝试将这个机器学习资料之描述档显示出来
print(digits['DESCR'])
```

    Optical Recognition of Handwritten Digits Data Set
    ===================================================

    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 5620
        :Number of Attributes: 64
        :Attribute Information: 8x8 image of integer pixels in the range 0..16.
        :Missing Attribute Values: None
        :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
        :Date: July; 1998

    This is a copy of the test set of the UCI ML hand-written digits datasets
    http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

    The data set contains images of hand-written digits: 10 classes where
    each class refers to a digit.

    Preprocessing programs made available by NIST were used to extract
    normalized bitmaps of handwritten digits from a preprinted form. From a
    total of 43 people, 30 contributed to the training set and different 13
    to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of
    4x4 and the number of on pixels are counted in each block. This generates
    an input matrix of 8x8 where each element is an integer in the range
    0..16. This reduces dimensionality and gives invariance to small
    distortions.

    For info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.
    T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.
    L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,
    1994.

    References
    ----------
      - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
        Applications to Handwritten Digit Recognition, MSc Thesis, Institute of
        Graduate Studies in Science and Engineering, Bogazici University.
      - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.
      - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.
        Linear dimensionalityreduction using relevance weighted LDA. School of
        Electrical and Electronic Engineering Nanyang Technological University.
        2005.
      - Claudio Gentile. A New Approximate Maximal Margin Classification
        Algorithm. NIPS. 2000.



这个描述档说明了这个资料集是在 1998年时建立的，由`E. Alpaydin, C. Kaynak ，Department of Computer Engineering
Bogazici University, Istanbul Turkey ` 建立的。数字的笔迹总共来自43个人，一开始取像时为32x32的点阵影像，之后经运算处理形成 8x8影像，其中灰阶记录的范围则为 0~16的整数。

## (三)应用范例介绍
在整个scikit-learn应用范例中，有以下几个范例是利用了这组手写辨识资料集。这个资料集的使用最适合机器学习初学者来理解分类法的原理以及其进阶应用

 * [分类法 Classification](../Classification/Classification.md)
   * [Ex 1: Recognizing hand-written digits](../Classification/ex1_Recognizing_hand-written_digits.md)
 * [特徵选择 Feature Selection](../Feature_Selection/intro.md)
   * [Ex 2: Recursive Feature Elimination](../Feature_Selection/ex2_Recursive_feature_elimination.md)
   * [Ex 3: Recursive Feature Elimination with Cross-Validation](../Feature_Selection/ex3_rfe_crossvalidation__md.md)


```python

```
