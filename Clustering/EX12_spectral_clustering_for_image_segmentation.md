# **范例十二:Spectral clustering for image segmentation**

http://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html

此范例是利用Spectral clustering来区别重叠的圆圈，将重叠的圆圈分为个体。

1. 建立一个100x100的影像包含四个不同半径的圆
2. 透过```np.indices```改变影像颜色複杂度
3. 用```spectral_clustering```区分出各个不同区域特徵


## (一)引入函式库
引入函式库如下：
1. ```numpy```:产生阵列数值
2. ```matplotlib.pyplot```:用来绘製影像
3. ```sklearn.feature_extraction import image```:将每个像素的梯度关系图像化
4. ```sklearn.cluster import spectral_clustering```:将影像正规化切割


```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering```


## (二)建立要被区分的重叠圆圈影像

* 产生一个大小为输入值得矩阵(此范例为100x100)，其内部值为沿著座标方向递增(如:0,1,...)的值。


```python
l = 100
x, y = np.indices((l, l))```

* 建立四个圆圈的圆心座标并给定座标值
* 给定四个圆圈的半径长度
* 将圆心座标与半径结合产生四个圆圈图像

```python
center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2
```
* 将上一段产生的四个圆圈影像合併为```img```使其成为一体的物件
* ```mask```为布林形式的```img```
* ```img```为浮点数形式的```img```
* 用乱数产生的方法将整张影像作乱数处理


```python
# 4 circles
img = circle1 + circle2 + circle3 + circle4
mask = img.astype(bool)
img = img.astype(float)

img += 1 + 0.2 * np.random.randn(*img.shape)```



接著将产生好的影像化为可使用```spectral_clustering```的影像

* ```image.img_to_graph``` 用来处理边缘的权重与每个像速间的梯度关联有关
* 用类似Voronoi Diagram演算法的概念来处理影像

```python
graph = image.img_to_graph(img, mask=mask)

graph.data = np.exp(-graph.data / graph.data.std())
```
最后用```spectral_clustering```将连在一起的部分切开，而```spectral_clustering```中的各项参数设定如下:
* ```graph```: 必须是一个矩阵且大小为nxn的形式
* ```n_clusters=4```: 需要提取出的群集数
* ```eigen_solver='arpack'```: 解特徵值的方式

开一张新影像```label_im```用来展示```spectral_clustering```切开后的分类结果

```python
labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
label_im = -np.ones(mask.shape)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)
```
![](http://scikit-learn.org/stable/_images/plot_segmentation_toy_001.png)
![](http://scikit-learn.org/stable/_images/plot_segmentation_toy_002.png)


## (三)完整程式码
Python source code:plot_segmentation_toy.py

http://scikit-learn.org/stable/_downloads/plot_segmentation_toy.py


```python
print(__doc__)

# Authors:  Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#           Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

###############################################################################
l = 100
x, y = np.indices((l, l))

center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

###############################################################################
# 4 circles
img = circle1 + circle2 + circle3 + circle4
mask = img.astype(bool)
img = img.astype(float)

img += 1 + 0.2 * np.random.randn(*img.shape)

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(img, mask=mask)

# Take a decreasing function of the gradient: we take it weakly
# dependent from the gradient the segmentation is close to a voronoi
graph.data = np.exp(-graph.data / graph.data.std())

# Force the solver to be arpack, since amg is numerically
# unstable on this example
labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
label_im = -np.ones(mask.shape)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)

###############################################################################
# 2 circles
img = circle1 + circle2
mask = img.astype(bool)
img = img.astype(float)

img += 1 + 0.2 * np.random.randn(*img.shape)

graph = image.img_to_graph(img, mask=mask)
graph.data = np.exp(-graph.data / graph.data.std())

labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
label_im = -np.ones(mask.shape)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)

plt.show()```
