#机器学习：使用Python

这份文件的目的是要提供Python 之机器学习套件 scikit-learn (http://scikit-learn.org/) 的中文使用说明。一开始的主要目标是详细说明scikit-learn套件中的[范例程式](http://scikit-learn.org/stable/auto_examples/index.html )的使用流程以及相关函式的使用方法。目前使用版本为 scikit-learn version 0.17 以上


本书原始资料在 Github 上公开，欢迎大家共同参与维护： [https://github.com/htygithub/machine-learning-python](https://github.com/htygithub/machine-learning-python)。

## 本文件主要的版本发展
* 0.0: 2015/12/21
    * 开始本文件「机器学习：使用Python」的撰写
    * 初期以scikit-learn套件的范例介绍为主轴
* 0.1: 2016/4/15
    * 「机器学习：使用Python」文件
    *  Contributor: 陈巧宁、曾裕胜、黄腾毅 、蔡奕甫
    *  新增章节: Classification, Clustering, cross_decomposition, Datasets, feature_selection, general_examples
    *  新增 introduction: 说明简易的Anaconda安装，以及利用数字辨识范例来入门机器学习的方法
    *  第 10,000个 pageview 达成
![](images/pg10000.PNG)
* 0.2: 2016/8/30
    *  新增应用章节，Contributor: 吴尚真
    *  增修章节: Classification, Datasets, feature_selection, general_examples
* 0.3: 2017/2/16
    *  新增应用章节，Contributor: 杨采玲、欧育年
    *  增修章节: Neural_Network, Decision tree
    *  2016年，使用者约四万人次，页面流量约15万次。
![](images/2016year.PNG)
##  Scikit-learn 套件

Scikit-learn (http://scikit-learn.org/) 是一个机器学习领域的开源套件。整个专案起始于 2007年由David Cournapeau所执行的`Google Summer of Code` 计画。而2010年之后，则由法国国家资讯暨自动化研究院（INRIA, http://www.inria.fr） 继续主导及后续的支援及开发。近几年(2013-2015)则由 INRIA 支持 Olivier Grisel (http://ogrisel.com) 全职负责该套件的维护工作。以开发者的角度来观察，会发现Scikit-learn的整套使用逻辑设计的极其简单。往往能将繁杂的机器学习理论简化到一个步骤完成。Python的机器学习相关套件相当多，为何Scikit-learn会是首选之一呢？其实一个开源套件的选择，最简易的指标就是其`contributor: 贡献者` 、 `commits:版本数量` 以及最新的更新日期。下图是2016/1/3 经过了美好的跨年夜后，笔者于官方开源程式码网站(https://github.com/scikit-learn/scikit-learn) 所撷取的画面。我们可以发现最新`commit`是四小时前，且`contributor`及`commit`数量分别为531人及 20,331个。由此可知，至少在2016年，这个专案乃然非常积极的在运作。在众多机器学习套件中，不论是贡献者及版本数量皆是最庞大的。也因此是本文件介绍机器学习的切入点。未来，我们希望能介绍更多的机器学习套件以及理论，也欢迎有志之士共同参与维护。

![](images/sklearn_intro.PNG)
