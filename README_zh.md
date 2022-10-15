Galaxy_VAE
=======

这个工作的目的是将无监督的提取星系图像形态特征，主要分为两个部分。

- **第一部分**

  首先从高维星系图像中提取隐变量，模型结构确定的情况下，隐变量维度是网络结构中唯一的超参。

  分析寻找一个好的隐变量维度后，得到许多星系图像的隐变量构成隐变量空间。

  探究隐变量空间的性质。从隐空间降维可视化、隐空间分类和离群点分析方面进行探究。
- **第二部分**

  方法：域适应

  原理：用域适应的方法将DECaLS巡天计划迁移到非DECaLS巡天计划，实现不同天区之间的应用。

  目的：DECaLS训练出来的模型用于提取其他巡天的形态特征，域适应训练对其他巡天进行降噪，使得第一部分训练的网络可以更好应用到其他巡天来提取形态特征。


## 环境配置要求
-  numpy
-  pandas
-  torch>=1.8.1
-  torchvision>=0.8.0
-  cudatoolkit>=0.7.1
-  scikit-learn>=1.0.2
-  matplotlib>=1.0.2
-  warnings
-  tqdm
-  astropy
-  pyarrow
-  datetime
-  random

# 第一部分 ``VAE``
## 数据
`DECaLS`: 根据此文章[论文](http://dx.doi.org/10.1093/mnras/stab2093)中的数据，对形状大小`3×256×256`筛选，而后用阈值进行选择。

用以下命令产生训练需要的`train.txt`
```
python make_datatxt.py
```
## 运行
### 训练
用以下命令直接训练`train.txt`中的星系图像。
```
python Galaxy_VAE_n.py
```
模型结构如下:

<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/outline/VAE_NN.png" width="600">

您可以通过单击访问它
[Github-models](https://github.com/xuquanfeng/Galaxy_VAE/blob/master/VAE_model/models.py)
.
### 分析隐变量
- `analysis/decals_1.py`: 星表中筛选合适阈值的星系图像。
- `analysis/decals_2_1.py`: 筛选的星系和DECaLSE中的星系图像进行配准，保存索引。
- `analysis/decals_2_2.py`: 根据上面的索引提出对应隐变量特征。
- `analysis/decals_3.py`: 降维可视化对应类别的隐变量特征。
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/outline/all.jpg" width="600">

- `analysis/decals_4.py`: 筛选离群点。
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/outline/lqd.jpg" width="600">

### 分析离群点

- `outline/visualization.ipynb`: 通过重构不同隐变量维度，查看模型效果。
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/outline/xiang.jpg" width="600">

- `outline/outline.ipynb`: 同一个星系通过不同巡天得到的星系图像，通过VAE得到的隐变量之间的距离，排序可视化。

由于不同巡天计划中观测波段、望远镜和PSF等因素影响，导致同一个星系的形态识别有差异：
* `相似`
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/transfer/in_out_like.jpg" width="800">

* `不相似`
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/transfer/in_out_dislike.jpg" width="800">

# 第二部分 ``域适应``

旨在克服上述在不同巡天计划中噪声影响的问题。
代码内容具体在`tansfer`文件夹中。
- `tansfer/dataset_resu.ipynb`: 两个不同巡天的星表进行配准筛选，将配准图像制作成数据集。
- `tansfer/no_tf.ipynb`: 得到未迁移的训练结果。
- `tansfer/transfer_learning.ipynb`: 得到迁移的训练结果。
