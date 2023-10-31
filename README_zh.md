Galaxy_VAE
=======

[English](README.md) | 简体中文

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
- `analysis/decals_select.py`: 星表中筛选合适阈值的星系图像。
- `analysis/decals_match.py`: 筛选的星系和DECaLSE中的星系图像进行配准，保存索引。
- `analysis/decals_take.py`: 根据上面的索引提出对应隐变量特征。
- `analysis/decals_visualization.py`: 降维可视化对应类别的隐变量特征。
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/outline/all.jpg" width="600">

- `analysis/decals_outliner.py`: 筛选离群点。
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
- `tansfer/dataset_result.ipynb`: 两个不同巡天的星表进行配准筛选，将配准图像制作成数据集。
- `tansfer/no_tf.ipynb`: 得到未迁移的训练结果。
- `tansfer/transfer_learn.ipynb`: 得到迁移的训练结果。
- `tansfer/select_overlap.py`: 通过阈值选出重叠天区的标签。
- `tansfer/RF_class_accuracy.ipynb`: 计算不同巡天计划下域迁移前后各个特征的精度。

下表为 VAE 域适应前后在不同巡天计划下各个特征的精度

| 问题           | 特征数 | 样本数 | Before DA<br/>DECaLS    | Before DA<br/>BASS+MzLS | After DA<br/>DECaLS      | After DA<br/>BASS+MzLS         |
|--------------|-----|-----|---------------|-----------|---------------|-----------------|
| How round    | 3   | 625 | 76.59\%       | 74.47 \%  | **82.45 \%**  | 77.66 \%        |
| Edge-on      | 2   | 1024 | 88.60 \%      | 87.62 \%  | **90.23 \%**  | 89.25 \%        |
| Bar          | 2   | 503 | **96.69 \%**  | 92.72 \%  | 96.03 \%      | 94.70 \%        |
| Have Arm     | 2   | 784 | 83.40 \%      | 80.43 \%  | 82.55 \%      | **84.68 \%**    |
| Arm Tightness | 3   | 225 | 61.76 \%      | 66.18 \%  | 67.65 \%      | **69.12 \%**    |
| Bulge Size   | 3   | 285 | 73.26 \%      | 70.93 \%  | **76.74 \%**  | 74.42 \%        |
| Merger       | 2   | 1135 | 97.95\%       | 97.65 \%  | 97.95 \%      | **98.53 \%**    |

## 引用

```
@article{10.1093/mnras/stad3181,
    author = {Xu, Quanfeng and Shen, Shiyin and de Souza, Rafael S and Chen, Mi and Ye, Renhao and She, Yumei and Chen, Zhu and Ishida, Emille E O and Krone-Martins, Alberto and Durgesh, Rupesh},
    title = "{From images to features: unbiased morphology classification via variational auto-encoders and domain adaptation}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {526},
    number = {4},
    pages = {6391-6400},
    year = {2023},
    month = {10},
    issn = {0035-8711},
    doi = {10.1093/mnras/stad3181},
    url = {https://doi.org/10.1093/mnras/stad3181},
}
```
