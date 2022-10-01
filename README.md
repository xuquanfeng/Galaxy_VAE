Galaxy_VAE
=======
English | [简体中文](README_zh.md)

The aim of this work is the unsupervised extraction of galaxy image morphological features, which is divided into two main parts.
- **Part I**

  Firstly, the latent variables are extracted from the high-dimensional galaxy images, and the latent variable dimensions are the only super parameters in the network structure when the model structure is determined.

  After analytically finding a good latent variable dimension, the latent variables of many galaxy images are obtained to constitute the latent variable space.

  Explore the nature of the latent variable space. Explore in terms of dimensionality reduction visualization of latent spaces, classification of latent spaces and outlier analysis.
- **Part II**

  Method: Domain adaptation

  Theory: The transfer of DECaLS survey to non-DECaLS survey using domain adaptation to enable applications between different sky survey.

  Goal: The model trained by DECaLS is used to extract morphological features from other surveys, and the domain adaptation training performs denoising on other surveys, so that the network trained in the first part can be better applied to other surveys to extract morphological features.


Requirements
============
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

Part I ``VAE``
========================
# Datasets
`DECaLS`: Based on the data in this article [(paper)](http://dx.doi.org/10.1093/mnras/stab2093), the shape size `3×256×256` was filtered, while the selection was made with a threshold value afterwards.

Generate the `train.txt` needed for training with the following command:
```
python make_datatxt.py
```
# Usage
## Training
Train the galaxy images in `train.txt` directly with the following command.
```
python Galaxy_VAE_n.py
```
The model structure is as follows:
<img src="http://cluster.shao.ac.cn/wiki/index.php?title=File:VAE_NN.png" width="600">

You can access it by clicking on 
[Github-models](https://github.com/xuquanfeng/Galaxy_VAE/blob/master/VAE_model/models.py)
.
## Analysis of latent variables
- `analysis/decals_1.py`: 星表中筛选合适阈值的星系图像。
- `analysis/decals_2_1.py`: 筛选的星系和DECaLSE中的星系图像进行配准，保存索引。
- `analysis/decals_2_2.py`: 根据上面的索引提出对应隐变量特征。
- `analysis/decals_3.py`: 降维可视化对应类别的隐变量特征。
- `analysis/decals_4.py`: 筛选离群点。
## Analysis of outliers
- `outliers/visualization.ipynb`: The model effects are viewed by reconstructing different hidden variable dimensions.
<img src="http://cluster.shao.ac.cn/wiki/index.php?title=File:VAE_NN.png" width="600">
- `outliers/outline.ipynb`: The same galaxy images obtained by different surveys are visualized by sorting the distances between the latent variables obtained by VAE.
<img src="http://cluster.shao.ac.cn/wiki/index.php?title=File:VAE_NN.png" width="600">

The morphological identification of the same galaxy varies due to the effects of observing bands, telescopes and PSFs in different sky survey programs.
- `like`
<img src="http://cluster.shao.ac.cn/wiki/index.php?title=File:VAE_NN.png" width="600">
- `dislike`
<img src="http://cluster.shao.ac.cn/wiki/index.php?title=File:VAE_NN.png" width="600">

Part II ``Domain Adaptation``
====================

The aim is to overcome the above problem of noise effects in different sky surveys.
The code content is specified in the `tansfer` folder.
- `tansfer/dataset_resu.ipynb`: The catalogs of two different surveys were aligned and filtered to produce a dataset of aligned images.
- `tansfer/no_tf.ipynb`: The un-transfer training results are obtained.
- `tansfer/ts_resu2.ipynb`: The transfer training results are obtained.
