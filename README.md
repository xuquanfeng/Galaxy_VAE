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


## Requirements
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

# Part I ``VAE``
## Datasets
`DECaLS`: Based on the data in this article [(paper)](https://doi.org/10.1093/mnras/stab2093), the shape size `3×256×256` was filtered, while the selection was made with a threshold value afterwards.

Generate the `train.txt` needed for training with the following command:
```
python make_datatxt.py
```
## Usage
### Training
Train the galaxy images in `train.txt` directly with the following command.
```
python Galaxy_VAE_n.py
```
The model structure is as follows:
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/outline/VAE_NN.png" width="600">

You can access it by clicking on 
[Github-models](https://github.com/xuquanfeng/Galaxy_VAE/blob/master/VAE_model/models.py)
.
### Analysis of latent variables
- `analysis/decals_select.py`: The catalog filters images of galaxies with appropriate thresholds.
- `analysis/decals_match.py`: The filtered galaxies are aligned with the galaxy images in DECaLSE and the index is saved.
- `analysis/decals_take.py`: The corresponding latent variable features are proposed according to the above index.
- `analysis/decals_visualization.py`: Reduced dimensional visualization of the latent variable features of the corresponding category.
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/outline/all.jpg" width="600">

- `analysis/decals_outliner.py`: Screening outlier points.
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/outline/lqd.jpg" width="600">

### Analysis of outliers

- `outliers/visualization.ipynb`: The model effects are viewed by reconstructing different hidden variable dimensions.
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/outline/xiang.jpg" width="600">

- `outliers/outline.ipynb`: The same galaxy images obtained by different surveys are visualized by sorting the distances between the latent variables obtained by VAE.

The morphological identification of the same galaxy varies due to the effects of observing bands, telescopes and PSFs in different sky survey programs.
- `like`
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/transfer/in_out_like.jpg" width="800">

- `dislike`
<img src="https://github.com/xuquanfeng/Galaxy_VAE/blob/master/transfer/in_out_dislike.jpg" width="800">

# Part II ``Domain Adaptation``

The aim is to overcome the above problem of noise effects in different sky surveys.
The code content is specified in the `tansfer` folder.
- `tansfer/dataset_result.ipynb`: The catalogs of two different surveys were aligned and filtered to produce a dataset of aligned images.
- `tansfer/no_tf.ipynb`: The un-transfer training results are obtained.
- `tansfer/transfer_learn.ipynb`: The transfer training results are obtained.
- `tansfer/select_overlap.py`: Obtain labels for overlapping sky areas through thresholds.
- `tansfer/RF_class_accuracy.ipynb`: Calculate the accuracy of each feature before and after domain adaption under different sky survey.

The following table shows the accuracy of each feature before and after the VAE domain adaptation under different sky survey:

| Question        | Num. of<br/>types | Total<br/>Count | Before DA<br/>DECaLS    | Before DA<br/>BASS+MzLS | After DA<br/>DECaLS      | After DA<br/>BASS+MzLS         |
|-----------------|---------|-------|---------------|-----------|---------------|-----------------|
| How round       | 3       | 625   | 76.59\%       | 74.47 \%  | **82.45 \%**  | 77.66 \%        |
| Edge-on         | 2       | 1024  | 88.60 \%      | 87.62 \%  | **90.23 \%**  | 89.25 \%        |
| Bar             | 2       | 503   | **96.69 \%**  | 92.72 \%  | 96.03 \%      | 94.70 \%        |
| Have Arm        | 2       | 784   | 83.40 \%      | 80.43 \%  | 82.55 \%      | **84.68 \%**    |
| Arm Tightness   | 3       | 225   | 61.76 \%      | 66.18 \%  | 67.65 \%      | **69.12 \%**    |
| Bulge Size      | 3       | 285   | 73.26 \%      | 70.93 \%  | **76.74 \%**  | 74.42 \%        |
| Merger          | 2       | 1135  | 97.95\%       | 97.65 \%  | 97.95 \%      | **98.53 \%**    |

# Citation

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
