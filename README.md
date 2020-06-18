# PhyDNet - Disentangling Physical Dynamics from Unknown Factors for Unsupervised Video Prediction
[Vincent Le Guen](https://www.linkedin.com/in/vincentleguen/),  [Nicolas Thome](http://cedric.cnam.fr/~thomen/)

Code for our CVPR 2020 paper "Disentangling Physical Dynamics from Unknown Factors for Unsupervised Video Prediction": https://arxiv.org/abs/2003.01460

<img src="https://github.com/vincent-leguen/PhyDNet/blob/master/images/fig1.png" width="500">

## Abstract
Leveraging physical knowledge described by partial differential equations (PDEs) is an appealing way to improve unsupervised video prediction methods. Since physics is too restrictive for describing the full visual content of generic videos, we introduce PhyDNet, a two-branch deep architecture, which explicitly disentangles PDE dynamics from unknown complementary information. A second contribution is to propose a new  recurrent physical cell (PhyCell), inspired from data assimilation techniques, for performing PDE-constrained prediction in latent space. Extensive experiments conducted on four various datasets show the ability of PhyDNet to outperform state-of-the-art methods. Ablation studies also highlight the important gain brought out by both disentanglement and PDE-constrained prediction. Finally, we show that PhyDNet presents interesting features for dealing with  missing data and long-term forecasting.

## Code

In main.py, there is an example on how to run PhyDNet on the Moving MNIST dataset.

If you find this code useful for your research, please cite our [paper](https://papers.nips.cc/paper/8672-shape-and-time-distortion-loss-for-training-deep-time-series-forecasting-models):

```
@incollection{leguen20phydnet,
title = {Disentangling Physical Dynamics from Unknown Factors for Unsupervised Video Prediction},
author = {Le Guen, Vincent and Thome, Nicolas},
booktitle = {Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```
