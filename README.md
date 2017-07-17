# Stochastic-Quantization
## Introduction
This repository contains the codes for training and testing <b>Stocastic Quantization</b> described in the paper "Learning Accurate Low-bit Deep Neural Networks with Stochastic Quantization" (BMVC 2017, Oral).

We implement our codes based on [Caffe](https://github.com/BVLC/caffe) framework. Our codes can be used for training BWN (Binary Weighted Networks), TWN (Ternary Weighted Networks), SQ-BWN and SQ-TWN.

## Usage
### Build Caffe
Please follow the standard [installation](http://caffe.berkeleyvision.org/installation.html) of Caffe.

```shell
cd caffe/
make
```
We add the BinaryConvolution, BinaryInnerProduct, TernaryConvolution and TernaryInnerProduct layers.

### Training and Testing
#### CIFAR
For CIFAR-10(100), we provide two network architectures VGG-9 and ResNet-56 (See details in the paper). For example, use the following commands to train ResNet-56:

* FWN

```shell
./CIFAR/ResNet-56/FWN/train.sh
```
* BWN

```shell
./CIFAR/ResNet-56/BWN/train.sh
```
* TWN

```shell
./CIFAR/ResNet-56/TWN/train.sh
```
* SQ-BWN

```shell
./CIFAR/ResNet-56/SQ-BWN/train.sh
```
* SQ-TWN

```shell
./CIFAR/ResNet-56/SQ-TWN/train.sh
```

#### ImageNet
For ImageNet, we provide AlexNet-BN and ResNet-18 network architectures. For example, use the following commands to train ResNet-18:

* FWN

```shell
./ImageNet/ResNet-18/FWN/train.sh
```
* BWN

```shell
./ImageNet/ResNet-18/BWN/train.sh
```
* TWN

```shell
./ImageNet/ResNet-18/TWN/train.sh
```
* SQ-BWN

```shell
./ImageNet/ResNet-18/SQ-BWN/train.sh
```
* SQ-TWN

```shell
./ImageNet/ResNet-18/SQ-TWN/train.sh
```

### Note
Our codes can only run appropriately on GPU. CPU version should be further implemented.