# Stochastic-Quantization
## Introduction
This repository contains the codes for training and testing <b>Stocastic Quantization</b> described in the paper "[Learning Accurate Low-bit Deep Neural Networks with Stochastic Quantization](https://arxiv.org/abs/1708.01001)" (BMVC 2017, Oral).

We implement our codes based on [Caffe](https://github.com/BVLC/caffe) framework. Our codes can be used for training BWN (Binary Weighted Networks), TWN (Ternary Weighted Networks), SQ-BWN and SQ-TWN.

## Usage
### Build Caffe
Please follow the standard [installation](http://caffe.berkeleyvision.org/installation.html) of Caffe.

```shell
cd caffe/
make
cd ..
```

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

## Implementation

### Layers
We add [BinaryConvolution](caffe/src/caffe/layers/conv_layer_binary.cu), [BinaryInnerProduct](caffe/src/caffe/layers/inner_product_layer_binary.cu), [TernaryConvolution](caffe/src/caffe/layers/conv_layer_ternary.cu) and [TernaryInnerProduct](caffe/src/caffe/layers/inner_product_layer_ternary.cu) layers to train binary or ternary networks. We also put useful functions of low-bits DNNs in [lowbit-functions](caffe/src/caffe/util/lowbit_functions.cu).

### Params
We add two more parameters in `convolution_param` and `inner_product_param`, which are `sq` and `ratio`. `sq` means whether to use stochastic quantization (default to `false`). `ratio` is the SQ ratio (default to 100).

### Note
Our codes can only run appropriately on GPU. CPU version should be further implemented.

Have fun to deploy your own low-bits DNNs!
