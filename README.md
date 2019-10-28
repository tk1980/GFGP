# Global Feature Guided Local Pooling

The Pytorch implementation for the ICCV2019 paper of "[Global Feature Guided Local Pooling](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/ICCV2019.pdf)" by [Takumi Kobayashi](https://staff.aist.go.jp/takumi.kobayashi/).

### Citation

If you find our project useful in your research, please cite it as follows:

```
@inproceedings{kobayashi2019iccv,
  title={Global Feature Guided Local Pooling},
  author={Takumi Kobayashi},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  pages={3365-3374},
  year={2019}
}
```

## Contents

1. [Introduction](#introduction)
2. [Install](#install)
3. [Usage](#usage)
4. [Results](#results)

## Introduction

This work provides two contributions.
One is the parametric pooling function, maximum-entropy pooling, which is theoretically derived from the maximum entropy principle.
The other is the novel module, global feature guided pooling (GFGP), to estimate the pooling parameters by means of the global features extracted from the feature map.
By simply replacing the existing pooling layer with the proposed one (Max-Ent pooling + GFGP), we can enjoy performance improvement.
For the more detail, please refer to our [paper](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/ICCV2019.pdf).

<img width=400 src="https://user-images.githubusercontent.com/53114307/67689158-3cd29380-f9de-11e9-9da3-408d6115ff1c.png">

Figure: GFGP Architecture

## Install

### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch(>=1.0.0)](http://pytorch.org)

### Compile
Compile the maximum-entropy pooling module as follows.
```python
cd models/modules/maxentpool_cuda
python setup.py build
cp build/lib.linux-<LINUX_ARCH>-<PYTHON_VER>/* build/
```

## Usage

### Training
The layer of GFGP + Max-Ent pooling is simply incorporated as in the other pooling layer by

```python
from modules.mylayers import MaxEntPoolingCuda2d
pool = MaxEntPoolingCuda2d(num_features=num_features, kernel_size=kernel_size, stride=stride, padding=padding)
```

For example, the ResNet-50 with the GFGP+Max-Ent pooling on ImageNet is also trained by

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_train.py  --dataset imagenet  --data ./datasets/imagenet12/images/  --arch resnet50 --pool max_ent  --config-name imagenet  --out-dir ./results/imagenet/resnet50/max_ent/  --dist-url 'tcp://127.0.0.1:8080'  --dist-backend 'nccl'  --multiprocessing-distributed  --world-size 1  --rank 0
```

Note that the ImageNet dataset must be downloaded at `./datasets/imagenet12/` before the training.

## Results
These performance results are not the same as those reported in the paper because the methods were implemented by MatConvNet in the paper and accordingly trained in a (slightly) different training procedure.

#### ImageNet

| Network  | Pooling | Top-1 Err. |
|---|---|---|
| VGG-16 mod [1]|  Max | 22.99 |
| VGG-16 mod [1]|  Max-Ent. + GFGP | 22.64 |
| VGG-16 [2]|  Max | 25.04 |
| VGG-16 [2]|  Max-Ent. + GFGP | 24.50 |
| ResNet-50 [3]|  Skip | 23.45 |
| ResNet-50 [3]|  Max-Ent. + GFGP | 21.77 |
| ResNeXt-50 [4]|  Skip | 22.42 |
| ResNeXt-50 [4]|  Max-Ent. + GFGP | 21.55 |
| DenseNet-169 [5]|  Avg. | 23.03 |
| DenseNet-169 [5]|  Max-Ent. + GFGP | 22.24 |

## References

[1] T. Kobayashi. "Analyzing Filters Toward Efficient ConvNets." In CVPR, pages 5619-5628, 2018. [pdf](https://staff.aist.go.jp/takumi.kobayashi/publication/2018/CVPR2018.pdf)

[2] K. Simonyan and A. Zisserman. "Very Deep Convolutional Networks For Large-Scale Image Recognition." CoRR, abs/1409.1556, 2014.

[3] K. He, X. Zhang, S. Ren, and J. Sun. "Deep Residual Learning For Image Recognition." In CVPR, pages 770–778, 2016.

[4] S. Xie, R. Girshick, P. Dollar, Z. Tu, and K. He. "Aggregated Residual Transformations For Deep Neural Networks." In CVPR, pages 5987–5995, 2017.

[5] G. Huang, Z. Liu, L. Maaten and K.Q. Weinberger. "Densely Connected Convolutional Networks." In CVPR, pages 2261-2269, 2017.