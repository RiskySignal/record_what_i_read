# Problems in Coding

[TOC]



## Python





## Tensorflow

#### Keras  和 Multiprocessing 组合 Bug

我在` windows` 上面运行的很好，但是放到 Linux 服务器上面后，子进程中的 `keras.models.load_model()` 就卡住不动了。Bug 原理和解决方法参考博客 “[keras使用多进程](https://www.cnblogs.com/zongfa/p/12193561.html)”，写的非常棒，体会到了进程拷贝的问题。Bug 在 github 上面的链接参考 [Keras is not multi-processing safe](https://github.com/keras-team/keras/issues/9964) ；

#### 



## PyTorch

#### Python 的 `@staticmethod / @classmethod` 方法

主要参考知乎大佬 [正确理解Python中的 @staticmethod@classmethod方法](https://zhuanlan.zhihu.com/p/28010894)，这里需要注意的是 PyTorch 的 `torch.nn.Function` 类中的 `forward/backward` 方法是比较特殊的；

#### PyTorch 的 `torch.nn.RNN` 源码分析

主要参考知乎大佬 [读PyTorch源码学习RNN（1）](https://zhuanlan.zhihu.com/p/32103001)，这里注意 PyTorch 的输入输出，以及如何进行时间片上的状态传递的；

#### PyTorch 镜像翻转实现

主要参考博客大佬 [Tensor的镜像翻转](https://heroinlin.github.io/2018/03/12/Pytorch/Pytorch_tensor_flip/)，镜像翻转的代码如下：

```python
import pytorch
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
```