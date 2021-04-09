# DeepLearning



[TOC]



## 神经网络中的 Normalization 的发展历程

> 参考链接：[[笔记] 神经网络中 Normalization 的发展历程](https://zhuanlan.zhihu.com/p/75539170)





## 人工智能前言讲习

> “他山之石，可以攻玉”，站在巨人的肩膀才能看得更高，走的更远。

- [【他山之石】整理 Deep Learning 调参 tricks](https://mp.weixin.qq.com/s/Gw8K0GggRcahwLf3tu4LrA)
- [【他山之石】深度学习中的那些 Trade-off](https://mp.weixin.qq.com/s/RoEwx7qAUlSvjB608zOx1g)
- [【他山之石】tensorflow2.4性能调优最佳实践](https://mp.weixin.qq.com/s/BI2BjAJGXzRk4k9d99PgLQ)
  - [【梦想做个翟老师】浅谈Tensorflow分布式架构：ring all-reduce算法](https://zhuanlan.zhihu.com/p/69797852)
  - [【瓦特兰蒂斯】单机多卡的正确打开方式（二）：TensorFlow](https://fyubang.com/2019/07/14/distributed-training2/)

- [【强基固本】机器学习常用评价指标总览](https://mp.weixin.qq.com/s/MVw3IIno4iyTNaEOjBLzAQ)



## Problems in Coding

- Keras  和 Multiprocessing 组合 Bug：

  我在 windows 上面运行的很好，但是放到 Linux 服务器上面后，子进程中的 `keras.models.load_model()` 就卡住不动了。Bug 原理和解决方法参考博客 “[keras使用多进程](https://www.cnblogs.com/zongfa/p/12193561.html)”，写的非常棒，体会到了进程拷贝的问题。Bug 在 github 上面的链接参考 [Keras is not multi-processing safe](https://github.com/keras-team/keras/issues/9964) ；