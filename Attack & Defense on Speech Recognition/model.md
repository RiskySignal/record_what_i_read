# Model on Speech Recognition



[TOC]

## Todo List

1. Chiu, Chung-Cheng, et al. "State-of-the-art speech recognition with sequence-to-sequence models." *2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2018.
2. Zou, Wei, et al. "Comparable study of modeling units for end-to-end mandarin speech recognition." *2018 11th International Symposium on Chinese Spoken Language Processing (ISCSLP)*. IEEE, 2018.
3. Park, Daniel S., et al. "Specaugment: A simple data augmentation method for automatic speech recognition." *arXiv preprint arXiv:1904.08779* (2019).
4. Hannun, Awni, et al. "Deep speech: Scaling up end-to-end speech recognition." *arXiv preprint arXiv:1412.5567* (2014).
5. Amodei, Dario, et al. "Deep speech 2: End-to-end speech recognition in english and mandarin." *International conference on machine learning*. 2016.
6. Battenberg, Eric, et al. "Exploring neural transducers for end-to-end speech recognition." *2017 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)*. IEEE, 2017.
7. T. N. Sainath and C. Parada. Convolutional neural networks for small-footprint keyword spotting. In Sixteenth Annual Conference ofthe International Speech Communication Association, 2015





## Optimizer in Deep Learning

### 梯度下降法（Gradient Descent）

梯度下降法的计算过程就是沿梯度下降的方向求解极小值，也可以沿梯度上升方向求解最大值。使用梯度下降法更新参数：
$$
\theta_{t+1} = \theta_t - \alpha \cdot \nabla_{\theta} J(\theta)
$$

#### 	批量梯度下降法（BGD）

在整个训练集上计算梯度，对参数进行更新：
$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{1}{n} \cdot \sum_{i=1}^{n}\nabla_\theta J_i(\theta, x^i, y^i)
$$
因为要计算整个数据集，收敛速度慢，但其优点在于更趋近于全局最优解；

#### 	随机梯度下降法（SGD）

每次只随机选择一个样本计算梯度，对参数进行更新：
$$
\theta_{t+1} = \theta_t - \alpha \cdot \nabla_\theta J_i(\theta,x^i,y^i)
$$
训练速度快，但是容易陷入局部最优点，导致梯度下降的波动非常大；

#### 	小批量梯度下降法（Mini-batch Gradient Descent）

每次随机选择 n 个样本计算梯度，对参数进行更新：
$$
\theta_{t+1} = \theta_t  - \alpha \cdot \frac{1}m \cdot \sum_{i=x}^{i=x+m-1} \cdot \nabla_\theta J_i(\theta, x^i, y^i)
$$
这种方法是 BGD 和 SGD 的折衷；

### 动量优化法（Momentum）

参数更新时在一定程度上保留之前更新的方向，同时又利用当前 batch 的梯度微调最终的更新方向。在 SGD 的基础上增加动量，则参数更新公式如下：
$$
\m_{t+1} = \mu \cdot m_t + \alpha \cdot \nabla_\theta J(\theta) \\
\theta_{t+1} = \theta_t - m_{t+1}
$$
在梯度方向发生改变时，Momentum 能够降低参数更新速度，从而减少震荡；在梯度方向相同时，Momentum 可以加速参数更新，从而加速收敛。


### Links

- 参考链接：https://zhuanlan.zhihu.com/p/55150256





## Listen, Attend and Spell

### Notes

1. 模型架构：分成两个模块，一个是 Listen Encoder 模块，从语音时序序列中提取出高维特征，采用 pBLSTM (pyramid BLSTM) 的架构；另一个是 Attend and Spell 模块，从语音高维特征中输出单词，采用 Attention + LSTM 架构。架构图如下：

   <img src="./pictures/image-20201130105452791.png" alt="image-20201130105452791" style="zoom: 70%;" />

2. Listen Encoder 模块，使用 pBLSTM 的架构，每层在时间维度上减少一倍，带来的优点有两个：

   (1) 减少模型的复杂性；

   (2) 加快模型的拟合速度（作者发现直接用 BLSTM 的话，用一个月的时间训练都没有办法得到好的结果）；

   形式化的公式为：

   <img src="pictures/image-20201130113050326.png" alt="image-20201130113050326" style="zoom: 24%;" />

3. Attend and Spell 模块，该模块采用 2 层 LSTM 单元来记忆模型当前的状态 s (由模型上一次的状态、输出字符和上下文信息转化而来)，Attention 单元根据当前的状态 s 从特征 h 中分离出“当前模型关心的”上下文信息 c，最后 MLP 单元根据模型的状态 s 和上下文信息 c 输出最可能的字符 y。形式化的公式如下：

   <img src="pictures/image-20201130203431907.png" alt="image-20201130203431907" style="zoom: 31%;" />

   其中 Attention 单元在模型中的实现：将模型状态 s 和特征 h 分别经过两个不同的 MLP 模型，计算出一个标量能量 (Scalar Energy，相当于一个相关性系数) e，然后用 softmax 处理一下这个概率后，和原来的特征 h 加权生成上下文信息 c。形式化的公式如下：

   <img src="pictures/image-20201130211024317.png" alt="image-20201130211024317" style="zoom: 38%;" />

4. Learning. 模型的目标是，在给定 **全部** 语音信号和 **上文** 解码结果的情况下，模型输出正确字符的概率最大。形式化的公式如下：

   <img src="pictures/image-20201130214339602.png" alt="image-20201130214339602" style="zoom:29%;" />

   在训练的时候，我们给的 y 都是 ground truth，但是解码的时候，模型不一定每个时间片都会产生正确的标签。虽然模型对于这种错误是具有宽容度，单训练的时候可以增加 **trick**：以 **10%** 的概率从前一个解码结果中挑选 (根据前一次的概率分布) 一个标签作为 ground truth 进行训练。形式化公式如下：

   <img src="pictures/image-20201130233851764.png" alt="image-20201130233851764" style="zoom: 28%;" />

   另外，作者发现预训练 (主要是预训练 Listen Encoder 部分) 对 LAS 模型没有作用。

5. Decoding & Rescoring. 解码的时候使用 Beam-Search 算法，目标是希望得到概率最大的字符串。形式化公式如下：

   <img src="pictures/image-20201130234826758.png" alt="image-20201130234826758" style="zoom: 18%;" />

   可以用语言模型对最后一轮 Beam-Search 的结果进行重打分，形式化公式如下：

   <img src="pictures/image-20201201000044972.png" alt="image-20201201000044972" style="zoom: 20%;" />

   增加解码结果的长度项 |y| 来**平衡产生长句、短句的权重**，另外语言模型的权重 lambda 可以通过验证集数据来确定。

6. 实验结果：

   (1) 使用 log-mel filter bank 特征

   (2) 整体对比，LAS 刚出来的时候并打不过传统的 DNN-HMM 模型；

   <img src="pictures/image-20201201001625745.png" alt="image-20201201001625745" style="zoom: 25%;" />

   (3) Attention 模块确实更加关注对应时间片段的特征；

   <img src="pictures/image-20201201002008091.png" alt="image-20201201002008091" style="zoom: 50%;" />

   (4) 模型对于较短的语句或者较长的语句效果都不是很好；

   <img src="pictures/image-20201201003246685.png" alt="image-20201201003246685" style="zoom: 50%;" />

### Shortcoming

1. 必须要得到整个语音后才能解码，限制了模型的流式处理能力；
2. Attention 机制需要消耗大量的计算量；
3. 输入长度对于模型的影响较大；

### Links

- 论文链接：[Listen, Attend and Spell](https://arxiv.org/abs/1508.01211)
- LAS 模型缺点参考链接：[LAS 语音识别框架发展简述](https://blog.csdn.net/weixin_39529413/article/details/103570831)
- Pytorch 实现：[End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch) ( <u>**暂未阅读代码**</u> )





## Lingvo: a modular and scalable framework for sequence-to-sequence modeling

谷歌开源的基于tensorflow的序列模型框架。

### Notes

### Links

- 论文链接：[Lingvo: a modular and scalable framework for sequence-to-sequence modeling](https://arxiv.org/abs/1902.08295)
- Github：[Lingvo](https://github.com/tensorflow/lingvo)

