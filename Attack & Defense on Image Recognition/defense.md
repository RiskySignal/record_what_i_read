# Defense on Image Recognition



[TOC]



## Todo List

- Muzammal Naseer, Salman Khan, and Fatih Porikli. Local gradients smoothing: Defense against localized adversarial attacks. In 2019 IEEE Winter Conference on Applications ofComputer Vision (WACV), pp. 1300–1307. IEEE, 2019.
- Jamie Hayes. On visible adversarial perturbations & digital watermarking. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 1597–1604, 2018.
- Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. arXiv preprint arXiv:1802.00420, 2018.
- Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Uesato, Timothy Mann, and Pushmeet Kohli. On the effectiveness of interval bound propagation for training verifiably robust models. arXiv preprint arXiv:1810.12715, 2018.
- Matthew Mirman, Timon Gehr, and Martin Vechev. Differentiable abstract interpretation for provably robust neural networks. In International Conference on Machine Learning, pp. 3575–3583, 2018.
- Alexander Levine and Soheil Feizi. Robustness certificates for sparse adversarial attacks by randomized ablation. arXiv preprint arXiv:1911.09272, 2019.
- Gotta Catch’Em All: Using Honeypots to Catch Adversarial Attacks on Neural Networks



## Towards Deep Learning Models Resistant to Adversarial Attacks

> 由于时间原因，该文章的笔记借鉴自 “前人分享”（链接见下）。

### Contribution

1. 建模了对抗训练过程；
2. 使用 PGD生成的对抗样本 来做对抗训练；

### Notes

1. ⭐ 问题建模，**从优化的角度来看模型鲁棒性问题**。深度学习中，我们经常根据下面这个目标来训练我们的网络：即我们希望我们训练得到的模型在训练样本**上**的经验损失能够达到最小。
   $$
   \min_\theta \rho(\theta), \;\; \text{where} \;\; \rho(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[
   L(\theta, x, y)
   \right]
   $$
   但是这样的训练目标，使得模型容易受到对抗样本的攻击。故作者将对抗样本的攻击防御问题总结为以下公式，该问题原文中作者称为**鞍点问题（saddle point problem）**，即我们希望我们训练得到的模型在训练样本**周围**的经验损失能够达到最小。
   $$
   \min_\theta \rho(\theta), \;\; \text{where} \;\; \rho(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\max_{\delta \in \mathcal{S}}
   L(\theta, x+\delta, y)
   \right]
   $$
   建模完问题以后，那么以前的对抗样本领域的工作就可以进行简单地分类：（<u>稍微有点绕</u>）

   - 提出一个好的对抗攻击算法，来寻找使得（内层）经验损失最大化的扰动；
   - 提出一个鲁棒性好的模型，来使得（外层）最小化（内层的最大的）经验损失；

2. 文章中作者采用投影梯度下降算法（PGD）来生成对抗样本：

   <img src="pictures/image-20210131225428946.png" alt="image-20210131225428946" style="zoom: 19%;" />

3. 实验发现：

   (1) Loss 下降趋势和对抗样本算法迭代轮数的关系：无论是原始模型还是使用对抗训练得到的模型，两者使用 PGD算法 生成对抗样本时，随着迭代轮数的上升，样本的loss都会上升，且到最后趋于收敛；

   <img src="pictures/image-20210131225722450.png" alt="image-20210131225722450" style="zoom: 40%;" />

   (2) Loss 分布的差异：于原始模型相比，在对抗训练得到的模型上生成对抗样本，得到的loss更小，更集中且没有异常值；

   <img src="pictures/image-20210131230018586.png" alt="image-20210131230018586" style="zoom: 50%;" />

   (3) 鲁棒性与模型规模的关系：相对而言，模型越复杂，鲁棒性也越好。同时，经过对抗训练的模型，在原始任务上会有一定的损失，是因为出现了如“过拟合”的现象，使得模型在测试集上面的效果并不好；

   <img src="pictures/image-20210131230643571.png" alt="image-20210131230643571" style="zoom: 55%;" />

   (4) 范数限制的影响：$l_\infty$ 范数比 $l_2$ 范数成功的扰动量要小；（<u>这个对比合理吗？</u>）

   <img src="pictures/image-20210131231034909.png" alt="image-20210131231034909" style="zoom:48%;" />

### Links

- 论文链接：[Madry A, Makelov A, Schmidt L, et al. Towards deep learning models resistant to adversarial attacks[J]. arXiv preprint arXiv:1706.06083, 2017.](https://arxiv.org/abs/1706.06083?spm=5176.12281978.0.0.5f797a4dKc7U2f&file=1706.06083)
- 论文代码 - mnist：https://github.com/MadryLab/mnist_challenge
- 论文代码 - cifar10：https://github.com/MadryLab/cifar10_challenge

> 从代码上来看，作者提供的模型的输入是 $32*32*3$ 大小的，这样经过压缩的输入维度，是否导致了对抗样本算法难以实现呢？或者说生成对抗样本的过程能否利用一下这个特点？

- 论文模型：https://github.com/MadryLab/cifar10_challenge
- 前人分享：https://zhuanlan.zhihu.com/p/45684812



# Unlabeled Data Improves Adversarial Robustness

### Notes



### Links

- [Carmon Y, Raghunathan A, Schmidt L, et al. Unlabeled data improves adversarial robustness[J]. arXiv preprint arXiv:1905.13736, 2019.](https://arxiv.org/abs/1905.13736)