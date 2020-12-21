# Attack on Image Recognition



[TOC]

## Todo List

1. Kurakin, A., Goodfellow, I., and Bengio, S. Adversarial examples in the physical world. 2016. 
2. Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., and Fergus, R. Intriguing properties of neural networks. 2013.
3. Goodfellow, I. J., Shlens, J., and Szegedy, C. Explaining and harnessing adversarial examples. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2015.
4. Carlini, N. and Wagner, D. Towards evaluating the robustness of neural networks. In IEEE Symposium on Security & Privacy, 2017c.
5. Evtimov, I., Eykholt, K., Fernandes, E., Kohno, T., Li, B., Prakash, A., Rahmati, A., and Song, D. Robust PhysicalWorld Attacks on Deep Learning Models. 2017.
6. Tom B Brown, Dandelion Man´e, Aurko Roy, Mart´ın Abadi, and Justin Gilmer. Adversarial patch. arXiv preprint arXiv:1712.09665, 2017.
7. Danny Karmon, Daniel Zoran, and Yoav Goldberg. Lavan: Localized and visible adversarial noise. arXiv preprint arXiv:1801.02608, 2018.
8. Zuxuan Wu, Ser-Nam Lim, Larry Davis, and Tom Goldstein. Making an invisibility cloak: Real world adversarial attacks on object detectors. arXiv preprint arXiv:1910.14667, 2019.
9. Cassidy Laidlaw and Soheil Feizi. Functional adversarial attacks, 2019
10. Mahmood Sharif, Sruti Bhagavatula, Lujo Bauer, and Michael K Reiter. Adversarial generative nets: Neural network attacks on state-of-the-art face recognition. arXiv preprint arXiv:1801.00349, 2017.
11. Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. arXiv preprint arXiv:1802.00420, 2018.
12. advbox
13. paddlepaddle 对抗工具箱

## Synthesizing Robust Adversarial Examples

### Contribution

1. 提出了一种增加物理环境下对抗样本鲁棒性的一般化方法 EOT；
2. 不仅在 2D 下测试，而且在 3D 下测试；
3. 模拟物理变换的想法十分具有借鉴意义，已被后续的对抗攻击算法广泛使用；

### Notes

1. **白盒**的、**针对物理环境**下的、**有目标**的对抗攻击算法。攻击的算法不仅在 2D 下可行，同时在 **3D** 下也可以生成成功的对抗样本；

2. 已有的对抗攻击算法，训练的目标如下：

   <img src="pictures/image-20201205002305615.png" alt="image-20201205002305615" style="zoom:13%;" />

   但是这样生成的对抗样本，在视角等物理环境发生改变时**无法保持对抗性**。故作者提出改进后的训练目标 **EOT (Expectation Over Transformation)**：

   <img src="pictures/image-20201205003104148.png" alt="image-20201205003104148" style="zoom:15%;" />

   其含义是，**在保证对抗样本经过物理变换的 “感受” 修改量在一定范围内时，使得对抗样本（经过物理变换）能够尽可能地被分类为目标类别**。这类物理变换可以是 2D/3D 的变换，包括随机旋转、平移、噪声、视角变化、光照等。作者将公式转换为 [Carlini &Wagner (2017c)](https://arxiv.org/abs/1608.04644) 的形式，并使用**二级范数**和 **PGD** (Projected Gradient Descent) 优化器进行计算：

   <img src="pictures/image-20201205100259271.png" alt="image-20201205100259271" style="zoom:12%;" />

   其中 *LAB* 代表指的是 [LAB 色域](https://blog.csdn.net/gdymind/article/details/82357139) 。

3. Distributions of Transformations:

   (1) **2D Case**

   <img src="pictures/image-20201205133736692.png" alt="image-20201205133736692" style="zoom: 20%;" />

   (2) **3D Case**

   <img src="pictures/image-20201205134021170.png" alt="image-20201205134021170" style="zoom:22%;" />

   (3) **Physical Case**

   <img src="pictures/image-20201205134140775.png" alt="image-20201205134140775" style="zoom:25%;" />

4. Evaluation：

   (1) 攻击基于数据集 ImageNet 的 **Inception V3** 模型 (`Top-1 Accuracy = 78.0%`)，随机选择目标分类；

   (2) **Robust 2D adversarial examples**：在 2D 下考虑的物理变换有 **缩放 、旋转、亮度调节、高斯噪声和平移**。每个样本都在 **1000** 个随机的模拟物理变换上进行测试，结果如下：

   <img src="C:/Users/Ceres/AppData/Roaming/Typora/typora-user-images/image-20201205105608634.png" alt="image-20201205105608634" style="zoom: 27%;" />

   (3) **Robust 3D adversarial examples**：在 3D 下考虑**不同的相机距离、照明条件、对象的平移和旋转以及纯色背景色**。挑选了 10 个 3D 模型 —— 木桶、棒球、够、橘子、海龟、小丑鱼、沙发、泰迪熊、汽车和出租车。每个 3D 模型都挑选 20 个随机的目标分类标签；每个样本都在 100 个随机的模拟物理变换上进行测试，结果如下：

   <img src="pictures/image-20201205113737288.png" alt="image-20201205113737288" style="zoom: 27%;" />

   (4) **Physical adversarial examples**：在 3D 的基础上，考虑**摄像机的噪声、照明的影响和颜色的失真**。作者考虑将 ”海龟“ 错误分类成 ”手枪“、 ”棒球“ 错误分类成 ”咖啡“ 两种情况，将对抗样本经过 3D 打印后，拍 100 张照片进行测试，结果如下：

   <img src="pictures/image-20201205115918823.png" alt="image-20201205115918823" style="zoom:26%;" />

   (5) **Perturbation budget**：在物理环境下越鲁棒，需要模拟更多的物理变换，添加的噪声也会更多；

### Links

- 论文链接：[Athalye, Anish, et al. "Synthesizing robust adversarial examples." *International conference on machine learning*. PMLR, 2018.](https://arxiv.org/abs/1707.07397)
- 开源代码：[prabhant/synthesizing-robust-adversarial-examples (github.com)](https://github.com/prabhant/synthesizing-robust-adversarial-examples)