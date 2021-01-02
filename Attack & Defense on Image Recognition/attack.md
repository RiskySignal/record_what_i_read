# Attack on Image Recognition



[TOC]

## Todo List

1. Kurakin, A., Goodfellow, I., and Bengio, S. Adversarial examples in the physical world. 2016. 
2. Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., and Fergus, R. Intriguing properties of neural networks. 2013.
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
14. Stealthy Porn: Understanding Real-World Adversarial Images for Illicit Online Promotion
15. Stealing Hyperparameters in Machine Learning
16. Phantom of the ADAS: Securing Advanced Driver-Assistance Systems from Split-Second Phantom Attacks
17. Text Captcha Is Dead? A Large Scale Deployment and Empirical Study
18. A Tale of Evil Twins: Adversarial Inputs versus Poisoned Models
19. Adversarial Sensor Attack on LiDAR-based Perception in Autonomous Driving
20. Privacy Risks of Securing Machine Learning Models against Adversarial Examples
21. Procedural Noise Adversarial Examples for Black-Box Attacks on Deep Convolutional Networks
22. Seeing isn’t Believing: Towards More Robust Adversarial Attack Against Real World Object Detectors
23. Model-Reuse Attacks on Learning Systems
24. A. Ilyas, L. Engstrom, A. Athalye, and J. Lin, “Black-box adversarial attacks with limited queries and information,” in ICML, 2018.
25. A. Kurakin, I. J. Goodfellow, and S. Bengio, “Adversarial examples in the physical world,” in ICLR, 2017.
26. Yue Zhao, Hong Zhu, Ruigang Liang, Qintao Shen, Shengzhi Zhang, Kai Chen, "Seeing isn’t Believing: Towards More Robust Adversarial Attack Against Real World Object Detectors", CCS 2019





## 范数

> （**向量**）范数在对抗样本、模型可解释性等方向经常遇到，主要用来限制扰动的“形状”和范围，本人经常是看一次忘一次，故将这个知识点列在最前面；

### 向量范数

#### 	1-范数

向量元素绝对值之和：$\lVert \mathbf{x} \rVert_1 = \sum_{i=1}^N |x_i|$；

#### 	2-范数

向量的欧几里得长度：$\lVert \mathbf{x} \rVert_2 = \sqrt{\sum_{i=1}^N x_i^2}$；

#### 	无穷范数

(1) $\infty$-范数，向量元素绝对值中的最大值：$\lVert \mathbf{x} \rVert = \max_i{|x_i|}$；

(2) $-\infty$-范数，向量元素绝对值中的最小值：$\lVert \mathbf{x} \rVert = \min_i{|x_i|}$；

#### 	p-范数

向量元素绝对值的 **$p$ 次方和**的 **$p$ 次幂**：$\lVert \mathbf{x} \rVert = (\sum_{i=1}^N |x|^p)^{\frac{1}p}$；

### * 矩阵范数

#### 	1-范数

矩阵**列向量**绝对值之和的最大值：$\lVert \boldsymbol{A} \rVert_1 = \max_j \sum_{i=1}^m |x_{i,j}|$；

#### 	2-范数

$\lVert \boldsymbol{A} \rVert_2 = \sqrt{\lambda_{max}}$，其中$\lambda_{max}$ 是 $\boldsymbol{A}^T \boldsymbol{A}$ 的最大特征值；

#### 	无穷范数

矩阵**行向量**绝对值之和的最大值：$\lVert \boldsymbol{A} \rVert_\infty = \max_i \sum_{j=1}^N |a_{i,j}|$

### 参考链接

- 知乎回答：https://www.zhihu.com/question/20473040





## Intriguing properties of neural networks

### Contribution

1. 首次提出了对抗样本（Adversarial Examples）概念；

### Notes

1. 生成对抗样本：添加一个扰动，使得原始输入的分类变成目标输入，且保证添加的扰动的 `2-范数` 最小；

   <img src="pictures/image-20210102105605711.png" alt="image-20210102105605711" style="zoom:15%;" />

   作者通过 梯度下降法求解损失函数最小化 来实现上述对抗样本生成：

   <img src="pictures/image-20210102105818702.png" alt="image-20210102105818702" style="zoom:23%;" />

2. 文章提到的几个要点：

   (1) 神经网络隐层的语义：神经网络某个隐藏层中携带的语义信息并不只在单个神经元中，而是这个隐层所表示的整个空间（线性关系）；作者通过 Word Embedding 举例说明了这个观点，词向量之间的距离代表了两个词之间的语义相似性，并且对词向量进行旋转变换后这部分语义并不会消失，但是词向量则完全发生了改变；

   (2) 对抗样本的普遍存在性；

   (3) 对抗样本的迁移性：同一个对抗样本可能在不同的模型上都可以使模型错误分类；

   (4) 对抗样本具有跨数据集的泛化能力：在 $D_1$ 数据集训练的模型 $A$ 上生成的对抗样本可能使得 $D_2$ 数据集训练的模型 $B$ 错误分类；

### Links

- 论文链接：[Szegedy C, Zaremba W, Sutskever I, et al. Intriguing properties of neural networks[J]. arXiv preprint arXiv:1312.6199, 2013.](https://arxiv.org/abs/1312.6199)

- 前人笔记：[Jun Tao 的个人博客](https://tanjuntao.github.io/2019/07/10/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0%EF%BC%9AIntriguing-Properties-of-Neural-Network/)





## Explaining and Harnessing Adversarial Examples

### Notes

1. 线性解释：作者提出了对抗样本存在的**线性解释**。如果将神经网络泛化成如下所示的**线性**点积形式：

   <img src="pictures/image-20201231010621616.png" alt="image-20201231010621616" style="zoom: 10%;" />

   其中 $\tilde{x}$ 为目标类的一个样本，$x$ 为原始样本，$\eta$ 为添加的对抗扰动。当输入的维度无限扩大时，很显然我们可以保证 $\lVert \eta \rVert_\infty$ 很小的情况下，而点积后的值却变化很大，使得上式两侧的值相等，即生成了一个成功的对抗扰动；

2. 对抗样本生成算法：作者提出了基于对抗样本线性解释的快速对抗样本生成算法，**Fast Gradient Sign Method（FGSM）生成对抗样本只需要计算一次梯度，然后在梯度上走一小步**：

   <img src="pictures/image-20201231011859924.png" alt="image-20201231011859924" style="zoom: 12%;" />

3. 对抗训练：作者提出了在深度神经网络中可以通过对抗训练的方法来防御对抗攻击。下面利用 FGSM 进行对抗训练：

   <img src="pictures/image-20201231012210113.png" alt="image-20201231012210113" style="zoom:27%;" />

   公式的含义为，**在训练网络的过程中，不仅要保证现有样本能够被成功分类，对于那些添加了一小步的对抗样本应该同样被正确分类**；

4. 泛化原因：作者解释**对抗样本的存在位置并不是一个特定的点，而是一个区域，或称为子空间**。由于子空间的这个属性，以及深度学习模型对训练集拟合的相同效果，可能出现了**对抗样本子空间的重叠，使得对抗样本可以在不同的模型之间进行迁移**；

### Links

- 论文链接：[Goodfellow I J, Shlens J, Szegedy C. Explaining and harnessing adversarial examples[J]. arXiv preprint arXiv:1412.6572, 2014.](https://arxiv.org/abs/1412.6572)
- 论文代码：https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/papers/maxout
- 前人笔记：https://zhuanlan.zhihu.com/p/32784766





## ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models

### Contribution

1. 提出了第一个基于梯度估计的黑盒优化攻击算法；
2. 针对梯度估计过程中需要大量访问黑盒模型的问题，提出了 3 中可行的缓解方法（<u>访问次数依然很大</u>）；

### Notes

1. 作者提出了一种**针对黑盒的优化攻击（Optimization Attack）算法**；

2. Introduction：（<u>文章比较早，故作者用较多的篇幅介绍了对抗攻击领域的工作</u>）

   (1) 白盒攻击算法：FGSM（Fast Gradient Sign Method），JSMA（Jacobian-based Saliency Map Attack），DeepFool，Carlini & Wagner (C&W) Attack；

   (2) 本文攻击场景：攻击黑盒模型，攻击者只知道输入和相应的输出；

   (3) 对抗攻击防御：Detection-based Defense，Gradient and Representation Masking，Adversarial training；

   > 企业更加看重黑盒攻击和防御，所以如果以工作为目标的同学，需要多学习、思考这方面的内容

3. ZOO 攻击算法：

   (1) 借鉴 C&W Attack，将生成对抗样本的过程转换成一个最优化问题：

   <img src="pictures/image-20201231093224292.png" alt="image-20201231093224292" style="zoom: 15%;" />

   ​	其中 $f(x,t)$ 为损失函数；

   (2) 损失函数：

   - 有目标攻击的损失函数如下：

     <img src="pictures/image-20201231095157405.png" alt="image-20201231095157405" style="zoom:20%;" />

   - 无目标攻击的损失函数如下：

     <img src="pictures/image-20201231095257739.png" alt="image-20201231095257739" style="zoom: 20.5%;" />

   (3) 零阶优化：

   - 一阶导数估计：

     <img src="pictures/image-20201231100111028.png" alt="image-20201231100111028" style="zoom: 15%;" />

   - 二阶导数估计：

     <img src="pictures/image-20201231100158156.png" alt="image-20201231100158156" style="zoom: 19%;" />

     其中 $h$ 为一个极小的固定值，文章中作者取 $0.0001$，$e_i$ 为只有第 $i$ 个值为 $1$ 的矩阵。如果输入的矩阵（图像）含有 $p$ 个像素点的话，那么通过作者的方法需要访问模型 $2p$ 次。

   (4) Stochastic Coordinate Descent：（<u>直译过来为”随机坐标下降”</u>）随机从输入中挑选一个点，使用梯度下降算法进行修改；

   - Stochastic Coordinate Descent：

     <img src="pictures/image-20201231101508857.png" alt="image-20201231101508857" style="zoom: 24%;" />

   - ZOO-ADAM：

     <img src="pictures/image-20201231101554869.png" alt="image-20201231101554869" style="zoom:33%;" />

   - ZOO-Newton：

     <img src="pictures/image-20201231101634223.png" alt="image-20201231101634223" style="zoom: 29%;" />

     作者实验中发现，ADAM 比 Newton 生成对抗样本来得更快；

   (5) 缩小迭代空间：为了减少 ZOO 的 Query 数量，从而加快算法的运行。大致的思想是进行**对抗扰动特征空间的映射**，定义一个 （更小的）扰动特征空间 $\mathbb{R}^p$ 和特征映射函数 $D(\cdot)$，那么转换（原特征空间-像素空间的）最优化问题为扰动特征空间的最优化问题：

   <img src="pictures/image-20201231103044288.png" alt="image-20201231103044288" style="zoom: 16%;" />

   其中 $y$ 表示在扰动特征空间的对抗扰动；作者提到的特征映射方法有 Up-Sampling （升采样） 和 DCT（时频变换）；

   (6) 分层递进攻击：前一个方法可以大大减小对抗样本的搜索空间，但是由于搜索空间的受限，会导致无法生成成功的对抗样本的问题。大致的思想是定义**多个**对抗扰动特征空间的映射 $D_1(\cdot), D_2(\cdot), \dots$，攻击过程中首先使用 $D_1$ 生成对抗样本，如果在一定轮数后仍未生成成功的对抗样本，那么将最后一轮的样本转换到 $D_2$ 的特征空间（**后面使用的特征空间应保证比前面的特征空间更广**），继续生成对抗样本。

   (7) 重要像素点优先迭代：作者虽然缩小了查询的特征空间（$32*32*3 \text{ for example}$），但是在这个空间中生成对抗样本还是需要花费大量的 Query 次数，并且不一定能够生成成功的对抗样本。大致的思想是**将图像切块，分块定义像素点被随机采样的概率，概率的大小和区域中像素值的变化大小成正相关**。作者给出了大致的采样概率变化示意图：

   <img src="C:/Users/Ceres/AppData/Roaming/Typora/typora-user-images/image-20201231105134731.png" alt="image-20201231105134731" style="zoom:55%;" />

   作者指出在小的扰动空间时，并不采用这种优先采样算法；

4. Evaluation 1：

   (1) 实验 1 的目标：这是第一个在黑盒模型上做的优化攻击，所以作者的目标是和已有的白盒攻击（C&W Attack）和迁移攻击（CleverHans）做对比，希望能够达到这样的效果：

   - 攻击的成功率和添加的对抗扰动大小能够和白盒攻击算法相近；
   - 攻击的成功率应该远优于迁移攻击；

   (2) 黑盒模型：

   <img src="pictures/image-20201231130226748.png" alt="image-20201231130226748" style="zoom: 90%;" /><img src="pictures/image-20201231130309608.png" alt="image-20201231130309608" style="zoom:90%;" />

   (3) 样本数量：

   - 有目标攻击，生成 900 个对抗样本；（其他细节见原文）
   - 无目标攻击，生成 200 个对抗样本；

   (4) 实验结果：

   <img src="pictures/image-20201231130656635.png" alt="image-20201231130656635" style="zoom: 50%;" />

5. Evaluation 2:

   (1) 实验 2 目标：作者尝试将这种攻击运用在更大的模型上，并且探讨文章提出的缓解方法的作用；

   (2) 黑盒模型：Inception-V3;

   (3) 实验设定：

   - 无目标攻击：

     生成150张对抗样本；保证每张对抗样本的大小都大于 $299*299$；不使用分层递进方法，只使用一个 $32*32*3$ 的对抗扰动域进行攻击；限制算法的迭代轮数为 1500 轮（$1500*128$ 次 Query）；

   - 有目标攻击：

     只选择了一张在无目标攻击中无法成功攻击的样本；扩大对抗扰动域为 $64*64*3$ 和 $128*128*3$；最大迭代轮数上升为 20000 轮（$20000*128$ 次 Query）；（<u>可以看到，黑盒下面的 Query 数量是十分惊人的</u>）

   (4) 实验结果：

   - 无目标攻击：

     <img src="pictures/image-20201231143233361.png" alt="image-20201231143233361" style="zoom:22%;" />

   - 有目标攻击：

     <img src="pictures/image-20201231143330484.png" alt="image-20201231143330484" style="zoom:23%;" />

### Links

- 论文链接：[Chen P Y, Zhang H, Sharma Y, et al. Zoo: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models[C]//Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security. 2017: 15-26.](https://arxiv.org/abs/1708.03999)
- 论文代码：https://github.com/IBM/ZOO-Attack
- C&W Attack 代码：https://github.com/carlini/nn_robust_attacks
- CleverHans：https://github.com/cleverhans-lab/cleverhans





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





## NES: Black-box Adversarial Attacks with Limited Queries and Information

### Contribution

1. 利用 NES 算法大大减少黑盒攻击的访问次数；

### Notes

1. 黑盒设定：

   - Query-limited Setting: 限制访问次数 ;
   - Partial-information Setting: 只知道 Top-K 的结果 (包括概率);
   - Label-only Setting: 只知道 Top-K 的标签 (不包括概率); (<u>这一项我觉得没必要看</u>)

2. NES (Natural Evolutionary Strategies) 进行梯度估计: 最小化期望的损失大小, 算法伪代码如下 (<u>如何挑选这个参数?</u>)

   <img src="pictures/image-20201228005801444.png" alt="image-20201228005801444" style="zoom: 28%;" />

3. PGD (Projected Gradient Descent) 进行梯度更新:

   <img src="pictures/image-20201228010620801.png" alt="image-20201228010620801" style="zoom:15%;" />

4. 仅知道 Top-K 的概率:

   <img src="pictures/image-20201228011853945.png" alt="image-20201228012415004" style="zoom: 60%;" />

   - 使用目标分类的样本来初始化扰动，从而减少 query 的数量；
   - 在保证目标分类在 Top-K 中的前提下，不断缩小对抗扰动，直至生成对抗样本且满足修改量的限制；

5. Evaluation：

   (1) 参数的选择：

   <img src="pictures/image-20201228014735192.png" alt="image-20201228014735192" style="zoom: 25%;" />

   (2) On ImageNet：这里大概的 query 数量级为上万级别的

   <img src="pictures/image-20201228015018920.png" alt="image-20201228015018920" style="zoom:28%;" />

### Links

- 论文链接: [Ilyas, Andrew, et al. "Black-box adversarial attacks with limited queries and information." *PRML* (2018).](https://arxiv.org/abs/1804.08598)
- 论文代码: https://github.com/labsix/limited-blackbox-attacks





## Hybrid Batch Attacks: Finding Black-box Adversarial Examples with Limited Queries

### Contribution



### Notes

1. 作者结合 迁移攻击（Transfer Attack） 和 优化攻击（Optimization Attack），并且利用种子优先级策略对模型进行黑盒攻击，目标是在保证攻击成功率的前提下，减少 Query 的数量；

2. 现有的梯度优化攻击算法：

   <img src="pictures/image-20201230225437994.png" alt="image-20201230225437994" style="zoom: 35%;" />

   

### Links

- 论文链接：[Suya F, Chi J, Evans D, et al. Hybrid batch attacks: Finding black-box adversarial examples with limited queries[C]//29th {USENIX} Security Symposium (USENIX Security 2020). 2020: 1327-1344.](https://www.usenix.org/conference/usenixsecurity20/presentation/suya)
- 论文代码：https://github.com/suyeecav/Hybrid-Attack