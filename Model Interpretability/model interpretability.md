# Model Interpretability



[TOC]



## Todo List

1. Bach S, Binder A, Montavon G, et al. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation [J]. PloS one, 2015, 10(7): e0130140.
3. Guo, Wenbo, et al. "Lemna: Explaining deep learning based security applications." *Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security*. 2018.
4. Tao Guanhong, Ma Shiqing, Liu Yingqi, et al. Attacks meet interpretability: Attribute-steered detection of adversarial samples [C] //Proc of the 32st Int Conf on Neural Information Processing Systems. USA: Curran Associates Inc., 2018: 7717-7728
5. Liu Ninghao, Yang Hongxia, Hu Xia. Adversarial detection with model interpretation [C] //Proc of the 24th ACM SIGKDD Int Conf on Knowledge Discovery & Data Mining. New York: ACM, 2018: 1803-1811
6. Carlini N, Wagner D. Towards evaluating the robustness of neural networks [C] //Proc of the 38th IEEE Symposium on Security and Privacy. Piscataway, NJ: IEEE, 2017: 39-57
7. Papernot N, Mcdaniel P, Jha S, et al. The limitations of deep learning in adversarial settings [C] //Proc of the 1st IEEE European Symp on Security and Privacy. Piscataway, NJ: IEEE, 2016: 372-387
8. Papernot N, Mcdaniel P, Goodfellow I, et al. Practical blackbox attacks against machine learning [C] //Proc of the 12th ACM Asia Conf on Computer and Communications Security. New York: ACM, 2017: 506-519
10. Ghorbani A, Abid A, Zou J. Interpretation of neural networks is fragile [J]. arXiv preprint arXiv:1710.10547, 2017
11. Zhang Xinyang, Wang Ningfei, Ji Shouling, et al. Interpretable Deep Learning under Fire [C] //Proc of the 29th USENIX Security Symp. Berkele, CA: USENIX Association, 2020
11. GRADIENTS OF COUNTERFACTUALS(ICLR 2017)
12. Simonyan, Karen, Vedaldi, Andrea, and Zisserman, Andrew. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.
13. Explainable Neural Network based on Generalized Additive Model.
13. Montavon, G., Lapuschkin, S., Binder, A., Samek, W., M¨uller, K.R.: Explaining nonlinear classification decisions with deep Taylor decomposition. Pattern Recogn. 65, 211–222 (2017)





## On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation

### Contribution

1. 作者提出了 LRP 的可解释性方法；

### Links

- 论文链接：[Bach S, Binder A, Montavon G, et al. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation[J]. PloS one, 2015, 10(7): e0130140.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)





## Layer-wise Relevance Propagation for Neural Networks with Local Renormalization Layers

### Contribution

1. 作者将 LRP 可解释方法扩展到了 `Local Renormalization Layer` 这种非线性的网络层；

### Notes

1. LRP（Layer-wise Relevance Propagation）：该方法利用 **神经元之间的数值关系**，回传 **相关性系数** （注意和梯度是有区别的，<u>我认为他和梯度的方法最大的不同点在于，该方法考虑了神经元之间的相关性</u>）

   (1) 神经元的激活公式如下：

   <img src="images/image-20210407000254371.png" alt="image-20210407000254371" style="zoom: 10%;" />

   (2) 给定一张图片 $x$ 和一个神经网络 $f$ ，LRP 方法的目的在于给图像的每一个像素点都给定一个 **相关性系数**：

   <img src="images/image-20210407000605063.png" alt="image-20210407000605063" style="zoom:9%;" />

   (3) 给定神经网络 $l+1$ 层的神经元，它的 **相关性系数** 存在如下公式：

   <img src="images/image-20210407000931610.png" alt="image-20210407000931610" style="zoom:10%;" />

   (4) 那么，我们就可以得到 $l$ 层的神经元，它的 **相关性系数** 存在如下公式：（即完成了系数的回传）

   <img src="images/image-20210407001056776.png" alt="image-20210407001056776" style="zoom:10%;" />

   ​	其中，最后一层的 相关性系数 等于 $f(x)$；

   (5) 在 [前文](#On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation) 中，有两套公式来计算系数的回传：

   - $\epsilon-rule$：

     <img src="images/image-20210407090115029.png" alt="image-20210407090115029" style="zoom:15%;" />

     其中，<img src="images/image-20210407090227665.png" alt="image-20210407090227665" style="zoom:19%;" />；$\epsilon$ 用来防止除 $0$；

   - $\beta-rule$：

     <img src="images/image-20210407090516976.png" alt="image-20210407090516976" style="zoom:18%;" />

     其中，$\beta$ 用来控制相关性再分配中对负相关系数的关注程度，其值越大生成的热力图越锐化；

   - 合并上面两套公式：

     <img src="images/image-20210407091700377.png" alt="image-20210407091700377" style="zoom:19%;" />

2. Extending LRP：将 LRP 方法扩展到 LRN 层；

   (1) 扩展的原因：我们先看一下 LRN 层的激活公式

   <img src="images/image-20210407091943735.png" alt="image-20210407091943735" style="zoom:15%;" />

   ​	很显然，它不满足第一部分讲得一般的神经网络激活公式；

   (2) ⭐ 泰勒展开式：假设存在 <img src="images/image-20210407093831787.png" alt="image-20210407093831787" style="zoom:12%;" />，我们可以对其使用一阶泰特展开式展开

   <img src="images/image-20210407093727551.png" alt="image-20210407093727551" style="zoom:25%;" />

   ​	那么，我们可以得到相关性系数的公式为

   <img src="images/image-20210407093954658.png" alt="image-20210407093954658" style="zoom:29%;" />

   (3) 推导 LRN 层：对 $y_k$ 求偏导，
   $$
   \frac{\partial y_{k}}{\partial x_{j}}
   =
   \begin{cases}
   \frac{1}{\left ( 1+b\sum_{i=1}^{n}x_{i}^{2} \right )^{c}}-\frac{2bcx_{j}x_{k}}{\left ( 1+b \sum_{i=1}^{n}x_{i}^{2}\right )^{c+1}} & ,j=k
   \\
   -\frac{2bcx_{j}x_{k}}{\left ( 1+b \sum_{i=1}^{n}x_{i}^{2}\right )^{c+1}} & ,j\ne k
   \end{cases}
   $$
   ​	整理一下就可以得到原文公式

   <img src="images/image-20210407095219010.png" alt="image-20210407095219010" style="zoom:23%;" />

   (4) **特殊**泰勒展开点：（<u>我很困惑，为什么明知道这个展开点不好展开，还要在这个点去进行展开</u>）假设现在有实际输入 $z_{1}=\left ( x_{1},x_{2},...,x_{n} \right )$ 和 泰勒展开点 $z_{2}=\left ( 0,0,...,x_{k},...,0 \right )$，不难得到如下公式

   <img src="images/image-20210407103615172.png" alt="image-20210407103615172" style="zoom:19%;" />

   ​	可以看到，这里的泰特一阶展开式是无效的。所以，下面对其进行相应的修改

   <img src="images/image-20210407104106897.png" alt="image-20210407104106897" style="zoom:25%;" />

   ​	然后可以用 (2) 进行回传；

3. 实验：（只简单地看一下其中两个实现结果）

   (1) 验证得到的重要特征是否是有效的

   - 检验手段：模糊化目标像素点，观察模型置信度的变化；

   - 实验结果：文章的方法可以找到图像中的重要特征点；

     <img src="images/image-20210407191847142.png" alt="image-20210407191847142" style="zoom:35%;" />

   (2) 检验文章方法是否优于原有方法（原有方法直接将相关性权重整个回传给中心特征点 $x_k$）

   - 检验手段：同样模糊化重要特征，观察模型 `AUC` 的变化（越小越好）；

   - 实验结果：文章方法更优；

     <img src="images/image-20210407192122910.png" alt="image-20210407192122910" style="zoom: 40%;" />

### Links

- 论文链接：[Binder A, Montavon G, Lapuschkin S, et al. Layer-wise relevance propagation for neural networks with local renormalization layers[C]//International Conference on Artificial Neural Networks. Springer, Cham, 2016: 63-71.](https://arxiv.org/abs/1604.00825)
- 参考链接：[Tensorflow的LRN是怎么做的](https://www.jianshu.com/p/c06aea337d5d)
- 参考链接：[【阅读笔记】神经网络中的LRP及其在非线性神经网络中的运用](https://blog.csdn.net/ChangHao888/article/details/109320053)
- 参考链接：[Implementation of LRP for pytorch](https://github.com/fhvilshoj/TorchLRP)





## SmoothGrad: removing noise by adding noise

### Contributes

1. 在一定程度上去除 Grad 方法中的噪声;

### Notes

1. 已有的基于梯度的算法能够得到一个重要性关联的图, 但是带有很多的噪点, 即直接计算梯度, 会出现梯度饱和的问题 (在当前图中, 羊的特征对当前的分类概率影响不大, 因为它的分类结果可能已经是0.999了)

   <img src="images/image-20201210234923895.png" alt="image-20201210234923895" style="zoom: 22%;" />

2. 👍 作者提出的想法是添加扰动后再对图片求梯度, 然后将这些梯度求一个均值, 故论文名称为 Removing noise by adding noise, 求导的时候添加扰动, 是为了除去在重要性结果上的噪声:
   $$
   \hat{M}_c(x) = \frac{1}{n} \sum_1^n M_c(x+\mathcal{N}(0, \sigma^2))
   $$

3. Evaluation:

   (1) 模型+数据集: Inception v3 trained on ILSVRC-2013 dataset, convolutional MNIST model

   (2) 噪声大小的影响: 噪声过大的情况下也会使得效果变差, 从图中的效果来看, 添加 10% 的噪声效果是最好的

   <img src="images/image-20201211002446839.png" alt="image-20201211002446839" style="zoom: 50%;" />

   (3) 求梯度次数的影响: 从图中的效果来看, 计算的次数越多越好

   <img src="images/image-20201211002936728.png" alt="image-20201211002936728" style="zoom: 43%;" />

   (4) 和其他工作的对比: **作者在对比的过程中, 只是简单地抽取出几张图片对比了一下效果, 说明这个方向缺乏一个 ground truth 来检验各种可解释性方法的好坏**

   <img src="images/image-20201211003218352.png" alt="image-20201211003218352" style="zoom: 67%;" />

   (5) 最后作者在 MNIST 上面发现, 训练的时候就添加噪声进行训练, 也同样能够达到一定的 "**重要性降噪**" 的作用:

   <img src="images/image-20201211003543788.png" alt="image-20201211003543788" style="zoom: 33%;" />

### Links

- 论文链接: [Smilkov, Daniel, et al. "Smoothgrad: removing noise by adding noise." *ICML* (2017).](https://arxiv.org/abs/1706.03825)
- 论文主页: https://pair-code.github.io/saliency/
- Pytorch 实现: [pytorch-smoothgrad](https://github.com/hs2k/pytorch-smoothgrad)
- Tensorflow 实现 (论文源码): [saliency](https://github.com/PAIR-code/saliency)





## Visualizing and Understanding Neural Machine Translation

> 上过刘洋老师的 NLP 课后，一直很崇拜刘洋老师！:smiley::smiley::smiley: 

### Contribution

1. 👍 将 LRP（Layer-wise Relevance Propagation）方法应用到“基于注意力机制的机器翻译任务”中，用于**判断不同层不同神经元（或称为隐藏状态）对结果的贡献度**；

### Notes

> 原文在各个细节方面都讲地非常清楚，并且逻辑清晰、言语流畅，因此十分**推荐直接阅读原文**，下面只是对原文的一些摘要；

1. 基于注意力机制的机器翻译任务：

   <img src="images/image-20210331203551689.png" alt="image-20210331203551689" style="zoom: 45%;" />

   

2. LRP 算法的简单示例：（文章内容有些多，但其实如果你知道 LRP 算法是怎么做的话，这篇文章只是对这个方法的应用）

   以一个简单的**前向神经网络**举例，网络结构如下：

   <img src="images/image-20210331204046048.png" alt="image-20210331204046048" style="zoom: 25%;" />

   现在我们关注**输入层**和**隐藏层**各神经元对**输出神经元 $v_1$** 的影响，首先我们计算**隐藏层的影响因子**：

   <img src="images/image-20210331204505568.png" alt="image-20210331204505568" style="zoom: 16%;" />

   然后我们计算**输入层的影响因子**：

   <img src="images/image-20210331204653043.png" alt="image-20210331204653043" style="zoom: 28%;" />

3. LRP 算法定义：（从算法的定义上来看，**算法并不需要依赖于可导的梯度计算**）

   <img src="images/image-20210331204924929.png" alt="image-20210331204924929" style="zoom: 50%;" />

   其中，$v \in \text{OUT}(u)$ 指在神经网络中神经元 $u$ 指向神经元 $v$；$w_{u \rightarrow z}$ 的定义如下（**在 LRP 原文中，作者发现这个相关系数和网络的激活函数是无关的，所以下面均没有考虑神经元的激活函数**）

   - 如果是形如 $\bold{v} = \bold{Wu}$ 这样的**矩阵相乘**，则

   <img src="images/image-20210331205356249.png" alt="image-20210331205356249" style="zoom: 13%;" />

   - 如果是形如 $\bold{v} = \bold{u_1} \circ \bold{u_2}$ 的**向量点积**，则

   <img src="images/image-20210331205733116.png" alt="image-20210331205733116" style="zoom:11%;" />

   - 如果是形如 $v=\max{\{u_1, u_2 \}}$ 的**求最值**，则

   <img src="images/image-20210331210242737.png" alt="image-20210331210242737" style="zoom: 17%;" />

   其中，$u' \in \text{IN}(v)$ 指在神经元中神经元 $u'$ 指向神经元 $v$；

4. 实验：从实验的结果来看，**LRP 方法不仅能够得到和 Attention 可视化方法相似的结果，并且能够分析更多的隐藏层内容**。（正例是对正常翻译结果的分析，而反例则是机器翻译中常出现错误翻译问题的举例）

   - （正例）Source Side，即 Encoder 侧的影响因子可视化：

     <img src="images/image-20210331210628539.png" alt="image-20210331210628539" style="zoom:40%;" />

   - （正例）Target Side，即 Decoder 侧的印象因子可视化：

     <img src="images/image-20210331210847350.png" alt="image-20210331210847350" style="zoom:43%;" />

   - （反例）Word Omission，即过早结束翻译：

     <img src="images/image-20210331211453468.png" alt="image-20210331211453468" style="zoom:39%;" />

   - （反例）Word Repetition，即重复翻译；

     <img src="images/image-20210331211617034.png" alt="image-20210331211617034" style="zoom: 45%;" />

   - （反例）Unrelated Words，即翻译出了完全无关的词汇；

     <img src="images/image-20210331211746706.png" alt="image-20210331211746706" style="zoom:45%;" />

   - （反例）Negation Reversion，即翻译时把否定词遗漏了；（<u>文章的这个地方，我觉得是问题最大的，翻译过程中把“不”给遗漏了，不一定分析“about”这个词，作者对这个词的分析位置的选择，我觉得存粹是从中文的角度来看问题的，因为“谈 (talk)”后面跟的是“不”，他就来分析这个“about”</u>）

     <img src="images/image-20210331211924440.png" alt="image-20210331211924440" style="zoom: 45%;" />

   - （反例）Extra Words，即翻译中出现了多余的词；

     <img src="images/image-20210331212412328.png" alt="image-20210331212412328" style="zoom:45%;" />

### Links

- 论文链接：[Ding Y, Liu Y, Luan H, et al. Visualizing and understanding neural machine translation[C]//Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2017: 1150-1159.](https://www.aclweb.org/anthology/P17-1106/)





## SHAP - SHapley Additive exPlanation

> SHAP 提出的是一种特征重要性的度量方式，在具体的使用上它可以结合已有的可解释性方法（LIME、GRAD 等）一起运行，所以该部分不仅仅只关注其中某篇论文，而是关注 SHAP 这个方法；
>
> 因为 SHAP 实在是太难理解了，纯靠论文我实在是无法理解，所以下面我参考了大量网上的博客，如果有什么不清楚的地方，还是建议去读一读博客的原文和论文；

### Contribution



### Notes

#### SHAP 的基本运作原理

> 参考博客：[SHAP Values Explained Exactly How You Wished Someone Explained to You](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30)
>
> 中文翻译：[SHAP 值的解释，以一种你期望的方式解释给你听](https://www.6aiq.com/article/1588001724187) （<u>这个翻译某些部分有点糟糕，但大致能够看懂，建议直接看英文原文</u>）

##### Game Theory and Machine Learning

- Shapley Values，是一个博弈论的概念；

- 博弈论中，涉及到 “Game”（游戏） 和 “Players”（玩家） 两个概念。这两个概念和我们要解决的“**模型可解释性**”问题的对应关系为：

  - <font color=blue>Game</font> 对应的是 <font color=blue>模型的输出值</font>；（作者特别强调了，“**One Game one observation**"，“一场游戏 对应着 一个样本输入到模型中得到的输出值”）
  - <font color=red>Players</font> 对应的是 <font color=red>模型的特征</font>；

  ⭐故 **Shapley Values 和 SHAP 方法的对应关系**为：

  - Shapley Values 是用来度量每个 “Player” 对 “Game” 的贡献的；
  - SHAP 是用来度量每个特征对模型输出值的贡献的；

##### A Power Set of Features

- 作者以 “**用年龄（Age）、性别（Gender）和工作（Job）来预测一个人的收入**” 进行举例（SHAP 是一中黑盒的可解释性方法，故这里我们并不需要关心实际的模型是个线性模型、决策树模型还是其他更加复杂的模型）；

- Shapley Values 的基本思想：**Each possible combination (or coalition) of players should be considered to determine the importance of a single player**。<u>翻译到模型可解释性方向的话，就是**计算单个特征的重要性时，我们需要考虑全部特征的可能组合**</u>；

- Power Set（**幂集**）：即全部特征的、所有可能的组合，可以用树的形式来表示，如下所示；

  <img src="images/1_GOwxZ1ApAidTIDoa2l98ew.png" alt="1_GOwxZ1ApAidTIDoa2l98ew" style="zoom:80%;" />

  其中，每个节点代表一个 **特征联盟**（A coalition of features），每一条边表示（上面的特征联盟中）**不存在的特征**。$f$ 指的是某个特征联盟中特征的数量，$F$ 指的是全部特征的数量。

  ⭐**SHAP 在解释模型时，需要为每个特征联盟训练一个独立的预测模型**，即在这里，SHAP 需要训练 $2^F=8$ 个模型（这些模型的架构、超参数的设定和他们的训练数据都是一致的，唯一不同的是他们考虑的特征是不一样的）。作者举例，我们已经训练得到了 $8$ 个独立的预测模型，他们对某个输入 $x_0$ 在模型中的输出值都做了相应的预测，如下图所示：

  <img src="images/1_9oHGJ9kAMaUUEYtnfKKKEA.png" alt="1_9oHGJ9kAMaUUEYtnfKKKEA" style="zoom:80%;" />

  :warning: 一个可能存在的误区，先列出我们存在的两种不同的观点：

  - <font color=red>（错误）对于每个节点，采用的是原始的我们要解释的模型，而不需要对每个节点训练一个模型</font>；
  - <font color=green>（正确）对于每个节点，需要用相同的方法训练一个独立的模型</font>；

  这里正确的是第二个观点，之所以会产生第一种错误的观点，是把 $Predict_{\varnothing}$ 当成了把所有特征取 $0$，而正确的理解应该是 $Predict_{\varnothing}$ 指的是每个特征都可以取任何值；

##### Marginal Contributions of a Feature

- 如上所述，**两个相连的节点之间只有一个特征的差异**，（前面还指出每个特征联盟的预测模型都使用的相同参数等），故**两个相连的模型的预测值之间的差异也主要是这个差异特征导致的**；我们把这种预测值之间的差异称为 <font color=red>**该差异特征**的 “Marginal Contribution”（边缘贡献）</font>；为了防止我理解的有偏差，这里给出原文截图：

  <img src="images/image-20210521171044218.png" alt="image-20210521171044218" style="zoom:33%;" />

- 举个例子，对于节点 ①，这个模型它不考虑任何特征，所以这个模型给出的就是（训练集的）平均收入水平 $50k\ \$$；再看节点 ②，这个模型它只考虑样本 $x_0$ 的年龄（Age）特征，它给出的预测结果是 $40k\ \$$；这说明，当我们考虑了 $x_0$ 的年龄特征后，模型的预测值下降了 $10k\ \$$；我们把这种差异称为 年龄（Age）的 边缘贡献（之一），用公式表示如下：
  $$
  MC_{Age,\{Age\}} = Predict_{\{Age\}}(x_0) - Predict_{\varnothing}(x_0) = 40k\ \$ - 50k\ \$ = -10k\ \$
  $$
  下图中画出了年龄（Age） 的全部边缘贡献：

  <img src="images/1_7keBLSQszepu5jITz8SKxA.png" alt="1_7keBLSQszepu5jITz8SKxA" style="zoom:80%;" />

  最终，SHAP 对这些边缘贡献进行 **加权求和**（权重的大小由下一小节解释）：
  $$
  \begin{align}
  SHAP_{Age}(x_0) & =\ w_1 \times MC_{Age,\{Age \}}(x_0) \\
  &\ + w_2 \times MC_{Age,\{Age, Gender\}}(x_0) \\
  &\ + w_3 \times MC_{Age, \{Age, Job\}}(x_0) \\
  &\ + w_4 \times MC_{Age, \{Age, Gender, Job\}}(x_0)\\
  \end{align} \\
  \text{where }\ w_1+w_2+w_3+w_4=1
  $$

##### Weighting the Marginal Contributions

- 以上面的为例，我们直接给出每个边缘贡献的权重，如下图所示：

  ![1_wOLS6BO2vGTSB4sCoAb6fQ](images/1_wOLS6BO2vGTSB4sCoAb6fQ.png)

  简单来说，每个边缘贡献的权重 等于 每层边缘贡献数量的倒数；

- **数学公式对上面的规律进行表示**：
  $$
  weight = f \times \dbinom{F}{f}
  $$

- 现在，我们可以求出最终的 SHAP 值：
  $$
  \begin{align}
  SHAP_{Age}(x_0) & =\ [1 \times\dbinom{3}{1}]^{-1} \times MC_{Age,\{Age \}}(x_0) \\
  &\ + [2 \times\dbinom{3}{2}]^{-1} \times MC_{Age,\{Age, Gender\}}(x_0) \\
  &\ + [2 \times\dbinom{3}{2}]^{-1} \times MC_{Age, \{Age, Job\}}(x_0) \\
  &\ + [3 \times\dbinom{3}{3}]^{-1} \times MC_{Age, \{Age, Gender, Job\}}(x_0)\\
  &\ = \frac{1}{3} \times(-10k\ $) + \frac{1}{6} \times (-9k \ $) + \frac{1}{6} \times(-15k \ $) + \frac{1}{3} \times(-12k \ $) \\
  &\ = -11.33k \ $
  \end{align}
  $$

##### Wrapping it up

- 最后，我们就得到了 **SHAP 文章中的公式**（<u>现在就很好理解这个公式了</u>）：
  $$
  SHAP_{feature}(x) = \sum_{set:\ feature\ \in\ set}[|set| \times \dbinom{F}{|set|}]^{-1}[Predict_{set}(x) - Predict_{set \setminus feature}(x)]
  $$

- 相应的，我们可以计算得到各个特征的 SHAP 值：
  $$
  SHAP_{Age}(x_0) = -11.33k \ \$ \\
  SHAP_{Gender}(x_0) = -2.33k \ \$ \\
  SHAP_{Job}(x_0) = +46.66k \ \$
  $$
  可以看到，把这些 SHAP 值全部加起来，正好等于 the full model（最下面的考虑全部特征的模型）减去 the null model（最上面的不考虑任何特征的模型）的预测值的差；这也正好是 **SHAP 方法的一个特点**，也是它取名的原由，为了不造成更多歧义，直接粘原文：

  <img src="images/image-20210522122406221.png" alt="image-20210522122406221" style="zoom:33%;" />

- 这里，虽然我们似乎已经做完了 SHAP 的全部步骤，但是在实际应用中，我们考虑的特征是非常多的，可以看到我们上面还要分别训练 $2^F$ 个模型，所以接下来的任务，就是**如何在拥有大量特征的情况下，用近似的方法来求解 SHAP 值**，论文原文的话如下所示；

  <img src="images/image-20210522134030671.png" alt="image-20210522134030671" style="zoom: 45%;" />

#### Kernel SHAP（Linear LIME + Shapley Values）

> 参考博客：[Complete SHAP tutorial for model explanation Part 3. KernelSHAP](https://summer-hu-92978.medium.com/complete-shap-tutorial-for-model-explanation-part-3-kernelshap-4859186dc8be)（建议优先看这篇博客）
>
> 参考博客：[5.10.2 KernelSHAP](https://christophm.github.io/interpretable-ml-book/shap.html#kernelshap)
>
> 第二篇博客的中文翻译：[模型解释–SHAP Value的简单介绍](https://mathpretty.com/10699.html)（<u>一样，中文翻译一般都挺坑的</u>）

- 符号表示

  为了让下面的内容更容易看懂，我们得先解释一下一些符号的含义：

  - $x$ 表示模型的输入，$f(x)$ 表示模型的输出；

  - $z'$ 表示简化后的**特征域**（如：图像中我们会把模型的输入 $x$ 切成小的超像素块域 $z'$），$M$ 表示特征的数量，$h_x(z')$ 表示将特征域 $z'$ 映射到原输入域 $x$（**注意：这个映射是一个 one-to-many mapping**）；下面展示论文中的描述：

    <img src="images/image-20210522144253177.png" alt="image-20210522144253177" style="zoom: 43%;" />

  - $g$ 表示目标模型的替代解释模型；

  - $\pi_x(z')$ 表示为样本 $z'$ 的权重；

- LIME 可解释方法

  LIME 的目标是最小化如下**目标函数**：
  $$
  \xi = \mathop{\arg\min}\limits_{g\in\mathcal{G}} L(f,g,\pi_x(z')) + \Omega(g)
  $$
  即，LIME 希望最终**得到的线性回归模型在采样点 $z'$ 上的加权误差尽可能的小**，并且**根据样本的修改程度来确定不同样本的权重** $\pi_x(z')$，使用 $\Omega(g)$ 来**限制回归模型的复杂度**；

- ⭐ **Kernel SHAP 运行步骤**

  1. 采样特征域样本 $z'$ ，一般地，我们需要采样 $2^M$ 个特征域样本；

  2. 计算模型在采样样本上的预测结果 $f(h_x(z'))$；

     因为上面提到，映射变换是一个一对多的变换，所以这里 $f(h_x(z'))=E[f(h(z'))]$；（:question: ​<u>是否会为了简化计算，把这个一对多的映射变成一个一对一的映射？这种映射可能还存在一个问题，就是说它的前提是每个特征需要相互独立的，但是在实际的数据分布中，并不是这样的；</u>）

  3. 用如下 **SHAP Kernel** 计算每个特征域样本的权重：

     <img src="images/1_8QAcN7ZQeMo6tXps-UoYEw.png" alt="1_8QAcN7ZQeMo6tXps-UoYEw" style="zoom: 60%;" />

  4. 借助 LIME，构造**线性回归模型**：

     <img src="images/1_Bas1PB1Dxh5XmYJfIvR6cw.png" alt="1_Bas1PB1Dxh5XmYJfIvR6cw" style="zoom: 63%;" />

     去除原目标函数中的复杂度约束项 $\Omega(g)$，并将新的 SHAP Kernel 代入公式中，得到 **新的目标函数**：
     $$
     \xi = \mathop{\arg\min}\limits_{g\in\mathcal{G}} L(f,g,\pi_x(z')) \\
     L(f,g,\pi_x(z')) = \sum_{z' \in Z} \left[f(h_x(z')) - g(z') \right]\cdot \pi_x(z')
     $$

  5. 拟合线性回归模型，得到的 $\phi_j$ 即为 **Shapley Values 的估计值**；（<u>论文中证明了，这种方法可以保证得到 Shapley Values 的估计值，具体的证明细节就不再探讨</u>）

- 理解 SHAP Kernel

  为了理解这个核函数，博客作者首先画了一下它的取值，如下图所示：

  ![1_TqxlGgfg7Zx4K5CW3fvvjw](images/1_TqxlGgfg7Zx4K5CW3fvvjw.png)

  可以看到，联盟中的特征很多或者很少时，联盟的权重相对更高；这造成的影响是，最后的回归模型对于小联盟或者大联盟的拟合想过更好；给出这一段的博客原文描述：

  <img src="images/image-20210522152922044.png" alt="image-20210522152922044" style="zoom: 33%;" />

- 理解 Kernel SHAP 方法（<u>是我个人的理解，谨慎参考</u>）

  1. <u>Kernel SHAP 可解释方法，可以简单地认为是 LIME 方法的一个变种，因为我们是在 LIME 方法上对目标函数和样本权重公式做了相应的修改，最后还是拟合了一个线性回归模型；</u>
  2. <u>但是，我认为这种方法可能并没有 LIME 方法来的好，因为**我可能更希望通过样本间的距离来赋值我的样本权重**；</u>
  3. <u>Kernel SHAP 运行还是比较耗时，另外 Tree SHAP 在效率上得到了提高；</u>
  4. :warning: <u>再讨论前面提到的是否训练了一个独立模型的问题，这边我觉得：如果这里采用的是一对多的映射函数的话，可以认为它训练了一个独立的模型（求期望）；如果这里采用的是一对一的映射函数的话，应该是没有训练独立的模型的；</u>

#### Tree SHAP



### Links

- 论文链接：[Lundberg S, Lee S I. A unified approach to interpreting model predictions[J]. arXiv preprint arXiv:1705.07874, 2017.](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- 论文链接：[Consistent Individualized Feature Attribution for Tree Ensembles](Lundberg S M, Erion G G, Lee S I. Consistent individualized feature attribution for tree ensembles[J]. arXiv preprint arXiv:1802.03888, 2018.)
- 论文代码：[slundberg/shap: A game theoretic approach to explain the output of any machine learning model. (github.com)](https://github.com/slundberg/shap)
- API 文档：https://shap.readthedocs.io/en/latest/
- SHAP 简单入门博客：[SHAP：Python的可解释机器学习库](https://zhuanlan.zhihu.com/p/83412330)





## LEMNA: Explaining Deep Learning based Security Applications

> 这篇文章，实际上我看了两遍：第一遍研一的时候看的，是文献阅读课，讲得很差，被老板批评了；第二遍研二项目中需要用到一系列的可解释性方法，和大佬一起又重新拜读了一下这篇文章，这次是结合代码来学习的，所以对 LEMNA 这种方法算是又有了新的认识；

### Contribution

1. LEMNA (Local Explanation Method using Nonlinear Approximation)，一种新的黑盒解释方法；
2. Mixture Regression Model 拟合一个局部非线性（分段线性）模型；（<u>用我的话来说，其实是用高斯混合分布筛选扰动数据集</u>）
3. Fused Lasso 限制的临近特征权重之间的差异；

### Notes

1. 先从作者总结的表格来看一下这些已有的可解释性方法：

   ![image-20210508001350035](images/image-20210508001350035.png)

   这里简单将现有的工作分为三类：

   - White-box method (forward)：通过修改/去除前向传播的一些特征，观察模型输出的变化大小，来确定特征的重要性；
   - White-box method (backward)：通过比较梯度的反向传播值的大小、正负，来确定特征的重要性；
   - Black-box method：通过拟合模型的边界来确定特征的重要性；（<u>LEMNA 同样属于这一类</u>）

   同时，作者也列出了一些理由，为什么不选择已有的这些工作：

   - 是否支持 RNN/MLP，像 CAM 家族的工作都是依赖于卷积层的位置对应关系才得以将热力图回传的，所以这类工作不支持非 CNN 网络；
   - 边界是否为非线性的；（<u>挺好奇的是，为什么作者这里认为白盒的方法不支持非线性？</u>）
   - 方法是否支持黑盒（<u>这一点，我认为是比较 Weak</u>）；

2. LEMNA 的方法设计：主要分为两块，一个是 Fused Lasso，一个是 Mixture Regression Model；

   - Fused Lasso：作者的解释是，为了使得 LEMNA 方法将相关的（文中用的 relevant）或者是靠近的（文中用的 adjacent）特征关联起来，来生成解释。<u>换成我的话说，就是让相邻的（和 relevant 没有一点关系）特征的权重尽可能得相近</u>，看公式便一清二楚；

     在 LIME 等前面的可解释性方法中，是使用最大似然估计 MLE 的方法来是如下的损失函数最小化：

     <img src="images/image-20210509170637459.png" alt="image-20210509170637459" style="zoom: 12%;" />

     LEMNA 使用 Fused Lasso 方法后，其目标就变成了在一定限制下使得损失函数最小化：

     <img src="images/image-20210509171347666.png" alt="image-20210509171347666" style="zoom: 14%;" />

     作者也给出了，基于 Fused Lasso 方法他希望得到的最终的可解释性形式：

     <img src="images/image-20210509171600046.png" alt="image-20210509171600046" style="zoom:30%;" />

   - Mixture Regression Model：作者的解释是，一个混合回归模型能够拟合一个局部非线性边界。换成我的话说，就是拟合一个局部分段线性模型。从作者的图里面比较容易理解：

     <img src="images/image-20210509172134945.png" alt="image-20210509172134945" style="zoom:33%;" />

     可以看到，LIME 拟合的是一个局部线性模型，LEMNA 拟合的是一个分段线性模型，<u>从示意图上可以感觉到用 **分段线性模型** 是更好的</u>；下面给出这个混合回归模型的公式：

     <img src="images/image-20210509172341626.png" alt="image-20210509172341626" style="zoom:8%;" />

     使用混合回归模型的目的，就是要找到上图中红色的这个线性模型，因为它对 $x$ 处的拟合效果是最好的；

     <img src="images/image-20210509172642015.png" alt="image-20210509172642015" style="zoom: 30%;" />

3. LEMNA 的具体实现：<u>这部分我们结合代码一起来看会比较有感觉</u>；

   具体的代码实现，在 GitHub 上有三个相关的库：

   - Henrygwb 的 [Explaining-DL](https://github.com/Henrygwb/Explaining-DL) 库（<u>大佬好像是跟我说这个是作者后来的工作</u>）：这个库中的 LEMNA 实现只有 Fused Lasso 部分（<u>我们暂时用的是这个库</u>）；
   - nitishabharathi 的 [LEMNA](https://github.com/nitishabharathi/LEMNA/blob/master/lemna.py) 库：这个库中的 LEMNA 实现只有 Mixture Regression Model 部分；
   - ZhihaoMeng 的 [CPSC8580](https://github.com/ZhihaoMeng/CPSC8580) 库：这个库中的 LEMNA 实现很 nice，非常全，强烈建议看这个库；（<u>虽然我还没运行过他的代码，但是我简单看了一下，实现上来说和第一个库差不多，所以花点功夫应该是可以跑起来的，有时间我也会跑一下，甚至替换到我们的项目中</u>）

   代码的实现过程中，Fused Lasso 和 Mixture Regression Model 是几乎混在一起的，下面还是简单地拆开来分析；

   - Fused Lasso：调用 R 语言的实现，下面简单列一下相关的代码；

   ```python
   # 导入rpy的包等等操作
   from scipy import io
   from rpy2 import robjects
   from rpy2.robjects.packages import importr
   r = robjects.r
   rpy2.robjects.numpy2ri.activate()
   # 导入r的包
   importr('genlasso')
   importr('gsubfn')
   # 使用r来实现Fused Lasso
   X = r.matrix(data_explain)
   Y = r.matrix(lable_explain)
   results = r.fusedlasso1d(y=Y, X=X)
   ```

   - Mixture Regression Model：（<u>希望看官不要嫌弃我的数学</u>）对于原问题，最小化如下损失函数

     <img src="images/image-20210509175725889.png" alt="image-20210509175725889" style="zoom: 19%;" />

     作者的实现**依赖于“单个线性模型的预测结果的误差 满足 高斯分布”这个假设**。我们将 $P(y_i|x_i)$ 写成高斯混合分布的形式：

     <img src="images/image-20210509180210672.png" alt="image-20210509180210672" style="zoom:11%;" />

     这样就通过**拟合高斯混合分布的方法**来计算模型的参数，即 EM 算法；但是因为我们需要结合 Fused Lasso 方法，对 EM 算法还需要做一些修改：

     - E Step：和原来的 EM 步骤一样，估计各个样本属于哪个子分布，计算期望值；

     - M Step：对于参数 $\pi_k$ 和 $\sigma_k$ 的更新和原来的 EM 步骤是一样的，但是对于 $\beta_k$ 的更新，改成使用 Fused Lasso 方法来对其进行更新；即已经用 E Step 得到了每个样本的子分布，在 M Step 中直接调用 Fused Lasso 方法对每个分布的样本进行拟合，下面给出作者的表达

       <img src="images/image-20210509181202246.png" alt="image-20210509181202246" style="zoom:30%;" />

     这里代码就不粘了，得说一下 ZhihaoMeng 里面三个脚本的区别：

     - [lemna.py](https://github.com/ZhihaoMeng/CPSC8580/blob/master/MLM/lemna_O1/lemna.py)：（<u>算是一个阉割版</u>）直接用高斯混合分布对扰动数据集（Perturbation Samples）进行拟合，挑选出属于最优（$\pi_k$ 最大）的子高斯分布的样本点，用 Fused Lasso 对这些点进行拟合；
     - [lemna-no-GMM.py](https://github.com/ZhihaoMeng/CPSC8580/blob/master/MLM/lemna_O1/lemna-no-GMM.py)：（<u>算是一个精简版</u>）只用了 Fused Lasso 方法，这个和 Henrygwb 的库的实现是一样的；
     - [lemna-customGMM.py](https://github.com/ZhihaoMeng/CPSC8580/blob/master/MLM/lemna_O1/lemna-customGMM.py)：这是完整的 LEMNA 实现；

   LEMNA 方法最后给出的特征重要性即为 $\pi_k$ 最大的线性分布模型的权重 $\beta_k$，下面给出作者的表达：

   <img src="images/image-20210509183138541.png" alt="image-20210509183138541" style="zoom:30%;" />

4. 我的理解：

   - 为啥不用白盒梯度的方法呢：折腾来折腾去，就是拟合一个边界，感觉黑盒方法的优势并不存在；
   - 筛选扰动数据集：黑盒方法的好坏很大层面上取决于你构造的扰动数据集的好坏；我们换个角度来看，LEMNA 算法最终得到的还是一个局部线性模型（混合回归模型就是多拟合几条线，我们来选一条最好的），他其实不就是用高斯混合模型对扰动数据集做了一个筛选么！
   - 黑盒算法的性能受到特征维度的制约；
   - 实现和理论不近相符：前面图片中我们想要得到的是红色的线性模型，即经过样本点的线性模型，但是实现过程中挑选最优的模型，则是判断混合分布的 $\pi_k$ 的大小；
   - 作者在实现过程中还有一个 Trick，他在 Binary 这个模型的分析过程中，设置了一个大小为 40 的分析窗口；

5. 论文实验：

   - 实验针对的模型：

     <img src="images/image-20210509183335534.png" alt="image-20210509183335534" style="zoom:30%;" />

     - Binary Function Start：标记二进制代码函数起始位置的模型，用的 RNN 模型，其中 $O0\sim O3$ 表示 gcc 的编译优化等级；
     - PDF Malware：PDF 恶意脚本检测的模型，人工提取出相应的特征，用的 MLP 模型，因为相邻特征之间没有绝对的相关性，所以这里没有用到 Fused Lasso；

   - Baseline：和 LIME 方法和 Random 方法进行对比；

   - 如何度量精度（Fidelity）

     - Local Approximation Accuracy：即计算线性模型预测的概率和神经网络预测的概率之间的差别，用 RMSE 度量，公式如下

       <img src="images/image-20210509184040483.png" alt="image-20210509184040483" style="zoom:13%;" />

     - End-to-end Fidelity Tests：主要思想是保留、删除、生成重要特征，观察模型结果的变化，都用 PCR 来度量

       - Feature Deduction Test：在原始数据中删除重要特征，PCR 越小说明解释方法越好；
       - Feature Augmentation Test：挑选一个相反类（opposite class）数据，在数据中增加（从原始数据中分析得到的）重要特征，PCR 越大说明解释方法越好；
       - Synthetic Test：在空白数据中只添加（从原始数据中分析得到的）重要特征，PRC 越大说明解释方法越好；

   - 实验结果：

     - Local Approximation Accuracy：LEMNA 方法的 RMSE 结果优于 LIME；

       <img src="images/image-20210509184737868.png" alt="image-20210509184737868" style="zoom:33%;" />

     - End-to-end Fidelity Tests

       - Feature Deduction Test

         <img src="images/image-20210509184907443.png" alt="image-20210509184907443" style="zoom: 40%;" />

       - Feature Augmentation Test

         <img src="images/image-20210509184944966.png" alt="image-20210509184944966" style="zoom:40%;" />

       - Synthetic Test

         <img src="images/image-20210509185031068.png" alt="image-20210509185031068" style="zoom:40%;" />

       - 分析：LEMNA 方法均优于 LIME，并且发现对于 PDF Malware 这个网络，LIME 方法的效果和 Random 差不多；

         <img src="images/image-20210509185328932.png" alt="image-20210509185328932" style="zoom:38%;" />

   - 如何利用可解释方法：作者最后给出了几种有意思的可解决方法的利用场景，有兴趣可以再看看；

### Links

- 论文链接：[Guo W, Mu D, Xu J, et al. Lemna: Explaining deep learning based security applications[C]//proceedings of the 2018 ACM SIGSAC conference on computer and communications security. 2018: 364-379.](https://dl.acm.org/doi/10.1145/3243734.3243792)
- 代码链接：[Black-box explanation of deep learning models using mixture regression models](https://github.com/Henrygwb/Explaining-DL)
- 代码链接：https://github.com/nitishabharathi/LEMNA
- 代码链接：https://github.com/ZhihaoMeng/CPSC8580





## 机器学习模型可解释性方法、应用与安全研究综述

### Notes

> 这篇是纪老师关于模型可解释性文章的综述，以下均为个人对文章做的笔记，建议自己阅读原文。另外纪老师在这方面有很多不错的工作，值得关注。
>
> 模型可解释性方向有特别多的工作，文章中也有一些不全的地方需要自己去思考和补充。

#### 机器学习可解释性问题

1. **模型可解释性问题**：可解释性旨在帮助人们理解机器学习模型是如何学习的，它从数据中学到了什么，针对每一个输入它为什么做出如此决策以及它所做的决策是否可靠；
2. **模型的复杂度与模型准确性相关联，又与模型的可解释性相对立**。
3. 根据**选择结构简单易于解释的模型然后训练它**，还是**训练复杂的最优模型然后开发可解释性技术解释它**，将机器学习模型可解释性总体上分为：**ante-hoc 可解释性**和 **post-hoc 可解释性**；

#### * Ante-hoc 可解释性

1. Ante-hoc 可解释性指**模型本身内置可解释性**，即对于一个已训练好的学习模型，我们无需额外的信息就可以理解模型的决策过程或决策依据；

2. 自解释模型：

   (1) 可模拟性：在一定时间可以预测模型的每一步计算；

   (2) 可分解性：模型的每个部分都可以得到一个直观的解释；

   (3) 结构简单：由于人类认知的局限性，自解释模型的内置可解释性受模型的复杂度制约，这要求自解释模型结构一定不能过于复杂；

3. 广义加性模型：在简单模型和复杂问题之间的一个折中；
   $$
   g(y) = f_1(x_1) + f_2(x_2) + \cdots + f_n(x_n)
   $$

4. 注意力机制：通过 Attention 的权重来分析模型关注的重点是什么；

#### Post-hoc 可解释性

1. Post-hoc 可解释性的重点在于设计高保真的解释方法或构建高精度的解释模型，根据解释目的和解释对象的不同，可分为全局可解释性和局部可解释性；

2. 经典的解释方法如下：

   <img src="images/image-20201208125757884.png" alt="image-20201208125757884" style="zoom: 60%;" />

3. 全局解释：全局可解释性旨在帮助人们从整体上理解模型背后的复杂逻辑以及内部的工作机制，如模型是如何学习的、模型从训练数据中学到了什么、模型是如何进行决策的 等；

   (1) * 规则提取：通过受训模型中提取解释规则的方式，提供对复杂模型尤其是黑盒模型整体决策逻辑的理解（比较早期的做法）；

   (2) 模型蒸馏：

   - 定义：通过**降低模型复杂度**，解决理解受训模型比较困难的问题，是一种经典的**模型压缩方法**；

   - 核心思想：利用**结构紧凑的学生模型来模拟结构复杂的教师模型**，从而完成从教师模型到学生模型的知识迁移过程，实现对复杂教师模型的知识的 ”蒸馏“；

   - 训练损失函数定义：
     $$
     L_{student}=\alpha L^{(soft)} + (1-\alpha)L^{(hard)}
     $$
     其中，$L^{(soft)}$ 为**软目标损失**，期望学生模型能够学到教师模型相似的概率分布输出；$L^{(hard)}$ 为**硬目标损失**，要求学生模型能够保留教师模型决策的类别；

   - 模型蒸馏解释方法实现简单，易于理解，且不依赖待解释模型的具体结构信息，因而作为一种模型无关的解释方法，常被用于解释黑盒机器学习模型；

   - 蒸馏模型只是对原始复杂模型的一种全局近似，基于蒸馏模型所做出的解释不一定能够反映待解释模型的真实形为；

   > 我的想法：1. 能否通过模型蒸馏的方法去生成一个黑盒模型的替代模型，从而去辅助生成对抗样本？（这个关键在于生成的替代模型能否较好地逼近黑盒模型）2. 模型蒸馏的方法，最终能够得到怎样的模型解释？能否通过模型蒸馏的方法来解释现有的语音识别模型？（这个关键在于如何去分析蒸馏以后的模型）

   (3) 激活最大化 (Activation Maximization)：

   - 定义：通过在特定的层上**找到神经元的首选输入来最大化神经元激活**，来理解 DNN 中每一层隐含层神经元所捕获的表征；

   - 核心思想：通过寻找有界范数的输入模式，最大限度地激活给定地隐藏神经元，而一个单元最大限度地响应的输入模式可能是 ”一个单元正在做什么的“ 良好的一阶表示；

   - 形式化定义：
     $$
     x^* = \arg \mathop{\max}\limits_{x} f_l(x)-\lambda \lVert x \rVert ^2
     $$
     其中左边项期望 $x$ 能够使得当前神经元的激活值最大；右边项期望 $x$ 与原样本尽可能接近（<u>右边应该改用 $\Delta x$ 来表示</u>）；

   - 激活最大化解释方法是一种模型相关的解释方法，相比规则提取解释和模型蒸馏解释，其解释结果更准确，更能反映待解释模型的真实形为；

   - 激活最大化本身是一个优化问题，在通过激活最大化寻找原型样本的过程中，优化过程中的噪音和不确定性可能导致产生的原型样本难以解释；

   - 激活最大化解释方法**难以用于解释自然语言处理模型和图神经网络模型**；

4. 局部解释：模型的局部可解释性以输入样本为导向，通常可以通过分析输入样本的每一维特征对模型最终决策结果的贡献来实现。

   (1) 敏感性分析 (Sensitivity Analysis)：

   - 核心思想：通过**逐一改变自变量的值来解释因变量受自变量变化影响大小**的规律；

   - 根据**是否需要利用模型的梯度信息**，敏感性分析方法可分为模型相关方法和模型无关方法；

   - **模型相关方法**：利用模型的局部梯度信息评估特征与决策结果的相关性，常见的相关性定义如下：
     $$
     R_i(x) = (\frac{\partial f(x)}{\partial x_i})^2
     $$
     即为模型梯度的 $l_2$ 范数分解；

   - 模型无关方法：无需利用模型的梯度信息，只关注**待解释样本特征值变化对模型最终决策结果的影响**。具体地，该方法通过观察去掉某一特定属性前后模型预测结果的变化来确定该属性对预测结果的重要性，即：
     $$
     R_i(x) = f(x) - f(x \setminus x_i)
     $$

   - 敏感性分析方法提供的解释结果通常**相对粗糙且难以理解**，且无法分析**多个特征之间的相关关系**；

   (2) 局部近似：

   - 核心思想：利用**结构简单的可解释模型拟合待解释模型针对某一输入实例的决策结果**，然后基于解释模型对该决策结果进行解释；
   - 基于局部近似的解释方法实现简单，易于理解且不依赖待解释模型的具体结构，适于**解释黑盒机器学习模型**；
   - 对于每个输入样本都需要训练一个解释模型，**效率不高**，并且**该方法基于特征相互独立的假设**；

   (3) ⭐ 反向传播：

   - 核心思想：利用 **DNN 的反向传播机制**将模型的决策重要性信号从模型的输出层神经元逐层传播到模型的输入以推导输入样本的特征重要性；
   - 其中，**SmoothGrad 方法**的核心思想：通过向待解释样本中添加噪声对相似的样本进行采样，然后利用反向传播方法求解每个采样样本的决策显著图，最后将所有求解得到的显著图进行平均并将其作为对模型针对该样本的决策结果的解释；
   - **分层相关性传播(LRP)方法**的核心思想：利用反向传播将高层的相关性分值递归地传播到底层直至传播到输入层；
   - 基于反向传播地解释方法通常实现**简单、计算效率高且充分利用了模型的结构特性**；
   - 如果预测函数在输入附近变得平坦，那么预测函数相对于输入的梯度在该输入附近将变得很小，进而导致无法利用梯度信息定位样本的决策特征；

   > 反向传播的想法应该可以在语音领域进行尝试。

   (4) * 特征反演 (Feature Inversion)：

   - 定义：特征反演作为一种可视化和理解 DNN 中间特征表征的技术，可以充分利用模型的中间层信息，以提供对模型整体行为及模型决策结果的解释；
   - 特征反演解释方法分为：模型级解释方法和实例级解释方法；

   (5) 类激活映射：

   <img src="images/image-20201208195015000.png" alt="image-20201208195015000" style="zoom: 25%;" />

   - ⭐ 前提：**CNN 不同层次的卷积单元包含大量的位置信息，使其具有良好的定位能力**。然而，传统 CNN 模型通常在卷积和池化之后采用全连接层对卷积层提取的特征图进行组合用于最终的决策，因而**导致网络的定位能力丧失**，这个问题需要在这些工作中解决。
   - **类激活映射 (Class Activation Mapping, CAM) 解释方法**：利用全局平均池化(Global Average Pooling)层来替代传统 CNN 模型中除 softmax 层以外的所有全连接层，并通过将输出层的权重投影到卷积特征图来识别图像中的重要区域 => (需要**用全局平均池化层替换模型中的全连接层并重新训练**)。
   - **梯度加权类激活映射 (Grad-CAM) 解释方法**：给定一个输入样本，Grad-CAM 首先计算目标类别相对于最后一个卷积层中每一个特征图的梯度并对梯度进行全局平均池化，以获得每个特征图的重要性权重；然后，基于重要性权重计算特征图的加权激活，以获得一个粗粒度的梯度加权类激活图，用于定位输入样本中具有类判别性的重要区域 => (**不需要修改网络后进行重训练**)；
   - **导向梯度加权类激活 (Guided Grad-CAM) 解释方法**：即将 GuidedBP 方法和 Grad-CAM 方法进行结合；

#### 可解释性应用

1. 模型验证：由于数据集中可能存在偏差，并且验证集也可能与训练集同分布，我们很难简单地通过评估模型在验证集上的泛化能力来验证模型的可靠性，也很难验证模型是否从训练数据中学到了真正的决策只是。这里**通过可解释性方法，去分析模型在做决策时到底更加关注哪些特征，这些特征是否是合理的**。如下：

<img src="images/image-20201208201334366.png" alt="image-20201208201334366" style="zoom: 30%;" />

2. \* 模型诊断：诊断模型中的缺陷；

   > 这个东西感觉不太靠谱，模型可解释性的方法很大程度上并不能得到很直观的一个解释，又如何依靠模型可解释性来做模型的诊断。

3. \* 辅助分析：如医疗领域，使用可解释性方法来辅助医护人员进行检查；

   <img src="images/image-20201208212314074.png" alt="image-20201208212314074" style="zoom: 22%;" />

4. 知识发现：辅助人学习基于大量数据训练的模型中的知识；

   - **LEMNA** 解释方法可以挖掘出检测模型从数据中学到的新知识；

#### 可解释性与安全性分析

> 应该仔细思考：**可解释性方法和安全领域的相关关系，从而去发掘新的应用场景、攻击场景、防御手段等**。

1. 安全隐患消除：模型可解释性方法用于检测对抗样本，并增强模型的鲁棒性；

2. 安全隐患：**攻击者也同样可以利用模型可解释性的方法来生成更好的对抗样本**；

3. ⭐ 自身安全问题：由于采用了近似处理或是基于优化手段，大多数解释方法只能提供近似的解释，因而解释结果与模型的真是形为之间存在一定的不一致性

   - 不改变模型的决策结果的前提下，使解释方法解释出错：
     $$
     \arg \mathop{\max}_\delta D(I(x_t;N), I(x_t+\delta;N))\\
     s.t. \lVert\delta\rVert_\infty \leq \epsilon , f(x_t+\delta) = f(x_t)
     $$
     其中，$I(x_t;N)$ 为解释系统对神经网络 $N$ 针对样本 $x_t$ 决策结果 $f(x_t)$ 的解释。

   - 模型结果出错，单不改变解释方法解释出错；

#### 未来方向

1. ⭐如何设计更精确、更友好的解释方法，消除解释结果与模型真实形为之间的不一致；
2. ⭐如何设计更科学、更统一的可解释性评估指标，以评估可解释方法解释性能和安全性；

> <u>探讨一下</u>：（我）老板问的问题是：“模型可解释性这个东西做出来，你希望它能有怎样的效果？我们要用这个要求去检验模型可解释性”；老板第二句话是：“我们应该是去做一些开创性的东西，而不是说，在别人的方法上面修修补补，或者是迁移到另一个领域，然后发一篇文章就了事了”。最后一句：“这个评估是这个领域亟待解决的问题，你能不能提出一个通用、合理的方法来评估这样方法，如果不行的话，针对不同的分类，我们能不能来做评估”。

### Links

- 论文链接：[纪守领, et al. "机器学习模型可解释性方法, 应用与安全研究综述." *计算机研究与发展* 56.10 (2019).](https://nesa.zju.edu.cn/download/%E6%A8%A1%E5%9E%8B%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%80%A7%E5%85%B3%E9%94%AE%E6%8A%80%E6%9C%AF%E3%80%81%E5%BA%94%E7%94%A8%E5%8F%8A%E5%85%B6%E5%AE%89%E5%85%A8%E6%80%A7%E7%A0%94%E7%A9%B6%E7%BB%BC%E8%BF%B0.pdf)





## Layer-Wise Relevance Propagation: An Overview

### Notes

1. ⭐**信息守恒**：LRP 实现的传播过程服从守恒性质，神经元接收到的信息必须等量地重新分配到下一层。简单来看，LRP 就是一种权重不断回传的算法，如下图所示

   <img src="images/image-20210428105041271.png" alt="image-20210428105041271" style="zoom: 37%;" />

2. LRP Rules for Deep Rectifier Networks

   该部分主要针对 ReLU 神经网络，为了一般化，文章设 $a_0=1$ 并且 $w_{0k}$ 是偏置项：

   <img src="images/image-20210428110035537.png" alt="image-20210428110035537" style="zoom: 17%;" />

   - **Basic Rule (LRP-0)**：即直接按照比例进行回传；

     <img src="images/image-20210428110317923.png" alt="image-20210428110317923" style="zoom: 17%;" />

     作者指出：尽管这条规则看起来很直观，但可以证明，这条规则应用在整个神经网络中时，等效于 $Grad \times Input$；

   - **Epsilon Rule (LRP-$\pmb{\epsilon}$)**：即在相关性中引入一个 $\epsilon$ ；

     <img src="images/image-20210428111230669.png" alt="image-20210428111230669" style="zoom: 17%;" />

     作者指出：添加 $\epsilon$ 项可以用来稀释贡献较弱（weak）或相反（contradictory）的项，当其值变大时，只有那些影响大的项的结果才会被保留，这可以用来产生更稀疏，噪声更小的相关性解释；

   - **Gamma Rule (LRP-$\pmb{\gamma}$)** ：即在相关性中引入一个正向系数的比例因子 $\gamma$；

     <img src="images/image-20210428145011393.png" alt="image-20210428145011393" style="zoom: 20%;" />

   - 算法实现：将上述三种规则统一为如下形式

     <img src="images/image-20210428152211271.png" alt="image-20210428152211271" style="zoom: 20%;" />

     这里，$\rho$ 主要和 $\gamma$ 相关。上述公式在实际中可以分成 4 步进行实现

     <img src="images/image-20210428152552631.png" alt="image-20210428152552631" style="zoom: 45%;" />

     其中第三步可以借助深度学习框架中的梯度下降算法来实现

     <img src="images/image-20210428152753536.png" alt="image-20210428152753536" style="zoom:16%;" />

     作者给出了可行的 **PyTorch** 实现

     <img src="images/image-20210428152944695.png" alt="image-20210428152944695" style="zoom:45%;" />

3. ⭐ **不同的神经网络层选择不同的 LRP 策略**

   - 不同的 LRP 策略给解释结果带来的影响，如下图：

     <img src="images/image-20210428205228055.png" alt="image-20210428205228055" style="zoom:50%;" />

   - $\text{Softmax-Layer}$：神经网络的最后一层常见的都是使用 Softmax 输出模型最后的概率，公式如下
     $$
     \text{affine:  }z_c = \sum_{0,k} a_k w_{kc} \\
     \text{softmax:   }P(\omega_c) = \frac{\exp(z_c)}{\sum_{c'}\exp{z_{c'}}} 
     $$
     如下图中间图所示

     <img src="images/image-20210428235525813.png" alt="image-20210428235525813" style="zoom:50%;" />

     如果直接对 $z_c$ 进行解释，可以发现模型结果与“乘用车”部分有较大的相关性，但是对于“火车头”仍然有较大的相关性，所以这样的可解释结果是不准确的（<u>如何区分模型的可解释性结果是模型导致的问题，还是解释性方法导致的问题？</u>）；

     作者指出，可以转换成对 $\eta_c$ 的解释，效果如上图右边图所示，可以发现这次的模型解释结果更多得是在关注“乘用车”，而“火车头”部分则是出现了负相关的情况。公式如下：
     $$
     \eta_c = \log[P(\omega_c)/(1-P(\omega_c))]
     $$
     该公式可以通过两层计算得到：（<u>具体的实现如何做？</u>）

     <img src="images/image-20210429000835187.png" alt="image-20210429000835187" style="zoom: 18%;" />

     第一层公式用来计算  $\text{the log-probability ratios  - } \log[P(\omega_c)/P(\omega_{c'})]$ ，第二层公式用来计算 $\text{log-sum-exp pooling}$。对于这种池化操作，可以通过如下公式进行相关性系数的回传：

     <img src="images/image-20210429001309618.png" alt="image-20210429001309618" style="zoom: 25%;" />

   - $\text{Special Pooling Layers}$：在卷积网络中经常用到一些池化层

     - $\text{Sum Pooling}$：可以采用正常的 linear-ReLU 层对这层进行替换；
     - $\text{Max Pooling}$：可以采用 winner-take-all 策略，或是和 $\text{Sum Pooling}$ 采用相同的策略；

   - $\text{Batch Normalization Layers}$：批量池化层在固定后起到的作用只是对输入起到了修改均值和缩放的作用，所以可以借助想用的线性仿射变换进行替代；

   - $\text{Input Layer}$：对于像素点的输入层，作者指出可以使用 $z^\mathcal{B}\text{-rule}$ ，具体公式如下

     <img src="images/image-20210429002843076.png" alt="image-20210429002843076" style="zoom: 20%;" />

   - $\text{LSTM Layer}$：其公式如下，作者指出一种常用的实现是将相关性系数沿着信号传递（即只沿着第二项进行传递）

     <img src="images/image-20210429092849280.png" alt="image-20210429092849280" style="zoom: 23%;" />

   - $\text{Pairwise Matching}$：常用在推荐系统或者图像-特征图匹配中（<u>这个场景太模糊了，有没有更具体的解释？</u>），作者指出一种可行的实现是构造如下神经元

     <img src="images/image-20210429093506456.png" alt="image-20210429093506456" style="zoom: 20%;" />

     其相关性的传递公式如下

     <img src="images/image-20210429093602351.png" alt="image-20210429093602351" style="zoom: 15%;" />

   - 常用的实现列表：

     <img src="images/image-20210429093846601.png" alt="image-20210429093846601" style="zoom: 43%;" />

### Links

- 论文链接：[Montavon G, Binder A, Lapuschkin S, et al. Layer-wise relevance propagation: an overview[J]. Explainable AI: interpreting, explaining and visualizing deep learning, 2019: 193-209.](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)

- 论文代码：https://git.tu-berlin.de/gmontavon/lrp-tutorial

