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
- Certified adversarial robustness via randomized smoothing

## Towards Deep Learning Models Resistant to Adversarial Attacks

> 由于时间原因，该文章的笔记借鉴自 “前人分享”（链接见下）。

### Contribution

1. Madry.
2. 建模了对抗训练过程；
3. 使用 PGD生成的对抗样本 来做**对抗训练**；

### Notes

1. ⭐ 问题建模，**从优化的角度来看模型鲁棒性问题**。深度学习中，我们经常根据下面这个目标来训练我们的网络：即我们希望我们训练得到的模型在训练样本**上**的经验损失能够达到最小。
   
   $$
   \min_\theta \rho(\theta), \;\; \text{where} \;\; \rho(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[
   L(\theta, x, y)
   \right]
   $$
   
   但是这样的训练目标，使得模型容易受到对抗样本的攻击。故作者将对抗样本的攻击防御问题总结为以下公式，该问题原文中作者称为 **鞍点问题（saddle point problem）**，即我们希望我们训练得到的模型在训练样本**周围**的经验损失能够达到最小。
   
   $$
   \min_\theta \rho(\theta), \;\; \text{where} \;\; \rho(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\max_{\delta \in \mathcal{S}}
   L(\theta, x+\delta, y)
   \right]
   $$
   
   建模完问题以后，那么以前的对抗样本领域的工作就可以进行简单地分类：（<u>稍微有点绕</u>）
   
   - 提出一个好的对抗攻击算法，来寻找使得（内层）经验损失最大化的扰动；
   - 提出一个鲁棒性好的模型，来使得（外层）最小化（内层的最大的）经验损失；

2. 文章中作者采用投影梯度下降算法（PGD）来生成对抗样本：（<u>作者的代码中，在生成对抗样本之前，会添加均匀分布的随机噪声，然后再生成对抗样本</u>）
   
   <img src="pictures/image-20210131225428946.png" alt="image-20210131225428946" style="zoom: 19%;" />

3. 实验发现：
   
   (1) Loss 下降趋势和对抗样本算法迭代轮数的关系：无论是原始模型还是使用对抗训练得到的模型，两者使用 PGD算法 生成对抗样本时，随着迭代轮数的上升，样本的 loss 都会上升，且到最后趋于收敛；
   
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

## Theoretically Principled Trade-off between Robustness and Accuracy

### Contribution

1. 简称：Trades.
2. 从理论上证明了 成功率和鲁棒性 对于分类问题来说是一个权衡利弊的问题；（<u>虽然我并不太关注这个证明</u>）
3. 在理论的基础上，提出了新的对抗训练的损失函数；

### Notes

1. 文章思想：（证明部分直接忽略不看）
   
   (1) 文章整体的思想如下图，即为在保证模型分类准确的前提下，希望模型的边界能够离这些真实的样本远一些：
   
   <img src="pictures/image-20210202165954840.png" alt="image-20210202165954840" style="zoom: 43%;" />
   
   (2) 文章中提到了 鲁棒误差（Robust Error）、自然误差（Natural Error）和 边界误差（Boundary Error）：
   
   - 鲁棒误差：<img src="pictures/image-20210202175415560.png" alt="image-20210202175415560" style="zoom: 25%;" />
   
   - 自然误差：<img src="pictures/image-20210202175455598.png" alt="image-20210202175455598" style="zoom:18%;" />
   
   - 边界误差：<img src="pictures/image-20210202175542702.png" alt="image-20210202175542702" style="zoom:25%;" />
   
   - 三者关系：<img src="pictures/image-20210202175626692.png" alt="image-20210202175626692" style="zoom:13.5%;" />
     
     理解起来有点困难，借鉴前人分享的图（<u>链接见下</u>）：
     
     ![img](pictures/v2-1d2fbe6fc48155cfdaf2082b50233caf_720w.jpg)
     
     > 画个示意图：假设中间是决策边界，A点~G点是x，外面的圈圈是给定的扰动，虚线是决策边界的边界。

2. 最优化问题：
   
   <img src="pictures/image-20210202164152015.png" alt="image-20210202164152015" style="zoom: 25%;" />
   
   左半部分用来保证模型的**准确性**，而右半部分则用来保证模型的**鲁棒性**；上式其实表述的是一个二分类的问题，对于多分类问题，作者修改上式为：
   
   <img src="pictures/image-20210202171523939.png" alt="image-20210202171523939" style="zoom: 25%;" />
   
   其中 $\mathcal{L}(\cdot,\cdot)$ 为交叉熵损失函数；

3. 训练方法：
   
   <img src="pictures/image-20210202172815570.png" alt="image-20210202172815570" style="zoom:50%;" />
   
   首先在样本上面添加高斯随机噪声（[cifar10-challenge](#Towards Deep Learning Models Resistant to Adversarial Attacks) 中用的是均匀分布随机噪声），然后再利用 PGD 生成对抗样本（这里希望对抗样本的概率分布和原始概率分布能够尽可能的不同），最后微调模型。

4. 实验：
   
   (0) 实验评估：使用两个成功率来评估方法的好坏，为了说明实验方法即保证较高的成功率，又有很好的鲁棒性，我们应该希望实验得到的数据 $\mathcal{A}_{rob} \ and \ \mathcal{A}_{nat}$ 都比较大；
   
   $$
   \mathcal{A}_{rob}(f) = 1 - \mathcal{R}_{rob}(f) \\
   \mathcal{A}_{nat}(f) = 1 - \mathcal{R}_{nat}(f)
   $$
   
   (1) 参数 $\lambda$ 的作用：
   
   - 参数设置：
     
     <img src="pictures/image-20210202203007605.png" alt="image-20210202203007605" style="zoom: 50%;" />
   
   - 实验结果：在 MNIST 上面对原始分类成功率的影响不大，但是对 CIFAR10 的影响还是比较大的。所以后面的实验中，作者分别使用 $\lambda=1\ or\ \lambda=6$，分别保证成功率和鲁棒性；
     
     <img src="pictures/image-20210202203422196.png" alt="image-20210202203422196" style="zoom: 43%;" />
   
   (2) 横向对比 - 白盒攻击：
   
   - 参数设置：
     
     <img src="pictures/image-20210202204046392.png" alt="image-20210202204046392" style="zoom:43%;" />
     
     <img src="pictures/image-20210202204113833.png" alt="image-20210202204113833" style="zoom:43%;" />
   
   - 实验结果：
     
     <img src="pictures/image-20210202204443598.png" alt="image-20210202204443598" style="zoom:70%;" />
   
   (3) 横向对比 - 黑盒攻击：
   
   - 参数设置：
     
     <img src="pictures/image-20210202204046392.png" alt="image-20210202204046392" style="zoom:43%;" />
     
     <img src="pictures/image-20210202205001998.png" alt="image-20210202205001998" style="zoom:45%;" />
   
   - 实验结果：
     
     <img src="pictures/image-20210202205057265.png" alt="image-20210202205057265" style="zoom:43%;" />
     
     <img src="pictures/image-20210202205235451.png" alt="image-20210202205235451" style="zoom:43%;" />

### Links

- 论文链接：[Zhang H, Yu Y, Jiao J, et al. Theoretically principled trade-off between robustness and accuracy[C]//International Conference on Machine Learning. PMLR, 2019: 7472-7482.](https://arxiv.org/abs/1901.08573?spm=5176.12281978.0.0.5f793e46VQJsFW&file=1901.08573)
- 论文代码：https://github.com/yaodongyu/TRADES
- 前人分享：https://zhuanlan.zhihu.com/p/337989683

## Unlabeled Data Improves Adversarial Robustness

### Contribution

1. 简称：RST（**R**obust **S**elf-**T**raining）.
2. 在 [TRADES](#Theoretically Principled Trade-off between Robustness and Accuracy) 的基础上修改了损失函数，添加了无标签数据；
3. 文章给我的感觉是，就是利用更多数据来训练网络，不过这个数据可以是无标签的数据，这样的话就不需要进行大量的认为标注；

### Notes

1. 背景：使用更多的训练数据有助于获得更好的模型鲁棒性，但是收集数据以及打标签是一个十分昂贵的过程；

2. 思路：在对抗训练过程中加上自监督学习；

3. 训练方法：（证明部分直接忽略不看）
   
   <img src="pictures/image-20210202073617084.png" alt="image-20210202073617084" style="zoom:50%;" />
   
   - 首先使用有标签数据（正常地）训练网络，这里使用 $Standard\ Loss$（即交叉熵损失函数）为：
     
     <img src="pictures/image-20210202073956120.png" alt="image-20210202073956120" style="zoom: 19%;" />
   
   - 使用训练好的网络，给无标签的数据打标签；
   
   - 使用有标签和“自标签”的数据继续训练网络，这里使用$Robust\ Loss$（和Trades一样）为：
     
     <img src="pictures/image-20210202074235297.png" alt="image-20210202074235297" style="zoom: 31%;" />
     
     其中 $L_{reg}$ 项保证了在样本的领域内模型的输出概率是稳定的。这又回到了经典的如何拟合 $L_{reg}(\theta, x)$ 项（因为寻找邻域内的最大值太困难），作者提出了两种方法（从实验的结果来看，Adversarial Training 的效果略优于 Stability Training）：
     
     - **Adversarial Training**：使用 PGD 获取邻域最大值
       
       <img src="pictures/image-20210202075550864.png" alt="image-20210202075550864" style="zoom: 21%;" />
     
     - **Stability Training**：使用 高斯分布 采样邻域的值
       
       <img src="pictures/image-20210202080012169.png" alt="image-20210202080012169" style="zoom:29%;" />
       
       使用 Stability Training 的网络在测试的时候也做了改变，**模型输出的是 高斯分布 采样邻域中的可能性最大的分类**
       
       <img src="pictures/image-20210202080219912.png" alt="image-20210202080219912" style="zoom:38%;" />

4. 实验：
   
   (0) 实验参数：
   
   - 数据集：大致意思就是从 **80M 的 CIFAR10-TINY** 中为每个类挑选出 50K 张图片组成一个 500K 大小的无标签数据集；
   
   <img src="pictures/image-20210202210522050.png" alt="image-20210202210522050" style="zoom:45%;" />
   
   - 模型 & 训练参数：
   
   <img src="pictures/image-20210202211117325.png" alt="image-20210202211117325" style="zoom:45%;" />
   
   (1) 经验性防御（heuristic defense）：主要关注 $L_\infty$ 攻击；（⭐ <u>针对作者提供的这种方法，作者还自己对 PGD 做了修改以达到最好的攻击成功率，这一点在做攻击的时候可以借鉴一下。</u>）
   
   <img src="pictures/image-20210202081540739.png" alt="image-20210202081540739" style="zoom:45%;" />
   
   (2) 证明性防御（certified defense）：同时关注 $L_\infty$ 和 $L_2$ 攻击；
   
   <img src="pictures/image-20210202082314261.png" alt="image-20210202082314261" style="zoom: 50%;" />
   
   (3) SVHN（Street View House Numbers）实验：实验的结果表明，数据的标签对于模型的鲁棒性影响并不大；（<u>我这里比较好奇的是，在 SVHN 这个任务上作者提出的方法并没有优于 $Baseline_{adv}(604K)$ 的效果</u>）
   
   <img src="pictures/image-20210202214606301.png" alt="image-20210202214606301" style="zoom:47%;" />

### Links

- 论文链接：[Carmon Y, Raghunathan A, Schmidt L, et al. Unlabeled data improves adversarial robustness[J]. arXiv preprint arXiv:1905.13736, 2019.](https://arxiv.org/abs/1905.13736)

- 论文代码：https://github.com/yaircarmon/semisup-adv

## Defense Against Adversarial Attacks Using Feature Scattering-based Adversarial Training

### Contribution

1. 使用  Wasserstein Distance 来生成对抗样本；

### Notes

1. Background：
   
   (1) Adversarial Training，可以被形式化为 $min-max$ 问题：
   
   <img src="pictures/image-20210204222303690.png" alt="image-20210204222303690" style="zoom: 12%;" />
   
   ​    其中内层最大值经常用对抗样本生成来近似，如使用单步的 $FGSM$ 算法，或者是多步的 $PGD$ 算法（算法第一次迭代时首先会添加上一个随机噪声，然后再迭代生成对抗样本），$PGD$ 算法公式如下：
   
   <img src="pictures/image-20210204222702970.png" alt="image-20210204222702970" style="zoom:20%;" />
   
   (2) 对抗训练的问题：
   
   - 标签泄露（Label Leaking）：生成的对抗扰动本身和目标类别是密切相关的，导致训练好的模型看到测试集的对抗扰动便知道目标分类，而与原始的图片无关；
   - 梯度隐藏（Gradient Masking）：指的是训练后的模型学习到了尽可能生成一些无用的梯度（<u>这个和我想象中的“梯度隐藏”的概念不同，从作者的解释来看是模型学习到了这种生成无用梯度的能力，可能是我没有真正理解这个含义，因为我感觉即使是梯度隐藏也是可以起到防御对抗攻击的效果的</u>）；

2. 最优运输理论（Optimal Transport Theory）
   
   (1) 参考链接：
   
   - [【数学】Wasserstein Distance](https://zhuanlan.zhihu.com/p/58506295)
   - [机器学习工具（二）Notes of Optimal Transport](https://zhuanlan.zhihu.com/p/82424946)
   - [令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)
   
   (2) Wasserstein Distance的优点：**Wasserstein距离相比KL散度、JS散度的优越性在于，即便两个分布没有重叠，Wasserstein距离仍然能够反映它们的远近**

3. 训练方法：
   
   <img src="pictures/image-20210207160456570.png" alt="image-20210207160456570" style="zoom: 43%;" />
   
   <u>我的理解：</u>
   
   - 简单的来看，作者提出的方法和已有的方法的不同之处在于使用了 Wasserstein Distance 作为生成对抗样本时的损失函数；
   - 思考一下，变换了这个损失函数的作用有可能那么大么？从 Wasserstein Distance 自身的有点来看，它的优点主要是体现在两个分布没有重叠时，还能够有效地指导网络训练。所以从对抗样本的生成角度来看，Wasserstein Distance 对于较大扰动时的对抗样本生成是更加有效的，因为此时两个样本的解码概率分布可能已经差异十分大了，用 KL 散度并不能更好地指导生成对抗样本；**概括一下，能够指导生成更多样化的对抗样本，从而更好地进行对抗训练**；
   - 再思考一下，这篇文章作者一直是从 feature 这个角度来说的，其实我一直不太明白作者这样说有什么意义，但是我们揣测一下的话，因为在利用 Wasserstein Distance 生成对抗样本的时候，其实是要利用到一整个batch的样本来调整扰动的，这也是这个方法和前面方法的不同之处；**概括一下，用到了batch来生成对抗样本**；
   
   方法示意图：
   
   <img src="pictures/image-20210207162115193.png" alt="image-20210207162115193" style="zoom: 43%;" />

4. 实验：
   
   (0) 实验参数：
   
   - 数据集 & 基准方法：
     
     <img src="pictures/image-20210207175245420.png" alt="image-20210207175245420" style="zoom:49%;" />
   
   - 测试参数：
     
     <img src="pictures/image-20210207175032840.png" alt="image-20210207175032840" style="zoom:50%;" />
   
   (1) 白盒攻击结果：
   
   <img src="pictures/image-20210207175417561.png" alt="image-20210207175417561" style="zoom:50%;" />
   
   <img src="pictures/image-20210207175438864.png" alt="image-20210207175438864" style="zoom:50%;" />

### Links

- 论文链接：[Zhang H, Wang J. Defense against adversarial attacks using feature scattering-based adversarial training[J]. arXiv preprint arXiv:1907.10764, 2019.](https://arxiv.org/abs/1907.10764?spm=5176.12281978.0.0.5f797a4dBpnylV&file=1907.10764)
- 论文代码：https://github.com/Haichao-Zhang/FeatureScatter

## * Using Pre-Training Can Improve Model Robustness and Uncertainty

### Contribution

1. 验证了预训练的作用：不仅可以提高模型收敛速率，而且可以提高模型的鲁棒性和不确定性；

### Links

- 论文链接：[Hendrycks D, Lee K, Mazeika M. Using pre-training can improve model robustness and uncertainty[C]//International Conference on Machine Learning. PMLR, 2019: 2712-2721.](https://arxiv.org/abs/1901.09960?spm=5176.12281978.0.0.5f793e46aHiE80&file=1901.09960)
- 论文代码： https://github.com/hendrycks/pre-training

## Boosting Adversarial Training with Hypersphere Embedding

### Contribution

1. 修改了loss函数，希望生成对抗样本的时候能够只学习角度的变换；

### Notes

1. 思想：作者希望在生成对抗样本的时候，尽可能地旋转样本的角度，而不是缩放大小；
   
   <img src="pictures/image-20210212110243153.png" alt="image-20210212110243153" style="zoom: 50%;" />

2. 方法：对参数和模型提取出来的特征，在最后一个 softmax 层之前做 normalization，然后计算 loss；
   
   <img src="pictures/image-20210212110557246.png" alt="image-20210212110557246" style="zoom:50%;" />
   
   该处的loss实现方法和 [CosFace](https://arxiv.org/abs/1801.09414) 相同。

3. 实验：
   
   <img src="pictures/image-20210212110722459.png" alt="image-20210212110722459" style="zoom: 67%;" />

### Links

- 论文链接：[Pang T, Yang X, Dong Y, et al. Boosting adversarial training with hypersphere embedding[J]. arXiv preprint arXiv:2002.08619, 2020.](https://arxiv.org/abs/2002.08619)

- 论文代码：https://github.com/ShawnXYang/AT_HE

## * Overfitting in adversarially robust deep learning

### Contribution

1. 使用大量的实验来验证：对抗训练的模型需要设置一个 `early stop` ，并且这种模型过拟合的问题并不能通过现有的一些抑制过拟合的方法来解决；

### Notes

1. 实验 1：过拟合会对对抗训练的鲁棒性产生负面影响；
   
   - 在 $l_\infty$ 对抗攻击下：
   
   <img src="pictures/image-20210212195423596.png" alt="image-20210212195423596" style="zoom: 60%;" />
   
   - 在 $l_2$ 和 $l_\infty$ 对抗攻击下的多个数据集中：
   
   <img src="pictures/image-20210212195726133.png" alt="image-20210212195726133" style="zoom:48%;" />

2. 实验 2：不同的学习率退化算法对过拟合现象的影响；（都会出现过拟合现象）
   
   <img src="pictures/image-20210212200229043.png" alt="image-20210212200229043" style="zoom:50%;" />

3. 实验 3：可以在实验中设置一个验证集来判断对抗训练是否已经过拟合；
   
   <img src="pictures/image-20210212200852708.png" alt="image-20210212200852708" style="zoom:50%;" />

4. 实验 4：不同的模型大小对过拟合的影响；（都会出现过拟合现象，但是复杂网络能够达到最优鲁棒性）
   
   <img src="pictures/image-20210212201311036.png" alt="image-20210212201311036" style="zoom: 75%;" />

5. 实验 5：现有的抑制过拟合的方法对过拟合的影响；（都会出现过拟合现象，并且除了 `Semi-Supervised` 方法，其他方法效果都和 `early stop` 相似）
   
   <img src="pictures/image-20210212201718555.png" alt="image-20210212201718555" style="zoom:45%;" />

### Links

- 论文链接：[Rice L, Wong E, Kolter Z. Overfitting in adversarially robust deep learning[C]//International Conference on Machine Learning. PMLR, 2020: 8093-8104.](https://arxiv.org/abs/2002.11569?spm=5176.12281978.0.0.5f793e4672ozXT&file=2002.11569)
- 论文代码：https://github.com/locuslab/robust_overfitting

## Rethinking Softmax Cross-Entropy Loss for Adversarial Robustness

### Contribution

### Notes

### Links

- 论文链接：[Pang T, Xu K, Dong Y, et al. Rethinking softmax cross-entropy loss for adversarial robustness[J]. ICLR, 2020.](https://arxiv.org/abs/1905.10626?spm=5176.12281978.0.0.5f793e46BkBcJw&file=1905.10626)
- 论文代码：

## Adversarial Weight Perturbation Helps Robust Generalization

### Contribution

1. 简称：AWP（**A**dversarial **W**eight **P**erturbation）；
2. 提出了一种在高训练集鲁棒性的前提下，获得高测试机鲁棒性的算法（训练的时候不断对抗地修改输入样本和参数）；

### Notes

1. **Weight Loss Landscape** 和 **Robust Generalization Gap** 的关系：
   
   (1) 首先要解释下这两个词：
   
   - Weight Loss Landscape：指的是微小地修改模型的参数，目标 Loss（这个在下面提到）值的改变情况；
   - Robust Generalization Gap：指的是训练集和测试集上模型鲁棒性的差距；
   
   (2) 目标 Loss 度量：
   
   <img src="pictures/image-20210305193954917.png" alt="image-20210305193954917" style="zoom: 25%;" />
   
   $\bold{w}$ 指的是模型的参数，$\bold{d}$ 指的是随机高斯分布（实验中采样 10 次）的参数修改方向，$\alpha$ 指的是参数修改量的大小；另外，文章中使用的 loss 函数 $\mathcal{l}$ 是交叉熵函数，用 PGD 算法动态生成对抗样本使得 loss 最大化。
   
   (3) 实验结果：
   
   <img src="pictures/image-20210305195019886.png" alt="image-20210305195019886" style="zoom: 45%;" />
   
   - **The Connection in the Learning Process of Adversarial Training**：上图 a 中可以看出，Weight Loss Landscape 越平坦，Robust Generalization Gap 就越小；
   - **The Connection across Different Adversarial Training Methods**：上图 b 中可以看出，AT-ES 对抗训练算法的 Weight Loss Landscape 最平坦，并且 Robust Generalization Gap 也最小；
   - **Does Flatter Weight Loss Landscape Certainly Lead to Higher Test Robustness**：一个平坦的减肥景观确实直接导致一个较小的鲁棒推广差距，但只有在训练过程足够的条件下（即训练鲁棒性高），才有利于最终测试的鲁棒性；

2. 文章算法：
   
   (1) 根据上面实验的分析，我们原始的目标是”希望**训练集鲁棒性强**，且**测试集和训练集的鲁棒性差距小**“，可以转换成新的目标是”希望**训练集鲁棒性强**，且**新提出的度量 Loss 小**“，即：
   
   <img src="pictures/image-20210305200548393.png" alt="image-20210305200548393" style="zoom: 25%;" />
   
   (2) 然后，我们需要展开上式，得到：
   
   <img src="pictures/image-20210305200657910.png" alt="image-20210305200657910" style="zoom: 30%;" />
   
   其中，$\bold{v}$ 是模型参数的一个可能的修改量，这个修改量的范围由模型各层参数的大小分别确定，即：
   
   <img title="" src="pictures/image-20210305200845013.png" alt="image-20210305200845013" style="zoom: 8%;" width="132" data-align="center">
   
   (3) 参数更新过程：
   
   - 修改输入 $\bold{x}$ ：使用 PGD 算法生成对抗样本
     
     <img src="pictures/image-20210305202000184.png" alt="image-20210305202000184" style="zoom: 20%;" />
   
   - 改变模型参数的修改量 $\bold{v}$ ：使用梯度上升算法
     
     <img src="pictures/image-20210305202259907.png" alt="image-20210305202259907" style="zoom:25%;" />
     
     其中，$m$ 是 batch size，另外这里 $\bold{v}$ 的更新也是分层更新的，即
     
     <img src="pictures/image-20210305202943936.png" alt="image-20210305203048988" style="zoom:25%;" />
   
   - 更新模型参数：

<img src="pictures/image-20210305202450224.png" alt="image-20210305202450224" style="zoom: 25%;" />

​        (4) 伪代码：

<img src="pictures/image-20210305202648140.png" alt="image-20210305202648140" style="zoom:50%;" />

​            注：<u>伪代码中可能忘记了每一步 $t$ 之前，应该将 $\bold{v}$ 初始化为 0</u>；

3. 实验结果：
   
   <img src="pictures/image-20210305204419577.png" alt="image-20210305204419577" style="zoom: 42%;" />
   
   <img src="pictures/image-20210305204511222.png" alt="image-20210305204511222" style="zoom: 43%;" />

### Links

- 论文链接：[Wu D, Xia S T, Wang Y. Adversarial weight perturbation helps robust generalization[J]. Advances in Neural Information Processing Systems, 2020, 33.](https://arxiv.org/abs/2004.05884)
- 论文代码：https://github.com/csdongxian/AWP

## Fast is better than free: Revisiting adversarial training

### Contribution

### Notes

### Links

- 论文链接：[Wong E, Rice L, Kolter J Z. Fast is better than free: Revisiting adversarial training[J]. arXiv preprint arXiv:2001.03994, 2020.](https://arxiv.org/abs/2001.03994)
- 论文代码：https://github.com/locuslab/fast_adversarial

## Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples

### Contribution

1. 作者探讨了已有的对对抗训练存在影响的音素，并训练了一个目前为止最优的对抗训练模型；

### Notes

1. 先看一眼文章列出的研究进程：这篇文章的主旨是去分析对抗训练中各个音素可能给对抗鲁棒性带来的影响；
   
   <img src="pictures/image-20210313113536476.png" alt="image-20210313113536476" style="zoom:50%;" />
   
   梯度隐藏：<u>看过这篇文章，我对梯度隐藏这个概念有了新的认识，它指的是模型的鲁棒性其实并没有真正的提高，即在 $\epsilon$ 邻域内仍然可以找到一个成功的对抗样本，但是通过计算梯度的方法，很可能得到的梯度为 0，故使得梯度的方法并不能很好地生成一个成功的对抗样本，在测试时模型的鲁棒性可能会有提升，但通过更加严格的攻击时，模型的鲁棒性会立刻下降</u>；

2. 外层优化目标（训练网络）对对抗鲁棒性的影响：
   
   <img src="pictures/image-20210313215127927.png" alt="image-20210313215127927" style="zoom:48%;" />
   
   - 作者对比了 AT、TRADES 和 MART 这三种外层优化损失函数，发现相对而言 TRADES 的效果是最好的；
   - MART 产生了比较强的梯度隐藏效果；
   - 对抗鲁棒性的验证应该使用比 $PGD^{20}$ 攻击效果更好的攻击，这样才更具有标准型；

3. 内层优化目标（生成对抗样本）对对抗鲁棒性的影响：
   
   <img src="pictures/image-20210313221504531.png" alt="image-20210313221504531" style="zoom: 48%;" />
   
   - 作者对比了 XENT、KL 和 MARGIN 这三种内层优化目标，发现在少样本训练情况中 TRADES-XENT 的效果最好，在负样本训练情况中 TRADES-KL的效果最好；
   - MARGIN 产生了比较强的梯度隐藏效果；

4. 内层优化时的最大扰动（生成对抗样本）对对抗鲁棒性的影响：
   
   <img src="pictures/image-20210313232503970.png" alt="image-20210313232503970" style="zoom:48%;" />
   
   - 对于 AT 来说，提高最大扰动量可以在一定程度上缩小它和 TRADES 的差距；
   - 对于 TRADES 来说，提高最大扰动量并没有什么明显的作用；

5. 额外数据集的数量和质量对对抗鲁棒性的影响：
   
   <img src="pictures/image-20210313233611254.png" alt="image-20210313233611254" style="zoom:25%;" />
   
   - 额外数据集的增加，会对模型鲁棒性带来先增后减的影响；

6. batch 中有标签和无标签数据的比例对对抗鲁棒性的影响：
   
   <img src="pictures/image-20210313234126133.png" alt="image-20210313234126133" style="zoom: 37%;" />
   
   - 可以看到，在 $3:7$ 的时候，模型的效果最好，所以可以在训练的时候不断扩大这个比例，直到模型的鲁棒性不再上升；
   - Label Smoothing 并没有太大的效果（<u>Label Smoothing 指的到底是什么？</u>）；

7. 模型大小对对抗鲁棒性的影响：
   
   <img src="pictures/image-20210313235110733.png" alt="image-20210313235110733" style="zoom:43%;" />
   
   - 增大模型每层的神经元个数，或者是直接增加模型的层数，对模型的鲁棒性都有不错的效果；
   - 不过不是在每种情况下，增加模型大小的作用都是一样的，有的好一些，有的差一些；

8. 加权平均（Weight Average）对对抗鲁棒性的影响：
   
   <img src="pictures/image-20210313235928491.png" alt="image-20210313235928491" style="zoom:44%;" />
   
   - WA 确实可以增强模型的对抗鲁棒性；

9. 模型激活函数对对抗鲁棒性的影响：
   
   <img src="pictures/image-20210314000257608.png" alt="image-20210314000257608" style="zoom:42%;" />
   
   - 激活函数确实可能影响模型的对抗鲁棒性；
   - 实验中作者发现 $Swish/SiLU$ 激活函数相对来说效果最好；

10. 最终模型：
    
    - 模型训练时所用的参数：
      
      <img src="pictures/image-20210314000758749.png" alt="image-20210314000758749" style="zoom:50%;" />
    
    - 实验得到的结果：
      
      <img src="pictures/image-20210314000907884.png" alt="image-20210314000907884" style="zoom: 50%;" />

### Links

- 论文链接：[Gowal S, Qin C, Uesato J, et al. Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples[J]. arXiv preprint arXiv:2010.03593, 2020.](https://arxiv.org/abs/2010.03593)



## PatchGuard: A Provably Robust Defense against Adversarial Patches via Small Receptive Fields and Masking

### Contribution

### Notes

### Links

- 论文链接：[Xiang C, Bhagoji A N, Sehwag V, et al. Patchguard: A provably robust defense against adversarial patches via small receptive fields and masking[C]//30th {USENIX} Security Symposium ({USENIX} Security 21). 2021.](https://www.usenix.org/conference/usenixsecurity21/presentation/xiang)
- 代码链接：https://github.com/inspire-group/PatchGuard



## DetectorGuard: Provably Securing Object Detectors against Localized Patch Hiding Attacks

### Contribution

### Notes

### Links

- 论文链接：[Xiang C, Mittal P. DetectorGuard: Provably Securing Object Detectors against Localized Patch Hiding Attacks[J]. CCS, 2021.](https://arxiv.org/abs/2102.02956)
- 代码链接：https://github.com/inspire-group/DetectorGuard



## Recent Advances in Adversarial Training for Adversarial Robustness

### Contribution

1. 一篇很好的SOK文章，对对抗训练的相关工作进行了相应的总结；

> 可以看到，在一定程度上，这样的对抗训练算法，我们可以直接拿到语音领域来使用；所以查阅一下语音领域是否有相关工作？它们遇到了什么问题？为什么没有被大量采用？负采样的思想是否可行？

### Notes

1. 现有的对抗训练算法存在的问题：
   
   a. Standard Generalization：对抗训练会影响正常数据集的精度；
   
   b. Adversarial Robust Generalization：对抗训练的效果在训练集上比测试集上好得多；
   
   c. Generalization on Unseen Attacks：对抗训练对没有见过的攻击可能泛化性能差；

2. 现有的算法：

   <img src="pictures/image-20220316160905490.png" alt="image-20220316160905490" style="zoom: 33%;" />

### Links

- 论文链接：[Bai T, Luo J, Zhao J, et al. Recent advances in adversarial training for adversarial robustness[J]. arXiv preprint arXiv:2102.01356, 2021.](https://arxiv.org/abs/2102.01356)
