# Attack on Speech Recognition



[TOC]



## Todo List

1. Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., and Fergus, R. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.
2. Biggio, B., Corona, I., Maiorca, D., Nelson, B., ˇSrndi´c, N., Laskov, P., Giacinto, G., and Roli, F. Evasion attacks against machine learning at test time. In Joint European conference on machine learning and knowledge discovery in databases, pp. 387–402. Springer, 2013.
6. N. Carlini, P. Mishra, T. Vaidya, Y. Zhang, M. Sherr, C. Shields, D. Wagner, and W. Zhou. Hidden voice commands. In 25th USENIX Security Symposium (USENIX Security 16), Austin, TX, 2016.
6. A. Nguyen, J. Yosinski, and J. Clune, “Deep neural networks are easily fooled: High confidence predictions for unrecognizable images,” in Conference on Computer Vision and Pattern Recognition. IEEE, Jun. 2015, pp. 427–436.
7. N. Carlini and D. Wagner, “Towards evaluating the robustness of neural networks,” in Symposium on Security and Privacy. IEEE, May 2017, pp. 39–57.
8. I. Evtimov, K. Eykholt, E. Fernandes, T. Kohno, B. Li, A. Prakash, A. Rahmati, and D. Song, “Robust physical-world attacks on machine learning models,” CoRR, vol. abs/1707.08945, pp. 1–11, Jul. 2017.
10. Moustapha Ciss´e, Yossi Adi, Natalia Neverova, and Joseph Keshet. Houdini: Fooling deep structured visual and speech recognition models with adversarial examples. In Proceedings of the 31st Annual Conference on Neural Information Processing Systems, pages 6980–6990, 2017.
8. Dibya Mukhopadhyay, Maliheh Shirvanian, and Nitesh Saxena. 2015. All your voices are belong to us: Stealing voices to fool humans and machines. In Proceedings ofthe European Symposium on Research in Computer Security. Springer, 599–621.





## Cocaine Noodles: Exploiting the Gap between Human and Machine Speech Recognition

### Contribution

1. 这篇文章可以说是语音识别对抗攻击的开山之祖；
2. 攻击特征提取模块；

### Notes

1. 攻击 **特征提取(MFCC)** 模块的对抗攻击；

2. 👍  攻击方法的流程图如下：

   <img src="pictures/image-20201205201454735.png" alt="image-20201205201454735" style="zoom:28%;" />

   作者的攻击思路非常简单，从语音中提取出 MFCC 特征（这个过程会 **丢失一些语音的信息**，如 MFCC 特征往往只取前 13 个参数，而高维的 MFCC 特征代表着能量变化的高频信息），然后**把 MFCC 特征逆转为语音信号**，只要该样本同时满足 “**人耳无法理解**”，“**机器可以理解**” 两个条件，那么这就是一个成功的对抗样本。

### Shortcoming

1. 需要大量的时间去生成一个对抗样本，因为要调参数使得满足 **“人耳无法理解“，”机器可以理解“** 两个条件；
2. （<u>纯属吐槽</u>）文章的编写我觉得挺烂的，你想知道的你都没有知道，你不想知道的他介绍了一堆。如果你熟悉语音识别、MFCC的话，会发现整篇文章就只有两块（介绍了语音识别和做了个问卷调查），对于实际的攻击算法的实现、如何去调参生成一个对抗样本（作者的描述是：我生成了 500 个样本）并没有提及，甚至没有介绍相关的一些参数，代码也是没有开源的（计算 MFCC 的链接在下面）；
3. （<u>猜测一下</u>）整个算法的流程大概是：正常计算得到MFCC特征，然后用 **逆 DCT 变换**（对应 Mel Filter Bank Energy 到 MFCC 过程）和 **逆 DFT 变换**（对应 时域信号到频谱 过程）。虽然算法特别简单，但是因为涉及到帧之间的重叠（分帧的时候一般 `step_length < frame_length`），整个调试过程应该是比较麻烦的事情；

### Links

- 论文链接：[Vaidya, Tavish, et al. "Cocaine noodles: exploiting the gap between human and machine speech recognition." *9th {USENIX} Workshop on Offensive Technologies ({WOOT} 15)*. 2015.](https://www.usenix.org/conference/woot15/workshop-program/presentation/vaidya)
- MFCC 实现：[PLP and RASTA (and MFCC, and inversion) in Matlab using melfcc.m and invmelfcc.m (columbia.edu)](https://www.ee.columbia.edu/~dpwe/resources/matlab/rastamat/)





## DolphinAttack: Inaudible voice commands

### Contribution

1. 利用麦克风非线性的特征，使得高频信号采样后出现额外的低频分量；

### Notes

1. 攻击**语音前端采集模块（麦克风）**的对抗攻击（或者说，更像是一种**漏洞**）。

2. 语音信号是一种波，被麦克风捕获，麦克风把波的声压转换成电信号，再通过对电信号进行采样便可获得离散时间的波形文件（引自 [李理的博客](http://fancyerii.github.io/dev287x/ssp/)）。这个过程中，LPC 模块会过滤掉超过 20kHz 的信号，ADC 模块负责采样信号：

   <img src="pictures/image-20201205153918038.png" alt="image-20201205153918038" style="zoom:25%;" />

3. 👍  麦克风的非线性特征，能够使得高频的语音信号被 **downconversion** (或可以理解为**解调**) 出低频的能量分量，如下图：

   <img src="pictures/image-20201205171056036.png" alt="image-20201205171056036" style="zoom: 33%;" />

   其中第一行是**原始语音的频谱**，第二行是 **MEMS 麦克风接收后的频谱**，第三行是 **ECM 麦克风接收后的频谱**。这里采用的载波信号为 20kHz，语音信号为 2kHz。作者接着测试如果是正常的语音能否被解调：

   <img src="pictures/image-20201205174850556.png" alt="image-20201205174850556" style="zoom: 41%;" />

   其中第一行是原始的 TTS 语音（发音为 Hey）的 MFCC 特征，第二行是正常播放-录音后的 MFCC 特征，第三行是经过 “调制-解调”后的 MFCC 特征。计算得到他们的 MCD 距离分别为 3.1 和 7.6。（**我不是很理解 Amplitude of MFCC 是什么意思，时频的 MFCC 特征应该像热力图才对？**）

4. Voice Command Generation，生成指令用于后续调制过程：

   (1) Activation Commands Generation：使用 TTS (Text to Speech) 的方法或者是语音拼接的方法生成一个唤醒词。（可以看到在智能语音助手邻域，说话人的识别并不是很有效的，可以被说话声音相似的人激活）；

   (2) General Control Commands Generation：直接使用 TTS 生成；

5. Evaluation：这部分和信号的调制非常相关，不太易懂，直接简略地看下结果

   <img src="pictures/image-20201205192628736.png" alt="image-20201205192628736" style="zoom: 40%;" />

### Shortcoming:

这篇文章的攻击非常有效，因为他利用的是麦克风的 “漏洞”，所以几乎能够攻击全部平台设备。但**它的缺点是需要一台超声波发生设备**。

### Links

- 论文链接：[Roy, Nirupam, et al. "Inaudible voice commands: The long-range attack and defense." *15th {USENIX} Symposium on Networked Systems Design and Implementation ({NSDI} 18)*. 2018.](https://arxiv.org/abs/1708.09537)
- Github 主页：[USSLab/DolphinAttack: Inaudible Voice Commands (github.com)](https://github.com/USSLab/DolphinAttack)





## * Did you hear that? Adversarial Examples Against Automatic Speech Recognition

### Contribution

1. 针对关键词识别模型进行黑盒攻击；

### Notes

1. **黑盒**、**有目标**的**针对语音关键词识别**的对抗攻击算法。攻击的模型是 **Speech Commands Classification Model**，其中涉及的关键词有 `yes`、`no`、`up`、`down`、`left`、`right`、`on`、`off`、`stop` 和 `go` ；

2. 算法流程：

   <img src="pictures/image-20201202213545932.png" alt="image-20201202213545932" style="zoom: 50%;" />

   使用**遗传算法**生成对抗样本；

### Links

- 论文链接：[Alzantot, Moustafa, Bharathan Balaji, and Mani Srivastava. "Did you hear that? adversarial examples against automatic speech recognition." *NIPS Machine Deception Workshop* (2017).](https://arxiv.org/abs/1801.00554)
- 论文主页：[Adversarial Speech Commands | adversarial_audio (nesl.github.io)](https://nesl.github.io/adversarial_audio/)
- 论文代码：[nesl/adversarial_audio (github.com)](https://github.com/nesl/adversarial_audio)





## Audio Adversarial Examples: Targeted Attacks on Speech-to-Text

### Contribution

1. 白盒、有目标的、攻击端到端 DeepSpeech 模型（CTC）的对抗攻击算法；

### Notes

1. **白盒**、**有目标**的对抗攻击算法。攻击的模型为 **DeepSpeech** 模型，攻击的指令为**任意长度**；

2. **基础**的攻击方法，loss 函数（ 后半部分为 CTC-Loss ）如下：

   <img src="./pictures/image-20201129193714655.png" alt="image-20201129193714655" style="zoom: 60%;" />

   作者提到使用 2 范数而不用无穷范数的原因是，无穷范数可能会导致不收敛的问题，难以训练。在参数的选择上，作者使用 Adam 算法，学习率为 5，迭代论述为 5000。在实验过程中，作者发现**目标指令越长**，需要添加越多的扰动来生成对抗样本；而如果**原始指令越长**，似乎更加容易生成对抗样本（这一点我的想法是，**如果原始指令越长，原始存在更多的音素和能量可以被梯度下降过程利用**）。

3. 👍  **改进**的攻击方法（<u>作者称：这种改进的攻击方法只能在 DeepSpeech 使用 Greedy-Search 的情况下有效</u>），loss 函数如下：

   <img src="./pictures/image-20201129201614604.png" alt="image-20201129201614604" style="zoom: 40%;" />

   其中<img src="./pictures/image-20201129202011269.png" alt="image-20201129202011269" style="zoom: 33%;" /> 表示对于当前 alignment，第 i 帧的 loss 值。作者这样修改 loss 函数的原因大致有两个：

   (1) 如果使用 CTC-Loss，会添加不必要的修改。**如果已经解码出 ”ABCX“，目标指令为 ”ABCD“，在使用 CTC-Loss时，梯度下降算法仍然会在 ”A“ 上添加扰动使得其变得更像 ”A“**；

   (2) **不同的字符生成的难易程度是不同的**，所以把权重系数 c 移到了累加的里面。( **这一点作者称是在 [Hidden Voice Command]() 中发现的规律，但其实只是在附录中给出了不同的单词可能需要的最短音素帧的数量是不同的，并没有给出字符难易程度的结论；并且这篇文章开源的代码中也没有给出这个改进的 loss 函数，所以可以直接把这个 c 移出去作为单个参数进行调参** )；

   训练的 **trick**：首先用 CTC-Loss 生成一个对抗样本，以这个对抗样本为参照固定 alignment（在 CTC 中，可能有许多种alignment，作者通过这种方法来确定选择其中一种），然后用改进的 loss 函数来生成；（这边，我的想法是，**改进的攻击方法会使得对抗样本丧失其迁移性，因为它只是恰好将特征拟合到模型的边界而已，而没有去进一步地逼近泛化的特征上**）

4. Evaluation：

   (1) 作者在原始指令的基础上通过非常小（约为 -30dB）的扰动生成对抗样本，并且在白盒的情况下最多可以在 **1s** 的语音中插入 **50** 个字符；

   (2) **对于 Non-Speech，作者发现更难生成对抗样本**；

   (3) 作者还对比了 FGSM 和 Iterative Optimization 两种生成对抗样本的算法，发现**在语音识别领域 FGSM 只适合生成 un-targeted 样本，而不适合生成 targeted 样本（或者说生成的效率很差，几乎没有办法生成）**；

   (4) 作者发现**这种方法生成的对抗样本是对噪声不鲁棒的**；

### Links

- 论文链接：[Carlini, Nicholas, and David Wagner. "Audio adversarial examples: Targeted attacks on speech-to-text." *2018 IEEE Security and Privacy Workshops (SPW)*. IEEE, 2018.](https://arxiv.org/abs/1801.01944)
- 论文主页：[Audio Adversarial Examples (carlini.com)](https://nicholas.carlini.com/code/audio_adversarial_examples)
- 论文代码：[carlini/audio_adversarial_examples: Targeted Adversarial Examples on Speech-to-Text systems (github.com)](https://github.com/carlini/audio_adversarial_examples)





## Adversarial Attacks Against Automatic Speech Recognition Systems via Psychoacoustic Hiding

### Contribution

1. 白盒、有目标的、针对API的、针对Kaldi DNN-HMM模型的对抗攻击算法；
2. 首次提出使用声学掩蔽效应；
3. 实验部分做的很全面，值得借鉴；

### Notes

1. **白盒**、**有目标**的、**只针对API**的对抗攻击算法。攻击的模型为 Kaldi 的 **WSJ** 模型（或称为 recipe）；

2. 攻击方法整体架构图如下：

   <img src="pictures/image-20201201013545300.png" alt="image-20201201013545300" style="zoom: 35%;" />

   (1) **Forced Alignment**：时序信号经过分帧后，每一帧都有对应音素的概率分布（Kaldi 中使用 [tri-phone](https://flashgene.com/archives/105296.html)，直接说音素比较好理解）。作者根据目标指令和原始语音找到一个最好的对齐方式，目的是为了**使得修改量最小**；

   (2) **Back-propagation in Feature Extraction**：语音识别过程给网络的一般是 MFCC、Mel-log Filter Bank 等语音特征，把它简单地理解成是一张**二维热力图**，算法需要把梯度从这个特征回传到时域信号。（Kaldi 不像 tensorflow 那样直接就帮你把梯度计算好了，所以作者去推导了相关的梯度计算公式。不过，<u>这里作者只推导了对数能量谱的梯度，但是 WSJ 里面用的应该是 MFCC 才对。另外不清楚作者用的是优化器，还需要看一下 Kaldi 代码。</u>）。

   (3) 👍  **Hearing Thresholds**：**心理声学掩蔽效应**，可以计算出音频各个时间、各个频率点的**能量掩蔽值**，只要修改量不超过这个值，那么人就不会察觉。下图展示了在 1kHz 处 60dB 信号的能量掩蔽曲线（黑色），绿色的为人耳能够感受到声音的最小能量（如 20kHz 的声音，至少要达到 70dB 我们才听得到）：

   <img src="pictures/image-20201205193855101.png" alt="image-20201205193855101" style="zoom: 25%;" />

   作者计算样本的能量变化 D，并期望 D 在任何时间、频率点均小于掩蔽阈值 H，公式如下（<u>论文中的公式有个小错误，f 应该是 k</u>）：

   <img src="pictures/image-20201201101756118.png" alt="image-20201201101756118" style="zoom: 27%;" />

   变量 <img src="pictures/image-20201201101959466.png" alt="image-20201201101959466" style="zoom: 20%;" /> 度量能量变化 D 和 掩蔽阈值 H 之间的差值。如果 D 在任何点都不能超过 H ，这样的限制条件过于苛刻，可能会导致无法生成对抗样本。故作者添加一个系数来放宽这个限制条件，公式如下：

   <img src="pictures/image-20201201102258486.png" alt="image-20201201102258486" style="zoom: 11%;" />

   <img src="pictures/image-20201201102441645.png" alt="image-20201201102441645" style="zoom: 8%;" />

   将 <img src="pictures/image-20201201101959466.png" alt="image-20201201101959466" style="zoom: 20%;" /> 小于 0 的值置为 0 并归一化到 0~1，公式如下：

   <img src="pictures/image-20201201103739369.png" alt="image-20201201103739369" style="zoom: 63%;" />

   <img src="pictures/image-20201201103829555.png" alt="image-20201201103829555" style="zoom: 33%;" />

   只添加 <img src="pictures/image-20201201101959466.png" alt="image-20201201101959466" style="zoom: 20%;" /> 到梯度回传中时，作者发现差点意思。将 H 归一化到 0~1，公式如下：

   <img src="pictures/image-20201201104415164.png" alt="image-20201201104415164" style="zoom: 60%;" />

   最后作者将这两个系数结合到 DFT 的梯度回传上（声学特征的计算这里不做解释了，推荐 [Mel Frequency Cepstral Coefficient (MFCC) tutorial](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)），公式如下：

   <img src="pictures/image-20201201105201150.png" alt="image-20201201105201150" style="zoom: 25%;" />

   我对这一块的理解：**整体来看，作者想要使用 ”心理声学掩蔽效应“ 来生成更具隐藏性（或者说修改量小）的对抗样本，他认为 “当前掩蔽值大” 、并且“修改量远小于掩蔽值” 的点可以添加更多的扰动，回传的梯度可以更大；而 “当前掩蔽值小”、或者是 “修改量已经接近掩蔽值“ 的点不应该再做更多的修改，回传的梯度趋近于 0 。相对而言，我更喜欢 ”Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition“ 的工作，他们直接将 ”心理掩蔽效应“ 添加到了 loss 函数中去，让模型自己来选择梯度的变化。**

   最后思考一个问题：**这样用掩蔽阈值和 perturbation 的差值来度量真的是一种好的方法吗？可能不是，我们其实更希望的是去度量 频率掩蔽曲线 的变化有多大。举例来说，计算掩蔽阈值的时候首先得到的是 masker（可以理解为一个频率点，其能量是个极值点），我们在 masker 处增加能量来抬高 masker（完全可以做到增加的能量低于 masker 处的掩蔽值，因为这个点的掩蔽值等于 masker 的能量，这个值是很大的），这样人耳的听觉感受已经发生了改变。但是如果要这么来做，就要用可求解梯度的方法来实现 ”计算掩蔽值“ 的过程，过程实在是很复杂，这也可能是大家不这么做的原因（代价太大，做出来还不知道能不能收敛，效果好不好）。**

3. 👍  Evaluation：

   (1) 目标指令：

   <img src="pictures/image-20201201144854938.png" alt="image-20201205194302490" style="zoom: 33%;" />

   (2) 原始音频：Speech (从 WSJ 数据集中获取) + Music

   (3) 评估指标：**WER** 和 **平均修改能量**，前者越小越好，后者越大越好；

   (4) 分析 Hearing Threshold 和 Forced Alignment 的效果，学习率 **0.05 (这个和其他的工作相差挺大，猜测可能是因为有像 librosa 那样的 normalization)**，迭代 **500** 轮：

   <img src="pictures/image-20201201150329215.png" alt="image-20201201150329215" style="zoom:38%;" />

   (5) 分析单位时间嵌入词数量的影响：

   <img src="pictures/image-20201201151047824.png" alt="image-20201201151047824" style="zoom:30%;" />

   (6) 分析迭代轮数的影响：

   <img src="pictures/image-20201201151217121.png" alt="image-20201201151217121" style="zoom: 53%;" />

   (7) 和 CommandSong 进行对比，对比的指标为 SNR ：

   <img src="pictures/image-20201201151416298.png" alt="image-20201201151416298" style="zoom: 36%;" />

   

### Links

- 论文链接：[Schönherr, Lea, et al. "Adversarial attacks against automatic speech recognition systems via psychoacoustic hiding." *arXiv preprint arXiv:1808.05665* (2018).](https://arxiv.org/abs/1808.05665)
- 论文主页：[Adversarial Attacks (adversarial-attacks.net)](https://adversarial-attacks.net/)
- 论文代码：[rub-ksv/adversarialattacks: Adversarial Attacks (github.com)](https://github.com/rub-ksv/adversarialattacks)





## * Targeted adversarial examples for black box audio systems

### Contribution

1. 黑盒、有目标的攻击 DeepSpeech 的对抗攻击算法；
2. 结合遗传算法和梯度下降算法 (**在语音上没人这么来做过，但是其实在图像识别上面这种黑盒攻击不是新鲜事了，所以在算法上面没有创新型，而且直接应用到语义领域，出现了 query 次数巨大的问题**)；
3. 思路是值得借鉴的，黑盒攻击一定有比白盒攻击更加 interesting 的问题，但是不能照搬图像领域的方法；

### Notes

1. **黑盒**、**有目标**的对抗攻击算法。攻击的模型为 **DeepSpeech** 模型，选择的方式是 **遗传算法和梯度下降算法的结合**；

2. 从 CommonVoice 测试机中挑选出前100个样本作为原始音频，目标指令都是 **2** 个词的指令，比较短；

3. 作者假设 DeepSpeech 模型是不可知探知的，但是知道模型最后的概率分布的输出，并且针对 Greedy Decoding 进行攻击 （我的想法：**这样的攻击场景其实是不常见的，所以这个工作可能指导意义不大，但是我们应该思考一下，如果 ASR 模型经过了 LM 模型的修饰，还能不能用黑盒探测的方法来生成对抗样本？如果能，代价又有多大？**）；

4. 算法流程：

   <img src="pictures/image-20201202110759945.png" alt="image-20201202110759945" style="zoom: 55%;" />

   (1) 当样本解码的字符串距离目标指令较大时，使用遗传算法生成对抗样本，遗传算法的评分函数使用 CTC-Loss，其变异概率 p 由函数 `MomentumUpdate`进行更新；

   <img src="pictures/image-20201202112220375.png" alt="image-20201202112220375" style="zoom: 28%;" />

   (2) 当样本解码的字符串距离目标指令较小时，使用黑盒-梯度下降算法生成对抗样本，对每个序列样本点 ( <u>花费巨大，对于一个 16kHz 的 5s 语音，每轮都要调用目标模型进行解码 80k 次</u> ) 都分别添加小的扰动，根据 CTC-Loss 值的变化，确定扰动的影响是正面的还是负面的、是重要的还是不重要的；

   <img src="pictures/image-20201202112442039.png" alt="image-20201202112442039" style="zoom:22%;" />

5. Evaluation：

   (1) 使用 100 条原始语音，每个语音的目标指令是随机从 1000 个最常用的英语单词中抽取的 2 个单词，每个对抗样本，设置生成 3000 轮；

   (2) 使用 Success Rate 来评估成功率，对抗样本的成功率为 35%；使用 Cross Correlation Coefficient 来评估相似性，对抗样本与原始语音的相似性为 94.6% (这里只看成功的对抗样本)；

### Links

- 论文链接：[Taori, Rohan, et al. "Targeted adversarial examples for black box audio systems." *2019 IEEE Security and Privacy Workshops (SPW)*. IEEE, 2019.](https://arxiv.org/abs/1805.07820)
- 论文代码：[rtaori/Black-Box-Audio: Targeted Adversarial Examples for Black Box Audio Systems (github.com)](https://github.com/rtaori/Black-Box-Audio)





## Robust Audio Adversarial Example for a Physical Attack

### Contribution

1. 引入脉冲响应；
2. 实现了较高的物理攻击成功率，并且用了两组播放和接收设备；
3. 指令过短，实验应该增加更多的物理环境；

### Notes

1. **白盒**、**有目标**的**针对物理环境**的对抗攻击算法。攻击的模型为 **DeepSpeech** 模型，选取的指令都比较**短**；

2. 在图像领域的对抗攻击算法中，[Athalye et al., 2018] 等人提出了用一个抽象函数 t 来模拟物理环境下打印和拍照在样本上带来的扰动。将这个抽象函数 t结合到对抗样本的生成过程中去，可以大大增强生成的对抗样本的鲁棒性；

3. 作者提出的方法，关键点有三个：

   (1) **带通滤波器**。因为人的听觉频率范围是有限的，听筒-扬声器在工作的时候很多会直接丢弃其他频率的能量，所以作者设置了一个 **1000~4000** 范围的带通滤波器来减少修改量（我的看法：**我觉得 4000 这个上界是比较靠谱的，而 1000 这个下界可能并不合理，因为语音中低频的能量是比较多的，这部分的能量应该也是比较重要的；而这种带通滤波器的方法是否真的能够减少修改量也是存在问题的，因为依靠梯度下降算法，可能你限制了它修改的频带范围，需要的修改量可能是更多的**）。形式化公式如下：

   <img src="pictures/image-20201201234713041.png" alt="image-20201201234713041" style="zoom:22%;" />

   (2) 👍  **脉冲响应**。作者在生成对抗样本的过程中，添加脉冲响应的卷积来增强对抗样本对不同房间环境的鲁棒性。形式化公式如下：

   <img src="pictures/image-20201201235718462.png" alt="image-20201201235718462" style="zoom: 21%;" />

   (3) **高斯白噪声**。作者在生成对抗样本的过程中，添加高斯白噪声来增强对抗样本对背景白噪声的鲁棒性。形式化公式如下：

   <img src="pictures/image-20201202000105905.png" alt="image-20201202000105905" style="zoom: 27%;" />

4. Evaluation：

   (1) 作者其他的实现的细节与文章 [”Audio Adversarial Examples: Targeted Attacks on Speech-to-Text“](#Audio Adversarial Examples: Targeted Attacks on Speech-to-Text) 是一样的，Adam 迭代器和 CTC-Loss 函数；

   (2) 提到了一个比较有意思的攻击场景：FM radio；

   <img src="pictures/image-20201202001721712.png" alt="image-20201202001721712" style="zoom:35%;" />

   (3) 分析针对 API 的攻击：

   <img src="pictures/image-20201202002001695.png" alt="image-20201202002001695" style="zoom:28%;" />

   (4) 分析针对物理的攻击：

   <img src="pictures/image-20201202002630244.png" alt="image-20201202002630244" style="zoom:33%;" />

   每个样本尝试 10 次（**测试次数太少了**），统计成功率，并且发现只保证成功率高于 50% 的情况下，可以适当减少修改量。

### Links

- 论文链接：[Yakura, Hiromu, and Jun Sakuma. "Robust audio adversarial example for a physical attack." *arXiv preprint arXiv:1810.11793* (2018).](https://arxiv.org/abs/1810.11793)

- 论文主页：[Robust Audio Adversarial Example for a Physical Attack (yumetaro.info)](https://yumetaro.info/projects/audio-ae/)

- 论文代码：[hiromu/robust_audio_ae: Robust Audio Adversarial Example for a Physical Attack (github.com)](https://github.com/hiromu/robust_audio_ae)

- 冲击响应：

   [The reverb challenge: A common evaluation framework for dereverberation and recognition of reverberant speech (ieee.org)](https://ieeexplore.ieee.org/document/6701894)

   [A binaural room impulse response database for the evaluation of dereverberation algorithms (ieee.org)](https://ieeexplore.ieee.org/document/5201259)

   [Acoustical Sound Database in Real Environments for Sound Scene Understanding and Hands-Free Speech Recognition (naist.jp)](https://library.naist.jp/dspace/handle/10061/8220)

   [Evaluation of speech dereverberation algorithms using the MARDY database (2006) (ist.psu.edu)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.727.1869)

   [Acoustic measurement data from the varechoic chamber (nist.gov)](https://www.nist.gov/document-14705)





## Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition

### Contribution

1. 白盒、有目标的、针对端到端 LAS 模型的对抗攻击算法；
2. 心理掩蔽效应；
3. 模拟房间声学响应；

### Notes

1. **白盒**、**有目标**的对抗攻击算法。攻击的模型为  Lingvo 框架的 **LAS** 模型，攻击的指令选取了 1000 条**中等长度**的字符串;

2. 关键点在于两方面，一点是使用了**心理掩蔽效应**来提高对抗样本的隐蔽性，另一点是**模拟房间声学响应**来提高对抗样本的鲁棒性;

3. **心理掩蔽效应，（简单来讲）就是能量大的声音可以频闭能量小的声音，主要分为时间掩蔽和频率掩蔽**。与 [“Adversarial Attacks Against Automatic Speech Recognition Systems via Psychoacoustic Hiding”](#Adversarial Attacks Against Automatic Speech Recognition Systems via Psychoacoustic Hiding) 相同，作者也用频率掩蔽效应。添加心理掩蔽效应后的 loss 函数：
   
    <img src="pictures/Snipaste_2020-11-29_15-54-36.png" alt="Snipaste_2020-11-29_15-54-36" style="zoom: 40%;" />
    
    <img src="pictures/Snipaste_2020-11-29_15-57-05.png" alt="Snipaste_2020-11-29_15-57-05" style="zoom: 35%;" />
    
    
    前面部分保证**样本的成功率**，后面部分保证**样本的隐藏性**，**alpha** 控制两者的权重。作者生成对抗样本的时候有一个 **trick**（因为作者把两个放在一起时发现很难生成对抗样本）：( **Stage-1** ) 先根据前面的 loss 函数生成一轮对抗样本，( **Stage-2** ) 然后根据后面的 loss 函数生成一轮对抗样本，如果 stage-2 迭代 **20** 轮后，成功生成了对抗样本，那就把 alpha 增大一些（**说明可以增加一些隐藏性**）；如果 stage-2 迭代 **50** 轮，都没能生成对抗样本，那就把 alpha 减小一些（**说明需要牺牲一些隐藏性**）。具体的迭代生成算法如下：
    
    <img width="33%" src="./pictures/Snipaste_2020-11-29_16-45-46.png"/>
    
4. 模拟房间声学响应，简单来说，当固定了房间的参数和你设备的参数，你可以将整个物理信道用一个函数 t(x) 来建模。添加房间声学响应后的 loss 函数：

    <img src="./pictures/image-20201129170241799.png" alt="image-20201129170241799" style="zoom: 45%;" />

    训练的 **trick** ：( **Stage-1** ) 使用  `lr_1=50` 迭代 *2000* 轮保证在其中 **1 个房间声学响应**下能够生成对抗样本，( **Stage-2** ) 然后使用 `lr_2=5` 迭代 *5000* 轮来保证在另外随机采样的 **10 个房间声学响应**下都能够生成对抗样本（这个期间不再减小 **perturbation** 的上限）。

5. 👍  结合心理掩蔽效应和模型房间声学响应。结合后的 loss 函数:

    <img src="./pictures/image-20201129180621070.png" alt="image-20201129180621070" style="zoom: 65%;" />

    训练的 **trick** ：( 在 **4** 的对抗样本基础上 ) 结合整个 loss 函数来生成具有隐藏性的对抗样本，和 3 中的分两步生成不同。

6. Evaluation 部分，我觉得封面给的 100% 其实不是关键，因为该实验给的样本只是使用了心理掩蔽效应，考虑到这个攻击是个白盒攻击，所以白盒的非物理攻击其实在现实中意义不大，但是可以去论文主页听一下效果，<u>确实添加的扰动几乎无法听到</u>。所以主要还是关注它的第二个实验结果：

    <img src="./pictures/image-20201129183521406.png" alt="image-20201129183521406" style="zoom: 60%;" />

    作者采用的对抗样本的评价指标分别是：Accuracy - 整句话的成功率，WER - 词错率，和隐藏性。其中隐藏性没有采用常用的 SNR 来度量，而是直接采用**问卷调查**的形式，作者的问卷调查的问题分别为：

    (1) 音频是否清晰；

    (2) 分辨两个音频哪一个是原始音频；

    (3) 判断两个音频是否相同；

### Links
- 论文链接：[Qin, Yao, et al. "Imperceptible, robust, and targeted adversarial examples for automatic speech recognition." International Conference on Machine Learning. PMLR, 2019](https://arxiv.org/abs/1903.10346).

- 论文主页：[Imperceptible, Robust and Targeted Adversarial Examples for Automatic Speech Recognition (ucsd.edu)](http://cseweb.ucsd.edu/~yaq007/imperceptible-robust-adv.html)
- 论文代码：[cleverhans/examples/adversarial_asr at master · tensorflow/cleverhans (github.com)](https://github.com/tensorflow/cleverhans/tree/master/examples/adversarial_asr)





