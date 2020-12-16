# Competition in NLP



[TOC]



## 新手上路: (天池) 零基础入门NLP - 新闻文本分类

### 赛题相关

- 训练集: 20W; 目标样本集: 5W;
- 分类标签: 14个;

| 科技   | 股票   | 体育   | 娱乐   | 时政   | 社会   | 教育   | 财经   | 家居   | 游戏   | 房产   | 时尚   | 彩票   | 星座   |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 0.1945 | 0.1847 | 0.1571 | 0.1106 | 0.0750 | 0.0611 | 0.0499 | 0.0442 | 0.0392 | 0.0293 | 0.0246 | 0.0156 | 0.0091 | 0.0045 |

- 数据经过脱敏, 全用 索引 代替;
- 每个新闻段落的文字长度分布: 2 到 57921不等;

| 675  | 748  | 826  | 914  | 1013 | 1130 | 1275 | 1472 | 1795 | 2457 | 57921 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- |
| 50%  | 55%  | 60%  | 65%  | 70%  | 75%  | 80%  | 85%  | 90%  | 95%  | 100%  |

### 遇到的问题

> 我对 NLP 不是很了解, 所以做题前问了同学几个比较 **基础** 的问题 🤣.

### 实现过程

#### TextCNN

1. 实验结果:

   ```shell
   Epoch 1/10
   2672/2672 - 1095s - loss: 0.5272 - accuracy: 0.8741 - val_loss: 0.2553 - val_accuracy: 0.9362
   Epoch 2/10
   2672/2672 - 1095s - loss: 0.3031 - accuracy: 0.9240 - val_loss: 0.2288 - val_accuracy: 0.9396
   Epoch 3/10
   2672/2672 - 1094s - loss: 0.2718 - accuracy: 0.9319 - val_loss: 0.2180 - val_accuracy: 0.9440
   Epoch 4/10
   2672/2672 - 1092s - loss: 0.2474 - accuracy: 0.9379 - val_loss: 0.2194 - val_accuracy: 0.9441
   Epoch 5/10
   2672/2672 - 1092s - loss: 0.2298 - accuracy: 0.9430 - val_loss: 0.2144 - val_accuracy: 0.9487
   Epoch 6/10
   2672/2672 - 1096s - loss: 0.2162 - accuracy: 0.9470 - val_loss: 0.2260 - val_accuracy: 0.9442
   Epoch 7/10
   2672/2672 - 1095s - loss: 0.1955 - accuracy: 0.9520 - val_loss: 0.2297 - val_accuracy: 0.9472
   Epoch 8/10
   2672/2672 - 1095s - loss: 0.1856 - accuracy: 0.9556 - val_loss: 0.2226 - val_accuracy: 0.9467
   Epoch 9/10
   2672/2672 - 1096s - loss: 0.1743 - accuracy: 0.9584 - val_loss: 0.2324 - val_accuracy: 0.9489
   Epoch 10/10
   2672/2672 - 1087s - loss: 0.1655 - accuracy: 0.9615 - val_loss: 0.2343 - val_accuracy: 0.9466
   ```

3. 提交结果: `0.9393`

#### TextCNN + Pre-trained Word2Vec Embedding

1. 实验结果:

   ```shell
   Epoch 1/25
   2672/2672 - 1080s - loss: 1.1481 - accuracy: 0.6738 - val_loss: 0.6634 - val_accuracy: 0.8097
   Epoch 2/25
   2672/2672 - 1089s - loss: 0.9464 - accuracy: 0.7137 - val_loss: 0.6362 - val_accuracy: 0.8121
   Epoch 3/25
   2672/2672 - 1086s - loss: 0.9360 - accuracy: 0.7143 - val_loss: 0.6417 - val_accuracy: 0.8107
   Epoch 4/25
   2672/2672 - 1094s - loss: 0.9261 - accuracy: 0.7184 - val_loss: 0.6391 - val_accuracy: 0.8137
   Epoch 5/25
   2672/2672 - 1095s - loss: 0.9188 - accuracy: 0.7206 - val_loss: 0.6284 - val_accuracy: 0.8168
   Epoch 6/25
   2672/2672 - 1095s - loss: 0.9081 - accuracy: 0.7234 - val_loss: 0.6292 - val_accuracy: 0.8176
   Epoch 7/25
   2672/2672 - 1095s - loss: 0.9101 - accuracy: 0.7234 - val_loss: 0.6186 - val_accuracy: 0.8222
   Epoch 8/25
   2672/2672 - 1093s - loss: 0.8789 - accuracy: 0.7348 - val_loss: 0.5981 - val_accuracy: 0.8324
   ... ...
   Epoch 25/25
   2672/2672 - 1094s - loss: 0.8803 - accuracy: 0.7332 - val_loss: 0.5908 - val_accuracy: 0.8350
   ```

2. Pre-trained word2vec 似乎没能实现比较好的效果, 暂时有待解决;

#### FastText