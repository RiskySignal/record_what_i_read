# Outlier Detection



## Bilibili-异常样本检测算法详解

> 思考：能否应用到我们的工程中来；

视频链接：https://www.bilibili.com/video/BV1rq4y1C7R3

群组异常样本：其特征是出现的个体会在视图中呈现聚集；

离群点异常样本：不同于其他样本的分布；

时间序列异常样本：其特征是随着时间的变化，出现异常；

### FRAUDAR —— 群组异常检测算法

![image-20220512211718741](pictures/image-20220512211718741.png)



### SLICENDICE —— 群组异常检测算法

![image-20220512233742910](pictures/image-20220512233742910.png)



### SDNE —— 群组异常检测算法

![image-20220516120124502](pictures/image-20220516120124502.png)



### ONE —— 离群点异常检测算法

![image-20220516133802070](pictures/image-20220516133802070.png)

![image-20220516143819893](pictures/image-20220516143819893.png)

![image-20220516143946521](pictures/image-20220516143946521.png)



### DAGMM —— 离群点异常检测算法

![image-20220516145257359](pictures/image-20220516145257359.png)



### MSCRED —— 时序异常检测算法

![image-20220516151227824](pictures/image-20220516151227824.png)

![image-20220517184736032](pictures/image-20220517184736032.png)



### TadGAN —— 时序异常检测算法

![image-20220517184915957](pictures/image-20220517184915957.png)

![image-20220518003321450](pictures/image-20220518003321450.png)

![image-20220518003528687](pictures/image-20220518003528687.png)

![image-20220518003602650](pictures/image-20220518003602650.png)



### TS2Vec —— 时序异常检测算法

![image-20220518004323170](pictures/image-20220518004323170.png)

![image-20220518091300634](pictures/image-20220518091300634.png)

![image-20220518091623968](pictures/image-20220518091623968.png)

![image-20220518093138094](pictures/image-20220518093138094.png)

![image-20220518093452291](pictures/image-20220518093452291.png)



### FlowScope —— 资金关系异常检测算法

![image-20220518104427095](pictures/image-20220518104427095.png)



## Detecting Lateral Movement in Enterprise Computer Networks with Unsupervised Graph AI

> 如何处理网页间的关联关系？如何无穷尽的网页边界？

### Contribution

1. GNN实现了横移检测；

### Notes

1. 算法流程：

   ![image-20220523094431455](pictures/image-20220523094431455.png)

2. 实验结果：

   <img src="pictures/image-20220523111533662.png" alt="image-20220523111533662" style="zoom: 25%;" />

3. 最后给出了三条可行的减少误报的方案：
   1. 只报告首次异常登录事件；
   2. 只报告和用户相关的异常登录事件；
   3. 只报告和其他异常事件相关联的异常登录事件； -> 这一点比较有用、可借鉴；

### Links

- 论文链接：[Bowman B, Laprade C, Ji Y, et al. Detecting lateral movement in enterprise computer networks with unsupervised graph {AI}[C]//23rd International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2020). 2020: 257-268.](https://www.usenix.org/system/files/raid20-bowman.pdf)



## EULER: Detecting Network Lateral Movement via Scalable Temporal Link Prediction

> 如何和用户行为基线进行关联？无法关联的话，可能无法使用用户群组来减少模型误报；
>
> 如何处理非固定IP的情况？数据中，如果无法标识每一个用户，可能无法使用这个模型；
>
> 如何溯源横移异常？某个用户指向某个节点的异常访问会被标识；
>
> 为什么需要分布式的worker？分布式的worker能实现多大的效率提升？

### Contribution

1. 提出了一种分布式的基于GNN和RNN的横移检测算法，提高了模型检测执行的效率的同时，保证了相近的模型检测能力；
2. 主要的贡献可能集中在GNN和RNN这种多worker的联合训练框架上；

### Notes

1. 模型架构：

   ![image-20220522154630298](pictures/image-20220522154630298.png)

2. 流量横移检测：在LANL 2015数据集上进行测试

   <img src="pictures/image-20220522233758612.png" alt="image-20220522233758612" style="zoom: 33%;" />

### Links

- 论文链接：[King I J, Huang H H. EULER: Detecting Network Lateral Movement via Scalable Temporal Link Prediction[J]. NDSS 2022.](https://www.ndss-symposium.org/ndss-paper/auto-draft-227/)
- 论文代码：https://github.com/iHeartGraph/Euler
- LANL 2015数据集：https://csr.lanl.gov/data/cyber1/
