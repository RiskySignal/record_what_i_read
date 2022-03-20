# Funny Implementation in ASR

## 神秘鸭

### 简介

神秘鸭可以通过智能语音助手（集成 Siri、小爱同学等于一体），来实现远程控制电脑等设备。可以执行的指令分为六种：打开、关闭、删除、创建、执行、回放；

网页截图：

<img src="pictures/image-20210101230919988.png" alt="image-20210101230919988" style="zoom: 80%;" />

### 原理

这里，我只有 iPhone，所以就关注的是 Siri 的远程控制实现原理。大致的过程可以分为 $k=5$ 个阶段：

(1) 用户唤醒 Siri 后，向 Siri 下达命令；

(2) Siri 收到语音，并且在 shortcuts 中找到对应的指令（shortcut）；

(3) Siri 根据 shortcut 定义好的上传格式，向服务器 `https://smya.cn/client/do` 发送一个 `put` 请求并且带上一个指令id `run_id=<...>` （一台设备上的一条指令对应一个指令id ）；

(4) 服务器接收到请求后，找到相应的设备，发出执行相应命令的指令；

(5) 已经与服务器认证连接（根据 **设备号和一个安全码** 与服务器取得信任连接，和 TeamViewer、向日葵等软件的原理可能是类似的）的设备在收到指令后执行；

### 链接

- 神秘鸭官网：https://smya.cn/





## 华为 AI 安全

一起看一下，华为对 AI 安全的整体布局：

![AI安全防御架构](pictures/ai-security-1.jpg)

### 链接

- 华为官网：https://www.huawei.com/cn/trust-center/ai-section



## AI 实时声音克隆

分享链接：[【有趣的项目分享】尝试用AI合成特朗普的语音](https://juejin.cn/post/6872260516966842382)

Github 项目：[Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)



## 智能客服

todo there



## 攻击版权检测系统

![](/Users/zjs/workspace/record_what_I_read/pictures/2022-03-12-10-36-06-image.png)

这篇文章虽然做得不怎么样，但是利用对抗攻击来攻击版权检测系统，确实是个十分有意思的方向；



## OPPO 语音唤醒实现方案

![](/Users/zjs/workspace/record_what_I_read/pictures/2022-03-12-11-23-00-image.png)

参考链接：https://www.shenzhenware.com/articles/13081

很有意思，语音唤醒的背后涉及非常多的技术，也增加了我们攻击这个系统的难度；



## 声学回声消除

参考链接：https://www.cnblogs.com/LXP-Never/p/11703440.html

文章详细讲解了声学回声消除的原理，和推荐使用的开源库；