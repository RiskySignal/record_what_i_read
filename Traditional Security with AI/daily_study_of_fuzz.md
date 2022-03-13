# Daily Study of Fuzz

> 从`README`可以发现我本身并不会传统安全相关技术，但是在室友的熏陶和公司项目（AI赋能安全相关）的支持下，我开始了我的Fuzz学习之旅。
>
> 我能感觉到，AI赋能安全必然会到来，作为一个AI安全的研究人员，我觉得不仅需要掌握AI内生安全的知识，同时应该掌握传统安全的知识。不过很遗憾，我们实验室现在在这两个方向上是相对孤立的，我作为AI内生安全的研究人员，被选择去支持AI赋能安全相关工作，得此契机，我决定鼓起勇气更加深入地学习一下Fuzz相关知识。我不期望我能像那些大佬一样，成为这个领域的专家，我的目标是，传统安全的专家给我一个攻击点，我能对这个方向有一定的认识，从而发现其与AI之间的结合点。
>
> 当然，我最近也在思考，对于一家企业，AI安全能力在软件开发流程（模型部署）的最佳实践位置，如何能够保证AI的安全性，又不会改变原有的软件开发流程，不增加开发人员的工作量。
>
> 一起加油吧，Ceres！



## Learning Path

### Day 1

<img src="pictures/image-20220314004009609.png" alt="image-20220314004009609" style="zoom:67%;" />

学习链接：https://foxglovesecurity.com/2016/03/15/fuzzing-workflows-a-fuzz-job-from-start-to-finish/

该链接主要使用 `AFL++` 来对 `YAML-CPP` 工程进行 Fuzz，**最新版本的 `YAML-CPP` 好像没有 Crash**，效果复现不出来，后面换到博主提供的版本继续复现博主的实验。

[@BrandonPrry](https://twitter.com/BrandonPrry) 在博客中分享了一个通过并行加快 `afl-tmin` 的工具，将其成为 `afl-ptmin` ，其代码如下：

```bash
#!/bin/bash
cores=$1
inputdir=$2
outputdir=$3
pids=""
total=`ls $inputdir | wc -l`

for k in `seq 1 $cores $total`
do
  for i in `seq 0 $(expr $cores - 1)`
  do
    file=`ls -Sr $inputdir | sed $(expr $i + $k)"q;d"`
    echo $file
    afl-tmin -i $inputdir/$file -o $outputdir/$file -- ~/parse &
  done

  wait
done
```

思考问题：

- Fuzz是什么，主要的流程是什么，关键点是什么？ ❓
- `AFL` 的原理？它和还有一些 Fuzz 工具有什么异同点？ ❓
- `afl-cmin` 和 `afl-tmin` 两个工具他们的区别的是什么？ ❓
- GDB 插件 `exploitable` 和 `CrashWalk` 分别是什么作用？ ❓
- `afl-cov` 是如何实现的？❓

