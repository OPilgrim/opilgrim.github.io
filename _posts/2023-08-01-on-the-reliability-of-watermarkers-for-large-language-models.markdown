---
layout: post
title:  On the Reliability of Watermarks for Large Language Models
date:   2023-08-01 15:58:35 +0300
tags:   Translation
lang: zh
---
![image-20230801160058799](https://cdn.jsdelivr.net/gh/OPilgrim/Typoter-TC/img/image-20230801160058799.png)

[arxiv](https://arxiv.org/pdf/2306.04634.pdf)

<!--这篇跟 Can AI 是同一个机构不同组的人做的，比那篇更新一些，跟Greeen List是同一作者-->

本文研究现实中使用水印的可靠性，结论是，带水印的文本在被人类重写、由非带水印的LLM转述或混合成更长的手写文档后，水印仍然是可检测的；此外提出了对短跨度水印敏感的新检测方案





### 3 How to improve watermark reliability?

本节针对 Green List 进行改进，前面稍微介绍了一下 Green List 

#### 3.1 Improved Hashing Schemes

Green List 的实验专注于一个简单的方案，其中随机数生成器使用$h = 1$进行播种，即仅使用位置$t−1$的单个token 来为位置 $t$ 的令牌上色。我们将此方案称为**LeftHash**。由于 green list 仅依赖于一个token，第三方观察者可以通过搜索位置$t$上的后续单词来学习与位置$t−1$的 token 相关的 green list ，这些单词在非水印分布下不太可能出现。在某些情况下，水印方案需要在API后面保密，因此需要更安全的方案。

Kirchenbauer等人[2023]也提到了一种方案(算法3)，其中位置 $t$ 的 green list 是通过包括位置 $t$ 本身(尚未生成)的令牌来确定的，除了$f$输入中$t$左边的令牌。我们称这种哈希方案为**SelfHash**。该方法有效地将上下文宽度$h$增加了$1$，使得暴力破解方法更难发现水印规则。我们将此方案推广到包括任意函数$f$和文本生成例程，我们将在附录中的**算法1**中详细描述。

当上下文宽度$h$增加以保持红/绿列表规则的保密性时，我们发现检测可靠性在很大程度上取决于哈希方案。我们定义了以下函数$f: \mathbb{N}^h→\mathbb{N}$，它们将标记{$x_i$}的空间映射到伪随机数上。每个都依赖于一个秘密值$s∈\mathbb{N}$和一个标准整数PRF $P: \mathbb{N→N}$。

- **Additive**：这是 Green List 原文提到的函数，这边将其扩展到$h>1$的情况，通过定义：$f_{Additive-LeftHash}(x)=P(s\sum^{h}_{i=1}{x_i})$。虽然上下文$x$的排列不会改变结果，但从$x$中删除或交换单个token会改变哈希值，从而破坏该token处的水印。

- **Skip**：这个函数在上下文中只使用最左边的标记：$f_{Skip-LeftHash}(x)=P(sx_h)$。该散列对于非最左边令牌的更改是健壮的，但它容易受到插入/删除的影响。

- **Min**：这个函数定义为：$f_{Min-LeftHash}(x)=min_{i \in 1,...,h}P(sx_i)$。它对上下文中的排列具有鲁棒性，对插入/删除具有部分鲁棒性。假设所有$P(sx_i)$都是伪随机的，并且同样可能是最小的值，那么该方案失败的可能性与从上下文中删除的值的数量成比例，即如果$h = 4$并且从上下文中删除/丢失了2个令牌，PRF仍然有50%的可能性生成相同的哈希。

**图2**显示，较小的上下文宽度$h$为机器释义提供了最佳的鲁棒性。在更宽的上下文宽度下，Skip和Min变体在攻击下仍然强大，而Additive变体则受到攻击。然而，我们看到这种鲁棒性的改进是以牺牲文本质量为代价的，因为最小方案产生的输出较少。尽管如此，在上下文宽度$h = 4$时，Min-SelfHash方案(棕色圆圈标记)实现了与宽度$h = 1$(黑色圆圈)时的原始Additive-LeftHash方案相同的多样性，同时更加鲁棒。这表明我们可以使用Min和SelfHash提供的额外强度来运行更长的上下文宽度，从而保护水印。在主体工作的其余部分中，我们将这两种方案分别称为“SelfHash”和“LeftHash”。在附录A.2中，我们进一步探讨了图式选择对文本质量的句法和语义方面的影响。

#### 3.2 Improved Watermark Detection











