---
layout: page
title: Survey
permalink: /Survey/
image: 08.jpg
---



### 1 Hand-crafted Prompt

基于人工知识来定义prompt模板

- prefix prompt，在一个前缀的基础上填后续文本

- cloze prompt，在句子中填空
- 人工构造的prompt依赖人工经验，并且效果也难以保障，一般采用构造多组prompt，对每组prompt的效果分别进行验证对比，或者多组prompt融合的方法提升效果

#### 1.1 methods

***Language Models as Knowledge Bases?（2019）***

- 探讨预训练语言模型中学习到的语言知识
- 利用多种数据集构造cloze prompt，看预训练模型是否能预测出缺失词。例如Dante was born in (?)就是一个cloze prompt，模型预测空缺位置的词，预测正确说明预训练语言模型学到了这些知识
- 利用一些知识库构造了一批cloze prompt去对比不同预训练模型的效果，并发现Bert-Large能够取得很好的效果。这篇文章最后构造了LAMA数据集，根据多个数据得到的cloze prompt模板，用来检验预训练语言模型中包含的知识情况

***Template-Based Named Entity Recognition Using BART（ACL 2021）***

- 采用cloze prompt解决小样本下的NER任务
- 对于一个句子，如果某个词组是实体，那么其对应的模板就是$<x_i:j> is\ a <y_k>$；如果某个词组不是实体，那么其对应的模板为$<x_i:j> is\ not\ an\ entity$。例如对于一个输入文本*ACL will be held in Bangkok*来说，需要构造出多组模板文本，对应每个词组是否为某个entity，如*Bangkok is a location entity*。将构建好的句子送到预训练+其他领域finetune的BART上打分，根据打分高低判断每句话描述的对应词是否为实体的正确性

<img src="https://oss-emcsprod-public.modb.pro/wechatSpider/modb_20220812_52325dfa-1a2e-11ed-aef3-fa163eb4f6be.png" alt="img" style="zoom:50%;" />

***The Power of Scale for Parameter-Efficient Prompt Tuning（2021）***

- 将所有NLP任务都看成是文本生成任务，通过加入prefix prompt给模型一个额外的条件信息来指导模型生成后续的文本
- 当给定一个输入token序列后，正常对这些token进行embedding，然后拼接在prefix prompt的embedding之后，输入到encoder-decoder中
- 对于prompt embedding的初始化，除了随机初始化外，本文提出了采用预训练模型中某个word的embedding作为初始化，或者对于分类问题使用分类标签对应的token embedding进行初始化，指导生成阶段产出相应类别对应的文本

<img src="https://oss-emcsprod-public.modb.pro/wechatSpider/modb_20220812_526b98b8-1a2e-11ed-aef3-fa163eb4f6be.png" alt="img" style="zoom:50%;" />

***Language Models are Few-Shot Learners（2020）***

- 设计了多种prefix prmopt模板用来解决各种NLP任务。例如下面的例子中将翻译任务转换成了prompt，让模型预测句子末尾的单词，在文前提供了对于任务的描述文本。

<img src="https://oss-emcsprod-public.modb.pro/wechatSpider/modb_20220812_52c7205c-1a2e-11ed-aef3-fa163eb4f6be.png" alt="img" style="zoom:50%;" />

***Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference（ACL 2021）***

- 采用 cloze prompt + finetune 的方法解决文本分类问题
- 对于每种任务，会构造一组PVP。这里的PVP有两个组成部分，一个用来将该任务的输入样本转换成一个cloze prompt任务，另一个用来将分类的label映射到一个具体的word上。例如某个判断两个句子是否矛盾的任务样本为(Mia likes pie, Mia hates pie)，那么会将其转换为Mia likes pie? , [x] Mia hates pie，目标为预测[x]位置应该填什么词。同时，分类任务的每个label都会映射到某一个词上
- 首先会使用少量这样的数据进行finetune，然后在inference阶段预测[x]位置为各个label对应词的概率，选概率最大的
- 由于不同的prompt构造方法会影响效果，本文采用了一种知识蒸馏的方法，对于一个任务会构造多个prompt，每一个prompt finetune生成一个模型，最后使用知识蒸馏的方法融合各个finetune模型的预测结果
- 为了让各个prompt模板的信息能够融合，会进行多轮的finetune，每轮每个prompt的finetune数据使用上一个某些finetune模型产出的标签扩充训练数据

<img src="https://oss-emcsprod-public.modb.pro/wechatSpider/modb_20220812_52cb78fa-1a2e-11ed-aef3-fa163eb4f6be.png" alt="img" style="zoom:50%;" />



### 2 Generation Prompt

自动化的构造出大量的prompt模板（还是Hard prompt）

- 在大量语料中去挖掘一些可以作为prompt的模板。例如当确定了prompt的输入X和label对应的Y后，可以去海量文本库做匹配，看哪些句子包含[X] ... [Y]，就用这些模板作为构造prompt的依据，***How Can We Know What Language Models Know?（2019）***一文中就采用了这种方法
- prompt paraphrasing，首先生成一个种子prompt，然后在此基础上利用一些诸如回译（翻译成一种语言再翻译回来）、关键词替换等方法，扩展出更多的prompt模板，然后对比各个prompt模板的效果选择最优的prompt模板，或对多个prompt模板结果进行融合
- 将prompt模板的构造看成是一个生成式的任务

#### 1.1 methods

***AutoPrompt: Eliciting knowledge from language models with automatically generated prompts（2020）***

- 提出了采用自动搜索prompt模板词的方法
- 从词表中遍历所有词，看哪些词组成的prompt模板能最终生成训练数据中待填充的词，相当于一个逆向操作
- Prompt模板需要填充的词最开始用[MASK]初始化，然后去看使用其他词替换[MASK]会让label的概率最大，逐步替换[MASK]，得到template

<img src="https://oss-emcsprod-public.modb.pro/wechatSpider/modb_20220812_52f4d2f4-1a2e-11ed-aef3-fa163eb4f6be.png" alt="img" style="zoom: 67%;" />

***Making pre-trained language models better few-shot learners（2021）***

- 提出不同的label到word的映射方法，以及不同的prompt模板会对效果造成很大影响
- 对于label的选择，使用每个类别的训练样本，让预训练的语言模型预测[MASK]为各个词的概率，选择概率最大的几个词。这几个词都会作为label到word的映射方法，分别finetune模型，最终再选择在验证集上表现最好的
- 对于prompt模板的生成，本文使用了预训练的T5模型。T5模型在预训练阶段采用了mask span任务，输入一个被mask掉多个span的文本，在decoder处对mask掉的span进行还原，这正好可以用于prompt生成。具体例子如下图，对于任务的每个类别构造如下的输入，将prompt部分的除了已经确定好的label对应的词外都mask掉，让T5模型去生成各个模板的各个位置应该填什么，最后再进行finetune看哪个生成的prompt效果最好

<img src="https://oss-emcsprod-public.modb.pro/wechatSpider/modb_20220812_52f9d434-1a2e-11ed-aef3-fa163eb4f6be.png" alt="img" style="zoom:50%;" />



### 3 隐空间中的prompt

- 相比于文本prompt，隐空间的prompt不需要强制让prompt模板必须是真实的文本表示，而是在隐空间学习一个文本向量，它可能无法映射到具体的单词，但是和各个词的embedding在同一个向量空间下

- 这种自动生成的prompt也可以不用保证必须是真实的文本，给prompt的生成带来了更大的灵活空间

#### 3.1 methods

***Prefix-Tuning: Optimizing Continuous Prompts for Generation（2021）***

- 只finetune 0.1%的参数就取得和finetune相当的效果，并且在少样本任务上效果由于finetune
- 针对自然语言生成任务（如摘要生成、table-to-text等任务）的迁移预训练大模型的方法
- 具体实现，将预训练的Transformer模型参数整体Freeze住，当正常输入文本序列的时候，在最前端添加几个prefix id，每一个prefix id都对应一个随机初始化的embedding，不同的任务有不同的prefix id。这样在模型中，prefix之后每个时刻的表示都会受到prefix的影响，prefix代表某个对应具体任务的上下文信息。在Finetune过程中，模型的其他参数都Freeze，只Finetune prefix的embedding，以及prefix后面接的一些全连接层，Finetune参数量只占整体模型的0.1%，远小于其他的Finetune方法。该方法核心思想利用了prefix embedding去学习对于某个任务来说，需要从预训练语言模型中提取什么样的信息。这里用到的prompt并非直接生成具体的文本模板，而是在向量空间中生成的一个embedding，是一个隐式的prompt模板，每个任务都有一个隐式的prefix prompt

<img src="https://oss-emcsprod-public.modb.pro/wechatSpider/modb_20220812_52ff8744-1a2e-11ed-aef3-fa163eb4f6be.png" alt="img" style="zoom: 67%;" />

***The Power of Scale for Parameter-Efficient Prompt Tuning（2021）***

- 将所有NLP任务都看成是文本生成任务，通过加入prefix prompt给模型一个额外的条件信息来指导模型生成后续的文本
- 当给定一个输入token序列后，正常对这些token进行embedding，然后拼接在prefix prompt的embedding之后，输入到encoder-decoder中
- 对于prompt embedding的初始化，除了随机初始化外，本文提出了采用预训练模型中某个word的embedding作为初始化，或者对于分类问题使用分类标签对应的token embedding进行初始化，指导生成阶段产出相应类别对应的文本

<img src="https://oss-emcsprod-public.modb.pro/wechatSpider/modb_20220812_526b98b8-1a2e-11ed-aef3-fa163eb4f6be.png" alt="img" style="zoom:50%;" />

- 另一种方式是在文本prompt的基础上进行finetune，相当于将明文prompt作为一个初始点，然后在embedding空间finetune一个更适用于当前任务的隐式prompt。**Factual Probing Is [MASK]: Learning vs. Learning to Recall（2020）**的基本思路也是在隐空间学习prompt模板中各个token的embedding，在此基础上，本文提出使用一些文本prompt模板进行初始化，也就是隐空间prompt包含的单词数量、位置以及初始化参数都用一个人工定义好的文本prompt，然后在此基础上进行finetune得到更好的prompt

***GPT Understands, Too（2021）***

- 采用的思路是将人工定义的prompt的token对应的embedding从预训练模型输出的改为一个可学习的hidden向量，让模型去优化。为了建立模板中各个token之间的关系，文中采用了双向LSTM这种序列建模的方式生成每个prompt token的表示，第i个token的向量可以表示为如下公式：

<img src="https://oss-emcsprod-public.modb.pro/wechatSpider/modb_20220812_530b77a2-1a2e-11ed-aef3-fa163eb4f6be.png" alt="img" style="zoom:50%;" />



<img src="https://oss-emcsprod-public.modb.pro/wechatSpider/modb_20220812_532ec5fe-1a2e-11ed-aef3-fa163eb4f6be.png" alt="img" style="zoom: 67%;" />