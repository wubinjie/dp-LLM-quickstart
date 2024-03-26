#!/usr/bin/env python
# coding: utf-8

# # Hugging Face Transformers 微调训练入门
# 
# 本示例将介绍基于 Transformers 实现模型微调训练的主要流程，包括：
# - 数据集下载
# - 数据预处理
# - 训练超参数配置
# - 训练评估指标设置
# - 训练器基本介绍
# - 实战训练
# - 模型保存

# ## YelpReviewFull 数据集
# 
# **Hugging Face 数据集：[ YelpReviewFull ](https://huggingface.co/datasets/yelp_review_full)**
# 
# ### 数据集摘要
# 
# Yelp评论数据集包括来自Yelp的评论。它是从Yelp Dataset Challenge 2015数据中提取的。
# 
# ### 支持的任务和排行榜
# 文本分类、情感分类：该数据集主要用于文本分类：给定文本，预测情感。
# 
# ### 语言
# 这些评论主要以英语编写。
# 
# ### 数据集结构
# 
# #### 数据实例
# 一个典型的数据点包括文本和相应的标签。
# 
# 来自YelpReviewFull测试集的示例如下：
# 
# ```json
# {
#     'label': 0,
#     'text': 'I got \'new\' tires from them and within two weeks got a flat. I took my car to a local mechanic to see if i could get the hole patched, but they said the reason I had a flat was because the previous patch had blown - WAIT, WHAT? I just got the tire and never needed to have it patched? This was supposed to be a new tire. \\nI took the tire over to Flynn\'s and they told me that someone punctured my tire, then tried to patch it. So there are resentful tire slashers? I find that very unlikely. After arguing with the guy and telling him that his logic was far fetched he said he\'d give me a new tire \\"this time\\". \\nI will never go back to Flynn\'s b/c of the way this guy treated me and the simple fact that they gave me a used tire!'
# }
# ```
# 
# #### 数据字段
# 
# - 'text': 评论文本使用双引号（"）转义，任何内部双引号都通过2个双引号（""）转义。换行符使用反斜杠后跟一个 "n" 字符转义，即 "\n"。
# - 'label': 对应于评论的分数（介于1和5之间）。
# 
# #### 数据拆分
# 
# Yelp评论完整星级数据集是通过随机选取每个1到5星评论的130,000个训练样本和10,000个测试样本构建的。总共有650,000个训练样本和50,000个测试样本。
# 
# ## 下载数据集

# In[1]:


from datasets import load_dataset

dataset = load_dataset("yelp_review_full")


# In[2]:


dataset


# In[3]:


dataset["train"][111]


# In[4]:


import random
import pandas as pd
import datasets
from IPython.display import display, HTML


# In[5]:


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


# In[6]:


show_random_elements(dataset["train"])


# ## 预处理数据
# 
# 下载数据集到本地后，使用 Tokenizer 来处理文本，对于长度不等的输入数据，可以使用填充（padding）和截断（truncation）策略来处理。
# 
# Datasets 的 `map` 方法，支持一次性在整个数据集上应用预处理函数。
# 
# 下面使用填充到最大长度的策略，处理整个数据集：

# In[7]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


# In[8]:


show_random_elements(tokenized_datasets["train"], num_examples=1)


# ### 数据抽样
# 
# 使用 1000 个数据样本，在 BERT 上演示小规模训练（基于 Pytorch Trainer）
# 
# `shuffle()`函数会随机重新排列列的值。如果您希望对用于洗牌数据集的算法有更多控制，可以在此函数中指定generator参数来使用不同的numpy.random.Generator。

# In[9]:


small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets["test"]


# ## 微调训练配置
# 
# ### 加载 BERT 模型
# 
# 警告通知我们正在丢弃一些权重（`vocab_transform` 和 `vocab_layer_norm` 层），并随机初始化其他一些权重（`pre_classifier` 和 `classifier` 层）。在微调模型情况下是绝对正常的，因为我们正在删除用于预训练模型的掩码语言建模任务的头部，并用一个新的头部替换它，对于这个新头部，我们没有预训练的权重，所以库会警告我们在用它进行推理之前应该对这个模型进行微调，而这正是我们要做的事情。

# In[10]:


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)


# ### 训练超参数（TrainingArguments）
# 
# 完整配置参数与默认值：https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/trainer#transformers.TrainingArguments
# 
# 源代码定义：https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/training_args.py#L161
# 
# **最重要配置：模型权重保存路径(output_dir)**

# In[11]:


from transformers import TrainingArguments

model_dir = "models/bert-base-cased-finetune-yelp"

# logging_steps 默认值为500，根据我们的训练数据和步长，将其设置为100
training_args = TrainingArguments(output_dir=model_dir,
                                  per_device_train_batch_size=16,
                                  num_train_epochs=5,
                                  logging_steps=100)


# In[12]:


# 完整的超参数配置
print(training_args)


# ### 训练过程中的指标评估（Evaluate)
# 
# **[Hugging Face Evaluate 库](https://huggingface.co/docs/evaluate/index)** 支持使用一行代码，获得数十种不同领域（自然语言处理、计算机视觉、强化学习等）的评估方法。 当前支持 **完整评估指标：https://huggingface.co/evaluate-metric**
# 
# 训练器（Trainer）在训练过程中不会自动评估模型性能。因此，我们需要向训练器传递一个函数来计算和报告指标。 
# 
# Evaluate库提供了一个简单的准确率函数，您可以使用`evaluate.load`函数加载

# In[13]:


import numpy as np
import evaluate

metric = evaluate.load("accuracy")


# 
# 接着，调用 `compute` 函数来计算预测的准确率。
# 
# 在将预测传递给 compute 函数之前，我们需要将 logits 转换为预测值（**所有Transformers 模型都返回 logits**）。

# In[14]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# #### 训练过程指标监控
# 
# 通常，为了监控训练过程中的评估指标变化，我们可以在`TrainingArguments`指定`evaluation_strategy`参数，以便在 epoch 结束时报告评估指标。

# In[15]:


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir=model_dir,
                                  evaluation_strategy="epoch", 
                                  per_device_train_batch_size=16,
                                  num_train_epochs=3,
                                  logging_steps=30)


# ## 开始训练
# 
# ### 实例化训练器（Trainer）
# 
# `kernel version` 版本问题：暂不影响本示例代码运行

# In[16]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)


# ## 使用 nvidia-smi 查看 GPU 使用
# 
# 为了实时查看GPU使用情况，可以使用 `watch` 指令实现轮询：`watch -n 1 nvidia-smi`:
# 
# ```shell
# Every 1.0s: nvidia-smi                                                   Wed Dec 20 14:37:41 2023
# 
# Wed Dec 20 14:37:41 2023
# +---------------------------------------------------------------------------------------+
# | NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
# |-----------------------------------------+----------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# |                                         |                      |               MIG M. |
# |=========================================+======================+======================|
# |   0  Tesla T4                       Off | 00000000:00:0D.0 Off |                    0 |
# | N/A   64C    P0              69W /  70W |   6665MiB / 15360MiB |     98%      Default |
# |                                         |                      |                  N/A |
# +-----------------------------------------+----------------------+----------------------+
# 
# +---------------------------------------------------------------------------------------+
# | Processes:                                                                            |
# |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
# |        ID   ID                                                             Usage      |
# |=======================================================================================|
# |    0   N/A  N/A     18395      C   /root/miniconda3/bin/python                6660MiB |
# +---------------------------------------------------------------------------------------+
# ```

# In[ ]:


trainer.train()


# In[ ]:


small_test_dataset = tokenized_datasets["test"].shuffle(seed=64).select(range(100))


# In[ ]:


trainer.evaluate(small_test_dataset)


# ### 保存模型和训练状态
# 
# - 使用 `trainer.save_model` 方法保存模型，后续可以通过 from_pretrained() 方法重新加载
# - 使用 `trainer.save_state` 方法保存训练状态

# In[ ]:


trainer.save_model(model_dir)


# In[ ]:





# In[ ]:


trainer.save_state()


# In[ ]:





# In[ ]:


# trainer.model.save_pretrained("./")


# ## Homework: 使用完整的 YelpReviewFull 数据集训练，看 Acc 最高能到多少

# In[ ]:




