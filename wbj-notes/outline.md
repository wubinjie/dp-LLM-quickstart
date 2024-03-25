

# 视频学习大纲


6.5 UniPELT 大模型 PEFT 统一框架

        关键字： 统一PELT 门控网络 门控参数

    04:08  UniPELT 实验结果
        关键：同样本量，复合方法比单一方法更好。样本量足够大，单一方法逼近复合方法。



6.6 (IA)^3 极简主义增量训练方法

    通过学习向量来对激活层加权进行缩放

    03:35  (IA)^3 探索新的增量训练方法  ---  相对于 LoRA的优势
    05:24  (IA)^3 探索新的增量训练方法  --- 网络设计
    06:41  (IA)^3 实验结果    ---  和其他方法对比的坐标轴图
    09:40   (IA)^3 实验结果   --- 和其他模型对比的坐标图
    10:14  大模型高效微调技术  未来发展趋势
        1. 更高效的参数优化
        1. 适应性和灵活性的提升
        1. 跨模态和多任务学习
        1. 模型压缩和加速
        1. 低资源语言和任务的支持

    
7.2 Hugging Face PEFT 快速入门

    00:23 Hugging Face PEFT 库 
    01:56 Hugging Face PEFT 库支持的模型和微调方法
        Hugging Face 官方网站有线上可运行的官方工具
    06:04 PEFT与Transformers 如何集成
    06:20 复习: Transformer 预训练模型常见分类
    07:24 应用: Transformers AutoModel 常见模型
    13:45 Transformers Auto Class 设计
    17:33 PEFT AutoPeftModels 与 Adapters 设计
    21:57 PEFT类型配置与PeftModel 实例化
    24:19 AutoPeftModel 简化流程
    24:36 以task_type为核心简化 PEFT 流程
    25:17 PEFT无缝对接Transformers Trainer 训练模型
    26:17 PEFT库微调原理与最佳实践 （以LoRA为例）

7.3 Open Whisper 模型介绍

    00:43 OpenAI Whisper 语音识别模型
    05:36 Whisper 网页示例 
    06:49 OpenAI Whisper 模型鲁棒性实验结果
    08:30 PEFT库LoRA实战
    08:51 OpenAI Whisper 模型系列
    09:13 Common Voice 数据集项目
    09:41 Common Voice 数据集 历史版本
    10:42 实战使用 Common Voice 11.0 版本 
    
7.4 实战 LoRA 微调 Whisper-Large-V2 中文语音识别

    14:09 模型支持转录和翻译，本次实验金仅仅操作转录。
    15:15 全局参数的解释。
    19:18 Common Voice 预训练数据集如何选择。里面已经包括了粤语等方言。
    27:37 移除数据集中不必要的字段代码讲解
    28:24 降采样率参数讲解
    32:32 完整数据集讲解
    35:00 语音长度填充的理解
    35:49 int8精度加载 模型
    36:30 加载之后显存占用的计算
    38:25 Int8精度加载之后，只能做推理，不能做训练，做训练精度太低。 但是使用 prepare_model_for_int8_training 函数之后，一部分重新进行转换，就可以做训练了。
    40:00 Lora参数讲解
    43:08 到底要训练多久的两套策略讲解。基于Epochs设置，基于step 设置。
    全量 用3轮比较稳妥
    49:19 保存perf的model再次加载回来
    54:38 加载做评估的数据集和训练数据集的区别

8.1 大模型量化技术（quantization）

8.2 模型显存占用与量化技术简介

    0:0 模型参数与显存占用计算方法
    9:47 模型量化技术 
    13:00 量化技术核心变量

8.3 GPTQ 面向 预训练 Transformer 模型设计的量化技术

    05:41 GPTQ 与 RTN（baseline） 实验结果对比
    09:33 GPTQ 在不同规模大模型下的实验结果
    10:57 GPTQ量化算法实现原理 
        关键逻辑，使用存储在Cholesky分解中的逆Hessian信息量化连续列的块，并在步骤结束时更新剩余的权重，在每个块内递归地应用量化过程。
    28:12 GPTQ在千亿级大模型上的实验结果  
    29:23 GPTQ量化前后实验结果对比 

8.4 激活感知权重量化  

    不是对模型中所有权重进行量化，而是仅保留小部分1%对LLM性能至关重要的权重。

    04:53 AWQ量化算法核心流程
    13:37 AWQ在OPT系列模型上的实验结果 
    15:00 AWQ在LLMA系列模型上的实验结果 
    15:27 AWQ在视觉语言模型上的实验结果 
    18:43 AWQ与GPTQ技术结合使用
    21:02 量化算法对比 AWQ 与 GPTQ

8.5 BnB 量化包

    0:06 温故： QLoRA 模型量化算法
    01:05 BitsAndBytes BNB 软件包简介
    04:30 Transformers 3种量化方案基准测试

8.6 实战 Transformers 模型量化 Facebook OPT

    01:14 软件包注意更新到最新版本
    01:44 代码目录quantization
    03:16 使用的外部数据集
    03:43 量化超参数配置
    05:06 量化过程中额外消耗显存
    05:19 检查模型量化的正确性
    08:42 使用自定义的数据集进行量化
    10:54 为何没有导入就用到了auto-gptq. AutoConfig 扩展
    13:02 AWQ例子讲解开始
    13:58 量化前模型测试文本生成任务
    14:21 AutoAWQ进行量化
    16:25 Transformers兼容性测试
    18:18 使用GPU加载量化模型
    18:54 BNB代码例子讲解
    19:35 加载模型的关键参数
    20:05 如何看模型如何消耗显存
    21:46 使用NF4精度加载模型
    22:54 使用双量化加载模型
    23:30 使用QLoRA所有量化技术加载模型


