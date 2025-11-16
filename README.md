
# 代码说明

## 训练配置

**1. 硬件资源**  
- **GPU**: 8× NVIDIA H200
- **CPU**: 2× Intel Xeon Platinum 8558 
- **内存**: 2.0 TiB 
- **存储**: 根分区878G + /baai_data21 5.1T  

**2. 软件栈版本**  
- **OS**: Ubuntu 22.04
- **Python**: 3.11 
- **PyTorch**: 2.9.0+cu128  
- **Transformers**: 4.57.1  
- **CUDA**: 12.8 
- **驱动**: 550.54.14  

- 建议用 `torchrun --nproc_per_node=4` 启动多卡任务。


## 环境配置

- 参照init.sh进行配置

## 数据

- 使用了外部于Ego4D、EPIC-KITCHENS、LVIS等数据集和组委提供的数据作为训练和测试集

## 数据预处理
>对原始数据标签打分，规则过滤 ，模型过滤，三种粒度去重
- 1. 对原始数据的(instruction + input + output)字段进行打分，打分区间为1-6分
A相关度：计算原始数据的instruction + input字段和output字段之间的语义相似度，如果语义相似度低于一定阈值，删除掉。
- 2. nooutput输出,模型输出格式不符合
- 3. 基于Skywork-VL Reward：多模态奖励模型，得分从高到低排序，选取高得分的数据
- 4. simhash、minhash和基于语义编码的语义相似度去重，语义相似度去重基于bce模型和knn算法。

- 初赛⽅案中，我们对训练和验证数据集进⾏了详细规划，在本轮决赛中，我们从多个开源数据集中选取适合的⼦集进⾏训练：仅抽取⼀20k样本组成本轮训练⽤的 MU-Data（Minimal Usable Data



## 预训练模型

- 使用Qwen/Qwen2.5-VL-7B-Instruct作为预训练模型，可通过(https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct) 获得

## 算法

### 整体思路介绍

- 在本次决赛中，浅试团队基于RoboBrain2.0-7B模型，结合RoboBrain-1.0和Qwen3-VL的训练框架，在14天的周期内实现了SFT(+结构化CoT) → DPO的轻量闭环训练，并在空间理解、场景理解、基础空间和数量理解等指标上取得了显著的提升。通过基于Ego4D、EPIC-KITCHENS、LVIS等数据集，本次训练重点突破了7B⻓链规划、结构化思维链和偏好对⻬等瓶颈。


### 方法的创新点

#### **核心创新点与技术突破**

##### 1. 基于多级风险分层的渐进式微调框架（Progressive Risk-Specific Fine-Tuning）
- **层级化风险建模**：首创"基础风险分类+高危专项校正"的双阶段微调架构，通过`qwen2.5-vl-7b-sft`实现全风险谱系粗粒度识别后，采用`qwen2.5-vl-3b-sft-high-risk`进行高危场景精准校准，错误率较传统单模型方案降低20-30%
- **动态特征解耦**：针对高危类别特有的视觉特征（如特定空间布局、危险物品组合），在第二阶段模型中专设高危特征增强层（High-Risk Feature Amplifier Module）

##### 2. 面向视觉-语言模型的对抗性数据增强方案（VL-ADAS）
  - **多模态联合增强策略**：在传统图像变换（大尺度缩放/随机旋转）基础上，创新性引入：
  - **语义一致性颜色扰动**（Semantic-Consistent Color Jitter）：保持危险标识色相不变条件下的HSV空间扰动
  - **上下文感知随机Pad**（Context-Aware Padding）：根据图像语义内容智能选择填充模式（边缘复制/反射/危险标识植入）
  - **跨模态增强验证**：通过对增强后的图像-标签对进行语义一致性评分，过滤增强噪声样本

##### 3. 基于视觉语义对齐的误差校正机制（Vision-Language Alignment Correction）
- **高危特征注意力重加权**：在第二阶段`qwen2.5-vl-3b-sft-high-risk模型中采用
- **双模型置信度融合**：最终预测结果 = 基础模型置信度 × 高危模型校正系数


## 训练流程

- 参照train.sh

- SFT-Align (阶段 1):
为确保在14天内实现可⾏的训练⽬标，我们选择了三阶段训练⽅案，其中包括SFT训练、结构化CoT嵌⼊以及DPO偏好对⻬。通过分阶段训练，我们能在较短时间内逐步提升模型性能，并确保每个阶段的效果可验证、可量化。
在这⼀阶段，我们⾸先对RoboBrain2.0-7B模型进⾏预训练微调（SFT），确保其在视觉理解和语⾔⽣成上具有基本的能⼒。在这个过程中，我们使⽤标准的监督学习⽅法，结合适当的学习率和batch size。

- SFT-Inject (阶段 2)
在这⼀阶段，我们引⼊了结构化CoT，将任务指令转化为结构化的JSON输出。通过这种⽅式，我 们能够使模型⽣成的推理过程更加可控制。在SFT-Align的基础上，进⼀步训练模型⽣成结构化的思维链，并进⾏反馈调整。

- SFT-Fuse + DPO (阶段 3)
该阶段我们将继续使⽤SFT训练，融⼊更多的任务规划能⼒，同时加⼊DPO偏好对⻬，通过⾃制的“失败→修正”数据对进⾏偏好学习。使⽤DPO算法，通过正负样本对进⾏训练，使模型在任务规划过程中能够做出更符合期望的决策。


## trick

- 结构化CoT
从 VG/LVIS 等原始数据提取字段生成旁路 JSONL，将 “看、做、期望”写为可执行步骤，显著提升 EmbSpatial等空间推理和多图整合基准表现。
- 7B规划模板
由 Al2-THOR episode 压缩 3-5 步高层计划，适配7B 处理能力，在 EgoPlan 等规划/多视角任务上收益可观。
- DPO偏好对齐
在结构化计划上做偏好学习，使模型更贴近人类偏好完成任务，对多选/干扰题型(RealWorldQA等)帮助明显。

- 工程化补偿：我们正在采用的更优的动态校正策略去有效地收敛了离散结果。
因模型参数限制能力，训练数据阶段引入更强的闭源多模态大模型作裁判。对VG/LVIS等转出的`converted/*.jsonl`，用Qwen3-vl-plus-0923模型判断CoT自洽性与描述冗余度，筛除错误样本；在AI2-THOR多候选计划中，用Qwen3-VL做“好/坏”初筛。遵循“LLM-as-a-Judge”生成偏好数据、DPO对齐的套路，提升训练数据信噪比。因此，EmbSpatial等基准分数显著提升，归因于更干净一致的CoT/plan监督及DPO使用高质量偏好对。


## Bug solve
OOM/通信死锁:
现象:全量 SFT 出现 CUDA OOM 或某 rank 崩溃触发 NCCL 死锁。处置:bs↓/grad-acc↓/ZeR0-2/3个/checkpointing;
问题处理
ROPE 形状不匹配(长序列)现象:样本编码长度超上限，位置编码 broadcast 失败。处置:强制长度截断/像素上限/帧下采样;统- max_sequence length 与预处理策略(多图/视频场景)。
评测与数据规范:BLINK test 的 GT 隐藏，统一评 val; RefSpatial/here2Place 强制归一化坐标输出;多选题严格“只给选项”

