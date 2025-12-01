
# 代码说明
- 模型权重：/model
- 训练代码：/train_code_base_lllamafactory
- 训练日志：/running.txt
- 训练超参数：/training_args.yaml
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


## 环境配置

- 参照init.sh进行配置
```sh
#创建虚拟环境
conda create -n roborain python=3.10 -y  
conda activate roborain

#拉取LLaMA-factory仓库代码，安装依赖
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
#如有问题，可参照LLaMA-Factory源码安装环境, 源码链接：https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md?plain=1#L471



#下面是从魔搭社区拉取RoboBrain2.0-7B模型，如有问题，可参照魔搭社区模型下载文档, 链接：https://modelscope.cn/docs/models/download
pip install modelscope
modelscope download --model BAAI/RoboBrain2.0-7B --local_dir /baai/baai_data21/RoboBrain2.0-7B


```

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

- 使用BAAI/RoboBrain2.0-7B 作为预训练模型，可通过(https://modelscope.cn/models/BAAI/RoboBrain2.0-7B ) 获得

## 算法

### 整体思路介绍

- 在本次决赛中，浅试团队基于RoboBrain2.0-7B模型，结合RoboBrain-1.0和Qwen3-VL的训练框架，在14天的周期内实现了SFT(+结构化CoT) → DPO的轻量闭环训练，并在空间理解、场景理解、基础空间和数量理解等指标上取得了显著的提升。通过基于Ego4D、EPIC-KITCHENS、LVIS等数据集，本次训练重点突破了7B⻓链规划、结构化思维链和偏好对⻬等瓶颈。




## 训练流程

- 参照train.sh

```sh
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /baai_data21/BAAI/RoboBrain2.0-7B \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template qwen2_vl \
    --flash_attn fa2 \
    --dataset_dir /root/hsc/LLaMA-Factory/data \
    --dataset robobrain_train \
    --cutoff_len 8192 \
    --learning_rate 2e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 150 \
    --warmup_steps 80 \
    --packing False \
    --enable_thinking False \
    --report_to none \
    --output_dir /baai_data21/hsc_train/robobrain/train_2025-11-07-full-14-00 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --freeze_vision_tower False \
    --freeze_multi_modal_projector False \
    --image_max_pixels 589824 \
    --image_min_pixels 1024 \
    --video_max_pixels 65536 \
    --video_min_pixels 256 \
    --val_size 0.01 \
    --eval_strategy steps \
    --eval_steps 150 \
    --per_device_eval_batch_size 4 \
    --deepspeed llamaboard_cache/ds_z2_config.json
```

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

## 测评
本地自测使用FlagEvalMM vllm server ,便于和在线FlagEval EmbodiedVerse
评测平台等统⼀评测标准。

```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve /baai_data21/hsc_train/robobrain/train_2025-11-14-base4datasets_full_export_model \
--served-model-name RoboBrain-full-3 \
--tensor-parallel-size=4 \
--limit-mm-per-prompt image=32 \
--port 8000 > /baai_data21/ceping/sec-11-14-base4datasets_full_export_model——vllm_front.log 2>&1 &
```

```
curl http://172.2.28.155:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RoboBrain-full-first",
    "messages": [
      {"role": "user", "content": "说中文?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'

  
  curl http://106.112.142.114:30201/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RoboBrain-full-first",
    "messages": [
      {"role": "user", "content": "说中文?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'

  curl http://172.2.28.155:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RoboBrain-full-sec",
    "messages": [
      {"role": "user", "content": "说中文?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'

  

```

## 消融实验

|  | 32b_baseline | 1107full | 1112full | 1113video | 7b_baseline |
| --- | --- | --- | --- | --- | --- |
| all_angles | \ | 42 | 46 | 47 | 41 |
| blink | 59 | 66 | 58 | 65 | 49 |
| cvbench | \ | 83 | 95 | 95 | 74 |
| egoplan | \ | 20 | 21 | 26 | 19 |
| embspatial | \ | 88 | 95 | 95 | 76 |
| erqa | 42 | 70 | 38 | 46 | 38 |
| mmsi | \ | 64 | 82 | 88 | 27 |
| omini_spatial | 28 | 75 | 36 | 35 | 39 |
| omini_spatial_test | 35 | 40 | \ | 27 | 41 |
| realworldqa | \ | 51 | 90 | 93 | 68 |
| refspatial_location | 61 | 20 | \ | 33 | 55 |
| refspatial_planing | 55 | 38 | \ | 5 | 42 |
| refspatial_unseen | 41 | 33 | \ | 10 | 38 |
| robo_spatial | 50 | 62 | 44 | 45 | 56 |
| sat | 76 | 83 | 77 | 63 | 74 |
| vsibench | \ | 37 | 34 | 35 | 40 |
| where2place | 76 | 44 | 12 | 13 | 61 |


## 训练log
```
{"current_steps": 5, "total_steps": 93, "loss": 5.7013, "lr": 1.0000000000000002e-06, "epoch": 0.1646090534979424, "percentage": 5.38, "elapsed_time": "0:02:03", "remaining_time": "0:36:15", "throughput": 13061.0, "total_tokens": 1614784}
{"current_steps": 10, "total_steps": 93, "loss": 4.169, "lr": 2.25e-06, "epoch": 0.3292181069958848, "percentage": 10.75, "elapsed_time": "0:04:02", "remaining_time": "0:33:29", "throughput": 13174.21, "total_tokens": 3189568}
{"current_steps": 15, "total_steps": 93, "loss": 1.2572, "lr": 3.5e-06, "epoch": 0.49382716049382713, "percentage": 16.13, "elapsed_time": "0:06:00", "remaining_time": "0:31:13", "throughput": 13107.65, "total_tokens": 4722976}
{"current_steps": 20, "total_steps": 93, "loss": 0.5741, "lr": 4.75e-06, "epoch": 0.6584362139917695, "percentage": 21.51, "elapsed_time": "0:07:58", "remaining_time": "0:29:07", "throughput": 13009.35, "total_tokens": 6230208}
{"current_steps": 25, "total_steps": 93, "loss": 0.4713, "lr": 6e-06, "epoch": 0.823045267489712, "percentage": 26.88, "elapsed_time": "0:09:55", "remaining_time": "0:27:00", "throughput": 13034.74, "total_tokens": 7763968}
{"current_steps": 30, "total_steps": 93, "loss": 0.478, "lr": 7.25e-06, "epoch": 0.9876543209876543, "percentage": 32.26, "elapsed_time": "0:11:55", "remaining_time": "0:25:03", "throughput": 13062.27, "total_tokens": 9350240}
{"current_steps": 35, "total_steps": 93, "loss": 0.3954, "lr": 8.5e-06, "epoch": 1.131687242798354, "percentage": 37.63, "elapsed_time": "0:13:40", "remaining_time": "0:22:40", "throughput": 13113.1, "total_tokens": 10764576}
{"current_steps": 40, "total_steps": 93, "loss": 0.3949, "lr": 9.75e-06, "epoch": 1.2962962962962963, "percentage": 43.01, "elapsed_time": "0:15:35", "remaining_time": "0:20:39", "throughput": 13195.61, "total_tokens": 12342240}
{"current_steps": 45, "total_steps": 93, "loss": 0.3868, "lr": 1.1000000000000001e-05, "epoch": 1.4609053497942388, "percentage": 48.39, "elapsed_time": "0:17:32", "remaining_time": "0:18:42", "throughput": 13207.2, "total_tokens": 13894304}
{"current_steps": 50, "total_steps": 93, "loss": 0.3852, "lr": 1.2250000000000001e-05, "epoch": 1.625514403292181, "percentage": 53.76, "elapsed_time": "0:19:33", "remaining_time": "0:16:49", "throughput": 13192.26, "total_tokens": 15478240}
{"current_steps": 55, "total_steps": 93, "loss": 0.3429, "lr": 1.3500000000000001e-05, "epoch": 1.7901234567901234, "percentage": 59.14, "elapsed_time": "0:21:31", "remaining_time": "0:14:52", "throughput": 13179.49, "total_tokens": 17023008}
{"current_steps": 60, "total_steps": 93, "loss": 0.3337, "lr": 1.4750000000000003e-05, "epoch": 1.954732510288066, "percentage": 64.52, "elapsed_time": "0:23:30", "remaining_time": "0:12:55", "throughput": 13163.99, "total_tokens": 18563872}
{"current_steps": 65, "total_steps": 93, "loss": 0.2922, "lr": 1.6000000000000003e-05, "epoch": 2.0987654320987654, "percentage": 69.89, "elapsed_time": "0:25:11", "remaining_time": "0:10:50", "throughput": 13154.8, "total_tokens": 19879136}
{"current_steps": 70, "total_steps": 93, "loss": 0.2547, "lr": 1.7250000000000003e-05, "epoch": 2.263374485596708, "percentage": 75.27, "elapsed_time": "0:27:04", "remaining_time": "0:08:53", "throughput": 13174.04, "total_tokens": 21403360}
{"current_steps": 75, "total_steps": 93, "loss": 0.2494, "lr": 1.8500000000000002e-05, "epoch": 2.42798353909465, "percentage": 80.65, "elapsed_time": "0:29:05", "remaining_time": "0:06:58", "throughput": 13170.29, "total_tokens": 22982976}
{"current_steps": 80, "total_steps": 93, "loss": 0.259, "lr": 1.9750000000000002e-05, "epoch": 2.5925925925925926, "percentage": 86.02, "elapsed_time": "0:31:00", "remaining_time": "0:05:02", "throughput": 13207.95, "total_tokens": 24572768}
{"current_steps": 85, "total_steps": 93, "loss": 0.2441, "lr": 1.568064746731156e-05, "epoch": 2.757201646090535, "percentage": 91.4, "elapsed_time": "0:33:04", "remaining_time": "0:03:06", "throughput": 13188.79, "total_tokens": 26175904}
{"current_steps": 90, "total_steps": 93, "loss": 0.2805, "lr": 4.319352532688444e-06, "epoch": 2.9218106995884776, "percentage": 96.77, "elapsed_time": "0:35:02", "remaining_time": "0:01:10", "throughput": 13203.14, "total_tokens": 27757184}
{"current_steps": 93, "total_steps": 93, "epoch": 3.0, "percentage": 100.0, "elapsed_time": "0:37:00", "remaining_time": "0:00:00", "throughput": 12832.23, "total_tokens": 28499296}
```
