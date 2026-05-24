# Qwen2.5-7B 中医药命名实体识别：从零开始到优化实验

本项目完成的任务是：从中医药领域文本中抽取完整实体词，并判断它属于
`中药`、`方剂`、`临床表现`、`西医诊断` 等十种类别。

主方案使用 `Qwen/Qwen2.5-7B-Instruct`，同时提供：

- `LoRA` 与 `QLoRA` 两种可选微调方式；
- 适合一步一步执行的 Jupyter Notebook；
- 以**完整实体词**严格匹配计算的每类 F1；
- 数据审计、泄漏清理副本、低频类别增强、误差分析、消融对比等优化工具。

说明：默认采用 `Instruct` 版本，是因为本项目把 NER 写成“用户提问，模型输出
实体 JSON”的指令式 SFT 任务。需要替换基座模型时，只修改 YAML 中的
`model_name_or_path`，但不同模型架构应重新训练 adapter。

## 先看懂数据

第一次接触 NER 时，请先阅读 [data/DATASET_GUIDE.md](data/DATASET_GUIDE.md)。
它用真实数据解释：

- 为什么原始数据一行只有一个字；
- `B-中药`、`I-中药`、`O` 如何拼成完整实体；
- 十种实体类别各是什么含义；
- 为什么最终评价“词的 F1”而不是“单字 F1”。

最短示例：

```text
口 B-临床表现
苦 I-临床表现
```

应读为一个完整实体：

```json
{"entity": "口苦", "type": "临床表现"}
```

## 文件地图

建议按下表顺序理解项目，而不是一次打开全部代码。

| 顺序 | 文件 | 小白需要理解的事情 |
| ---: | --- | --- |
| 1 | `data/DATASET_GUIDE.md` | 原始 BIO 数据究竟是什么意思 |
| 2 | `prepare_sft_data.py` | 怎样把逐字标签转换成大模型训练答案 |
| 3 | `audit_dataset.py` | 为什么训练前必须检查数据泄漏和标注冲突 |
| 4 | `create_clean_splits.py` | 怎样生成不覆盖原文件的干净实验副本 |
| 5 | `rebalance_train_data.py` | 怎样只增强低频训练类别，不污染测试集 |
| 6 | `configs/qlora_optimized.yaml` | 推荐主实验的可修改配置 |
| 7 | `train_sft.py` | LoRA/QLoRA 如何真正训练 |
| 8 | `evaluate_ner.py` | 怎样生成预测并打印每类完整实体 F1 |
| 9 | `analyze_predictions.py` | 怎样查看漏检、边界错误和难样本 |
| 10 | `make_ablation_configs.py`、`compare_experiments.py` | 怎样用验证集比较配置 |

核心共用逻辑在 `ner_utils.py`；逐格学习版本在
`notebooks/中医药命名实体识别_Qwen2.5_LoRA_QLoRA.ipynb`。

## 运行环境

### 1. 进入项目目录

```bash
cd /Users/huwei/PyCharmMiscProject/大模型微调实战项目-中医药命名实体识别
```

### 2. 创建 Python 环境

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

请先依据你的 NVIDIA 显卡和 CUDA 版本，从
[PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择安装
`torch` 的命令，再执行：

```bash
pip install -r requirements.txt
```

没有 NVIDIA GPU 时，可以执行数据处理、审计和测试，但无法合理完成
Qwen2.5-7B 的正式训练。

## 路线选择

本项目提供两条路线：

| 路线 | 适用场景 | 重要说明 |
| --- | --- | --- |
| 快速学习路线 | 先跑通作业流程、理解每个文件用途 | 遇到冲突时可显式选择 `keep_first`，但这不等于人工纠正了标签 |
| 正式实验路线 | 用于最终报告或追求可信结果 | 必须先人工复核冲突条目，再重新生成干净数据并训练 |

下面的命令先给出快速学习路线；每一步都会指出正式实验需要注意什么。

## 第 1 步：把 BIO 数据转换成 SFT 数据

### 你运行哪个文件

`prepare_sft_data.py`

### 你执行的命令

```bash
python prepare_sft_data.py --data_dir data --output_dir processed_data
```

### 它读取什么

```text
data/medical.train
data/medical.dev
data/medical.test
```

### 它生成什么

```text
processed_data/train.jsonl
processed_data/validation.jsonl
processed_data/test.jsonl
processed_data/summary.json
```

每条 JSONL 记录包含原文、金标准实体，以及供 Qwen 学习的
`prompt` / `completion`。答案形式示例：

```json
[{"entity":"口苦","type":"临床表现","start":3,"end":5}]
```

`start` 从 0 开始；`end` 为右开区间，因此
`"现头昏口苦"[3:5] == "口苦"`。

### 当前数据转换结果

| 划分 | 样本数 | 最大文本字符数 |
| --- | ---: | ---: |
| train | 5259 | 143 |
| validation | 657 | 127 |
| test | 658 | 118 |

## 第 2 步：训练前先审计数据质量

### 你运行哪个文件

`audit_dataset.py`

### 你执行的命令

```bash
python audit_dataset.py --data_dir data --report_dir reports
```

### 它生成什么

```text
reports/data_audit.md
reports/data_audit.json
```

先打开容易阅读的 Markdown 报告：

```bash
sed -n '1,200p' reports/data_audit.md
```

### 本数据集已发现的真实问题

| 检查项 | 审计结果 | 为什么重要 |
| --- | ---: | --- |
| train 内部重复文本组 | 43 | 模型可能反复看到同一句话 |
| train 内部同文本标注冲突组 | 13 | 同一句话被教成不同答案 |
| validation 内部标注冲突组 | 1 | 验证分数可能受噪声影响 |
| 跨划分相同文本 | 19 | 训练内容可能提前出现在评测中 |
| 跨划分标注冲突 | 4 | 相同原文在不同集合答案不一致 |
| 训练集低于 100 个实体的类别 | `其他治疗` | 小类别可能很难学好 |

此外，多数文本含全角数字或符号。NER 的实体边界依赖字符位置，因此本工程不会
擅自将全角字符改为半角，否则 `start` / `end` 可能错位。

## 第 3 步：生成无精确文本泄漏的数据副本

### 你运行哪个文件

`create_clean_splits.py`

### 正式实验应先这样运行

```bash
python create_clean_splits.py \
  --input_dir processed_data \
  --output_dir processed_data/clean \
  --report_file reports/clean_split_report.json
```

由于审计确认存在标注冲突，这条命令会停止并提示你人工复核。这样设计是为了
不让程序在医学标签冲突中自行猜一个答案。

### 小白快速跑通流程的命令

只为了先完成训练流程练习，可以明确选择保留每个重复文本首次出现的标注：

```bash
python create_clean_splits.py \
  --input_dir processed_data \
  --output_dir processed_data/clean \
  --report_file reports/clean_split_report.json \
  --within_conflict_policy keep_first
```

该命令不覆盖原始文件，已生成的副本规模为：

| 划分 | 原样本数 | clean 样本数 | 移除数量 |
| --- | ---: | ---: | ---: |
| train | 5259 | 5184 | 75 |
| validation | 657 | 654 | 3 |
| test | 658 | 658 | 0 |

规则为：优先保留 `test`，其次保留 `validation`，最后保留 `train`；这样测试
文本不会同时存在于训练集中。

### 正式报告前如何处理冲突

1. 打开 `reports/data_audit.json` 和 `reports/clean_split_report.json`。
2. 找到 `annotation_conflict` 或 `same_split_annotation_conflict_keep_first` 条目。
3. 按课程标签规范或请领域人员确认应该保留的标注。
4. 修订原始 `data/medical.*` 或建立一份人工确认的数据版本。
5. 重新执行第 1 至第 3 步，并保存新的报告。

## 第 4 步：仅增强训练集低频类别

### 你运行哪个文件

`rebalance_train_data.py`

### 你执行的命令

```bash
python rebalance_train_data.py \
  --input_file processed_data/clean/train.jsonl \
  --output_file processed_data/optimized/train_balanced.jsonl \
  --target_types 其他治疗 \
  --target_entity_count 300 \
  --max_extra_copies_per_record 4 \
  --report_file reports/rebalance_report.json
```

### 它做了什么

- 只复制含 `其他治疗` 实体的训练样本；
- 每个副本保存来源 id，便于追踪是否过拟合；
- 不复制 `validation` 或 `test`，因此不会虚高最终分数；
- 每条原样本最多增加四个副本，避免某几句话被无限重复。

当前实际生成结果：

| 项目 | 增强前 | 增强后 |
| --- | ---: | ---: |
| 训练样本数 | 5184 | 5408 |
| `其他治疗` 实体数 | 65 | 300 |

## 第 5 步：选择 QLoRA 或 LoRA 训练

### 推荐先用 QLoRA

QLoRA 以 4-bit NF4 量化方式加载 7B 基座，更适合显存有限的首次实验：

```bash
python train_sft.py --config configs/qlora_optimized.yaml
```

### 显存较充分时用 LoRA

```bash
python train_sft.py --config configs/lora_optimized.yaml
```

### 两个优化配置包含什么改进

| 配置项 | 作用 |
| --- | --- |
| `processed_data/optimized/train_balanced.jsonl` | 使用去泄漏且增强低频类别后的训练数据 |
| `processed_data/clean/validation.jsonl` | 使用未重采样的干净验证数据 |
| `processed_data/clean/test.jsonl` | 使用保留完整测试样本的干净测试数据 |
| `group_by_length: true` | 将长度相近文本组成批次，减少 padding |
| `early_stopping_patience: 3` | 验证 loss 长期不改善时提前停止 |
| `attn_implementation: sdpa` | 使用 PyTorch 内置高效注意力实现 |
| `experiment_manifest.json` | 自动保存模型、数据 SHA256 和关键包版本 |

训练结果将保存到：

```text
outputs/qwen2_5_7b_qlora_optimized/
outputs/qwen2_5_7b_lora_optimized/
```

### 显存不足怎么办

打开相应 YAML，按顺序调小：

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

若仍不足，继续将批大小改为 `1`，并增大梯度累积步数维持近似等效批量。

### 如何替换基座模型

只需在 YAML 中修改：

```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
```

不同架构模型还需确认聊天模板、结束符和 `lora.target_modules`，并重新训练
adapter；不能把 Qwen adapter 直接装到另一个架构上。

## 第 6 步：先在验证集评价模型

不要先看测试集分数再调参数。训练完成后，先使用验证集判断方案是否值得保留。

### QLoRA 优化模型验证命令

```bash
python evaluate_ner.py \
  --config configs/qlora_optimized.yaml \
  --adapter_path outputs/qwen2_5_7b_qlora_optimized \
  --data_file processed_data/clean/validation.jsonl \
  --prediction_file outputs/qwen2_5_7b_qlora_optimized/validation_predictions.jsonl \
  --metrics_file outputs/qwen2_5_7b_qlora_optimized/validation_metrics.json
```

### 评价脚本现在额外做了什么优化

| 优化 | 默认行为 |
| --- | --- |
| 批量推理 | QLoRA 每批 4 条，LoRA 每批 2 条；显存不足可传 `--batch_size 1` |
| JSON 格式纠错 | 回答不是合法数组或跨度不合法时，自动重试一次 |
| 更严格指标 | 每种类别都打印完整实体 F1，同时打印 micro 与 macro F1 |
| 稳定性说明 | 默认对 micro F1 重采样 1000 次，给出 95% 区间 |

模型只有在 `type`、`entity`、`start`、`end` 全部正确时才计为真正例。例如
金标准是 `腹痛`，只预测出 `痛` 仍属于错误。

## 第 7 步：查看模型具体错在哪里

### 你运行哪个文件

`analyze_predictions.py`

### 你执行的命令

```bash
python analyze_predictions.py \
  --prediction_file outputs/qwen2_5_7b_qlora_optimized/validation_predictions.jsonl \
  --json_report reports/qlora_optimized_validation_errors.json \
  --markdown_report reports/qlora_optimized_validation_errors.md
```

### 它会告诉你什么

- 模型漏检了多少实体；
- 有多少是只识别到半个词的边界错误；
- 有多少是完整词正确但类别判断错；
- 哪些额外预测是假阳性；
- 短文本、长文本、含数字、含剂量单位、多实体文本分别表现如何；
- 十种实体类别中，哪类 F1 最低。

根据报告再判断是否需要调高 `其他治疗` 重采样比例、增加 rank 或修改提示词。

## 第 8 步：进行 QLoRA 消融实验

### 生成实验配置

```bash
python make_ablation_configs.py \
  --base_config configs/qlora_optimized.yaml \
  --output_dir configs/ablations
```

已生成的五种变体包括：

```text
configs/ablations/qlora_optimized_rank_r8.yaml
configs/ablations/qlora_optimized_rank_r32.yaml
configs/ablations/qlora_optimized_lr_1e-4.yaml
configs/ablations/qlora_optimized_lr_3e-4.yaml
configs/ablations/qlora_optimized_attention_only.yaml
```

### 逐个训练

例如：

```bash
python train_sft.py --config configs/ablations/qlora_optimized_rank_r8.yaml
```

每个实验训练完成后，都按第 6 步的方法在
`processed_data/clean/validation.jsonl` 上评价，并将指标文件保存到各自输出
目录中。

### 用验证集 F1 排名

以下命令中的指标路径请按你实际已跑的实验填写：

```bash
python compare_experiments.py \
  --metric_files \
    outputs/qwen2_5_7b_qlora_optimized/validation_metrics.json \
    outputs/ablations/rank_r8/validation_metrics.json \
    outputs/ablations/rank_r32/validation_metrics.json \
  --output_file reports/validation_leaderboard.md
```

只根据验证集排行榜决定最终配置，不能根据测试集分数反复调参。

## 第 9 步：只对选定模型运行一次测试集

假设验证集最终选中了优化 QLoRA 主模型：

```bash
python evaluate_ner.py \
  --config configs/qlora_optimized.yaml \
  --adapter_path outputs/qwen2_5_7b_qlora_optimized \
  --data_file processed_data/clean/test.jsonl \
  --prediction_file outputs/qwen2_5_7b_qlora_optimized/test_predictions.jsonl \
  --metrics_file outputs/qwen2_5_7b_qlora_optimized/test_metrics.json
```

该命令会打印你作业最终需要报告的十种实体类别 F1。

## 第 10 步：可选工业对比实验

### 1. 判别式 NER 基线

工业应用中，小型 token-classification 模型通常推理更快。因此项目提供可选基线：

```bash
python train_token_baseline.py --config configs/token_baseline.yaml
```

它使用与优化 QLoRA 相同的 clean/balanced 数据，且最终仍按完整实体 F1 评价。
这个实验用于比较成本和时延，不替代作业要求的 Qwen2.5-7B 微调结果。

### 2. 两张 GPU 训练

如果你的机器确有两张 GPU，可先检查并修改
`configs/accelerate_multi_gpu.yaml` 中的 `num_processes`，然后执行：

```bash
accelerate launch \
  --config_file configs/accelerate_multi_gpu.yaml \
  train_sft.py --config configs/qlora_optimized.yaml
```

### 3. FlashAttention 试验

默认 `attn_implementation: sdpa` 不需要额外安装。如果 GPU 和软件环境兼容，并
安装了 `flash-attn`，可复制一个 YAML，将这一项改为：

```yaml
attn_implementation: flash_attention_2
```

再通过验证集 F1 与训练耗时判断它是否值得使用。

## 优化实施状态说明

完整状态与决策注意事项见 [INDUSTRIAL_OPTIMIZATION.md](INDUSTRIAL_OPTIMIZATION.md)。

已经实际执行且产生文件的优化包括：

| 优化 | 已生成产物 |
| --- | --- |
| 数据审计 | `reports/data_audit.md`、`reports/data_audit.json` |
| 教学用 clean 副本 | `processed_data/clean/*.jsonl`、`reports/clean_split_report.json` |
| 低频类别重采样 | `processed_data/optimized/train_balanced.jsonl`、`reports/rebalance_report.json` |
| 消融配置准备 | `configs/ablations/*.yaml` |

需要模型训练完成后才能产生真实结果的项目包括：验证/测试 F1、错误分析报告、
JSON 自动纠错成功数、消融排行榜与判别式基线成绩。

## Jupyter 版本

可打开：

```text
notebooks/中医药命名实体识别_Qwen2.5_LoRA_QLoRA.ipynb
```

Notebook 适合学习 BIO 数据与基础训练流程。命令行步骤更适合完整执行数据审计、
清理、重采样和多实验比较，因为每一步会在磁盘上保留可审查报告。

## 本地已完成的验证

本工程已运行以下无需 GPU 的检查：

```bash
python prepare_sft_data.py --data_dir data --output_dir processed_data
python audit_dataset.py --data_dir data --report_dir reports
python create_clean_splits.py \
  --input_dir processed_data \
  --output_dir processed_data/clean \
  --report_file reports/clean_split_report.json \
  --within_conflict_policy keep_first
python rebalance_train_data.py \
  --input_file processed_data/clean/train.jsonl \
  --output_file processed_data/optimized/train_balanced.jsonl \
  --target_types 其他治疗 \
  --target_entity_count 300 \
  --max_extra_copies_per_record 4 \
  --report_file reports/rebalance_report.json
python make_ablation_configs.py --base_config configs/qlora_optimized.yaml --output_dir configs/ablations
python -m unittest discover -s tests -v
```

当前环境未启动 7B 模型训练或推理生成，因为这需要适配的 GPU、CUDA/PyTorch
环境和模型权重下载。

## 官方参考

- [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/v1.4.0/sft_trainer)
- [PEFT LoRA documentation](https://huggingface.co/docs/peft/main/en/developer_guides/lora)
- [Transformers bitsandbytes quantization documentation](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes)
- [Accelerate launch documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
