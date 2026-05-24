"""训练一个可选的中文 Token Classification NER 基线，与 Qwen 方案比较成本和质量。"""

# 导入未来注解功能，改善类型注解可读性。
from __future__ import annotations

# 导入命令行模块，让用户指定基线配置文件。
import argparse
# 导入 JSON 模块，保存完整实体评价报告。
import json
# 导入 Path，安全处理配置、数据与输出目录。
from pathlib import Path
# 导入 Any，用于配置和评价数组类型标注。
from typing import Any

# 导入 YAML，读取适合初学者编辑的参数配置。
import yaml

# 导入本项目统一的 BIO/实体转换与严格词级 F1 工具。
from ner_utils import ENTITY_TYPES, bio_to_entities, compute_entity_f1, entities_to_bio, format_metric_report, read_jsonl


# 建立判别式模型需要预测的全部标签：一个 O 标签和每类的 B/I 标签。
BIO_LABELS = ["O"] + [
    f"{prefix}-{entity_type}" for entity_type in ENTITY_TYPES for prefix in ["B", "I"]
]
# 将文字标签映射为模型训练使用的数字 id。
LABEL_TO_ID = {label: index for index, label in enumerate(BIO_LABELS)}
# 将数字 id 映射回可阅读的文字标签。
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    """读取基线实验 YAML 位置。"""

    # 创建参数解析器，说明这是非主实验的可选参照。
    parser = argparse.ArgumentParser(description="训练可选 Token Classification NER 基线供工业对比。")
    # 默认使用项目提供的公平数据配置。
    parser.add_argument(
        "--config", type=Path, default=Path("configs/token_baseline.yaml"), help="基线 YAML 配置路径。"
    )
    # 返回用户参数。
    return parser.parse_args()


def load_yaml(file_path: Path) -> dict[str, Any]:
    """读取 YAML 配置并检查文件存在。"""

    # 配置文件缺失时停止执行。
    if not file_path.exists():
        # 抛出明确错误提示。
        raise FileNotFoundError(f"找不到配置文件：{file_path}")
    # 打开 UTF-8 YAML 文件。
    with file_path.open("r", encoding="utf-8") as reader:
        # 返回解析后的配置字典。
        return yaml.safe_load(reader)


def load_examples(file_path: str) -> list[dict[str, Any]]:
    """读取 SFT JSONL，并将词级答案转回 Token Classification 所需 BIO。"""

    # 读取与大模型完全相同的数据记录。
    records = read_jsonl(file_path)
    # 准备判别式模型训练示例。
    examples: list[dict[str, Any]] = []
    # 逐条转换实体答案。
    for record in records:
        # 将每个中文字符组成一个 token 对齐单元。
        chars = list(record["text"])
        # 把词级实体重新转换为逐字符 BIO 标签。
        labels = entities_to_bio(record["text"], record["gold_entities"])
        # 保存供 tokenizer 对齐的字符和标签。
        examples.append({"id": record["id"], "chars": chars, "labels": labels})
    # 返回数据列表。
    return examples


def main() -> None:
    """训练较小判别式模型，并使用相同完整实体 F1 汇报测试结果。"""

    # 读取命令行参数和 YAML 配置。
    args = parse_args()
    # 加载本次基线配置。
    config = load_yaml(args.config)
    # 在真正训练时才导入深度学习依赖，便于不安装模型库时查看 --help。
    import numpy as np
    # 导入 PyTorch 检查混合精度支持情况。
    import torch
    # Datasets 将 Python 记录转换为 Trainer 数据集。
    from datasets import Dataset
    # Transformers 提供分类模型、数据整理器和标准 Trainer。
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    # 读取训练设置。
    training = config["training"]
    # 固定随机种子以便比较。
    set_seed(int(training["seed"]))
    # 读取基线模型名称或本地目录。
    model_name = config["model_name_or_path"]
    # 加载中文 tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # 分别读取训练、验证和测试记录。
    raw_examples = {
        split: load_examples(config["data"][f"{split}_file"])
        for split in ["train", "validation", "test"]
    }
    # 保存未 token 化的验证和测试字符，供严格实体指标还原边界。
    raw_eval_chars = {
        split: [example["chars"] for example in raw_examples[split]]
        for split in ["validation", "test"]
    }

    # 定义逐条 tokenizer 标签对齐函数。
    def tokenize_and_align(example: dict[str, Any]) -> dict[str, Any]:
        """将字符列表编码，并只给每个原字符的第一个子 token 分配 BIO 标签。"""

        # 使用 is_split_into_words 保留每个原字符与 token 的对应关系。
        encoded = tokenizer(
            example["chars"],
            is_split_into_words=True,
            truncation=True,
            max_length=int(training["max_length"]),
        )
        # 读取 tokenizer 为当前序列返回的原字符编号。
        word_ids = encoded.word_ids()
        # 保存与 token 序列等长的监督标签 id。
        aligned_labels: list[int] = []
        # 保存前一个字符编号，识别同一字符产生多个子 token 的情况。
        previous_word_id = None
        # 逐 token 安排监督标签。
        for word_id in word_ids:
            # 特殊 token 没有原字符对应，不参与损失计算。
            if word_id is None:
                # -100 表示 Trainer 忽略该位置损失。
                aligned_labels.append(-100)
            # 只有一个原字符的第一个 token 接受其 BIO 标签。
            elif word_id != previous_word_id:
                # 将文字 BIO 标签转换为整数标签 id。
                aligned_labels.append(LABEL_TO_ID[example["labels"][word_id]])
            else:
                # 同一字符后续子 token 不重复计分。
                aligned_labels.append(-100)
            # 更新上一 token 对应的原字符编号。
            previous_word_id = word_id
        # 将标签加入 tokenizer 编码字典。
        encoded["labels"] = aligned_labels
        # 返回 Trainer 可以使用的单条样本。
        return encoded

    # 将三个 Python 列表转为 Dataset 并执行 token 对齐。
    tokenized_datasets = {
        split: Dataset.from_list(examples).map(
            tokenize_and_align, remove_columns=["id", "chars", "labels"]
        )
        for split, examples in raw_examples.items()
    }

    # 定义从预测矩阵恢复完整实体指标的函数。
    def evaluate_predictions(predictions: Any, label_ids: Any, split: str) -> dict[str, Any]:
        """把 token 分类结果恢复为 BIO 实体并计算严格词级评价。"""

        # 在标签维度取概率最高的类别 id。
        predicted_label_ids = np.argmax(predictions, axis=-1)
        # 准备每条样本恢复出的人工实体。
        gold_batches: list[list[dict[str, Any]]] = []
        # 准备每条样本恢复出的预测实体。
        predicted_batches: list[list[dict[str, Any]]] = []
        # 逐条将有效 token 标签还原到原字符序列。
        for row_index, (predicted_row, gold_row) in enumerate(zip(predicted_label_ids, label_ids)):
            # 提取未被 -100 屏蔽的人工标签文字。
            gold_labels = [
                ID_TO_LABEL[int(label_id)] for label_id in gold_row if int(label_id) != -100
            ]
            # 使用同一有效位置提取模型预测标签文字。
            predicted_labels = [
                ID_TO_LABEL[int(predicted_id)]
                for predicted_id, gold_id in zip(predicted_row, gold_row)
                if int(gold_id) != -100
            ]
            # 截取与有效标签长度一致的原始字符。
            chars = raw_eval_chars[split][row_index][: len(gold_labels)]
            # 将金标准 BIO 还原为完整实体。
            gold_batches.append(bio_to_entities(chars, gold_labels, strict=True))
            # 模型可能产生不合法 I 标签，因此以容错方式恢复预测实体。
            predicted_batches.append(bio_to_entities(chars, predicted_labels, strict=False))
        # 按主实验相同的完整实体严格匹配规则评价。
        return compute_entity_f1(gold_batches, predicted_batches)

    # 定义 Trainer 在验证过程中调用的精简指标函数。
    def compute_validation_metrics(eval_prediction: Any) -> dict[str, float]:
        """返回 Trainer 选最优 checkpoint 所需的数值指标。"""

        # 解包模型预测分数和人工标签 id。
        predictions, label_ids = eval_prediction
        # 计算验证集实体指标。
        metrics = evaluate_predictions(predictions, label_ids, split="validation")
        # 返回标准 Trainer 可记录的几个浮点数。
        return {
            "micro_f1": metrics["overall"]["f1"],
            "macro_f1": metrics["overall"]["macro_f1"],
            "precision": metrics["overall"]["precision"],
            "recall": metrics["overall"]["recall"],
        }

    # 加载带 21 个 BIO 输出类别的 token classification 模型。
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(BIO_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    # 创建基线结果输出目录。
    output_dir = Path(training["output_dir"])
    # 确保目录存在。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 判断当前硬件能否启用 BF16。
    use_bf16 = bool(
        training.get("precision") == "bf16"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    # 定义 Hugging Face 标准训练循环配置。
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=config["experiment_name"],
        num_train_epochs=float(training["num_train_epochs"]),
        learning_rate=float(training["learning_rate"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=int(training["save_total_limit"]),
        weight_decay=float(training["weight_decay"]),
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        report_to="none",
        seed=int(training["seed"]),
    )
    # 用动态 padding 整理每一批 token 分类样本。
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # 创建标准 Trainer。
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_validation_metrics,
    )
    # 正式训练判别式基线。
    trainer.train()
    # 保存验证 micro F1 最优的基线模型与 tokenizer。
    trainer.save_model(str(output_dir))
    # 保存 tokenizer。
    tokenizer.save_pretrained(str(output_dir))
    # 对测试集产生 token 分类预测。
    test_prediction = trainer.predict(tokenized_datasets["test"])
    # 使用与 Qwen 方案一致的严格实体规则计算完整测试报告。
    test_metrics = evaluate_predictions(
        test_prediction.predictions, test_prediction.label_ids, split="test"
    )
    # 写入结构化测试指标。
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as writer:
        # 保留中文类别名。
        json.dump(test_metrics, writer, ensure_ascii=False, indent=2)
    # 打印各类和总体 F1，便于与 Qwen 结果比较。
    print(format_metric_report(test_metrics))
    # 显示指标保存路径。
    print(f"判别式基线测试指标已保存到：{output_dir / 'test_metrics.json'}")


# 只有直接运行脚本时才启动可选基线训练。
if __name__ == "__main__":
    # 执行主训练流程。
    main()
