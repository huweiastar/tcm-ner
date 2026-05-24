"""只对训练集中的低频实体类别做可控重采样，保持验证集和测试集不变。"""

# 导入未来注解功能，使类型提示易于阅读。
from __future__ import annotations

# 导入命令行解析模块，让用户可以指定目标类别与目标数量。
import argparse
# 导入 JSON 模块，用来保存重采样前后的类别数量报告。
import json
# 导入随机数模块，用固定种子打乱增强训练集顺序。
import random
# 导入 Counter，统计各类完整实体和每条记录复制次数。
from collections import Counter
# 导入 Path，以便读写训练集和报告文件。
from pathlib import Path
# 导入 Any，表达 SFT 字典记录包含多种字段类型。
from typing import Any

# 导入固定类别、JSONL 读取与写入函数，确保输出仍能被训练脚本使用。
from ner_utils import ENTITY_TYPES, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    """读取重采样脚本的可调整参数。"""

    # 创建命令行解析器，强调该脚本只应输入训练集。
    parser = argparse.ArgumentParser(description="仅增强低频类别的训练 JSONL，禁止用于 validation/test。")
    # 默认使用已经消除精确文本泄漏的训练副本。
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("processed_data/clean/train.jsonl"),
        help="输入训练集 JSONL 文件。",
    )
    # 默认将增强后的训练集保存到 optimized 子目录。
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("processed_data/optimized/train_balanced.jsonl"),
        help="增强训练集输出位置。",
    )
    # 默认只增强审计中确认极低频的其他治疗类别。
    parser.add_argument(
        "--target_types",
        nargs="+",
        default=["其他治疗"],
        choices=ENTITY_TYPES,
        help="需要增加曝光次数的实体类别。",
    )
    # 设置每个目标类别希望至少达到的完整实体出现次数。
    parser.add_argument(
        "--target_entity_count",
        type=int,
        default=300,
        help="重采样后每个目标类别期望至少拥有的实体数量。",
    )
    # 限制同一原样本最多额外复制次数，防止极少量句子被过度背诵。
    parser.add_argument(
        "--max_extra_copies_per_record",
        type=int,
        default=4,
        help="每条原始训练样本允许新增的最大副本数。",
    )
    # 固定随机种子，确保多次生成训练集顺序一致。
    parser.add_argument("--seed", type=int, default=42, help="打乱增强训练集使用的随机种子。")
    # 指定记录类别分布变化的 JSON 报告位置。
    parser.add_argument(
        "--report_file",
        type=Path,
        default=Path("reports/rebalance_report.json"),
        help="重采样报告输出位置。",
    )
    # 返回解析后的所有参数。
    return parser.parse_args()


def count_entities(records: list[dict[str, Any]]) -> Counter[str]:
    """按完整实体词统计一批 SFT 记录的类别数量。"""

    # 创建空计数器。
    counter: Counter[str] = Counter()
    # 遍历每条训练记录。
    for record in records:
        # 遍历当前记录标注的实体词。
        for entity in record["gold_entities"]:
            # 按类型累计一个完整实体，而不是累计字符。
            counter[entity["type"]] += 1
    # 返回统计结果。
    return counter


def contains_target_type(record: dict[str, Any], target_type: str) -> bool:
    """判断一条样本是否包含指定低频实体类别。"""

    # 只要任一完整实体类别相同，就允许该样本成为候选副本。
    return any(entity["type"] == target_type for entity in record["gold_entities"])


def copy_training_record(record: dict[str, Any], copy_index: int, target_type: str) -> dict[str, Any]:
    """生成带追踪字段的训练样本副本，便于之后知道它为何出现多次。"""

    # 浅复制顶层字典；内部 prompt 和实体只读使用，不会被改写。
    copied_record = dict(record)
    # 为副本建立新 id，避免误以为它是新的人工样本。
    copied_record["id"] = f"{record['id']}-repeat-{copy_index:02d}"
    # 保存原样本 id，方便做过拟合误差分析。
    copied_record["oversampled_from_id"] = record["id"]
    # 保存该副本是为了增强哪个低频类别。
    copied_record["oversampled_for_type"] = target_type
    # 返回可以直接用于 SFT 的记录副本。
    return copied_record


def main() -> None:
    """读取干净训练集、增强低频类别并输出可复现实验数据。"""

    # 读取用户提供的重采样参数。
    args = parse_args()
    # 输入训练文件不存在时，提示先生成 clean 数据副本。
    if not args.input_file.exists():
        # 抛出带步骤提示的异常。
        raise FileNotFoundError(f"找不到 {args.input_file}，请先运行 create_clean_splits.py。")
    # 拒绝明显输入 validation 或 test 的文件名，防止无意污染最终评测。
    if args.input_file.name in {"validation.jsonl", "test.jsonl"}:
        # 明确说明训练增强只能针对 train。
        raise ValueError("重采样只能应用于训练集，不能应用于 validation 或 test。")
    # 读取要增强的训练记录。
    original_records = read_jsonl(args.input_file)
    # 统计增强前的完整实体类别数量。
    before_counts = count_entities(original_records)
    # 复制原样本列表作为最终增强列表的起点。
    balanced_records = list(original_records)
    # 复制类别计数器，使后续新增实体能持续更新。
    after_counts = Counter(before_counts)
    # 记录每条原训练样本已经被额外复制了几次。
    copy_counts: Counter[str] = Counter()
    # 记录因每个目标类别新增了多少条训练样本。
    added_by_target: Counter[str] = Counter()
    # 逐个增强用户选择的低频类别。
    for target_type in args.target_types:
        # 按原始顺序找到所有包含当前低频类别的候选句子。
        candidate_records = [
            record for record in original_records if contains_target_type(record, target_type)
        ]
        # 如果训练集中完全不存在该类别，复制无法凭空创造人工标注。
        if not candidate_records:
            # 输出提醒并继续处理其他类别。
            print(f"警告：训练集中没有 {target_type} 实体，无法通过重采样增强该类。")
            # 继续下一个目标类别。
            continue
        # 反复遍历候选样本，直到达到目标数量或触及复制上限。
        while after_counts[target_type] < args.target_entity_count:
            # 标记本轮是否成功加入过新副本，用于发现复制额度耗尽。
            added_in_round = False
            # 遍历每一条包含当前目标类别的候选样本。
            for record in candidate_records:
                # 读取原始样本 id。
                source_id = record["id"]
                # 如果该样本已达到副本上限，则跳过它。
                if copy_counts[source_id] >= args.max_extra_copies_per_record:
                    # 继续寻找还可复制的候选记录。
                    continue
                # 为该原样本累计一次新增副本。
                copy_counts[source_id] += 1
                # 创建并加入带来源标记的训练副本。
                balanced_records.append(
                    copy_training_record(record, copy_counts[source_id], target_type)
                )
                # 该类别本轮确实加入了新样本。
                added_in_round = True
                # 按该句中全部实体类别更新分布，因为复制会同时增加共现类别。
                for entity in record["gold_entities"]:
                    # 累加复制带来的完整实体数量。
                    after_counts[entity["type"]] += 1
                # 记录该目标类别对应新增了一条样本。
                added_by_target[target_type] += 1
                # 达到目标实体数后立即结束本轮添加。
                if after_counts[target_type] >= args.target_entity_count:
                    # 跳出候选样本遍历。
                    break
            # 如果所有候选样本都达到复制上限，避免 while 无限循环。
            if not added_in_round:
                # 打印实际达到的数量，提醒用户可调整复制上限。
                print(
                    f"警告：{target_type} 已触及副本上限，仅达到 {after_counts[target_type]} 个实体，"
                    f"未达到目标 {args.target_entity_count}。"
                )
                # 停止增强当前类别。
                break
    # 创建带固定随机种子的生成器，以稳定打乱训练顺序。
    random_generator = random.Random(args.seed)
    # 打乱增强列表，避免所有重采样样本集中出现在一个 epoch 结尾。
    random_generator.shuffle(balanced_records)
    # 输出增强后的训练 JSONL。
    write_jsonl(balanced_records, args.output_file)
    # 构建类别变化与重采样来源统计报告。
    report = {
        "input_file": str(args.input_file),
        "output_file": str(args.output_file),
        "target_types": args.target_types,
        "target_entity_count": args.target_entity_count,
        "max_extra_copies_per_record": args.max_extra_copies_per_record,
        "seed": args.seed,
        "original_num_samples": len(original_records),
        "balanced_num_samples": len(balanced_records),
        "added_samples": len(balanced_records) - len(original_records),
        "added_samples_by_target_type": dict(added_by_target),
        "before_entity_counts": {entity_type: before_counts[entity_type] for entity_type in ENTITY_TYPES},
        "after_entity_counts": {entity_type: after_counts[entity_type] for entity_type in ENTITY_TYPES},
    }
    # 创建报告父目录。
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    # 将重采样详情写入 JSON 文件。
    with args.report_file.open("w", encoding="utf-8") as writer:
        # 保留中文并使用缩进格式。
        json.dump(report, writer, ensure_ascii=False, indent=2)
    # 打印样本总量变化。
    print(f"训练样本数：{len(original_records)} -> {len(balanced_records)}")
    # 逐一打印目标类别在增强前后的实体数量。
    for target_type in args.target_types:
        # 输出当前目标类别变化。
        print(f"{target_type} 实体数：{before_counts[target_type]} -> {after_counts[target_type]}")
    # 打印增强数据输出路径。
    print(f"增强训练集已保存到：{args.output_file}")
    # 打印重采样报告路径。
    print(f"重采样报告已保存到：{args.report_file}")


# 仅当用户直接运行文件时才进行训练集重采样。
if __name__ == "__main__":
    # 启动主处理流程。
    main()
