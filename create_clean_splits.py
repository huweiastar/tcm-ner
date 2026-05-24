"""从 SFT JSONL 建立无精确文本跨划分泄漏的实验副本，不覆盖原始数据。"""

# 导入未来注解功能，让类型标注更加直观。
from __future__ import annotations

# 导入参数解析模块，让用户选择输入与输出文件夹。
import argparse
# 导入 JSON 模块，用来保存去重动作明细。
import json
# 导入 defaultdict，按原始文本聚合同一划分中的记录。
from collections import defaultdict
# 导入 Path，以跨系统方式操作文件路径。
from pathlib import Path
# 导入 Any，标注记录字典包含多种类型字段。
from typing import Any

# 导入现有 JSONL 读写函数，使数据格式与预处理阶段完全一致。
from ner_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    """读取输入目录、输出目录与报告文件位置。"""

    # 创建命令行解析器并解释该脚本保留原始数据。
    parser = argparse.ArgumentParser(description="建立不含完全相同文本跨划分重叠的 SFT 数据副本。")
    # 输入是 prepare_sft_data.py 已生成的 JSONL 目录。
    parser.add_argument(
        "--input_dir", type=Path, default=Path("processed_data"), help="原始 SFT JSONL 数据目录。"
    )
    # 输出放在新的子目录中，避免改变可复现的原始处理结果。
    parser.add_argument(
        "--output_dir", type=Path, default=Path("processed_data/clean"), help="无泄漏副本输出目录。"
    )
    # 报告记录具体丢弃了哪些重复记录。
    parser.add_argument(
        "--report_file", type=Path, default=Path("reports/clean_split_report.json"), help="去重明细报告。"
    )
    # 默认不自动处理同文本标注冲突；教学试跑可显式选择保留第一条并记录风险。
    parser.add_argument(
        "--within_conflict_policy",
        choices=["error", "keep_first"],
        default="error",
        help="同一划分同文本标注冲突时的处理策略；正式实验建议保留 error 并人工复核。",
    )
    # 返回解析后的参数。
    return parser.parse_args()


def annotations_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """判断两条相同文本记录的完整实体答案是否完全相同。"""

    # 直接比较词级实体列表，其中包含类别、原文和起止位置。
    return left["gold_entities"] == right["gold_entities"]


def deduplicate_inside_split(
    split: str, records: list[dict[str, Any]], conflict_policy: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """移除同一划分中的完全重复文本；遇到标注冲突则停止并要求复核。"""

    # 按文本保存第一次遇到的记录。
    first_by_text: dict[str, dict[str, Any]] = {}
    # 保存没有重复的干净记录序列。
    kept_records: list[dict[str, Any]] = []
    # 保存因重复而被移除的记录说明。
    removed_records: list[dict[str, Any]] = []
    # 按原顺序检查每条样本，以便结果具有确定性。
    for record in records:
        # 读取当前样本原始文本。
        text = record["text"]
        # 第一次看到该文本时直接保留。
        if text not in first_by_text:
            # 将当前记录设为该文本的代表记录。
            first_by_text[text] = record
            # 保持该记录进入干净输出文件。
            kept_records.append(record)
            # 继续检查下一个样本。
            continue
        # 读取当前文本此前保留的记录。
        existing_record = first_by_text[text]
        # 同一划分同文本却标注不同，脚本不能替用户武断选择答案。
        if not annotations_equal(existing_record, record):
            # 默认抛出详细错误，提示先人工复核数据审计报告。
            if conflict_policy == "error":
                # 停止生成，防止不知情地把任意标签当作标准答案。
                raise ValueError(
                    f"{split} 中同文本出现标注冲突：{existing_record['id']} 与 {record['id']}；"
                    "请先人工修订标注，或仅为教学试跑显式传入 --within_conflict_policy keep_first。"
                )
            # 教学试跑模式下保留第一次标注，并将冲突丢弃动作记录到报告。
            removed_records.append(
                {
                    "removed_id": record["id"],
                    "kept_id": existing_record["id"],
                    "text": text,
                    "reason": "same_split_annotation_conflict_keep_first",
                    "kept_entities": existing_record["gold_entities"],
                    "removed_entities": record["gold_entities"],
                }
            )
            # 当前冲突记录已经记录且不再保留。
            continue
        # 记录该条样本因同划分完全重复而被移除。
        removed_records.append(
            {
                "removed_id": record["id"],
                "kept_id": existing_record["id"],
                "text": text,
                "reason": "same_split_exact_duplicate",
            }
        )
    # 返回去重后的记录及丢弃说明。
    return kept_records, removed_records


def remove_cross_split_overlap(
    split: str,
    records: list[dict[str, Any]],
    protected_texts: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从较低优先级划分移除已经被保留在评测划分中的相同文本。"""

    # 准备当前划分仍将保留的记录。
    kept_records: list[dict[str, Any]] = []
    # 准备当前划分因跨集合重叠被移除的记录。
    removed_records: list[dict[str, Any]] = []
    # 遍历当前划分中的每条唯一文本记录。
    for record in records:
        # 读取原始文本作为泄漏比较键。
        text = record["text"]
        # 如果该文本已属于高优先级划分，就从当前划分移除。
        if text in protected_texts:
            # 保存移除原因和优先保留在哪个划分中。
            removed_records.append(
                {
                    "removed_id": record["id"],
                    "protected_split": protected_texts[text],
                    "text": text,
                    "reason": "cross_split_exact_text_overlap",
                }
            )
            # 不把该记录添加到输出文件。
            continue
        # 当前文本尚未在更高优先级划分出现，因此可以保留。
        kept_records.append(record)
        # 标记它属于当前划分，使更低优先级数据不能再次出现。
        protected_texts[text] = split
    # 返回当前划分保留记录和移除明细。
    return kept_records, removed_records


def main() -> None:
    """生成保留测试集完整性的无泄漏数据副本，并保存去重报告。"""

    # 读取命令行参数。
    args = parse_args()
    # 按文件名定义原 JSONL 的三个划分。
    input_files = {
        "train": args.input_dir / "train.jsonl",
        "validation": args.input_dir / "validation.jsonl",
        "test": args.input_dir / "test.jsonl",
    }
    # 准备加载所有输入记录的字典。
    loaded_records: dict[str, list[dict[str, Any]]] = {}
    # 逐个检查并读取输入文件。
    for split, file_path in input_files.items():
        # 缺少预处理数据时无法安全去重。
        if not file_path.exists():
            # 提示用户应先运行 prepare_sft_data.py。
            raise FileNotFoundError(f"找不到 {file_path}，请先运行 prepare_sft_data.py。")
        # 读取当前 JSONL 文件。
        loaded_records[split] = read_jsonl(file_path)
    # 准备用于输出的干净记录。
    clean_records: dict[str, list[dict[str, Any]]] = {}
    # 准备记录所有删除动作。
    removed_by_split: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    # 首先在各自划分内部移除完全重复，冲突数据会直接阻止继续操作。
    for split, records in loaded_records.items():
        # 调用内部重复清理函数。
        unique_records, removed_records = deduplicate_inside_split(
            split, records, args.within_conflict_policy
        )
        # 暂时保存内部去重后的记录。
        clean_records[split] = unique_records
        # 记录内部去重删除动作。
        removed_by_split[split].extend(removed_records)
    # 维护已经被较高优先级划分占用的文本。
    protected_texts: dict[str, str] = {}
    # 按 test、validation、train 顺序处理，优先保留最终评价数据完整性。
    for split in ["test", "validation", "train"]:
        # 去除当前划分与高优先级划分重复的文本。
        kept_records, removed_records = remove_cross_split_overlap(
            split, clean_records[split], protected_texts
        )
        # 覆盖为最终应写入文件的记录。
        clean_records[split] = kept_records
        # 保存跨划分去重删除动作。
        removed_by_split[split].extend(removed_records)
    # 确保干净数据输出目录存在。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 按训练代码预期文件名写出三份干净 JSONL。
    for split, records in clean_records.items():
        # 输出当前划分记录。
        write_jsonl(records, args.output_dir / f"{split}.jsonl")
    # 构建便于查看去除规模和具体文本的报告。
    report = {
        "priority_rule": "保留优先级为 test > validation > train；不覆盖原始 processed_data。",
        "within_conflict_policy": args.within_conflict_policy,
        "original_sample_counts": {split: len(records) for split, records in loaded_records.items()},
        "clean_sample_counts": {split: len(records) for split, records in clean_records.items()},
        "removed_sample_counts": {split: len(removed_by_split[split]) for split in loaded_records},
        "removed_records": dict(removed_by_split),
    }
    # 创建报告父目录。
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    # 写入包含具体移除样本的 JSON 报告。
    with args.report_file.open("w", encoding="utf-8") as writer:
        # 保留中文且用缩进格式输出。
        json.dump(report, writer, ensure_ascii=False, indent=2)
    # 向用户展示每个划分在清理前后的样本数。
    for split in ["train", "validation", "test"]:
        # 打印当前划分变化。
        print(
            f"{split:<10} 原样本={len(loaded_records[split]):>5}  "
            f"clean样本={len(clean_records[split]):>5}  移除={len(removed_by_split[split]):>3}"
        )
    # 打印数据副本和报告所在位置。
    print(f"无精确文本泄漏的数据副本已保存到：{args.output_dir}")
    # 打印审计动作明细路径。
    print(f"去重明细已保存到：{args.report_file}")


# 只在用户直接运行本文件时执行去重流程。
if __name__ == "__main__":
    # 启动干净划分生成入口。
    main()
