"""审计中医药 BIO 数据的质量、集合泄漏风险、标注冲突和类别不平衡情况。"""

# 导入未来注解功能，使类型标注在较旧 Python 中也更清晰。
from __future__ import annotations

# 导入命令行解析模块，让初学者可以指定数据与报告目录。
import argparse
# 导入哈希模块，用于为每份原始数据生成可追踪的数据指纹。
import hashlib
# 导入 JSON 模块，用于保存可被程序继续读取的完整审计报告。
import json
# 导入 Unicode 工具，用于检查全角半角归一化可能改变的文本。
import unicodedata
# 导入 defaultdict 与 Counter，方便汇总重复文本和实体数量。
from collections import Counter, defaultdict
# 导入 combinations，逐对比较 train、validation、test 是否存在重复文本。
from itertools import combinations
# 导入 Path，以跨平台方式处理数据和报告文件位置。
from pathlib import Path
# 导入 Any，清楚表达报告字典可包含不同类型字段。
from typing import Any

# 导入数据转换函数与固定类别列表，确保审计与训练遵循同一口径。
from ner_utils import ENTITY_TYPES, make_sft_record, read_bio_sentences


def parse_args() -> argparse.Namespace:
    """读取审计脚本支持的输入与输出路径。"""

    # 建立命令行解析器，并说明脚本不会修改任何原始标注。
    parser = argparse.ArgumentParser(description="检查 BIO 数据质量、泄漏风险和类别分布，不修改原始文件。")
    # 接受原始数据所在文件夹，默认使用作业提供的 data 目录。
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="原始 medical.* 文件目录。")
    # 接受报告保存目录，默认统一放到 reports 文件夹。
    parser.add_argument("--report_dir", type=Path, default=Path("reports"), help="审计报告输出目录。")
    # 返回解析完成的参数对象。
    return parser.parse_args()


def sha256_file(file_path: Path) -> str:
    """计算一个文件的 SHA256 指纹，便于确认实验使用的是哪一版数据。"""

    # 创建 SHA256 计算器。
    digest = hashlib.sha256()
    # 以二进制模式打开文件，以保持原始字节内容不被换行转换影响。
    with file_path.open("rb") as reader:
        # 每次读取一小块数据，避免大文件一次性占用过多内存。
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            # 将当前数据块纳入哈希计算。
            digest.update(chunk)
    # 返回十六进制表示的数据指纹。
    return digest.hexdigest()


def entity_signature(entities: list[dict[str, Any]]) -> str:
    """将实体标注转换为稳定字符串，以比较相同文本是否被标成不同答案。"""

    # 用固定字段顺序和中文原文序列化实体列表，保证可直接比较。
    return json.dumps(entities, ensure_ascii=False, sort_keys=True)


def load_records(data_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """读取三个 BIO 划分并转为包含文本和实体的审计记录。"""

    # 建立训练框架中的划分名称与原始文件名映射。
    split_files = {
        "train": data_dir / "medical.train",
        "validation": data_dir / "medical.dev",
        "test": data_dir / "medical.test",
    }
    # 准备保存每一划分记录的字典。
    all_records: dict[str, list[dict[str, Any]]] = {}
    # 准备保存每一原始文件数据指纹的字典。
    file_hashes: dict[str, str] = {}
    # 依次读取训练、验证和测试数据。
    for split, file_path in split_files.items():
        # 数据文件缺失时，报告无法可信生成，因此立即指出问题。
        if not file_path.exists():
            # 抛出带文件位置的异常，帮助小白检查文件放置位置。
            raise FileNotFoundError(f"缺少原始数据文件：{file_path}")
        # 用训练相同的严格读取函数检查标签格式和 BIO 合法性。
        sentences = read_bio_sentences(file_path)
        # 将每个句子还原成文本和词级实体，便于比较泄漏和冲突。
        records = [
            make_sft_record(sentence, split=split, sample_index=index)
            for index, sentence in enumerate(sentences, start=1)
        ]
        # 保存当前划分全部记录。
        all_records[split] = records
        # 保存当前原始文件的 SHA256 数据指纹。
        file_hashes[split] = sha256_file(file_path)
    # 返回审计所需的记录和原始文件指纹。
    return all_records, file_hashes


def summarize_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    """统计一个数据划分的样本数、文本长度和实体类别分布。"""

    # 统计每条文本长度，便于评估最大上下文需求。
    lengths = [len(record["text"]) for record in records]
    # 创建实体类别计数器。
    entity_counts: Counter[str] = Counter()
    # 遍历每一条数据记录。
    for record in records:
        # 遍历当前文本中标注的完整实体词。
        for entity in record["gold_entities"]:
            # 按实体类别累计完整实体数量。
            entity_counts[entity["type"]] += 1
    # 返回适合写入报告的统计结构。
    return {
        "num_samples": len(records),
        "min_text_characters": min(lengths) if lengths else 0,
        "max_text_characters": max(lengths) if lengths else 0,
        "average_text_characters": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "entity_counts": {entity_type: entity_counts[entity_type] for entity_type in ENTITY_TYPES},
    }


def find_within_split_duplicates(records: list[dict[str, Any]]) -> dict[str, Any]:
    """查找一个划分内部文本完全相同的记录以及其中的标注冲突。"""

    # 按原始文本聚合同一划分中的所有记录。
    by_text: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    # 遍历当前划分的每条记录。
    for record in records:
        # 将相同文本的记录加入同一列表。
        by_text[record["text"]].append(record)
    # 找到出现超过一次的文本组。
    duplicated_groups = [items for items in by_text.values() if len(items) > 1]
    # 准备用于展示的重复示例。
    examples: list[dict[str, Any]] = []
    # 统计同文本却出现不同标注的组数。
    conflict_group_count = 0
    # 逐组判断标注答案是否一致。
    for items in duplicated_groups:
        # 取出当前文本所有不同的实体标注签名。
        signatures = {entity_signature(item["gold_entities"]) for item in items}
        # 超过一个签名说明同样的文本在本划分中存在标注冲突。
        has_conflict = len(signatures) > 1
        # 对冲突组进行计数。
        if has_conflict:
            # 累加冲突组数量。
            conflict_group_count += 1
        # 报告只展示前十组示例，避免文本过多影响可阅读性。
        if len(examples) < 10:
            # 保存样本 id、文本和冲突状态供人工复查。
            examples.append(
                {
                    "text": items[0]["text"],
                    "ids": [item["id"] for item in items],
                    "occurrences": len(items),
                    "annotation_conflict": has_conflict,
                }
            )
    # 返回内部重复概况和示例。
    return {
        "duplicate_text_groups": len(duplicated_groups),
        "duplicate_extra_samples": sum(len(items) - 1 for items in duplicated_groups),
        "annotation_conflict_groups": conflict_group_count,
        "examples": examples,
    }


def compare_split_overlap(
    left_name: str,
    left_records: list[dict[str, Any]],
    right_name: str,
    right_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """比较两个划分是否包含相同原文，从而发现评测泄漏风险。"""

    # 为左侧划分建立文本到第一条记录的快速查找表。
    left_by_text = {record["text"]: record for record in left_records}
    # 为右侧划分建立文本到第一条记录的快速查找表。
    right_by_text = {record["text"]: record for record in right_records}
    # 取得两个划分中都出现过的完全相同文本。
    shared_texts = sorted(set(left_by_text) & set(right_by_text))
    # 统计同文本且标注完全一致的重叠数量。
    same_annotation_count = 0
    # 统计同文本但标注不一致的冲突数量。
    conflict_count = 0
    # 准备有限数量的人工复核示例。
    examples: list[dict[str, Any]] = []
    # 逐个检查重叠文本的实体标注。
    for text in shared_texts:
        # 读取左侧文本对应的词级实体答案。
        left_entities = left_by_text[text]["gold_entities"]
        # 读取右侧文本对应的词级实体答案。
        right_entities = right_by_text[text]["gold_entities"]
        # 比较稳定序列化后的实体答案是否一致。
        same_annotation = entity_signature(left_entities) == entity_signature(right_entities)
        # 按比较结果累计一致或冲突数量。
        if same_annotation:
            # 这是潜在数据泄漏：测试内容已在训练或验证中出现。
            same_annotation_count += 1
        else:
            # 这是更严重的同文本标注冲突问题。
            conflict_count += 1
        # 仅保存前十个重叠例子以保持报告清爽。
        if len(examples) < 10:
            # 把可人工检查的两个 id 与答案写入报告。
            examples.append(
                {
                    "text": text,
                    f"{left_name}_id": left_by_text[text]["id"],
                    f"{right_name}_id": right_by_text[text]["id"],
                    "same_annotation": same_annotation,
                    f"{left_name}_entities": left_entities,
                    f"{right_name}_entities": right_entities,
                }
            )
    # 返回当前两个划分的重叠结果。
    return {
        "left_split": left_name,
        "right_split": right_name,
        "shared_text_count": len(shared_texts),
        "same_annotation_count": same_annotation_count,
        "annotation_conflict_count": conflict_count,
        "examples": examples,
    }


def find_unicode_normalization_risks(all_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """查找 NFKC 归一化会改变原文或让不同原文变相相同的样本。"""

    # 按 NFKC 归一化后的文本收集原始文本与所在划分。
    normalized_groups: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)
    # 统计每个划分有多少文本会被 NFKC 改写。
    changed_by_split: Counter[str] = Counter()
    # 保存少量发生变化的样例，说明为什么不能贸然归一化。
    changed_examples: list[dict[str, str]] = []
    # 遍历三份数据中的所有文本。
    for split, records in all_records.items():
        # 遍历当前划分每条记录。
        for record in records:
            # 使用 NFKC 模拟将全角数字和符号转换为常见半角表示。
            normalized_text = unicodedata.normalize("NFKC", record["text"])
            # 将当前样本加入其归一化文本组。
            normalized_groups[normalized_text].append((split, record["id"], record["text"]))
            # 判断归一化是否实际改变了原始医学文本。
            if normalized_text != record["text"]:
                # 按划分统计变化样本数。
                changed_by_split[split] += 1
                # 仅保存最前面的十个变化例子。
                if len(changed_examples) < 10:
                    # 同时展示原文和归一化版本，提醒位置可能变化。
                    changed_examples.append(
                        {"split": split, "id": record["id"], "original": record["text"], "nfkc": normalized_text}
                    )
    # 准备查找归一化后发生碰撞的不同原文。
    collision_examples: list[dict[str, Any]] = []
    # 统计归一化碰撞的文本组数量。
    collision_count = 0
    # 遍历按归一化文本聚集的样本。
    for normalized_text, items in normalized_groups.items():
        # 取得该组中原本不同的文本形式。
        original_texts = {item[2] for item in items}
        # 至少两个原文变为同一字符串时才属于归一化碰撞。
        if len(original_texts) > 1:
            # 累加碰撞组数。
            collision_count += 1
            # 保存前十组可读示例。
            if len(collision_examples) < 10:
                # 展示碰撞前后的文本及划分位置。
                collision_examples.append(
                    {
                        "normalized_text": normalized_text,
                        "original_occurrences": [
                            {"split": split, "id": sample_id, "text": original}
                            for split, sample_id, original in items
                        ],
                    }
                )
    # 返回归一化风险概况。
    return {
        "changed_sample_counts": {split: changed_by_split[split] for split in all_records},
        "changed_examples": changed_examples,
        "normalization_collision_groups": collision_count,
        "collision_examples": collision_examples,
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    """将结构化审计结果写成适合初学者查看的 Markdown 文本。"""

    # 创建报告标题和结论提示。
    lines = [
        "# 数据质量审计报告",
        "",
        "该报告由 `audit_dataset.py` 自动生成。它只检查并报告风险，不会修改原始数据。",
        "",
        "## 1. 文件指纹与数据规模",
        "",
        "| 划分 | 样本数 | 最长字符数 | SHA256 前 12 位 |",
        "| --- | ---: | ---: | --- |",
    ]
    # 向规模表格写入每一数据划分。
    for split in ["train", "validation", "test"]:
        # 读取当前划分的统计信息。
        split_summary = report["split_summary"][split]
        # 将样本量、最长文本与短哈希写入表格。
        lines.append(
            f"| {split} | {split_summary['num_samples']} | "
            f"{split_summary['max_text_characters']} | {report['file_sha256'][split][:12]} |"
        )
    # 添加实体类别数量表。
    lines.extend(
        [
            "",
            "## 2. 完整实体类别数量",
            "",
            "| 类别 | train | validation | test |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    # 逐类别展示训练、验证和测试中的真实实体数。
    for entity_type in ENTITY_TYPES:
        # 写入当前类别在三份数据中的数量。
        lines.append(
            f"| {entity_type} | {report['split_summary']['train']['entity_counts'][entity_type]} | "
            f"{report['split_summary']['validation']['entity_counts'][entity_type]} | "
            f"{report['split_summary']['test']['entity_counts'][entity_type]} |"
        )
    # 添加划分内部重复检查结果。
    lines.extend(["", "## 3. 划分内部重复与标注冲突", "", "| 划分 | 重复文本组 | 多出的重复样本 | 标注冲突组 |", "| --- | ---: | ---: | ---: |"])
    # 写入每个划分的内部重复情况。
    for split in ["train", "validation", "test"]:
        # 读取当前划分内部重复结果。
        result = report["within_split_duplicates"][split]
        # 添加当前划分的一行统计。
        lines.append(
            f"| {split} | {result['duplicate_text_groups']} | "
            f"{result['duplicate_extra_samples']} | {result['annotation_conflict_groups']} |"
        )
    # 添加跨划分泄漏检查结果。
    lines.extend(["", "## 4. 跨划分相同文本检查", "", "| 比较划分 | 相同文本数 | 同标注重叠 | 标注冲突 |", "| --- | ---: | ---: | ---: |"])
    # 逐对写入 train-validation、train-test、validation-test 重叠情况。
    for overlap in report["cross_split_overlap"]:
        # 组合当前比较的两个划分名称。
        pair_name = f"{overlap['left_split']} / {overlap['right_split']}"
        # 写入相同文本与冲突计数。
        lines.append(
            f"| {pair_name} | {overlap['shared_text_count']} | "
            f"{overlap['same_annotation_count']} | {overlap['annotation_conflict_count']} |"
        )
    # 读取归一化风险结果。
    unicode_risk = report["unicode_normalization_risks"]
    # 添加 Unicode 风险说明。
    lines.extend(
        [
            "",
            "## 5. 全半角与 Unicode 风险",
            "",
            "NER 使用字符位置作为边界，因此不能在未同步修正标注位置时直接将全角数字或符号转为半角。",
            "",
            "| 划分 | NFKC 会改写的样本数 |",
            "| --- | ---: |",
        ]
    )
    # 写入各划分会发生文本变化的样本数量。
    for split in ["train", "validation", "test"]:
        # 加入当前划分的一行。
        lines.append(f"| {split} | {unicode_risk['changed_sample_counts'][split]} |")
    # 输出归一化碰撞总量。
    lines.append(f"\n归一化后不同原文变为同一文本的组数：**{unicode_risk['normalization_collision_groups']}**。")
    # 输出自动生成的处理建议。
    lines.extend(["", "## 6. 建议动作", ""])
    # 逐条写出面向执行的建议。
    for recommendation in report["recommendations"]:
        # 使用 Markdown 列表显示一条建议。
        lines.append(f"- {recommendation}")
    # 添加 JSON 报告用途说明。
    lines.extend(["", "更详细的重复文本示例和归一化样例请查看同目录下的 `data_audit.json`。", ""])
    # 拼接全部 Markdown 行。
    return "\n".join(lines)


def main() -> None:
    """读取真实数据、执行全部审计规则并保存两种报告文件。"""

    # 读取用户指定的数据目录和输出目录。
    args = parse_args()
    # 加载并严格校验三份 BIO 数据。
    all_records, file_hashes = load_records(args.data_dir)
    # 按三份数据分别统计规模和类别分布。
    split_summary = {split: summarize_split(records) for split, records in all_records.items()}
    # 检查每份数据内部是否重复或冲突。
    within_duplicates = {
        split: find_within_split_duplicates(records) for split, records in all_records.items()
    }
    # 准备跨划分文本重复检查结果。
    cross_overlap: list[dict[str, Any]] = []
    # 两两比较三个划分，避免漏掉 validation 与 test 的重叠。
    for left_name, right_name in combinations(["train", "validation", "test"], 2):
        # 保存当前一对划分的重复和冲突结果。
        cross_overlap.append(
            compare_split_overlap(left_name, all_records[left_name], right_name, all_records[right_name])
        )
    # 检查全角数字、符号等 Unicode 归一化可能带来的边界风险。
    unicode_risks = find_unicode_normalization_risks(all_records)
    # 汇总所有与训练或测试相关的跨划分重叠文本数量。
    cross_overlap_count = sum(result["shared_text_count"] for result in cross_overlap)
    # 汇总所有跨划分标注冲突数量。
    cross_conflict_count = sum(result["annotation_conflict_count"] for result in cross_overlap)
    # 读取训练集类别计数，以发现极低频类别。
    train_entity_counts = split_summary["train"]["entity_counts"]
    # 将少于 100 个实体的类别视为需要重点观察的小类别。
    rare_train_types = [
        entity_type for entity_type in ENTITY_TYPES if train_entity_counts[entity_type] < 100
    ]
    # 准备可执行建议列表。
    recommendations: list[str] = []
    # 根据跨划分重复情况追加泄漏处理建议。
    if cross_overlap_count:
        # 相同文本跨集合出现时，建议先复核再决定是否重划数据。
        recommendations.append(
            f"发现 {cross_overlap_count} 条跨划分相同文本；正式报告前应复核其来源，必要时按文档去重后重新划分。"
        )
    else:
        # 没有发现精确文本泄漏时也记录这一正面结果。
        recommendations.append("未发现完全相同文本跨划分泄漏，可继续保留当前划分用于基线实验。")
    # 标注冲突会直接影响训练与测试一致性，需要优先复核。
    if cross_conflict_count:
        # 指明冲突数量和需要查看的详细文件。
        recommendations.append(
            f"发现 {cross_conflict_count} 组跨划分标注冲突，请在 `data_audit.json` 中逐条人工复核后再训练。"
        )
    # 如果存在低频类别，则建议只在训练集做重采样。
    if rare_train_types:
        # 把类别名称写入建议并引导使用重采样脚本。
        recommendations.append(
            f"训练集低频类别包括 {', '.join(rare_train_types)}；可运行 `rebalance_train_data.py` 仅增强训练集。"
        )
    # 总是提醒用户保留原始全角内容以维持实体跨度。
    recommendations.append("保持原文全角字符不变；若未来需要文本归一化，必须同步重新计算全部实体 start/end。")
    # 组织完整的机器可读报告结构。
    report = {
        "file_sha256": file_hashes,
        "split_summary": split_summary,
        "within_split_duplicates": within_duplicates,
        "cross_split_overlap": cross_overlap,
        "unicode_normalization_risks": unicode_risks,
        "rare_train_entity_types_below_100": rare_train_types,
        "recommendations": recommendations,
    }
    # 创建报告目录；目录已经存在时不会报错。
    args.report_dir.mkdir(parents=True, exist_ok=True)
    # 指定 JSON 完整报告文件位置。
    json_report_path = args.report_dir / "data_audit.json"
    # 用 UTF-8 写入完整结构化审计结果。
    with json_report_path.open("w", encoding="utf-8") as writer:
        # 保留中文并添加缩进，让初学者也能直接查看。
        json.dump(report, writer, ensure_ascii=False, indent=2)
    # 指定更适合人阅读的 Markdown 报告文件位置。
    markdown_report_path = args.report_dir / "data_audit.md"
    # 写入经过表格排版的 Markdown 报告。
    markdown_report_path.write_text(build_markdown_report(report), encoding="utf-8")
    # 在终端打印报告生成位置。
    print(f"数据审计完成：{markdown_report_path}")
    # 在终端打印结构化明细位置。
    print(f"详细 JSON 明细：{json_report_path}")
    # 打印跨划分重叠摘要，便于不打开文件也先看到关键风险。
    print(f"跨划分相同文本总数：{cross_overlap_count}；跨划分标注冲突总数：{cross_conflict_count}")
    # 打印训练集中需要关注的低频类别。
    print(f"训练集实体数少于 100 的类别：{rare_train_types if rare_train_types else '无'}")


# 只有用户直接执行脚本时才运行审计，不影响被其他文件导入。
if __name__ == "__main__":
    # 启动数据审计主流程。
    main()
