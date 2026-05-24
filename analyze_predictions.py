"""基于评价脚本保存的逐句预测，生成实体错误类型与文本切片分析报告。"""

# 导入未来注解功能，让复杂报告字段的类型说明更自然。
from __future__ import annotations

# 导入命令行工具，让用户指定某一次实验的预测文件。
import argparse
# 导入 JSON 模块，将分析报告同时保存成机器可读格式。
import json
# 导入正则表达式模块，用于识别剂量、数字等医学文本特征。
import re
# 导入 defaultdict，按实体类别收集错误示例。
from collections import defaultdict
# 导入 Path，以跨平台方式指定输入输出文件。
from pathlib import Path
# 导入 Any 和 Callable，描述报告记录和切片判断函数。
from typing import Any, Callable

# 导入已有严格词级 F1 与 JSONL 读取函数，保证报告指标口径一致。
from ner_utils import ENTITY_TYPES, compute_entity_f1, entity_to_key, read_jsonl


def parse_args() -> argparse.Namespace:
    """读取预测明细路径和分析报告输出路径。"""

    # 建立命令行解析器。
    parser = argparse.ArgumentParser(description="分析 evaluate_ner.py 输出的预测明细并生成错误报告。")
    # 要分析的预测文件通常来自优化版模型的验证集或测试集评价。
    parser.add_argument(
        "--prediction_file",
        type=Path,
        required=True,
        help="evaluate_ner.py 生成的预测 JSONL 文件。",
    )
    # JSON 报告便于后续程序读取或画图。
    parser.add_argument(
        "--json_report",
        type=Path,
        default=Path("reports/error_analysis.json"),
        help="结构化误差报告输出文件。",
    )
    # Markdown 报告便于小白直接阅读。
    parser.add_argument(
        "--markdown_report",
        type=Path,
        default=Path("reports/error_analysis.md"),
        help="可读误差报告输出文件。",
    )
    # 控制每个错误类别最多展示几条原文例子。
    parser.add_argument("--max_examples", type=int, default=5, help="每个错误类别保存的最大例子数。")
    # 返回解析完成的参数。
    return parser.parse_args()


def spans_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """判断两个实体字符区间是否至少共享一个字符位置。"""

    # 左实体开始位置小于右实体结束且右实体开始小于左实体结束时发生重叠。
    return left["start"] < right["end"] and right["start"] < left["end"]


def classify_errors(record: dict[str, Any]) -> list[dict[str, Any]]:
    """将一条样本的严格实体错误分成漏检、边界错误、类别错误和多余预测。"""

    # 读取人工标注实体列表。
    gold_entities = record["gold_entities"]
    # 读取已经过协议校验的预测实体列表。
    predicted_entities = record["predicted_entities"]
    # 将金标准实体转为可精确比较的集合。
    gold_keys = {entity_to_key(entity) for entity in gold_entities}
    # 将预测实体转为可精确比较的集合。
    predicted_keys = {entity_to_key(entity) for entity in predicted_entities}
    # 找到未被完整正确预测的金标准实体。
    missed_gold = [entity for entity in gold_entities if entity_to_key(entity) not in predicted_keys]
    # 找到不是完整正确答案的模型预测实体。
    extra_predictions = [
        entity for entity in predicted_entities if entity_to_key(entity) not in gold_keys
    ]
    # 准备当前样本的错误事件列表。
    events: list[dict[str, Any]] = []
    # 逐个分析没有被精确匹配的真实实体。
    for gold_entity in missed_gold:
        # 找到文本边界相同但类别错误的预测。
        type_mismatches = [
            predicted
            for predicted in extra_predictions
            if predicted["start"] == gold_entity["start"]
            and predicted["end"] == gold_entity["end"]
            and predicted["entity"] == gold_entity["entity"]
            and predicted["type"] != gold_entity["type"]
        ]
        # 如果存在相同完整词但类别错误，将其记为类别混淆。
        if type_mismatches:
            # 保存类别错误事件。
            events.append(
                {
                    "error_type": "type_error",
                    "gold": gold_entity,
                    "prediction": type_mismatches[0],
                }
            )
            # 当前金标准实体的主要错误已经解释完成。
            continue
        # 找到与真实实体有字符重叠但边界未完全一致的预测。
        boundary_mismatches = [
            predicted for predicted in extra_predictions if spans_overlap(gold_entity, predicted)
        ]
        # 若存在重叠实体，将它视为边界错误示例。
        if boundary_mismatches:
            # 保存边界错误事件。
            events.append(
                {
                    "error_type": "boundary_error",
                    "gold": gold_entity,
                    "prediction": boundary_mismatches[0],
                }
            )
            # 当前漏检已有更具体原因。
            continue
        # 没有对应或重叠预测时，属于完全漏检。
        events.append({"error_type": "missed_entity", "gold": gold_entity, "prediction": None})
    # 逐个检查没有被解释为类型或边界替代的多余预测。
    for predicted_entity in extra_predictions:
        # 判断该预测是否已作为某个类别或边界错误的对应预测记录。
        already_linked = any(
            event["prediction"] is not None
            and entity_to_key(event["prediction"]) == entity_to_key(predicted_entity)
            for event in events
        )
        # 未被关联的多余实体属于纯粹假阳性。
        if not already_linked:
            # 保存假阳性事件。
            events.append(
                {"error_type": "spurious_entity", "gold": None, "prediction": predicted_entity}
            )
    # 返回该文本中的全部解释性错误事件。
    return events


def slice_records(
    records: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    """对满足文本特征的记录单独计算严格实体指标。"""

    # 按条件筛选目标切片样本。
    selected_records = [record for record in records if predicate(record)]
    # 没有任何样本时返回空切片结果，避免计算函数报错。
    if not selected_records:
        # 表明该切片在当前评价文件中不存在。
        return {"num_samples": 0, "support": 0, "micro_f1": None}
    # 取得切片中的金标准实体批次。
    gold_batches = [record["gold_entities"] for record in selected_records]
    # 取得切片中的预测实体批次。
    predicted_batches = [record["predicted_entities"] for record in selected_records]
    # 使用与最终评价相同函数计算该切片指标。
    metrics = compute_entity_f1(gold_batches, predicted_batches)
    # 返回最适合对比的样本数、实体数与 micro F1。
    return {
        "num_samples": len(selected_records),
        "support": metrics["overall"]["support"],
        "micro_f1": metrics["overall"]["f1"],
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    """把结构化误差统计转换成便于查看的 Markdown 报告。"""

    # 建立报告标题和总体指标区。
    lines = [
        "# 模型预测误差分析报告",
        "",
        f"输入预测文件：`{report['prediction_file']}`",
        "",
        "## 1. 总体严格实体指标",
        "",
        "| 样本数 | 真实实体数 | Precision | Recall | micro F1 | macro F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {report['num_samples']} | {report['overall']['support']} | "
            f"{report['overall']['precision']:.4f} | {report['overall']['recall']:.4f} | "
            f"{report['overall']['f1']:.4f} | {report['overall']['macro_f1']:.4f} |"
        ),
        "",
        "## 2. 错误类型统计",
        "",
        "| 错误类型 | 数量 | 说明 |",
        "| --- | ---: | --- |",
    ]
    # 提供四种错误类别的人类可读解释。
    error_descriptions = {
        "missed_entity": "真实实体完全没有对应预测",
        "boundary_error": "预测与真实实体有重叠，但词边界不一致",
        "type_error": "完整实体边界正确，但类别判断错误",
        "spurious_entity": "模型额外预测了不存在的实体",
    }
    # 按固定顺序展示各类错误数量。
    for error_type in ["missed_entity", "boundary_error", "type_error", "spurious_entity"]:
        # 写入当前错误类型统计。
        lines.append(
            f"| {error_type} | {report['error_counts'].get(error_type, 0)} | "
            f"{error_descriptions[error_type]} |"
        )
    # 添加按文本特征分组的切片指标。
    lines.extend(
        [
            "",
            "## 3. 文本切片指标",
            "",
            "| 切片 | 样本数 | 真实实体数 | micro F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    # 逐切片写入表现，用于找到更难的文本类型。
    for slice_name, slice_result in report["slices"].items():
        # 空切片使用短横线代替不存在的 F1 数值。
        f1_text = "-" if slice_result["micro_f1"] is None else f"{slice_result['micro_f1']:.4f}"
        # 添加当前切片一行结果。
        lines.append(
            f"| {slice_name} | {slice_result['num_samples']} | "
            f"{slice_result['support']} | {f1_text} |"
        )
    # 添加各实体类别指标部分。
    lines.extend(
        [
            "",
            "## 4. 各实体类别 F1",
            "",
            "| 类别 | Support | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    # 按类别固定顺序显示指标。
    for entity_type in ENTITY_TYPES:
        # 读取当前类别评价结果。
        result = report["per_type"][entity_type]
        # 添加类别指标一行。
        lines.append(
            f"| {entity_type} | {result['support']} | {result['precision']:.4f} | "
            f"{result['recall']:.4f} | {result['f1']:.4f} |"
        )
    # 添加错误示例说明。
    lines.extend(["", "## 5. 错误示例", ""])
    # 逐错误类别展示有限个样例。
    for error_type in ["missed_entity", "boundary_error", "type_error", "spurious_entity"]:
        # 添加小标题。
        lines.append(f"### {error_type}")
        # 读取当前类别示例列表。
        examples = report["error_examples"].get(error_type, [])
        # 如果没有该类型错误，直接说明。
        if not examples:
            # 添加无错误提示。
            lines.append("\n无此类错误样例。\n")
            # 继续下一错误类型。
            continue
        # 逐条打印文本与金标准/预测信息。
        for example in examples:
            # 添加样本 id 和原文。
            lines.append(f"- `{example['id']}` 文本：{example['text']}")
            # 添加金标准实体信息。
            lines.append(f"  - 金标准：`{example['gold']}`")
            # 添加模型预测实体信息。
            lines.append(f"  - 预测：`{example['prediction']}`")
        # 在小节间添加空行。
        lines.append("")
    # 拼接报告内容。
    return "\n".join(lines)


def main() -> None:
    """读取逐句预测、分析失败模式并输出报告。"""

    # 读取命令行参数。
    args = parse_args()
    # 检查评价脚本是否已生成预测明细。
    if not args.prediction_file.exists():
        # 提示用户先执行模型评价步骤。
        raise FileNotFoundError(f"找不到 {args.prediction_file}，请先运行 evaluate_ner.py。")
    # 读取每句话的预测明细。
    records = read_jsonl(args.prediction_file)
    # 输入文件为空时没有可分析内容。
    if not records:
        # 中止并明确说明原因。
        raise ValueError("预测文件为空，无法分析。")
    # 取得金标准实体列表用于总体指标。
    gold_batches = [record["gold_entities"] for record in records]
    # 取得最终有效预测实体列表用于总体指标。
    predicted_batches = [record["predicted_entities"] for record in records]
    # 计算与评价阶段同口径的整体和各类别指标。
    metrics = compute_entity_f1(gold_batches, predicted_batches)
    # 准备错误类型数量统计。
    error_counts: defaultdict[str, int] = defaultdict(int)
    # 准备每种错误类型的有限例子。
    error_examples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    # 遍历每条预测明细，拆解其错误事件。
    for record in records:
        # 对当前样本分类错误。
        events = classify_errors(record)
        # 逐个汇总错误事件。
        for event in events:
            # 取得错误类别。
            error_type = event["error_type"]
            # 累计错误数量。
            error_counts[error_type] += 1
            # 只保留规定条数的人工可读示例。
            if len(error_examples[error_type]) < args.max_examples:
                # 保存样本 id、原文及相关实体。
                error_examples[error_type].append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "gold": event["gold"],
                        "prediction": event["prediction"],
                    }
                )
    # 定义可帮助业务判断模型弱点的文本切片。
    slices = {
        "短文本(<=20字)": lambda record: len(record["text"]) <= 20,
        "中等文本(21-50字)": lambda record: 20 < len(record["text"]) <= 50,
        "长文本(>50字)": lambda record: len(record["text"]) > 50,
        "含数字": lambda record: bool(re.search(r"[0-9０-９]", record["text"])),
        "含剂量单位": lambda record: bool(re.search(r"[gGｇ克毫升片剂]", record["text"])),
        "含多个真实实体": lambda record: len(record["gold_entities"]) >= 2,
    }
    # 对每一个文本切片计算严格实体 F1。
    slice_results = {
        slice_name: slice_records(records, predicate) for slice_name, predicate in slices.items()
    }
    # 组合完整结构化分析报告。
    report = {
        "prediction_file": str(args.prediction_file),
        "num_samples": len(records),
        "overall": metrics["overall"],
        "per_type": metrics["per_type"],
        "error_counts": dict(error_counts),
        "error_examples": dict(error_examples),
        "slices": slice_results,
    }
    # 创建 JSON 报告父目录。
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    # 写入结构化错误分析。
    with args.json_report.open("w", encoding="utf-8") as writer:
        # 使用缩进并保留中文。
        json.dump(report, writer, ensure_ascii=False, indent=2)
    # 创建 Markdown 报告父目录。
    args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
    # 将可读报告写入 Markdown 文件。
    args.markdown_report.write_text(build_markdown_report(report), encoding="utf-8")
    # 打印报告位置与总体表现摘要。
    print(f"误差分析报告已保存到：{args.markdown_report}")
    # 打印结构化报告位置。
    print(f"结构化明细已保存到：{args.json_report}")
    # 输出总体 micro F1 便于快速核对。
    print(f"总体 micro F1：{metrics['overall']['f1']:.4f}")


# 用户直接执行该脚本时才启动误差分析。
if __name__ == "__main__":
    # 运行误差分析主函数。
    main()
