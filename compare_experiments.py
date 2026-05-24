"""比较多个验证集 metrics.json，以严格实体 F1 选择最优实验配置。"""

# 导入未来注解功能，使返回值类型写法简洁。
from __future__ import annotations

# 导入参数解析模块，让用户一次提供多个实验指标文件。
import argparse
# 导入 JSON 模块，读取 evaluate_ner.py 保存的结构化指标。
import json
# 导入 Path，以便检查指标文件并输出排行榜。
from pathlib import Path
# 导入 Any，为结构化比较结果提供类型标注。
from typing import Any


def parse_args() -> argparse.Namespace:
    """读取若干验证集指标文件和排行榜输出位置。"""

    # 创建命令行解析器，强调必须使用 validation 指标选方案。
    parser = argparse.ArgumentParser(description="按 validation 的严格实体 micro F1 排序多个实验。")
    # 接收一个或多个 metrics JSON 路径。
    parser.add_argument(
        "--metric_files",
        type=Path,
        nargs="+",
        required=True,
        help="一个或多个验证集 evaluate_ner.py 指标 JSON 文件。",
    )
    # 指定排行榜 Markdown 输出路径。
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("reports/validation_leaderboard.md"),
        help="实验比较报告输出路径。",
    )
    # 返回参数。
    return parser.parse_args()


def read_metric_file(file_path: Path) -> dict[str, Any]:
    """读取一份 metrics JSON 并提取排行榜需要的字段。"""

    # 指标文件缺失时不能进行公平比较。
    if not file_path.exists():
        # 抛出清楚的缺失文件提示。
        raise FileNotFoundError(f"找不到指标文件：{file_path}")
    # 以 UTF-8 打开包含中文类别名的指标 JSON。
    with file_path.open("r", encoding="utf-8") as reader:
        # 将结构化指标还原为 Python 字典。
        metrics = json.load(reader)
    # 读取总体评价字段。
    overall = metrics["overall"]
    # 读取可能存在的结构化生成质量字段。
    generation_quality = metrics.get("generation_quality", {})
    # 返回用于排序和展示的扁平记录。
    return {
        "metrics_file": str(file_path),
        "experiment": file_path.parent.name,
        "micro_f1": overall["f1"],
        "macro_f1": overall["macro_f1"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "support": overall["support"],
        "invalid_entities": generation_quality.get("final_invalid_entity_count", "-"),
    }


def build_markdown(rows: list[dict[str, Any]]) -> str:
    """将按 F1 排序的实验记录排版为排行榜。"""

    # 添加标题、选模原则和表头。
    lines = [
        "# Validation 实验排行榜",
        "",
        "只能根据验证集指标选择配置；确定配置后，测试集只运行一次用于最终报告。",
        "",
        "| 排名 | 实验目录 | micro F1 | macro F1 | Precision | Recall | 无效实体条目 | 指标文件 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    # 按排序后的顺序逐行输出实验。
    for rank, row in enumerate(rows, start=1):
        # 添加当前实验结果表格行。
        lines.append(
            f"| {rank} | {row['experiment']} | {row['micro_f1']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['invalid_entities']} | `{row['metrics_file']}` |"
        )
    # 如果存在至少一个实验，标出第一名供用户后续测试使用。
    if rows:
        # 添加最优实验结论。
        lines.extend(
            [
                "",
                f"当前验证集最优实验：**{rows[0]['experiment']}**，micro F1 = **{rows[0]['micro_f1']:.4f}**。",
                "",
            ]
        )
    # 返回完整 Markdown 文本。
    return "\n".join(lines)


def main() -> None:
    """读取指标文件、按验证集 micro F1 排序并写出结果。"""

    # 读取用户参数。
    args = parse_args()
    # 解析每个实验的指标文件。
    rows = [read_metric_file(file_path) for file_path in args.metric_files]
    # 先按 micro F1、再按 macro F1 从高到低排列。
    rows.sort(key=lambda row: (row["micro_f1"], row["macro_f1"]), reverse=True)
    # 创建排行榜输出父目录。
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    # 写出可读的实验排行榜。
    args.output_file.write_text(build_markdown(rows), encoding="utf-8")
    # 在终端打印报告位置。
    print(f"验证集排行榜已保存到：{args.output_file}")
    # 显示当前最佳实验，便于下一步只对它运行一次测试集评价。
    if rows:
        # 打印最优指标摘要。
        print(f"最优实验：{rows[0]['experiment']}，validation micro F1={rows[0]['micro_f1']:.4f}")


# 只有直接运行本文件时才读取和比较实验。
if __name__ == "__main__":
    # 启动排行榜生成流程。
    main()
