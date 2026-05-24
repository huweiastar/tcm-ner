"""把原始中医药 BIO 数据转换为可用于大模型 SFT 的对话式 JSONL 数据。"""

# 导入命令行参数模块，让用户可以修改输入输出文件夹。
import argparse
# 导入 JSON 模块，将数据统计摘要写成易读文件。
import json
# 导入 Counter，用来统计十种实体各有多少个完整词。
from collections import Counter
# 导入 Path，安全处理项目内文件路径。
from pathlib import Path

# 从共用工具文件中导入实体类别、转换函数与 JSONL 写入函数。
from ner_utils import ENTITY_TYPES, make_sft_record, read_bio_sentences, write_jsonl


def parse_args() -> argparse.Namespace:
    """读取用户在命令行提供的数据路径参数。"""

    # 创建参数解析器，并写清楚脚本的学习目标。
    parser = argparse.ArgumentParser(description="将字符级 BIO 标注转换成对话式 SFT JSONL 数据。")
    # 允许用户指定原始 data 文件夹，默认就是本作业给出的目录。
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="原始 BIO 数据目录。")
    # 允许用户指定预处理结果目录，以便保留不同实验版本。
    parser.add_argument(
        "--output_dir", type=Path, default=Path("processed_data"), help="输出 JSONL 数据目录。"
    )
    # 返回解析完成的命令行参数对象。
    return parser.parse_args()


def main() -> None:
    """逐个划分转换数据，并打印便于检查的统计信息。"""

    # 读取命令行参数。
    args = parse_args()
    # 建立原始文件名与训练框架常用 split 名之间的映射。
    split_files = {
        "train": args.data_dir / "medical.train",
        "validation": args.data_dir / "medical.dev",
        "test": args.data_dir / "medical.test",
    }
    # 准备整个数据集的统计摘要字典。
    summary: dict[str, dict[str, object]] = {}

    # 按训练集、验证集、测试集顺序分别处理文件。
    for split, source_path in split_files.items():
        # 如果任一输入文件不存在，就立即指出具体缺少哪个文件。
        if not source_path.exists():
            # 明确报错比训练到一半才失败更容易排查。
            raise FileNotFoundError(f"找不到原始数据文件：{source_path}")
        # 读取一份 BIO 文件，并按空行切分为句子。
        sentences = read_bio_sentences(source_path)
        # 将每个句子转换为 prompt/completion SFT 记录。
        records = [
            make_sft_record(sentence, split=split, sample_index=index)
            for index, sentence in enumerate(sentences, start=1)
        ]
        # 指定当前数据划分输出为一行一条记录的 JSONL 文件。
        output_path = args.output_dir / f"{split}.jsonl"
        # 把已经转换的记录保存到磁盘。
        write_jsonl(records, output_path)

        # 初始化实体数量计数器。
        entity_counter: Counter[str] = Counter()
        # 遍历当前划分中的每条 SFT 记录。
        for record in records:
            # 遍历一条样本中恢复出的完整实体词。
            for entity in record["gold_entities"]:
                # 按实体类别累加完整词数量，而不是统计单个字符。
                entity_counter[entity["type"]] += 1
        # 计算每条文本包含的字符数，方便确定训练最大长度。
        lengths = [len(record["text"]) for record in records]
        # 保存当前划分的样本数、文本长度和各类实体数量。
        summary[split] = {
            "source_file": str(source_path),
            "output_file": str(output_path),
            "num_samples": len(records),
            "max_text_characters": max(lengths) if lengths else 0,
            "average_text_characters": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "entity_counts": {entity_type: entity_counter[entity_type] for entity_type in ENTITY_TYPES},
        }
        # 向用户报告当前文件已经处理完成。
        print(f"{split:<10} 样本数={len(records):>5}  最大文本字符数={summary[split]['max_text_characters']:>3}  输出={output_path}")

    # 为统计摘要准备输出文件路径。
    summary_path = args.output_dir / "summary.json"
    # 打开摘要文件，并使用缩进保留中文以便人工阅读。
    with summary_path.open("w", encoding="utf-8") as writer:
        # 把统计结果写入 JSON 文件，便于实验报告直接引用。
        json.dump(summary, writer, ensure_ascii=False, indent=2)
    # 打印摘要保存位置，提醒用户可以检查转换质量。
    print(f"统计摘要已保存到：{summary_path}")


# 只有直接运行此脚本时才执行 main；被 Notebook 导入时不会自动处理数据。
if __name__ == "__main__":
    # 启动数据转换流程。
    main()
