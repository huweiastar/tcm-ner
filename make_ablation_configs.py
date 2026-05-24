"""根据优化版 QLoRA 配置生成一组适合初学者逐一运行的消融实验 YAML。"""

# 导入未来注解功能，使类型标注更加清楚。
from __future__ import annotations

# 导入参数解析模块，让用户更换模板配置或输出目录。
import argparse
# 导入深拷贝函数，确保修改一个变体不会污染其他变体。
from copy import deepcopy
# 导入 Path，以便定位 YAML 模板和新配置文件。
from pathlib import Path
# 导入 Any，表示 YAML 配置中包含字符串、数字和列表等字段。
from typing import Any

# 导入 YAML 模块，用于读取模板并写出实验配置。
import yaml


def parse_args() -> argparse.Namespace:
    """读取模板 YAML 和新配置保存目录。"""

    # 创建命令行解析器。
    parser = argparse.ArgumentParser(description="从优化版 QLoRA 模板生成 rank、学习率和目标层消融配置。")
    # 默认基于已经使用 clean/balanced 数据的优化版配置生成。
    parser.add_argument(
        "--base_config",
        type=Path,
        default=Path("configs/qlora_optimized.yaml"),
        help="消融实验的基础 YAML 配置。",
    )
    # 输出配置与正式配置分目录保存，方便查看。
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("configs/ablations"),
        help="生成的消融配置目录。",
    )
    # 返回解析结果。
    return parser.parse_args()


def load_yaml(file_path: Path) -> dict[str, Any]:
    """读取基础 YAML 配置。"""

    # 模板文件不存在时，不能生成可靠的实验变体。
    if not file_path.exists():
        # 提醒用户检查是否已保留优化版配置。
        raise FileNotFoundError(f"找不到基础配置：{file_path}")
    # 以 UTF-8 打开带中文注释语义的 YAML 文件。
    with file_path.open("r", encoding="utf-8") as reader:
        # 安全解析 YAML 字典。
        return yaml.safe_load(reader)


def make_variant(
    base_config: dict[str, Any],
    suffix: str,
    output_suffix: str,
    changes: dict[str, Any],
) -> dict[str, Any]:
    """复制基础配置并应用一个简单、可解释的超参数变体。"""

    # 深复制完整配置，防止嵌套 training/lora 字段被共享修改。
    config = deepcopy(base_config)
    # 将变体名称追加到实验名，以便日志区分。
    config["experiment_name"] = f"{base_config['experiment_name']}_{suffix}"
    # 为每个变体设置独立的 adapter 输出文件夹。
    config["training"]["output_dir"] = f"outputs/ablations/{output_suffix}"
    # 应用通过点号表达的少量配置变更。
    for dotted_key, value in changes.items():
        # 将例如 lora.r 拆成一级段名和二级字段名。
        section, key = dotted_key.split(".", maxsplit=1)
        # 写入本变体的新值。
        config[section][key] = value
    # 返回可保存的新实验配置。
    return config


def main() -> None:
    """生成小规模可执行消融配置，并打印建议训练顺序。"""

    # 读取命令行参数。
    args = parse_args()
    # 载入已验证的数据与训练设置模板。
    base_config = load_yaml(args.base_config)
    # 定义初学者可理解的五组单因素变化实验。
    variants = {
        "rank_r8": {"lora.r": 8, "lora.alpha": 16},
        "rank_r32": {"lora.r": 32, "lora.alpha": 64},
        "lr_1e-4": {"training.learning_rate": 0.0001},
        "lr_3e-4": {"training.learning_rate": 0.0003},
        "attention_only": {
            "lora.target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        },
    }
    # 创建配置输出目录。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 保存生成的文件路径用于最后打印。
    generated_files: list[Path] = []
    # 逐个变体生成 YAML 文件。
    for suffix, changes in variants.items():
        # 创建本变体配置。
        variant_config = make_variant(base_config, suffix, suffix, changes)
        # 指定本变体 YAML 文件路径。
        output_path = args.output_dir / f"qlora_optimized_{suffix}.yaml"
        # 打开输出配置文件。
        with output_path.open("w", encoding="utf-8") as writer:
            # 写出可被训练脚本直接读取的 YAML。
            yaml.safe_dump(variant_config, writer, allow_unicode=True, sort_keys=False)
        # 保存已生成文件位置。
        generated_files.append(output_path)
    # 打印成功生成的配置文件。
    print("已生成以下消融配置：")
    # 逐文件显示路径。
    for generated_file in generated_files:
        # 打印一行路径。
        print(f"- {generated_file}")
    # 给出下一步运行方式，使初学者不需要猜命令格式。
    print("\n任选一个配置运行：python train_sft.py --config <上方某个yaml路径>")
    # 提醒在验证集评价后再比较，而不是窥视测试集选择模型。
    print("训练后请先在 validation.jsonl 上生成 metrics，再用 compare_experiments.py 选择方案。")


# 只有用户直接执行文件时才生成配置。
if __name__ == "__main__":
    # 启动配置生成流程。
    main()
