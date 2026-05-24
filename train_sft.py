"""使用 Qwen2.5-7B-Instruct 与 LoRA/QLoRA 完成中医药 NER 的 SFT 训练。"""

# 导入未来注解功能，使类型标注更易读。
from __future__ import annotations

# 导入命令行参数模块，让模型和微调方式都可以被覆盖。
import argparse
# 导入日期时间工具，为每次实验清单记录运行时间。
from datetime import datetime, timezone
# 导入哈希模块，为训练和验证数据记录可复现指纹。
import hashlib
# 导入 JSON 模块，将实验清单保存为结构化文件。
import json
# 导入包版本读取工具，记录关键训练库版本。
from importlib.metadata import PackageNotFoundError, version
# 导入 Path，便于跨操作系统组织配置和输出目录。
from pathlib import Path
# 导入 Any，说明配置字典中可以容纳多种值类型。
from typing import Any

# 导入 YAML，用于加载初学者容易修改的实验配置。
import yaml


def parse_args() -> argparse.Namespace:
    """读取训练脚本支持的命令行参数。"""

    # 创建命令行解析器并说明脚本用途。
    parser = argparse.ArgumentParser(description="使用 LoRA 或 QLoRA 对 Qwen2.5-7B 做中医药 NER SFT。")
    # 配置文件给出完整默认实验设置。
    parser.add_argument(
        "--config", type=Path, default=Path("configs/qlora.yaml"), help="YAML 实验配置路径。"
    )
    # 允许临时替换基座模型，而无需复制或编辑配置文件。
    parser.add_argument(
        "--model_name_or_path", type=str, default=None, help="覆盖配置中的基座模型名称或本地路径。"
    )
    # 允许一条命令切换 LoRA 与 QLoRA。
    parser.add_argument(
        "--method", choices=["lora", "qlora"], default=None, help="覆盖配置中的微调方式。"
    )
    # 允许临时指定输出目录，方便保留多次实验结果。
    parser.add_argument("--output_dir", type=Path, default=None, help="覆盖 adapter 输出目录。")
    # 返回已经解析好的参数。
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """读取 YAML 配置并确认它存在。"""

    # 如果用户提供的配置路径不存在，立即给出清楚提示。
    if not config_path.exists():
        # 通过异常中断训练，避免使用错误默认设置。
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    # 使用 UTF-8 打开中文注释配置文件。
    with config_path.open("r", encoding="utf-8") as reader:
        # 安全加载 YAML 内容为 Python 字典。
        config = yaml.safe_load(reader)
    # 返回可供脚本使用的完整配置。
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """把少量命令行覆盖项合并进 YAML 配置。"""

    # 如果用户传入了新模型路径，就覆盖 YAML 中的模型路径。
    if args.model_name_or_path is not None:
        # 这一设置实现“不改代码即可替换模型”。
        config["model_name_or_path"] = args.model_name_or_path
    # 如果用户明确指定 LoRA 或 QLoRA，就覆盖 YAML 中的方式。
    if args.method is not None:
        # 当覆盖项与配置文件原方法不同，提醒用户核对方法相关超参数。
        if args.method != config["method"]:
            # 批大小、优化器与目标层不会自动改写，使用成套 YAML 最易复现。
            print("提示：--method 已覆盖配置方法；请同时核对 lora.target_modules、optim 与批大小。")
        # 保存新的训练方式。
        config["method"] = args.method
    # 如果用户指定了输出目录，就覆盖默认实验目录。
    if args.output_dir is not None:
        # Path 转为字符串，使 YAML 序列化和 Trainer 都容易处理。
        config["training"]["output_dir"] = str(args.output_dir)
    # 返回已经应用覆盖项的配置。
    return config


def choose_precision(torch: Any, requested_precision: str) -> tuple[Any, bool, bool]:
    """根据 GPU 能力选择模型数据类型和 Trainer 混合精度开关。"""

    # 检查当前环境是否存在 CUDA GPU。
    cuda_available = torch.cuda.is_available()
    # 检查显卡是否原生支持 BF16；较新的数据中心卡和游戏卡通常支持。
    bf16_available = cuda_available and torch.cuda.is_bf16_supported()
    # 用户希望使用 BF16 且设备支持时，优先选用数值范围更稳健的 BF16。
    if requested_precision == "bf16" and bf16_available:
        # 返回模型类型、bf16 开关和 fp16 开关。
        return torch.bfloat16, True, False
    # 只要存在 CUDA GPU，无法使用 BF16 时就回退到常见的 FP16。
    if cuda_available:
        # 返回 FP16 设置，以降低 7B 模型显存占用。
        return torch.float16, False, True
    # CPU 教学检查可以使用 FP32，但实际训练 7B 模型会非常慢。
    print("未检测到 CUDA GPU：将使用 float32。正式训练 Qwen2.5-7B 建议使用 NVIDIA GPU。")
    # 返回 CPU 能运行的默认精度设置。
    return torch.float32, False, False


def file_sha256(file_path: str | Path) -> str:
    """计算实验输入文件的 SHA256，以确认不同运行使用了同一份数据。"""

    # 统一将字符串路径转换为 Path。
    path = Path(file_path)
    # 创建 SHA256 哈希对象。
    digest = hashlib.sha256()
    # 以二进制打开数据文件，避免文本换行规则影响指纹。
    with path.open("rb") as reader:
        # 分块读取文件，适配更大的训练数据。
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            # 将当前数据块加入哈希计算。
            digest.update(chunk)
    # 返回可以记录到实验报告中的十六进制指纹。
    return digest.hexdigest()


def package_version(package_name: str) -> str:
    """读取依赖版本；若未安装则用明确字符串表示。"""

    # 尝试从 Python 环境元数据读取指定包版本。
    try:
        # 返回已安装版本号。
        return version(package_name)
    # 当前环境没有安装该库时捕获查询异常。
    except PackageNotFoundError:
        # 返回可读提示，避免仅因记录版本而中断训练。
        return "not-installed"


def save_experiment_manifest(config: dict[str, Any], output_dir: Path) -> None:
    """保存训练配置、数据指纹和依赖版本，支持后续实验复现和比较。"""

    # 准备记录本实验输入数据文件的位置与指纹。
    data_manifest: dict[str, dict[str, str]] = {}
    # 只对训练阶段会读取的数据文件建立指纹。
    for split in ["train_file", "validation_file", "test_file"]:
        # 读取配置中的路径。
        data_path = Path(config["data"][split])
        # 文件不存在时立即提示用户先准备数据。
        if not data_path.exists():
            # 缺少数据不能开始可信训练。
            raise FileNotFoundError(f"配置中的数据文件不存在：{data_path}")
        # 保存路径与 SHA256。
        data_manifest[split] = {"path": str(data_path), "sha256": file_sha256(data_path)}
    # 组织本次实验最重要的可追踪元数据。
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config["experiment_name"],
        "model_name_or_path": config["model_name_or_path"],
        "method": config["method"],
        "data": data_manifest,
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "peft": package_version("peft"),
            "trl": package_version("trl"),
            "bitsandbytes": package_version("bitsandbytes"),
        },
    }
    # 将实验清单保存到本次输出目录。
    with (output_dir / "experiment_manifest.json").open("w", encoding="utf-8") as writer:
        # 保留中文并添加缩进，便于人工对照两次实验。
        json.dump(manifest, writer, ensure_ascii=False, indent=2)


def main() -> None:
    """加载数据、建立 PEFT 模型并启动监督微调。"""

    # 读取用户输入的配置路径和覆盖项。
    args = parse_args()
    # 加载 YAML 默认配置。
    config = load_config(args.config)
    # 应用可选命令行覆盖项。
    config = apply_overrides(config, args)

    # 在真正训练时再导入深度学习依赖，使 --help 和配置排查更加轻便。
    import torch
    # Datasets 根据 JSONL 文件生成训练与验证数据集对象。
    from datasets import load_dataset
    # PEFT 提供 LoRA 配置和 QLoRA 模型预处理工具。
    from peft import LoraConfig, prepare_model_for_kbit_training
    # Transformers 提供基座模型、分词器和 4-bit 量化配置。
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, set_seed
    # TRL 为对话式监督微调提供专用配置和 Trainer。
    from trl import SFTConfig, SFTTrainer

    # 读取随机种子，使相同配置的实验更容易复现。
    seed = int(config["training"]["seed"])
    # 为 Transformers、PyTorch 等组件同时设置随机种子。
    set_seed(seed)
    # 读取配置中选择的微调方法。
    method = config["method"].lower()
    # 只允许教学工程已经实现的两种 PEFT 方式。
    if method not in {"lora", "qlora"}:
        # 如果配置拼写错误，就给出清楚错误而不是静默运行。
        raise ValueError("method 必须是 lora 或 qlora。")
    # 读取模型名称或本地模型目录。
    model_name = config["model_name_or_path"]
    # 读取训练超参数部分，后续模型加载和 Trainer 都会使用它。
    training = config["training"]
    # 根据硬件选择 BF16、FP16 或 CPU 用的 FP32。
    model_dtype, use_bf16, use_fp16 = choose_precision(
        torch, config["training"].get("precision", "bf16")
    )
    # 打印本次实验最关键的选择，便于核对日志。
    print(f"模型={model_name}  方法={method}  dtype={model_dtype}  seed={seed}")

    # 加载与基座模型配套的 tokenizer，并保留模型自己的聊天模板。
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        use_fast=True,
    )
    # 某些生成模型没有独立 pad token，此时用结束 token 进行批量补齐。
    if tokenizer.pad_token is None:
        # 复用 eos token 可以避免额外修改词表大小。
        tokenizer.pad_token = tokenizer.eos_token
    # 训练因果语言模型时通常从右侧补齐序列。
    tokenizer.padding_side = "right"

    # QLoRA 需要先定义 4-bit 基座模型的量化方式。
    quantization_config = None
    # 仅当用户选择 QLoRA 时启用 4-bit 权重量化。
    if method == "qlora":
        # 使用论文与官方文档常见的 NF4、双重量化组合减少显存。
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_dtype,
        )

    # 按选择的精度和量化配置加载 Qwen 基座模型。
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        quantization_config=quantization_config,
        attn_implementation=training.get("attn_implementation", "sdpa"),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    # 训练时关闭 KV cache，因为梯度检查点与 cache 不配合。
    model.config.use_cache = False
    # 只有量化后的 QLoRA 模型需要做 k-bit 训练前准备。
    if method == "qlora":
        # 该步骤会正确处理量化层梯度与输入嵌入，使 LoRA 能被训练。
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
        )

    # 读取配置中的 LoRA 超参数。
    lora_settings = config["lora"]
    # 构建适用于因果语言模型的低秩适配器配置。
    peft_config = LoraConfig(
        r=int(lora_settings["r"]),
        lora_alpha=int(lora_settings["alpha"]),
        lora_dropout=float(lora_settings["dropout"]),
        target_modules=lora_settings["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 指定训练集与验证集 JSONL 文件。
    data_files = {
        "train": config["data"]["train_file"],
        "validation": config["data"]["validation_file"],
    }
    # 加载 JSONL 数据；每条数据含 prompt、completion 与人工检查信息。
    raw_datasets = load_dataset("json", data_files=data_files)
    # 训练器只需要对话式输入与答案两列，其他字段留给测试脚本使用。
    train_dataset = raw_datasets["train"].select_columns(["prompt", "completion"])
    # 验证数据同样仅保留监督微调所需的两列。
    validation_dataset = raw_datasets["validation"].select_columns(["prompt", "completion"])
    # 读取输出目录并确保它已经创建。
    output_dir = Path(config["training"]["output_dir"])
    # 创建日志和 adapter 所在文件夹。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 将实际生效的完整配置保存到输出目录，便于复现实验。
    with (output_dir / "effective_config.yaml").open("w", encoding="utf-8") as writer:
        # 写出命令行覆盖后的配置，并允许中文信息被直接阅读。
        yaml.safe_dump(config, writer, allow_unicode=True, sort_keys=False)
    # 写入本次使用的数据指纹和包版本，方便之后判断对比是否公平。
    save_experiment_manifest(config, output_dir)
    # 用 TRL 的 SFT 配置定义训练循环行为。
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=config["experiment_name"],
        max_length=int(training["max_length"]),
        num_train_epochs=float(training["num_train_epochs"]),
        learning_rate=float(training["learning_rate"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(training["eval_steps"]),
        save_strategy="steps",
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type=training["lr_scheduler_type"],
        weight_decay=float(training["weight_decay"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_fp16,
        optim=training["optim"],
        report_to=training["report_to"],
        seed=seed,
        group_by_length=bool(training.get("group_by_length", False)),
        packing=bool(training["packing"]),
        completion_only_loss=True,
        eos_token="<|im_end|>",
    )

    # 准备可选的训练回调列表。
    callbacks = []
    # 读取早停容忍次数；零或空值表示不启用早停。
    early_stopping_patience = training.get("early_stopping_patience")
    # 如果配置了正数，就在验证 loss 多次不改善时提前停止训练。
    if early_stopping_patience is not None and int(early_stopping_patience) > 0:
        # 注册 Hugging Face 官方早停回调。
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping_patience),
                early_stopping_threshold=float(training.get("early_stopping_threshold", 0.0)),
            )
        )
    # 创建专为生成模型监督微调设计的 Trainer。
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    # 打印可训练参数占比，直观展示 LoRA/QLoRA 为什么节省训练成本。
    trainer.model.print_trainable_parameters()
    # 读取可选断点目录；None 表示全新训练。
    resume_checkpoint = training.get("resume_from_checkpoint")
    # 启动训练，并在配置指定时从已有检查点继续。
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    # 保存最终或验证集最优的 LoRA adapter 权重。
    trainer.save_model(str(output_dir))
    # 保存 tokenizer，确保之后推理使用相同聊天模板和特殊 token。
    tokenizer.save_pretrained(str(output_dir))
    # 把训练阶段的 loss、速度等指标记录到日志。
    trainer.log_metrics("train", train_result.metrics)
    # 把训练指标保存成 JSON 文件。
    trainer.save_metrics("train", train_result.metrics)
    # 保存 Trainer 状态，支持继续训练和复现实验过程。
    trainer.save_state()
    # 提示用户下一步应使用测试脚本获得词级分类 F1。
    print(f"训练完成，adapter 已保存到 {output_dir}。请运行 evaluate_ner.py 生成词级 F1 报告。")


# 只有从终端直接执行该文件时才进入训练入口。
if __name__ == "__main__":
    # 启动训练主流程。
    main()
