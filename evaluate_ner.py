"""加载训练好的 LoRA/QLoRA adapter，并打印完整实体词级 F1 报告。"""

# 导入未来注解功能，使代码中的类型标注意义更明确。
from __future__ import annotations

# 导入命令行参数模块，使用户能够自由选择模型和测试规模。
import argparse
# 导入 JSON 模块，用于保存结构化指标结果。
import json
# 导入 Path，便于检查模型和数据文件位置。
from pathlib import Path
# 导入 Any，为配置字典提供清晰类型说明。
from typing import Any

# 导入 YAML，用来读取与训练一致的实验配置。
import yaml

# 导入本任务的数据读取、预测清洗和词级评分工具。
from ner_utils import (
    bootstrap_micro_f1_interval,
    compute_entity_f1,
    format_metric_report,
    normalize_predicted_entities,
    parse_model_json_with_status,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    """读取测试阶段常用的参数。"""

    # 创建参数解析器并说明本脚本计算的是完整实体 F1。
    parser = argparse.ArgumentParser(description="生成预测并计算中医药 NER 的词级实体 F1。")
    # 默认读取 QLoRA 配置；使用 LoRA 时传入 configs/lora.yaml 即可。
    parser.add_argument(
        "--config", type=Path, default=Path("configs/qlora.yaml"), help="训练时使用的 YAML 配置。"
    )
    # 允许直接指定 adapter 目录，否则采用配置里的输出目录。
    parser.add_argument("--adapter_path", type=Path, default=None, help="训练完成的 PEFT adapter 目录。")
    # 允许测试另一个基础模型，适合替换模型后的实验。
    parser.add_argument("--model_name_or_path", type=str, default=None, help="覆盖基座模型路径。")
    # 允许覆盖量化方式，必须与 adapter 对应的推理方案相符。
    parser.add_argument("--method", choices=["lora", "qlora"], default=None, help="覆盖 lora/qlora 方法。")
    # 允许测试自定义数据文件，例如先在 validation 上快速调试。
    parser.add_argument("--data_file", type=Path, default=None, help="覆盖默认测试 JSONL 路径。")
    # 控制回答最大新 token 数，实体列表较长时可调大。
    parser.add_argument("--max_new_tokens", type=int, default=None, help="覆盖每个回答允许生成的最大 token 数。")
    # 允许批量生成以提高完整测试集评价速度。
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖推理批大小；显存不足时设置为 1。")
    # 允许显式开启或关闭一次 JSON 格式纠错重试。
    parser.add_argument(
        "--retry_invalid_output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="第一次回答格式无效时是否自动重试一次。",
    )
    # 允许覆盖置信区间的重采样次数；设置为 0 可关闭。
    parser.add_argument(
        "--bootstrap_resamples",
        type=int,
        default=None,
        help="覆盖 micro F1 bootstrap 重采样次数；0 表示不计算区间。",
    )
    # 可只评价前若干样本进行冒烟测试，留空则评价整个测试集。
    parser.add_argument("--max_samples", type=int, default=None, help="仅评价前 N 条样本。")
    # 指定保存每条模型原始回答和清洗后实体的位置。
    parser.add_argument(
        "--prediction_file",
        type=Path,
        default=Path("outputs/test_predictions.jsonl"),
        help="预测明细输出文件。",
    )
    # 指定结构化 F1 报告保存位置。
    parser.add_argument(
        "--metrics_file",
        type=Path,
        default=Path("outputs/test_metrics.json"),
        help="指标 JSON 输出文件。",
    )
    # 返回用户选择的测试参数。
    return parser.parse_args()


def load_yaml(config_path: Path) -> dict[str, Any]:
    """加载训练阶段 YAML 配置。"""

    # 配置不存在时不能知道测试文件和基础模型，因此提前报错。
    if not config_path.exists():
        # 指出缺少的路径，帮助用户改正命令。
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    # 打开 YAML 配置文件。
    with config_path.open("r", encoding="utf-8") as reader:
        # 安全地把 YAML 内容解析为字典。
        return yaml.safe_load(reader)


def choose_inference_dtype(torch: Any) -> Any:
    """为推理选择在当前机器上可工作的浮点计算类型。"""

    # 如果 GPU 支持 BF16，就优先用它获得较稳定的生成计算。
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        # 返回 BF16 类型。
        return torch.bfloat16
    # 如果有 CUDA GPU 但不支持 BF16，则使用显存友好的 FP16。
    if torch.cuda.is_available():
        # 返回 FP16 类型。
        return torch.float16
    # 没有 CUDA 时使用 CPU 支持良好的 FP32，仅适合小规模调试。
    return torch.float32


def build_retry_messages(record: dict[str, Any], invalid_response: str) -> list[dict[str, str]]:
    """把错误回答作为上下文，请模型仅重写为协议要求的 JSON 数组。"""

    # 复制训练时相同的系统和用户消息。
    messages = list(record["prompt"])
    # 将模型首次错误回答加入对话历史，让它知道需要修正的内容。
    messages.append({"role": "assistant", "content": invalid_response})
    # 追加简短纠错要求，重申只允许输出 JSON 和原文一致的跨度。
    messages.append(
        {
            "role": "user",
            "content": (
                "上一个回答未通过格式或跨度校验。请重新检查原文，"
                "仅输出合法 JSON 数组；每项必须满足 text[start:end] 等于 entity，"
                "不要输出解释文字。"
            ),
        }
    )
    # 返回可再次应用聊天模板的消息列表。
    return messages


def main() -> None:
    """逐条生成实体结果、进行合法性校验并打印 F1 表格。"""

    # 读取命令行参数。
    args = parse_args()
    # 加载训练时的配置。
    config = load_yaml(args.config)
    # 在需要加载模型时再导入深度学习依赖。
    import torch
    # PEFT 用来把已经训练好的 adapter 装回基础模型。
    from peft import PeftModel
    # Transformers 提供模型加载、分词和 QLoRA 推理量化支持。
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # 优先使用命令行覆盖值，否则沿用训练配置中的基座模型。
    model_name = args.model_name_or_path or config["model_name_or_path"]
    # 优先使用命令行覆盖值，否则沿用训练配置中的微调方法。
    method = (args.method or config["method"]).lower()
    # 读取配置文件中用于推理和评价的默认设置。
    generation_settings = config.get("generation", {})
    # 命令行指定时优先采用命令行，否则使用配置或保守默认值。
    max_new_tokens = args.max_new_tokens or int(generation_settings.get("max_new_tokens", 256))
    # 推理批大小影响速度和显存占用。
    batch_size = args.batch_size or int(generation_settings.get("batch_size", 1))
    # 非法 JSON 自动重试的默认行为可在 YAML 中清楚设置。
    retry_invalid_output = (
        args.retry_invalid_output
        if args.retry_invalid_output is not None
        else bool(generation_settings.get("retry_invalid_output", False))
    )
    # 获取 bootstrap 次数；零代表不额外估计置信区间。
    bootstrap_resamples = (
        args.bootstrap_resamples
        if args.bootstrap_resamples is not None
        else int(generation_settings.get("bootstrap_resamples", 0))
    )
    # 批大小必须为正整数。
    if batch_size <= 0:
        # 提前报错而不是在 tokenizer 处出现难懂异常。
        raise ValueError("batch_size 必须大于 0。")
    # 根据配置输出目录找到 adapter，除非用户明确传入其他位置。
    adapter_path = args.adapter_path or Path(config["training"]["output_dir"])
    # 根据配置测试文件找到测试数据，除非用户指定其他 JSONL。
    data_file = args.data_file or Path(config["data"]["test_file"])
    # adapter 目录不存在通常说明尚未完成训练。
    if not adapter_path.exists():
        # 提示用户先训练或提供正确目录。
        raise FileNotFoundError(f"找不到 adapter：{adapter_path}，请先完成训练。")
    # 测试数据不存在通常说明还没有运行预处理脚本。
    if not data_file.exists():
        # 提示用户生成 JSONL 数据。
        raise FileNotFoundError(f"找不到测试数据：{data_file}，请先运行 prepare_sft_data.py。")
    # 读取测试 JSONL 中的文本、金标准实体和训练相同的 prompt。
    records = read_jsonl(data_file)
    # 如果用户设置了冒烟测试数量，只保留最前面的样本。
    if args.max_samples is not None:
        # 使用切片得到限定数量的测试记录。
        records = records[: args.max_samples]
    # 如果测试集为空，不能计算任何有意义的指标。
    if not records:
        # 明确报错，让用户检查数据路径或 --max_samples。
        raise ValueError("待评价的数据集为空。")

    # 推理时优先从 adapter 目录读取训练阶段保存的 tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True)
    # 若 tokenizer 没有 padding 符号，则复用结束符号。
    if tokenizer.pad_token is None:
        # 保证 generate 能为批次设置合法 pad_token_id。
        tokenizer.pad_token = tokenizer.eos_token
    # 批量生成的因果语言模型应从左侧补齐，使每条新答案从同一列开始。
    tokenizer.padding_side = "left"
    # 选择当前硬件可支持的推理精度。
    model_dtype = choose_inference_dtype(torch)
    # 默认不启用量化，普通 LoRA 会完整加载基础模型。
    quantization_config = None
    # QLoRA adapter 通常配合 4-bit 基础模型推理以保持低显存占用。
    if method == "qlora":
        # 使用与训练一致的 NF4 和双重量化参数。
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_dtype,
        )
    # 加载基础模型；推理阶段 device_map="auto" 会把模型放到可用设备。
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    # 将训练得到的 adapter 加载到对应基础模型之上。
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    # 切换到推理模式，关闭 dropout 等训练行为。
    model.eval()
    # 取出模型首个参数所在设备，将输入张量放到同一设备。
    input_device = next(model.parameters()).device
    # Qwen 聊天模板使用 <|im_end|> 结束一轮回答。
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # 首先保留 tokenizer 自带的通用句末 token。
    eos_token_ids = [tokenizer.eos_token_id]
    # 如果 <|im_end|> 是有效且不同的 token，就额外加入停止列表。
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_token_ids:
        # 这样模型输出完 assistant 回答后即可停止生成。
        eos_token_ids.append(im_end_id)

    # 准备逐句金标准实体列表。
    gold_batches: list[list[dict[str, Any]]] = []
    # 准备逐句有效预测实体列表。
    predicted_batches: list[list[dict[str, Any]]] = []
    # 准备包含原始模型回答的预测明细，方便错误分析。
    prediction_details: list[dict[str, Any]] = []
    # 记录实体字段或位置不合法的预测条目数。
    invalid_entity_count = 0
    # 记录首次生成未通过 JSON 协议或实体字段校验的样本数量。
    initial_invalid_sample_count = 0
    # 记录实际执行过格式重试的样本数量。
    retry_sample_count = 0
    # 记录经过重试后成功消除格式/跨度错误的样本数量。
    rescued_sample_count = 0

    # 定义批量生成函数，使首次回答和格式纠错回答共享完全相同的推理逻辑。
    def generate_batch(message_batches: list[list[dict[str, str]]]) -> list[str]:
        """对若干条聊天消息批量生成确定性答案文本。"""

        # 将每条聊天消息转换为 Qwen 能识别的完整模板字符串。
        prompt_texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in message_batches
        ]
        # 批量编码提示词；左填充可让生成的答案起点对齐。
        model_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
        # 把所有输入张量移动到模型入口所在设备。
        model_inputs = {name: value.to(input_device) for name, value in model_inputs.items()}
        # 填充后的统一输入宽度就是生成答案 token 开始前的位置。
        prompt_width = model_inputs["input_ids"].shape[1]
        # 推理阶段不计算梯度，降低显存占用。
        with torch.inference_mode():
            # 使用贪心解码，保持评价可重复。
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        # 只截取每条样本中新产生的回答 token。
        generated_answer_ids = generated_ids[:, prompt_width:]
        # 将整批回答 token 解码为 JSON 字符串候选。
        return [
            response.strip()
            for response in tokenizer.batch_decode(generated_answer_ids, skip_special_tokens=True)
        ]

    # 按配置的批大小遍历测试或验证记录。
    for batch_start in range(0, len(records), batch_size):
        # 取出当前批次记录。
        batch_records = records[batch_start : batch_start + batch_size]
        # 使用每条记录原始 prompt 批量生成第一次回答。
        batch_responses = generate_batch([record["prompt"] for record in batch_records])
        # 将当前批次每条记录与其回答配对处理。
        for record, original_response in zip(batch_records, batch_responses):
            # 校验第一次回答中的实体字段和跨度。
            predicted_entities, invalid_count = normalize_predicted_entities(
                record["text"], original_response
            )
            # 判断第一次回答是否至少满足 JSON 数组协议。
            _, valid_json_array = parse_model_json_with_status(original_response)
            # JSON 失败或存在无效实体字段均属于可尝试纠正的输出失败。
            needs_retry = not valid_json_array or invalid_count > 0
            # 保存最后将用于评分的回答，默认先采用第一次答案。
            final_response = original_response
            # 保存是否执行了自动纠错重试。
            retry_used = False
            # 首次失败时先计数，便于判断结构化生成优化是否有效。
            if needs_retry:
                # 累计首次无效样本数量。
                initial_invalid_sample_count += 1
            # 配置允许且首次回答无效时，再询问模型一次。
            if retry_invalid_output and needs_retry:
                # 记录本样本实际调用了第二次生成。
                retry_used = True
                # 累计重试样本数量。
                retry_sample_count += 1
                # 使用包含错误回答和纠错请求的新消息生成一次修正答案。
                corrected_response = generate_batch(
                    [build_retry_messages(record, original_response)]
                )[0]
                # 校验纠正答案中的实体。
                corrected_entities, corrected_invalid_count = normalize_predicted_entities(
                    record["text"], corrected_response
                )
                # 检查纠正答案 JSON 协议是否合法。
                _, corrected_valid_json = parse_model_json_with_status(corrected_response)
                # 只在纠正答案格式合法且无效条目更少时采用它。
                if corrected_valid_json and (
                    not valid_json_array or corrected_invalid_count < invalid_count
                ):
                    # 更新用于评分的原始回答文本。
                    final_response = corrected_response
                    # 更新用于评分的实体列表。
                    predicted_entities = corrected_entities
                    # 更新用于报告的无效条目数量。
                    invalid_count = corrected_invalid_count
                    # 纠正后完全不存在无效条目时统计一次成功救回。
                    if corrected_invalid_count == 0:
                        # 累计格式纠正成功样本数。
                        rescued_sample_count += 1
            # 累加最后采用的回答中无效实体数量。
            invalid_entity_count += invalid_count
            # 保存当前句的人工金标准实体。
            gold_batches.append(record["gold_entities"])
            # 保存当前句经过最终校验的预测实体。
            predicted_batches.append(predicted_entities)
            # 保存可供人工复盘的预测明细。
            prediction_details.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "gold_entities": record["gold_entities"],
                    "initial_response": original_response,
                    "final_response": final_response,
                    "retry_used": retry_used,
                    "predicted_entities": predicted_entities,
                    "invalid_entity_count": invalid_count,
                }
            )
        # 计算当前批次完成后的累计样本数。
        completed_count = min(batch_start + batch_size, len(records))
        # 每完成至少 50 条或者到最后一批时显示进度。
        if completed_count % 50 < batch_size or completed_count == len(records):
            # 输出已完成数量，便于长时间测试时观察运行状态。
            print(f"已生成并校验 {completed_count}/{len(records)} 条样本。")

    # 使用完整实体词而非单字标签计算各类别 F1。
    metrics = compute_entity_f1(gold_batches, predicted_batches)
    # 如果配置了正数重采样次数，则为总体 micro F1 计算 95% 区间。
    if bootstrap_resamples > 0:
        # 保存置信区间结果到结构化指标中。
        metrics["bootstrap_micro_f1_95_ci"] = bootstrap_micro_f1_interval(
            gold_batches,
            predicted_batches,
            num_resamples=bootstrap_resamples,
            seed=int(config["training"].get("seed", 42)),
        )
    # 将生成格式质量汇总进指标文件，便于比较纠错开关效果。
    metrics["generation_quality"] = {
        "batch_size": batch_size,
        "retry_invalid_output": retry_invalid_output,
        "initial_invalid_sample_count": initial_invalid_sample_count,
        "retry_sample_count": retry_sample_count,
        "rescued_sample_count": rescued_sample_count,
        "final_invalid_entity_count": invalid_entity_count,
    }
    # 创建预测明细输出文件所在目录。
    args.prediction_file.parent.mkdir(parents=True, exist_ok=True)
    # 保存每句话的模型回答与清洗后实体，便于误差分析。
    write_jsonl(prediction_details, args.prediction_file)
    # 创建指标文件所在目录。
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    # 打开结构化指标文件。
    with args.metrics_file.open("w", encoding="utf-8") as writer:
        # 保留中文类别名称并缩进输出，方便打开阅读。
        json.dump(metrics, writer, ensure_ascii=False, indent=2)
    # 打印十种类别及总体的完整实体级评价结果。
    print("\n词级实体严格匹配 F1 报告（类别、start、end、entity 必须全部正确）")
    # 打印格式化后的表格。
    print(format_metric_report(metrics))
    # 打印被丢弃的无效模型实体数量，辅助诊断输出格式问题。
    print(f"\n无效或重复的模型实体条目数：{invalid_entity_count}")
    # 打印结构化输出自动纠错效果概览。
    print(
        f"首次输出无效样本：{initial_invalid_sample_count}；"
        f"触发重试：{retry_sample_count}；重试后完全修复：{rescued_sample_count}"
    )
    # 打印预测文件位置。
    print(f"预测明细已保存到：{args.prediction_file}")
    # 打印指标文件位置。
    print(f"结构化指标已保存到：{args.metrics_file}")


# 只有从终端直接运行这个文件时才启动生成和评价流程。
if __name__ == "__main__":
    # 执行评价主函数。
    main()
