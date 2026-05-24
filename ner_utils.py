"""中医药命名实体识别任务共用的、适合教学阅读的数据与评测工具。"""

# 导入未来注解功能，使类型标注在较旧的 Python 版本里也更友好。
from __future__ import annotations

# 导入 JSON 模块，用来保存训练样本和解析模型输出。
import json
# 导入随机数工具，用固定随机种子重复抽样以计算 F1 置信区间。
import random
# 导入正则表达式模块，用来从模型回答中找到 JSON 数组。
import re
# 导入计数器，用来汇总每种实体类别的数量和评分结果。
from collections import Counter
# 导入 Path，让文件路径在 Windows、Linux 和 macOS 上都可读。
from pathlib import Path
# 导入类型工具，让初学者清楚每个函数接收和返回什么数据。
from typing import Any, Iterable


# 固定数据集中出现的 10 个实体类别，并保持打印报告时的稳定顺序。
ENTITY_TYPES = [
    "中医治则",
    "中医治疗",
    "中医证候",
    "中医诊断",
    "中药",
    "临床表现",
    "其他治疗",
    "方剂",
    "西医治疗",
    "西医诊断",
]

# 给大模型的系统指令明确任务、允许类别和输出约束。
SYSTEM_PROMPT = """你是一名中医药命名实体识别助手。
请从用户提供的文本中抽取实体，只能使用以下类别：
中医治则、中医治疗、中医证候、中医诊断、中药、临床表现、其他治疗、方剂、西医治疗、西医诊断。
请仅输出 JSON 数组，不要解释。每个实体的格式为：
{"entity":"实体原文","type":"实体类别","start":起始字符位置,"end":结束字符位置}
其中 start 从 0 开始计数，end 为右开区间，即 text[start:end] 必须等于 entity。
没有实体时输出 []。"""


def build_user_prompt(text: str) -> str:
    """把一段待识别文本包装成清晰的用户问题。"""

    # 使用固定模板，使训练阶段和推理阶段看到完全一致的任务表达。
    return f"请识别下列文本中的中医药相关实体：\n{text}"


def read_bio_sentences(file_path: str | Path) -> list[dict[str, list[str]]]:
    """读取以空行分句、每行为“字符 BIO标签”的原始标注文件。"""

    # 把字符串路径转成 Path 对象，便于后续打开文件和报告错误。
    path = Path(file_path)
    # 准备列表，用来收集文件中的全部句子。
    sentences: list[dict[str, list[str]]] = []
    # 准备当前句子的字符列表。
    current_chars: list[str] = []
    # 准备当前句子的 BIO 标签列表。
    current_labels: list[str] = []

    # 以 UTF-8 读取文件；newline=None 会自动兼容数据里的 CRLF 换行。
    with path.open("r", encoding="utf-8-sig", newline=None) as reader:
        # 逐行读取，同时保留行号以便数据错误时准确定位。
        for line_number, raw_line in enumerate(reader, start=1):
            # 只去掉行尾换行，正文中的全角或半角空格也可能是被标注字符。
            line = raw_line.rstrip("\r\n")
            # 空行代表一个句子结束。
            if line == "":
                # 只有当前句子不为空时才保存，避免连续空行产生空样本。
                if current_chars:
                    # 保存一份字符与标签列表的副本。
                    sentences.append({"chars": current_chars, "labels": current_labels})
                    # 清空字符列表，为下一句话重新收集数据。
                    current_chars = []
                    # 清空标签列表，使它与字符列表重新同步。
                    current_labels = []
                # 当前空行已经处理完成，直接进入下一行。
                continue

            # 只把 ASCII 空格或制表符当作字段分隔符，从而保留正文里的全角空格。
            matched_columns = re.match(r"^(.*)[ \t]+(O|[BI]-.+)$", line)
            # 如果无法取得字符与标签，说明原始数据格式不符合 BIO 约定。
            if matched_columns is None:
                # 立即抛出错误，避免错误标注悄悄进入训练集。
                raise ValueError(f"{path}:{line_number} 不是“字符 标签”格式：{line!r}")
            # 取出当前字符，例如“黄”。
            char = matched_columns.group(1)
            # 取出当前标签，例如“B-中药”。
            label = matched_columns.group(2)
            # 判断标签是否属于 O、B-类别、I-类别这三类合法形式。
            valid_label = label == "O" or (
                label[:2] in {"B-", "I-"} and label[2:] in ENTITY_TYPES
            )
            # 如果标签未知，就在预处理阶段暴露问题，而不是让模型学习噪声。
            if not valid_label:
                # 错误信息包含文件名和行号，方便人工修正数据。
                raise ValueError(f"{path}:{line_number} 出现未知标签：{label!r}")
            # 把当前字符加入当前句子。
            current_chars.append(char)
            # 把当前标签加入当前句子，保持与字符位置一一对应。
            current_labels.append(label)

    # 某些文件末尾没有空行，因此还要保存最后一个未提交的句子。
    if current_chars:
        # 将最后一句追加到句子列表中。
        sentences.append({"chars": current_chars, "labels": current_labels})
    # 返回已经解析完成的全部句子。
    return sentences


def bio_to_entities(
    chars: list[str], labels: list[str], strict: bool = True
) -> list[dict[str, Any]]:
    """将字符级 BIO 标签还原为完整实体及其字符跨度。"""

    # 字符与标签必须一样长，否则无法确定实体所在位置。
    if len(chars) != len(labels):
        # 抛出错误防止得到位置错乱的监督答案。
        raise ValueError("字符数量与 BIO 标签数量不一致。")
    # 拼接字符得到原始文本，之后可使用 Python 切片取出完整实体。
    text = "".join(chars)
    # 准备最终实体列表。
    entities: list[dict[str, Any]] = []
    # 使用 None 表示当前还没有正在读取的实体。
    current_start: int | None = None
    # 使用 None 表示当前还没有实体类别。
    current_type: str | None = None

    # 定义小工具函数，用当前位置结束正在读取的实体。
    def close_entity(end: int) -> None:
        # 告诉 Python 这里要读取外层函数中的两个状态变量。
        nonlocal current_start, current_type
        # 只有确实存在一个实体开始位置时才需要保存。
        if current_start is not None and current_type is not None:
            # 通过切片取得完整实体词，而不是把每个字单独评价。
            entity_text = text[current_start:end]
            # 按约定输出实体文本、类别和左闭右开的字符跨度。
            entities.append(
                {
                    "entity": entity_text,
                    "type": current_type,
                    "start": current_start,
                    "end": end,
                }
            )
        # 清空开始位置，表示当前实体已经关闭。
        current_start = None
        # 清空实体类别，为下一实体做好准备。
        current_type = None

    # 逐个位置检查 BIO 标签，以便恢复完整词语边界。
    for index, label in enumerate(labels):
        # O 表示当前位置不是实体的一部分。
        if label == "O":
            # 如果前方有实体，就在当前 O 字符之前结束它。
            close_entity(index)
            # 当前标签处理完毕，继续检查下一个字符。
            continue
        # 取出 B- 或 I- 之后的实际实体类别。
        entity_type = label[2:]
        # B 标签总是表示一个新实体从当前位置开始。
        if label.startswith("B-"):
            # 先关闭可能存在的前一个实体。
            close_entity(index)
            # 记录新实体的起始字符位置。
            current_start = index
            # 记录新实体的类别。
            current_type = entity_type
            # 新实体已开始，继续读取后续字符。
            continue
        # 合法的 I 标签必须紧跟同类型的 B 或 I 标签。
        if current_start is None or current_type != entity_type:
            # 严格模式下将不合法的 I 标签视为数据错误。
            if strict:
                # 错误信息指出问题位置和类别，方便排查原始标注。
                raise ValueError(f"位置 {index} 存在没有对应 B 标签的 {label}。")
            # 非严格模式会先关闭前一个实体，以尽可能恢复可用数据。
            close_entity(index)
            # 将这个孤立 I 标签当成新实体的起点。
            current_start = index
            # 保存它所声明的类别。
            current_type = entity_type

    # 到达句子结尾时，关闭最后一个可能尚未保存的实体。
    close_entity(len(chars))
    # 返回完整词级实体列表。
    return entities


def make_sft_record(
    sentence: dict[str, list[str]], split: str, sample_index: int
) -> dict[str, Any]:
    """把一个 BIO 句子转换为适合 SFT 的对话式训练记录。"""

    # 从解析后的句子中取得字符序列。
    chars = sentence["chars"]
    # 从解析后的句子中取得与字符对应的 BIO 标签。
    labels = sentence["labels"]
    # 拼接字符，恢复模型真正看到的自然语言文本。
    text = "".join(chars)
    # 将逐字标签恢复成词级实体监督答案。
    entities = bio_to_entities(chars, labels)
    # 把实体答案序列化为紧凑 JSON，减少训练所需 token 数量。
    completion_text = json.dumps(entities, ensure_ascii=False, separators=(",", ":"))
    # 返回 TRL 可识别的 conversational prompt-completion 结构。
    return {
        "id": f"{split}-{sample_index:06d}",
        "text": text,
        "gold_entities": entities,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text)},
        ],
        "completion": [{"role": "assistant", "content": completion_text}],
    }


def entities_to_bio(text: str, entities: list[dict[str, Any]]) -> list[str]:
    """将词级实体重新转换为逐字符 BIO 标签，供判别式 NER 基线复用同一数据。"""

    # 默认每个字符都在实体之外。
    labels = ["O"] * len(text)
    # 按起始位置排序，确保检查重叠时行为稳定。
    sorted_entities = sorted(entities, key=lambda entity: (entity["start"], entity["end"]))
    # 逐一把完整实体写回字符标签。
    for entity in sorted_entities:
        # 读取实体起始位置。
        start = entity["start"]
        # 读取实体结束位置。
        end = entity["end"]
        # 读取实体类别。
        entity_type = entity["type"]
        # 检查类别是否合法。
        if entity_type not in ENTITY_TYPES:
            # 未知类别不能进入 token classification 标签集合。
            raise ValueError(f"出现未知实体类别：{entity_type}")
        # 检查字符范围和实体原文是否一致。
        if not (0 <= start < end <= len(text) and text[start:end] == entity["entity"]):
            # 位置错误会导致 BIO 与原文错位，因此直接终止。
            raise ValueError(f"实体跨度与原文不一致：{entity}")
        # 检查当前实体是否与已经写入的实体发生重叠。
        if any(label != "O" for label in labels[start:end]):
            # 普通 BIO 不支持重叠实体，明确指出问题。
            raise ValueError(f"检测到重叠实体，无法转为 BIO：{entity}")
        # 在实体第一个字符处写入 B 标签。
        labels[start] = f"B-{entity_type}"
        # 在后续字符处写入 I 标签。
        for position in range(start + 1, end):
            # 保存实体内部标签。
            labels[position] = f"I-{entity_type}"
    # 返回与文本长度一致的字符标签列表。
    return labels


def write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> None:
    """将若干条字典记录保存成一行一个 JSON 对象的 JSONL 文件。"""

    # 将输出位置统一转换为 Path 对象。
    path = Path(output_path)
    # 自动创建父目录，使第一次运行也不会因为目录缺失而失败。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用 UTF-8 写文件并保留中文，方便直接人工检查处理结果。
    with path.open("w", encoding="utf-8") as writer:
        # 逐条写入记录，避免一次把整个数据集复制到新字符串中。
        for record in records:
            # 每条 JSON 记录单独占一行，这是 datasets 加载 JSONL 的标准方式。
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，供测试集生成式评价使用。"""

    # 将输入路径统一转换为 Path 对象。
    path = Path(file_path)
    # 准备列表保存解析后的每条记录。
    records: list[dict[str, Any]] = []
    # 以 UTF-8 方式打开预处理后的文件。
    with path.open("r", encoding="utf-8") as reader:
        # 逐行读取 JSONL，从 1 开始记录行号以便定位错误。
        for line_number, line in enumerate(reader, start=1):
            # 跳过理论上不应出现的空行，使读取更稳健。
            if not line.strip():
                # 直接处理下一行。
                continue
            # 将当前 JSON 文本还原为 Python 字典。
            try:
                # 成功解析的对象会被追加到结果列表。
                records.append(json.loads(line))
            # 捕获 JSON 格式错误，提供比默认报错更清楚的位置。
            except json.JSONDecodeError as error:
                # 抛出包含路径和行号的新异常。
                raise ValueError(f"{path}:{line_number} 不是合法 JSON。") from error
    # 返回文件中的所有记录。
    return records


def parse_model_json_with_status(response_text: str) -> tuple[list[Any], bool]:
    """解析模型回答，并同时告诉调用者回答是否是合法 JSON 数组。"""

    # 用正则寻找第一个以方括号包住的数组片段。
    matched_array = re.search(r"\[[\s\S]*\]", response_text.strip())
    # 没有数组就视为模型没有产生可评价的实体结果。
    if matched_array is None:
        # 返回空列表和失败标记，让推理脚本可以选择自动要求模型重写。
        return [], False
    # 尝试将找到的数组文本解析为 Python 对象。
    try:
        # 读取 JSON 数组内容。
        parsed = json.loads(matched_array.group(0))
    # 如果模型输出的 JSON 语法有误，就按无有效预测处理。
    except json.JSONDecodeError:
        # 返回失败标记，避免把语法错误误当作正确的“无实体”答案。
        return [], False
    # 只有列表才符合本任务的输出协议。
    if not isinstance(parsed, list):
        # 非列表结果不能用于实体评价。
        return [], False
    # 返回已经解析的候选实体数组，并标记 JSON 协议合法。
    return parsed, True


def parse_model_json(response_text: str) -> list[Any]:
    """从可能带 Markdown 包围符的模型回答中提取 JSON 数组。"""

    # 调用带状态的解析函数，并忽略只在生成诊断阶段需要的状态布尔值。
    parsed, _ = parse_model_json_with_status(response_text)
    # 保持原有接口：调用者直接得到候选实体数组。
    return parsed


def normalize_predicted_entities(
    text: str, response_text: str
) -> tuple[list[dict[str, Any]], int]:
    """校验模型实体的类别、跨度和原文一致性，并统计无效项。"""

    # 先从模型回答中取得候选 JSON 列表。
    candidate_entities = parse_model_json(response_text)
    # 准备经过合法性检查的实体列表。
    valid_entities: list[dict[str, Any]] = []
    # 使用集合去除模型可能重复生成的相同实体。
    seen_keys: set[tuple[str, int, int, str]] = set()
    # 记录输出中无法用于评分的实体数量。
    invalid_count = 0
    # 逐一检查模型生成的候选实体。
    for candidate in candidate_entities:
        # 一个实体必须是包含字段的 JSON 对象。
        if not isinstance(candidate, dict):
            # 统计这一条无效输出。
            invalid_count += 1
            # 跳过无效条目。
            continue
        # 读取模型给出的实体类别。
        entity_type = candidate.get("type")
        # 读取模型给出的实体文本。
        entity_text = candidate.get("entity")
        # 读取模型给出的开始位置。
        start = candidate.get("start")
        # 读取模型给出的结束位置。
        end = candidate.get("end")
        # 检查字段类型、允许类别、位置范围和原文切片是否完全一致。
        is_valid = (
            isinstance(entity_type, str)
            and entity_type in ENTITY_TYPES
            and isinstance(entity_text, str)
            and isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end <= len(text)
            and text[start:end] == entity_text
        )
        # 不满足协议的实体不能被算作正确预测。
        if not is_valid:
            # 累计无效条目数，供最终报告分析生成质量。
            invalid_count += 1
            # 继续检查下一条候选实体。
            continue
        # 构建不可变键值，用于去重和后续精确匹配。
        entity_key = (entity_type, start, end, entity_text)
        # 如果模型重复输出同一实体，只保留第一条。
        if entity_key in seen_keys:
            # 重复输出也属于生成格式问题，因此进行计数。
            invalid_count += 1
            # 跳过重复记录。
            continue
        # 将新实体键加入已经见过的集合。
        seen_keys.add(entity_key)
        # 保存字段顺序稳定的有效实体。
        valid_entities.append(
            {
                "entity": entity_text,
                "type": entity_type,
                "start": start,
                "end": end,
            }
        )
    # 返回有效实体和无效实体数量。
    return valid_entities, invalid_count


def entity_to_key(entity: dict[str, Any]) -> tuple[str, int, int, str]:
    """把一个实体字典转换成可用于集合精确匹配的键。"""

    # 同时包含类别、跨度和文本，确保完整词与类别都必须正确。
    return (entity["type"], entity["start"], entity["end"], entity["entity"])


def _safe_scores(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    """根据 TP、FP、FN 安全计算 precision、recall 与 F1。"""

    # 当没有预测实体时，precision 按 0 处理，避免除零错误。
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    # 当没有金标准实体时，recall 按 0 处理，避免除零错误。
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    # precision 与 recall 都为 0 时，F1 自然定义为 0。
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # 返回便于序列化和打印的评分字典。
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_entity_f1(
    gold_batches: list[list[dict[str, Any]]],
    predicted_batches: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """按完整实体跨度和类别精确匹配，计算每类及总体微平均 F1。"""

    # 测试集中的金标准句数与预测句数必须一致。
    if len(gold_batches) != len(predicted_batches):
        # 不一致时停止计算，防止评价样本错位。
        raise ValueError("金标准样本数与预测样本数不一致。")
    # 为每个类别初始化 TP、FP、FN 和金标准实体数统计。
    counts = {
        entity_type: Counter({"tp": 0, "fp": 0, "fn": 0, "support": 0})
        for entity_type in ENTITY_TYPES
    }
    # 将每句话的预测与对应金标准放在一起评价。
    for gold_entities, predicted_entities in zip(gold_batches, predicted_batches):
        # 将金标准实体转为集合，以完整跨度作为一个“词”进行判断。
        gold_keys = {entity_to_key(entity) for entity in gold_entities}
        # 将预测实体转为集合，重复实体不会被重复奖励。
        predicted_keys = {entity_to_key(entity) for entity in predicted_entities}
        # 对十种类别逐一累加混淆计数。
        for entity_type in ENTITY_TYPES:
            # 筛出当前类别的金标准实体键。
            typed_gold = {key for key in gold_keys if key[0] == entity_type}
            # 筛出当前类别的预测实体键。
            typed_predicted = {key for key in predicted_keys if key[0] == entity_type}
            # 交集代表类别和完整实体边界都正确的预测。
            counts[entity_type]["tp"] += len(typed_gold & typed_predicted)
            # 仅在预测中的实体是假阳性。
            counts[entity_type]["fp"] += len(typed_predicted - typed_gold)
            # 仅在金标准中的实体是假阴性。
            counts[entity_type]["fn"] += len(typed_gold - typed_predicted)
            # support 表示测试集中该类别的真实实体总数。
            counts[entity_type]["support"] += len(typed_gold)

    # 准备用于输出的逐类别评分字典。
    per_type: dict[str, dict[str, float | int]] = {}
    # 逐类别把计数转换为 P、R、F1。
    for entity_type in ENTITY_TYPES:
        # 取出当前类别的累计计数。
        category_counts = counts[entity_type]
        # 根据当前类别计数计算三个比率指标。
        scores = _safe_scores(
            category_counts["tp"], category_counts["fp"], category_counts["fn"]
        )
        # 保存评分及可解释的原始计数。
        per_type[entity_type] = {
            **scores,
            "support": category_counts["support"],
            "tp": category_counts["tp"],
            "fp": category_counts["fp"],
            "fn": category_counts["fn"],
        }
    # 汇总所有实体类别的 TP 总数。
    total_tp = sum(count["tp"] for count in counts.values())
    # 汇总所有实体类别的 FP 总数。
    total_fp = sum(count["fp"] for count in counts.values())
    # 汇总所有实体类别的 FN 总数。
    total_fn = sum(count["fn"] for count in counts.values())
    # 汇总所有实体类别的真实实体总数。
    total_support = sum(count["support"] for count in counts.values())
    # 计算最常用于 NER 汇报的总体微平均 P、R、F1。
    overall_scores = _safe_scores(total_tp, total_fp, total_fn)
    # 计算十类 F1 的简单平均值，用于观察小类表现。
    macro_f1 = sum(item["f1"] for item in per_type.values()) / len(ENTITY_TYPES)
    # 返回结构化结果，既能打印也能另存为 JSON。
    return {
        "overall": {
            **overall_scores,
            "macro_f1": macro_f1,
            "support": total_support,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "per_type": per_type,
    }


def bootstrap_micro_f1_interval(
    gold_batches: list[list[dict[str, Any]]],
    predicted_batches: list[list[dict[str, Any]]],
    num_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float | int]:
    """对句子进行有放回重采样，估计总体 micro F1 的 95% 置信区间。"""

    # 金标准和预测必须逐句对齐，才能按样本重新抽取。
    if len(gold_batches) != len(predicted_batches):
        # 如果数量不同，立即停止，避免产生没有意义的区间。
        raise ValueError("金标准样本数与预测样本数不一致。")
    # 无样本时不存在可解释的 bootstrap 区间。
    if not gold_batches:
        # 抛错提示调用者检查输入文件。
        raise ValueError("无法对空数据集计算置信区间。")
    # 至少需要一次重采样，零次会导致无法计算百分位点。
    if num_resamples <= 0:
        # 抛错提示命令行参数设置错误。
        raise ValueError("num_resamples 必须大于 0。")
    # 用局部随机数生成器和固定种子保证重复运行结果一致。
    random_generator = random.Random(seed)
    # 记录每轮有放回抽样得到的 micro F1。
    sampled_f1_values: list[float] = []
    # 保存句子总数，作为每轮需要抽取的样本量。
    sample_count = len(gold_batches)
    # 重复抽样指定次数，次数越高区间越稳定但计算时间略增加。
    for _ in range(num_resamples):
        # 有放回地抽取与原数据集同样数量的句子下标。
        sampled_indices = [random_generator.randrange(sample_count) for _ in range(sample_count)]
        # 按抽取下标组合本轮的金标准实体列表。
        sampled_gold = [gold_batches[index] for index in sampled_indices]
        # 按完全相同的下标组合本轮的模型预测列表。
        sampled_predictions = [predicted_batches[index] for index in sampled_indices]
        # 计算本轮抽样数据的严格词级 micro F1。
        sampled_metrics = compute_entity_f1(sampled_gold, sampled_predictions)
        # 保存本轮 F1，之后用经验百分位数形成区间。
        sampled_f1_values.append(float(sampled_metrics["overall"]["f1"]))
    # 从小到大排序，方便读取 2.5% 与 97.5% 两个百分位。
    sampled_f1_values.sort()
    # 通过下标取得经验分布下侧 2.5% 点。
    lower_index = int((num_resamples - 1) * 0.025)
    # 通过下标取得经验分布上侧 97.5% 点。
    upper_index = int((num_resamples - 1) * 0.975)
    # 计算原始未重采样测试集上的中心 F1 值。
    point_estimate = float(compute_entity_f1(gold_batches, predicted_batches)["overall"]["f1"])
    # 返回结构化区间，方便保存到指标 JSON 与终端展示。
    return {
        "point_estimate": point_estimate,
        "lower_95": sampled_f1_values[lower_index],
        "upper_95": sampled_f1_values[upper_index],
        "num_resamples": num_resamples,
        "seed": seed,
    }


def format_metric_report(metrics: dict[str, Any]) -> str:
    """把评分结果排版为适合终端与 Notebook 展示的表格文本。"""

    # 定义表头，让读者明确每列数字的意义。
    lines = [
        "类别             Precision    Recall        F1   Support    TP    FP    FN",
        "-------------------------------------------------------------------------",
    ]
    # 按固定类别顺序逐行写入每一种实体类别的结果。
    for entity_type in ENTITY_TYPES:
        # 读取当前类别对应的评分与计数。
        result = metrics["per_type"][entity_type]
        # 使用对齐格式保证中英文终端中尽量易读。
        lines.append(
            f"{entity_type:<12} {result['precision']:>9.4f} {result['recall']:>9.4f} "
            f"{result['f1']:>9.4f} {result['support']:>9} {result['tp']:>5} "
            f"{result['fp']:>5} {result['fn']:>5}"
        )
    # 取出所有类别共同汇总后的微平均结果。
    overall = metrics["overall"]
    # 在表格下追加总体指标和宏平均 F1。
    lines.extend(
        [
            "-------------------------------------------------------------------------",
            f"总体 micro       {overall['precision']:>9.4f} {overall['recall']:>9.4f} "
            f"{overall['f1']:>9.4f} {overall['support']:>9} {overall['tp']:>5} "
            f"{overall['fp']:>5} {overall['fn']:>5}",
            f"宏平均 macro F1: {overall['macro_f1']:.4f}",
        ]
    )
    # 如果评价脚本计算了置信区间，就把它追加在主指标下方。
    if "bootstrap_micro_f1_95_ci" in metrics:
        # 读取置信区间结果，便于排版显示。
        interval = metrics["bootstrap_micro_f1_95_ci"]
        # 向读者展示 F1 的不确定性范围及重采样次数。
        lines.append(
            "总体 micro F1 95% bootstrap CI: "
            f"[{interval['lower_95']:.4f}, {interval['upper_95']:.4f}] "
            f"(重采样 {interval['num_resamples']} 次)"
        )
    # 使用换行拼接各行，形成一个可直接 print 的字符串。
    return "\n".join(lines)
