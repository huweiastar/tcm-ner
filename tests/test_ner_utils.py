"""对数据转换与词级 F1 进行不需要 GPU 的基础验证。"""

# 导入临时文件夹工具，使测试不会污染项目数据。
import tempfile
# 导入 unittest，使用 Python 自带测试框架即可执行。
import unittest
# 导入 Path，便于创建临时 BIO 文件。
from pathlib import Path

# 导入本项目需要验证的核心函数。
from ner_utils import (
    ENTITY_TYPES,
    bio_to_entities,
    bootstrap_micro_f1_interval,
    compute_entity_f1,
    entities_to_bio,
    format_metric_report,
    make_sft_record,
    normalize_predicted_entities,
    parse_model_json_with_status,
    read_bio_sentences,
)


class NerUtilsTestCase(unittest.TestCase):
    """验证 BIO 转换、输出协议和严格完整实体 F1。"""

    def test_read_bio_and_convert_to_sft_record(self) -> None:
        """CRLF BIO 文件应转换为带完整实体跨度的 SFT 记录。"""

        # 创建运行完测试后会自动清理的临时目录。
        with tempfile.TemporaryDirectory() as temp_directory:
            # 在临时目录中建立一份最小 BIO 数据文件。
            bio_path = Path(temp_directory) / "sample.train"
            # 写入包含两个完整实体的 CRLF 示例，模拟原始数据格式。
            bio_path.write_text(
                "黄 B-中药\r\n芪 I-中药\r\n治 O\r\n疗 O\r\n头 B-临床表现\r\n痛 I-临床表现\r\n\r\n",
                encoding="utf-8",
            )
            # 读取临时 BIO 文件。
            sentences = read_bio_sentences(bio_path)
            # 把第一句话转成监督微调记录。
            record = make_sft_record(sentences[0], split="train", sample_index=1)

        # 检查句子能够从字符行恢复成自然文本。
        self.assertEqual(record["text"], "黄芪治疗头痛")
        # 检查实体按完整词保存，并且 end 为右开区间。
        self.assertEqual(
            record["gold_entities"],
            [
                {"entity": "黄芪", "type": "中药", "start": 0, "end": 2},
                {"entity": "头痛", "type": "临床表现", "start": 4, "end": 6},
            ],
        )
        # 检查训练答案中确实包含实体文本与跨度字段。
        self.assertIn('"entity":"黄芪"', record["completion"][0]["content"])

    def test_bio_rejects_orphan_inside_tag(self) -> None:
        """没有 B 开头的 I 标签在严格模式下应被拒绝。"""

        # 断言错误 BIO 不会静默变成训练监督信号。
        with self.assertRaises(ValueError):
            # 单个 I 标签没有对应的实体开始位置。
            bio_to_entities(["痛"], ["I-临床表现"])

    def test_full_width_space_character_is_preserved(self) -> None:
        """原始语句中的全角空格不能在读取时被误删。"""

        # 创建会在测试完成后自动删除的临时目录。
        with tempfile.TemporaryDirectory() as temp_directory:
            # 准备只包含全角空格字符及其 O 标签的数据文件。
            bio_path = Path(temp_directory) / "space.train"
            # 数据中的第一个字符是中文排版常见的全角空格。
            bio_path.write_text("　 O\r\n黄 B-中药\r\n芪 I-中药\r\n\r\n", encoding="utf-8")
            # 调用真实读取逻辑解析该句子。
            sentences = read_bio_sentences(bio_path)
        # 断言正文开头的全角空格仍然存在。
        self.assertEqual("".join(sentences[0]["chars"]), "　黄芪")
        # 断言实体起点已经正确跳过这个被保留的字符。
        self.assertEqual(bio_to_entities(sentences[0]["chars"], sentences[0]["labels"])[0]["start"], 1)

    def test_invalid_generated_span_is_removed(self) -> None:
        """文本切片与 entity 不一致的模型回答不能进入评分。"""

        # 构造一个位置错误的 JSON 回答，黄芪实际位于 0 到 2。
        response = '[{"entity":"黄芪","type":"中药","start":1,"end":3}]'
        # 清洗模型回答并取得无效项计数。
        entities, invalid_count = normalize_predicted_entities("黄芪治疗", response)
        # 错误跨度不应被当作有效预测。
        self.assertEqual(entities, [])
        # 一条错误预测应被明确计入诊断信息。
        self.assertEqual(invalid_count, 1)

    def test_entity_f1_requires_a_complete_word_match(self) -> None:
        """只猜中实体中的一个字，不能被算作完整实体正确。"""

        # 金标准中包含一个中药实体和一个临床表现实体。
        gold_batches = [
            [
                {"entity": "黄芪", "type": "中药", "start": 0, "end": 2},
                {"entity": "头痛", "type": "临床表现", "start": 4, "end": 6},
            ]
        ]
        # 预测完整猜对黄芪，却只猜到了“痛”一个字。
        predicted_batches = [
            [
                {"entity": "黄芪", "type": "中药", "start": 0, "end": 2},
                {"entity": "痛", "type": "临床表现", "start": 5, "end": 6},
            ]
        ]
        # 计算严格词级实体指标。
        metrics = compute_entity_f1(gold_batches, predicted_batches)
        # 完整预测正确的中药实体 F1 应为 1。
        self.assertEqual(metrics["per_type"]["中药"]["f1"], 1.0)
        # 部分猜中的临床表现不算正确，因此 F1 必须为 0。
        self.assertEqual(metrics["per_type"]["临床表现"]["f1"], 0.0)
        # 总体只有两个真实实体中的一个被完整识别，总体 F1 为 0.5。
        self.assertEqual(metrics["overall"]["f1"], 0.5)

    def test_report_prints_every_entity_category(self) -> None:
        """最终报告必须固定列出数据集中的全部十种实体类别。"""

        # 用一个没有实体的样本生成指标，模拟某些小类没有预测的情况。
        metrics = compute_entity_f1([[]], [[]])
        # 将结构化指标排版为评价脚本实际打印的文本。
        report = format_metric_report(metrics)
        # 逐类确认报告仍包含类别名称，保证不会漏报低频类别。
        for entity_type in ENTITY_TYPES:
            # 某类别即使 F1 为零，也必须在终端报告中出现。
            self.assertIn(entity_type, report)

    def test_json_parser_distinguishes_valid_empty_answer_from_format_error(self) -> None:
        """合法的无实体数组不应被误当作需要格式重试的失败回答。"""

        # 解析模型明确输出的空数组。
        empty_entities, is_valid_empty = parse_model_json_with_status("[]")
        # 解析缺少方括号的无效输出。
        invalid_entities, is_valid_invalid = parse_model_json_with_status("没有找到实体")
        # 合法空数组应保持没有实体。
        self.assertEqual(empty_entities, [])
        # 合法空数组必须被标记为 JSON 协议成功。
        self.assertTrue(is_valid_empty)
        # 无效自然语言回答也不会产生实体。
        self.assertEqual(invalid_entities, [])
        # 无效自然语言回答必须被标记为协议失败。
        self.assertFalse(is_valid_invalid)

    def test_bootstrap_interval_is_one_for_perfect_predictions(self) -> None:
        """所有实体均预测正确时，F1 置信区间上下界都应为 1。"""

        # 建立一个完全正确的最小金标准批次。
        gold = [[{"entity": "黄芪", "type": "中药", "start": 0, "end": 2}]]
        # 预测与金标准完全一致。
        predicted = [[{"entity": "黄芪", "type": "中药", "start": 0, "end": 2}]]
        # 用少量重采样即可检查确定性完全正确场景。
        interval = bootstrap_micro_f1_interval(gold, predicted, num_resamples=10, seed=42)
        # 中心 F1 必须为满分。
        self.assertEqual(interval["point_estimate"], 1.0)
        # 下界必须为满分。
        self.assertEqual(interval["lower_95"], 1.0)
        # 上界必须为满分。
        self.assertEqual(interval["upper_95"], 1.0)

    def test_entities_can_round_trip_back_to_bio_for_baseline(self) -> None:
        """生成式实体答案应可还原为判别式基线使用的 BIO 标签。"""

        # 准备一条带完整中药实体的文本。
        text = "服黄芪"
        # 准备生成式 SFT 所使用的实体答案。
        entities = [{"entity": "黄芪", "type": "中药", "start": 1, "end": 3}]
        # 将实体答案重新转换为逐字符标签。
        labels = entities_to_bio(text, entities)
        # 检查实体外、开始和内部标签均正确。
        self.assertEqual(labels, ["O", "B-中药", "I-中药"])


# 支持直接运行本文件以执行全部测试。
if __name__ == "__main__":
    # 启动 unittest 测试执行器。
    unittest.main()
