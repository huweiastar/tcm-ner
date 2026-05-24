"""验证数据清理、重采样和误差分类等优化辅助工具的关键行为。"""

# 导入 unittest，使用 Python 内置测试能力。
import unittest

# 导入误差事件分类函数。
from analyze_predictions import classify_errors
# 导入干净划分工具的内部与跨划分处理函数。
from create_clean_splits import deduplicate_inside_split, remove_cross_split_overlap
# 导入训练集实体计数与复制函数。
from rebalance_train_data import copy_training_record, count_entities


def make_record(sample_id: str, text: str, entities: list[dict]) -> dict:
    """构造最小 SFT 记录，便于无需文件即可验证优化规则。"""

    # 返回优化工具所需的核心字段。
    return {"id": sample_id, "text": text, "gold_entities": entities, "predicted_entities": []}


class OptimizationToolsTestCase(unittest.TestCase):
    """验证数据优化过程不会静默污染评测口径。"""

    def test_conflicting_duplicate_raises_by_default(self) -> None:
        """相同文本标注不同的记录默认必须阻止自动清理。"""

        # 建立同文本但类别不同的两条训练记录。
        records = [
            make_record("train-1", "腹痛", [{"entity": "腹痛", "type": "临床表现", "start": 0, "end": 2}]),
            make_record("train-2", "腹痛", [{"entity": "腹痛", "type": "中医诊断", "start": 0, "end": 2}]),
        ]
        # 默认 error 策略必须要求人工复核。
        with self.assertRaises(ValueError):
            # 调用同划分重复清理。
            deduplicate_inside_split("train", records, "error")

    def test_cross_split_rule_keeps_protected_test_text(self) -> None:
        """已在测试集出现的相同文本不得继续保留在训练集中。"""

        # 建立测试集中应被保护的文本字典。
        protected_texts = {"腹痛": "test"}
        # 建立训练集中发生重叠的一条记录。
        train_records = [make_record("train-1", "腹痛", [])]
        # 执行跨划分清理。
        kept, removed = remove_cross_split_overlap("train", train_records, protected_texts)
        # 重叠训练文本不应被保留。
        self.assertEqual(kept, [])
        # 报告中应记录一条移除动作。
        self.assertEqual(len(removed), 1)
        # 明确记录是测试集优先保护了该文本。
        self.assertEqual(removed[0]["protected_split"], "test")

    def test_oversampled_record_keeps_source_trace(self) -> None:
        """重采样副本必须记录来源，避免被误认为新增人工标注。"""

        # 建立带其他治疗实体的训练样本。
        source = make_record(
            "train-1",
            "心理疏导",
            [{"entity": "心理疏导", "type": "其他治疗", "start": 0, "end": 4}],
        )
        # 复制这条低频类别训练记录。
        copied = copy_training_record(source, copy_index=1, target_type="其他治疗")
        # 副本应记录来源样本 id。
        self.assertEqual(copied["oversampled_from_id"], "train-1")
        # 计数函数应能统计复制后的目标类别实体。
        self.assertEqual(count_entities([source, copied])["其他治疗"], 2)

    def test_error_analysis_recognizes_boundary_error(self) -> None:
        """只预测实体一部分时应被归类为边界错误而非正确结果。"""

        # 建立金标准为“腹痛”、预测仅为“痛”的样本。
        record = make_record(
            "test-1",
            "腹痛",
            [{"entity": "腹痛", "type": "临床表现", "start": 0, "end": 2}],
        )
        # 填入不完整预测。
        record["predicted_entities"] = [
            {"entity": "痛", "type": "临床表现", "start": 1, "end": 2}
        ]
        # 拆解错误事件。
        events = classify_errors(record)
        # 应找到一个边界错误事件。
        self.assertEqual(events[0]["error_type"], "boundary_error")


# 支持直接运行本测试文件。
if __name__ == "__main__":
    # 启动 unittest。
    unittest.main()
