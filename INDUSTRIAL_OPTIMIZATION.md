# 工业级优化实施状态

本文件记录最初优化清单中哪些项目已经在工程中实现，哪些需要 GPU、部署引擎
或业务人工确认后才能得出结论。不要把“有代码入口”误认为“已经得到最优模型”。

## 已实现并可立即运行

| 原优化项 | 当前实现文件 | 实现内容 | 当前真实结果 |
| --- | --- | --- | --- |
| 数据质量审计与泄漏检查 | `audit_dataset.py` | 检查 BIO 合法性、数据指纹、内部重复、跨集合重叠、标注冲突、全半角风险和类别分布 | `reports/data_audit.md` 显示跨划分相同文本 19 条、跨划分冲突 4 组；训练内部冲突 13 组、验证内部冲突 1 组 |
| 避免评测泄漏 | `create_clean_splits.py` | 生成新副本，优先保护 test，再从 validation/train 删除完全相同文本；默认遇冲突停止 | 教学模式 `keep_first` 已生成 `processed_data/clean/`：train 5184、validation 654、test 658 |
| 小类别增强 | `rebalance_train_data.py` | 仅复制训练集中含目标类别的样本，每个副本记录来源，绝不增强验证/测试 | `其他治疗` 完整实体从 65 增至 300；输出 `processed_data/optimized/train_balanced.jsonl` |
| 生成格式稳定性 | `evaluate_ner.py` | 检测非法 JSON/非法跨度后自动纠错重试一次，并报告救回数量 | 等训练 adapter 后获得实际救回结果 |
| 更严格的评价 | `ner_utils.py`、`evaluate_ner.py` | 完整实体严格 F1、每类 F1、macro/micro F1、bootstrap 95% CI | 指标逻辑已由单元测试验证 |
| 验证集误差分析 | `analyze_predictions.py` | 统计漏检/边界/类别/多余实体错误，按文本长度、数字、剂量、多实体切片 | 等生成验证集预测后运行 |
| 超参数与目标层消融 | `make_ablation_configs.py`、`compare_experiments.py` | 生成 rank、学习率、目标层 YAML；按 validation F1 排行 | 配置已生成到 `configs/ablations/` |
| 训练效率与复现 | `train_sft.py`、`configs/*_optimized.yaml` | 早停、长度分桶、SDPA/FlashAttention 可选入口、实验数据指纹和依赖版本清单 | 等 GPU 训练验证吞吐与 F1 |
| 多卡训练入口 | `configs/accelerate_multi_gpu.yaml` | 单机两 GPU 的 Accelerate 配置模板 | 需在实际 GPU 机器按卡数修改 |
| 判别式 NER 参照 | `train_token_baseline.py`、`configs/token_baseline.yaml` | 使用同一干净/增强数据与同一完整实体 F1，对比较小模型方案 | 可选，不替代 Qwen2.5-7B 作业主方案 |

## 必须人工确认的事项

审计已经证实存在同文本标注冲突。`create_clean_splits.py` 的默认策略为
`--within_conflict_policy error`，会停止运行并要求人工确认。项目中为演示后续
流程生成的 `processed_data/clean/` 使用了显式的 `keep_first` 策略，它适合先跑
通代码，不等于医学标注已经被权威修订。

用于正式报告或提交高可信结果前，应执行：

1. 打开 `reports/data_audit.json` 和 `reports/clean_split_report.json` 查看冲突条目。
2. 与课程标注规范或具备领域知识的人员确认同文本应该保留哪一份实体答案。
3. 在原始 `data/medical.*` 标注文件中完成修订，或建立经过确认的新数据版本。
4. 重新运行预处理、审计、clean 划分与训练，保存新的数据指纹。

## 属于部署阶段、当前不应假装完成的优化

| 优化方向 | 为什么现在未直接强行实现 | 后续实施条件 |
| --- | --- | --- |
| 真正 grammar/JSON-schema 约束解码 | 标准 Transformers 贪心生成本身不保证语法约束；当前实现为一次自动纠错重试 | 选择支持 JSON schema 或 grammar 的推理服务框架，并重新比较无效输出率与吞吐 |
| vLLM/量化在线服务与 SLA 调优 | 需要确定 GPU、并发量、延迟目标和部署环境 | 有实际服务硬件和压测数据后进行 |
| 线上数据漂移监控和合规处理 | 当前只有离线作业数据，没有新流量或隐私规则 | 接入新的脱敏线上样本和企业 MLOps/合规流程后建设 |

## 推荐实验顺序

1. 使用原始数据跑 `audit_dataset.py`，先确认数据问题。
2. 人工修订冲突；仅学习流程时可显式使用 `keep_first` 生成副本。
3. 仅对干净训练集执行 `rebalance_train_data.py`。
4. 先训练 `configs/qlora_optimized.yaml` 作为主优化模型。
5. 在 `processed_data/clean/validation.jsonl` 上评价并运行错误分析。
6. 再运行消融配置，用 `compare_experiments.py` 根据验证集 F1 选择模型。
7. 选定配置后，只在干净测试集上运行一次最终评价。
8. 若做工业扩展，再运行判别式基线与部署性能实验。
