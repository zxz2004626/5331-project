# 代码导览

## 数据流

`prepare → data.load_dataset → generate.build_cache → evaluate.run_evaluations → summaries → report`

1. `data.py` 读取固定上游样例，稳定排序时间戳，建立节点映射和时间划分。
2. `history.py` 维护滑动历史、局部结构、近期活动和 lifetime pair 信息。
3. `generate.py` 在严格过去的历史上生成异常和特征，再按时间批次更新历史。
4. `cache.py` 校验缓存及时间划分，通过列名对齐训练和测试特征。
5. `models.py` 只在训练/验证阶段选参，然后在训练加验证集上重拟合。
6. `evaluate.py` 管理模型断点、跨生成器预测和失败记录；`metrics.py` 统一评价指标。
7. `summaries.py` 只汇总来源与产物指纹一致的记录。`workflows/` 将这些能力组织成命令。

## 通用设施

- `paths.py`：环境变量或源码位置定位项目目录，避免个人绝对路径。
- `artifacts.py`：JSON/CSV 原子写入、文件哈希及 JSON 指纹。
- `provenance.py`：显式列出生成和评估的源码依赖，避免漏掉拆分模块；源码统一按 LF 计算。
- `constants.py`：数据集、种子、生成器和特征组名称。
- `shell.py`：发现 Bash，将 Python 和脚本路径作为参数传递。
- `report_inputs.py`：从缓存或已保存的历史审计读取报告所需摘要。

`evaluate.py` 继续导出旧公共函数，因此原有调用方式以及旧估计器的导入路径保持可解析。新代码建议直接导入职责模块，例如 `from graphbench.metrics import compute_metrics`。

`workflows/verify_artifacts.py` 的独立指标实现保留为审计参照，不复用生产指标函数，避免同一个错误同时影响计算和核验。

## 六人合作建议

| 负责人 | 主要文件 | 交接产物 |
| --- | --- | --- |
| 1 | `data.py`、`paths.py`、prepare workflow | 输入与数据 manifest |
| 2 | `history.py` | 特征定义、图历史测试 |
| 3 | `generate.py`、generate workflow | 缓存、注入与匹配诊断 |
| 4 | `models.py`、`evaluate.py`、`cache.py` | 模型、预测和断点记录 |
| 5 | `metrics.py`、`summaries.py`、verify workflow | 指标与独立审计 |
| 6 | report workflows、`reports/`、`docs/` | 图表、报告与说明 |

这是协作建议，不是六位成员实际 Contribution 的声明。共用协议 `configs/experiment.json` 的修改应先组内讨论；不要为使旧记录通过检查而修改历史哈希或已有数值。
