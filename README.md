# COMP5331: graph anomaly benchmark experiments

根据 `kdd_proposal.pdf` 实现的完整实验项目。研究问题是：**改变异常边的合成方式，是否会改变我们对轻量图异常检测模型的判断？** 所有结果来自实际运行；原始 proposal 和 PDF 保持原样。

## 阅读顺序

1. `EXPERIMENT_REPORT.md`：中文实验报告与主要结论（最终汇总脚本生成）。
2. `output/pdf/experiment_report.pdf`：英文实验报告，含可直接用于课程报告的图表。
3. `docs/experiment_plan.md`：数据、生成器、因果历史、模型选择和敏感性实验的完整协议。
4. `docs/official_audit.md`、`docs/generator_audit.md`、`docs/results_review.md`：官方复现、生成器有效性和结果复核。
5. `results/report/tables/`、`results/report/figures/`：机器可读表格以及 PNG/PDF 图片。

## 实验范围

- 两个完整示例数据集：MovieLens 100K、UCI Messages。
- 官方 TGF 原始代码复现，包括原始预处理、注入/检查、H1000 特征、原始 RF notebook 学习步骤；另做同数据、同划分的严格时间历史对照。
- 主实验：G0/G1/G2/G3 × RF/LR × TGF/扩展特征 × 5 个种子 × 2 个数据集，共 **160 个核心模型**。
- 完整跨生成器矩阵：上述模型共 **640 个评估单元**。
- 特征充分性与留一组消融：额外 **200 个 RF 模型**。
- 5 项敏感性：1%/10% 异常比例、H500/H2000、past-only 非边；两个数据集、G0/G3、两套特征、5 种子，共 **200 个 RF 模型**。
- 补充归因对照：移除 `pair_seen_before` 的 **40 个 RF 模型**，复用各自完整特征模型在验证集选定的参数。

扩展部分共600个最终模型、1080个测试评估单元；另有官方/严格时间对照20个模型结果。不要把交叉测试次数当成不同的训练次数。

## 环境与复现

实际执行环境：Windows、Python3.11.4、scikit-learn1.3.0。使用 `requirements.txt` 固定主要依赖。官方 notebook 依赖 pandas1.x 的旧接口，因而不要直接升级 pandas2.x。Python 可执行文件在本次机器上为 `E:\anaconda\python.exe`。

```powershell
cd E:\A_Study\COMP5331\project
python -m pip install -r requirements.txt
python scripts/prepare_sources.py
$env:PYTHONPATH='src'
python -m pytest tests -q
.\scripts\run_all.ps1
```

`run_all.ps1` 复用有效缓存和模型断点；源码或输入哈希变化会触发失效检查。首次完整执行需要持续 CPU 计算与数 GB 磁盘空间。可以通过 `-GeneratorWorkers`、`-ModelJobs` 和 `-NoPairJobs` 调整资源占用；本次主实验和敏感性模型使用2线程，补充 no-pair 模型使用1线程。官方 fresh-pipeline 的 shell 步骤还需要 Git Bash；本机路径为 `E:\Git\bin\bash.exe`。

只重新生成最终表格、图和中文报告：

```powershell
python scripts/report_results.py
python scripts/write_report.py
```

PDF 生成需要 `reportlab` 和 `pypdf`，不影响实验本身。本次使用 Codex 附带的 Python 运行 `scripts/build_pdf_report.py`；也可在自己的 Python 环境安装这些包后运行该脚本。PNG/PDF 图由 matplotlib 生成，无须 LaTeX。

## 关键定义

- 异常为正类。主指标 **AP (average precision)**；另保存梯形 PR-AUC，避免两个不同定义混称。次要指标包括 ROC-AUC、Precision@K、Recall@K。K=测试异常数，因此后两个指标相同。
- 扩展实验的“5%”是混合流中异常占比；官方“5%”是注入数/正常数，对应混合占比4.762%。两者分开报告。
- 60/20/20 按真实时间跨度划分，不是按交互条数划分；同时间戳不跨集合。所有特征只读取严格早于当前时间戳的混合历史。
- G1/G2/G3 使用128→16→4的候选排序筛选，属于**近似匹配**。有保留候选层级不等于已实现分布完全一致；必须结合诊断图解释。
- 全局非边和全量节点集合是明确声明的合成基准先验；不把这种生成流程描述为无需未来信息的线上异常生成器。
- UCI 有大量重复正常交互，而全局非边异常没有历史交互。这种“新边”信号是残余捷径；首次正常边子集和移除历史出现标志的对照会单独报告。
- 每个种子的测试集相同时间范围；种子间标准差只表示注入/模型随机性，不能替代跨数据集或跨时间段的不确定性。

## 文件结构

```text
configs/experiment.json        固定的实验参数
src/graphbench/                数据加载、历史特征、异常生成、模型评估
scripts/                      运行、汇总、报告脚本
tests/                        时间隔离、图特征、生成器、指标和断点测试
vendor/TGF/                   原作者仓库，固定提交且未修改受跟踪源码
data/manifest.json            数据来源、SHA-256、规模与时间边界
results/cache/<scenario>/     全事件特征、注入审计、分布诊断及配置
results/evaluation/<scenario>/ 模型、测试预测、逐次指标和聚合结果
results/official/             官方复现与时间协议对照
results/no_pair/              移除 lifetime pair 标志的补充消融
results/validation/           独立结果核验
results/report/               最终表格、图形和汇总 JSON
```

大体积缓存和上游仓库被 `.gitignore` 排除，但保留在本地；可通过固定配置重新构建。不要删除需要提交的机器可读指标、报告和源代码。

## 来源

主要论文：Latapy & Rajeh, *Trivial Graph Features and Classical Learning are Enough to Detect Random Anomalies*, ICDM2025, DOI10.1109/ICDM65498.2025.00142。

- 官方网站：https://stephanyrajeh.github.io/TGFWebsite/
- 原始代码与示例数据：https://github.com/StephanyRajeh/TGF
- 固定版本：`af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b`

原作者代码沿用其 GPL-2.0 许可（见 `vendor/TGF/LICENSE`）。本项目的结论仅针对这些受控合成异常；没有真实欺诈或攻击标签。proposal 中的 MIDAS/微簇研究是可选扩展，本次未运行。
