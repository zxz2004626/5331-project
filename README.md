# COMP5331 Graph Anomaly Benchmark

研究问题：**异常边的合成方式，如何影响轻量图异常检测模型的成绩、迁移能力和特征收益？**

在 MovieLens 100K 和 UCI Messages 上比较 G0–G3 四种异常生成方式，使用 RF/LR 和可解释图特征。扩展实验保存了 600 个最终模型的记录、1080 个测试评估单元；官方 TGF 复现及时间协议对照另行报告。生成器采用近似匹配，结论针对合成标签。

## 从哪里开始

- [实验报告](reports/experiment_report.md) / [英文实验 PDF](reports/experiment_report.pdf)：已有结果与解释。这是实验记录，课程最终论文维护在 Overleaf。
- [正式 proposal](docs/proposal.pdf)：项目依据；旧稿集中在 `docs/archive/`。
- [运行指南](docs/running.md)：安装、迁移、复现和常见问题。
- [代码导览](docs/architecture.md)：模块职责、数据流、六人协作边界。
- [实验协议](docs/experiment_plan.md) / [特征定义](docs/features.md)：科学定义。
- `results/report/`：可直接查看的机器可读表格与图形。

## 安装

使用 **64 位 Python 3.10 或 3.11（推荐 3.11）**。原始依赖固定在 `requirements.txt`，以兼容官方 notebook；不支持直接换成 pandas 2 或 NumPy 2。

```sh
python -m venv .venv
# 激活环境：Windows PowerShell 用 .venv\Scripts\Activate.ps1
# Linux/macOS 用 source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

安装后可从任意目录运行。源码脚本也继续可用：`python /path/to/project/scripts/build_experiments.py --help`。

```sh
python -m graphbench --help
python -m graphbench run --dry-run
python -m graphbench prepare
python -m graphbench generate --workers 1
python -m graphbench evaluate --ablations --n-jobs 2
```

完整复现：`python -m graphbench run`。首次运行需要下载上游代码、持续 CPU 计算和数 GB 存储。只有官方原始注入的 `--fresh-pipeline` 步骤依赖 Bash 及 Unix 命令工具；Windows 使用 Git Bash。先看 [运行指南](docs/running.md)。

## 目录

```text
configs/                  固定实验协议
src/graphbench/            可安装的 Python 包
  data.py, history.py      数据与历史图特征
  generate.py             G0–G3 注入与特征缓存
  cache.py, metrics.py     缓存校验、特征对齐、评价指标
  models.py, evaluate.py   模型选择与训练评估调度
  artifacts.py            原子读写与数据文件校验
  provenance.py           源码依赖指纹
  summaries.py            有效运行记录汇总
  workflows/              命令行流程与报告生成
scripts/                  兼容旧命令的轻量入口
tests/                    协议、模型与迁移测试
docs/                     方案、说明及方法审计
reports/                  人类阅读的实验报告
data/manifest.json        数据来源与规模
results/                  已保存的实验记录和图表
vendor/TGF/               prepare 下载的固定上游版本（不提交）
```

## 结果与重建的边界

仓库保留数值记录、表格、图形和历史审计。体积较大的主实验缓存、模型和预测未提交，克隆后不能直接进行完整事件级重评分，需先重建。`graphbench report` 可使用已保存诊断摘要重绘汇总图；原始特征 CDF 缺少缓存时会明确跳过。

重构不会把旧结果的来源哈希改成新代码。旧结果仍是原运行的记录；新代码更改了依赖指纹，应在独立检出目录中重新生成和验证。详见 [迁移说明](docs/running.md#已有结果和缓存)。

## 检查

```sh
python -m pytest -q
ruff check src scripts tests
ruff format --check src scripts tests
```

CI 在 Windows、Linux、macOS 上使用 Python 3.11 检查安装、单元测试和入口；完整研究实验不在 CI 中重跑。

主要来源：Latapy & Rajeh, *Trivial Graph Features and Classical Learning are Enough to Detect Random Anomalies*, ICDM 2025。官方代码：[TGF](https://github.com/StephanyRajeh/TGF)，固定提交 `af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b`。上游代码沿用其 GPL-2.0 许可。
