# 图异常检测项目

[English](README.md) · [项目仓库](https://github.com/zxz2004626/5331-project)

COMP5331 第 5 组：在 MovieLens 和 UCI Messages 上比较 G0–G3 异常生成器、
RF/LR 检测器，以及可解释图特征。

## 安装与编译

使用 **64 位 Python 3.10 或 3.11**，在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Linux/macOS 改用 `source .venv/bin/activate` 激活环境。项目使用 Python，
无需原生代码编译；可用 `python -m compileall -q src` 检查字节码编译。

`prepare` 需要 Git。官方 `--fresh-pipeline` 模式还需要 Bash、gzip 和 coreutils；
Windows 请使用 Git Bash。

## 运行方法

快速验证已有预测，不下载数据、不重新训练：

```sh
python -m graph_anomaly official --verify-output
```

单个数据集、生成器、种子和模型的运行示例：

```sh
python -m graph_anomaly prepare
python -m graph_anomaly generate --datasets ucimessages --generators G0 --seeds 11 --workers 1
python -m graph_anomaly evaluate --datasets ucimessages --generators G0 --seeds 11 --models RF --features tgf --same-generator-only --n-jobs 1
```

```sh
python -m graph_anomaly run --dry-run  # 预览完整实验步骤
python -m graph_anomaly run            # 执行完整实验
python -m graph_anomaly summarize      # 重新生成结果表格与图片
python -m graph_anomaly --help
```

安装后也可使用 `graph-anomaly` 命令。其他子命令包括 `no-pair`（重复交互控制）
和 `verify`（实验产物核验）；每个命令均支持 `--help`。

## 源码说明

实现集中在 `src/graph_anomaly/`。算法与对应运行入口放在同一模块，不再使用纯转发脚本。

| 文件 | 职责 |
|---|---|
| `config.py` | 实验标识、参数常量及可迁移的项目路径。 |
| `data.py` | 数据加载、时间划分、缓存校验和特征列对齐。 |
| `features.py` | 严格历史状态与 TGF、局部、时间特征（报告 3.1、3.3 节）。 |
| `generation.py` | G0–G3 候选筛选与时间批次生成（3.2 节）。 |
| `models.py` | RF/LR 模型及验证集参数选择（4.2 节）。 |
| `metrics.py` | AP、PR-AUC、ROC-AUC 与 top-K 指标（4.2 节）。 |
| `evaluation.py` | 模型保存与恢复、同生成器/跨生成器评估、消融（第 5 节）。 |
| `controls.py` | 移除 prior-pair 特征的控制实验（5.6 节）。 |
| `official.py` | 上游代码准备、官方复现与特征观察时机对照（5.1 节）。 |
| `results.py` | 汇总实验记录，生成 CSV/JSON 表格与 PNG/PDF 图片。 |
| `verify.py` | 独立核验预测、缓存来源、样本数量及汇总。 |
| `artifacts.py` | 原子文件读写和源码指纹。 |
| `cli.py` | 命令分发与完整实验调度。 |
| `__main__.py` | `python -m graph_anomaly` 的入口。 |
| `__init__.py` | 包初始化。 |

测试文件：`test_features.py` 检查历史与特征；`test_generators.py` 检查采样和因果缓存；
`test_evaluation.py` 检查指标、训练和断点恢复；`test_pair_novelty.py` 检查控制实验；
`test_portability.py` 检查目录迁移和命令入口。实验配置、打包配置及依赖分别位于
`configs/experiment.json`、`pyproject.toml` 和 `requirements.txt`。

`results/official/fresh_pipeline/` 保留上游脚本快照：`bip_check.sh`、`check.sh` 校验输入；
`bip_build_injectable.sh`、`build_injectable.sh` 构造候选；各数据集的 `inject.sh` 和
`random_sample.py` 执行原始注入与采样。

## 数据、结果与日志

`data/manifest.json` 记录准确的数据来源与校验值。清洗数据及预处理中间数据保存在
`results/official/fresh_pipeline/`；`prepare` 会将同版本输入下载至 `vendor/TGF/`。
实验结果位于 `results/evaluation/`、`official/`、`no_pair/`、`report/` 和 `validation/`。
原有日志全部保留，包括 `results/logs/`；失败运行记录保留在 `results/failures/`。

大型主实验缓存、扩展模型及主实验预测需重新生成。建议使用独立副本运行新实验；
历史结果保留原始指纹，源码变化后应重建缓存。`verify --available-only` 会说明
当前保存产物可核验的范围。

迁移目录：`python -m graph_anomaly --root "项目路径" 子命令`。
环境变量覆盖项：`GRAPH_ANOMALY_ROOT`、`GRAPH_ANOMALY_BASH`。

## 测试及已验证系统

```sh
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

已验证环境：**64 位 Windows build 26200、Python 3.11.9**，NumPy 1.24.3、
pandas 1.5.3、scikit-learn 1.3.0。检查覆盖单元/迁移测试、打包、包含空格路径下的
安装运行、已有预测核验、结果生成及真实 UCI G0/RF 流程。重构时不重跑完整实验矩阵。
缺少上游参考代码或 Bash 的测试会明确跳过。已配置 Linux/macOS CI，但未在本地验证。

上游：[TGF](https://github.com/StephanyRajeh/TGF)，GPL-2.0，固定提交
`af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b`。
