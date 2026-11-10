# 运行与迁移

## 环境

推荐 Python 3.11，支持 Python 3.10–3.11。实验依赖固定在 `requirements.txt`；测试和格式工具使用 `pip install -e ".[dev]"`，可选实验 PDF 使用 `pip install -e ".[pdf]"`。PDF 字体由 matplotlib 自带字体定位，不依赖 Windows 字体目录。

先创建虚拟环境，再安装；不要复用不兼容的全局 pandas/NumPy 环境。所有命令中的 `python` 应指向同一个虚拟环境。

## 项目定位

可编辑安装后，从任意目录执行：

```sh
python -m graphbench --root "/path/with spaces/COMP5331Project" generate --help
python -m graphbench --root "/path/with spaces/COMP5331Project" run --dry-run
```

`--root` 放在子命令之前。也可设置 `GRAPHBENCH_ROOT`。未指定时，程序从源码位置或当前目录向上寻找 `configs/experiment.json`。复制项目后，重新在新位置执行一次 `pip install -e ".[dev]"`；不要复制旧 `.venv`、`*.egg-info` 或绝对路径绑定的可编辑安装。

不安装包时可直接使用绝对脚本路径：`python /path/to/project/scripts/run_all.py --dry-run`。

## 各阶段

```sh
python -m graphbench prepare
python -m graphbench official
python -m graphbench official --paired-timing
python -m graphbench official --fresh-pipeline
python -m graphbench official --verify-output
python -m graphbench generate --workers 1
python -m graphbench evaluate --ablations --n-jobs 2
python -m graphbench no-pair --n-jobs 1
python -m graphbench verify
python -m graphbench report
python -m graphbench write-report
python -m graphbench pdf --check
```

五项敏感性实验的完整参数由 `python -m graphbench run --dry-run` 展示。`run` 是跨平台入口；PowerShell 的 `scripts/run_all.ps1` 只转发参数到同一个 Python 流程。

只查看/整理已提交结果不需要下载上游源码。`report` 在缺少完整缓存时使用已保存的诊断 CSV；不能重新计算原始 CDF 时保留已有图并显示提示。`write-report` 的划分计数可从历史 `final_audit.json` 读取。

## 官方 shell 步骤

仅 `official --fresh-pipeline` 需要 Bash、gzip 和常见 Unix 命令工具。Linux/macOS 从 PATH 找 `bash`；Windows 优先查找 Git 安装中的 Bash。特殊安装位置设置 `GRAPHBENCH_BASH` 为可执行文件路径。

路径带空格时通过参数数组传递，不拼接为 shell 代码。上游脚本使用固定 Git blobs，避免 CRLF 换行改变其行为。Python 调用使用当前环境的 `sys.executable`，不依赖某台机器的 Anaconda 路径。其他 Python 实验步骤不依赖 Bash。

## 已有结果和缓存

`results/` 中的数值和审计来自重构前真实运行，仍保留原始来源信息。重构后的源码依赖指纹会改变，不应把它们伪装成新代码生成的结果。

- 仅查看报告：直接打开 `reports/` 或 `results/report/`。
- 重绘汇总图：运行 `report`，读取已保存的运行记录与摘要。
- 完整复现：建议单独检出一份仓库，在该目录运行 `prepare` 和完整 `run`，避免混合旧产物。
- 如果已有生成缓存失效，生成器会报错；确认需要重建后使用 `generate --force`。旧指标不会因此自动获得新代码的来源标记。
- 在缺少缓存的克隆中运行严格 `verify` 会报告缺失；`verify --available-only` 仅检查可用产物，并不代表完整复现通过。

模型序列化受 Python/scikit-learn 版本影响。不要把任意版本的 joblib 文件当作可跨版本迁移格式；跨机器迁移优先保留配置、源码、输入数据及 CSV/JSON 指标，并在指定环境中重新训练。

## 测试边界

本地测试覆盖时间隔离、图统计、候选生成、模型选参、跨生成器列对齐、断点失效、路径迁移和源码换行。未下载 `vendor/TGF` 时，两项官方算术对照会明确跳过；下载后可重跑。CI 配置在三种操作系统上执行轻量测试，不会在每次提交中重跑完整研究实验。
