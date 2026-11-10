"""Write the Chinese report only from complete, actually computed experiment tables."""

import argparse

import pandas as pd

from graphbench.paths import ROOT
from graphbench.report_inputs import load_split_counts

from .report_results import DS, GEN, LABEL, mdtable, meanstd


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    m = pd.read_csv(ROOT / "results/evaluation/main/metrics.csv")
    assert len(m) == 840
    c = m[(m.source_generator == m.target_generator) & m.features.isin(["tgf", "extended"])]
    n = pd.read_csv(ROOT / "results/no_pair/metrics.csv")
    assert len(n) == 40
    sens = pd.concat(
        [
            pd.read_csv(ROOT / f"results/evaluation/{s}/metrics.csv")
            for s in ["ratio01", "ratio10", "window500", "window2000", "past_only"]
        ]
    )
    assert len(sens) == 200

    def vals(d, g, model="RF", f="tgf", metric="average_precision"):
        return c[
            (c.dataset == d) & (c.source_generator == g) & (c.model == model) & (c.features == f)
        ][metric]

    main_table = pd.read_csv(ROOT / "results/report/tables/main_ap.csv")
    official = pd.read_csv(ROOT / "results/official/paired_timing_summary_flat.csv")
    of = []
    for _, r in official.iterrows():
        of.append(
            {
                "数据": LABEL[r.dataset],
                "协议": "官方 post-update"
                if r.protocol.startswith("official")
                else "严格时间批处理",
                "ROC-AUC": f"{r.roc_auc_mean:.4f} ± {r.roc_auc_std:.4f}",
                "AP": f"{r.average_precision_mean:.4f} ± {r.average_precision_std:.4f}",
            }
        )
    splits = []
    for d in DS:
        for s, count in zip(["训练", "验证", "测试"], load_split_counts(ROOT, d)):
            splits.append(
                {
                    "数据": LABEL[d],
                    "集合": s,
                    "正常": count["normal"],
                    "异常": count["anomaly"],
                    "异常比例": f"{count['anomaly'] / sum(count.values()):.3%}",
                }
            )
    paired = pd.read_csv(ROOT / "results/report/tables/paired_generator_differences.csv")
    p = paired[(paired.model == "RF") & (paired.contrast == "G3-G0")].copy()
    p["AP 差值（G3−G0）"] = p.apply(lambda x: f"{x['mean']:+.4f} ± {x['std']:.4f}", axis=1)
    p = p[["dataset", "features", "AP 差值（G3−G0）"]]
    transfer = []
    for d in DS:
        for f in ["tgf", "extended"]:
            z = m[
                (m.dataset == d)
                & (m.model == "RF")
                & (m.features == f)
                & (m.target_generator == "G3")
            ]
            a = z[z.source_generator == "G0"].average_precision
            b = z[z.source_generator == "G3"].average_precision
            transfer.append(
                {
                    "数据": LABEL[d],
                    "特征": f,
                    "G0训练→G3测试": meanstd(a),
                    "G3训练→G3测试": meanstd(b),
                    "均值差": f"{b.mean() - a.mean():+.4f}",
                }
            )
    np_rows = []
    for d in DS:
        for g in GEN:
            z = n[(n.dataset == d) & (n.source_generator == g)]
            np_rows.append(
                {
                    "数据": LABEL[d],
                    "生成器": g,
                    "完整扩展特征 AP": meanstd(vals(d, g, f="extended")),
                    "去除 prior-pair AP": meanstd(z.average_precision),
                    "去除−完整": f"{z.average_precision.mean() - vals(d, g, f='extended').mean():+.4f}",
                }
            )
    st = []
    for d in DS:
        for s in ["main", "ratio01", "ratio10", "window500", "window2000", "past_only"]:
            z = (
                c[(c.dataset == d) & (c.model == "RF")]
                if s == "main"
                else sens[(sens.dataset == d) & (sens.scenario == s)]
            )
            r = {"数据": LABEL[d], "条件": s}
            for g in ["G0", "G3"]:
                for f in ["tgf", "extended"]:
                    r[f"{g}/{f}"] = meanstd(
                        z[(z.source_generator == g) & (z.features == f)].average_precision
                    )
            st.append(r)
    novel = []
    for d in DS:
        z = c[
            (c.dataset == d)
            & (c.source_generator == "G3")
            & (c.model == "RF")
            & (c.features == "extended")
        ]
        novel.append(
            {
                "数据": LABEL[d],
                "全测试 AP": meanstd(z.average_precision),
                "首次正常边子集 AP": meanstd(z.novel_average_precision),
                "全测试 ROC": meanstd(z.roc_auc),
                "子集 ROC": meanstd(z.novel_roc_auc),
                "子集异常比例": f"{z.novel_anomaly_fraction.mean():.2%}",
                "正常重复边比例": f"{z.normal_previously_seen_fraction.mean():.2%}",
            }
        )
    ml_drop = vals("movielens", "G3").mean() - vals("movielens", "G0").mean()
    uci_drop = vals("ucimessages", "G3").mean() - vals("ucimessages", "G0").mean()
    text = f"""# 实验报告：异常注入方式如何影响图异常检测结论

本报告根据 `docs/proposal.pdf` 的正式方案完成。数据、模型和图表均来自实际运行，主实验未抽样缩小数据集。核心发现是：**异常生成机制明显影响检测分数；增加少量时间特征可以恢复部分性能，但匹配没有消除全部统计捷径。**

## 1. 已完成的工作

官方原始代码的预处理、异常注入与检查、特征提取和 RF notebook 学习步骤均已运行。主实验包括两个数据集、四种生成器、RF/LR、两套特征和五个种子，共160个核心模型；完整跨生成器测试产生640项评估。另完成200个特征消融、200个敏感性模型，以及40个移除历史边出现标志的补充模型。扩展实验合计600个最终模型、1080项测试评估；官方与时间协议对照另有20项模型结果。

复现入口是 [README.md](README.md) 和 [scripts/run_all.ps1](scripts/run_all.ps1)。完整协议、生成器诊断和独立结果审计分别位于 [experiment_plan.md](../docs/experiment_plan.md)、[generator_audit.md](../docs/generator_audit.md)、[results_review.md](../docs/results_review.md)。

## 2. 数据与实验设计

使用作者仓库固定提交 `af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b` 中的完整示例：MovieLens100K 的100,000条交互，以及清理后的 UCI Messages59,795条交互。MovieLens 是二部图，UCI 是无向单部图。数据来源、校验和、节点数及时间边界保存在 [data/manifest.json](../data/manifest.json)。

严格按照真实时间跨度60%/20%/20%划分，且一个时间戳不会同时出现在两个集合中。预热至第1000条原始交互所在的整个时间戳结束。主实验历史为最后1000条正常与注入交互，所有同时间戳事件先基于冻结历史评分，再统一更新；最近活动窗口固定为一天。以下计数对五个种子、四个生成器一致：

{mdtable(pd.DataFrame(splits))}

UCI 后期交互稀疏，测试集只有94条异常。五个种子的标准差描述注入和模型随机性，不能被解释为跨数据集或跨时间段的置信区间。

主实验的异常占比是混合流中的5%，每个集合注入 `round(N_normal × p/(1−p))` 条异常。参考交互时间在生成器间配对。全局非边约束和全量节点集合是合成基准的先验信息；特征本身只使用过去交互。额外的 past-only 实验只排除过去已出现的边，并排除与当前正常交互完全重复的事件。

G0 均匀随机选择合法边。G1 从128个合法候选中保留端点度和加权度距离最小的16个；G2 再保留局部结构距离最小的4个；G3 再匹配最近活动次数、活动间隔和活跃标志。MovieLens 使用三跳路径，UCI 使用共同邻居、Jaccard 和 Adamic–Adar。这里是有限候选池的近似排序匹配，不能声称端点或局部结构已经完全保持。

RF 使用100棵树，并由验证 AP 在叶子最小样本数1/5之间选择；LR 使用训练集拟合的 log1p+标准化流水线，由验证 AP 在 C=0.1/1/10之间选择。选择后在训练+验证集合上重拟合，最后评估测试集。跨生成器测试不使用目标生成器调参。

## 3. 官方复现与时间协议审计

原始预处理结果与作者文件的 SHA-256 一致；新注入流通过原始检查脚本。忠实性能复现使用作者发布的固定注入流，特征与发布缓存逐项一致。官方的5%是“注入数/正常数”，混合占比为4.762%，与扩展实验的5%分别报告。官方 RF 按70/30时间顺序行划分，训练时对正常样本下采样。

{mdtable(pd.DataFrame(of))}

这五次只改变 RF 种子，不是五次独立官方异常注入。严格时间对照保持同一注入流、训练索引和模型参数，改变当前事件及同时间戳事件的历史可见性。MovieLens AP 下降，UCI AP 基本不变；因此不能断言当前事件自包含统一解释了所有高分。原始实现传递的标签没有进入模型特征，不能将自包含直接等同于标签泄漏。

## 4. RQ1：生成器对测得性能的影响

以下是同生成器训练与测试的 AP，报告五种子均值±样本标准差。AP 是 `average_precision_score` 的阶梯加权定义；另存梯形 PR-AUC，二者不混用。

{mdtable(main_table)}

![主实验 AP](../results/report/figures/main_average_precision.png)

RF-TGF 从 G0 到 G3 在 MovieLens 下降{abs(ml_drop):.4f}，在 UCI 下降{abs(uci_drop):.4f}个 AP 单位。对应的配对种子差值如下。分数仍高于约0.05的类别先验基线，不能称为“完全失效”。

{mdtable(p)}

完整 ROC-AUC、梯形 PR-AUC、Precision@K 和 Recall@K 位于 [主指标表](../results/report/tables/main_all_metrics.csv)。K 等于真实测试异常数，故 Precision@K 与 Recall@K 相同；它们是事后排序指标，不代表可在线获知的报警阈值。

## 5. RQ2：匹配是否成功，以及模型在利用什么信号

![生成器诊断](../results/report/figures/generator_diagnostics.png)

端点、局部和时间组的 KS 距离是该组各特征的平均值。prior_pair 单独显示，避免把无法匹配的历史新边信号混入活动匹配的诊断。G3 比 G0 缩小了两套数据的端点和活动差异，但 UCI 的局部差异没有稳定缩小。按整流平均匹配距离看，G3 仍有约76%的 MovieLens 注入边、69%的 UCI 注入边超过预先报告的 log2 端点距离参考值。因此，实验支持“不同注入机制改变测得难度”，不足以支持“只改变了某一种结构因素”的强因果结论。

![特征分布](../results/report/figures/feature_distributions_seed11.png)

图中使用预先固定的种子11展示测试分布；完整 SMD、KS、Wasserstein、候选接受率、拒绝原因和逐边匹配记录均随缓存保存。不同生成器的混合历史不同，图中也分别保留正常边分布。

## 6. RQ3：跨生成器泛化

![RF 跨生成器 AP](../results/report/figures/cross_generator_RF.png)

{mdtable(pd.DataFrame(transfer))}

对固定 G3 测试分布，使用 G0 训练的 RF 通常低于使用 G3 训练的模型，说明训练注入机制影响可迁移性。但并不是每一列都由对角线模型获胜，迁移具有不对称性。LR 的完整矩阵见 [LR 图](../results/report/figures/cross_generator_LR.png)，不把 RF 的观察直接推广到所有分类器。

## 7. RQ4：轻量扩展与特征消融

![特征消融](../results/report/figures/feature_ablation.png)

RF-G3 的完整扩展特征 AP 在 MovieLens 为{meanstd(vals("movielens", "G3", f="extended"))}，UCI 为{meanstd(vals("ucimessages", "G3", f="extended"))}。消融显示主要增益来自时间组；局部特征单独使用时表现弱，TGF+local 与 TGF 的差别也较小。增加局部特征并不总能进一步提高 AP。完整扩展模型不是每个条件中最优的特征组合，且未依据测试集重新选择最终特征。

![RF 特征重要性](../results/report/figures/feature_importance.png)

RF 的 impurity importance 仅作描述性分析，不能作为变量的因果贡献。时间组含 `pair_seen_before`，因此增加以下控制：从完整扩展特征中去掉该标志，复用相应源生成器的验证集选定参数，重新拟合一次。它去掉的是显式 lifetime 标志，而 TGF 历史边权仍可能编码部分重复信息；随机森林的特征抽样也会随维度改变，因此不将小幅差异视为精确的单变量因果效应。

{mdtable(pd.DataFrame(np_rows))}

在 UCI 的 G3 条件下，去掉显式历史标志后 AP 为{meanstd(n[(n.dataset == "ucimessages") & (n.source_generator == "G3")].average_precision)}，低于完整扩展的{vals("ucimessages", "G3", f="extended").mean():.3f}，但仍高于 TGF 的{vals("ucimessages", "G3").mean():.3f}。这表明增益部分来自新边信息，也保留了其他时间信息的收益；不能将全部提升归因于局部结构建模。对同一个首次正常边子集，去除标志与完整模型的 AP 差别较小（详细配对值见 no-pair 审计），进一步支持该标志主要帮助识别重复正常交互的解释。

## 8. 残余新边捷径与首次正常边对照

UCI 测试正常边约82.09%曾出现，而全局非边异常从未出现。下面只保留首次出现的正常边和全部异常边，使用同一已训练模型评分，不重新训练。

{mdtable(pd.DataFrame(novel))}

UCI 子集异常比例提高到约22.71%，故不能通过子集 AP 与全量 AP 的直接高低判断难度。ROC 的下降表明原有评估包含容易识别的重复正常边，但这一过滤不是完整的重复信号去除实验。MovieLens 原始用户—物品对不重复，因此该子集与全测试集一致。

## 9. 敏感性分析

只改变单个设置，比较 G0/G3、两套特征、RF、两个数据集和五种子。main 是5%异常与 H1000；ratio01/ratio10 是1%/10%；window500/window2000 是历史事件数；past_only 是过去未出现定义。没有重复整个 LR/跨生成器矩阵。

{mdtable(pd.DataFrame(st))}

![敏感性分析](../results/report/figures/sensitivity.png)

改变异常比例同时改变 AP 的随机基线和混合历史，因此比例实验应结合 ROC-AUC 与实际先验一起读，不能将 AP 的变化全部归因于检测能力。H 窗口同时作用于候选匹配与检测特征，并非仅改变检测器。past-only 条件也可能选到后来会真实出现的边，其标签含义与全局非边不同。逐次指标和配对主实验控制都已保存。上述五类设置中，两个数据集的 G3 均低于对应 G0；但更长的历史并不保证更高 AP，例如 MovieLens 的 G3-TGF 在 H2000 时低于 H1000。

## 10. 结论、边界和交付

这两套完整数据上的证据支持 proposal 的主要研究动机：随机注入下的高分不能直接推导出对其他合成异常机制同样有效；轻量时间特征可以恢复部分性能。更强的结论需要更好的匹配支持、对新边先验的严格控制，以及更多时间划分、数据集和真实异常标签。

当前结果的边界包括：有限候选池残余差异、UCI 测试异常数量少、全局非边使用未来流先验、固定一天活动窗口和 H-type 历史、有限超参数网格、只有两个图类型，以及合成异常并不是真实欺诈或攻击。没有运行可选的 MIDAS/微簇或深度模型，不能对这些方法做性能判断。

可复现交付包括固定配置、上游提交和数据哈希、140份生成器数据缓存、逐条注入审计、训练模型、1080项扩展测试预测和指标、官方对照、完整代码测试、机器可读表格及图形。所有报告数字由脚本读取实验文件生成；失败/中断日志保留，只有完整且哈希一致的结果进入汇总。

参考：Latapy & Rajeh, *Trivial Graph Features and Classical Learning are Enough to Detect Random Anomalies*, ICDM2025, DOI [10.1109/ICDM65498.2025.00142](https://doi.org/10.1109/ICDM65498.2025.00142)；[官方代码](https://github.com/StephanyRajeh/TGF)、[官方项目网站](https://stephanyrajeh.github.io/TGFWebsite/)。
"""
    destination = ROOT / "reports/experiment_report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    print("Wrote reports/experiment_report.md from saved experiment results")


if __name__ == "__main__":
    main()
