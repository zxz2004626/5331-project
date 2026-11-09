"""Build the final English experiment report from completed, saved results only.

Run with the Codex bundled Python (reportlab + pypdf). --check only validates
readiness; it does not create a PDF. This script never starts an experiment.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'results/report'
FIG = REPORT / 'figures'
TABLES = REPORT / 'tables'
OUTPUT = ROOT / 'output/pdf/experiment_report.pdf'
DS = ['movielens', 'ucimessages']
GEN = ['G0', 'G1', 'G2', 'G3']
LABEL = {'movielens': 'MovieLens 100K', 'ucimessages': 'UCI Messages'}
SCENARIOS = ['ratio01', 'ratio10', 'window500', 'window2000', 'past_only']


def csv_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def pick(rows, **criteria):
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in criteria.items())]


def values(rows, column):
    return [float(r[column]) for r in rows]


def mean(rows, column='average_precision'):
    return st.mean(values(rows, column))


def ms(rows, column='average_precision', digits=3):
    v = values(rows, column)
    assert len(v) == 5, (column, len(v))
    return f'{st.mean(v):.{digits}f} ± {st.stdev(v):.{digits}f}'


def load():
    required = [ROOT / 'results/evaluation/main/metrics.csv', ROOT / 'results/no_pair/metrics.csv',
                ROOT / 'results/official/paired_timing_metrics.csv', REPORT / 'summary.json',
                TABLES / 'diagnostic_group_statistics.csv', TABLES / 'matching_statistics.csv']
    required += [ROOT / f'results/evaluation/{s}/metrics.csv' for s in SCENARIOS]
    required += [FIG / f'{s}.png' for s in ['main_average_precision', 'cross_generator_RF',
                'cross_generator_LR', 'generator_diagnostics', 'feature_distributions_seed11',
                'feature_ablation', 'feature_importance', 'sensitivity']]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise RuntimeError('Report inputs are not complete: ' + ', '.join(missing))
    main = csv_rows(required[0])
    no_pair = csv_rows(required[1])
    official = csv_rows(required[2])
    sensitivities = {s: csv_rows(ROOT / f'results/evaluation/{s}/metrics.csv') for s in SCENARIOS}
    assert len(main) == 840, f'Expected 840 main evaluations, got {len(main)}'
    assert len(no_pair) == 40, f'Expected 40 no-pair evaluations, got {len(no_pair)}'
    assert len(official) == 20
    assert all(len(x) == 40 for x in sensitivities.values())
    for rows in [main, no_pair, official, *sensitivities.values()]:
        assert all(math.isfinite(float(r['average_precision'])) for r in rows)
    diagonal = [r for r in main if r['source_generator'] == r['target_generator']]
    return {'main': main, 'diagonal': diagonal, 'no_pair': no_pair, 'official': official,
            'sens': sensitivities, 'manifest': json.loads((ROOT / 'data/manifest.json').read_text()),
            'split_counts': {d: json.loads((ROOT / f'results/cache/main/{d}/seed_11/G0.json').read_text())['split_counts'] for d in DS},
            'diagnostic': csv_rows(TABLES / 'diagnostic_group_statistics.csv'),
            'matching': csv_rows(TABLES / 'matching_statistics.csv')}


def build(data):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
    from pypdf import PdfReader

    fonts = Path(sys.executable).parent / 'Lib/site-packages/matplotlib/mpl-data/fonts/ttf'
    regular, bold = fonts / 'DejaVuSans.ttf', fonts / 'DejaVuSans-Bold.ttf'
    if not regular.exists():
        regular, bold = Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/arialbd.ttf')
    pdfmetrics.registerFont(TTFont('ReportSans', str(regular)))
    pdfmetrics.registerFont(TTFont('ReportSans-Bold', str(bold)))
    pdfmetrics.registerFontFamily('ReportSans', normal='ReportSans', bold='ReportSans-Bold', italic='ReportSans', boldItalic='ReportSans-Bold')
    navy = colors.HexColor('#173044')
    teal = colors.HexColor('#17756f')
    pale = colors.HexColor('#eef3f5')
    gray = colors.HexColor('#4c5b65')
    W = A4[0] - 96
    styles = {
        'body': ParagraphStyle('body', fontName='ReportSans', fontSize=9.1, leading=13.3, textColor=navy, spaceAfter=8),
        'small': ParagraphStyle('small', fontName='ReportSans', fontSize=7.8, leading=11, textColor=gray, spaceAfter=6),
        'caption': ParagraphStyle('caption', fontName='ReportSans', fontSize=7.7, leading=10.8, textColor=gray, spaceAfter=10),
        'h1': ParagraphStyle('h1', fontName='ReportSans-Bold', fontSize=21, leading=26, textColor=navy, spaceAfter=14),
        'h2': ParagraphStyle('h2', fontName='ReportSans-Bold', fontSize=11.5, leading=16, textColor=teal, spaceBefore=8, spaceAfter=7),
        'kicker': ParagraphStyle('kicker', fontName='ReportSans-Bold', fontSize=8.3, leading=12, textColor=teal, spaceAfter=8),
        'cell': ParagraphStyle('cell', fontName='ReportSans', fontSize=7.65, leading=10.1, textColor=navy),
        'headcell': ParagraphStyle('headcell', fontName='ReportSans-Bold', fontSize=7.65, leading=10.1, textColor=colors.white),
        'cover': ParagraphStyle('cover', fontName='ReportSans-Bold', fontSize=28, leading=34, textColor=navy, spaceAfter=16),
    }
    story = []
    section = 0

    def p(text, style='body'):
        story.append(Paragraph(text, styles[style]))

    def title(number, name, eyebrow='EXPERIMENT REPORT'):
        nonlocal section
        if section:
            story.append(PageBreak())
        section += 1
        p(f'{number:02d} / {eyebrow}', 'kicker')
        p(name, 'h1')

    def table(headers, rows, widths=None, small=False):
        widths = widths or [W / len(headers)] * len(headers)
        cells = [[Paragraph(html.escape(str(x)), styles['headcell']) for x in headers]]
        cells += [[Paragraph(html.escape(str(x)), styles['cell']) for x in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), navy), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, pale]),
            ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 6 if not small else 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6 if not small else 4),
            ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor('#c9d3d8')),
        ]))
        story.extend([t, Spacer(1, 10)])

    def figure(name, caption, maxheight=320):
        from PIL import Image as PILImage
        path = FIG / f'{name}.png'
        with PILImage.open(path) as im:
            iw, ih = im.size
        scale = min(W / iw, maxheight / ih)
        story.append(Image(str(path), width=iw * scale, height=ih * scale, hAlign='CENTER'))
        story.append(Spacer(1, 5))
        p(caption, 'caption')

    def core(d, g, m='RF', f='tgf'):
        return pick(data['diagonal'], dataset=d, source_generator=g, model=m, features=f)

    def contrast(d, m='RF', f='tgf'):
        return mean(core(d, 'G3', m, f)) - mean(core(d, 'G0', m, f))

    def paired_delta(a, b):
        aa = {r['seed']: float(r['average_precision']) for r in a}
        return st.mean(float(r['average_precision']) - aa[r['seed']] for r in b)

    # 1: concise, data-driven cover and findings.
    title(1, 'Beyond Random Anomalies', 'COMP5331 / RESEARCH PROJECT')
    p('Graph anomaly detection under<br/>ranked, approximately matched injections', 'cover')
    p('Completed experiments on MovieLens 100K and UCI Messages<br/>September 2026 | Reproducible experimental report', 'small')
    p('Research question', 'h2')
    p('Do simple graph features remain sufficient when synthetic anomalous links are made closer to normal interactions in endpoint activity, local structure and temporal context? This study treats the injection procedure as an experimental variable and preserves a separate, faithful reproduction of the released TGF baseline [1, 2].')
    rows = []
    for d in DS:
        rows.append([LABEL[d], f'{mean(core(d,"G0")):.3f}', f'{mean(core(d,"G3")):.3f}',
                     f'{mean(core(d,"G3",f="extended")):.3f}', f'{contrast(d):+.3f}'])
    table(['Dataset', 'G0 / RF TGF', 'G3 / RF TGF', 'G3 / RF extended', 'G3 - G0 / TGF'], rows,
          [125, 91, 91, 105, W-412])
    p('Table 1. Mean test average precision over five paired injection/model seeds. G0 is a random legal non-edge; G3 applies endpoint, local and temporal candidate rankings. AP is a ranking metric, not classification accuracy.', 'caption')
    p('The injection procedure changes measured difficulty', 'h2')
    p(f'RF with TGF loses {-contrast("movielens"):.3f} AP on MovieLens and {-contrast("ucimessages"):.3f} on UCI from G0 to G3. Additional local and temporal features recover part of the G3 performance. The generators nevertheless retain substantial distribution gaps, and UCI contains an unmatched repeated-pair shortcut. These results support sensitivity to benchmark construction, rather than a claim of realistic fraud detection.')
    p('Coverage: 840 main evaluations, 200 sensitivity evaluations and 40 prior-pair feature ablations; plus 20 official-protocol/timing fits. All numerical tables and figures in this document are loaded from saved outputs.', 'small')

    # 2: dataset and causal chronology.
    title(2, 'Data and temporal evaluation')
    table(['Property', 'MovieLens 100K', 'UCI Messages'], [
        ['Observed events', '100,000', '59,795'], ['Nodes', '1,682 movies + 943 users', '1,899 users'],
        ['Unique observed pairs', '100,000', '13,838'], ['Graph type', 'Bipartite, movie then user', 'Undirected, repeated events'],
        ['Distinct timestamps', '49,282', '58,911'],
    ], [150, (W-150)/2, (W-150)/2])
    p('The complete cleaned examples are pinned to the TGF repository commit. UCI raw input contains 59,835 messages; official cleaning removes duplicated timestamp/pair events. Ratings and message direction are not predictor inputs. Source URLs and SHA-256 values are retained in data/manifest.json [2-4].')
    table(['Dataset', 'Train normals', 'Validation normals', 'Test normals', 'Test anomalies / mixed rows'], [
        [LABEL[d], data['split_counts'][d][0]['normal'],data['split_counts'][d][1]['normal'],
         data['split_counts'][d][2]['normal'],
         f'{int(float(core(d,"G0")[0]["n_anomalies"]))} / {int(float(core(d,"G0")[0]["n_test"]))}'] for d in DS],
         [115, 85, 99, 84, W-383])
    p('Table 2. Actual main-condition counts after warm-up. Split boundaries are 60% and 80% of elapsed observation time, not event-count quantiles. The late UCI period is sparse, making its test sample much smaller.', 'caption')
    p('Score first, then update the complete timestamp batch', 'h2')
    p('All features for time t are computed from events with timestamps strictly less than t. Normal and synthetic events at t are then inserted into a shared mixed history. H=1000 retains the last 1,000 events; recent activity uses an independent 86,400-second window. Warm-up includes the entire timestamp containing observed event 1,000. Earlier unlabeled test events can update later test history, as in a stream.')
    p('The main generator requests round(N_normal x 0.05 / 0.95) anomalies separately within each split. Seeds 11, 23, 37, 51 and 71 share reference timestamp schedules across G0-G3. Main pair exclusion uses the full observed pair set, an explicit offline construction oracle. The feature history itself remains causal.')
    p('Model selection uses validation AP only. Selected models are refit on train plus validation before final testing. The source-generator model is fixed when applied to other target generators; target data never selects its configuration.')

    # 3: generators and model definitions.
    title(3, 'Generators, features and classifiers')
    table(['Condition', 'Candidate selection rule'], [
        ['G0 / random', 'Uniformly sample one legal, previously unobserved pair.'],
        ['G1 / endpoint ranking', 'Draw 128 legal pairs; retain the 16 nearest reference endpoint log-degree/strength profiles; choose randomly.'],
        ['G2 / local ranking', 'From the endpoint 16, retain 4 nearest local profiles; choose randomly.'],
        ['G3 / temporal ranking', 'From the local 4, choose the nearest recent-activity/recency profile.'],
    ], [125, W-125])
    p('Distances are fixed RMS distances after prescribed transformations. Local and temporal ties use seeded randomness. These are finite-pool nearest-rank controls (128 -> 16 -> 4), not exact matching or hard tolerance guarantees. The chosen streams subsequently have different mixed histories; population-level closeness need not improve monotonically.')
    table(['Feature group', 'Definition and scope'], [
        ['TGF (44 / 33)', 'Active graph summaries; endpoint degrees and strengths; pair weights; equal-value and strictly-greater-value counts. Bipartite / unipartite counts shown.'],
        ['Local (2 / 3)', 'MovieLens: simple length-three path count and degree-normalized path count. UCI: common neighbors, Jaccard, Adamic-Adar.'],
        ['Temporal (7)', 'Endpoint counts in the prior day, capped log recencies, recent-active flags, and lifetime pair_seen_before.'],
        ['Extended (53 / 43)', 'TGF + local + temporal. Ordinary common neighbors are not used across MovieLens partitions because they are identically zero.'],
    ], [125, W-125])
    p('Random Forest', 'h2')
    p('100 trees, balanced_subsample class weights, validation choice of min_samples_leaf in {1, 5}, other recorded scikit-learn defaults. Logistic regression uses nonnegative log1p features, a training-fitted StandardScaler, balanced class weights, liblinear and validation choice of C in {0.1, 1, 10}.')
    p('The main factorial fits 160 final models (2 datasets x 4 generators x 2 feature sets x 2 classifiers x 5 seeds), each tested on four generators. Five RF feature-group ablations add 200 models. Mean +/- sample SD measures conditional seed variation, not population uncertainty across datasets or independent time folds.', 'small')

    # 4: verified official baseline.
    title(4, 'Faithful official reproduction')
    rows = []
    for d in DS:
        for proto, short in [('official_post_update_H1000','Official after arrival'), ('strict_timestamp_causal_pre_update_H1000','Strict prior timestamp')]:
            rr=pick(data['official'],dataset=d,protocol=proto)
            rows.append([LABEL[d],short,ms(rr,'roc_auc',4),ms(rr,'average_precision',4)])
    table(['Dataset', 'Feature timing', 'ROC-AUC', 'Average precision'], rows, [103,135,130,W-368])
    p('Table 3. Five RF seeds (42, 0, 1, 2, 3), with a fixed released injection and fixed training split. These SDs describe RF randomness only. The original notebook uses a chronological 70/30 row split, balanced normal downsampling and 100-tree RF; it does not randomly split train and test.', 'caption')
    p('Exact released-feature agreement', 'h2')
    p('The original generator was run without source edits. All 105,000 MovieLens rows and 62,784 UCI rows match the published feature cache after excluding the runtime cost field. Original learning cells execute unchanged; seed-42 ROC-AUC reproduces the notebook values 0.9880 and 0.9635. The first 1,000 featureless rows are removed, leaving 44 and 33 predictors.')
    p('Raw preprocessing and injection also executed', 'h2')
    p('All original raw-data preprocessing notebook cells were executed in isolated output directories. Both cleaned files are byte-identical to the committed versions. Original shell injection and checking passed for 100,000 + 5,000 MovieLens events and 59,795 + 2,989 UCI events. Git-blob LF files and a Windows stdout newline adapter preserve the original algorithm. The unseeded fresh injection is retained but does not replace the fixed stream used for faithful metric comparison.')
    p('Observation-time sensitivity', 'h2')
    p('Official features include the current event and earlier rows at the same timestamp. The paired alternative freezes each timestamp group and keeps the same event stream, row split, training indices and RF settings. MovieLens AP falls by 0.0653; UCI changes by +0.0021. This contrast changes both current-event and tied-event visibility. It is not evidence of label leakage, nor does it explain all random-injection discrimination.')
    p('Official 5% means anomalies / normals, giving about 4.762% mixed prevalence. Its injection excludes duplicate triples, whereas the main causal benchmark excludes globally observed pairs and uses 5% mixed prevalence. Therefore their scores are separate protocol results, not a direct method ranking.', 'small')

    # 5: main table and chart.
    title(5, 'Main result: injection changes difficulty')
    figure('main_average_precision','Figure 1. Diagonal tests train and test on the same generator. Error bars are sample SD over five seeds. The dotted 5% line marks main-condition prevalence, the random-ranking reference.',maxheight=270)
    rows=[]
    for d in DS:
        for g in GEN:
            rows.append([LABEL[d] if g=='G0' else '',g,ms(core(d,g)),ms(core(d,g,f='extended')),
                         ms(core(d,g,m='LR')),ms(core(d,g,m='LR',f='extended'))])
    table(['Dataset','Gen.','RF / TGF','RF / extended','LR / TGF','LR / extended'],rows,[105,34,88,94,88,W-409],small=True)
    p('Table 4. Average precision, mean +/- sample SD. AP is computed by average_precision_score; trapezoidal PR-AUC is saved separately. Different classifiers and feature groups need not follow identical trends. The full paired per-seed contrasts are supplied with the report tables.', 'caption')

    # 6 and 7: cross-generator tests.
    for page,m in [(6,'RF'),(7,'LR')]:
        title(page,f'Cross-generator transfer: {m}')
        figure(f'cross_generator_{m}',f'Figure {page-4}. Mean AP over five seeds. Rows identify the training generator; columns identify the test generator. Each row applies one fixed fitted model per seed, so off-diagonal differences are evaluation-distribution changes, not target-specific retraining.',maxheight=390)
        rows=[]
        for d in DS:
            for f in ['tgf','extended']:
                g0g0=pick(data['main'],dataset=d,source_generator='G0',target_generator='G0',model=m,features=f)
                g0g3=pick(data['main'],dataset=d,source_generator='G0',target_generator='G3',model=m,features=f)
                g3g3=core(d,'G3',m,f)
                rows.append([LABEL[d],f,f'{mean(g0g0):.3f}',f'{mean(g0g3):.3f}',f'{mean(g3g3):.3f}'])
        table(['Dataset','Features','G0 -> G0','G0 -> G3','G3 -> G3'],rows,[125,95,93,93,W-406],small=True)
        p('Table '+str(page-1)+'. Reading one controlled transfer contrast. G0 -> G3 tests a random-injection-trained model on G3; G3 -> G3 also adapts training to G3. The heatmaps make possible asymmetry visible. Numeric annotations keep the matrices interpretable in grayscale.', 'caption')

    # 8: diagnostics, distribution plots.
    title(8,'How closely were anomalies matched?')
    figure('generator_diagnostics','Figure 4. Mean two-sample KS statistic within feature groups, then mean +/- SD across seeds. Lower values indicate smaller marginal normal/anomaly gaps; the prior-pair indicator is shown separately.',maxheight=180)
    figure('feature_distributions_seed11','Figure 5. Test empirical CDFs for one fixed seed (11), comparing G0 and G3. Degree and local-count axes use log(1+x); recency is already transformed. Solid/dashed normal curves reflect different mixed histories. CDFs are illustrative, not additional independent replicates.',maxheight=278)
    ep=[]
    for d in DS:
        g=pick(data['matching'],dataset=d,generator='G3')
        ep.append(f'{LABEL[d]} {100*mean(g,"endpoint_distance_over_log2_fraction"):.1f}%')
    p('Matching is incomplete: the fraction of G3 injections whose endpoint RMS log-distance exceeds log(2) is '+', '.join(ep)+'. This threshold is a diagnostic, not an enforced caliper. Substantial degree/activity separation remains; UCI local marginal gaps need not shrink from G0 to G3. Avoid describing these conditions as perfectly structure-preserving.', 'small')

    # 9: ablation and importance.
    title(9,'Which features carry the signal?')
    figure('feature_ablation','Figure 6. RF feature-group ablations, five-seed mean AP. Local-only, temporal-only and local+temporal test feature sufficiency; TGF+local and TGF+temporal isolate additions to the baseline.',maxheight=208)
    figure('feature_importance','Figure 7. Mean RF impurity importance for the extended feature set. The displayed features are the largest across the generators within each dataset. Importances are descriptive and can be biased by correlated or high-cardinality predictors.',maxheight=219)
    gains=[]
    for d in DS:
        gain=paired_delta(core(d,'G3'),core(d,'G3',f='extended'))
        gains.append(f'{LABEL[d]} {gain:+.3f}')
    p('On G3, adding all local and temporal features changes RF AP by '+', '.join(gains)+'. This gain is a feature-set comparison under the same seed and generator. UCI interpretation requires the prior-pair controls on page 11 because global non-edge injection makes all synthetic pairs novel.')
    recovery=[]
    for d in DS:
        recovery.append(f'{LABEL[d]}: TGF+temporal {mean(core(d,"G3",f="tgf_temporal")):.3f}, TGF+local {mean(core(d,"G3",f="tgf_local")):.3f}')
    p('The G3 group ablations locate most recovery in the temporal group ('+'; '.join(recovery)+'). Local-only performance is much lower. Adding all features need not outperform TGF+temporal.', 'small')
    p('Neither feature importance nor group ablation is a causal attribution. Correlated endpoint, pair-weight and recency statistics can substitute for one another. The benchmark supports an empirical comparison of available signals, not identification of a unique mechanism.', 'small')

    # 10: sensitivity.
    title(10,'Sensitivity to prevalence and history')
    figure('sensitivity','Figure 8. Each condition changes one main-protocol parameter: mixed anomaly fraction 1% or 10%, H=500 or 2,000, or past-only pair exclusion. RF, G0/G3 and both feature sets are repeated over all five seeds on both datasets.',maxheight=330)
    rows=[]
    for d in DS:
        for f in ['tgf','extended']:
            base=mean(core(d,'G3',f=f))
            history=[mean(pick(data['sens'][s],dataset=d,source_generator='G3',features=f)) for s in ['window500','window2000']]
            past=mean(pick(data['sens']['past_only'],dataset=d,source_generator='G3',features=f))
            rows.append([LABEL[d],f,f'{base:.3f}',f'{history[0]:.3f}',f'{history[1]:.3f}',f'{past:.3f}'])
    table(['Dataset','Features','Main G3','H500 G3','H2000 G3','Past-only G3'],rows,[111,83,76,76,76,W-422],small=True)
    p('Table 7. Selected G3 mean AP sensitivity values. Past-only exclusion removes the full-future-pair absence constraint but still excludes current-timestamp observed duplicates. The full sensitivity CSV also records realized prevalence and ROC-AUC.', 'caption')
    p('AP depends on prevalence. Directly comparing AP at 1%, 5% and 10% mixes prevalence with detector ranking changes; those values are not a pure difficulty scale. History-size and absence sensitivities use the main prevalence. Fixed elapsed-time splits remain unchanged, so UCI late-period sparsity persists.')

    # 11: novelty and no-pair controls.
    title(11,'Residual novelty shortcut and controls')
    repeat=mean(core('ucimessages','G0'), 'normal_previously_seen_fraction')
    p(f'In the UCI test set, {100*repeat:.2f}% of normal events have a pair previously observed in lifetime mixed history, while globally absent injected pairs are all novel. G1-G3 do not match this property. MovieLens has no corresponding repeated-normal-pair signal. This constraint can make temporal features useful without showing that recency itself recovers performance.')
    rows=[]
    for d in DS:
        for g in ['G0','G3']:
            rr=core(d,g,f='extended')
            rows.append([LABEL[d],g,f'{mean(rr):.3f}',f'{mean(rr,"novel_average_precision"):.3f}',
                         f'{100*mean(rr,"novel_anomaly_fraction"):.1f}%',f'{mean(rr,"novel_n"):.0f}'])
    table(['Dataset','Gen.','Full AP','Novel-normal AP','Filtered prevalence','Filtered n'],rows,[110,35,69,103,107,W-424],small=True)
    p('Table 8. Existing RF extended predictions are reevaluated after removing repeated normal pairs. Models are not retrained. The filtered class prior changes, so a higher or lower filtered AP cannot be attributed only to difficulty. Counts and prevalence are reported alongside the metric.', 'caption')
    rows=[]
    for d in DS:
        for g in GEN:
            rr=pick(data['no_pair'],dataset=d,source_generator=g)
            original=core(d,g,f='extended')
            rows.append([LABEL[d] if g=='G0' else '',g,ms(original),ms(rr),f'{paired_delta(original,rr):+.3f}'])
    table(['Dataset','Gen.','Extended AP','Without prior-pair AP','Paired change'],rows,[110,35,125,142,W-412],small=True)
    p('Table 9. Forty additional fits remove only pair_seen_before. Each reuses the corresponding extended model\'s selected min_samples_leaf, refits on train+validation, and applies the same test split. This is a controlled feature deletion, not independently retuned feature selection. TGF still contains recent pair weight, so other novelty proxies remain.', 'caption')
    pair_drop=paired_delta(core('ucimessages','G3',f='extended'),pick(data['no_pair'],dataset='ucimessages',source_generator='G3'))
    p(f'For UCI G3, removing the explicit prior-pair bit changes AP by {pair_drop:+.3f}, confirming that this feature materially contributes under global non-edge injection. MovieLens\'s bit is constant zero; its small score changes reflect RF refitting/feature-sampling effects rather than removal of a novelty signal.', 'small')
    novel_full=mean(core('ucimessages','G3',f='extended'),'novel_average_precision')
    novel_drop=mean(pick(data['no_pair'],dataset='ucimessages',source_generator='G3'),'novel_average_precision')
    p(f'Within the UCI G3 novel-normal subset, full and no-pair AP are {novel_full:.3f} and {novel_drop:.3f}. The small change there, compared with the full-set drop, supports a role in separating repeated normal pairs from novel injected pairs. It does not establish better ranking of new normal interactions.', 'small')

    # 12: reproducibility.
    title(12,'Reproducibility and verification')
    table(['Experiment block','Final fitted models','Test evaluations'],[
        ['Core RF/LR x TGF/extended',160,640],['RF feature-group ablations',200,200],
        ['Five sensitivity conditions',200,200],['Remove pair_seen_before',40,40],
        ['Causal-extension total',600,1080],['Separate official/timing reproduction',20,20],
    ],[W-220,110,110])
    p('Hyperparameter candidate fits are additional internal validation fits, not counted as final models. The 140 cached generation conditions comprise 40 main streams and 100 sensitivity streams. Every scored event is retained; these experiments do not downsample normal observations in the causal extension.')
    p('Verification performed', 'h2')
    p('Independent tests compare original TGF formulas, active-population ranks and medians, history eviction, separate recency windows, timestamp-batch invariance, bipartite paths and legality. Generator audits verify requested class counts, unique legal injected pairs, strictly prior maximum history timestamps and reproducible cache fingerprints. Model-selection tests check source-only validation and feature alignment.')
    p('The official sensitivity outputs pass independent score-level verification: 20 prediction files, 497,350 scored rows, recomputed AP and ROC-AUC within 1e-12, paired labels and row indices equal. Fresh preprocessing has byte-for-byte source equality and original injection checks pass. Saved failures are excluded until successfully rerun; they are not substituted by fabricated scores.')
    p('Metric records also contain ROC-AUC, trapezoidal PR-AUC, Precision@K and Recall@K. K is the true test anomaly count, so precision and recall at K coincide; these are retrospective ranking summaries, not a deployment threshold selected without labels.', 'small')
    p('Artifact map', 'h2')
    table(['Location','Contents'],[
        ['data/manifest.json','Dataset sources, hashes, split boundaries and counts'],
        ['results/cache/','140 manifests, complete feature arrays and injection audit logs'],
        ['results/evaluation/','Fitted models, per-event scores, selection metadata and metrics'],
        ['results/no_pair/','Forty prior-pair deletion fits and predictions'],
        ['results/official/','Original-source reproduction, cached-feature checks and fresh pipeline'],
        ['results/report/','Regenerated figures, tables, summaries and per-seed contrasts'],
        ['docs/experiment_plan.md','Protocol details and deviations required for interpretation'],
    ],[160,W-160],small=True)
    p('Regenerate tables/figures: python scripts/report_results.py. Build this PDF using the bundled reportlab runtime: python scripts/build_pdf_report.py. Training/generation entry points are scripts/build_experiments.py and scripts/evaluate_experiments.py; saved manifests fix parameters and seeds.', 'small')

    # 13: interpretations and references.
    title(13,'Conclusions, limits and references')
    p('What the experiments establish', 'h2')
    p('The released TGF example results are reproducible. Within a fixed causal experiment, progressively ranked injections can substantially reduce TGF AP, and inexpensive additional features can recover some performance. Cross-generator testing makes transfer sensitivity visible; a detector trained on one injection distribution need not behave equivalently on another.')
    p('What they do not establish', 'h2')
    p('Synthetic labels identify generated edges, not confirmed real anomalies. Matching is approximate and finite-pool; G3 still has large endpoint/activity gaps, while UCI local gaps are not uniformly reduced. Global non-edge exclusion uses future observed pairs and leaves repeated-normal-pair signals unmatched. The past-only and prior-pair controls narrow these issues but do not remove every correlated novelty proxy.')
    p('There are only two datasets and one elapsed-time partition per dataset. UCI has few late test events; seed SD does not describe temporal-fold or deployment uncertainty. AP across changed priors is not directly comparable. The original-reproduction protocol differs from the causal extension in injection, feature timing, model weighting and splitting. Optional neural baselines, MIDAS, coordinated anomalies and real labeled fraud are outside this completed single-edge study.')
    p('References and primary sources', 'h2')
    refs=[
        '[1] M. Latapy and S. Rajeh. Trivial Graph Features and Classical Learning are Enough to Detect Random Anomalies. IEEE ICDM, 2025, pp. 1330-1339. DOI: 10.1109/ICDM65498.2025.00142. <link href="https://doi.org/10.1109/ICDM65498.2025.00142" color="#17756f">Publisher record</link>.',
        '[2] Authors\' TGF repository, pinned commit af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b. <link href="https://github.com/StephanyRajeh/TGF" color="#17756f">Source code and example datasets</link>; <link href="https://stephanyrajeh.github.io/TGFWebsite/" color="#17756f">project website</link>.',
        '[3] GroupLens Research. MovieLens 100K dataset. <link href="https://grouplens.org/datasets/movielens/100k/" color="#17756f">Original dataset page</link>. This report uses the pinned TGF/KONECT cleaned copy.',
        '[4] T. Opsahl and P. Panzarasa. Clustering in Weighted Networks. Social Networks 31(2), 2009, pp. 155-163. UCI messages are distributed as opsahl-ucsocial in KONECT and the pinned TGF repository.',
        '[5] Project artifacts: kdd_proposal.pdf (task specification); docs/experiment_plan.md (fixed protocol); docs/official_audit.md, docs/generator_audit.md and docs/features.md (implementation audits). All report numbers derive from saved CSV/JSON/NPZ experiment outputs.',
    ]
    for r in refs:p(r,'small')
    p('Metric convention: AP denotes non-interpolated average precision. Mean +/- SD uses sample SD over the stated five seeds. All feature importance and distribution diagnostics are descriptive.', 'small')

    def header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(navy)
        canvas.rect(0,A4[1]-37,A4[0],37,fill=1,stroke=0)
        canvas.setFont('ReportSans-Bold',8)
        canvas.setFillColor(colors.white)
        canvas.drawString(48,A4[1]-24,'BEYOND RANDOM ANOMALIES')
        canvas.setFont('ReportSans',7.3)
        canvas.drawRightString(A4[0]-48,A4[1]-24,'COMP5331 | EXPERIMENTAL REPORT')
        canvas.setStrokeColor(colors.HexColor('#c9d3d8'))
        canvas.line(48,42,A4[0]-48,42)
        canvas.setFont('ReportSans',7)
        canvas.setFillColor(gray)
        canvas.drawString(48,28,'Synthetic edge benchmark | Pinned data and saved results')
        canvas.drawRightString(A4[0]-48,28,f'{doc.page} / 13')
        canvas.restoreState()

    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(OUTPUT),pagesize=A4,rightMargin=48,leftMargin=48,
                          topMargin=61,bottomMargin=56,title='Beyond Random Anomalies: Experimental Report',
                          author='COMP5331 Project',subject='Reproducible TGF and structure-matched anomaly experiments')
    doc.build(story,onFirstPage=header_footer,onLaterPages=header_footer)
    reader=PdfReader(OUTPUT)
    assert len(reader.pages)==13,f'Layout overflow: expected 13 pages, got {len(reader.pages)}'
    for i,page in enumerate(reader.pages):
        text=page.extract_text() or ''
        assert len(text)>250,f'Unexpectedly empty page {i+1}'
        assert '\ufffd' not in text
    print(json.dumps({'status':'created','pages':len(reader.pages),'path':str(OUTPUT),'bytes':OUTPUT.stat().st_size}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    data=load()
    if args.check:
        print(json.dumps({'status':'ready','main':len(data['main']),'no_pair':len(data['no_pair']),
                          'sensitivity':sum(len(x) for x in data['sens'].values())}))
    else:
        build(data)


if __name__=='__main__':
    main()
