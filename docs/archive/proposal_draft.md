# Project Proposal

## Title

**Beyond Random Anomalies: Evaluating Graph Anomaly Detection with Structure-Preserving Anomaly Injection**

## Type of Project

**Research**

## Brief Description

### 1. Introduction

Anomaly detection on graph-structured data is an important task in data mining and knowledge discovery. Many real-world systems can naturally be represented as graphs or temporal link streams. For example, users interact with items in recommendation systems, people exchange messages in communication networks, accounts transfer money in financial networks, and computers communicate with each other in network traffic. Detecting unusual interactions in these systems is useful for identifying fraud, attacks, abnormal behavior, and unexpected events.

A major difficulty in graph anomaly detection is the lack of reliable ground-truth anomalies. Real datasets often contain no labels indicating which interactions are actually anomalous. As a result, many existing studies evaluate anomaly detection methods by artificially injecting anomalous links into otherwise normal graph data. A common approach is **random anomaly injection**, where two nodes and a timestamp are randomly sampled and a previously nonexistent link is inserted into the graph.

The recent work *Trivial Graph Features and Classical Learning are Enough to Detect Random Anomalies* by Latapy and Rajeh provides an interesting observation about this evaluation methodology [1]. Instead of using a complicated graph neural network, the authors compute simple graph features from historical graphs and train classical machine-learning models such as Random Forest. They show that this simple approach can detect randomly injected links extremely well. Their method is computationally inexpensive and interpretable, and their experiments suggest that sophisticated models may be unnecessary when anomalies are generated through random injection. The authors therefore argue that future anomaly detection research should consider more complex kinds of anomalies.

This project starts from this observation and asks a further question:

**Are simple graph features still sufficient when anomalous links are designed to preserve important structural and temporal properties of normal interactions?**

Rather than immediately developing a more complicated anomaly detector, this project investigates the anomaly-generation process itself. We hypothesize that randomly injected anomalies may introduce obvious statistical differences between normal and anomalous links. A detector may therefore achieve high accuracy partly because it recognizes artifacts of the injection procedure rather than genuinely difficult anomalous behavior.

The project will first reproduce the main experimental pipeline of Latapy and Rajeh and then replace purely random anomalies with several increasingly difficult **structure-preserving anomalies**. The objective is to determine how anomaly-injection assumptions affect measured detection performance and to identify which graph properties make anomalies easy or difficult to detect.

### 2. Background and Motivation

The baseline paper represents a temporal network as a sequence of interactions

$$
(t_i,u_i,v_i),
$$

where \(t_i\) is the timestamp and \(u_i\) and \(v_i\) are the endpoints of an interaction. The authors construct historical graphs over previous interactions and compute inexpensive graph features for each new link. A Random Forest classifier is then trained to distinguish normal links from artificially injected anomalous links. The released implementation includes preprocessing, anomaly injection, graph-feature generation, and learning/testing. It provides examples for both the bipartite MovieLens dataset and the unipartite UCI Messages dataset.

An important detail is how artificial anomalies are generated. In the released implementation, a timestamp and two nodes are randomly selected, subject to the constraint that the injected link does not already exist.

Although such random injection provides a convenient evaluation benchmark, it may generate anomalous interactions that are substantially different from realistic difficult anomalies. Consider a social network in which most users communicate within stable communities. A randomly generated link may connect two unrelated nodes with very different activity histories. Similarly, in a recommender system, randomly connecting a user to an arbitrary item may select an item whose popularity and historical interaction pattern are very different from those normally associated with that user.

Simple graph features such as node degree, previous interactions, neighborhood structure, or recent activity may easily separate these randomly generated links from real links. High anomaly-detection accuracy in this setting does not necessarily imply equally strong performance against anomalies that deliberately resemble normal interactions.

This motivates the central idea of this project: instead of treating anomaly injection as a fixed preprocessing step, we will treat it as an experimental variable.

### 3. Research Questions

The project will investigate three main research questions.

**RQ1: How much does anomaly-detection performance depend on the random anomaly-injection strategy?**

We will first reproduce the baseline results using randomly injected anomalies and then gradually constrain the anomaly-generation process to resemble normal interactions more closely.

**RQ2: Which structural or temporal properties make an anomalous link difficult to detect?**

We will separately control properties such as endpoint degree, local graph structure, community membership, and recent activity. This allows us to measure which information contributes most strongly to anomaly detectability.

**RQ3: Can lightweight structure-aware features recover detection performance on harder anomalies?**

If the original simple features fail on certain structured anomalies, we will investigate whether a small number of additional interpretable features can improve detection without requiring a large neural model.

### 4. Proposed Methodology

The first stage of the project will reproduce the TGF pipeline. The authors provide code covering dataset preprocessing, anomaly injection, feature generation, and Random Forest-based classification. Their repository demonstrates the complete pipeline using MovieLens as a bipartite graph and UCI Messages as a unipartite graph.

We will reproduce experiments on at least these two datasets and establish a **random-injection baseline**. Metrics will include ROC-AUC and PR-AUC, with particular attention to PR-AUC because anomalous links are naturally a minority class.

The research stage will introduce a hierarchy of harder anomaly-generation strategies.

**Level 0: Random anomalies.**
This follows the original paper and serves as the control condition. Endpoints are sampled largely independently from the existing graph structure.

**Level 1: Degree- or popularity-matched anomalies.**
Instead of selecting arbitrary endpoints, anomalous links will be generated between nodes whose historical degrees or activity levels resemble those of nodes participating in normal interactions. For example, if a normal interaction involves a highly active user and a moderately popular item, the injected interaction will use nodes from similar degree ranges. This reduces the possibility that a classifier detects anomalies simply because their endpoints have unusual popularity.

**Level 2: Local-structure-matched anomalies.**
The generator will additionally consider local neighborhood statistics. Candidate anomalous links can be selected so that properties such as neighborhood size, common-neighbor statistics, or local density resemble normal interactions. The generated edge remains artificial, but its immediate structural context is less obviously abnormal.

**Level 3: Temporal-context-matched anomalies.**
The generator will further preserve temporal activity patterns. Instead of pairing arbitrary nodes at an arbitrary sampled time, it will preferentially select nodes that were active within the corresponding historical window. This tests whether temporal inactivity or unusual timing is an important shortcut used by the detector.

These levels create a **difficulty-controlled anomaly benchmark**. Increasing the number of matched properties should progressively remove simple signals that distinguish injected anomalies from genuine interactions.

An optional extension will study **coordinated or micro-cluster anomalies**, where several related anomalous edges occur within a short period. This type of behavior is motivated by streaming-graph anomaly detection methods such as MIDAS, which specifically considers suddenly appearing groups of similar suspicious edges.

### 5. Evaluation and Analysis

For each dataset and anomaly type, we will train and evaluate the same baseline classifier under controlled experimental settings. Rather than reporting only one overall accuracy number, we will analyze how performance changes as anomaly generation becomes progressively more structure-preserving.

A central experiment will therefore have the form

$$
\text{Random}
\rightarrow
\text{Degree-Matched}
\rightarrow
\text{Structure-Matched}
\rightarrow
\text{Temporal-Matched}.
$$

If ROC-AUC or PR-AUC decreases substantially along this sequence, it would provide evidence that anomaly-generation assumptions strongly influence the apparent difficulty of the benchmark.

We will also inspect feature importance from the Random Forest model. This analysis can reveal which graph features distinguish random anomalies and whether their importance changes for structured anomalies. For example, if endpoint degree dominates detection under random injection but becomes ineffective after degree matching, this provides an interpretable explanation for the performance change.

An ablation study will independently enable or disable different matching constraints. This will help determine whether degree, local structure, or temporal context contributes most strongly to anomaly difficulty.

If substantial weaknesses are found, the final stage will investigate a lightweight extension to the baseline feature set. Potential additions include community-transition features, temporal activity deviations, and richer local-neighborhood statistics. The goal is not to build a large deep-learning system, but to examine whether the identified failure mode can be addressed while maintaining the efficiency and interpretability advantages of classical graph features.

A representative streaming anomaly detector such as MIDAS may also be evaluated on compatible experiments to determine whether the observed difficulty is specific to TGF or reflects a broader limitation of anomaly benchmarks. MIDAS detects micro-cluster anomalies in edge streams and operates online with constant time and memory per edge, making it relevant to this setting.

### 6. Expected Outcomes and Research Contribution

The expected contribution of this project is not simply a new anomaly-detection model. Instead, the project studies the **evaluation methodology** used for graph anomaly detection.

There are several possible outcomes. First, the experiments may show that simple graph features remain highly effective even for carefully structured anomalies. Such a result would strengthen the conclusion of the baseline paper and demonstrate that classical graph statistics are more powerful than expected.

A second possibility is that performance decreases substantially once obvious structural differences are removed. This would suggest that random anomaly injection produces an artificially easy benchmark and that reported detection performance depends strongly on the anomaly-generation procedure.

A third possible outcome is that different anomaly types expose different weaknesses. Degree-matched anomalies may remain easy while temporally matched or neighborhood-matched anomalies may become significantly harder. Such findings would provide a more detailed understanding of what current graph features actually detect.

Based on these observations, the project aims to provide three concrete outputs:

1. a reproduction of the TGF random-anomaly experiments;
2. a configurable structure-preserving anomaly-injection framework; and
3. an empirical analysis of anomaly difficulty, feature importance, and possible lightweight improvements.

This project is feasible within the scope of a graduate data-mining course because the main baseline has publicly available code and data-processing scripts, and the proposed research primarily modifies the anomaly-generation component rather than requiring a new large-scale neural architecture. The original implementation already provides MovieLens and UCI Messages examples as well as support for historical graph construction and Random Forest training.

More broadly, the project may help clarify an important issue in knowledge discovery: a model's measured performance depends not only on the learning algorithm but also on how evaluation data are constructed. By systematically controlling anomaly structure, we hope to better distinguish genuinely difficult anomaly detection from detection of artifacts introduced by synthetic benchmark generation.

---

## Papers to Be Read

**[1] Matthieu Latapy and Stephany Rajeh.**
*Trivial Graph Features and Classical Learning are Enough to Detect Random Anomalies.*
IEEE International Conference on Data Mining (ICDM), 2025.
This is the primary paper to reproduce and extend. It demonstrates that inexpensive graph features and classical learning are highly effective for randomly injected anomalies and motivates the study of more complex anomalies.

**[2] Siddharth Bhatia, Bryan Hooi, Minji Yoon, Kijung Shin, and Christos Faloutsos.**
*MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams.*
AAAI Conference on Artificial Intelligence, 2020.
MIDAS studies online detection of groups of suspicious edges and is particularly relevant to the optional coordinated-anomaly experiments.

**[3] Dhivya Eswaran and Christos Faloutsos.**
*SedanSpot: Detecting Anomalies in Edge Streams.*
IEEE International Conference on Data Mining (ICDM), 2018.
SedanSpot detects anomalous streaming edges using both burst information and graph connectivity and provides useful background on structural signals for edge anomalies.

**[4] Dhivya Eswaran, Christos Faloutsos, Sudipto Guha, and Nina Mishra.**
*SpotLight: Detecting Anomalies in Streaming Graphs.*
ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2018.
SpotLight studies streaming-graph anomalies, including the appearance or disappearance of unusual dense structures, and provides a broader perspective on structured graph anomalies.

**[5] Wenchao Yu, Wei Cheng, Charu C. Aggarwal, Kai Zhang, Haifeng Chen, and Wei Wang.**
*NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks.*
ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2018.
NetWalk represents a different family of approaches based on dynamically updated graph embeddings and is useful for understanding how learned representations capture structural anomalies.

**[6] Li Zheng, Zhenpeng Li, Jian Li, Zhao Li, and Jun Gao.**
*AddGraph: Anomaly Detection in Dynamic Graph Using Attention-Based Temporal GCN.*
International Joint Conference on Artificial Intelligence (IJCAI), 2019.
AddGraph combines structural and temporal information using a temporal graph neural network and provides a useful comparison with feature-based approaches.

**[7] Xiaoxiao Ma, Jia Wu, Shan Xue, Jian Yang, Chuan Zhou, Quan Z. Sheng, Hui Xiong, and Leman Akoglu.**
*A Comprehensive Survey on Graph Anomaly Detection With Deep Learning.*
IEEE Transactions on Knowledge and Data Engineering, 2023.
This survey provides a taxonomy of graph anomaly detection problems, datasets, methods, evaluation practices, and open research challenges and will be used for the background and related-work sections.
