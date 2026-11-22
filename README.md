# Reward Systems- rubric judge学习笔记

## 摘要

Rubric 是一套把抽象目标拆成具体判断标准的规范，既可以指导离线数据合成，也可以作为 Agentic RL 的奖励接口。现有研究已经从“把评分标准写得更详细”，推进到“让标准可核验、随策略变化而调整，并真正促进学习”。本文围绕这一问题梳理诊断与合成、训练效用反馈、奖励聚合、在线演化、证据与过程信用等研究方向，比较各类方法的优化对象、实验结论和适用边界，并讨论训练效用识别、自然错奖修复、旧能力保持等开放问题。

## 一、问题背景

### 1.1 什么是 Rubric

Rubric 是一套**把抽象目标拆成具体判断标准的规范**，通常说明应该检查什么、什么情况下适用、依据什么证据、怎样判定满足程度，以及不同判断如何合并。

以“修改数据库中的指定记录，其他记录保持不变”为例，标准可以包括：

- **对象正确**：查询条件、对象标识与任务要求匹配
- **写入正确**：写入后读取的实际状态符合要求
- **无副作用**：操作范围与前后状态差异证明未误改其他对象
- **报告准确**：最终回复与实际执行结果一致

**Rubric 是标准，judge 是执行标准的评分者，reward 是最终进入训练的数值。** 相同 rubric 交给不同 judge，可能得到不同判断；逐项判断相同，采用不同聚合方式，也可能形成不同训练激励。

完整的 rubric system 还包括规则内容、适用条件、证据接口、判断器、聚合机制，以及规则生成、更新、选择、停用和接入训练的生命周期。因此，规则正确、证据充分、判断准确、聚合合理、训练有用，是不同层面的要求。

### 1.2 从评分标准到学习信号

“离线／在线”与“数据合成／RL”不是同一个划分：

- **离线固定**：在主要生成或训练前构建规则，使用期间基本不更新
- **在线演化**：根据当前策略的输出、错误或训练状态修改规则
- **数据合成**：指导样本生成、筛选、排序或标注，也可以生成训练 judge 的反馈
- **RL 奖励**：对当前策略的 rollout 评分，形成奖励或偏好信号

预先生成的固定 rubric 完全可以用于 on-policy RL，RaR 就是例子。[7] 即时生成规则也不等于维护了持续演化的规则系统：即时实例化、长期规则记忆和生成器参数更新是不同机制。

实际奖励可以来自已有 LLM 按显式 rubric 逐项评分，也可以来自偏好数据训练的 RM，InstructGPT 是后一类流程的代表。[22] 前者便于修改和审计，后者能吸收样本中的细粒度偏好；两者都依赖已有参数，也都可能产生误判和被策略利用。只要最终给出适用的标量奖励，都可接入 PPO、GRPO 等训练流程。

### 1.3 问题的重要性

**Rubric 同时位于“目标定义”和“学习信号”之间。** 离线合成中，它决定模型看到什么数据；RL 中，它决定哪些行为被强化。系统性的小偏差会通过数据筛选和反复策略优化被放大。

离线流程容易出现规范覆盖不足、模板与风格偏置、标签循环、过滤器误拒有效替代解法，以及评分质量与训练价值脱节。在线 RL 中，规则还可能随策略变化而饱和、全部失败、被利用或偏离原任务；旧规则退出后，相应行为也可能退化。

对 agentic RL，模型声称完成任务并不等于外部状态真的满足目标。**语言上的合理性、执行上的正确性、训练上的有效性，需要在同一个闭环中对齐。**

## 二、问题形成机制的分析

### 2.1 六个关键机制

**目标分解不完整**：抽象目标包含适用条件、例外和优先级。拆分标准时可能遗漏条件，也可能把可替代的成功路径误写成唯一流程。关键是区分**真正的任务约束**与**某一种实现策略**。

**证据不足**：工具调用被发出、工具返回成功、实际状态满足目标，是不同事实。若 judge 只看最终回答，自然语言润色无法补齐缺失的状态证据。

**聚合改变目标**：加权求和允许不同标准相互补偿，关键失败可能被其他易得分项抵消；但把所有要求都改成严格合取，也可能使奖励极其稀疏。ProRubric 展示了评分上涨但回答适当性下降的现象。[9]

**策略变化造成测量失效**：一项要求可以始终重要，但当前已经人人满足；也可以非常重要，但当前模型一次都做不到。两者都缺少组内差异。**语义重要性、当前可学习性、统计区分度，是三个不同属性。**[18]

**代理目标与真实收益存在距离**：规则合理、评分准确、当前有区分度、能指导一次答案修改、能促进多步参数学习，是五种不同结果，不能相互替代。[11]

**反馈闭环产生选择偏差**：生成器、judge、规则筛选器和最终评测若共享同一套偏好，系统可能越来越符合自己的测量方式。固定评测标准、更换评测模型、引入独立事实，分别约束不同问题。

### 2.2 核心矛盾

> **Rubric 既要准确表达稳定任务目标，又要随策略变化持续提供可学习信号；但当前最有效的测量标准，不一定是最有长期训练价值的标准。**

一个编辑可能短期帮助探索，长期却增加无效步骤；也可能前期收益不明显，随后才促进能力形成。训练初始化、采样和环境差异还会扰乱候选排序。RHyVE 对阶段与验证跨度的研究，以及 RARE 的配对续训分析，都说明真实学习过程需要直接测量。[6][13]

## 三、主要研究方向

### 3.1 诊断校准与规模化规则合成

代表工作：LLM-Rubric [1]、RIFT [2]、Prometheus 2 [3]、OpenRubrics [4]

**核心思路**：先把 rubric 当成测量工具，检查标准、权重和 judge 输出是否可靠；再用强模型或偏好对合成规则与评价反馈，降低人工成本。

**方法要点**：LLM-Rubric 使用多维人工标准、评分分布和校准映射预测人工评价；RIFT 用可控破坏和诊断探针检查规则质量。Prometheus 2 训练支持自定义标准的开放评价模型；OpenRubrics 从 chosen/rejected 对比生成规则，再按偏好标签一致性筛选。

**实验结论**：

- 校准评分和专门诊断探针在各自评价设置中优于未校准评分或直接询问模型
- 开放评价模型和合成规则在直接评分、偏好比较及奖励建模基准上取得改善

**局限性**：评价一致性不能代替策略优化后的独立能力验证；合成监督还可能继承来源模型或偏好标签的偏差。质量诊断适合做候选编辑的准入检查，训练效用仍需另外验证。

### 3.2 训练效用反馈与联合学习

代表工作：OptimSyn [5]、RHyVE [6]、Rubric-ARM [10]、EvoRubrics [11]

**核心思路**：不再只凭规则文本和当前评分统计选择规则，而是用目标模型的优化信息、真实续训结果或偏好判断训练规则生成器与评价者。

**方法要点**：OptimSyn 用目标模型梯度及影响估计衡量合成样本效用，再以 RL 优化 rubric 生成器；RHyVE 从相同训练状态启动短分支，结合能力阶段、验证跨度和稳定性部署奖励候选。Rubric-ARM 交替优化生成器与 judge；EvoRubrics 让策略与生成器共同演化，并用区分度、多样性、标准对齐和建设性等反馈。

**实验结论**：

- OptimSyn 在医学、人文社科合成数据任务中报告下游收益
- RHyVE 显示奖励选择与策略阶段、验证跨度有关；联合学习方法也报告了奖励建模或下游对齐改善

**局限性**：影响估计、偏好判断准确性和一次答案修订收益，都不等于在线多轮 actor 的真实参数学习收益。仅仅“用 RL 训练生成器”不足以形成新贡献，关键是证明效用更适合在线 agent 学习。

### 3.3 奖励接口、聚合与在线规则演化

代表工作：RaR [7]、OpenRS [8]、ProRubric [9]、ARBOR [12]、RARE [13]

**核心思路**：即使规则文字不变，逐项反馈、偏好比较和必要条件也能改变训练激励；进一步还可以随当前策略更新规则内容和共享规则记忆。

**方法要点**：RaR 将实例级 rubric 用作 on-policy RL 奖励；OpenRS 根据候选答案差异实例化成对标准；ProRubric 用协议级维度和失败条款减少错误补偿。ARBOR 从不同质量轨迹生成局部规则，经筛选、归纳和去重进入共享缓冲区；RARE 替换饱和标准并更新任务级 rubric。

**实验结论**：

- RaR、OpenRS 和 ProRubric 分别在医学、科学、奖励建模或临床咨询设置中报告结构化奖励的收益
- ARBOR 使 Qwen3-8B 在四个多跳 QA 任务上的平均 EM 从 46.9 提升到 48.9；RARE 的配对续训中，更新分支 20 步后平均领先静态分支 0.79 分

**局限性**：当前区分度和重评分统计不保证后续效用；整条轨迹的过程评价也不等于逐步因果信用。新增规则前，还应先判断问题是否本应由聚合结构解决。

### 3.4 证据约束与反奖励利用

代表工作：MetaRubric [14]、RLR³ [15]、CHERRL [16]

**核心思路**：减少无依据的满足判定，利用可验证通道、反事实案例和攻击诊断，让得分对应真实满足。

**方法要点**：MetaRubric 使用反事实提示与证据感知的逐项反馈，并在原始意图和参考评价约束下调整规则；RLR³ 将可验证标准交给信息抽取与确定性检查，其余交给 LLM judge，再进行分层聚合；CHERRL 注入已知 judge 偏差，构造可控奖励利用场景。

**实验结论**：

- MetaRubric 和 RLR³ 分别在医学问答、部分可验证视觉语言任务中报告错误奖励减少或下游改善
- CHERRL 展示了策略利用已知偏差的过程，以及日志诊断能够识别的异常

**局限性**：文本证据不等于实际工具状态；注入式漏洞也不能替代自然产生的未知漏洞。检测成功不等于自动修复成功，真正需要建立的是自然错奖、最小修复和未见错误泛化之间的证据链。

### 3.5 过程信用、无效规则恢复与预算测量

代表工作：ARCO [17]、RuACT [18]、ProCredit [19]、RRT [20]

**核心思路**：把反馈放到更可能可学习的位置，对全部失败和全部通过分别恢复监督，并在有限 judge 调用下选择信息量更高的标准。

**方法要点**：ARCO 学习与终局反馈一致的步骤级 rubric；RuACT 对全部失败的标准使用特权信息和奖励门控自蒸馏，对全部通过的标准拆分子标准。ProCredit 在中间环境状态运行可执行验收检查，以进展变化分配信用；RRT 用项目反应理论建模难度和区分度，以 Fisher 信息选择评分标准。

**实验结论**：

- ARCO、RuACT 和 ProCredit 分别在多跳 QA、医学科学任务或 AppWorld 等环境中报告训练收益
- RRT 在四个数据集上改善奖励质量与策略表现，半量标准预算下接近全量评分基线

**局限性**：与终局相容的步骤分数不唯一，不能直接称为因果信用；恢复当前无效监督也不等于长期保持旧能力。若环境已有可靠程序化过程奖励，它应当成为自然语言 rubric 的强基线。

## 四、方法比较与分析

### 4.1 各方法的设计维度对比

| 方法类别 | 核心机制 | 优化目标 | 主要优势 | 仍需证明 |
|---|---|---|---|---|
| 诊断与合成 | 校准、探针、偏好合成 | 测量质量与偏好一致性 | 可定位问题并扩展初始资源 | 分布变化后的训练收益 |
| 训练反馈与联合学习 | 影响估计、短分支、生成器训练 | 训练效用或判断准确性 | 更接近最终学习目标 | 代理误差、迁移与验证成本 |
| 接口、聚合与在线演化 | 逐项比较、门控、规则缓冲区 | 当前区分度与结果相关性 | 改变错误激励并适应策略 | 后续效用与原目标保持 |
| 证据与反利用 | 状态检查、反事实、攻击诊断 | 可核验违反与异常 | 错误具体，便于因果对照 | 未见漏洞与正常案例误拒 |
| 过程、恢复与预算 | 步骤反馈、自蒸馏、信息选择 | 终局一致性、通过率与信息量 | 改善稀疏反馈并节约评分 | 信用正确性与长期保持 |

这些机制可以组合，但实验需要隔离主要改变。规则文字、judge 输入和聚合方式同时变化时，收益很难归因于“更好的 rubric 编辑”。

### 4.2 核心权衡

**规则效度与训练效用**：更合理的标准可能让当前评分更准确，却不一定产生更多可学习行为；更有区分度的规则也可能只是区分了任务无关特征。

**监督锚点与共享 Judge**：系统普遍依赖人工偏好、黄金 rubric、终局验收或可执行检查。训练、修订、筛选和评测若都由同一 judge 完成，改善可能只是迎合该 judge，需要程序事实、独立模型或盲评提供对照。

**局部优化与真实总成本**：规则更严格可能增加误拒，聚合更严格可能造成零奖励，评分更省钱可能漏掉低频关键要求。规则生成、重评分、候选验证和编辑器训练都应计入成本，并与“同预算直接多训 actor”比较。

### 4.3 共同的关键发现

尽管各方法出发点不同，以下几点是共识：

1. **评分更好不等于训练更好**：与独立标注更一致，不能推出策略学习一定改善
2. **规则需要证据和适用条件**：文本合理性、执行正确性和训练有效性必须分别验证
3. **当前区分度不等于长期价值**：全部通过和全部失败都缺少组内信号，但需要不同处理
4. **不同实验结论不能直接横比**：医学评分、脚本质量、问答 EM 和工具完成率不属于同一排行榜

## 五、开放问题与未来方向

### 5.1 怎样度量“对学习有用”

需要建立规则统计、影响估计、短分支收益与长程真实收益之间的关系。可检验的进展是：低成本效用预测器在未见学习状态上，比方差、相关性和固定日程更准确地选择有效编辑，并且排序能跨随机种子、学习阶段和任务复用。

### 5.2 怎样发现自然错奖并修复机制

实际 agent 可能通过新操作序列、状态边界或对象混淆获得错误奖励。系统需要从自然轨迹与主动搜索中定位错误，再判断应修改规则、证据接口、judge 还是聚合。关键证据是**未见错误机制上的改善，同时正常成功案例通过率保持稳定**。

### 5.3 动态规则怎样保持任务语义与旧能力

标准可以变细，但原任务目标不能随训练悄悄变化。每条标准应保留来源、适用条件和对应任务要求，将可演化建议与持续有效的硬约束分开。低频但严重的错误也不能因当前很少出现而停止检查，评价应同时报告新任务表现、旧要求违反率和评分成本。

### 5.4 怎样获得可识别的过程信用并控制预算

给每一步打分不难，难的是判断这一步是否促成成功。状态检查、反事实分支和动作干预可以提供更强证据，但成本与环境可恢复性不同。应区分“与终局相容的步骤分解”和“经干预验证的动作贡献”，同时防止预算调度漏掉低频关键约束。

### 5.5 我们的三个潜在切入点

三条路线应共用基础设施、分别验证：一是用真实 RL 分支收益学习可迁移的**规则效用预测器**；二是以真实工具状态发现自然错奖，完成**可执行反例驱动的最小修复**；三是在固定评分与回放预算下，研究**旧能力监测与规则重新激活**。

共同基础是可信终局验收加待研究的过程奖励：$r(\tau)=r_{\mathrm{outcome}}(\tau)+\alpha s_{\mathcal R}(\tau)$。初期固定 judge、聚合方式和优化器，先建设训练分支收益测量器与逐项环境事实核验器，再由 pilot 结果选择主线。Actor 训练、元训练、开发和最终测试必须分开，最终测试不能反馈给编辑器。[21]

## 六、总结

Rubric systems 的关键问题可以归为三层：**目标是否表达正确、评分是否对应事实、产生的信号是否促进真实学习。** 离线合成通过数据选择影响模型，在线 RL 则把规则偏差带入持续变化的策略优化，两者都需要独立下游验证。

已有研究已经覆盖诊断、合成、训练反馈、奖励聚合、联合学习、在线演化、证据约束、过程信用和预算测量。简单加入动态更新、训练一个生成器或安排几个短分支，都不足以单独支撑新增贡献。

更有价值的切口是：真实 RL 编辑效用能否被学习并复用；可执行错奖能否被修复并改善未见行为；有限预算下能否在学习新能力时保持原有要求。最终需要形成可重复、可迁移、成本清楚的证据。

## 参考文献

[1] **LLM-Rubric.** *LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts.* ACL 2024。[论文与元数据](https://aclanthology.org/2024.acl-long.745/)

[2] **RIFT.** *The Hitchhikers Guide to Rubric Quality Understanding and Enrichment.* arXiv:2604.01375，v4。[论文](https://arxiv.org/abs/2604.01375v4)

[3] **Prometheus 2.** *Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models.* EMNLP 2024。[论文与元数据](https://aclanthology.org/2024.emnlp-main.248/)

[4] **OpenRubrics.** *OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment.* ACL 2026。[论文与元数据](https://aclanthology.org/2026.acl-long.791/)

[5] **OptimSyn.** *OptimSyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation.* arXiv:2604.00536，v1。[论文](https://arxiv.org/abs/2604.00536v1)

[6] **RHyVE.** *RHyVE: Competence-Aware Verification and Phase-Aware Deployment for LLM-Generated Reward Hypotheses.* arXiv:2604.28056，v1。[论文](https://arxiv.org/abs/2604.28056v1)

[7] **RaR.** *Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains.* ICLR 2026。[会议论文页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cfd7bee7a651ee9af525098ef67a9e45-Abstract-Conference.html)

[8] **OpenRS.** *Open Rubric System: Scaling Reinforcement Learning with Pairwise Adaptive Rubric.* arXiv:2602.14069，v3。[论文](https://arxiv.org/abs/2602.14069v3)

[9] **ProRubric.** *Scoring Higher, Answering Worse: Mitigating Reward Hacking in Rubric-Based RL via Protocol-Level Rubrics.* arXiv:2609.38847，v1。[论文](https://arxiv.org/abs/2609.38847v1)

[10] **Rubric-ARM.** *Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training.* arXiv:2602.01511，v2。[论文](https://arxiv.org/abs/2602.01511v2)

[11] **EvoRubrics.** *EvoRubrics: Dynamic Rubrics as Rewards via Adversarial Co-Evolution for LLM Reinforcement Learning.* arXiv:2606.23038，v1。[论文](https://arxiv.org/abs/2606.23038v1)

[12] **ARBOR.** *ARBOR: Online Process Rewards via a Reusable Rubric Buffer for Search Agents.* arXiv:2606.03239，v1。[论文](https://arxiv.org/abs/2606.03239v1)

[13] **RARE.** *RARE: Reinforcement Learning with Adaptive Rubric Evolution for Open-Ended Generation.* arXiv:2610.25434，v1。[论文](https://arxiv.org/abs/2610.25434v1)

[14] **MetaRubric.** *MetaRubric: Learning to Reward for Rubric-Based Reinforcement Learning.* arXiv:2610.02824，v1。[论文](https://arxiv.org/abs/2610.02824v1)

[15] **RLR³.** *Reinforcement Learning with Robust Rubric Rewards.* arXiv:2605.30244，v1。[论文](https://arxiv.org/abs/2605.30244v1)

[16] **CHERRL.** *Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning.* arXiv:2606.04923，v3。[论文](https://arxiv.org/abs/2606.04923v3)

[17] **ARCO.** *ARCO: Adaptive Rubrics with Co-Evolution for Multi-Step LLM-Based Agents.* arXiv:2606.21262，v2。[论文](https://arxiv.org/abs/2606.21262v2)

[18] **RuACT.** *RuACT: Recovering Inactive Rubrics as Adaptive Critics and Teachers.* arXiv:2611.09597，v1。[论文](https://arxiv.org/abs/2611.09597v1)

[19] **ProCredit.** *ProCredit: From Outcome Rewards to Progress Credit in Agentic Reinforcement Learning.* arXiv:2609.27532，v3。[论文](https://arxiv.org/abs/2609.27532v3)

[20] **RRT.** *Rubric Rewards from Item Response Theory.* arXiv:2609.35646，v1。[论文](https://arxiv.org/abs/2609.35646v1)

[21] **AppWorld.** 官方仓库、环境说明与数据使用约束。[项目与文档](https://github.com/StonyBrookNLP/appworld)

[22] **InstructGPT / RLHF.** *Training language models to follow instructions with human feedback.* arXiv:2203.02155，v1。[论文](https://arxiv.org/abs/2203.02155v1)
