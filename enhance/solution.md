# TriSkill：Definition-Guided & Metric-Conditioned Creativity Elicitation Framework

## 0. 目标

本文件用于指导实现一个完整的 LLM 创造力激发框架，服务于现有 Creativity Benchmark。

该框架不是为每个任务手写 prompt，而是：

```text
三种创造力定义 → 三条 workflow
异构任务指标 → canonical metrics
canonical metrics → skills
skills → inference-time elicitation
benchmark → 评估 profile shift
```

核心思想：

```text
Definition chooses the workflow.
Metrics choose the skills.
Skills guide the inference.
Benchmark evaluates the profile shift.
```

中文解释：

```text
创造力定义决定大流程；
指标语义决定用哪些技能；
技能控制模型推理过程；
benchmark 衡量激发前后创造力画像的变化。
```

---

## 1. 方法名称

推荐名称：

```text
TriSkill: Definition-Guided and Metric-Conditioned Creativity Elicitation
```

完整解释：

```text
TriSkill 是一个 test-time creativity elicitation framework。
它先根据任务所属创造力层级选择 workflow，
再将任务的 raw metrics 映射到 canonical creativity metrics，
最后根据 canonical metrics 组合 inference-time skills，
使固定 LLM 在不同创造力任务上更充分地表现其创造能力。
```

---

## 2. 总体架构

### 2.1 高层流程

```text
Input Task Item
    ↓
Task Profiler
    - task name
    - creativity level
    - raw metrics
    - output schema
    - visible constraints
    ↓
Metric Abstraction Layer
    - raw metrics → canonical metrics
    ↓
Workflow Router
    - combinational workflow
    - exploratory workflow
    - transformational workflow
    ↓
Skill Composer
    - 根据 level + canonical metrics 选择 skills
    ↓
Workflow Executor
    - 顺序执行 skills
    - 维护 state
    - 保存中间 artifacts
    ↓
Verifier / Selector
    - 约束检查
    - 新颖性 / 多样性 / 有效性平衡
    ↓
Output Normalizer
    - 输出 benchmark adapter 可解析格式
    ↓
Final Answer
```

### 2.2 一句话概括

```text
TriSkill = 三层创造力 workflow + 统一指标抽象层 + skill registry + workflow executor
```

---

## 3. 核心原则

### 3.1 不要做成 task-specific prompt hacking

错误做法：

```text
DAT 写一个 prompt
BATS 写一个 prompt
RAT 写一个 prompt
AUT 写一个 prompt
Transformation 写一个 prompt
```

正确做法：

```text
组合型创造力使用组合型 workflow
探索型创造力使用探索型 workflow
转换型创造力使用转换型 workflow

每个 workflow 内部根据 canonical metrics 插入 skills
每个任务只做必要的 schema / constraint / metric mapping
```

---

### 3.2 不要直接按 raw metric 名字调用 skill

因为不同任务指标名不统一：

```text
AUT: originality
CreativeMath: novelty
CS4: novelty
DAT: pairwise semantic distance
BATS/RAT/Metaphor: accuracy
Transformation: judge score / goals
```

所以必须加入中间层：

```text
raw metric name
    ↓
canonical creativity metric
    ↓
skill
```

---

### 3.3 不允许使用 hidden gold answer

任何 skill 都不能读取或利用 scoring-only 字段。

禁止使用：

```text
reference answer
gold answer
hidden candidate_answers
test-set aggregate statistics
post-hoc score
```

可以使用：

```text
task text
visible choices
visible constraints
task name
creativity level
output schema
canonical metric objectives
```

尤其注意：

```text
BATS / RAT / Metaphor 中的 candidate_answers 如果是评分用参考答案，不能放进 prompt。
```

---

### 3.4 中间过程可以长，最终输出必须短

中间 workflow 可以产生：

```text
candidate pool
relation hypotheses
constraint map
breakage map
new primitives
verification scores
```

但提交给 benchmark adapter 的最终答案必须严格符合原任务格式。

例如：

```json
{"word": "cheese"}
```

或：

```json
{"target": "hive"}
```

或只输出任务要求的文本 / 代码 / 故事 / 方案。

---

## 4. 三种创造力 workflow

---

# 4.1 Combinational Workflow

## 4.1.1 定义

组合型创造力：

```text
通过重新组合已有概念、属性、关系，产生非显然但有意义的连接。
```

对应任务：

```text
DAT
BATS
RAT
Metaphor
```

核心思想：

```text
先扩大联想范围，
再检查关系是否成立，
最后输出格式化答案。
```

---

## 4.1.2 Workflow

```text
Combinational Workflow

1. Schema Parsing
   解析任务输出格式、是否需要 word / target / list。

2. Association Expansion
   扩大概念或候选词空间。

3. Relation / Property Bridging
   建立候选与题目之间的关系桥接。

4. Relation / Validity Verification
   检查关系是否成立，防止瞎联想。

5. Selection
   选择最符合任务目标的候选。

6. Output Normalization
   输出 adapter 可解析格式。
```

简写：

```text
expand → bridge → verify → select → normalize
```

---

## 4.1.3 组合型 canonical metrics

```yaml
semantic_diversity:
  meaning: 概念之间语义距离大
  typical_tasks:
    - DAT

relation_validity:
  meaning: 候选答案与题目之间的关系成立
  typical_tasks:
    - BATS
    - RAT
    - Metaphor

associative_bridge:
  meaning: 能在多个提示词 / 概念之间建立共同桥接
  typical_tasks:
    - RAT

metaphorical_fit:
  meaning: 源域词与目标语境之间存在恰当隐喻映射
  typical_tasks:
    - Metaphor

lexical_validity:
  meaning: 输出词合法、常见、非专名、非短语
  typical_tasks:
    - DAT
    - RAT
    - BATS
    - Metaphor

format_validity:
  meaning: 输出能被 adapter 正确解析
  typical_tasks:
    - all
```

---

## 4.1.4 组合型 skills

### Skill: semantic_domain_expansion

用途：

```text
从多个远距语义域生成候选词或候选概念。
```

目标指标：

```text
semantic_diversity
fluency
```

适用任务：

```text
DAT
RAT
Metaphor
```

输入：

```json
{
  "task_text": "...",
  "num_candidates": 30,
  "domains": ["nature", "technology", "body", "law", "music", "food", "architecture"]
}
```

输出：

```json
{
  "candidate_concepts": ["violin", "volcano", "microscope", "..."],
  "domain_tags": {
    "violin": "music",
    "volcano": "nature"
  }
}
```

Prompt 模板：

```text
Generate a diverse pool of common English nouns or concepts from distant semantic domains.

Requirements:
- Use broad and different domains.
- Avoid proper nouns.
- Avoid obscure technical terms.
- Avoid multi-word phrases unless the task explicitly allows them.
- Avoid near-synonyms and same-category clusters.
- Prefer common words that are semantically far apart.

Return JSON:
{
  "candidates": [
    {"word": "...", "domain": "..."}
  ]
}
```

---

### Skill: relation_abstraction

用途：

```text
抽象 A:B 中的关系，用于类比任务。
```

目标指标：

```text
relation_validity
```

适用任务：

```text
BATS
```

输入：

```json
{
  "source_pair": ["bird", "nest"],
  "target_source": "bee"
}
```

输出：

```json
{
  "relation_hypotheses": [
    {
      "relation": "animal -> dwelling",
      "confidence": 0.9
    }
  ]
}
```

Prompt 模板：

```text
Infer the relation between A and B.

Consider multiple possible relation types:
- part-whole
- function
- agent-instrument
- animal-habitat
- profession-tool
- cause-effect
- category-member
- synonymy
- antonymy
- object-location
- singular-plural

Return JSON:
{
  "relations": [
    {
      "relation": "...",
      "explanation": "...",
      "confidence": 0.0
    }
  ]
}
```

---

### Skill: bridge_search

用途：

```text
寻找能连接多个 clue 或 target/source domain 的桥接候选。
```

目标指标：

```text
associative_bridge
relation_validity
```

适用任务：

```text
RAT
Metaphor
```

输入：

```json
{
  "clues": ["cottage", "swiss", "cake"],
  "candidate_pool": [...]
}
```

输出：

```json
{
  "bridge_candidates": [
    {
      "word": "cheese",
      "links": ["cottage cheese", "Swiss cheese", "cheesecake"]
    }
  ]
}
```

Prompt 模板：

```text
Find candidate bridge words that connect all given clues.

The connection may involve:
- compound words
- common phrases
- idioms
- category relation
- shared properties
- prefix/suffix relation

Reject candidates that only fit one or two clues.

Return JSON:
{
  "bridge_candidates": [
    {
      "word": "...",
      "fits": [
        {"clue": "...", "connection": "..."}
      ],
      "all_clues_fit": true
    }
  ]
}
```

---

### Skill: metaphorical_property_mapping

用途：

```text
识别目标语境中的含义和属性，并映射到源域词。
```

目标指标：

```text
metaphorical_fit
relation_validity
context_fit
```

适用任务：

```text
Metaphor
```

输入：

```json
{
  "sentence": "...",
  "blank_or_target": "...",
  "visible_choices": null
}
```

输出：

```json
{
  "target_meaning": "...",
  "implied_properties": ["force", "suddenness"],
  "candidate_source_domains": ["volcano", "fire", "storm"],
  "word_candidates": [...]
}
```

Prompt 模板：

```text
Analyze the metaphorical mapping in the sentence.

Steps:
1. Identify the target meaning.
2. Identify the implied property or image.
3. Suggest possible source domains.
4. Propose candidate words that fit the metaphor and the sentence.

Do not choose words only because they are unusual.
The word must fit the context and intended meaning.

Return JSON:
{
  "target_meaning": "...",
  "implied_properties": ["..."],
  "candidates": [
    {
      "word": "...",
      "source_domain": "...",
      "mapping": "..."
    }
  ]
}
```

---

### Skill: relation_verification

用途：

```text
检查候选关系是否真正成立。
```

目标指标：

```text
relation_validity
appropriateness
```

适用任务：

```text
BATS
RAT
Metaphor
```

输入：

```json
{
  "task_type": "RAT",
  "task_text": "...",
  "candidates": [...]
}
```

输出：

```json
{
  "scored_candidates": [
    {
      "answer": "...",
      "score": 3,
      "valid": true,
      "reason": "..."
    }
  ]
}
```

评分规范：

```text
0 = relation does not hold
1 = weak or partial relation
2 = plausible relation
3 = strong and direct relation
```

Prompt 模板：

```text
Verify each candidate against the task relation.

Rules:
- Do not reward novelty alone.
- Prefer candidates that directly satisfy the relation.
- Reject candidates that only partially fit.
- Prefer common direct answers over clever but weak answers.

Return JSON:
{
  "scored_candidates": [
    {
      "candidate": "...",
      "score": 0,
      "valid": true,
      "reason": "..."
    }
  ],
  "best_candidate": "..."
}
```

---

### Skill: lexical_validity_check

用途：

```text
检查输出词是否合法。
```

目标指标：

```text
lexical_validity
format_validity
```

适用任务：

```text
DAT
BATS
RAT
Metaphor
```

检查项：

```text
是否是单词
是否是常见词
是否非专名
是否非短语
是否非乱码
是否符合词性
是否符合输出字段
```

---

### Skill: diversity_filtering

用途：

```text
去除同类、近义、重复候选，提升语义分散性。
```

目标指标：

```text
semantic_diversity
```

适用任务：

```text
DAT
AUT
```

实现方式：

```text
优先使用 LLM 自检版本。
可选 embedding rerank 版本作为 ablation。
```

注意：

```text
如果使用 embedding rerank，必须标注为 optional rerank version。
不要在主方法中显得直接优化 DAT metric。
```

---

# 4.2 Exploratory Workflow

## 4.2.1 定义

探索型创造力：

```text
在既有规则空间中探索更多、更不同、更新颖但仍然有效的解。
```

对应任务：

```text
AUT
CreativeMath
CS4
NeoCoder
```

核心思想：

```text
先解析约束，
再从多个策略方向生成候选，
再筛选新颖、多样、有效的答案。
```

---

## 4.2.2 Workflow

```text
Exploratory Workflow

1. Constraint Parsing
   解析任务硬约束、输出要求、禁止条件。

2. Strategy Axis Expansion
   构造多个探索方向。

3. Candidate Multiplication
   多路径生成候选。

4. Deduplication / Clustering
   去重、分类、识别类别覆盖。

5. Novelty / Flexibility Selection
   选择更不常规、更跨类别的候选。

6. Appropriateness / Correctness Verification
   检查是否满足题意、是否可执行、是否正确。

7. Pareto Selection
   在 novelty 与 appropriateness 之间选平衡点。

8. Output Normalization
   输出最终答案。
```

简写：

```text
parse constraints → expand strategies → generate candidates → deduplicate → verify → select → normalize
```

---

## 4.2.3 探索型 canonical metrics

```yaml
fluency:
  meaning: 有效候选数量
  typical_tasks:
    - AUT
    - CreativeMath
    - CS4

flexibility:
  meaning: 候选覆盖不同类别、策略或结构
  typical_tasks:
    - AUT
    - CreativeMath
    - CS4
    - NeoCoder

novelty:
  meaning: 答案远离常规模式，但仍有意义
  aliases:
    - originality
  typical_tasks:
    - AUT
    - CreativeMath
    - CS4
    - NeoCoder

appropriateness:
  meaning: 答案满足题意、约束、现实合理性
  typical_tasks:
    - AUT
    - CreativeMath
    - CS4
    - NeoCoder

coherence:
  meaning: 整体表达、故事或方案连贯
  typical_tasks:
    - CS4

constraint_satisfaction:
  meaning: 明确满足题目约束
  typical_tasks:
    - CS4
    - CreativeMath
    - NeoCoder

execution_validity:
  meaning: 代码或构造能通过执行 / 测试 / 验证
  typical_tasks:
    - NeoCoder
    - CreativeMath

elaboration:
  meaning: 答案细节充分
  typical_tasks:
    - AUT
    - CS4
```

---

## 4.2.4 探索型 skills

### Skill: constraint_parser

用途：

```text
解析任务硬约束、软约束、输出格式。
```

目标指标：

```text
appropriateness
constraint_satisfaction
```

适用任务：

```text
AUT
CreativeMath
CS4
NeoCoder
Transformation
```

输出：

```json
{
  "hard_constraints": [],
  "soft_constraints": [],
  "forbidden_actions": [],
  "output_schema": "...",
  "free_dimensions": []
}
```

Prompt 模板：

```text
Extract the constraints of the task.

Separate:
- hard constraints
- soft preferences
- forbidden outputs
- output format
- dimensions where creativity is allowed

Return JSON:
{
  "hard_constraints": ["..."],
  "soft_constraints": ["..."],
  "forbidden": ["..."],
  "creative_degrees_of_freedom": ["..."],
  "output_schema": "..."
}
```

---

### Skill: strategy_axis_expansion

用途：

```text
为任务建立多个探索方向。
```

目标指标：

```text
flexibility
novelty
```

适用任务：

```text
AUT
CreativeMath
CS4
NeoCoder
```

可用策略轴：

```text
conventional
minimal
cross-domain
reverse-assumption
mechanism-based
edge-case
aesthetic
algorithmic
social
physical
symbolic
```

Prompt 模板：

```text
Create diverse strategy axes for solving this task.

Each strategy should explore a different region of the solution space.
Do not generate final answers yet.

Return JSON:
{
  "strategies": [
    {
      "name": "...",
      "principle": "...",
      "why_different": "..."
    }
  ]
}
```

---

### Skill: candidate_multiplication

用途：

```text
基于多个策略生成多个候选。
```

目标指标：

```text
fluency
flexibility
```

适用任务：

```text
AUT
CreativeMath
CS4
NeoCoder
```

Prompt 模板：

```text
Generate candidate solutions using the provided strategies.

For each candidate:
- state which strategy it uses
- ensure it satisfies hard constraints
- make it distinct from other candidates

Return JSON:
{
  "candidates": [
    {
      "candidate": "...",
      "strategy": "...",
      "constraint_notes": "..."
    }
  ]
}
```

---

### Skill: semantic_deduplication

用途：

```text
删除重复或本质相同的候选。
```

目标指标：

```text
fluency
flexibility
```

适用任务：

```text
AUT
CS4
CreativeMath
NeoCoder
```

Prompt 模板：

```text
Remove candidates that are semantically or structurally redundant.

Two candidates are redundant if:
- they use the same core idea
- they differ only in wording
- they solve the task by the same mechanism
- they belong to the same narrow category

Return JSON:
{
  "unique_candidates": [...],
  "removed_duplicates": [
    {
      "candidate": "...",
      "duplicate_of": "...",
      "reason": "..."
    }
  ]
}
```

---

### Skill: category_coverage

用途：

```text
保证候选覆盖不同类别或策略。
```

目标指标：

```text
flexibility
```

适用任务：

```text
AUT
CS4
CreativeMath
NeoCoder
```

输出：

```json
{
  "clusters": {
    "physical use": [...],
    "symbolic use": [...],
    "tool use": [...]
  },
  "coverage_score": 0.0
}
```

---

### Skill: novelty_shift

用途：

```text
推动答案远离常规模式，但不破坏合理性。
```

目标指标：

```text
novelty
originality
```

适用任务：

```text
AUT
CreativeMath
CS4
NeoCoder
Transformation
```

Prompt 模板：

```text
Improve the novelty of the candidate without violating constraints.

Do not make the answer bizarre or infeasible.
Increase novelty by:
- using an uncommon but valid mechanism
- combining distant domains
- avoiding the most obvious solution
- adding a non-obvious perspective

Return JSON:
{
  "revised_candidate": "...",
  "novelty_change": "...",
  "constraint_preservation": "..."
}
```

---

### Skill: appropriateness_check

用途：

```text
检查答案是否符合题意、约束、现实合理性。
```

目标指标：

```text
appropriateness
constraint_satisfaction
```

适用任务：

```text
AUT
CreativeMath
CS4
NeoCoder
Transformation
```

Prompt 模板：

```text
Check whether each candidate satisfies the task constraints.

Reject candidates that:
- violate hard constraints
- are irrelevant
- are infeasible
- are incoherent
- sacrifice correctness for novelty

Return JSON:
{
  "scored_candidates": [
    {
      "candidate": "...",
      "valid": true,
      "appropriateness_score": 0.0,
      "violations": []
    }
  ]
}
```

---

### Skill: coherence_check

用途：

```text
检查故事、方案或长答案的连贯性。
```

目标指标：

```text
coherence
story_quality
```

适用任务：

```text
CS4
Transformation
CreativeMath
```

检查项：

```text
因果是否连贯
叙事是否完整
前后是否矛盾
人物 / 机制 / 变量是否一致
是否满足约束
```

---

### Skill: execution_verification

用途：

```text
对代码、数学构造或可执行方案进行验证。
```

目标指标：

```text
execution_validity
correctness
appropriateness
```

适用任务：

```text
NeoCoder
CreativeMath
```

实现方式：

```text
NeoCoder:
  优先调用已有代码执行 / test cases / static checks。

CreativeMath:
  可用 LLM checker + symbolic sanity check。
  如果有可程序验证部分，可以调用程序验证。
```

注意：

```text
execution_verification 不能伪造测试结果。
如果无法实际执行，应标记为 llm_self_check，而不是 execution_pass。
```

---

### Skill: pareto_selection

用途：

```text
在 novelty / flexibility / appropriateness / correctness 之间做平衡选择。
```

目标指标：

```text
novelty
flexibility
appropriateness
correctness
```

适用任务：

```text
AUT
CreativeMath
CS4
NeoCoder
Transformation
```

选择原则：

```text
优先满足 hard constraints
然后最大化 novelty / flexibility
如果 novelty 与 appropriateness 冲突，优先保住 appropriateness
```

Prompt 模板：

```text
Select the best final answer by balancing:
1. hard constraint satisfaction
2. appropriateness / correctness
3. novelty
4. flexibility / diversity
5. clarity and format compliance

Do not select a candidate that is novel but invalid.

Return JSON:
{
  "selected": "...",
  "selection_reason": "...",
  "scores": {
    "appropriateness": 0.0,
    "novelty": 0.0,
    "flexibility": 0.0
  }
}
```

---

# 4.3 Transformational Workflow

## 4.3.1 定义

转换型创造力：

```text
当底层规则、假设、公理或系统边界发生变化时，
重新构造新的机制、解释或系统。
```

对应任务：

```text
Transformation
```

核心思想：

```text
不是机械遵守新规则，
而是在新规则世界下重建系统。
```

---

## 4.3.2 Workflow

```text
Transformational Workflow

1. Rule Parsing
   解析 5 条 changed rules 和 3 个 goals。

2. Old Dependency Mapping
   识别旧系统依赖了哪些旧规则 / 旧假设。

3. Breakage Propagation
   推导新规则破坏哪些模块。

4. New Primitive Induction
   提出新概念、新变量、新接口、新测量方式、新协调原则。

5. Architecture Reconstruction
   重建新规则世界下的核心系统。

6. Performance Restoration
   说明如何恢复关键性能。

7. Norm Establishment
   建立新标准、新解释、新培训、新验证体系。

8. Old-world Residue Audit
   检查方案是否仍偷偷依赖旧世界假设。

9. Goal Coverage Check
   确认三个 goals 都被覆盖。

10. Final Answer
   输出最终重建方案。
```

简写：

```text
parse rules → map old dependencies → propagate breakage → induce primitives → reconstruct → restore → establish norms → audit residue → final
```

---

## 4.3.3 转换型 canonical metrics

```yaml
rule_utilization:
  meaning: 是否真正使用了所有 changed rules，而不是只复述
  typical_tasks:
    - Transformation

system_reconstruction:
  meaning: 是否重建了新规则世界下的核心机制
  typical_tasks:
    - Transformation

performance_restoration:
  meaning: 是否恢复关键性能
  typical_tasks:
    - Transformation

norm_establishment:
  meaning: 是否建立新规范、接口、解释、培训或验证体系
  typical_tasks:
    - Transformation

old_assumption_removal:
  meaning: 是否去除了旧世界假设残留
  typical_tasks:
    - Transformation

interface_coordination:
  meaning: 是否考虑基础设施、接口、组织协同
  typical_tasks:
    - Transformation

cognitive_execution:
  meaning: 是否考虑术语、公众理解、培训、执行语言
  typical_tasks:
    - Transformation

goal_coverage:
  meaning: 是否覆盖 3 个 goals
  typical_tasks:
    - Transformation
```

---

## 4.3.4 转换型 skills

### Skill: rule_parser

用途：

```text
解析 5 条 changed rules 和 3 个 goals。
```

目标指标：

```text
rule_utilization
goal_coverage
```

输出：

```json
{
  "rules": [
    {
      "slot": "Core Rule",
      "content": "...",
      "strength": "strong",
      "type": "reversal"
    }
  ],
  "goals": [
    {
      "name": "rebuild_core_mechanism",
      "content": "..."
    }
  ]
}
```

---

### Skill: old_dependency_mapping

用途：

```text
识别旧系统依赖哪些旧规则、旧假设、旧接口、旧测量、旧语言。
```

目标指标：

```text
rule_utilization
old_assumption_removal
```

Prompt 模板：

```text
Identify what the old system implicitly depended on before the rule changes.

For each component, specify:
- old assumption
- old mechanism
- old interface or infrastructure
- old coordination norm
- old language or cognitive convention

Return JSON:
{
  "old_dependencies": [
    {
      "component": "...",
      "old_assumption": "...",
      "dependent_rule": "...",
      "why_it_mattered": "..."
    }
  ]
}
```

---

### Skill: breakage_propagation

用途：

```text
推导新规则导致哪些模块失效。
```

目标指标：

```text
system_reconstruction
interface_coordination
cognitive_execution
```

必须区分五类失效：

```text
core mechanism failure
secondary structural failure
interface / infrastructure failure
institutional coordination failure
cognitive / language execution failure
```

Prompt 模板：

```text
For each changed rule, analyze how it breaks the old system.

Distinguish:
1. core mechanism failure
2. secondary structural failure
3. interface / infrastructure failure
4. institutional coordination failure
5. cognitive / language execution failure

Return JSON:
{
  "breakages": [
    {
      "rule": "...",
      "broken_components": ["..."],
      "failure_type": "...",
      "failure_chain": "..."
    }
  ]
}
```

---

### Skill: new_primitive_induction

用途：

```text
提出新规则世界下所需的新抽象、新变量、新机制、新接口。
```

目标指标：

```text
system_reconstruction
novelty
norm_establishment
```

新 primitive 类型：

```text
new measurement
new state variable
new causal primitive
new interface protocol
new coordination rule
new terminology
new validation method
```

Prompt 模板：

```text
Invent the minimum necessary new primitives for the new rule-world.

Possible primitives:
- new measurement
- new state variable
- new causal mechanism
- new interface protocol
- new coordination standard
- new terminology
- new training or validation method

Do not invent decorative concepts.
Each primitive must solve a breakage identified earlier.

Return JSON:
{
  "new_primitives": [
    {
      "name": "...",
      "type": "...",
      "solves_breakage": "...",
      "why_needed": "...",
      "how_used": "..."
    }
  ]
}
```

---

### Skill: architecture_reconstruction

用途：

```text
基于新 primitives 重建系统架构。
```

目标指标：

```text
system_reconstruction
goal_coverage
```

Prompt 模板：

```text
Design a coherent system architecture for the new rule-world.

The architecture must:
- not depend on invalid old-world assumptions
- use the new primitives
- satisfy the changed rules
- rebuild the core mechanism
- be executable or operationally clear

Return JSON:
{
  "architecture": {
    "core_mechanism": "...",
    "modules": [
      {
        "name": "...",
        "function": "...",
        "uses_primitives": ["..."],
        "handles_rules": ["..."]
      }
    ],
    "system_flow": "..."
  }
}
```

---

### Skill: performance_restoration

用途：

```text
说明新系统如何恢复关键性能。
```

目标指标：

```text
performance_restoration
```

Prompt 模板：

```text
Explain how the reconstructed system restores key performance under the new rules.

Include:
- target performance dimension
- why old performance failed
- new mechanism for restoring it
- trade-offs
- verification method

Return JSON:
{
  "performance_restoration": [
    {
      "performance": "...",
      "old_failure": "...",
      "new_restoration_mechanism": "...",
      "verification": "..."
    }
  ]
}
```

---

### Skill: norm_establishment

用途：

```text
建立新规范、新标准、新接口、新培训、新验证体系。
```

目标指标：

```text
norm_establishment
interface_coordination
cognitive_execution
```

Prompt 模板：

```text
Establish the new norms required for the reconstructed system.

Include:
- shared standards
- interface protocols
- terminology
- training or education
- verification or auditing process
- institutional coordination

Return JSON:
{
  "new_norms": [
    {
      "norm": "...",
      "purpose": "...",
      "who_uses_it": "...",
      "how_it_is_verified": "..."
    }
  ]
}
```

---

### Skill: residue_audit

用途：

```text
检查最终方案是否仍然偷偷依赖旧世界假设。
```

目标指标：

```text
old_assumption_removal
appropriateness
```

Prompt 模板：

```text
Audit the proposed system for residual old-world assumptions.

Check whether any part still depends on:
- old core rule
- old secondary structure
- old interface
- old institutional standard
- old terminology or cognitive convention

If residue is found, propose a revision.

Return JSON:
{
  "residue_findings": [
    {
      "location": "...",
      "old_assumption": "...",
      "severity": "low|medium|high",
      "revision": "..."
    }
  ],
  "residue_free": true
}
```

---

### Skill: goal_coverage_check

用途：

```text
检查 3 个 goals 是否全部覆盖。
```

目标指标：

```text
goal_coverage
appropriateness
```

Prompt 模板：

```text
Check whether the final design satisfies each goal.

Goals:
1. rebuild_core_mechanism
2. restore_key_performance
3. establish_new_norm

Return JSON:
{
  "goal_coverage": {
    "rebuild_core_mechanism": {
      "covered": true,
      "evidence": "..."
    },
    "restore_key_performance": {
      "covered": true,
      "evidence": "..."
    },
    "establish_new_norm": {
      "covered": true,
      "evidence": "..."
    }
  },
  "missing_goals": []
}
```

---

## 5. Task Profile 配置

所有任务都应该有一个 task profile 配置文件。

推荐路径：

```text
evalscope/elicitation/config/task_profiles.yaml
```

---

### 5.1 DAT profile

```yaml
DAT:
  level: combinational
  workflow: combinational
  raw_metrics:
    - pairwise_semantic_distance
  canonical_metrics:
    semantic_diversity: high
    lexical_validity: medium
    format_validity: high
  output_schema:
    type: word_list
    field: null
  skills:
    - semantic_domain_expansion
    - lexical_validity_check
    - diversity_filtering
    - output_normalization
  budgets:
    candidate_count: 50
    final_count: 10
    generation_temperature: 0.9
    verification_temperature: 0.0
```

---

### 5.2 BATS profile

```yaml
BATS:
  level: combinational
  workflow: combinational
  raw_metrics:
    - bats_accuracy
  canonical_metrics:
    relation_validity: high
    lexical_validity: high
    format_validity: high
  output_schema:
    type: json
    required_fields:
      - target
  skills:
    - relation_abstraction
    - candidate_generation
    - relation_verification
    - output_normalization
  budgets:
    relation_hypotheses: 3
    candidates_per_relation: 3
    generation_temperature: 0.6
    verification_temperature: 0.0
```

---

### 5.3 RAT profile

```yaml
RAT:
  level: combinational
  workflow: combinational
  raw_metrics:
    - rat_accuracy
  canonical_metrics:
    associative_bridge: high
    relation_validity: high
    lexical_validity: high
    format_validity: high
  output_schema:
    type: json
    required_fields:
      - word
  skills:
    - association_pool_generation
    - bridge_search
    - three_way_fit_verification
    - output_normalization
  budgets:
    candidate_count: 30
    generation_temperature: 0.8
    verification_temperature: 0.0
```

---

### 5.4 Metaphor profile

```yaml
Metaphor:
  level: combinational
  workflow: combinational
  raw_metrics:
    - metaphor_accuracy
  canonical_metrics:
    metaphorical_fit: high
    context_fit: high
    relation_validity: high
    lexical_validity: high
    format_validity: high
  output_schema:
    type: json
    required_fields:
      - word
  skills:
    - context_meaning_parser
    - metaphorical_property_mapping
    - relation_verification
    - output_normalization
  budgets:
    candidate_count: 12
    generation_temperature: 0.7
    verification_temperature: 0.0
```

---

### 5.5 AUT profile

```yaml
AUT:
  level: exploratory
  workflow: exploratory
  raw_metrics:
    - fluency
    - flexibility
    - originality
    - elaboration
  canonical_metrics:
    fluency: high
    flexibility: high
    novelty: high
    elaboration: medium
    appropriateness: high
    format_validity: high
  output_schema:
    type: list_or_text
  skills:
    - constraint_parser
    - candidate_multiplication
    - category_coverage
    - novelty_shift
    - semantic_deduplication
    - appropriateness_check
    - output_normalization
  budgets:
    candidate_count: 30
    final_count: null
    generation_temperature: 0.9
    verification_temperature: 0.0
```

---

### 5.6 CreativeMath profile

```yaml
CreativeMath:
  level: exploratory
  workflow: exploratory
  raw_metrics:
    - fluency
    - novelty
    - flexibility
    - appropriateness
  canonical_metrics:
    fluency: medium
    novelty: high
    flexibility: high
    appropriateness: high
    correctness: high
    format_validity: high
  output_schema:
    type: solution_text
  skills:
    - constraint_parser
    - strategy_axis_expansion
    - candidate_multiplication
    - correctness_check
    - novelty_selection
    - pareto_selection
    - output_normalization
  budgets:
    strategy_count: 5
    candidates_per_strategy: 1
    generation_temperature: 0.8
    verification_temperature: 0.0
```

---

### 5.7 CS4 profile

```yaml
CS4:
  level: exploratory
  workflow: exploratory
  raw_metrics:
    - fluency
    - grammar
    - coherence
    - likability
    - flexibility
    - appropriateness
    - novelty
    - QUC
    - RCS
  canonical_metrics:
    fluency: medium
    novelty: high
    flexibility: medium
    appropriateness: high
    coherence: high
    constraint_satisfaction: high
    story_quality: high
    format_validity: high
  output_schema:
    type: story_text
  skills:
    - constraint_parser
    - strategy_axis_expansion
    - plot_variant_generation
    - novelty_shift
    - coherence_check
    - constraint_satisfaction_check
    - pareto_selection
    - output_normalization
  budgets:
    plot_variants: 4
    generation_temperature: 0.9
    verification_temperature: 0.0
```

说明：

```text
QUC / RCS 保留为 CS4 task-specific raw metrics。
在高层分析中只映射到 constraint_satisfaction / story_quality / coherence，不强行解释其内部公式。
```

---

### 5.8 NeoCoder profile

```yaml
NeoCoder:
  level: exploratory
  workflow: exploratory
  raw_metrics:
    - execution
    - test_pass
    - novelty
    - flexibility
    - appropriateness
  canonical_metrics:
    execution_validity: high
    correctness: high
    appropriateness: high
    novelty: medium
    flexibility: medium
    format_validity: high
  output_schema:
    type: code
  skills:
    - constraint_parser
    - algorithmic_strategy_expansion
    - code_candidate_generation
    - execution_verification
    - code_diversity_selection
    - pareto_selection
    - output_normalization
  budgets:
    strategy_count: 4
    candidates_per_strategy: 1
    generation_temperature: 0.7
    verification_temperature: 0.0
```

---

### 5.9 Transformation profile

```yaml
Transformation:
  level: transformational
  workflow: transformational
  raw_metrics:
    - judge_score
    - rebuild_core_mechanism
    - restore_key_performance
    - establish_new_norm
  canonical_metrics:
    rule_utilization: high
    system_reconstruction: high
    performance_restoration: high
    norm_establishment: high
    old_assumption_removal: high
    interface_coordination: high
    cognitive_execution: high
    goal_coverage: high
    format_validity: high
  output_schema:
    type: reconstruction_text
  skills:
    - rule_parser
    - old_dependency_mapping
    - breakage_propagation
    - new_primitive_induction
    - architecture_reconstruction
    - performance_restoration
    - norm_establishment
    - residue_audit
    - goal_coverage_check
    - output_normalization
  budgets:
    generation_temperature: 0.7
    verification_temperature: 0.0
    max_final_tokens: 1200
```

---

## 6. 推荐代码结构

推荐新增目录：

```text
evalscope/
  elicitation/
    __init__.py

    core/
      __init__.py
      state.py
      profile.py
      registry.py
      executor.py
      router.py
      normalizer.py
      logging.py

    workflows/
      __init__.py
      base.py
      combinational.py
      exploratory.py
      transformational.py

    skills/
      __init__.py
      base.py

      combinational/
        __init__.py
        semantic_domain_expansion.py
        association_pool_generation.py
        relation_abstraction.py
        bridge_search.py
        metaphorical_property_mapping.py
        relation_verification.py
        lexical_validity_check.py
        diversity_filtering.py

      exploratory/
        __init__.py
        constraint_parser.py
        strategy_axis_expansion.py
        candidate_multiplication.py
        semantic_deduplication.py
        category_coverage.py
        novelty_shift.py
        appropriateness_check.py
        coherence_check.py
        execution_verification.py
        pareto_selection.py

      transformational/
        __init__.py
        rule_parser.py
        old_dependency_mapping.py
        breakage_propagation.py
        new_primitive_induction.py
        architecture_reconstruction.py
        performance_restoration.py
        norm_establishment.py
        residue_audit.py
        goal_coverage_check.py

      common/
        __init__.py
        output_normalization.py
        json_repair.py
        format_validation.py

    prompts/
      combinational.yaml
      exploratory.yaml
      transformational.yaml
      common.yaml

    config/
      task_profiles.yaml
      skill_registry.yaml
      ablations.yaml

    runners/
      run_elicited_eval.py
      run_ablation.py
```

---

## 7. 核心数据结构

### 7.1 ElicitationState

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ElicitationState:
    task_name: str
    item_id: Optional[str]
    raw_item: Dict[str, Any]
    original_prompt: str

    level: str
    workflow: str

    raw_metrics: List[str] = field(default_factory=list)
    canonical_metrics: Dict[str, str] = field(default_factory=dict)

    output_schema: Dict[str, Any] = field(default_factory=dict)
    visible_choices: Optional[List[str]] = None

    constraints: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected_candidate: Optional[Any] = None

    artifacts: Dict[str, Any] = field(default_factory=dict)
    skill_trace: List[Dict[str, Any]] = field(default_factory=list)

    final_answer: Optional[str] = None
    parse_success: bool = False

    warnings: List[str] = field(default_factory=list)
```

---

### 7.2 TaskProfile

```python
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class TaskProfile:
    task_name: str
    level: str
    workflow: str
    raw_metrics: List[str]
    canonical_metrics: Dict[str, str]
    output_schema: Dict[str, Any]
    skills: List[str]
    budgets: Dict[str, Any]
```

---

### 7.3 BaseSkill

```python
from abc import ABC, abstractmethod


class BaseSkill(ABC):
    name: str = "base_skill"

    def __init__(self, config=None):
        self.config = config or {}

    @abstractmethod
    def run(self, state: ElicitationState, llm) -> ElicitationState:
        pass
```

---

### 7.4 Skill Registry

```python
SKILL_REGISTRY = {
    # combinational
    "semantic_domain_expansion": SemanticDomainExpansionSkill,
    "association_pool_generation": AssociationPoolGenerationSkill,
    "relation_abstraction": RelationAbstractionSkill,
    "bridge_search": BridgeSearchSkill,
    "metaphorical_property_mapping": MetaphoricalPropertyMappingSkill,
    "relation_verification": RelationVerificationSkill,
    "lexical_validity_check": LexicalValidityCheckSkill,
    "diversity_filtering": DiversityFilteringSkill,

    # exploratory
    "constraint_parser": ConstraintParserSkill,
    "strategy_axis_expansion": StrategyAxisExpansionSkill,
    "candidate_multiplication": CandidateMultiplicationSkill,
    "semantic_deduplication": SemanticDeduplicationSkill,
    "category_coverage": CategoryCoverageSkill,
    "novelty_shift": NoveltyShiftSkill,
    "appropriateness_check": AppropriatenessCheckSkill,
    "coherence_check": CoherenceCheckSkill,
    "execution_verification": ExecutionVerificationSkill,
    "pareto_selection": ParetoSelectionSkill,

    # transformational
    "rule_parser": RuleParserSkill,
    "old_dependency_mapping": OldDependencyMappingSkill,
    "breakage_propagation": BreakagePropagationSkill,
    "new_primitive_induction": NewPrimitiveInductionSkill,
    "architecture_reconstruction": ArchitectureReconstructionSkill,
    "performance_restoration": PerformanceRestorationSkill,
    "norm_establishment": NormEstablishmentSkill,
    "residue_audit": ResidueAuditSkill,
    "goal_coverage_check": GoalCoverageCheckSkill,

    # common
    "output_normalization": OutputNormalizationSkill,
}
```

---

### 7.5 Workflow Executor

```python
class WorkflowExecutor:
    def __init__(self, skill_registry, logger=None):
        self.skill_registry = skill_registry
        self.logger = logger

    def execute(self, state: ElicitationState, profile: TaskProfile, llm):
        for skill_name in profile.skills:
            skill_cls = self.skill_registry[skill_name]
            skill = skill_cls(config=profile.budgets)

            before = state
            state = skill.run(state, llm)

            state.skill_trace.append({
                "skill": skill_name,
                "num_candidates": len(state.candidates),
                "artifacts_keys": list(state.artifacts.keys()),
                "warnings": state.warnings[-3:],
            })

            if self.logger:
                self.logger.log_skill_state(state, skill_name)

        return state
```

---

## 8. LLM 调用规范

所有 LLM skills 应该尽量使用结构化输出。

### 8.1 推荐通用调用函数

```python
def call_llm_json(llm, prompt: str, temperature: float = 0.0, max_tokens: int = 1024):
    response = llm.generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    parsed = try_parse_json(response)

    if parsed is None:
        repaired = repair_json_with_llm_or_regex(response)
        parsed = try_parse_json(repaired)

    return parsed, response
```

---

### 8.2 JSON 修复原则

````text
先尝试 json.loads。
失败后尝试提取 ```json ... ```。
仍失败则使用轻量 regex 提取字段。
必要时调用 json_repair skill。
所有修复失败都要记录 warning。
````

---

## 9. 输出 Normalization

### 9.1 OutputNormalizationSkill

该 skill 必须是所有 workflow 的最后一步。

任务级输出规则：

```python
def normalize_output(state: ElicitationState) -> str:
    task = state.task_name

    if task == "DAT":
        return normalize_dat(state)

    if task == "BATS":
        return normalize_json_field(state, field="target")

    if task == "RAT":
        return normalize_json_field(state, field="word")

    if task == "Metaphor":
        return normalize_json_field(state, field="word")

    if task == "AUT":
        return normalize_aut(state)

    if task == "CreativeMath":
        return normalize_solution_text(state)

    if task == "CS4":
        return normalize_story_text(state)

    if task == "NeoCoder":
        return normalize_code(state)

    if task == "Transformation":
        return normalize_reconstruction_text(state)

    return str(state.selected_candidate or state.final_answer)
```

---

### 9.2 BATS / RAT / Metaphor 输出

必须输出：

```json
{"target": "answer"}
```

或：

```json
{"word": "answer"}
```

不要输出解释。

---

### 9.3 DAT 输出

输出任务原本 adapter 能解析的格式。

推荐：

```text
word1, word2, word3, word4, word5, word6, word7, word8, word9, word10
```

或按照已有 DAT adapter 要求输出。

---

### 9.4 NeoCoder 输出

必须保证：

```text
只输出代码或 adapter 需要的字段
不要在代码外写解释
如果原任务要求 markdown code block，则遵循原任务格式
```

---

## 10. 任务级实现细节

---

# 10.1 DAT 实现

## 10.1.1 目标

提升：

```text
semantic_diversity
```

同时控制：

```text
lexical_validity
format_validity
```

---

## 10.1.2 Skill chain

```text
semantic_domain_expansion
→ lexical_validity_check
→ diversity_filtering
→ output_normalization
```

---

## 10.1.3 流程

```text
1. 生成 30-50 个候选常见英文名词。
2. 每个候选带 domain tag。
3. 删除：
   - proper nouns
   - obscure words
   - multi-word phrases
   - near-synonyms
   - same-domain duplicates
4. 选择 10 个语义域尽量不同的词。
5. 输出逗号分隔词表。
```

---

## 10.1.4 DAT prompt

```text
You are completing a divergent association task.

Generate a pool of common English nouns from maximally different semantic domains.

Rules:
- Avoid proper nouns.
- Avoid obscure technical terms.
- Avoid multi-word phrases.
- Avoid synonyms or close associates.
- Prefer common concrete nouns.
- Use at most one word from each broad semantic domain.

Use diverse domains such as:
nature, artifact, music, science, law, food, architecture, geography, body, tool, material, social institution, weather, mathematics, emotion-related object.

Return JSON:
{
  "candidates": [
    {"word": "...", "domain": "..."}
  ]
}
```

---

## 10.1.5 DAT selection prompt

```text
Select exactly 10 words from the candidate pool.

Selection criteria:
1. Maximize semantic distance among selected words.
2. Avoid words from the same broad domain.
3. Avoid proper nouns, rare terms, multi-word phrases.
4. Prefer common nouns.
5. Do not include synonyms or close associates.

Return JSON:
{
  "selected_words": ["...", "..."]
}
```

---

# 10.2 BATS 实现

## 10.2.1 目标

提升：

```text
relation_validity
```

---

## 10.2.2 Skill chain

```text
relation_abstraction
→ candidate_generation
→ relation_verification
→ output_normalization
```

---

## 10.2.3 流程

```text
1. 抽象 A:B 的可能关系。
2. 每个关系生成 C:? 的候选。
3. 验证 A:B 与 C:answer 是否同构。
4. 选择最直接、最常见、关系最强的词。
5. 输出 {"target": "..."}。
```

---

## 10.2.4 BATS prompt

```text
You are solving an analogy task.

Infer the relation between A and B, then apply exactly the same relation to C.

Do not choose an unusual word only because it is creative.
The final answer must preserve the same relation.

Return JSON:
{
  "relations": [
    {
      "relation": "...",
      "candidate_answers": ["...", "..."],
      "confidence": 0.0
    }
  ]
}
```

---

## 10.2.5 BATS verification prompt

```text
Verify each candidate answer.

For each candidate y, check whether A:B and C:y instantiate the same relation.

Score:
0 = relation does not hold
1 = weak relation
2 = plausible relation
3 = strong same relation

Prefer common direct answers over clever indirect ones.

Return JSON:
{
  "scored_candidates": [
    {
      "candidate": "...",
      "score": 0,
      "valid": true,
      "reason": "..."
    }
  ],
  "best_candidate": "..."
}
```

---

# 10.3 RAT 实现

## 10.3.1 目标

提升：

```text
associative_bridge
relation_validity
```

---

## 10.3.2 Skill chain

```text
association_pool_generation
→ bridge_search
→ three_way_fit_verification
→ output_normalization
```

---

## 10.3.3 流程

```text
1. 对每个 clue 生成关联词 / 固定搭配 / compound。
2. 找能同时连接三个 clue 的 bridge word。
3. 验证候选是否 fit all three clues。
4. 优先选择 common English word。
5. 输出 {"word": "..."}。
```

---

## 10.3.4 RAT prompt

```text
You are solving a Remote Associates Test.

Find one common English word that connects all three clue words.

The connection may involve:
- compound words
- common phrases
- idioms
- category membership
- shared properties
- prefix/suffix relation

Do not output a word that only fits one or two clues.
Generate candidate bridge words and test each against all three clues.

Return JSON:
{
  "candidates": [
    {
      "word": "...",
      "connections": [
        {"clue": "...", "connection": "..."}
      ],
      "all_three_fit": true
    }
  ]
}
```

---

## 10.3.5 RAT verification prompt

```text
Verify whether each candidate connects all three clues.

A candidate is invalid if it only fits one or two clues.
Prefer candidates that form common compounds, phrases, or widely known expressions.

Return JSON:
{
  "scored_candidates": [
    {
      "candidate": "...",
      "fits_clue_1": true,
      "fits_clue_2": true,
      "fits_clue_3": true,
      "score": 0,
      "reason": "..."
    }
  ],
  "best_candidate": "..."
}
```

---

# 10.4 Metaphor 实现

## 10.4.1 目标

提升：

```text
metaphorical_fit
context_fit
relation_validity
```

---

## 10.4.2 Skill chain

```text
context_meaning_parser
→ metaphorical_property_mapping
→ relation_verification
→ output_normalization
```

---

## 10.4.3 流程

```text
1. 解析语境中的目标含义。
2. 提取 implied property。
3. 从多个 source domains 生成候选词。
4. 检查候选是否符合隐喻映射和语法语境。
5. 输出 {"word": "..."}。
```

---

## 10.4.4 Metaphor prompt

```text
You are solving a metaphorical word-mapping task.

Analyze:
1. the target meaning in the sentence
2. the implied property or image
3. possible source domains
4. candidate words that fit both the metaphor and context

Do not choose a word only because it is unusual.
The word must fit the context and intended metaphor.

Return JSON:
{
  "target_meaning": "...",
  "implied_properties": ["..."],
  "candidates": [
    {
      "word": "...",
      "source_domain": "...",
      "mapping": "...",
      "context_fit": "..."
    }
  ]
}
```

---

# 10.5 AUT 实现

## 10.5.1 目标

提升：

```text
fluency
flexibility
novelty/originality
appropriateness
```

---

## 10.5.2 Skill chain

```text
constraint_parser
→ candidate_multiplication
→ category_coverage
→ novelty_shift
→ semantic_deduplication
→ appropriateness_check
→ output_normalization
```

---

## 10.5.3 流程

```text
1. 解析物体和用途约束。
2. 生成 30 个候选用途。
3. 按功能类别聚类：
   - physical tool
   - container
   - weight
   - decoration
   - safety
   - signal
   - educational
   - artistic
   - social
   - scientific
4. 删除重复用途。
5. 挑选覆盖不同类别且可行的用途。
6. 对部分常规用途做 novelty shift。
7. 检查是否现实可行。
8. 输出最终用途列表。
```

---

## 10.5.4 AUT prompt

```text
Generate alternative uses for the given object.

Requirements:
- Generate many valid uses.
- Cover different functional categories.
- Avoid duplicates and trivial rewordings.
- Include some unusual but feasible uses.
- Do not include impossible or irrelevant uses.

Return JSON:
{
  "uses": [
    {
      "use": "...",
      "category": "...",
      "novelty": "low|medium|high",
      "feasibility": "low|medium|high"
    }
  ]
}
```

---

# 10.6 CreativeMath 实现

## 10.6.1 目标

提升：

```text
novelty
flexibility
appropriateness
correctness
```

---

## 10.6.2 Skill chain

```text
constraint_parser
→ strategy_axis_expansion
→ candidate_multiplication
→ correctness_check
→ novelty_selection
→ pareto_selection
→ output_normalization
```

---

## 10.6.3 数学策略轴

```text
constructive proof
counterexample
geometric interpretation
algebraic transformation
invariant
extremal argument
symmetry
probabilistic construction
algorithmic construction
```

---

## 10.6.4 CreativeMath prompt

```text
Solve the creative math task using diverse strategies.

First identify the constraints.
Then generate candidate solution ideas from distinct mathematical strategies:
- constructive
- algebraic
- geometric
- invariant-based
- extremal
- counterexample-based
- algorithmic

For each candidate, check correctness and constraint satisfaction.

Return JSON:
{
  "candidates": [
    {
      "strategy": "...",
      "solution": "...",
      "correctness_check": "...",
      "novelty": "low|medium|high"
    }
  ],
  "selected_solution": "..."
}
```

---

# 10.7 CS4 实现

## 10.7.1 目标

提升：

```text
novelty
coherence
constraint_satisfaction
story_quality
appropriateness
```

---

## 10.7.2 Skill chain

```text
constraint_parser
→ strategy_axis_expansion
→ plot_variant_generation
→ novelty_shift
→ coherence_check
→ constraint_satisfaction_check
→ pareto_selection
→ output_normalization
```

---

## 10.7.3 流程

```text
1. 解析故事约束。
2. 生成 3-4 个 plot variant。
3. 每个 variant 保证满足约束。
4. 对 plot premise / conflict / resolution 做 novelty shift。
5. 检查 coherence、grammar、constraint satisfaction。
6. 选择最平衡版本。
7. 输出 final story。
```

---

## 10.7.4 CS4 prompt

```text
Create several story plans that satisfy the given constraints.

Each plan should include:
- premise
- main conflict
- key events
- resolution
- how each constraint is satisfied
- what makes it novel

Then select the plan that best balances:
- constraint satisfaction
- coherence
- novelty
- story quality

Return JSON:
{
  "plans": [
    {
      "premise": "...",
      "conflict": "...",
      "resolution": "...",
      "constraints_satisfied": ["..."],
      "novelty_note": "..."
    }
  ],
  "selected_plan": "..."
}
```

---

# 10.8 NeoCoder 实现

## 10.8.1 目标

提升：

```text
execution_validity
correctness
flexibility
novelty
appropriateness
```

---

## 10.8.2 Skill chain

```text
constraint_parser
→ algorithmic_strategy_expansion
→ code_candidate_generation
→ execution_verification
→ code_diversity_selection
→ pareto_selection
→ output_normalization
```

---

## 10.8.3 流程

```text
1. 解析输入输出和约束。
2. 生成多个算法策略：
   - brute force
   - optimized
   - dynamic programming
   - graph
   - greedy
   - data structure
   - mathematical simplification
3. 每个策略生成代码候选。
4. 调用已有执行 / test 验证。
5. 优先选择通过测试的代码。
6. 在通过测试的前提下考虑简洁性和多样性。
7. 输出 final code。
```

---

## 10.8.4 重要要求

```text
如果有真实 execution/test harness，必须优先使用真实执行结果。
如果无法执行，只能标记为 llm_check，不得伪装成 execution_pass。
```

---

# 10.9 Transformation 实现

## 10.9.1 目标

提升：

```text
rule_utilization
system_reconstruction
performance_restoration
norm_establishment
old_assumption_removal
goal_coverage
```

---

## 10.9.2 Skill chain

```text
rule_parser
→ old_dependency_mapping
→ breakage_propagation
→ new_primitive_induction
→ architecture_reconstruction
→ performance_restoration
→ norm_establishment
→ residue_audit
→ goal_coverage_check
→ output_normalization
```

---

## 10.9.3 核心原则

Transformation 任务不能只让模型：

```text
遵守规则
解释规则
增加约束
写更长答案
```

必须让模型：

```text
识别旧系统依赖
推导新规则破坏
提出新中间抽象
重建系统机制
恢复性能
建立新规范
检查旧假设残留
```

---

## 10.9.4 Transformation 主 prompt

```text
You are solving a transformational creativity task.

The world rules have changed.
Your task is not to optimize the old system.
Your task is to reconstruct a working system in the new rule-world.

Follow the reconstruction protocol:

1. Old Dependency Mapping
Identify what the old system relied on.

2. Breakage Propagation
For each changed rule, identify which mechanisms fail and why.

3. New Primitive Induction
Invent the minimum necessary new concepts, measurements, interfaces, mechanisms, or coordination principles.

4. System Reconstruction
Build a coherent new system that works under the changed rules.

5. Performance Restoration
Explain how the new system restores key performance.

6. Norm Establishment
Create new standards, terminology, training, verification, or coordination norms.

7. Old-world Residue Audit
Check whether your solution still depends on invalid old assumptions.
Revise if necessary.

8. Goal Coverage
Ensure the solution satisfies:
- rebuild_core_mechanism
- restore_key_performance
- establish_new_norm

Return JSON with all intermediate artifacts and a final answer.
```

---

## 10.9.5 Transformation final answer format

最终提交给 benchmark 的答案建议包含三部分：

```text
1. Rebuilt Core Mechanism
2. Restored Key Performance
3. New Norm / Standard / Training / Verification System
```

但不要输出过长中间推理。

推荐 final answer 模板：

```text
### Rebuilt Core Mechanism
...

### Restoring Key Performance
...

### New Norms, Interfaces, and Verification
...

### Old-Assumption Safeguards
...
```

如果 benchmark adapter 不接受 markdown，则按原 adapter 要求转成纯文本。

---

## 11. Ablation 设置

推荐配置：

```yaml
ablations:
  direct:
    description: 原始 prompt，不使用 TriSkill

  generic_creativity_prompt:
    description: 在原始 prompt 前加 "be creative / think outside the box"

  cot_structured:
    description: 通用结构化 reasoning，不使用 level-specific workflow

  high_temperature:
    description: 提高 temperature，多样性 baseline

  multi_sample:
    description: 多次采样后选择，不使用 skills

  self_refine:
    description: draft → feedback → revise

  triskill_full:
    description: 完整方法

  triskill_level_only:
    description: 只使用 level workflow，不根据 canonical metrics 选择额外 skills

  triskill_metric_only:
    description: 只根据 metrics 选 skills，不使用三层 workflow 顺序

  triskill_without_verifier:
    description: 去掉 verification / appropriateness / residue audit skills

  triskill_wrong_skill_assignment:
    description: 故意把不匹配的 skills 用到任务上

  triskill_length_matched:
    description: 控制 final answer 长度

  direct_long:
    description: direct prompt，但要求输出与 TriSkill 相同长度
```

---

## 12. 最重要的 ablation 预期

### 12.1 w/o verifier

预期：

```text
DAT 可能仍提升。
BATS / RAT / Metaphor 可能下降。
AUT / CS4 novelty 可能上升，但 appropriateness 下降。
Transformation old-world residue 增加。
```

解释：

```text
创造力激发不能只发散，必须有收束。
```

---

### 12.2 wrong skill assignment

例子：

```text
给 DAT 用 transformational residue audit。
给 BATS 用 pure novelty shift。
给 Transformation 用 simple semantic domain expansion。
```

预期：

```text
用错 skill 不提升，甚至下降。
```

解释：

```text
创造力激发需要和创造力层级 / 指标语义匹配。
```

---

### 12.3 length-matched baseline

目的：

```text
证明提升不是因为答案更长。
```

实现：

```text
统计 TriSkill final answer token 数。
让 direct_long 输出相近长度。
或者在评估中加入 output length covariate。
```

---

### 12.4 budget-matched baseline

目的：

```text
证明提升不是因为调用次数更多。
```

实现：

```text
如果 TriSkill 使用 5 次 LLM call，
multi_sample baseline 也允许 5 次 call。
如果 TriSkill 使用 N tokens，
direct budget baseline 也允许近似 N tokens。
```

---

## 13. 实验日志设计

每个 item 需要保存完整日志，但最终提交只用 final answer。

推荐日志结构：

```json
{
  "task_name": "Transformation",
  "item_id": "...",
  "method": "triskill_full",
  "level": "transformational",
  "raw_metrics": ["judge_score"],
  "canonical_metrics": {
    "system_reconstruction": "high",
    "old_assumption_removal": "high"
  },
  "skills": [
    "rule_parser",
    "old_dependency_mapping",
    "breakage_propagation"
  ],
  "skill_trace": [
    {
      "skill": "rule_parser",
      "artifacts": {...},
      "warnings": []
    }
  ],
  "final_answer": "...",
  "parse_success": true,
  "output_length": 842,
  "warnings": []
}
```

---

## 14. 统计分析字段

每条输出建议额外记录：

```text
method
model
task
item_id
level
raw_score
canonical_score_group
output_length
num_llm_calls
total_prompt_tokens
total_completion_tokens
parse_success
```

用于后续分析：

```text
task-level improvement
level-level improvement
profile shift
novelty vs appropriateness trade-off
length-controlled analysis
budget-controlled analysis
failure mode analysis
```

---

## 15. Failure Mode 标注

尤其 Transformation 需要标注失败模式。

推荐字段：

```yaml
transformation_failure_modes:
  rule_paraphrase:
    meaning: 只是复述规则，没有重建机制

  local_patching:
    meaning: 只做局部补丁，没有系统重构

  old_world_residue:
    meaning: 方案仍依赖被改变的旧规则

  no_new_primitives:
    meaning: 没有提出新抽象、新变量、新接口或新机制

  missing_performance_restoration:
    meaning: 没有说明关键性能如何恢复

  missing_norm_establishment:
    meaning: 没有建立新标准、培训、解释或验证体系

  interface_neglect:
    meaning: 忽略旧基础设施、接口或组织协同约束

  cognitive_execution_neglect:
    meaning: 忽略术语、公众理解、培训或执行语言
```

TriSkill full 的预期是：

```text
rule_paraphrase ↓
local_patching ↓
old_world_residue ↓
no_new_primitives ↓
missing_norm_establishment ↓
```

---

## 16. 评估方式

### 16.1 Task-level

每个任务保留原始 benchmark metric：

```text
DAT: pairwise semantic distance
BATS: bats_accuracy
RAT: rat_accuracy
Metaphor: metaphor_accuracy
AUT: fluency / flexibility / originality / elaboration
CreativeMath: fluency / novelty / flexibility / appropriateness
CS4: task-specific metrics + canonical mapping
NeoCoder: execution/test + novelty/flexibility/appropriateness
Transformation: judge score + goal-level scores + failure diagnostics
```

---

### 16.2 Level-level

对每个任务先算 method 相对 baseline 的提升。

推荐：

```text
delta = score(method) - score(direct)
standardized_delta = delta / baseline_task_std
```

然后按 level 汇总：

```text
combinational_gain = average standardized_delta over DAT/BATS/RAT/Metaphor
exploratory_gain = average standardized_delta over AUT/CreativeMath/CS4/NeoCoder
transformational_gain = item-level bootstrap over Transformation
```

---

### 16.3 Profile shift

输出模型的 creativity profile：

```text
Before:
[combinational, exploratory, transformational]

After:
[combinational, exploratory, transformational]
```

关注：

```text
TriSkill 是否改变模型 profile
哪个 level 提升最大
是否 novelty 提升但 appropriateness 下降
Transformation 是否真实提升而非更长答案
```

---

## 17. 推理参数建议

### 17.1 默认参数

```yaml
default_generation_temperature: 0.7
default_verification_temperature: 0.0
default_max_json_tokens: 1024
default_max_final_tokens: 1200
```

---

### 17.2 分阶段参数

```yaml
candidate_generation:
  temperature: 0.7-0.9

verification:
  temperature: 0.0-0.2

final_normalization:
  temperature: 0.0
```

---

### 17.3 分任务建议

```yaml
DAT:
  generation_temperature: 0.9
  verification_temperature: 0.0

BATS:
  generation_temperature: 0.6
  verification_temperature: 0.0

RAT:
  generation_temperature: 0.8
  verification_temperature: 0.0

Metaphor:
  generation_temperature: 0.7
  verification_temperature: 0.0

AUT:
  generation_temperature: 0.9
  verification_temperature: 0.0

CreativeMath:
  generation_temperature: 0.8
  verification_temperature: 0.0

CS4:
  generation_temperature: 0.9
  verification_temperature: 0.0

NeoCoder:
  generation_temperature: 0.7
  verification_temperature: 0.0

Transformation:
  generation_temperature: 0.7
  verification_temperature: 0.0
```

---

## 18. 最小可运行版本

如果时间有限，先实现 MVP。

### 18.1 MVP 目标

```text
先让框架跑通。
先支持 9 个任务。
每个任务至少能通过 workflow 生成 final answer。
先不做复杂 embedding rerank。
先不做自动 controller，只用 YAML 配置选择 skills。
```

---

### 18.2 MVP skills

组合型：

```text
semantic_domain_expansion
relation_abstraction
bridge_search
relation_verification
output_normalization
```

探索型：

```text
constraint_parser
strategy_axis_expansion
candidate_multiplication
appropriateness_check
pareto_selection
output_normalization
```

转换型：

```text
rule_parser
old_dependency_mapping
breakage_propagation
new_primitive_induction
architecture_reconstruction
residue_audit
goal_coverage_check
output_normalization
```

---

### 18.3 MVP 方法组

先跑：

```text
direct
generic_creativity_prompt
cot_structured
triskill_full
triskill_without_verifier
```

之后再加：

```text
multi_sample
self_refine
wrong_skill_assignment
length_matched
budget_matched
```

---

## 19. 与 evalscope 集成方式

推荐不要改原 benchmark adapter 的评分逻辑。

做法：

```text
原始 adapter 负责：
- 读取数据
- 构造原始 prompt
- 解析最终输出
- 计算 metric

TriSkill 负责：
- 接收原始 prompt / item
- 生成 elicited final answer
- 将 final answer 返回给 adapter
```

---

### 19.1 推荐集成点

新增 runner：

```text
evalscope/elicitation/runners/run_elicited_eval.py
```

该 runner 做：

```python
for item in dataset:
    original_prompt = adapter.build_prompt(item)

    if method == "direct":
        answer = llm.generate(original_prompt)

    elif method.startswith("triskill"):
        answer = triskill_elicit(
            task_name=task_name,
            item=item,
            original_prompt=original_prompt,
            llm=llm,
            method_config=method_config
        )

    prediction = adapter.parse_prediction(answer)
    score = metric(prediction, item)
```

---

### 19.2 TriSkill 主函数

```python
def triskill_elicit(task_name, item, original_prompt, llm, method_config):
    profile = load_task_profile(task_name)

    state = ElicitationState(
        task_name=task_name,
        item_id=str(item.get("id", "")),
        raw_item=item,
        original_prompt=original_prompt,
        level=profile.level,
        workflow=profile.workflow,
        raw_metrics=profile.raw_metrics,
        canonical_metrics=profile.canonical_metrics,
        output_schema=profile.output_schema,
        visible_choices=extract_visible_choices(item),
    )

    profile = apply_ablation(profile, method_config)

    executor = WorkflowExecutor(SKILL_REGISTRY)
    state = executor.execute(state, profile, llm)

    final_answer = state.final_answer

    return final_answer
```

---

## 20. 安全检查与数据泄露防护

实现一个统一函数：

```python
def extract_visible_choices(item):
    # 只返回题面明确提供给模型的 choices/options
    # 不返回 reference / candidate_answers / gold_answers
    pass
```

必须维护黑名单字段：

```python
GOLD_FIELD_BLACKLIST = {
    "answer",
    "answers",
    "reference",
    "references",
    "gold",
    "gold_answer",
    "target",
    "label",
    "candidate_answers",
    "correct_answer",
}
```

但注意：

```text
有些任务的 target 字段可能是输入的一部分，也可能是 gold。
需要根据具体 adapter 确认。
如果不确定，默认不放入 prompt。
```

---

## 21. Skill 输出错误处理

每个 skill 要能容忍 LLM 输出坏 JSON。

策略：

```text
1. 尝试解析 JSON。
2. 失败则 regex 提取关键字段。
3. 仍失败则降级为 direct continuation。
4. 记录 warning。
5. 不要让单个 skill 崩掉整个评估。
```

伪代码：

```python
try:
    parsed = call_llm_json(...)
except Exception as e:
    state.warnings.append(f"{self.name} failed: {e}")
    parsed = fallback_output(state)

state.artifacts[self.name] = parsed
return state
```

---

## 22. 最终论文表述

可以直接写入论文的方法段：

```text
We propose TriSkill, a definition-guided and metric-conditioned creativity elicitation framework. TriSkill first assigns each benchmark task to one of three workflows derived from the theoretical definitions of combinational, exploratory, and transformational creativity. It then maps heterogeneous task-specific metrics into canonical creativity objectives, such as semantic diversity, relation validity, fluency, flexibility, novelty, appropriateness, execution validity, rule utilization, and system reconstruction. Finally, TriSkill composes a sequence of inference-time skills that target these objectives and executes the corresponding workflow to generate a final answer. This design separates high-level creativity mechanisms from task-specific output schemas, allowing the same framework to adapt to heterogeneous creativity tasks without using hidden answers or task-specific prompt hacking.
```

中文版本：

```text
我们提出 TriSkill，一个由创造力定义驱动、由指标语义条件化的创造力激发框架。TriSkill 首先根据组合型、探索型和转换型创造力的理论定义，为每个任务分配对应的 workflow；随后将不同任务中的异构指标映射到一组统一的创造力目标，例如语义分散性、关系有效性、流畅性、灵活性、新颖性、合理性、可执行性、规则利用和系统重建；最后根据这些目标组合一系列 inference-time skills，并执行对应 workflow 生成最终答案。该设计将高层创造力机制与任务级输出格式解耦，使同一框架能够适配异构创造力任务，而不依赖隐藏答案或任务特定 prompt hacking。
```

---

## 23. 最终人话版总结

```text
TriSkill 不是一个 prompt。
它是一个固定的创造力激发框架。

组合型创造力：
先联想，再验证关系。

探索型创造力：
先解析约束，再多方向探索，再筛选新颖且合理的解。

转换型创造力：
先找旧系统依赖，再推导新规则破坏，再发明新抽象，最后重建系统并检查旧假设残留。

不同任务的指标名不统一，
所以先映射成 canonical metrics，
再根据这些 metrics 选择 skills。

最后所有中间过程只用于帮助模型生成答案，
提交给 benchmark 的仍然是原 adapter 能解析的 final answer。
```

---

## 24. 实现优先级

### Phase 1：跑通框架

```text
实现 task_profiles.yaml
实现 ElicitationState
实现 SkillRegistry
实现 WorkflowExecutor
实现 OutputNormalizationSkill
实现每个任务最小 skill chain
跑通 direct vs triskill_full
```

---

### Phase 2：完善 skills

```text
组合型：
完善 relation_verification 和 bridge_search

探索型：
完善 constraint_parser、novelty_shift、pareto_selection

转换型：
完善 old_dependency_mapping、breakage_propagation、new_primitive_induction、residue_audit
```

---

### Phase 3：实验对照

```text
direct
generic_creativity_prompt
cot_structured
multi_sample
self_refine
triskill_full
triskill_without_verifier
triskill_wrong_skill_assignment
direct_long
budget_matched
```

---

### Phase 4：分析

```text
task-level improvement
level-level profile shift
novelty vs appropriateness trade-off
length-controlled analysis
budget-controlled analysis
transformation failure mode reduction
ablation contribution
```

---

## 25. 最重要的成功标准

该框架实现后，需要能回答以下问题：

```text
1. TriSkill 是否比 direct prompt 更好？
2. 是否比 generic creativity prompt 更好？
3. 是否比 CoT / self-refine / multi-sample 更好？
4. 提升是否不是因为答案更长？
5. 提升是否不是因为调用次数更多？
6. novelty 是否牺牲了 appropriateness？
7. 错误 skill 分配是否无效或有害？
8. 去掉 verifier 是否导致关系任务和探索任务掉 appropriateness？
9. Transformation 是否真的减少 old-world residue？
10. 三层 creativity profile 是否发生可解释变化？
```

---

## 26. 预期主要结论

理想结果应该是：

```text
1. Combinational:
   DAT 提升明显；
   BATS/RAT/Metaphor 在 verifier 存在时持平或小幅提升；
   没有 verifier 时 accuracy-like 任务可能下降。

2. Exploratory:
   fluency / flexibility / novelty 提升；
   appropriateness 在 verifier 和 pareto_selection 下保持稳定。

3. Transformational:
   提升最大；
   old-world residue 减少；
   local patching 减少；
   new primitive 和 system reconstruction 更充分。

4. Ablation:
   wrong skill assignment 不提升；
   length-matched direct 无法复现 Transformation 提升；
   budget-matched multi-sample 不如 TriSkill 稳定。
```

---

## 27. 最终核心 claim

```text
Creativity elicitation should not be a single generic prompt.
Different creativity levels require different inference workflows.
Different heterogeneous metrics should be abstracted into canonical creativity objectives.
A model can then be guided by metric-conditioned skills to search, verify, and reconstruct solutions in a way aligned with the target form of creativity.
```

中文：

```text
创造力激发不应该是一个通用 prompt。
不同创造力层级需要不同的推理 workflow。
不同任务的异构指标需要先抽象成统一创造力目标。
然后系统根据这些目标组合 skills，引导模型在对应创造力空间中搜索、验证和重建答案。
```

---

## 28. 当前实现状态与约束

当前 `enhance/` 实现遵循以下约束：

```text
1. 不修改 evalscope 测评代码。
2. 不使用 hidden gold answer、candidate_answers、答案表、任务专用符号规则或测试集统计。
3. 只根据组合任务的通用特性优化：
   - direct answer seed
   - relation / context verifier
   - canonical metric guidance
   - output normalization
   - analogy direction、entity type、abstraction level consistency
4. optional multi-seed 只作为可配置软证据，默认不开启硬投票覆盖。
```

当前组合任务 workflow：

```text
Task item
  ↓
safe_item_view：剔除评分字段
  ↓
direct_seed：用原始可见 prompt 产生一个保守候选
  ↓
skill workflow：
  - DAT: semantic_domain_expansion → lexical_validity_check → diversity_filtering
  - BATS: relation_abstraction → relation_verification → lexical_validity_check
  - RAT: bridge_search → relation_verification → lexical_validity_check
  - Metaphor: metaphorical_property_mapping → relation_verification → lexical_validity_check
  ↓
output_normalization：只输出 benchmark schema
```

`limit=50` 验证结果显示，当前非泄露版本在至少一个模型上满足四个组合任务全面增益：

```text
Run: models2_combination_limit50_general_entitytype_triskill_full
Models: /root/benchmark/evalscope/run/models2.json
Sampling: limit=50, max_parallel=64

7B direct:
DAT 3.4429, BATS 0.86, RAT 0.04, Metaphor 0.16

7B TriSkill:
DAT 5.1548, BATS 0.90, RAT 0.06, Metaphor 0.28
```

更大模型目前是部分增益：

```text
32B: DAT 和 RAT 提升，BATS 持平，Metaphor 下降。
14B: BATS 持平，RAT 和 Metaphor 提升，DAT 下降。
```

论文中应如实报告这一点：当前框架已经证明在组合任务上存在可复现的 profile shift，
但不同模型对 DAT / Metaphor 的响应存在差异，后续需要做更系统的模型规模与任务类型分析。
