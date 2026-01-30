# 2026 MCM Problem C（DWTS）建模路线与可行性确认

## 0. 可行性结论（对齐 Gemini 评价）

本题的关键是“反推粉丝投票”这一**不可辨识（identifiability）**问题：官方数据只给了评委分与历史结果，不给绝对票数。

- **我们能可靠得到的量**：每周/每季的**相对粉丝投票强度**（fan vote share / fan vote index）及其不确定性区间。
- **我们无法唯一确定的量**：绝对投票人数（例如“百万票”），除非引入额外外生信息或强先验。

这在 MCM 场景下是完全可接受的：题目要求“estimate / analyze / propose”，并未要求还原绝对票数。我们会在文中把“粉丝票估计”明确表述为**比例/指数 + 区间**，并用稳健性验证支撑可信度。

外生数据（Wikipedia pageviews、州人口、可选 Google Trends）不是“解题必需”，但可以：
- 提供更有说服力的先验（让反推解更集中）；
- 帮助第三题归因（把“人气”拆成可解释的部分）。

## 1. 数据与工程落地（必须先做）

### 1.1 原始数据

- 官方数据：`mcm2026c/2026_MCM_Problem_C_Data.csv`
- 外生数据（已抓取）：
  - Wikipedia pageviews：`data/raw/dwts_wikipedia_pageviews.csv`
  - 2020 州人口：`data/raw/us_census_2020_state_population.csv`
  - （可选）Google Trends：`data/raw/dwts_google_trends.csv`
    - 当前文件可能存在大量 `TooManyRequestsError` 且 `n_points=0`（早期赛季尤甚），因此在主线建模中应作为“可用则用”的补充信号；默认以 Wikipedia pageviews 作为主要人气 proxy。

### 1.2 建模输入表（建议作为“唯一真源”）

#### A. weekly panel（核心）

粒度：`season, week, celebrity`

字段建议：
- `season`
- `week`
- `celebrity_name`
- `pro_name`（ballroom_partner）
- `judge_score_total`（该周合计；或保留 `judge1..judgeK`）
- `judge_score_pct`（百分比法用：该周个人分 / 全体分之和）
- `judge_rank`（排名法用：该周按分排序得到名次，平分要处理）
- `active_flag`（该周是否仍在赛）
- `eliminated_this_week`（该周是否淘汰）

输出位置建议：`data/processed/dwts_weekly_panel.parquet`（或 csv）

#### B. season features（解释与回归）

粒度：`season, celebrity`

字段建议：
- 官方静态特征：`celebrity_industry, celebrity_age_during_season, celebrity_homestate, ...`
- 外生特征：
  - `wiki_pageviews_sum/mean/max`（赛季窗口）
  - `state_population_2020`
  - （可选）`trends_mean/max`

输出位置建议：`data/processed/dwts_season_features.parquet`

### 1.3 Pipeline 文件命名（按 conventions）

建议按你们的约定：`src/mcm2026/pipelines/mcm2026c_q<k>_<verb>_<object>.py`

- `mcm2026c_q0_build_weekly_panel.py`（数据落地）
- `mcm2026c_q1_estimate_fan_votes.py`
- `mcm2026c_q2_compare_voting_systems.py`
- `mcm2026c_q3_explain_fan_votes_and_scores.py`
- `mcm2026c_q4_design_new_system.py`

## 2. 第一题：估算粉丝投票（核心：逆向约束 + 采样/优化）

### 2.1 目标与输出

**目标**：对每个赛季每周，估计每位选手的粉丝投票份额 `P_fan(i)`（或排名 `R_fan(i)`）的分布。

**主要输出**：
- `E[P_fan(i)]`、`median[P_fan(i)]`、`CI_95%`（或分位数区间）
- `fan_vote_index`：把份额映射为便于比较的指数（例如 `logit(P_fan)` 或标准化 z-score）
- `certainty`：确定性指标（例如后验熵、可行解比例、区间宽度）

### 2.2 关键假设（最小集合）

- 每周的淘汰由当周规则（rank-based / percent-based）决定；允许以“噪声”形式放宽（见 2.4）。
- 粉丝票只需要**相对比例**，不需要绝对人数。
- 同一周投票总量被归一化：`sum_i P_fan(i) = 1`。

### 2.3 约束构造

#### 百分比法（Percentage）

- 已知：`P_judge(i) = judge_score_total(i) / sum_j judge_score_total(j)`
- 未知：`P_fan(i)`，且 `P_fan(i) >= 0, sum P_fan = 1`
- 总分：`T(i) = P_judge(i) + P_fan(i)`
- 淘汰约束（硬约束版）：实际淘汰者 `e` 满足：`T(e) = min_i T(i)`

#### 排名法（Rank）

- 已知：`R_judge(i)`（按评委分排名）
- 未知：`R_fan(i)`（粉丝票排名，为 1..n 的排列）
- 总排名：`S(i) = R_judge(i) + R_fan(i)`
- 淘汰约束（硬约束版）：`S(e) = max_i S(i)`

### 2.4 求解方法（推荐：Monte Carlo/SMC；备选：优化）

#### 方法 A：Monte Carlo + 拒绝采样（易实现、可解释）

- 百分比法：
  - 先验：`P_fan ~ Dirichlet(alpha)`
  - `alpha` 默认全 1（均匀）；也可用外生数据增强（见 2.5）
  - 抽样 `P_fan`，计算 `T(i)`，保留满足淘汰约束的样本
  - 得到 `P_fan(i)` 的后验样本集合

- 排名法：
  - 先验：对 `R_fan` 取均匀随机排列
  - 计算 `S(i)`，保留满足淘汰约束的排列
  - 统计 `R_fan` 的后验分布（可转换成“相对票强度”）

#### 方法 B：软约束（更现实，也更稳健）

硬约束可能导致某些周可行解很少（或与现实不完全一致）。可将淘汰规则建成概率：

- 令 `Pr(eliminate=i) ∝ exp(-T(i)/tau)`（百分比法）或 `exp(S(i)/tau)`（排名法）
- 用 `tau` 控制噪声：`tau→0` 接近硬淘汰
- 通过重要性采样/SMC 得到后验

#### 方法 C：最优化（用于“点估计”）

构造一个目标让“预测的全季名次/淘汰周”与真实最接近：
- 决策变量：每周 `P_fan(i)` 或参数化 `P_fan(i)=softmax(z_i)`
- 约束：`P_fan` simplex
- 目标：最小化淘汰/名次的损失（例如 hinge loss）

推荐在论文主线用 A/B（输出区间、可解释），优化作为补充。

### 2.5 外生数据如何并入（让反推更“有自信”）

- Wikipedia pageviews：作为“人气先验”
  - 对 Dirichlet 先验取 `alpha_i = 1 + c * norm(pageviews_i)`（c 为超参数）
  - 解释：pageviews 越高，先验认为该选手粉丝票比例更可能高
- 州人口：作为粗 proxy
  - 可作为 pageviews 缺失时的退路：`alpha_i = 1 + c * norm(state_population)`
- Google Trends（可选）：与 pageviews 类似，但需对 `trends_status` 与 `n_points` 做质量控制；若缺失/报错则不纳入特征或先验（避免引入系统性偏差）。

注意：外生信号只影响“先验”，最终仍由淘汰约束与评委分纠正，不会变成纯外生回归。

### 2.6 验证与可解释性输出

- 每周：可行样本数、`certainty`（熵/区间宽度）
- 全季：
  - 预测冠军/进入决赛概率（由后验样本产生）
  - “争议指数”：评委排名与粉丝票排名分歧程度（Spearman 相关等）

## 3. 第二题：两种规则比较 + 争议案例 + judge save（核心：反事实模拟）

### 3.1 目标与输出

- 把同一季的数据放在两套规则下重算：
  - 淘汰者概率变化、冠军概率变化
  - 粉丝权重敏感性（人气断层在 percent 法下更强）

**输出建议**：
- “机制差异表”：每季在两规则下冠军变化率、淘汰周变化率
- “争议案例复盘”：例如 Bobby Bones
- 引入 judge save（评委二选一）的结果变化

### 3.2 模拟方式

基于第一题得到的后验样本：
- 对每个后验样本（每周一组 `P_fan` 或 `R_fan`），在另一套规则下重新计算淘汰/晋级
- 统计反事实下的概率分布（而不是给单一结论）

### 3.3 judge save 的建模

在每周淘汰前加入：
- 先按总分选出 bottom-2
- 评委从 bottom-2 中救一个

评委救人规则可取：
- 纯按评委分救高者（最小假设）
- 或引入噪声（以体现“节目效果/裁判偏好”）

## 4. 第三题：选手特征 + 专业舞伴影响（核心：回归/混合效应/层级）

### 4.1 两条线并行（分别解释“技术”和“人气”）

- 技术线（评委分）：
  - 因变量：`judge_score_pct`（周）或 `season_avg_judge_pct`（季）
- 人气线（粉丝票）：
  - 因变量：第一题输出的 `fan_vote_index`（周/季）

### 4.2 特征工程

- 选手：年龄、行业、home state
- 舞伴（pro）：
  - 固定效应/随机效应（建议 random intercept）
  - 可加“历史战绩”特征：例如过往进入决赛次数（需要从本数据计算即可，不必外抓）
- 外生：pageviews、州人口、（可选）trends

### 4.3 建模选择

- 快速 baseline（工程内已具备）：`mcm2026.models.baseline_ml`
  - 回归：Ridge
  - 分类：LogisticRegression（例如预测“能否进入决赛”）

- 论文更强方案（推荐）：statsmodels 混合效应
  - `fan_vote_index ~ age + industry + pageviews + (1|pro) + (1|season)`
  - 输出系数、置信区间、显著性与解释

### 4.4 验证

- 按 season 分组的交叉验证（leave-one-season-out）
- 与不用外生数据的模型对比（看解释力/泛化是否提升）

## 5. 第四题：提出新系统（核心：机制设计 + 指标 + 仿真）

### 5.1 定义“更好/更公平”的量化指标

建议至少包含：
- **专业性保护**：冠军的评委排名分位数（越靠前越好）
- **人气表达**：粉丝票对结果的边际影响（不能为 0，否则失去互动）
- **鲁棒性**：对极端粉丝票（断层）不敏感（避免“低技术高票”压倒性夺冠）
- **可解释/可执行**：规则简单，观众能理解

### 5.2 候选规则模板（可选其一作为主推荐）

- 非线性压缩粉丝票：
  - `P_fan' = softmax(log(P_fan + eps) / k)` 或 `sqrt(P_fan)`
  - 直觉：降低“人气断层”的压制力

- 动态权重：
  - 前期粉丝权重大，后期评委权重提高

- 技术门槛 + 粉丝加成：
  - 评委分低于阈值不能进决赛（阈值可由历史分布定）

- judge save 常态化：
  - 每周 bottom-2 由评委救一个

### 5.3 证明方式（必须仿真）

- 用第一题后验样本做仿真：
  - 在旧规则 vs 新规则下跑完整季
  - 汇总指标与争议案例（如 Bobby Bones）
- 给出 trade-off 图表：公平性提升 vs 观众投票影响保留

## 6. 交付物（对应论文/备忘录）

### 6.1 表格（outputs/tables）

- 每周 `fan_vote_index` 与区间
- 机制对比的冠军/淘汰变化概率
- 特征影响回归表（含 pro 效应）
- 新机制指标对比表

### 6.2 图（outputs/figures）

- 每季争议指数时间序列
- 案例季（如 27）在不同机制下冠军概率条形图
- pro 随机效应/固定效应可视化
- 新机制 trade-off 曲线

## 7. 我们的“自信来源”（写给评委看的）

- 反推不是拍脑袋：
  - 约束来自规则 + 已知淘汰/名次
  - 输出是分布与区间，并给出确定性指标
- 外生数据有据可查：
  - pageviews 与人口数据可复现、带元数据
  - 外生信号只做先验/解释，不强行替代规则约束
- 稳健性：
  - 机制切换点（如 season 28）当作参数做敏感性分析
  - 硬/软约束两版对照

## 8. 炫技（加分）模块（但仍保持可复现与可解释）

- **不确定性可视化**：
  - Q1 输出每周每人的后验区间宽度/熵（certainty）热图，展示“哪些周/哪些人可辨识性更强”。
- **反事实仿真 + 争议复盘**：
  - 用 Q1 后验样本在 Rank/Percent/Judge Save 下重跑整季，输出冠军概率与淘汰路径概率（而非单次模拟）。
- **层级/混合效应（解释 pro dancer）**：
  - 对 pro dancer 做随机效应，给出效应分布与区间，并与 baseline 线性/岭回归对照。
- **机制设计的指标化 trade-off 曲线**：
  - 为新规则定义“技术保护/人气表达/鲁棒性/可解释性”指标，给出旧规则 vs 新规则的对比表与 trade-off 图。

---

## 附：下一步最小可执行顺序

1. `q0_build_weekly_panel`：把官方 wide 表转成 weekly panel（含 rank/pct）
2. `q1_estimate_fan_votes`：先做百分比法（Dirichlet + rejection），输出后验与 certainty
3. `q2_compare_voting_systems`：用后验做反事实 + judge save
4. `q3_explain_*`：混合效应解释 pro 与特征
5. `q4_design_new_system`：定义指标 + 仿真对比
