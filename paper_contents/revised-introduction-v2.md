# Revised Introduction (Version 2) / 引言修改版（第二版）

> 编辑说明(2026-09-04):本版在v1(`revised-introduction.md`)基础上,把叙事腔改成陈述性学术语态,并把ECU严格锚定为"Edge Computing Unit"这一核心定义,去掉了智慧城市/IoT/MEC这类偏离本文实际问题(服务到边缘计算单元的在线分配)的背景包装。
>
> **⚠ 未解决事项,发正文前必须您本人确认**:原稿引用了"Trust-Oriented IoT Service Placement for Smart Cities in Edge Computing (Xu et al., IEEE JIOT 2020)"这篇文献,包括中文版里"徐小龙教授团队"这一具体归属。我无法验证这篇文献是否真实存在、内容是否与引用处的论断吻合——本版暂时保留这条引用,但整体背景已经从"智慧城市/IoT"改成了通用边缘计算表述,如果这篇文献的原始内容确实是面向智慧城市/IoT场景的,现在的引用位置可能不再贴合,需要您核实后确认是否保留、以及保留在什么位置。

---

## 1. English Version (英文版)

### 1. Introduction
The proliferation of distributed edge computing has established the dynamic deployment of services onto geographically distributed **Edge Computing Units (ECUs)** as a standard operational pattern for delivering low-latency computation closer to end users [5, 6]. As multi-tenant services increasingly share physical ECU hardware, co-location of services handling conflicting, sensitive, or competitive data introduces a risk of privacy leakage through shared hardware resources and side-channel exposure [13]. Edge service provisioning is therefore not solely a resource-efficiency optimization problem: the orchestrator must place services to satisfy capacity constraints while strictly preventing conflicting service pairs from being co-located on the same ECU. **This work formulates the problem as a single-objective optimization under two hard constraints, not a multi-objective trade-off**: the sole optimization objective is to maximize resource utilization (AR), subject to (i) a capacity constraint and (ii) a privacy-conflict constraint, both of which must hold with zero violation rather than being weighed against the objective.

This work is organized around three progressively layered questions: what optimization algorithms can solve, what Reinforcement Learning (RL) adds beyond that, and where Safe RL becomes necessary, relative to standard RL, once privacy conflicts are introduced as a hard constraint.

Mathematical programming — Integer Linear Programming (ILP) and Mixed-Integer Linear Programming (MILP) [9, 10] — is the classical optimization approach to service placement. In offline, perfect-information settings, ILP computes mathematically optimal deployment configurations and is the exact-optimization reference method. **ILP relies on explicit mathematical modeling of the environment: constraints and decision variables must be hand-designed for the specific environment, and the scale of variables and constraints required grows as environmental complexity increases (more constraint types, tighter conditional dependencies among them); RL, by contrast, is an environment-perceptive method — a policy learns its decisions through interaction with the environment, without relying on an explicit mathematical characterization of it.** In practical edge deployment settings, however, the ECU set and the service set for a given scenario are fixed and known in advance, but the orchestrator must commit to each service's placement one at a time, in sequence, and each decision, once made, cannot be revisited or revised. **This work uses ILP's offline exact solution as the comparison baseline against which RL and Safe RL are evaluated.**

ILP's offline exact solution is used strictly as an evaluation baseline, solved under the same two hard constraints — capacity and privacy co-location — that define the problem throughout this work, without participating in the online decision process itself. Benchmarking a Safe RL controller against this baseline reveals a stable **0.10-to-0.12 structural gap in resource utilization**, which constitutes the upper bound achievable under an online, single-step, non-reversible decision paradigm; this gap quantifies the competence of Safe RL as an online executor relative to the offline optimum.

Among the two constraints, the **privacy co-location constraint** in particular reveals a structural advantage of Safe RL over classical mathematical programming: **the tractability of formulating complex, non-linear constraints**. Encoding state-dependent, multi-tenant co-location privacy conflicts into an ILP formulation requires linearizing complex conditional logic — for instance, whether placing service $A$ on $ECU_j$ dynamically blocks the co-location of service $B$ — through a large number of binary helper variables and Big-M constants, which substantially increases the NP-hard problem's solving complexity and degrades solver tractability, and requires re-deriving the formulation whenever the constraint set changes. The proposed Safe RL framework, by contrast, decouples the safety-enforcement mechanism from the underlying optimization mathematics. A lightweight, state-dependent **Action Masking** mechanism implemented directly in the simulator environment dynamically excludes privacy-violating ECU actions from the agent's action space at every step, guaranteeing a deterministic zero-violation privacy co-location boundary at runtime without requiring any algebraic modification of the optimization formulation.

A systematic evaluation of diverse learning architectures — masked PPO, Lagrangian dual PPO, and standard DQN/DDQN — under these constraints clarifies two distinct levels of learning-based service placement. First, regarding reward shaping and strategy learning: continuous, quality-graded reward signals, rather than binary success/failure rewards, are the primary driver of policy convergence; under graded rewards, unconstrained RL models learn valid placement policies with success rates rising from zero to approximately 30%. Second, regarding constraint enforcement and safety boundaries: reward shaping determines whether a policy can be learned at all, but it does not substitute for dedicated safety mechanisms in enforcing the privacy boundary. Under identical graded rewards, unconstrained RL models retain a residual privacy-violation rate of 30%-40%, rendering them unusable against a hard privacy requirement regardless of their placement quality; Action Masking structurally eliminates invalid actions and guarantees a zero-violation rate, whereas Lagrangian dual methods provide only soft, probabilistic boundary control.

In summary, this work connects theoretical optimality with practical, secure edge deployment: the offline ILP solution, solved under the full problem of capacity and privacy co-location constraints together, is retained as the evaluation baseline, and Safe RL's constraint-formulation tractability is leveraged to enforce privacy boundaries without mathematical-formulation overhead, yielding a deployable design pattern for secure, edge-native service placement.

---

## 2. Chinese Version (中文版)

### 一、引言 (Introduction)
分布式边缘计算的普及，使服务动态部署到地理分布的**边缘计算单元（Edge Computing Unit, ECU）**上，成为在终端用户附近提供低延迟计算的标准做法 [5, 6]。随着多租户服务日益共享物理 ECU 硬件，处理冲突、敏感或竞争性数据的服务被共置于同一 ECU 时，共享硬件资源与侧信道暴露带来隐私泄露风险 [13]。边缘服务供给因此不只是资源效率优化问题：服务编排必须在满足容量约束的同时，严格避免将存在冲突关系的服务对共置于同一 ECU 上。**本文将该问题形式化为一个优化目标、两条硬性约束的单目标优化问题,而非多目标权衡**:唯一的优化目标是最大化资源利用率(AR),该目标需在(i)容量约束与(ii)隐私冲突约束这两条硬约束下实现——二者是必须零违反的约束条件,不是与优化目标相互权衡的另外两个目标。

本文围绕三个层层递进的问题展开：优化算法能求解到什么程度、强化学习（RL）在此基础上补足了什么、以及在隐私冲突这一硬约束下 Safe RL 相较于普通 RL 的必要性体现在哪里。

数学规划——整数线性规划（ILP）与混合整数线性规划（MILP）[9, 10]——是求解服务放置问题的经典优化算法。在离线且信息完备的场景下，ILP 能够计算出数学上的全局最优部署配置，是精确求解的参照方法。**ILP 依赖对环境的显式数学建模：约束与决策变量需要针对具体环境手工设计，环境复杂度上升（约束种类增多、约束间的条件依赖增强）时，所需建模的变量与约束规模随之增长；RL 则是环境感知型方法，策略通过与环境交互学习决策，不依赖对环境的显式数学刻画。**而在边缘计算的实际部署场景中，对于给定场景，ECU 集合与服务集合（SVC 集合）从一开始就是固定且完整已知的，但编排器必须逐个服务、按顺序依次做出放置决策，且每一次决策一旦执行便不可回溯或修改。**本文以 ILP 的离线精确解作为评估基准，RL 与 Safe RL 的表现均以此为对比对象。**

ILP 的离线精确解严格作为评估基准使用，在本文自始至终采用的容量约束与隐私共置约束这两条硬约束下求解，不参与在线决策过程本身。以此基准衡量 Safe RL 控制器，可得到一个稳定的 **0.10 至 0.12 的资源利用率结构性差距**，这一差距即在线、单步、不可撤回决策范式下可达到的性能上限，量化了 Safe RL 作为在线执行器相对离线最优解的能力边界。

在两条约束中，**隐私共置约束**尤其体现出 Safe RL 相较于传统数学规划（如 ILP）的一项结构性优势：**面对复杂非线性约束时的建模可行性**。将与状态相关的多租户共置隐私冲突纳入 ILP 建模——例如判定服务 A 部署在 ECU_j 上是否会动态阻断服务 B 的共置——需要通过大量二值辅助变量和 Big-M 常数对复杂条件逻辑进行线性化处理，这会显著增加 NP-hard 问题的求解复杂度、降低求解器的可行性，且约束集合每次变化都需要重新推导整套公式。本文提出的 Safe RL 框架将安全执行机制与底层优化数学解耦：在仿真环境中直接实现轻量级、状态相关的**动作掩码（Action Masking）**，在智能体每一步决策时动态排除所有导致隐私冲突的 ECU 动作，在无需对优化建模做任何代数修改的前提下，保证隐私共置边界的确定性零违规。

对多种学习架构（带掩码的 PPO、拉格朗日对偶 PPO、基础 DQN 与 DDQN）在上述约束下的系统性评估，澄清了学习型服务放置中两个不同层面的问题。其一，就奖励塑造与策略学习而言：按分配质量连续分级的奖励信号，而非二元的成败奖励，是策略收敛的主要驱动因素；在分级奖励下，不具备约束机制的普通 RL 模型也能学出有效的放置策略，成功率从零提升至约 30%。其二，就约束执行与安全边界而言：奖励塑造决定的是策略能否学得动，并不能替代专用安全机制对隐私边界的守护——在同样的分级奖励下，不具备约束机制的 RL 模型仍保留 30%~40% 的残余隐私违规率，无论其放置质量如何，都无法满足硬性隐私要求，因此不可用；动作掩码在结构上排除违规动作，保证隐私零违规；拉格朗日对偶提供的则是软性的、概率意义上的边界控制，不构成结构性保证。

综上，本文将理论最优性与实际安全部署相连接：离线 ILP 的精确解在容量约束与隐私共置约束同时存在的完整问题下作为评估基准被保留，Safe RL 在复杂约束下的建模可行性被用于在无需额外数学建模开销的前提下执行隐私边界，二者共同构成一种面向安全、边缘原生服务部署的可落地设计模式。








# 个人评论
ILP作为我们基准，我们不管是RL还是Safe RL都是和同标准的ILP进行比较。
1. 目前RL和Safe RL进行了对比，主要体现RL没有Constraint是完全不行的。
2. ILP目前在实验中已经没有不带Constraint的结果了，都是带Constraint的，所以不能再说什么额外再加上隐私约束
