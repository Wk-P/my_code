# Problem 3: RL for Service Deployment — Without Constraint Consideration

## 目录

1. [问题背景回顾](#1-问题背景回顾)
2. [为什么需要 RL？](#2-为什么需要-rl)
3. [MDP 建模](#3-mdp-建模)
4. ["不考虑约束" 的设计含义](#4-不考虑约束-的设计含义)
5. [环境设计](#5-环境设计)
6. [算法选择](#6-算法选择)
7. [网络架构](#7-网络架构)
8. [训练配置](#8-训练配置)
9. [评估指标](#9-评估指标)
10. [实现路线图](#10-实现路线图)
11. [预期行为与局限性](#11-预期行为与局限性)

---

## 1. 问题背景回顾

### P_simple 问题定义

| 符号 | 含义 |
|------|------|
| $N$ | ECU 数量 |
| $M$ | 服务数量（$M \leq N$） |
| $e_j$ | 第 $j$ 个 ECU 的 VM 容量 |
| $n_i$ | 第 $i$ 个服务的 VM 需求 |
| $x_{ij} \in \{0,1\}$ | 决策变量：服务 $i$ 是否分配到 ECU $j$ |

**目标（最大化平均资源利用率 AR）：**

$$\max \; AR = \frac{1}{M} \sum_{j=0}^{N-1} \sum_{i=0}^{M-1} x_{ij} \cdot \frac{n_i}{e_j}$$

**约束（P2 ILP 中显式满足，P3 RL 中不显式处理）：**

$$\sum_{j} x_{ij} = 1 \quad \forall i \quad \text{（每个服务必须部署到且仅部署到一个 ECU）}$$

$$\sum_{i} x_{ij} \leq 1 \quad \forall j \quad \text{（每个 ECU 最多承载一个服务）}$$

$$\sum_{i} x_{ij} \cdot n_i \leq e_j \quad \forall j \quad \text{（容量约束：需求不超过 ECU 容量）}$$

### P2（ILP）的结果作为上界基准

P2 通过 PuLP 求解器（CBC）得到**全局最优解**，其 AR 值作为 P3 RL 方法的理论上界对比基准。

---

## 2. 为什么需要 RL？

| 对比维度 | P2（ILP） | P3（RL） |
|---------|-----------|---------|
| 求解质量 | 全局最优 | 近似解 |
| 计算复杂度 | NP-hard，大规模时指数爆炸 | 推理时 $O(1)$（常数时间） |
| 可扩展性 | $N > 100$ 时开始变慢 | $N = 10^5$ 仍可实时推理 |
| 约束处理 | 硬约束，数学保证满足 | **不显式处理**（P3）→ 可能违反 |
| 泛化能力 | 每次实例需重新建模求解 | 训练后可泛化到新实例 |

**RL 的核心价值**：将组合优化问题转化为序贯决策问题，训练后可以在毫秒内给出可用解。

---

## 3. MDP 建模

### 3.1 状态空间（State Space）

每一步的状态 $s_t$ 需要包含当前决策所需的完整信息：

$$s_t = \left[ \underbrace{d_t}_{\text{当前服务需求}}, \underbrace{AR_t}_{\text{累积利用率}}, \underbrace{r_0^t, r_1^t, \ldots, r_{N-1}^t}_{\text{各 ECU 剩余容量百分比}} \right]$$

| 维度 | 内容 | 值域 | 说明 |
|------|------|------|------|
| 0 | 当前服务需求（归一化） | $[0, 1]$ | $n_t / \max(e_j)$ |
| 1 | 当前累积 AR | $[0, 1]$ | 已分配步的平均利用率 |
| 2 ~ N+1 | 各 ECU 剩余容量百分比 | $[0, 1]$ | $\text{remaining}_j / e_j$ |

> **向量维度**：$N + 2$

### 3.2 动作空间（Action Space）

$$a_t \in \{0, 1, \ldots, N-1\}$$

离散动作：选择将当前服务 $t$ 分配给第 $a_t$ 个 ECU。

> **动作空间大小**：$N$（随 ECU 数量线性增长）

### 3.3 奖励函数（Reward Function）— 不考虑约束版本

```
P3（不考虑约束）的奖励设计原则：
  - 不对约束违反行为施加显式惩罚
  - 仅在 episode 末尾给出 AR 作为奖励
  - 中间步骤奖励为 0（稀疏奖励）
```

$$r_t = \begin{cases} AR_T & \text{if } t = T \text{（最后一步）} \\ 0 & \text{otherwise} \end{cases}$$

**与 P2 的区别**：
- P2 的 ILP 约束是**硬约束**（数学上强制满足）
- P3 的 RL 奖励函数**不包含任何约束惩罚项**（无 Lagrangian 乘子、无 action masking、无安全层）
- 如果 agent 将服务分配到容量不足的 ECU，**不终止 episode，不给负奖励**，只记录违反情况

### 3.4 Episode 结构

```
episode 开始
  ├── reset(): 随机生成 N 个 ECU，M 个服务
  │
  ├── step 0: 分配 service_0 → ECU[a_0]
  ├── step 1: 分配 service_1 → ECU[a_1]
  ├── ...
  ├── step M-1: 分配 service_{M-1} → ECU[a_{M-1}]
  │                ↓
  └── episode 结束，reward = AR（所有已分配服务的平均利用率）
```

---

## 4. "不考虑约束" 的设计含义

### 与 P2、P4 的三方对比

| | P2（ILP） | P3（RL，无约束） | P4（RL，有约束） |
|--|-----------|-----------------|-----------------|
| 容量约束处理 | 硬约束（LP内） | ❌ 不处理 | ✅ Action Masking / Lagrangian |
| 单ECU单服务约束 | 硬约束（LP内） | ❌ 不处理 | ✅ 显式处理 |
| 约束违反率 | 0%（保证） | 可能 > 0% | 接近 0% |
| AR 性能 | 最优上界 | 较高但违约 | 略低但合法 |

### P3 环境中的"无约束"实现方式

1. **允许超载分配**：agent 选择容量不足的 ECU 时，不终止 episode，也不给 -1 惩罚，而是记录违反并继续
2. **允许重复分配**：agent 可以将多个服务分配到同一 ECU（违反单ECU单服务约束），同样只记录不惩罚
3. **AR 计算仍然进行**：即使违约，AR 仍基于实际（可能违约的）分配结果计算
4. **评估时统计违反率**：运行时记录 `constraint_violation_rate`，与 P4 进行对比

---

## 5. 环境设计

### 5.1 与现有 `env/env.py` 的对比

| 特性 | 现有 `env.py`（P_base） | P3 环境 |
|------|------------------------|---------|
| 容量不足时 | reward=-1，episode 终止 | 记录违反，episode 继续 |
| 重复分配 ECU | 允许（容量扣减） | 允许，记录违反 |
| 奖励 | 最终 AR 或 -1 | 最终 AR（始终） |
| 观测 | $[d_t, AR_t, r_0, \ldots, r_{N-1}]$ | 相同 |

### 5.2 P3 环境伪代码

```python
class P3Env(gym.Env):
    """
    Service Deployment Environment - WITHOUT constraint enforcement
    Agent is allowed to make infeasible assignments; violations are tracked but NOT penalized.
    """
    
    def step(self, action: int):
        service = self.services[self._step]
        
        # ─── 约束检查（只记录，不惩罚）───
        violation = False
        if self.remaining_vms[action] < service.required_vms:
            self.capacity_violations += 1  # 记录容量违反次数
            violation = True
        if self.ecu_assigned[action]:
            self.single_service_violations += 1  # 记录单ECU多服务违反
            violation = True
        
        # ─── 无论是否违反，都执行分配 ───
        ru = service.required_vms / self.initial_vms[action]
        self.remaining_vms[action] -= service.required_vms   # 允许变负（超载）
        self.ecu_assigned[action] = True
        
        # ─── AR 更新 ───
        self.ar = (self.ar * self._step + ru) / (self._step + 1)
        self._step += 1
        
        done = (self._step >= self.M)
        reward = float(self.ar) if done else 0.0  # 稀疏奖励，仅末尾给 AR
        
        info = {
            "ar": self.ar,
            "violation": violation,
            "capacity_violations": self.capacity_violations,
            "single_service_violations": self.single_service_violations,
        }
        return self._get_obs(), reward, done, False, info
```

### 5.3 关键追踪指标（Info 字典）

```python
info = {
    "ar": float,                        # 当前累积 AR
    "capacity_violations": int,         # 容量超载次数（本 episode）
    "single_service_violations": int,   # 单ECU多服务违反次数
    "total_violations": int,            # 总违反次数
    "violation_rate": float,            # 违反步数 / M
}
```

---

## 6. 算法选择

### 推荐：PPO（Proximal Policy Optimization）

```
PPO 适合本问题的理由：

1. 离散动作空间（Discrete(N)）→ PPO 的 softmax policy 直接适配
2. 稀疏奖励（仅末尾有 AR）→ PPO 的 GAE 优势估计比 DQN 更稳定
3. Episode 长度固定为 M 步 → 可以精确计算 V(s) baseline
4. 状态空间连续（Box(N+2,)）→ MLP policy 即可
5. 实现简单，stable-baselines3 直接可用
```

| 算法 | 适合场景 | 不适合 |
|------|---------|--------|
| **PPO** ✅ | $N \leq 2000$，稳定训练 | 超大 $N$（$>10^4$）动作空间 |
| DQN | 大 $N$，off-policy 样本复用 | 稀疏奖励收敛慢 |
| A2C | 快速迭代实验 | 收敛质量不如 PPO |

### PPO 关键超参数解释

```python
PPO(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,      # Adam 学习率
    n_steps       = 1024,      # 每次更新收集的步数（rollout buffer size）
    batch_size    = 128,       # minibatch 大小
    n_epochs      = 10,        # 每次 rollout 数据重复利用次数
    gamma         = 0.99,      # 折扣因子（episode 末尾奖励需要 gamma^M 的折扣）
    gae_lambda    = 0.95,      # GAE λ（偏差-方差权衡）
    clip_range    = 0.2,       # PPO clip 系数（防止策略更新过大）
)
```

> ⚠️ **稀疏奖励注意**：因为奖励只在最后一步出现，`gamma=0.99` 在 $M=10$ 时折扣因子为 $0.99^{10} \approx 0.90$，影响不大；但 $M=100$ 时折扣因子为 $0.99^{100} \approx 0.37$，需要考虑使用 `gamma=0.999` 或**稠密奖励 shaping**。

---

## 7. 网络架构

### MLP Policy 架构

```
输入层: (N+2,)
    ↓
隐藏层 1: Linear(N+2, 256) + ReLU
    ↓
隐藏层 2: Linear(256, 256) + ReLU
    ↓
    ├── Actor head:  Linear(256, N) + Softmax  → π(a|s)
    └── Critic head: Linear(256, 1)            → V(s)
```

### 自适应架构（根据 N）

| $N$ | 隐藏层配置 |
|-----|----------|
| $N \leq 100$ | `[256, 256]` |
| $N \leq 1000$ | `[512, 512]` |
| $N > 1000$ | `[1024, 512]` |

```python
policy_kwargs = dict(net_arch=[256, 256])  # 隐藏层配置
```

---

## 8. 训练配置

### 8.1 推荐训练参数

```python
# 训练配置
N         = 10        # ECU 数量（从小开始验证）
M         = 10        # 每 episode 的服务数量
SEED      = 42
TOTAL_STEPS = 200_000  # 总训练步数

# 环境参数
VMS_RANGE = (10, 101)  # ECU VM 容量范围 [10, 100]
REQ_RANGE = (1, 10)    # 服务需求范围 [1, 9]
```

### 8.2 训练流程

```
Phase 1: 环境验证
  → 手动跑几个 episode，确认 step/reset/obs 正确
  → 检查 info 中 violation 统计是否正常

Phase 2: 随机基线测试（Random Baseline）
  → 随机策略跑 300 episodes
  → 记录 AR 均值、标准差、违反率

Phase 3: PPO 训练
  → train 200k steps
  → 用 ARCallback 记录每 episode 的 AR
  → 监控 violation_rate 变化趋势

Phase 4: 评估
  → 用训练好的模型跑 300 episodes（确定性策略，deterministic=True）
  → 统计 AR 和约束违反率
  → 与 P2（ILP 最优解）和随机基线对比
```

### 8.3 训练监控回调

```python
class P3Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_ars         = []
        self.episode_violations  = []   # 每 episode 的违反次数
    
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_ars.append(info["episode"]["r"])
                self.episode_violations.append(info.get("total_violations", 0))
        return True
```

---

## 9. 评估指标

评估时对比 P2（ILP）、随机基线、P3（RL 无约束）三者：

| 指标 | 含义 | 期望（P3 vs P2） |
|------|------|----------------|
| **平均 AR** | 所有 episode 的 AR 均值 | P3 ≈ P2 的 80%~95% |
| **AR 标准差** | AR 稳定性 | P3 > P2（P2 每次都是最优） |
| **约束违反率** | `violation_steps / total_steps` | P3 > 0%，P2 = 0% |
| **容量违反次数/episode** | 平均每 episode 超载次数 | P3 可能 > 0 |
| **单ECU多服务违反/episode** | 平均每 episode 重复分配次数 | P3 可能 > 0 |

### 评估输出格式（示例）

```
==========================================
        Evaluation Results (300 episodes)
==========================================
Method          | AR (mean±std)  | Violation Rate
----------------|----------------|---------------
P2 (ILP)        | 0.823 ± 0.000  | 0.00%
Random Baseline | 0.651 ± 0.042  | 0.00%*
P3 (RL w/o C)   | 0.754 ± 0.031  | 12.3%
==========================================
* Random baseline 在原始 env 中会 early stop，此处为 P3 env 结果
```

---

## 10. 实现路线图

### 文件结构

```
problem3_single/
├── docs.md                 ← 本文档
├── env_p3.py               ← P3 专用环境（无约束惩罚版）
├── train_p3.py             ← 训练主脚本（PPO）
├── evaluate_p3.py          ← 评估脚本（对比 P2、Random、P3）
├── callbacks.py            ← 训练回调（记录 AR + 违反情况）
├── config.py               ← 超参数配置
└── results/                ← 训练结果输出
    ├── training_curve.png
    ├── comparison_bar.png
    └── results.json
```

### 实现步骤（有序）

```
Step 1: 实现 env_p3.py
  → 复制 env/env.py 的结构
  → 修改 step() 的违反处理逻辑（记录但不惩罚，不终止）
  → 添加 violation 统计到 info 字典
  → 运行 __main__ 手动测试

Step 2: 实现 train_p3.py
  → 加载 P3Env，用 Monitor 包装
  → 配置 PPO（stable-baselines3）
  → 用 P3Callback 记录 AR 和 violation
  → 保存训练好的模型（model.save()）

Step 3: 实现 evaluate_p3.py
  → 加载保存的模型
  → 跑 300 episodes，统计 AR + 违反率
  → 读取 P2 的 ILP 结果（从 problem2/results/ 加载）
  → 生成对比图表

Step 4: 分析结果
  → 关注：RL 是否学会了"自然地"避免违约？
  → 关注：与 ILP 的 AR 差距有多大？
  → 关注：违反率随训练步数的变化趋势？
```

---

## 11. 预期行为与局限性

### 预期训练行为

```
训练初期（0 ~ 50k steps）:
  → agent 几乎随机选择 ECU
  → 高违反率（~30-50%），AR 接近随机基线

训练中期（50k ~ 150k steps）:
  → agent 开始学习"选容量大的ECU"提升 AR
  → 违反率开始下降（因为选大容量 ECU 自然不容易超载）
  → AR 开始超越随机基线

训练后期（150k ~ 200k steps）:
  → AR 趋于稳定，可能仍有少量违反
  → 违反率不一定降为 0（因为没有显式惩罚）
```

### 局限性

| 局限性 | 原因 | P4 中的解决方案 |
|--------|------|----------------|
| 约束违反不为零 | reward 中无约束惩罚 | Action Masking / Lagrangian 惩罚 |
| AR 低于 P2 最优 | RL 是近似方法 | 更多训练步数，更好的 reward shaping |
| 泛化性有限 | 训练时固定 N、M | 随机化 N、M 进行 curriculum learning |
| 稀疏奖励收敛慢 | 只有最后一步有奖励 | 添加 reward shaping（中间步骤给增量 AR） |

### P3 的研究价值

> P3 的核心研究问题：  
> **"当 RL agent 不被显式告知约束时，它能否通过纯粹的奖励最大化隐式学会满足约束？"**
> 
> 答案的对比将直接揭示 P4（约束感知 RL）的必要性与改进幅度。

---

*文档版本：v1.0 | 日期：2026-03-11*
