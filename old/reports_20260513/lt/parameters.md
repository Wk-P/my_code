# 实验参数说明 — ecu_lt_svc_p（ECU < SVC，N=10, M=15）

## 1. 场景参数

| 参数 | 值 | 说明 |
|------|----|------|
| N_ECUS | 10 | ECU 数量 |
| N_SVCS | 15 | 服务（SVC）数量，**多于 ECU 数量** |
| ECU 容量范围 | [50, 200]，步长 5 | 每个 ECU 的 VM slot 容量，从该区间均匀采样（无放回） |
| SVC 需求范围 | [10, 100]，步长 5 | 每个服务所需 VM slot 数，从该区间均匀采样（无放回） |
| 冲突集数量 | 10 | 每个 scenario 生成 10 条冲突约束 |
| 冲突集大小 | [2, N_SVCS=15] | 每条冲突集随机选取 2~15 个服务的 ID |
| 场景总数 | 200 | 独立生成的 scenario 数 |
| 训练集 | 160 （80%） | 由 SEED 决定的确定性随机划分 |
| 测试集 | 40 （20%） | 训练时不可见，用于最终评估 |

**与 eq 场景的区别：** N_SVCS=15（多出 5 个服务），需求超过供给，放置难度最高。ILP 最优 AR（0.74）最高，因为更多服务竞争有限 ECU 容量，迫使密集打包。lt 场景中冲突约束可能强制部分服务无法放置（infeasible），使约束处理能力成为关键差异。

**lt 场景专属特性：** 场景生成时至少包含一条大小 > N_ECUS 的冲突集，保证存在至少一个理论上不可满足的极端冲突，从而测试方法的违反容忍和优化能力。

**约束定义：**
- **容量约束**：部署到某 ECU 的所有服务的 requirement 之和 ≤ 该 ECU 的 capacity
- **冲突约束**：同一冲突集中的任意两个服务不能被部署到同一 ECU

---

## 2. 训练基础参数

| 参数 | 值 | 说明 |
|------|----|------|
| TRAIN_SEED | 0, 1, 2, 3, 4 | 控制 train/test 划分和 RL 初始化的随机种子，通过环境变量 `TRAIN_SEED` 传入 |
| 实验种子数 | 5 | 取 5 个独立种子的结果进行均值±标准差统计 |
| 配置文件 | `training_steps_config.py` | 集中管理各 problem 的训练步数上限 |

---

## 3. PPO 算法参数（P3 / P4 / P6 共用）

> 适用于：`problem3_ppo`、`problem4_ppo_mask`、`problem6_ppo_opt`

| 参数 | 值 | 说明 |
|------|----|------|
| 总训练步数 | 5,000,000 | 与环境交互的总 timestep 数 |
| 并行环境数 (N_ENVS) | 40 | DummyVecEnv 并行收集轨迹 |
| 学习率 (PPO_LR) | 3×10⁻⁴ | Adam 优化器学习率 |
| Rollout 长度 (N_STEPS) | 256 | 每次更新前从每个并行环境收集的步数 |
| Mini-Batch 大小 (BATCH_SIZE) | 128 | SGD mini-batch 大小 |
| 更新轮数 (N_EPOCHS) | 10 | 每次 rollout 数据重复使用 10 次 |
| 折扣因子 (GAMMA) | 0.999 | 高折扣，M=15 步 episode 更需要长程信用分配 |
| GAE lambda | 0.95 | 广义优势估计偏差-方差权衡 |
| Clip 范围 (CLIP_RANGE) | 0.2 | PPO 截断比率 |
| 网络结构 (NET_ARCH) | [256, 256] | Actor 和 Critic 共享的两层 MLP |
| 设备 (DEVICE) | auto | 优先使用 CUDA，否则 CPU |
| Torch 线程数 | 8 | CPU 推理并发线程数 |

### PPO 原理简述

PPO（Proximal Policy Optimization）是一种 on-policy actor-critic 算法。每轮迭代：
1. 用当前策略 π_old 在 N_ENVS 个并行环境中采集 N_STEPS 步轨迹
2. 计算广义优势估计 (GAE)：Â_t = Σ (γλ)^k δ_{t+k}，其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
3. 对同一批数据重复 N_EPOCHS 次梯度更新：
   - Actor loss：L_CLIP = E[min(r_t Â_t, clip(r_t, 1-ε, 1+ε) Â_t)]，r_t = π/π_old
   - Critic loss：均方误差 (V(s_t) - V_target)²
4. 更新 π_old ← π_new，开始下一轮采集

---

## 4. Lagrangian PPO 参数（P5 专用）

> 适用于：`problem5_ppo_lagrangian`

| 参数 | 值 | 说明 |
|------|----|------|
| 总训练步数 | 5,000,000 | 同上 |
| 并行环境数 | 40 | 同上 |
| 学习率 | 3×10⁻⁴ | 同上 |
| N_STEPS | 128 | 较短 rollout，M=15 步 episode 下更频繁更新 |
| BATCH_SIZE | 256 | 较大 batch |
| N_EPOCHS | 10 | 同上 |
| GAMMA | 0.99 | 略低于 P3/P4/P6 |
| GAE lambda | 0.95 | 同上 |
| CLIP_RANGE | 0.2 | 同上 |
| Actor 网络 (pi) | [256, 256] | Actor MLP |
| Critic 网络 (vf) | [512, 512] | Critic 使用更宽网络 |
| **LAMBDA_INIT** | 0.1 | λ 初始值 |
| **LAMBDA_LR** | 0.005 | 对偶上升步长 |
| **LAMBDA_TARGET** | 0.0 | 目标违反率：零违反 |
| **LAMBDA_MAX** | 5.0 | λ 上限 |
| **LAMBDA_UPDATE_WINDOW** | 20 | 每 20 个 episode 更新一次 λ |
| **LAMBDA_WARMUP_EPISODES** | 0 | 无预热，从第 1 episode 开始对偶上升 |

### Lagrangian 松弛原理

```
maximize  E[Σ (r_t - λ · c_t)]
λ ← clip(λ + LAMBDA_LR × (avg_viol_rate - LAMBDA_TARGET), 0, LAMBDA_MAX)
```

在 lt 场景中，由于资源紧缺且存在强制不可满足的冲突集，λ 往往需要上升到更高值才能控制违反率，但同时也会压低 AR。这一张力在 lt 场景中表现最为突出（见 P5 违反率 0.089 vs eq 场景的 0.0015）。

---

## 5. DQN / DDQN 参数

> 适用于：`problem_dqn`、`problem_ddqn`

| 参数 | 值 | 说明 |
|------|----|------|
| 总训练步数 | 3,000,000 | off-policy 效率更高 |
| 并行环境数 | 12 | DummyVecEnv |
| 学习率 (DQN_LR) | 1×10⁻³ | 比 PPO 高一个量级 |
| 经验回放缓冲区 | 100,000 | 存储历史转移 |
| 开始学习步数 | 64 | 最少 64 条经验后开始更新 |
| Mini-Batch 大小 | 64 | 从 replay buffer 中随机采样 |
| 软更新系数 (TAU) | 1.0 | 硬更新目标网络 |
| 折扣因子 (GAMMA) | 0.99 | Q-learning 折扣 |
| 训练频率 (TRAIN_FREQ) | 4 | 每 4 步做 1 次梯度更新 |
| 梯度步数 | 1 | 每次 1 步梯度下降 |
| 目标网络更新间隔 | 500 steps | 每 500 步复制权重到目标网络 |
| 探索衰减比例 | 0.1 | ε 在前 10% 训练步内线性衰减 |
| 最终探索率 | 0.0 | 训练后期完全贪心 |
| 网络结构 | [128, 128] | Q 网络两层 MLP |

**lt 场景中 DQN/DDQN 表现最差：** 高违反率（DQN 68.1%，DDQN 64.4%）说明纯靠终止惩罚无法在 M=15 的紧缺环境中有效学习约束满足。

---

## 6. ILP 基线参数

| 参数 | 说明 |
|------|------|
| 求解器 | CBC（PuLP 默认整数线性规划求解器） |
| 决策变量 | x_{i,j} ∈ {0,1}，服务 i 是否部署到 ECU j |
| 目标函数 | maximize Σ_{i,j} x_{i,j} · req_i / cap_j / |active ECUs| |
| 容量约束 | Σ_i x_{i,j} · req_i ≤ cap_j，∀j |
| 冲突约束 | Σ_{i∈conflict_set} x_{i,j} ≤ 1，∀j, ∀conflict_set |
| 唯一部署约束 | Σ_j x_{i,j} ≤ 1，∀i |
| 作用 | 提供每个 scenario 的全局最优解上界 |

---

## 7. 评估参数

| 参数 | 值 | 说明 |
|------|----|------|
| EVAL_EPS | 40 | 在测试集全部 40 个 scenario 上各跑 1 个 episode |
| SMOOTH_W | 1000 | 训练曲线滑动平均窗口大小 |
| PROGRESS_LOG_EVERY_STEPS | 200,000 | 每 20 万步打印一次进度日志 |
