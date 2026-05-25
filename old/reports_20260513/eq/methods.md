# 实验方法说明 — ecu_eq_svc_p（ECU = SVC，N=10, M=10）

## 1. 问题定义

**任务**：将 M=10 个服务（SVC）依次分配到 N=10 个边缘计算单元（ECU），最大化平均资源利用率（AR），同时满足容量约束和冲突约束。

**目标函数（AR）**：

```
AR = (1 / |active ECUs|) × Σ_{i,j: x_{i,j}=1} req_i / cap_j
```

- active ECUs：至少承载一个服务的 ECU
- req_i：服务 i 的 VM slot 需求
- cap_j：ECU j 的 VM slot 容量
- AR 越高表示资源利用越密集

**约束**：
1. **容量约束**：∀j, Σ_{i: placed on j} req_i ≤ cap_j
2. **冲突约束**：∀ conflict_set C, ∀j, |{i ∈ C : placed on j}| ≤ 1（冲突集中的服务不能共存于同一 ECU）

**eq 场景特点**：N=M=10，供需均衡。每个 ECU 平均要承载 1 个服务。ILP 最优 AR ≈ 0.54，约束较易满足，但 RL 策略仍需学习精确的配对关系（哪个服务应放到哪个 ECU）。

---

## 2. 建模方式：顺序决策（MDP）

将放置问题建模为一个有限步 MDP：
- **状态** s_t：当前待放置服务的特征 + 所有 ECU 的剩余容量 + 冲突标记 + 已放置历史
- **动作** a_t：选择目标 ECU j ∈ {0, …, N-1}
- **奖励** r_t：放置当前服务后的 AR 增量（稠密奖励信号）
- **终止条件**：放置完所有 M 个服务（episode 固定长度 M=10 步）

每个 episode 对应一个 scenario 的一次完整放置过程。

---

## 3. 六种方法详解

### 3.1 ILP 最优基线（problem2_ilp）

**性质**：整数线性规划，全局最优，非 RL。

**流程**：
1. 对每个测试 scenario，用 PuLP + CBC 求解器直接求解 ILP
2. 决策变量 x_{i,j} ∈ {0,1}
3. 得到全局最优放置方案和对应 AR

**作用**：提供每个 scenario 的理论上界，用于衡量 RL 方法的次优性 gap。

**eq 场景中**：ILP 最优 AR = 0.5418 ± 0.0049，零违反。

---

### 3.2 P3：无约束 PPO（problem3_ppo）

**约束处理**：无，仅对容量做软屏蔽（剩余容量可为负）。

**环境特点**：
- Episode 始终运行 M=10 步，不因违反而提前终止
- 剩余容量可以变为负数（允许超载）
- 奖励：每步给予 AR 增量，不含任何违反惩罚

**状态空间**（约 56 维）：
```
[当前服务需求(归一化), 累计AR, 剩余总容量(归一化), 剩余总需求,
 有足够容量的ECU比例, 剩余服务比例,
 各ECU初始容量(×10), 各ECU剩余容量(×10),
 各ECU冲突标记(×10), 有效动作标记(容量检查)(×10),
 剩余服务需求降序排列(×10)]
```

**动作空间**：Discrete(10)，选择 ECU

**作用**：消融基线，展示完全无约束时 RL 能达到的 AR 上限及违反代价。

**eq 场景中**：AR = 0.865（远超 ILP 0.542），因为允许超载；143 次冲突违反（无法部署）。

---

### 3.3 P4：动作掩码 PPO（problem4_ppo_mask，MaskablePPO）

**约束处理**：硬约束，通过动作掩码完全屏蔽非法动作。

**掩码规则**：当且仅当同时满足以下条件，ECU j 的动作才为有效（mask=True）：
1. ECU j 剩余容量 ≥ 当前服务需求
2. ECU j 未被当前服务所在的任何冲突集的其他成员占用

**实现**：使用 `sb3_contrib.MaskablePPO`，在采样动作时仅从有效动作集合中采样，无效动作的概率被置为 0（softmax 后）。

**结果保证**：所有放置决策均可行，理论上违反率 = 0。

**等价于约束满足搜索**：策略实际上在学习"在所有可行放置中选择最优的那个"。

**eq 场景中**：AR = 0.535（接近 ILP 0.542），零违反。

---

### 3.4 P5：Lagrangian 约束松弛 PPO（problem5_ppo_lagrangian）

**约束处理**：软约束，通过自适应拉格朗日乘子 λ 将约束违反转化为奖励惩罚。

**奖励函数**：
```
r_t = (req_i / cap_j) - λ · c_t
```
- req_i / cap_j：本步放置的 AR 增量
- c_t = 1 若本步产生违反（容量或冲突），否则 0
- λ：当前拉格朗日乘子（动态调整）

**λ 更新机制**（每 20 个 episode）：
```
λ ← clip(λ + 0.005 × (avg_viol_rate - 0.0), 0.0, 5.0)
```

**观测扩展**：将 λ/LAMBDA_MAX 加入观测向量，使策略感知当前约束压力强度，减少非平稳性。

**特点**：不保证零违反，但通过对偶上升迭代逼近零违反目标；在 AR 和约束满足之间实现软权衡。

**eq 场景中**：AR = 0.537，0.6 次冲突违反，近似零违反。

---

### 3.5 P6：PPO + 局部修复启发式（problem6_ppo_opt）

**约束处理**：后处理修复，当 RL 策略选择一个已被占用的 ECU 时，通过 Best-Fit 启发式重新定位。

**修复逻辑**：
1. RL 策略选择目标 ECU j
2. 若 j 已被冲突集中的其他服务占用，触发修复：
   - 枚举所有备选 ECU，按剩余容量降序（Best-Fit）选择可行的替代 ECU
   - 若无可行替代，跳过该服务（placed_mean < M）
3. 容量违反不触发修复，可能仍会产生容量超载

**特点**：混合 RL + 组合优化，保证冲突约束满足，但可能牺牲 AR（因为最优 ECU 可能已不可用）。

**eq 场景中**：AR = 0.537，2.8 次容量违反，0 次冲突违反。P6 在容量违反上高于 P4，因为修复逻辑不处理容量超载。

---

### 3.6 DQN / DDQN（problem_dqn / problem_ddqn）

**约束处理**：终止惩罚，违反约束时 episode 提前终止并给予负奖励。

**奖励设计**：
- 合法放置：r = +req_i / cap_j（AR 增量）
- 违反（容量或冲突）：r = -1，episode 终止

**算法差异**：
- **DQN**：用目标网络的 max Q 计算 TD 目标，存在高估偏差
- **DDQN**：用在线网络选动作、目标网络估值，降低高估

**状态中包含有效动作标记**：每个 ECU 的容量是否足够的 binary flag，帮助策略识别合法动作（但不强制掩码）。

**eq 场景中**：DQN AR=0.528，DDQN AR=0.532，均有少量违反（DQN 5.5% 违反率）。

---

## 4. 实验执行流程

### 4.1 单次单 seed 执行

```
TRAIN_SEED=0 python ecu_eq_svc_p/{problem}/run_all.py
```

每个 `run_all.py` 执行：
1. 加载 YAML 配置 → 解析 200 个 scenario
2. 按 SEED 划分 160 训练 / 40 测试 scenario
3. 对测试 scenario 逐一运行 ILP 求解，记录最优 AR 基线
4. 训练 RL 算法（PPO/DQN）：
   - N_ENVS=40 个并行环境，循环采集-更新直到 TOTAL_STEPS
5. 对 40 个测试 scenario 各评估 1 个 episode，记录 AR、violations 等指标
6. 生成输出：
   - `results/summary.csv`：汇总行（方法名、AR均值、放置数、违反率等）
   - `results/results.json`：详细指标
   - `results/training_curve.png`：训练曲线（AR + 违反率 + 服务放置数）
   - `results/comparison.png`：ILP vs 方法的 AR/违反对比

### 4.2 多 seed 并行执行

```bash
bash scripts/run_multi_seed.sh --seeds "0 1 2 3 4" --total-timesteps 5000000
```

流程：
1. 对每个 seed：
   a. 清理上轮结果（`scripts/clean_all_results.sh`）
   b. 设置 `TRAIN_SEED={seed}`
   c. 并行启动 3 个环境组（eq/gt/lt），每组内顺序运行 6 个 problem
   d. 等待全部完成，归档 `summary.csv` 到 `logs/multi_seed_results/seed_{seed}/{group}/`
2. 5 个 seed 全部完成后，运行 `scripts/aggregate_results.py` 计算 mean±std
3. 运行 `scripts/plot_aggregate_combined.py` 生成 3×4 综合可视化图

### 4.3 结果汇总

`aggregate_results.py` 读取所有 `seed_*/group/problem.csv`，对每个（group, problem, method）组合计算跨 seed 的均值和标准差，输出 `aggregate_summary.csv` 和 `aggregate_report.txt`。

---

## 5. 观测空间设计要点

各 problem 的观测维度有所不同，但核心信息一致：

| 信息类型 | 维度 | 说明 |
|----------|------|------|
| 当前服务特征 | 1 | 当前待放置服务的 requirement（归一化） |
| 全局进度 | 3-4 | 累计 AR、剩余容量总量、已放置服务数等 |
| ECU 状态 | N × 3 | 每个 ECU 的初始容量、剩余容量、冲突标记 |
| 有效动作 | N | 每个 ECU 是否可放置当前服务（容量检查） |
| 剩余服务队列 | M | 剩余服务需求降序排列 |
| λ（P5 专属） | 1 | 归一化拉格朗日乘子（λ / LAMBDA_MAX） |

P5 的观测多了 λ 维度（总维度 ≈ 57），Critic 使用更宽的 [512,512] 网络以适应更复杂的价值估计。
