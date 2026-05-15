# ECU 服务部署优化：强化学习 vs 整数线性规划

---

## 1. 实验背景和目的

### 1.1 研究背景

车载 ECU（Electronic Control Unit，电子控制单元）服务部署问题：将 $M$ 个软件服务分配至 $N$ 台 ECU，在满足**容量约束**与**冲突约束**的前提下，最大化**平均资源利用率（AR）**。

**目标函数：**

$$AR = \frac{1}{|\text{active ECUs}|} \sum_{\substack{i,j \\ x_{i,j}=1}} \frac{\text{req}_i}{\text{cap}_j}$$

- $\text{req}_i$：服务 $i$ 所需 VM slot 数
- $\text{cap}_j$：ECU $j$ 的 VM slot 容量
- active ECUs：至少承载一个服务的 ECU 集合

**约束：**

1. **容量约束**：$\forall j,\ \sum_{i: \text{placed on } j} \text{req}_i \leq \text{cap}_j$
2. **冲突约束**：$\forall C \in \text{ConflictSets},\ \forall j,\ |\{i \in C : \text{placed on } j\}| \leq 1$

**研究目的：** 对比六种方法在三种资源场景下的 AR 性能与约束满足能力，评估 RL 方法能否以接近 ILP 最优解的水平完成安全可行的服务部署。

### 1.2 实验场景

| 场景 | ECU 数 N | 服务数 M | 资源关系 |
|------|----------|----------|----------|
| LT（资源紧缺） | 10 | 15 | ECU < SVC，每个 ECU 平均承载 1.5 个服务 |
| EQ（供需均衡） | 10 | 10 | ECU = SVC，每个 ECU 平均承载 1 个服务 |
| GT（资源充裕） | 15 | 10 | ECU > SVC，服务有更多 ECU 可选 |

### 1.3 参与对比的六种方法

| 方法 | 约束处理策略 | 算法基础 |
|------|-------------|----------|
| ILP（基线） | 硬约束，全局最优 | PuLP + CBC 整数线性规划 |
| P3 — 无约束 PPO | 无约束（AR 上界参考） | PPO |
| P4 — 动作掩码 PPO | 硬约束，非法动作掩码为 0 | MaskablePPO |
| P5 — Lagrangian PPO | 软约束，自适应拉格朗日乘子 | PPO + 对偶上升 |
| P6 — 修复启发式 PPO | 冲突违反后 Best-Fit 修复 | PPO + 局部搜索 |
| DQN / DDQN | 违反时终止并惩罚 | DQN / Double DQN |

> P3 仅保存在 `logs/multi_seed_results/aggregate_summary.csv` 中用于 CSV 对比，不出现在图表中。

### 1.4 机器设备配置

| 硬件 | 规格 |
|------|------|
| GPU | 3 × NVIDIA GeForce GTX 1080 Ti（各 11 GiB 显存） |
| CPU | 56 核 |
| 内存 | 125 GiB |
| 操作系统 | Linux（Ubuntu） |
| Python 环境 | `.venv/bin/python`（虚拟环境） |
| 主要框架 | stable-baselines3、sb3-contrib、PuLP、PyTorch |

---

## 2. 实验具体参数配置

### 2.1 环境参数配置

#### 场景生成参数

| 参数 | LT | EQ | GT |
|------|----|----|----|
| ECU 数量 N | 10 | 10 | 15 |
| 服务数量 M | 15 | 10 | 10 |
| ECU 容量范围 | \[50, 200\]，步长 5 | 同左 | 同左 |
| SVC 需求范围 | \[10, 100\]，步长 5 | 同左 | 同左 |
| 冲突集数量（每 scenario） | 10 | 10 | 10 |
| 冲突集大小范围 | \[2, 15\] | \[2, 10\] | \[2, 10\] |
| 场景总数 | 200 | 200 | 200 |
| 训练集 / 测试集 | 160 / 40（80/20） | 同左 | 同左 |

#### MDP 建模

- **状态** $s_t$：当前待放置服务需求（归一化）+ 累计 AR + 各 ECU 初始容量/剩余容量/冲突标记 + 有效动作标记 + 剩余服务队列
- **动作** $a_t$：选择目标 ECU $j \in \{0, \ldots, N-1\}$
- **奖励** $r_t$：本步 AR 增量 $= \text{req}_i / \text{cap}_j$（P5 额外减去约束惩罚项）
- **Episode 长度**：固定 M 步（每步放置一个服务）
- **测试评估**：对 40 个测试 scenario 各跑 1 个 episode

### 2.2 模型参数配置

#### PPO 基础参数（P3 / P4 / P6 共用）

| 参数 | 值 | 说明 |
|------|----|------|
| 总训练步数 | 5,000,000 | 与环境的交互总步数 |
| 并行环境数 | 40 | DummyVecEnv 并发收集轨迹 |
| 学习率 | 3×10⁻⁴ | Adam 优化器 |
| Rollout 长度（N_STEPS） | 256 | 每次更新前各环境收集步数；共 256×40=10,240 步/更新 |
| Mini-Batch 大小 | 128 | 从 rollout buffer 随机采样 |
| 更新轮数（N_EPOCHS） | 10 | 每批 rollout 数据复用 10 次梯度更新 |
| 折扣因子 γ | 0.999 | 高折扣保证末步奖励能反向传播到起始步 |
| GAE λ | 0.95 | 广义优势估计偏差-方差权衡 |
| Clip 范围 ε | 0.2 | PPO 截断比率 |
| 网络结构 | \[256, 256\] | Actor/Critic 共享两层 MLP |

#### PPO 算法原理

每轮迭代：

1. 用当前策略 $\pi_\theta$ 在 $N_\text{envs}$ 个并行环境中采集 $N_\text{steps}$ 步轨迹
2. 计算广义优势估计（GAE）：$\hat{A}_t = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}$，其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
3. 对同一批数据重复 $N_\text{epochs}$ 次梯度更新：
   - **Actor loss**：$L^{\text{CLIP}} = \mathbb{E}\left[\min\!\left(r_t \hat{A}_t,\ \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$，$r_t = \pi_\theta / \pi_{\theta_\text{old}}$
   - **Critic loss**：$\mathbb{E}\left[(V(s_t) - V_\text{target})^2\right]$
4. 更新 $\pi_{\theta_\text{old}} \leftarrow \pi_\theta$，开始下一轮采集

#### Lagrangian PPO 专用参数（P5）

在 PPO 基础上，奖励函数修改为：

$$r_t = \frac{\text{req}_i}{\text{cap}_j} - \lambda \cdot c_t$$

其中 $c_t = 1$ 若本步产生违反（容量或冲突），否则 $c_t = 0$。

**对偶上升更新**（每 `LAMBDA_UPDATE_WINDOW` 个 episode）：

$$\lambda \leftarrow \text{clip}\!\left(\lambda + \alpha_\lambda \cdot (\bar{v} - v^*),\ 0,\ \lambda_{\max}\right)$$

| 参数 | 值 | 说明 |
|------|----|------|
| N_STEPS | 128 | 更短 rollout，适配 M=10 步的短 episode |
| BATCH_SIZE | 256 | 更大 batch，减小梯度方差 |
| γ | 0.99 | 略低于 P3/P4/P6 |
| Critic 网络 | \[512, 512\] | 更宽网络适配含 λ 的扩展观测 |
| LAMBDA_INIT | 0.1 | λ 初始值 |
| LAMBDA_LR（$\alpha_\lambda$） | 0.005 | 对偶上升步长 |
| LAMBDA_TARGET（$v^*$） | 0.0 | 目标违反率：零违反 |
| LAMBDA_MAX（$\lambda_{\max}$） | 5.0 | λ 上限，防止惩罚项崩溃 |
| LAMBDA_UPDATE_WINDOW | 20 episodes | λ 更新周期 |
| 观测扩展 | +1 维 | 将 $\lambda / \lambda_{\max}$ 加入观测，缓解非平稳性 |

#### DQN / DDQN 参数

| 参数 | 值 | 说明 |
|------|----|------|
| 总训练步数 | 3,000,000 | off-policy 效率更高，步数少于 PPO |
| 并行环境数 | 12 | DQN 不需要大批量 on-policy 收集 |
| 学习率 | 1×10⁻³ | 比 PPO 高一个量级 |
| Replay Buffer 大小 | 100,000 | 存储历史转移 $(s,a,r,s')$ |
| Mini-Batch 大小 | 64 | 从 replay buffer 随机采样 |
| 软更新系数 TAU | 1.0 | 硬更新目标网络 |
| 折扣因子 γ | 0.99 | Q-learning 折扣 |
| 目标网络更新间隔 | 500 steps | 定期同步在线网络权重 |
| ε 探索衰减比例 | 0.1 | 前 10% 步从 1.0 线性衰减至 0.0 |
| 网络结构 | \[128, 128\] | Q 网络两层 MLP |
| 违反处理 | 终止并给 −1 奖励 | episode 在第一次违反时终止 |

**DQN vs DDQN TD 目标计算：**

$$y_{\text{DQN}} = r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a')$$

$$y_{\text{DDQN}} = r + \gamma \cdot Q_{\text{target}}\!\left(s',\ \arg\max_{a'} Q_{\text{online}}(s', a')\right)$$

DDQN 用在线网络选动作、目标网络估值，缓解高估偏差。

#### ILP 基线公式

决策变量 $x_{i,j} \in \{0,1\}$（服务 $i$ 是否部署到 ECU $j$）：

$$\text{maximize} \quad \frac{1}{|\text{active ECUs}|} \sum_{i,j} x_{i,j} \cdot \frac{\text{req}_i}{\text{cap}_j}$$

$$\text{s.t.} \quad \sum_i x_{i,j} \cdot \text{req}_i \leq \text{cap}_j \quad \forall j$$

$$\sum_{i \in C} x_{i,j} \leq 1 \quad \forall j,\ \forall C \in \text{ConflictSets}$$

$$\sum_j x_{i,j} \leq 1 \quad \forall i$$

#### 实验种子配置

| 参数 | 值 |
|------|-----|
| 种子列表 | 0, 1, 2, 3, 4 |
| 种子数量 | 5 |
| 控制范围 | train/test 划分 + RL 初始化 |
| 传入方式 | 环境变量 `TRAIN_SEED` |
| 统计方式 | 5 个种子的均值 ± 标准差 |

---

## 3. 实验脚本运行指南

### 3.1 运行脚本指令

#### 完整多 seed 实验（推荐）

自动后台运行全部 3 组 × 6 方法 × 5 个 seed，完成后自动汇总：

```bash
bash scripts/run_multi_seed.sh --seeds "0 1 2 3 4" --total-timesteps 5000000
```

快速冒烟测试（仅验证流程）：

```bash
bash scripts/run_multi_seed.sh --seeds "0 1" --total-timesteps 10000
```

脚本行为：
- 自动进入后台，终端立即返回控制权
- 每轮 seed 完成后归档至 `logs/multi_seed_results/seed_{N}/{group}/{problem}.csv`
- 全部完成后自动运行 `scripts/aggregate_results.py` 生成汇总报告

#### 单次并行实验（单 seed）

并行启动三个场景组（eq/gt/lt），各自在后台顺序运行 6 个 problem：

```bash
export TRAIN_SEED=0
bash scripts/run_all_parallel.sh --total-timesteps 5000000
```

#### 汇总跨 seed 结果

```bash
.venv/bin/python scripts/aggregate_results.py \
    --archive-dir logs/multi_seed_results \
    --seeds 0 1 2 3 4
```

输出：
- `logs/multi_seed_results/aggregate_summary.csv` — 完整汇总表（含均值 ± 标准差）
- `logs/multi_seed_results/aggregate_report.txt` — 可读报告

#### 生成对比图表

```bash
.venv/bin/python scripts/plot_metrics.py
```

输出至 `figures/` 目录：

| 图表 | 路径 | 内容 |
|------|------|------|
| Fig 1 | `figures/fig1_ar_comparison/` | 各方法 AR 对比（3 场景并排） |
| Fig 2 | `figures/fig2_violations/` | 约束违反总数（容量 + 冲突堆叠） |
| Fig 3 | `figures/fig3_tradeoff/` | AR vs 违反率安全-性能散点图 |

### 3.2 查看日志指令

| 目的 | 命令 |
|------|------|
| 多 seed 进度 | `tail -f logs/multi_seed_results/run_multi_seed.log` |
| 单次并行进度 | `tail -f logs/run_all_parallel.log` |
| 三组实时日志（同时） | `bash scripts/log_all_parallel.sh` |
| 单组详细日志（eq 示例） | `tail -f ecu_eq_svc_p/run_scripts/run_background.log` |

各组日志路径：

```
ecu_eq_svc_p/run_scripts/run_background.log
ecu_gt_svc_p/run_scripts/run_background.log
ecu_lt_svc_p/run_scripts/run_background.log
```

### 3.3 停止指定任务指令

#### 停止全部并行实验

```bash
bash scripts/stop_all_parallel.sh
```

依次终止顶层调度进程和三个场景组后台进程，清理所有 PID 文件。

#### 手动停止单个进程

```bash
# 停止多 seed 调度进程
kill $(cat pids/run_multi_seed.pid)

# 停止并行调度进程
kill $(cat pids/run_all_parallel.pid)

# 停止单个场景组（以 eq 为例）
kill $(cat ecu_eq_svc_p/run_scripts/run_background.pid)
```

#### 清理所有实验结果（谨慎使用）

```bash
bash scripts/clean_all_results.sh
```

> `logs/multi_seed_results/` 目录不会被清除，已归档的 seed 结果保留。
