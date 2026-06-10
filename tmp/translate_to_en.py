#!/usr/bin/env python3
"""Translate all tmp/docs/ and tmp/code/ HTML files to English."""
import re
import os
import glob
import unicodedata

def has_cjk(text):
    return any('一' <= c <= '鿿' for c in text)

def cjk_ratio(text):
    if not text.strip():
        return 0
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    return cjk / max(len(text.strip()), 1)

# ─── Shared replacements (docs + code) ──────────────────────────────────────
SHARED = [
    ('<html lang="zh">', '<html lang="en">'),
    ('<html lang="zh-CN">', '<html lang="en">'),
    ('← 返回列表', '← Back to list'),

    # Scenario subtitle after h1
    ('ECU 数量等于服务数量，平均每个 ECU 分配一个服务。',
     'Number of ECUs equals number of services; one service per ECU on average.'),
    ('ECU 数量大于服务数量，容量较充足，约束相对宽松。',
     'ECU count exceeds service count; capacity is abundant and constraints are loose.'),
    ('ECU 数量多于服务数量，资源较为充足，放置约束相对宽松。',
     'ECU count exceeds service count; ample resources make placement constraints loose.'),
    ('服务数量多于 ECU 数量，是最具挑战性的场景，约束最严格。',
     'Service count exceeds ECU count; the most challenging scenario with the tightest constraints.'),
    ('ECU 容量更充足，资源约束相对宽松，较易求解。',
     'ECU capacity is more abundant, resource constraints are relatively loose, easier to solve.'),
    ('服务数量远大于 ECU 数量，资源最为紧张，求解难度最高。',
     'Service count far exceeds ECU count; the most resource-constrained, highest-difficulty scenario.'),

    # H2 sections (docs + shared structure)
    ('<h2>问题定义</h2>', '<h2>Problem Definition</h2>'),
    ('<h2>约束条件</h2>', '<h2>Constraints</h2>'),
    ('<h2>马尔可夫决策过程（MDP）</h2>', '<h2>Markov Decision Process (MDP)</h2>'),
    ('<h2>观测空间</h2>', '<h2>Observation Space</h2>'),
    ('<h2>奖励函数</h2>', '<h2>Reward Function</h2>'),
    ('<h2>约束处理</h2>', '<h2>Constraint Handling</h2>'),
    ('<h2>动作掩码</h2>', '<h2>Action Masking</h2>'),
    ('<h2>理论保证</h2>', '<h2>Theoretical Guarantees</h2>'),
    ('<h2>拉格朗日乘子更新</h2>', '<h2>Lagrangian Multiplier Update</h2>'),
    ('<h2>修复机制</h2>', '<h2>Repair Mechanism</h2>'),
    ('<h2>实验结果</h2>', '<h2>Experiment Results</h2>'),
    ('<h2>ILP 问题定义</h2>', '<h2>ILP Problem Definition</h2>'),
    ('<h2>Dinkelbach 变换</h2>', '<h2>Dinkelbach Transform</h2>'),
    ('<h2>求解器</h2>', '<h2>Solver</h2>'),
    ('<h2>算法：Deep Q-Network (DQN)</h2>', '<h2>Algorithm: Deep Q-Network (DQN)</h2>'),
    ('<h2>算法：Double DQN (DDQN)</h2>', '<h2>Algorithm: Double DQN (DDQN)</h2>'),
    ('<h2>算法：MaskablePPO</h2>', '<h2>Algorithm: MaskablePPO</h2>'),
    ('<h2>算法：PPO（无约束）</h2>', '<h2>Algorithm: PPO (Unconstrained)</h2>'),
    ('<h2>算法：Lagrangian PPO</h2>', '<h2>Algorithm: Lagrangian PPO</h2>'),
    ('<h2>算法：Repair PPO</h2>', '<h2>Algorithm: Repair PPO</h2>'),
    ('<h2>整数线性规划（ILP）</h2>', '<h2>Integer Linear Programming (ILP)</h2>'),

    # H3 sections
    ('<h3>容量约束</h3>', '<h3>Capacity Constraint</h3>'),
    ('<h3>冲突约束</h3>', '<h3>Conflict Constraint</h3>'),
    ('<h3>步奖励（Step Reward）</h3>', '<h3>Step Reward</h3>'),
    ('<h3>步奖励的几何意义</h3>', '<h3>Geometric Interpretation of Step Reward</h3>'),
    ('<h3>终止奖励（Terminal Bonus）</h3>', '<h3>Terminal Bonus</h3>'),
    ('<h3>Bellman 方程</h3>', '<h3>Bellman Equation</h3>'),
    ('<h3>损失函数</h3>', '<h3>Loss Function</h3>'),
    ('<h3>超参数</h3>', '<h3>Hyperparameters</h3>'),
    ('<h3>掩码策略分布</h3>', '<h3>Masked Policy Distribution</h3>'),
    ('<h3>网络架构</h3>', '<h3>Network Architecture</h3>'),
    ('<h3>PPO 目标（同 P3）</h3>', '<h3>PPO Objective (same as P3)</h3>'),
    ('<h3>GAE 优势估计</h3>', '<h3>GAE Advantage Estimation</h3>'),
    ('<h3>PPO-Clip 目标</h3>', '<h3>PPO-Clip Objective</h3>'),
    ('<h3>拉格朗日惩罚</h3>', '<h3>Lagrangian Penalty</h3>'),
    ('<h3>对偶上升（Dual Ascent）</h3>', '<h3>Dual Ascent</h3>'),
    ('<h3>修复函数 Best-fit</h3>', '<h3>Best-fit Repair Function</h3>'),
    ('<h3>修复后奖励</h3>', '<h3>Post-repair Reward</h3>'),
    ('<h3>选择与估值解耦</h3>', '<h3>Decoupled Selection and Evaluation</h3>'),
    ('<h3>双重动作掩码</h3>', '<h3>Dual Action Masking</h3>'),
    ('<h3>双网络架构</h3>', '<h3>Dual Network Architecture</h3>'),
    ('<h3>探索策略 ($\\varepsilon$-greedy)</h3>',
     '<h3>Exploration Strategy (ε-greedy)</h3>'),
    ('<h3>DDQN vs DQN</h3>', '<h3>DDQN vs DQN</h3>'),

    # Table headers (docs)
    ('<th>元素</th><th>定义</th>', '<th>Element</th><th>Definition</th>'),
    ('<th>维度</th><th>含义</th><th>公式</th>', '<th>Dimension</th><th>Description</th><th>Formula</th>'),
    ('<th>参数</th><th>值</th>', '<th>Parameter</th><th>Value</th>'),
    ('<th>网络</th><th>结构</th>', '<th>Network</th><th>Architecture</th>'),
    ('<th>指标</th><th>值</th>', '<th>Metric</th><th>Value</th>'),
    ('<th>场景</th><th>AR</th>', '<th>Scenario</th><th>AR</th>'),

    # Table cells — MDP elements
    ('<td><strong>状态</strong> $s_t$</td><td>观测向量，见下方观测空间</td>',
     '<td><strong>State</strong> $s_t$</td><td>Observation vector; see Observation Space below</td>'),
    ('<td><strong>动作</strong> $a_t$</td><td>选择 ECU 索引 $a_t \\in \\{0,\\ldots,N-1\\}$</td>',
     '<td><strong>Action</strong> $a_t$</td><td>Choose ECU index $a_t \\in \\{0,\\ldots,N-1\\}$</td>'),
    ('<td><strong>转移</strong></td><td>确定性：将服务 $t$ 分配到 ECU $a_t$，更新剩余容量</td>',
     '<td><strong>Transition</strong></td><td>Deterministic: assign service $t$ to ECU $a_t$, update remaining capacity</td>'),
    ('<td><strong>回合长度</strong></td><td>$T = M$ 步（每步放置一个服务）</td>',
     '<td><strong>Episode Length</strong></td><td>$T = M$ steps (one service placed per step)</td>'),
    ('<td><strong>折扣因子</strong></td><td>$\\gamma = 0.99$</td>',
     '<td><strong>Discount Factor</strong></td><td>$\\gamma = 0.99$</td>'),

    # Observation space table cells
    ('<td>[0]</td><td>当前服务需求（归一化）</td>', '<td>[0]</td><td>Current service demand (normalised)</td>'),
    ('<td>[1]</td><td>当前累计 AR</td>', '<td>[1]</td><td>Cumulative AR so far</td>'),
    ('<td>[2]</td><td>剩余总容量（归一化）</td>', '<td>[2]</td><td>Total remaining capacity (normalised)</td>'),
    ('<td>[3]</td><td>剩余服务总需求（归一化）</td>', '<td>[3]</td><td>Total remaining service demand (normalised)</td>'),
    ('<td>[4]</td><td>可用 ECU 比例</td>', '<td>[4]</td><td>Fraction of ECUs with sufficient remaining capacity</td>'),
    ('<td>[5]</td><td>剩余服务比例</td>', '<td>[5]</td><td>Fraction of services yet to be placed</td>'),
    ('<td>[6:6+N]</td><td>初始容量比例（每 ECU）</td>', '<td>[6:6+N]</td><td>Initial capacity ratio per ECU</td>'),
    ('<td>[6+N:6+2N]</td><td>剩余容量（每 ECU，可为负）</td>', '<td>[6+N:6+2N]</td><td>Remaining capacity per ECU (can be negative)</td>'),
    ('<td>[6+2N:6+3N]</td><td>冲突标志（每 ECU）</td>', '<td>[6+2N:6+3N]</td><td>Conflict flag per ECU</td>'),
    ('<td>[6+3N:6+4N]</td><td>合法容量标志（每 ECU）</td>', '<td>[6+3N:6+4N]</td><td>Valid-capacity flag per ECU</td>'),
    ('<td>[6+3N:6+4N]</td><td>可用服务比例（每 ECU）</td>', '<td>[6+3N:6+4N]</td><td>Fraction of allowed services per ECU</td>'),
    ('<td>[6+4N:6+5N]</td><td>合法动作标志（容量+冲突）</td>', '<td>[6+4N:6+5N]</td><td>Valid action flag (capacity + conflict)</td>'),
    ('<td>中间 M 维</td><td>合法 ECU 数（每未来服务）</td>', '<td>Middle M dims</td><td>Valid ECU count per future service</td>'),
    ('<td>后 M 维</td><td>剩余服务需求列表</td>', '<td>Last M dims</td><td>Remaining service demand list</td>'),

    # Hyperparameter table cells
    ('<td>网络结构</td><td>MLP [128, 128]</td>', '<td>Network</td><td>MLP [128, 128]</td>'),
    ('<td>学习率</td><td>1×10⁻³</td>', '<td>Learning rate</td><td>1×10⁻³</td>'),
    ('<td>$\\tau$（目标网络更新）</td><td>1.0（硬更新）</td>',
     '<td>$\\tau$ (target update)</td><td>1.0 (hard copy)</td>'),
    ('<td>策略网络 $\\pi_\\theta$</td><td>MLP [256, 256]</td>',
     '<td>Policy $\\pi_\\theta$</td><td>MLP [256, 256]</td>'),
    ('<td>价值网络 $V_\\phi$</td><td>MLP [512, 512]（EQ）/ [256, 256]（LT）</td>',
     '<td>Value $V_\\phi$</td><td>MLP [512, 512] (EQ) / [256, 256] (LT)</td>'),
    ('<td>$\\varepsilon$ 衰减期</td><td>训练总步数的 50%（LT）/ 10%（EQ）</td>',
     '<td>$\\varepsilon$ decay period</td><td>50% of total steps (LT) / 10% (EQ)</td>'),

    # Common paragraph text
    ('<p>状态向量 $s_t \\in \\mathbb{R}^{dim}$，按顺序拼接：</p>',
     '<p>State vector $s_t \\in \\mathbb{R}^{\\text{dim}}$ concatenated in order:</p>'),
    ('<p>其中：</p>', '<p>where:</p>'),

    # Scenario subtitle variants
    ('ECU 数量多于服务数量，容量约束宽松，分配更灵活。',
     'ECU count exceeds service count; capacity constraints are loose and placement is more flexible.'),
    ('ECU 数量少于服务数量，每个 ECU 须承载多个服务，容量与冲突约束更紧。',
     'ECU count is less than service count; each ECU must host multiple services; capacity and conflict constraints are tighter.'),

    # Scale variants
    ('<p><strong>规模：</strong>$N$ 个 ECU，$M$ 个服务，N > M（N/M > 1）</p>',
     '<p><strong>Scale:</strong> $N$ ECUs, $M$ services, N &gt; M (N/M &gt; 1)</p>'),
    ('<p><strong>规模：</strong>$N$ 个 ECU，$M$ 个服务，N < M（N/M < 1）</p>',
     '<p><strong>Scale:</strong> $N$ ECUs, $M$ services, N &lt; M (N/M &lt; 1)</p>'),

    # AR description
    ('AR = 平均资源利用率，目标为最大化 AR，同时满足容量约束和冲突约束。',
     'AR = Average Resource Utilisation; objective is to maximise AR while satisfying capacity and conflict constraints.'),

    # Conflict set description
    ('$\\mathcal{F} = \\{\\mathcal{F}_1, \\ldots, \\mathcal{F}_K\\}$ 为冲突集族，同一冲突集内的服务不可共驻同一 ECU。',
     '$\\mathcal{F} = \\{\\mathcal{F}_1, \\ldots, \\mathcal{F}_K\\}$ is the family of conflict sets; services within the same set cannot share an ECU.'),

    # Target network
    ('$\\theta^-$：目标网络参数（定期硬拷贝，间隔 500 步）。',
     '$\\theta^-$: target network parameters (hard-copied every 500 steps).'),

    # Step reward note
    ('放在已有 ECU 时 $ru_t/K^t = \\Delta AR_t > 0$；启用新 ECU 时保持正值，不惩罚必要扩张，\n    但激励在满足约束前提下尽量打包（packing）——因为相同 $ru_t$ 下，\n    $K^t$ 更小则奖励更大。',
     'When placed on an existing ECU, $ru_t/K^t = \\Delta AR_t > 0$. When a new ECU is activated the reward stays positive — necessary expansion is not penalised, but packing is incentivised: for the same $ru_t$, smaller $K^t$ yields a higher reward.'),
    ('放在已有 ECU 时 $ru_t/K^t = \\Delta AR_t > 0$；启用新 ECU 时保持正值，不惩罚必要扩张，但激励在满足约束前提下尽量打包（packing）——因为相同 $ru_t$ 下，$K^t$ 更小则奖励更大。',
     'When placed on an existing ECU, $ru_t/K^t = \\Delta AR_t > 0$. When a new ECU is activated the reward stays positive — necessary expansion is not penalised, but packing is incentivised: for the same $ru_t$, smaller $K^t$ yields a higher reward.'),

    # Decision variable description
    ('<p><strong>决策变量：</strong>分配矩阵 $x_{ij} \\in \\{0,1\\}$，其中 $x_{ij}=1$ 表示服务 $i$ 分配到 ECU $j$。</p>',
     '<p><strong>Decision variable:</strong> assignment matrix $x_{ij} \\in \\{0,1\\}$, where $x_{ij}=1$ means service $i$ is assigned to ECU $j$.</p>'),
    ('<p><strong>规模：</strong>$N$ 个 ECU，$M$ 个服务，N = M（N/M = 1）</p>',
     '<p><strong>Scale:</strong> $N$ ECUs, $M$ services, N = M (N/M = 1)</p>'),
    ('<p><strong>规模：</strong>$N$ 个 ECU，$M$ 个服务，N &gt; M（资源充足）</p>',
     '<p><strong>Scale:</strong> $N$ ECUs, $M$ services, N &gt; M (resource surplus)</p>'),
    ('<p><strong>规模：</strong>$N$ 个 ECU，$M$ 个服务，N &lt; M（资源紧张）</p>',
     '<p><strong>Scale:</strong> $N$ ECUs, $M$ services, N &lt; M (resource-constrained)</p>'),

    # DQN constraint handling
    ('DQN <strong>无动作掩码</strong>，约束处理完全依赖奖励信号：',
     'DQN has <strong>no action mask</strong>; constraint handling relies entirely on reward signals:'),
    ('<li>容量违规：$-2.0$ 惩罚，remaining_vms 可变负，ep 继续</li>',
     '<li>Capacity violation: $-2.0$ penalty, remaining capacity can go negative, episode continues</li>'),
    ('<li>冲突违规：$-2.0$ 惩罚，ep 继续</li>',
     '<li>Conflict violation: $-2.0$ penalty, episode continues</li>'),
    ('<li>Q 网络通过观测中的 valid_flag $\\mathbf{1}[\\text{rem}_j^t \\geq d_t]$ 间接学习约束</li>',
     '<li>The Q-network learns constraints indirectly via the valid_flag $\\mathbf{1}[\\text{rem}_j^t \\geq d_t]$ in the observation</li>'),

    # DQN note about hard termination
    ('EQ DQN 历史上使用硬终止（cap 违规即 done），导致 replay buffer 被 1 步 $-2.0$ 轨迹主导，Q 值全部收敛至 $-2.0$，训练曲线平坦。已改为软终止（episode 继续）。',
     'The EQ DQN originally used hard termination (cap violation → done), causing the replay buffer to be dominated by 1-step $-2.0$ trajectories and Q-values converging to $-2.0$ with flat training curves. Changed to soft termination (episode continues).'),

    # P4 action masking
    ('$\\text{conflict\\_ecus}(t)$：ECU $j$ 上已有与服务 $t$ 冲突的服务的集合。',
     '$\\text{conflict\\_ecus}(t)$: set of ECUs that already host a service conflicting with service $t$.'),
    ('若 $\\mathbf{m}^t = \\mathbf{0}$（无合法 ECU），回退到最大剩余容量的 ECU，施加 $-2.0$ 惩罚。',
     'If $\\mathbf{m}^t = \\mathbf{0}$ (no legal ECU), fall back to the ECU with the largest remaining capacity and apply a $-2.0$ penalty.'),
    ('非法动作的 logit 被置为 $-\\infty$，使其采样概率恰好为零。',
     'Logits for illegal actions are set to $-\\infty$, making their sampling probability exactly zero.'),
    ('熵系数 $c_2 = 0.005$（防止过早收敛），价值损失系数 $c_1 = 0.5$。',
     'Entropy coefficient $c_2 = 0.005$ (prevents premature convergence); value loss coefficient $c_1 = 0.5$.'),
    ('惩罚项：仅在强制溢出（fallback）时 $-2.0$，正常训练中掩码保证无违规。',
     'Penalty: only $-2.0$ on forced fallback; the mask guarantees zero violations during normal training.'),

    # P4 theoretical guarantees
    ('<strong>零违规保证：</strong>在正常训练中（掩码非全零），采样动作一定满足容量约束和冲突约束。',
     '<strong>Zero-violation guarantee:</strong> During normal training (mask not all-zero), sampled actions always satisfy capacity and conflict constraints.'),
    ('<strong>可行性保证：</strong>若问题本身可行，理论上 episode 总能完成全部 $M$ 步放置。',
     '<strong>Feasibility guarantee:</strong> If the problem is feasible, the episode can always complete all $M$ placement steps.'),
    ('LT 场景下，由于 $N &lt; M$，服务累积后约束更紧，仍有概率出现全零掩码（infeasible state），导致 success rate &lt; 100%。',
     'In the LT scenario ($N &lt; M$), constraints tighten as services accumulate; all-zero masks (infeasible states) can still occur, causing success rate &lt; 100%.'),

    # Violation continuation
    ('<p>违规时 $ru_t = 0$，episode <strong>继续</strong>（软约束，M 步完整运行）。</p>',
     '<p>On violation, $ru_t = 0$; episode <strong>continues</strong> (soft constraint, full $M$-step run).</p>'),

    # P5 Lagrangian specific
    ('其中 $c_t = \\mathbf{1}[\\text{conf\\_viol}_t]$，$\\kappa_{\\text{cap}} = 2.0$，$\\beta = 0.2$（基础冲突惩罚）。',
     'where $c_t = \\mathbf{1}[\\text{conf\\_viol}_t]$, $\\kappa_{\\text{cap}} = 2.0$, $\\beta = 0.2$ (base conflict penalty).'),

    # P3 terminal bonus math
    ('\\text{无条件，不惩罚违规}', '\\text{unconditional, violations not penalised}'),

    # LaTeX text fragments inside math blocks
    ('\\text{步奖励}', '\\text{step reward}'),
    ('\\text{若放在已有 ECU（}', '\\text{if placed on existing ECU (}'),
    ('\\text{若启用新 ECU（}', '\\text{if new ECU activated (}'),
    ('\\text{若无违规}', '\\text{if no violations}'),
    ('\\text{若有违规}', '\\text{if violations}'),
    ('\\text{若有强制违规}', '\\text{if forced violation}'),
    ('\\text{概率 }', '\\text{with probability }'),
    ('\\text{已被使用}', '\\text{in use}'),
]

# ─── Docs-only H1 title map ─────────────────────────────────────────────────
DOCS_H1 = [
    ('DQN — 深度 Q 网络', 'DQN — Deep Q-Network'),
    ('DDQN — 双重深度 Q 网络', 'DDQN — Double Deep Q-Network'),
    ('P3 PPO — 无约束基线', 'P3 PPO — Unconstrained Baseline'),
    ('P3 — 无约束 PPO', 'P3 — Unconstrained PPO'),
    ('P4 MaskablePPO — 动作掩码', 'P4 MaskablePPO — Action Masking'),
    ('P5 LagPPO — 拉格朗日约束', 'P5 LagPPO — Lagrangian Constraint'),
    ('P5 LagPPO — 容量掩码 + 冲突惩罚', 'P5 LagPPO — Capacity Mask + Conflict Penalty'),
    ('P6 RepairPPO — 修复启发式', 'P6 RepairPPO — Best-fit Repair Heuristic'),
    ('ILP — 整数线性规划', 'ILP — Integer Linear Programming'),
]

# ─── Additional docs replacements (inline Chinese in paragraphs) ─────────────
DOCS_EXTRA = [
    # P5 Lagrangian specific
    ('<p>其中 $c_t = \\mathbf{1}[\\text{conf\\_viol}_t]$，容量掩码软化时 $\\kappa_{\\text{cap}}$ 降低。</p>',
     '<p>where $c_t = \\mathbf{1}[\\text{conf\\_viol}_t]$; $\\kappa_{\\text{cap}}$ is reduced when the capacity mask is softened.</p>'),
    ('容量违规时 $ru_t=0$，episode 继续。',
     'On capacity violation, $ru_t=0$, episode continues.'),

    # P3 unconstrained
    ('<p>P3 <strong>无约束</strong>：容量与冲突违规仅记录，不施加惩罚。</p>',
     '<p>P3 is <strong>unconstrained</strong>: capacity and conflict violations are logged but not penalised.</p>'),
    ('<p>违规不终止 episode，不施加额外惩罚，AR 可超过 1.0。</p>',
     '<p>Violations do not terminate the episode and carry no extra penalty; AR can exceed 1.0.</p>'),

    # ILP
    ('<p><strong>变量：</strong>$x_{ij} \\in \\{0,1\\}$，服务 $i$ 是否分配到 ECU $j$；$y_j \\in \\{0,1\\}$，ECU $j$ 是否被激活。</p>',
     '<p><strong>Variables:</strong> $x_{ij} \\in \\{0,1\\}$ — service $i$ assigned to ECU $j$; $y_j \\in \\{0,1\\}$ — ECU $j$ activated.</p>'),
    ('<p><strong>目标：</strong>最大化 AR（平均资源利用率）</p>',
     '<p><strong>Objective:</strong> Maximise AR (average resource utilisation)</p>'),
    ('<p><strong>约束：</strong></p>', '<p><strong>Constraints:</strong></p>'),

    # Network architecture cells
    ('<td>价值网络 $V_\\phi$</td><td>MLP [256, 256]</td>',
     '<td>Value $V_\\phi$</td><td>MLP [256, 256]</td>'),
    ('<td>价值网络 $V_\\phi$</td><td>MLP [512, 512]</td>',
     '<td>Value $V_\\phi$</td><td>MLP [512, 512]</td>'),
]


# ─── Docs DDQN-specific ────────────────────────────────────────────────────
DOCS_DDQN = [
    ('<h2>算法：Double DQN</h2>', '<h2>Algorithm: Double DQN</h2>'),
    ('<h3>DQN 的过估计问题</h3>', '<h3>DQN Overestimation Problem</h3>'),
    ('<p>标准 DQN 用同一网络选动作和评估价值，导致系统性高估：</p>',
     '<p>Standard DQN uses the same network to both select and evaluate actions, leading to systematic overestimation:</p>'),
    ('<h3>Double DQN 修正</h3>', '<h3>Double DQN Fix</h3>'),
    ('<p>用在线网络 $\\theta$ 选动作，目标网络 $\\theta^-$ 评估价值：</p>',
     '<p>Use the online network $\\theta$ to select actions and the target network $\\theta^-$ to evaluate values:</p>'),
    ('\\text{在线网络选动作}', '\\text{online network selects action}'),
    ('<h3>DQN vs DDQN 对比</h3>', '<h3>DQN vs DDQN Comparison</h3>'),
    ('<tr><th>特性</th><th>DQN</th><th>DDQN</th></tr>',
     '<tr><th>Property</th><th>DQN</th><th>DDQN</th></tr>'),
    ('<tr><td>动作选择</td><td>目标网络</td><td>在线网络</td></tr>',
     '<tr><td>Action selection</td><td>Target network</td><td>Online network</td></tr>'),
    ('<tr><td>价值评估</td><td>目标网络</td><td>目标网络</td></tr>',
     '<tr><td>Value evaluation</td><td>Target network</td><td>Target network</td></tr>'),
    ('<tr><td>偏差</td><td>系统性高估</td><td>更低偏差</td></tr>',
     '<tr><td>Bias</td><td>Systematic overestimation</td><td>Lower bias</td></tr>'),
    ('<tr><td>其余超参数</td><td colspan="2" style="text-align:center">相同</td></tr>',
     '<tr><td>Other hyperparameters</td><td colspan="2" style="text-align:center">identical</td></tr>'),
    ('<p>与 DQN 相同：无掩码，通过奖励惩罚隐式学习约束。DDQN 更稳定的 Q 估计使约束学习更高效。</p>',
     '<p>Same as DQN: no mask; constraints learned implicitly through reward penalties. DDQN\'s more stable Q-estimates make constraint learning more efficient.</p>'),
]

# ─── Docs P3-specific ──────────────────────────────────────────────────────
DOCS_P3 = [
    ('<p>惩罚项：无（<code>penalties = 0</code>），所有违规仅记录不扣分。</p>',
     '<p>Penalties: none (<code>penalties = 0</code>); all violations are recorded but not penalised.</p>'),
    ('<h2>算法：Proximal Policy Optimization (PPO)</h2>',
     '<h2>Algorithm: Proximal Policy Optimization (PPO)</h2>'),
    ('<h3>目标函数</h3>', '<h3>Objective Function</h3>'),
    ('<p>其中概率比 $\\rho_t(\\theta) = \\dfrac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)}$，截断参数 $\\varepsilon = 0.2$。</p>',
     '<p>where the probability ratio $\\rho_t(\\theta) = \\dfrac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)}$ and clipping parameter $\\varepsilon = 0.2$.</p>'),
    ('<h3>广义优势估计（GAE）</h3>', '<h3>Generalised Advantage Estimation (GAE)</h3>'),
    ('<tr><td>$n_\\text{steps}$（每次 rollout 步数）</td><td>512</td></tr>',
     '<tr><td>$n_\\text{steps}$ (steps per rollout)</td><td>512</td></tr>'),
    ('<tr><td>$n_\\text{epochs}$（每次 PPO 更新轮数）</td><td>10</td></tr>',
     '<tr><td>$n_\\text{epochs}$ (PPO update epochs per rollout)</td><td>10</td></tr>'),
    ('<tr><td>$\\gamma$（折扣）</td><td>0.99</td></tr>',
     '<tr><td>$\\gamma$ (discount)</td><td>0.99</td></tr>'),
    ('<tr><td>学习率</td><td>3×10⁻⁴</td></tr>',
     '<tr><td>Learning rate</td><td>3×10⁻⁴</td></tr>'),
    ('<p>P3 为<strong>无约束基线</strong>：既无动作掩码，也无违规惩罚。</p>',
     '<p>P3 is the <strong>unconstrained baseline</strong>: no action mask and no violation penalty.</p>'),
    ('<p>违规后果：episode 继续，仅在 info 中记录计数。</p>',
     '<p>Violation consequence: episode continues; only counts are recorded in info.</p>'),
    ('<p class="note">预期行为：agent 学会将所有服务堆叠在同一个 ECU 上（AR > 1），\n  以最大化 $\\sum ru_t / K^t$（分母最小化）。这正是 P3 的设计目标——\n  作为对照，展示<em>缺乏约束时 RL 的失败模式</em>。</p>',
     '<p class="note">Expected behaviour: the agent learns to stack all services onto a single ECU (AR &gt; 1) to maximise $\\sum ru_t / K^t$ (minimise denominator). This is P3\'s design intent — a control showing the <em>failure mode of unconstrained RL</em>.</p>'),
]

# ─── Docs P5-specific ──────────────────────────────────────────────────────
DOCS_P5 = [
    ('<h2>拉格朗日约束松弛</h2>', '<h2>Lagrangian Constraint Relaxation</h2>'),
    ('<h3>原始问题</h3>', '<h3>Primal Problem</h3>'),
    ('<p>其中 $\\bar{c}(\\theta) = \\mathbb{E}[c_t]$ 为期望冲突违规率。</p>',
     '<p>where $\\bar{c}(\\theta) = \\mathbb{E}[c_t]$ is the expected conflict violation rate.</p>'),
    ('<h3>拉格朗日松弛</h3>', '<h3>Lagrangian Relaxation</h3>'),
    ('<p>其中 $\\hat{c}_k$ 为第 $k$ 个 rollout 内的冲突违规率估计，$\\alpha$ 为对偶学习率。$\\lambda$ 随冲突率上升而增大，自适应施加更重的约束压力。</p>',
     '<p>where $\\hat{c}_k$ is the estimated conflict violation rate over rollout $k$ and $\\alpha$ is the dual learning rate. $\\lambda$ grows with the conflict rate, adaptively increasing constraint pressure.</p>'),
    ('<h3>容量约束处理（EQ 特有）</h3>', '<h3>Capacity Constraint Handling (EQ-specific)</h3>'),
    ('<h3>容量约束处理（GT 特有）</h3>', '<h3>Capacity Constraint Handling (GT-specific)</h3>'),
    ('<h3>容量约束处理（LT 特有）</h3>', '<h3>Capacity Constraint Handling (LT-specific)</h3>'),
    ('<p>EQ/GT: 容量违规给固定惩罚 $-2.0$，episode 继续。</p>',
     '<p>EQ/GT: capacity violation applies a fixed $-2.0$ penalty; episode continues.</p>'),
    ('<p>LT: 动作掩码保证容量合法（capacity mask），仅在极端情况下回退并施加 $-2.0$ 惩罚。</p>',
     '<p>LT: action mask guarantees capacity legality; only in extreme cases does a fallback occur with a $-2.0$ penalty.</p>'),
    ('<h2>算法流程</h2>', '<h2>Algorithm Flow</h2>'),
    ('<div class="step"> 初始化策略 $\\theta$，乘子 $\\lambda = 0$</div>',
     '<div class="step"> Initialise policy $\\theta$, multiplier $\\lambda = 0$</div>'),
    ('<div class="step"> 用策略 $\\pi_\\theta$ 采集 rollout，计算 $\\hat{c}_k$</div>',
     '<div class="step"> Collect rollout with policy $\\pi_\\theta$, compute $\\hat{c}_k$</div>'),
    ('<div class="step"> PPO 更新 $\\theta$（固定 $\\lambda$）</div>',
     '<div class="step"> PPO update $\\theta$ (fixed $\\lambda$)</div>'),
    ('<div class="step"> 对偶更新：$\\lambda \\leftarrow \\max(0, \\lambda + \\alpha \\hat{c}_k)$</div>',
     '<div class="step"> Dual update: $\\lambda \\leftarrow \\max(0, \\lambda + \\alpha \\hat{c}_k)$</div>'),
    ('<div class="step"> 重复 2–4 直至收敛</div>',
     '<div class="step"> Repeat 2–4 until convergence</div>'),
]

# ─── Docs P6-specific ──────────────────────────────────────────────────────
DOCS_P6 = [
    ('<p>修复越多，终止奖励越低；若修复率 = 100%，终止奖励为 0。</p>',
     '<p>More repairs mean a lower terminal bonus; at 100% repair rate the terminal bonus is 0.</p>'),
    ('<h3>修复策略</h3>', '<h3>Repair Strategy</h3>'),
    ('<p>当 agent 选择的 ECU 违反冲突约束时，系统自动<strong>修复</strong>（repair）到合法 ECU：</p>',
     '<p>When the agent selects an ECU that violates the conflict constraint, the system automatically <strong>repairs</strong> to a legal ECU:</p>'),
    ('\\text{其中} \\quad \\mathcal{V}_t', '\\text{where} \\quad \\mathcal{V}_t'),
    ('<p>修复动作选择剩余容量最大的合法 ECU，同时施加 $-0.1$ 惩罚。</p>',
     '<p>The repair action selects the legal ECU with the largest remaining capacity, applying a $-0.1$ penalty.</p>'),
    ('<h3>设计意图</h3>', '<h3>Design Intent</h3>'),
    ('<p>P6 允许 agent "犯错后纠正"，相比 P4（严格掩码）探索性更强，\n  但终止奖励对修复次数线性惩罚，引导 agent 学习一次性做出合法决策。</p>',
     '<p>P6 lets the agent "correct mistakes", giving more exploration freedom than P4 (strict masking), but the terminal bonus penalises repair count linearly, guiding the agent to make legal decisions first time.</p>'),
    ('<h2>修复率与 AR 的关系</h2>', '<h2>Repair Rate vs AR</h2>'),
    ('<tr><th>修复率 $r$</th><th>终止奖励</th><th>含义</th></tr>',
     '<tr><th>Repair rate $r$</th><th>Terminal bonus</th><th>Meaning</th></tr>'),
    ('<tr><td>0（无修复）</td><td>$AR_T$</td><td>完美放置</td></tr>',
     '<tr><td>0 (no repair)</td><td>$AR_T$</td><td>Perfect placement</td></tr>'),
    ('<tr><td>0.5</td><td>$0.5 \\cdot AR_T$</td><td>一半需修复</td></tr>',
     '<tr><td>0.5</td><td>$0.5 \\cdot AR_T$</td><td>Half need repair</td></tr>'),
    ('<tr><td>1（全修复）</td><td>0</td><td>每步都冲突</td></tr>',
     '<tr><td>1 (all repair)</td><td>0</td><td>Conflict every step</td></tr>'),
]

# ─── Code-file code comment translations ─────────────────────────────────
CODE_COMMENTS = [
    # Common across files
    ('<span class="cm"># 读取与 ILP 相同的 YAML 配置文件</span>',
     '<span class="cm"># Load the same YAML config file as ILP</span>'),
    ('<span class="cm"># 有多少个 ECU</span>', '<span class="cm"># number of ECUs</span>'),
    ('<span class="cm"># 有多少个服务</span>', '<span class="cm"># number of services</span>'),
    ('<span class="cm"># 可选：训练场景列表</span>', '<span class="cm"># optional: training scenario list</span>'),
    ('<span class="cm"># N 个 ECU 对象</span>', '<span class="cm"># N ECU objects</span>'),
    ('<span class="cm"># M 个 SVC 对象</span>', '<span class="cm"># M service objects</span>'),
    ('<span class="cm"># 动作空间：从 0 到 N-1 选一个 ECU</span>',
     '<span class="cm"># action space: choose one ECU from 0 to N-1</span>'),
    ('<span class="cm"># 观测空间：形状 = 5N + 6 + M 的浮点向量，范围 [-1, 1]</span>',
     '<span class="cm"># observation space: float vector of shape 5N+6+M, range [-1, 1]</span>'),
    ('<span class="cm"># 注意：仅检查容量（capacity），不检查冲突（conflict）！</span>',
     '<span class="cm"># note: checks capacity only, not conflict!</span>'),
    ('<span class="cm"># 防止除以零</span>', '<span class="cm"># avoid division by zero</span>'),
    ('<span class="cm"># 当前待放服务的需求（归一化）</span>',
     '<span class="cm"># current service demand (normalised)</span>'),
    ('<span class="cm"># 冲突标志：对每个 ECU，1=放当前服务到这里会违反冲突约束</span>',
     '<span class="cm"># conflict flag: 1 = placing current service here violates a conflict constraint</span>'),
    ('<span class="cm"># 容量合法标志：1=该 ECU 有足够剩余容量</span>',
     '<span class="cm"># capacity-valid flag: 1 = ECU has sufficient remaining capacity</span>'),
    ('<span class="cm"># [0]       当前服务需求</span>', '<span class="cm"># [0]       current service demand</span>'),
    ('<span class="cm"># [1]       当前 AR</span>', '<span class="cm"># [1]       current AR</span>'),
    ('<span class="cm"># [2]       剩余可用总容量</span>', '<span class="cm"># [2]       total usable remaining capacity</span>'),
    ('<span class="cm"># [3]       剩余服务总需求</span>', '<span class="cm"># [3]       total remaining service demand</span>'),
    ('<span class="cm"># [4]       有足够容量的 ECU 比例</span>', '<span class="cm"># [4]       fraction of ECUs with enough capacity</span>'),
    ('<span class="cm"># [5]       剩余服务数量比例</span>', '<span class="cm"># [5]       fraction of services not yet placed</span>'),
    ('<span class="cm"># [6:6+N]   各 ECU 初始容量比例</span>', '<span class="cm"># [6:6+N]   per-ECU initial capacity fraction</span>'),
    ('<span class="cm"># [6+N:6+2N]  剩余容量（可能为负！）</span>', '<span class="cm"># [6+N:6+2N]  remaining capacity (may be negative!)</span>'),
    ('<span class="cm"># [6+2N:6+3N] 冲突标志</span>', '<span class="cm"># [6+2N:6+3N] conflict flag</span>'),
    ('<span class="cm"># [6+3N:6+4N] ECU 可接受服务比例</span>', '<span class="cm"># [6+3N:6+4N] fraction of conflict-free services per ECU</span>'),
    ('<span class="cm"># [6+4N:6+5N] 容量合法标志</span>', '<span class="cm"># [6+4N:6+5N] capacity-valid flag</span>'),
    ('<span class="cm"># [6+5N:6+5N+M] 剩余服务需求列表</span>', '<span class="cm"># [6+5N:6+5N+M] remaining service demand list</span>'),
    ('<span class="cm"># 当前要放的服务</span>', '<span class="cm"># current service to place</span>'),
    ('<span class="cm"># ① 检测违规</span>', '<span class="cm"># ① detect violations</span>'),
    ('<span class="cm"># ② 计算惩罚（可叠加！同时违规 = -4.0）</span>',
     '<span class="cm"># ② compute penalty (stackable! both violations = -4.0)</span>'),
    ('<span class="cm"># ③ 利用率奖励：违规时 ru=0（不给正奖励）</span>',
     '<span class="cm"># ③ utilisation reward: ru=0 on violation (no positive reward)</span>'),
    ('<span class="cm"># ④ 无论违规与否，都执行放置（remaining_vms 可能变负！）</span>',
     '<span class="cm"># ④ execute placement regardless of violation (remaining_vms may go negative!)</span>'),
    ('<span class="cm"># ⑤ 终局奖励（最后一步才触发）</span>',
     '<span class="cm"># ⑤ terminal bonus (triggered only on the last step)</span>'),
    ('<span class="cm"># 零违规 → 额外奖励 +AR（鼓励整集无违规）</span>',
     '<span class="cm"># zero violations → extra reward +AR (encourages clean episodes)</span>'),
    ('<span class="cm"># 有违规 → 额外惩罚 -AR（惩罚整集失败）</span>',
     '<span class="cm"># violations present → extra penalty -AR (punishes failed episodes)</span>'),
    ('<span class="cm"># 随机生成 K 个冲突集，每个集包含 2~M 个服务</span>',
     '<span class="cm"># randomly generate K conflict sets, each containing 2–M services</span>'),
    ('<span class="cm"># 同一冲突集内的服务不能放在同一 ECU 上</span>',
     '<span class="cm"># services in the same conflict set cannot share an ECU</span>'),
    ('<span class="cm"># 返回 True 表示：把 svc_idx 放到 ecu_idx 会违反冲突约束</span>',
     '<span class="cm"># returns True if placing svc_idx on ecu_idx violates a conflict constraint</span>'),
    ('<span class="cm"># 放置 svc_idx 后，从 ecu_allowed 中移除与它冲突的其他服务</span>',
     '<span class="cm"># after placing svc_idx, remove conflicting services from ecu_allowed</span>'),
    ('<span class="cm"># 使用全连接神经网络（MLP）作为 Q 网络</span>',
     '<span class="cm"># use MLP as Q-network</span>'),
    ('<span class="cm"># 100,000 条经验</span>', '<span class="cm"># 100,000 experiences</span>'),
    ('<span class="cm"># 先随机探索 2000 步</span>', '<span class="cm"># random exploration for first 2000 steps</span>'),
    ('<span class="cm"># 每次抽 64 条</span>', '<span class="cm"># sample 64 per update</span>'),
    ('<span class="cm"># 1.0 = 硬复制目标网络</span>', '<span class="cm"># 1.0 = hard copy of target network</span>'),
    ('<span class="cm"># 0.99 折扣</span>', '<span class="cm"># 0.99 discount</span>'),
    ('<span class="cm"># 每 4 步更新一次</span>', '<span class="cm"># update every 4 steps</span>'),
    ('<span class="cm"># 每 500 步同步目标网络</span>', '<span class="cm"># sync target network every 500 steps</span>'),
    ('<span class="cm"># 前 50% 探索</span>', '<span class="cm"># explore for first 50% of training</span>'),
    ('<span class="cm"># 最终 ε=0</span>', '<span class="cm"># final ε=0</span>'),
    ('<span class="cm"># DQN 的 Q 值更新公式（Bellman equation）：</span>',
     '<span class="cm"># DQN Q-value update formula (Bellman equation):</span>'),
    ('<span class="cm"># 注意：目标网络的 max_a 同时做了"选择"和"评估"两件事</span>',
     '<span class="cm"># note: target network\'s max_a does both "selection" and "evaluation"</span>'),
    ('<span class="cm"># 这会导致 Q 值高估（Overestimation Bias）→ DDQN 解决了这个问题</span>',
     '<span class="cm"># this causes Q-value overestimation bias → DDQN fixes this</span>'),
    ('<span class="cm"># 随机抽取一个训练场景</span>', '<span class="cm"># randomly sample a training scenario</span>'),
    ('<span class="cm"># 服务按需求从大到小排序（先放大服务，类似 First Fit Decreasing）</span>',
     '<span class="cm"># sort services by demand descending (like First Fit Decreasing)</span>'),
    ('<span class="cm"># 重置剩余容量</span>', '<span class="cm"># reset remaining capacity</span>'),
    ('<span class="cm"># 清空放置记录</span>', '<span class="cm"># clear placement records</span>'),
    ('<span class="cm"># 所有服务初始都允许</span>', '<span class="cm"># all services initially allowed</span>'),
    ('<span class="cm"># 每步决策：</span>', '<span class="cm"># per-step decision:</span>'),
    ('<span class="cm"># 随机探索</span>', '<span class="cm"># random exploration</span>'),
    ('<span class="cm"># 贪心利用</span>', '<span class="cm"># greedy exploitation</span>'),
    # DDQN specific
    ('<span class="cm"># ① 违规检测（与 DQNEnv 完全相同）</span>',
     '<span class="cm"># ① violation detection (identical to DQNEnv)</span>'),
    ('<span class="cm"># ② 计算惩罚（与 DQNEnv 完全相同）</span>',
     '<span class="cm"># ② compute penalty (identical to DQNEnv)</span>'),
    ('<span class="cm"># ③ 强制放置，remaining_vms 可能变负（与 DQNEnv 完全相同）</span>',
     '<span class="cm"># ③ force placement, remaining_vms may go negative (identical to DQNEnv)</span>'),
    ('<span class="cm"># ④ 终局奖励（与 DQNEnv 完全相同）</span>',
     '<span class="cm"># ④ terminal bonus (identical to DQNEnv)</span>'),
    ('<span class="cm"># DDQN 只需在策略参数中开启 double_q_network=True</span>',
     '<span class="cm"># DDQN only needs double_q_network=True in policy kwargs</span>'),
    ('<span class="cm"># SB3 的 DQN 内置支持 DDQN！</span>',
     '<span class="cm"># SB3\'s DQN has built-in DDQN support!</span>'),
    ('<span class="cm"># stable_baselines3 v2.x 默认启用 Double DQN</span>',
     '<span class="cm"># stable_baselines3 v2.x enables Double DQN by default</span>'),
    ('<span class="cm"># ↑ 与 DQNEnv.step() 返回值结构完全相同</span>',
     '<span class="cm"># ↑ return structure identical to DQNEnv.step()</span>'),
    ('<span class="cm"># 两个独立网络互相制衡 → 减少高估！</span>',
     '<span class="cm"># two independent networks check each other → reduces overestimation!</span>'),
    ('<span class="cm"># ... 其他超参数与 DQN 相同 ...</span>',
     '<span class="cm"># ... other hyperparameters identical to DQN ...</span>'),
    ('<span class="cm"># 分离"选择"和"评估"两个职责</span>',
     '<span class="cm"># separate "selection" and "evaluation" responsibilities</span>'),
    ('<span class="cm"># 同一个网络做两件事 → 高估偏差！</span>',
     '<span class="cm"># one network does both → overestimation bias!</span>'),
    ('<span class="cm"># 在线选，目标评</span>', '<span class="cm"># online selects, target evaluates</span>'),
    ('<span class="cm"># 目标网络同时负责：</span>', '<span class="cm"># target network handles both:</span>'),
    ('<span class="cm"># 目标网络同时选+评</span>', '<span class="cm"># target network selects + evaluates together</span>'),
    ('<span class="cm"># 相同</span>', '<span class="cm"># identical</span>'),
    ('<span class="cm"># 或通过子类 / 修改 train() 方法中的 target 计算逻辑实现</span>',
     '<span class="cm"># or via subclass / modifying target computation in train()</span>'),
    # P4 specific
    ('<span class="cm"># 1. 加载场景</span>', '<span class="cm"># 1. load scenarios</span>'),
    ('<span class="cm"># 2. 用 ILP（整数线性规划）求最优解 —— 作为性能上界</span>',
     '<span class="cm"># 2. solve ILP for optimal solution — used as performance upper bound</span>'),
    ('<span class="cm"># 3. 训练 MaskablePPO</span>', '<span class="cm"># 3. train MaskablePPO</span>'),
    ('<span class="cm"># 4. 评估 MaskablePPO（确定性策略）</span>',
     '<span class="cm"># 4. evaluate MaskablePPO (deterministic policy)</span>'),
    ('<span class="cm"># 5. 保存结果 JSON + CSV + 绘图</span>',
     '<span class="cm"># 5. save results JSON + CSV + plots</span>'),
    ('<span class="cm"># ILP 对所有测试场景求最优，给出 AR 上界</span>',
     '<span class="cm"># ILP solves all test scenarios optimally, giving the AR upper bound</span>'),
    ('<span class="cm"># 动作空间: Discrete(N)</span>', '<span class="cm"># action space: Discrete(N)</span>'),
    ('<span class="cm"># 动作空间: Discrete(N)  (相同)</span>', '<span class="cm"># action space: Discrete(N)  (same)</span>'),
    ('<span class="cm"># 观测空间: 5N+6+M</span>', '<span class="cm"># observation space: 5N+6+M</span>'),
    ('<span class="cm"># 观测空间: 5N+6+M  (相同)</span>', '<span class="cm"># observation space: 5N+6+M  (same)</span>'),
    # P6 specific
    ('<span class="cm"># else: PATH A — 智能体选择正确 ──</span>',
     '<span class="cm"># else: PATH A — agent chose correctly ──</span>'),
    ('<span class="cm"># ── PATH B: 使用修复后的 ECU ──</span>',
     '<span class="cm"># ── PATH B: use repaired ECU ──</span>'),
    ('<span class="cm"># ── PATH C: 修复失败 ──</span>',
     '<span class="cm"># ── PATH C: repair failed ──</span>'),
    ('<span class="cm"># 回合结束时触发 / triggered when done == True</span>',
     '<span class="cm"># triggered when done == True</span>'),
    ('<span class="cm"># 扣减容量</span>', '<span class="cm"># deduct capacity</span>'),
    ('<span class="cm"># 更新冲突集合</span>', '<span class="cm"># update conflict sets</span>'),
    ('<span class="cm"># ── 检测违规 / Detect violations ──</span>',
     '<span class="cm"># ── Detect violations ──</span>'),
    ('<span class="cm"># ── 计算奖励 / Compute reward ──</span>',
     '<span class="cm"># ── Compute reward ──</span>'),
    ('<span class="cm"># 记录部署</span>', '<span class="cm"># record deployment</span>'),
    ('<span class="cm"># 重置所有计数器 / reset all counters</span>',
     '<span class="cm"># reset all counters</span>'),
    # Inline arrow comments (not inside cm tags, but bare in code context)
    ('← 唯一不同！', '← only difference!'),
    ('← 在线网络选动作', '← online network selects action'),
    ('← 目标网络评估价值', '← target network evaluates value'),
]

# ─── Code-file specific replacements ────────────────────────────────────────
CODE_SPECIFIC = [
    # Nav bar scenario descriptions
    ('ECU 数等于服务数，每 ECU 平均分配一个服务',
     'N = M, one service per ECU on average'),
    ('ECU 数大于服务数，容量更充足，约束更宽松',
     'N > M, ECU count exceeds services, constraints are looser'),
    ('ECU 数多于服务数，容量充足，约束宽松',
     'N > M, abundant capacity, loose constraints'),
    ('服务数多于 ECU 数，资源最紧张',
     'N < M, services exceed ECUs, tightest constraints'),
    ('服务数大于 ECU 数，资源最紧张，难度最高',
     'N < M, services exceed ECUs, highest difficulty'),

    # H1 for code walkthroughs
    ('DQN 代码讲解', 'DQN Code Walkthrough'),
    ('DDQN 代码讲解', 'DDQN Code Walkthrough'),
    ('P3 代码讲解', 'P3 Code Walkthrough'),
    ('P4 代码讲解', 'P4 Code Walkthrough'),
    ('P4 MaskablePPO 代码讲解', 'P4 MaskablePPO Code Walkthrough'),
    ('P5 代码讲解', 'P5 Code Walkthrough'),
    ('P5 LagPPO 代码讲解', 'P5 LagPPO Code Walkthrough'),
    ('P6 代码讲解', 'P6 Code Walkthrough'),
    ('P6 RepairPPO 代码讲解', 'P6 RepairPPO Code Walkthrough'),
    ('ILP 代码讲解', 'ILP Code Walkthrough'),

    # Subtitle line in code walkthrough headers (Chinese + English Bilingual → remove)
    ('High School Level · Chinese + English Bilingual', ''),

    # Scenario label prefixes in header paragraph
    ('ECU EQ 场景 · ', 'EQ — '),
    ('ECU GT 场景 · ', 'GT — '),
    ('ECU LT 场景 · ', 'LT — '),
    ('无动作屏蔽 · 违规惩罚学习', 'no action mask · penalty-based constraint learning'),
    ('动作掩码（硬约束） · 零违规', 'action masking (hard constraint) · zero violations'),
    ('容量硬掩码 + 冲突软惩罚 λ', 'capacity hard mask + conflict soft penalty λ'),
    ('Best-fit 修复启发式', 'Best-fit repair heuristic'),
    ('Off-policy 算法 · 无动作屏蔽', 'off-policy · no action mask'),
    ('双网络 · 消除过高估计', 'dual network · eliminates overestimation'),

    # Table header in code files (bilingual)
    ('<th>参数</th><th>值</th><th>含义（中）</th><th>Meaning (EN)</th>',
     '<th>Parameter</th><th>Value</th><th>Meaning</th>'),
    ('<th>参数</th><th>值</th><th>含义</th>',
     '<th>Parameter</th><th>Value</th><th>Meaning</th>'),
    ('<th>参数</th><th>含义（中）</th><th>Meaning (EN)</th>',
     '<th>Parameter</th><th>Meaning</th>'),

    # Common paragraph starters in code files
    ('<p>在这个项目里：</p>', '<p>In this project:</p>'),
    ('<p>在 EQ 场景里：</p>', '<p>In the EQ scenario:</p>'),
    ('<p>在 GT 场景里：</p>', '<p>In the GT scenario:</p>'),
    ('<p>在 LT 场景里：</p>', '<p>In the LT scenario:</p>'),

    # DQN reward table header
    ('<tr><th>情况</th><th>终局奖励</th><th>含义</th></tr>',
     '<tr><th>Case</th><th>Terminal Bonus</th><th>Meaning</th></tr>'),
    ('<td>奖励：整集成功，获得等于当前 AR 的额外分</td>',
     '<td>reward: clean episode bonus equal to current AR</td>'),
    ('<td>惩罚：整集失败，扣除等于当前 AR 的分数</td>',
     '<td>penalty: failed episode deduction equal to current AR</td>'),

    # DQN valid/violated placement li items
    ('<li>ru = req / cap_initial（正数）</li>',
     '<li>ru = req / cap_initial (positive)</li>'),
    ('<li>remaining_vms 正常减少</li>',
     '<li>remaining_vms decremented normally</li>'),
    ('<li>remaining_vms <span style="color:var(--red)">变负！</span></li>',
     '<li>remaining_vms <span style="color:var(--red)">goes negative!</span></li>'),

    # DQN scenario description
    ('<p><strong>LT 场景</strong>（Large-scale Tight）：15 个服务，10 个 ECU，约束很紧。</p>',
     '<p><strong>LT scenario</strong> (Large-scale Tight): 15 services, 10 ECUs, tight constraints.</p>'),
    ('<tr><th>算法</th><th>）</th><th>ViolRate</th></tr>',
     '<tr><th>Algorithm</th><th>AR</th><th>ViolRate</th></tr>'),

    # P3 analogy content
    ('P3 = 记录违规（知道发生了）但不扣分（不学习避免它）。',
     'P3 = log violations (knows they happened) but no penalty (does not learn to avoid them).'),

    # ILP code nav items
    ('<li><a href="#s1">objects.py — Data Classes (数据类)</a></li>',
     '<li><a href="#s1">objects.py — Data Classes</a></li>'),
    ('<li><a href="#s2">What problem are we solving? (问题背景)</a></li>',
     '<li><a href="#s2">What problem are we solving?</a></li>'),
    ('<li><a href="#s4">Dinkelbach Outer Loop (丁克尔巴赫迭代外循环)</a></li>',
     '<li><a href="#s4">Dinkelbach Outer Loop</a></li>'),
    ('<li><a href="#s5">Decision Variables (决策变量 x, y)</a></li>',
     '<li><a href="#s5">Decision Variables (x, y)</a></li>'),
    ('<li><a href="#s7">Five Constraints (五个约束条件)</a></li>',
     '<li><a href="#s7">Five Constraints</a></li>'),
    ('<li><a href="#s8">λ Update and Convergence (λ更新与收敛)</a></li>',
     '<li><a href="#s8">λ Update and Convergence</a></li>'),
    # ILP h2 with Chinese subtitle
    ('<h2 id="s1">1. objects.py — Data Classes (数据类)</h2>',
     '<h2 id="s1">1. objects.py — Data Classes</h2>'),
    ('<h2 id="s2">2. What Problem Are We Solving? (问题背景)</h2>',
     '<h2 id="s2">2. What Problem Are We Solving?</h2>'),
    # ILP table headers
    ('<th>约束 Constraint</th>', '<th>Constraint</th>'),
    ('<th>说明 Meaning</th>', '<th>Meaning</th>'),
    # ILP table cells
    ('<td><span class="tag tag-blue">分配</span> Assignment</td>',
     '<td><span class="tag tag-blue">Assignment</span></td>'),
    ('每个服务必须被分配到 <em>恰好一个</em> ECU — Each service goes to exactly one ECU',
     'Each service goes to exactly one ECU'),
    # ILP ECU analogy
    ('<div class="analogy"><strong>类比 / Analogy:</strong>',
     '<div class="analogy"><strong>Analogy:</strong>'),
    # ILP li items
    ('<li><code>name</code>：ECU 的标识，如 <code>"ECU0"</code></li>',
     '<li><code>name</code>: ECU identifier, e.g. <code>"ECU0"</code></li>'),
    ('<li><code>capacity</code>：这个 ECU 能运行的虚拟机总数（总槽位数）</li>',
     '<li><code>capacity</code>: total VMs this ECU can run (total slot count)</li>'),
    ('<li><code>name</code>：服务名称，如 <code>"SVC0"</code></li>',
     '<li><code>name</code>: service name, e.g. <code>"SVC0"</code></li>'),
    ('<li><code>requirement</code>：该服务需要占用的 VM 数（床位数）</li>',
     '<li><code>requirement</code>: number of VMs this service needs</li>'),
    # = 每个活跃 ECU 的平均利用率
    ('   = 每个活跃 ECU 的平均利用率', '   = average utilisation per active ECU'),

    # Card labels (bilingual "Chinese English" → English only)
    # These are handled by regex below
]


def strip_chinese_from_label(text):
    """For 'Chinese English' label text, keep only the English portion."""
    # Find last continuous CJK block and strip everything before/including it
    # Strategy: keep ASCII-dominant trailing part
    parts = re.split(r'[一-鿿]+', text)
    # Last part is likely English
    candidate = parts[-1].strip()
    if candidate and not has_cjk(candidate):
        return candidate
    return text


def clean_bilingual_heading(heading_text):
    """For '中文 English' or '中文？ English?' headings, return English part only."""
    # Try to split at '？ ' (Chinese question mark + space)
    if '？ ' in heading_text:
        parts = heading_text.split('？ ', 1)
        en_part = parts[1].strip()
        if en_part and not has_cjk(en_part):
            return en_part
    # Try to find the last segment that has no CJK
    # Split on CJK blocks
    segments = re.split(r'[一-鿿]+', heading_text)
    # Find first segment from end that has meaningful ASCII content
    for seg in reversed(segments):
        seg = re.sub(r'^[^a-zA-Z0-9$\\*(]+', '', seg.strip()).strip()
        if seg and len(seg) > 2 and not has_cjk(seg):
            return seg
    return heading_text


def strip_cjk_parentheticals(text):
    """Remove Chinese parentheticals like (什么是...) or （中文） from text."""
    text = re.sub(r'\s*（[^）]*[一-鿿][^）]*）', '', text)
    text = re.sub(r'\s*\([^)]*[一-鿿][^)]*\)', '', text)
    return text.strip()


def process_labels(html):
    """Replace <div class="label">Chinese English</div> labels."""
    def repl_label(m):
        content = m.group(1)
        if has_cjk(content):
            cleaned = strip_chinese_from_label(content)
            return f'<div class="label">{cleaned}</div>'
        return m.group(0)
    return re.sub(r'<div class="label">([^<]+)</div>', repl_label, html)


def process_headings(html):
    """Clean bilingual h2/h3/h4 headings in code files."""
    def repl_h(m):
        open_tag = m.group(1)   # full opening tag e.g. <h2 id="s1">
        tag_name = m.group(2)   # just h2/h3/h4
        prefix = m.group(3) or ''
        text = m.group(4)
        close_tag = m.group(5)
        if not has_cjk(text):
            return m.group(0)
        # Strip Chinese parentheticals first
        cleaned = strip_cjk_parentheticals(text)
        if not has_cjk(cleaned):
            return f'{open_tag}{prefix}{cleaned.strip()}</{close_tag}>'
        # Fall back to bilingual heading extraction
        cleaned = clean_bilingual_heading(text)
        return f'{open_tag}{prefix}{cleaned}</{close_tag}>'

    # h2/h3/h4 with optional attributes and section-num span
    html = re.sub(
        r'(<(h[234])[^>]*>)(<span[^>]*>\d+</span>\s*)?([^<]+)</(h[234])>',
        lambda m: repl_h(m) if m.group(2) == m.group(5) else m.group(0),
        html
    )
    return html


def remove_chinese_only_paragraphs(html):
    """
    Remove <p>...</p> elements that contain mostly Chinese and have no style/class
    indicating they are English content. English equivalents have style with 'muted' or 'color'.
    """
    def repl_p(m):
        attrs = m.group(1) or ''
        content = m.group(2)
        # Skip elements with muted style (English)
        if 'muted' in attrs or 'color:var(--muted)' in attrs.lower():
            return m.group(0)
        # If pure/mostly Chinese paragraph and no attributes, remove
        plain_content = re.sub(r'<[^>]+>', '', content)
        if cjk_ratio(plain_content) > 0.35:
            return ''
        return m.group(0)

    # Match <p> or <p attrs>...</p>
    html = re.sub(
        r'<p([^>]*)>((?:(?!<p[^>]*>|</p>).)*?)</p>',
        repl_p,
        html,
        flags=re.DOTALL
    )
    return html


def remove_chinese_list_items(html):
    """
    For <li> elements: if content is mixed Chinese+English with the Chinese part
    dominant, try to strip the Chinese prefix and keep the English.
    This is conservative — only remove if CJK ratio > 80%.
    """
    def repl_li(m):
        content = m.group(1)
        plain = re.sub(r'<[^>]+>', '', content)
        if cjk_ratio(plain) > 0.8:
            return ''
        return m.group(0)
    html = re.sub(r'<li>((?:(?!<li>|</li>).)*?)</li>', repl_li, html, flags=re.DOTALL)
    return html


def remove_lang_labels(html):
    """Remove Chinese-only lang-label divs and lang-badge spans."""
    html = re.sub(r'<div class="lang-label">[^<]*</div>\s*', '', html)
    html = re.sub(r'<span class="lang-badge">[^<]*</span>\s*', '', html)
    return html


def fix_four_col_table_header(html):
    """Remove the Chinese 3rd-column header after prior replacements have run."""
    html = html.replace(
        '<th>Parameter</th><th>Value</th><th>含义（中）</th><th>Meaning (EN)</th>',
        '<th>Parameter</th><th>Value</th><th>Meaning</th>'
    )
    return html


def remove_chinese_td_column(html):
    """In 4-column table rows where the 3rd td is mostly Chinese, remove it."""
    def repl_row(m):
        row = m.group(0)
        cells = re.findall(r'<td(?:[^>]*)>.*?</td>', row, re.DOTALL)
        if len(cells) == 4:
            plain3 = re.sub(r'<[^>]+>', '', cells[2])
            if has_cjk(plain3) and cjk_ratio(plain3) > 0.3:
                return row.replace(cells[2], '', 1)
        return row
    return re.sub(r'<tr>.*?</tr>', repl_row, html, flags=re.DOTALL)


def clean_bilingual_th_td(html):
    """For th/td with 'Chinese English' mixed content, keep English portion."""
    def repl(m):
        tag, attrs, content = m.group(1), m.group(2) or '', m.group(3)
        if not has_cjk(content):
            return m.group(0)
        plain = re.sub(r'<[^>]+>', '', content)
        cleaned = strip_chinese_from_label(plain)
        if cleaned and cleaned != plain and not has_cjk(cleaned):
            new_content = content.replace(plain, cleaned)
            return f'<{tag}{attrs}>{new_content}</{tag}>'
        return m.group(0)
    return re.sub(r'<(th|td)((?:\s[^>]*)?)>(.*?)</\1>', repl, html, flags=re.DOTALL)


def clean_obs_headers(html):
    """Translate observation grid header divs."""
    html = html.replace(
        '<div class="obs-header">索引 Index</div>', '<div class="obs-header">Index</div>')
    html = html.replace(
        '<div class="obs-header">字段名 Field</div>', '<div class="obs-header">Field</div>')
    html = html.replace(
        '<div class="obs-header">含义 Meaning</div>', '<div class="obs-header">Meaning</div>')
    return html


def clean_obs_rows(html):
    """For observation grid value divs with bilingual content, keep English suffix."""
    def repl_div(m):
        content = m.group(1)
        if not has_cjk(content):
            return m.group(0)
        plain = re.sub(r'<[^>]+>', '', content)
        cleaned = clean_bilingual_heading(plain)
        if cleaned and cleaned != plain and not has_cjk(cleaned):
            return f'<div>{cleaned}</div>'
        return m.group(0)
    return re.sub(r'<div>((?:(?!</div>).)+)</div>', repl_div, html, flags=re.DOTALL)


def clean_bilingual_badge(html):
    """For badge/span elements with 'Chinese · English' content."""
    def repl(m):
        tag_class, content = m.group(1), m.group(2)
        if not has_cjk(content):
            return m.group(0)
        if ' · ' in content:
            parts = content.split(' · ')
            en_parts = [p.strip() for p in parts if p.strip() and not has_cjk(p)]
            if en_parts:
                return f'<{tag_class}>{" · ".join(en_parts)}</{tag_class.split()[0]}>'
        return m.group(0)
    html = re.sub(r'(<span class="badge">)(.*?)</span>', repl, html, flags=re.DOTALL)
    return html


def clean_bilingual_li(html):
    """For list items with bilingual content, extract English label."""
    def repl_li(m):
        content = m.group(1)
        if not has_cjk(content):
            return m.group(0)

        working = content

        # Step 1: Clean Chinese parentheticals from anchor text
        def repl_anchor(ma):
            inner = ma.group(1)
            href = ma.group(0).split('>')[0] + '>'
            text = ma.group(1)
            cleaned = strip_cjk_parentheticals(text)
            if not has_cjk(cleaned) and cleaned != text:
                return href + cleaned + '</a>'
            return ma.group(0)
        working = re.sub(r'<a[^>]*>(.*?)</a>', repl_anchor, working, flags=re.DOTALL)

        # Step 2: Clean bilingual strong tags
        def repl_strong(ms):
            inner = ms.group(1)
            if not has_cjk(inner):
                return ms.group(0)
            cleaned = strip_chinese_from_label(inner)
            if cleaned and not has_cjk(cleaned):
                return f'<strong>{cleaned}</strong>'
            return ms.group(0)
        working = re.sub(r'<strong>(.*?)</strong>', repl_strong, working, flags=re.DOTALL)

        # Step 3: Strip ：Chinese description suffix (after tags)
        working = re.sub(r'：\s*[^<]*[一-鿿][^<]*$', '', working).rstrip()

        # Step 4: Strip — Chinese description suffix
        working = re.sub(r'\s*[—–-]\s*[^<]*[一-鿿][^<]*$', '', working).rstrip()

        # Only accept if result is clean
        if working != content:
            plain = re.sub(r'<[^>]+>', '', working).strip()
            if plain and not has_cjk(plain):
                return f'<li>{working}</li>'

        return m.group(0)
    return re.sub(r'<li>(.*?)</li>', repl_li, html, flags=re.DOTALL)


def apply_replacements(html, replacements):
    for old, new in replacements:
        html = html.replace(old, new)
    return html


def translate_file(path, is_code=False):
    with open(path, encoding='utf-8') as f:
        html = f.read()

    # Apply shared replacements
    html = apply_replacements(html, SHARED)

    if is_code:
        # Code comments first (before structural processing)
        html = apply_replacements(html, CODE_COMMENTS)
        # Code-specific replacements
        html = apply_replacements(html, CODE_SPECIFIC)
        # Fix 4-column table header (after Parameter/Value already translated)
        html = fix_four_col_table_header(html)
        # Remove Chinese explanation td column from 4-col rows
        html = remove_chinese_td_column(html)
        # Remove lang-label / lang-badge elements
        html = remove_lang_labels(html)
        # Clean bilingual badge/span
        html = clean_bilingual_badge(html)
        # Clean observation grid headers and rows
        html = clean_obs_headers(html)
        html = clean_obs_rows(html)
        # Clean bilingual th/td
        html = clean_bilingual_th_td(html)
        # Clean bilingual li items
        html = clean_bilingual_li(html)
        # Process card labels
        html = process_labels(html)
        # Clean bilingual h2/h3 headings
        html = process_headings(html)
        # Remove Chinese-only paragraphs
        html = remove_chinese_only_paragraphs(html)
        # Remove Chinese-only list items
        html = remove_chinese_list_items(html)
    else:
        # Docs H1 titles
        html = apply_replacements(html, DOCS_H1)
        # Docs extra
        html = apply_replacements(html, DOCS_EXTRA)
        # Docs algorithm-specific
        html = apply_replacements(html, DOCS_DDQN)
        html = apply_replacements(html, DOCS_P3)
        html = apply_replacements(html, DOCS_P5)
        html = apply_replacements(html, DOCS_P6)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)

    return path


def main():
    base = '/home/soar009/github/my_code/tmp'
    processed = []

    # Docs files
    for path in sorted(glob.glob(f'{base}/docs/**/*.html', recursive=True)):
        translate_file(path, is_code=False)
        processed.append(path)
        print(f'  docs: {os.path.relpath(path, base)}')

    # Code files
    for path in sorted(glob.glob(f'{base}/code/**/*.html', recursive=True)):
        translate_file(path, is_code=True)
        processed.append(path)
        print(f'  code: {os.path.relpath(path, base)}')

    print(f'\nProcessed {len(processed)} files.')


if __name__ == '__main__':
    main()
