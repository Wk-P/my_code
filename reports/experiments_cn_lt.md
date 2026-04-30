# ecu_lt_svc_p 程序设计文档

## 与 eq/gt 的实验理念对齐（及 lt 专属修改）

### 对齐原则

lt 的实验设计以 eq/gt 的理念为基准，再做 lt 特有的适配：

| 方面 | eq/gt 做法 | lt 原始实现 | lt 修正后 |
|------|-----------|-----------|---------|
| **P3 ru 计算** | 始终加 `ru`（违规不清零） | 违规时 `ru=0` | ✅ 始终加 `ru` |
| **P3 terminal** | 始终 `+AR` | 有违规时 `-AR` | ✅ 始终 `+AR` |
| **P3 惩罚** | 无任何惩罚 | forced-overflow `-2.0` + 冲突软惩罚 | ✅ 无惩罚 |
| **P5 ru 计算** | 始终加 `ru`，冲突只有 λ 惩罚 | 冲突时 `ru=0` | ✅ 始终加 `ru` |
| **DQN/DDQN terminal** | 违规终止（隐式 signal） | 无 terminal bonus | ✅ `±AR` terminal bonus |
| **DQN/DDQN 终止** | 违规立即 done | 从不提前终止 | ⚙️ **lt 保留无终止**（bin packing 场景下频繁终止会导致稀疏信号） |
| **P4 约束** | 容量 mask + 冲突终止 | 容量+冲突双 mask | ⚙️ **lt 保留双 mask**（多服务共 ECU 中冲突空间更大，终止过于频繁） |
| **P6 机制** | PPO + best-fit repair | 双 mask + 丰富 terminal | ⚙️ **lt 保留 mask 方案**（lt 修复启发式复杂度更高，mask 已足够） |

### 为什么 lt 不完全照搬 eq/gt

1. **P4 不用冲突终止**：eq P4 容量 mask + 冲突终止。lt 中 M=15 服务塞进 N=10 ECU，每台 ECU 平均 1.5 个服务，冲突机会显著多于 eq（N=M=10 一机一服务）。如果冲突也触发终止，训练前期几乎每 episode 都提前结束，梯度极其稀疏。双 mask 保证 M 步完整跑完，信号更稳。

2. **DQN 不用违规终止**：同理，lt bin packing 下违规概率在 random policy 阶段高于 eq，终止会大幅缩短 episode，经验回放样本质量极差。保留软惩罚 + 增加 terminal bonus，让 DQN 获得与 PPO 同等量级的 episode 级奖励信号。

3. **P6 不用 repair 启发式**：eq P6 的 best-fit repair 在一机一服务时相对简单（找一台空的满足容量的 ECU）。lt 中 repair 需要考虑当前 ECU 上已有哪些服务、与新服务的冲突关系，复杂度大幅提升。当前 dual-mask + 连续 terminal bonus 已能区分 P6 vs P4。

## 核心场景定义

| 条件 | 说明 |
|------|------|
| `N < M` | ECU 数**少于**服务数，每台 ECU 必须承载多个服务 |
| `sum(cap) > sum(req)` | 总容量大于总需求，理论上全部可放置 |
| 冲突约束 | K 个子集，同子集服务不能共 ECU |

相比 `ecu_gt_svc_p`（N > M 可一机一服务），lt 变体最大的难点是 **bin packing + 冲突** 的组合：需要把多个服务塞进少数 ECU，同时避免冲突对。

---

## 状态空间：`5N + 6 + M`（P3/P4/P6）

比 `ecu_gt_svc_p` 多一维 `ecu_allowed_frac[j]`：

| 维度 | 变量名 | 含义 | 备注 |
|------|--------|------|------|
| `[0]` | `service_demand_norm` | 当前服务需求（归一化） | 通用 |
| `[1]` | `ar` | 累计 AR | 通用 |
| `[2]` | `remaining_usable_capacity_sum` | 剩余可用容量总和 | 通用 |
| `[3]` | `remaining_service_demand_sum` | 剩余服务需求总和 | 通用 |
| `[4]` | `remaining_usable_ecu_count` | 容量充足 ECU 占比 | 通用 |
| `[5]` | `remaining_services_count` | 剩余服务占比 | 通用 |
| `[6:6+N]` | `initial_cap_pct[j]` | 各 ECU 初始容量占比 | 通用 |
| `[6+N:6+2N]` | `remaining_pct[j]` | 各 ECU 剩余容量占比 | 通用 |
| `[6+2N:6+3N]` | `conflict_flag[j]` | 当前服务放该 ECU 是否冲突 | _p 系列 |
| **`[6+3N:6+4N]`** | **`ecu_allowed_frac[j]`** | **仍可无冲突放置的服务比例** | **lt 专有** |
| `[6+4N:6+5N]` | `valid_flag[j]` | 有效动作标志（容量 + 冲突双满足） | _p 系列 |
| `[6+5N:6+5N+M]` | `remaining_svcs[t]` | 剩余服务需求序列（M 维） | 通用 |

**为什么需要 `ecu_allowed_frac`**：在 lt 场景中，一台 ECU 放了若干服务后，它对未来服务的冲突状态会逐步收窄（`ecu_allowed` 集合随每次放置缩小）。这个动态信息让智能体知道"这台 ECU 未来还能接多少类服务"，从而做更长远的规划。

P5 在此基础上额外追加 `lambda_norm` 标量，观测维度为 `5N + 7 + M`。

---

## AR 计算（与 eq 变体的关键差异）

```
AR = Σ_j (total_req_on_j / cap_j) / active_ecus
```

- 分子：每台**活跃** ECU 的资源利用率之和
- 分母：**活跃 ECU 数**（有服务的 ECU 数，不是 N）
- 目标：把服务紧凑地集中到少数高匹配度的 ECU 上

这个分数目标（fractional objective）ILP 侧通过 Dinkelbach 迭代求解，RL 侧通过 `_total_ru / active_ecus` 增量维护。

代码实现（env.py 中）：

```python
_active = sum(1 for j in range(self.N) if self.ecu_placements[j])
self.ar = self._total_ru / _active if _active > 0 else 0.0
```

---

## 各问题的约束处理策略

| 问题 | 容量约束 | 冲突约束 | 主要特点 |
|------|----------|----------|----------|
| P1 | 硬终止 | 无 | 基础原型（单服 per ECU 旧逻辑，仅教学用） |
| P2 | ILP 强约束 | ILP 强约束 | Dinkelbach 全局最优，oracle 基线 |
| P3 | 硬掩码（容量不足强制跳转最大余量 ECU） | 软惩罚（`-req/total_cap`） | 容量安全，冲突可探索 |
| P4 | 硬掩码 | 硬掩码（双掩码合并） | 两类约束都强制，最保守 |
| P5 | 硬掩码 | Lagrangian 自适应惩罚（`λ` 对偶上升） | 软冲突 + obs 含 `λ_norm`，维度 `5N+7+M` |
| P6 | 硬掩码 | 硬掩码 | terminal bonus 按违规率连续缩放，奖励更丰富 |
| DQN/DDQN | 硬终止（非法即 done） | 硬终止 | 离策略值函数，经验回放 |

---

## Forced-overflow Fallback 设计

当 `action_masks()` 全为 False（所有 ECU 都超容量或冲突）时：

```python
# 选剩余容量最大的 ECU 作为 fallback
best = int(np.argmax(self.remaining_vms))
mask[best] = True
# step 内给予重惩罚 -2.0
```

这保证 episode **始终跑满 M 步**，不会因为无可行动作而提前终止（避免 sparse signal 问题）。

---

## `ecu_allowed` 集合的增量维护

冲突状态的核心数据结构：

```python
self.ecu_allowed = [set(range(self.M)) for _ in range(self.N)]
# ecu_allowed[j] 初始包含所有服务索引；
# 每次在 ECU j 放置服务 i 后，从所有含 i 的冲突子集中
# 移除同子集的其他服务索引。
```

更新函数（O(K) 时间复杂度，K 为冲突集数量）：

```python
def _update_ecu_allowed(self, ecu_idx: int, svc_idx: int) -> None:
    for subset in self.conflict_sets:
        if svc_idx in subset:
            self.ecu_allowed[ecu_idx] -= (subset - {svc_idx})
```

冲突检查退化为 O(1)：

```python
def _has_conflict(self, ecu_idx: int, svc_idx: int) -> bool:
    return svc_idx not in self.ecu_allowed[ecu_idx]
```

---

## 冲突集索引重映射

从 YAML 加载场景时，冲突集用的是原始服务顺序的索引。但 reset 时服务会按需求降序重排，`self._step` 代表的是排序后的位置，需要映射：

```python
sorted_order = sorted(range(len(self.services)),
                      key=lambda i: self.services[i].requirement, reverse=True)
orig_to_new = {orig: new for new, orig in enumerate(sorted_order)}
self.conflict_sets = [
    {orig_to_new[i] for i in cs if i < len(self.services)} for cs in _cs
]
```

如果跳过此重映射，冲突检查会错位（用新位置索引去查旧索引定义的冲突集）。

---

## 与 `ecu_gt_svc_p` 的核心设计差异对比

| 方面 | `ecu_gt_svc_p`（N > M） | `ecu_lt_svc_p`（N < M） |
|------|------------------------|------------------------|
| 每 ECU 服务数 | 最多 1 个 | 多个（bin packing） |
| 占用标志 `occupied_flag` | 有（一机一服务核心信息） | 无（ECU 可多次使用） |
| `ecu_allowed_frac` | 无 | 有（冲突状态动态演化） |
| obs 维度（P3/P4/P6） | `5N+6+M` | `5N+6+M`（但含义不同） |
| AR 分母 | active ECUs（通常接近 M） | active ECUs（通常接近 N，所有 ECU 都会活跃） |
| 主要优化难点 | 一对一最优匹配 + 冲突 | Bin packing + 冲突交叉限制 |

---

## 各问题奖励函数汇总（最终版）

### P3（无约束基线）

```
reward = ru                    # 始终 req/cap_i，违规不清零
       + terminal_bonus        # 始终 +AR（无惩罚，无约束上界）
```

### P4（双 hard mask）

```
reward = ru                    # 合法步（fallback 时 ru=0）
       + violation_penalty     # fallback 触发：-2.0
       + terminal_bonus        # +AR（零违规）或 +0.1*AR（有违规）
```

### P5（Lagrangian 冲突惩罚）

```
reward = match_gain                        # 始终 req/cap_i（对齐 eq P5）
       - (lambda_val + 0.2) * c_t          # 冲突 Lagrangian 软惩罚
       - forced_overflow_penalty           # 全满 fallback：-2.0
       + terminal_bonus                    # +AR（零违规）或 -AR
```

### P6（双 hard mask + 连续 terminal）

```
reward = ru                                # 合法步（fallback 时 ru=0）
       + violation_penalty                 # fallback 触发：-2.0
       + terminal_bonus                    # AR * max(0, 1 - violation_rate)
```

### DQN / DDQN（软惩罚 + terminal bonus）

```
reward = ru                                # 合法步
       + cap_penalty                       # 容量违规：-2.0
       + conflict_penalty                  # 冲突违规：-2.0
       + terminal_bonus                    # +AR（零违规）或 -AR（lt 专属适配）
```

---

## 场景生成约束（`generate_config.py`）

每个场景需同时满足：

1. `N < M`（ECU 数严格小于服务数）
2. `sum(ecu_capacity) > sum(svc_requirement)`（总容量大于总需求）
3. 冲突集中每个子集大小随机（2 到 M 之间），共 K 个子集

场景数量与 `ecu_eq_svc` 保持一致（200 个场景）。

---

## 设计原则总结

1. **Large-first 排序**：reset 时服务按需求降序排列，让大需求服务优先分配，早期步骤更容易满足容量约束，减少 forced-overflow 概率。

2. **`ecu_allowed` 增量更新而非重算**：每次放置后 O(K) 更新，避免每步 O(K×M) 的全量扫描。

3. **Fallback 保证 M 步完整执行**：即使所有 ECU 都无可行动作，也不提前终止，保持训练信号的稳定性。

4. **P5 将 `λ` 放入观测**：让策略网络感知当前惩罚强度，动态调整对冲突的容忍度。

5. **ILP 用 Dinkelbach 迭代**：AR 是分数目标（分母 active_ecus 依赖解），单次 LP 无法直接优化，Dinkelbach 方法每轮收敛后更新参数 λ ← total_util / active_ecus。
