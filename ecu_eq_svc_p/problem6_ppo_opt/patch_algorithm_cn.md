# 重复 ECU 分配的最优适配补救算法

## 概述

当 RL 智能体选择了一个已被服务 `s1` 占用的 ECU（`ecu_n`）时，
算法不会简单地记录违规并覆盖，而是将两个服务在 `ecu_n` 和一个选出的空闲 ECU（`ecu_m`）之间
**重新调度**，以最小化约束违规并最大化资源利用率。

---

## 所需新增状态变量

| 变量              | 类型                  | 描述                                               |
| ----------------- | --------------------- | -------------------------------------------------- |
| `ecu_service_idx` | `int[N]`，初始值 `-1` | 记录每个 ECU 上当前运行的服务下标（`-1` 表示空闲） |

该变量须在 `reset()` 中初始化为 `-1`，并在每次合法分配后更新。

---

## 触发条件

```
ecu_assigned[ecu_n] == True
```

- `s1` = `services[ecu_service_idx[ecu_n]]` — 已在 `ecu_n` 上运行的服务
- `s2` = `svc` — 当前步骤新来的服务

---

## 步骤 1 — 确定 `s_stay` 和 `s_move`

比较 `s1.req` 和 `s2.req`：

**情况 A：`s2.req > s1.req`**（新来的更大）

- `s_stay = s2`（较大，留在 `ecu_n`）
- `s_move = s1`（较小，将被迁移）
- 额外检查：若 `initial_vms[ecu_n] < s2.req` → `capacity_violations += 1`

**情况 B：`s2.req <= s1.req`**（新来的更小或相等）

- `s_stay = s1`（较大，留在 `ecu_n`）
- `s_move = s2`（较小，将被迁移）
- 无需检查容量（`s1` 原本已合法放置在 `ecu_n`）

---

## 步骤 2 — 寻找最优适配 `ecu_m`（正常路径）

在所有**空闲** ECU（`ecu_assigned[j] == False`，`j != ecu_n`）中搜索：

```
candidates = [j for j in range(N) if not ecu_assigned[j] and initial_vms[j] >= s_move.req]
ecu_m = 使 (remaining_vms[j] - s_move.req) 最小的候选 ECU   # 最紧适配
```

- **找到** → 执行步骤 3
- **找不到** → 进入步骤 5（Fallback）

---

## 步骤 3 — 执行迁移（正常路径）

```
remaining_vms[ecu_n]  = initial_vms[ecu_n] - s_stay.req
remaining_vms[ecu_m] -= s_move.req

ecu_service_idx[ecu_n] = s_stay 的下标
ecu_service_idx[ecu_m] = s_move 的下标
ecu_assigned[ecu_m]    = True
```

---

## 步骤 4 — 修正 AR（正常路径）

`s_move` 的 RU 贡献从 `cap_n` 变为了 `cap_m`，对运行 AR 做增量修正（无需重新遍历所有步骤）：

$$AR_{\text{new}} = AR_{\text{old}} + \frac{s\_move.req}{self.\_step} \cdot \left(\frac{1}{cap_m} - \frac{1}{cap_n}\right)$$

> `self._step` 为当前步骤前**已提交**的服务数量，
> 即 `s_move` 被计入 AR 时所用的分母。

---

## 步骤 5 — Fallback（找不到容量足够的空闲 ECU）

将所有空闲 ECU 作为 `ecu_m` 候选，枚举两种排列方案：

| 排列 | `ecu_n` 上运行 | `ecu_m` 上运行 |
| ---- | -------------- | -------------- |
| X    | s1             | s2             |
| Y    | s2             | s1             |

对每个 `(ecu_m, 排列)` 组合计算：

**违规数**（0、1 或 2）：

```
viol = 0
if s_on_ecu_n.req > initial_vms[ecu_n]:  viol += 1
if s_on_ecu_m.req > initial_vms[ecu_m]:  viol += 1
```

**得分**（越大越好）：
$$\text{score} = \frac{s1.req + s2.req}{cap_n + cap_m}$$

**选取规则**（两级优先）：

1. 最小化违规数 — **违规数 = 2 的组合永远不被选取**
2. 违规数相同时，取得分最高的组合

执行选中的排列，将实际违规数追加到 `capacity_violations`。  
`single_service_violations` **不增加**（两个服务最终分布在不同 ECU 上）。

---

## 汇总表

| 情形                                   | 路径     | `capacity_violations` | `single_service_violations` |
| -------------------------------------- | -------- | --------------------- | --------------------------- |
| 正常路径，`ecu_n` 无容量问题（情况 B） | 正常     | +0                    | +0                          |
| 正常路径，`ecu_n` 有容量问题（情况 A） | 正常     | +1                    | +0                          |
| Fallback，找到 0 违规方案              | Fallback | +0                    | +0                          |
| Fallback，找到 1 违规方案              | Fallback | +1                    | +0                          |
| 2 违规的组合                           | —        | 永不选取              | —                           |
