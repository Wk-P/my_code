# RL state 设计（P3-P6 PPO 标准）

当前 PPO 标准 observation 维度为 `4N + 6 + M`，顺序如下。

符号约定（用于所有变量注释）：

- `N`：ECU 数量。
- `M`：服务数量。
- `j`：ECU 下标，取值 `0..N-1`。
- `t`：服务序列下标，取值 `0..M-1`。
- `initial_vms[j]`：第 `j` 台 ECU 初始容量。
- `remaining_vms[j]`：第 `j` 台 ECU 当前剩余容量。
- `req_t`：当前 step 待部署服务需求量。
- `total_cap = sum(initial_vms)`：系统总初始容量。

| 参数名                          | 含义（语义）                                           | 计算方式（定义）                                                  | 取值范围/单位                            |
| ------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------- | ---------------------------------------- |
| `service_demand_norm`           | 当前待部署服务需求（当前任务强度）                     | `req_t / max(initial_vms)`                                        | `[0, 1]`，无量纲                         |
| `ar`                            | 当前累计平均资源利用率（Average Resource Utilization） | 环境内累计统计值（平均 `requirement/capacity`）                   | 一般在 `[0, 1]`，无量纲                  |
| `remaining_usable_capacity_sum` | 当前未占用 ECU 的可用容量总量（全局资源余量）          | `sum(max(remaining_vms[j], 0) for j if ECU j 未占用) / total_cap` | `[0, 1]`，无量纲                         |
| `remaining_service_demand_sum`  | 从当前到回合结束的总剩余需求（全局任务余量）           | `sum(remaining service requirements) / total_cap`                 | `[0, 1]`，无量纲                         |
| `remaining_usable_ecu_count`    | 当前可继续接单的 ECU 数量占比                          | `(# 可用 ECU) / N`                                                | `[0, 1]`，无量纲                         |
| `remaining_services_count`      | 当前尚未放置服务数量占比                               | `(# remaining services) / M`                                      | `[0, 1]`，无量纲                         |
| `initial_cap_pct[j]`            | 第 `j` 台 ECU 初始容量占比（静态结构信息）             | `initial_vms[j] / max(initial_vms)`                               | `[0, 1]`，无量纲                         |
| `remaining_pct[j]`              | 第 `j` 台 ECU 剩余容量占比（动态局部资源）             | `remaining_vms[j] / initial_vms[j]`                               | P4/P6/DQN 通常 `[0,1]`；P3/P5 可小于 `0` |
| `occupied_flag[j]`              | 第 `j` 台 ECU 是否已分配过服务                         | 已占用记 `1`，未占用记 `0`                                        | `{0, 1}`                                 |
| `valid_flag[j]`                 | 第 `j` 台 ECU 对当前服务是否可行动作                   | 满足“未占用且容量足够”记 `1`，否则 `0`                            | `{0, 1}`                                 |
| `remaining_svcs[t]`             | 当前到末尾的剩余服务需求序列（未来信息）               | 已按需求降序；已放置位置补 `0`；再除以 `max(initial_vms)`         | `[0, 1]`，无量纲                         |

备注：

- 当前 `services` 在每个 episode 的 `reset` 时会按 `requirement` 降序排序（largest-demand-first）。
- 这套状态是“全局汇总 + 逐 ECU 细粒度 + 剩余需求序列”的组合，既保留可行动作判别能力，也提供未来信息。
- P3、P4、P6 直接使用这套标准状态。
- P5 使用这套标准状态，并额外附加一个 `lambda_norm` 标量，用于表达当前 Lagrangian 乘子。

## 维度演进

- v1.1 初始版本：`N + 2`
    - 变量与含义：
        - `service_demand_norm`（1 维）：当前待放置服务需求，反映“当前任务压力”。
        - `ar`（1 维）：当前累计平均资源利用率，反映“当前已取得效果”。
        - `remaining_pct[j]`（N 维）：每台 ECU 剩余容量占比，反映“逐 ECU 可用资源”。
    - 说明：这是最早的简化状态，仅包含“当前任务 + 当前效果 + 逐 ECU 剩余容量”。

- 中间版本：`3N + 2 + M`
    - 在 `N+2` 基础上增加变量与含义：
        - `occupied_flag[j]`（N 维）：ECU 是否已被分配，显式表示“一机一服务”的占用状态。
        - `valid_flag[j]`（N 维）：对当前服务是否可行，显式表示动作可行性。
        - `remaining_svcs[t]`（M 维）：未来剩余服务需求序列，提供前瞻信息。
    - 中间版本完整变量含义：
        - `service_demand_norm`：当前任务强度。
        - `ar`：当前累计效果。
        - `remaining_pct[j]`：逐 ECU 剩余资源。
        - `occupied_flag[j]`：逐 ECU 占用状态。
        - `valid_flag[j]`：逐 ECU 对当前任务的可行动作标记。
        - `remaining_svcs[t]`：未来任务需求轨迹。

- 当前 PPO 标准（P3/P4/P6）：`4N + 6 + M`
    - 在 `3N+2+M` 基础上增加变量与含义：
        - `remaining_usable_capacity_sum`（1 维）：系统剩余可用容量总量（全局资源余量）。
        - `remaining_service_demand_sum`（1 维）：系统剩余服务需求总量（全局任务余量）。
        - `remaining_usable_ecu_count`（1 维）：剩余可用 ECU 数量占比（全局资源规模）。
        - `remaining_services_count`（1 维）：剩余服务数量占比（全局任务规模）。
        - `initial_cap_pct[j]`（N 维）：每台 ECU 初始容量占比（静态结构信息）。
    - 当前标准完整变量含义（11 组）：
        - `service_demand_norm`：当前任务强度。
        - `ar`：当前累计效果。
        - `remaining_usable_capacity_sum`：全局资源余量。
        - `remaining_service_demand_sum`：全局任务余量。
        - `remaining_usable_ecu_count`：可用资源规模。
        - `remaining_services_count`：待完成任务规模。
        - `initial_cap_pct[j]`：逐 ECU 初始容量结构。
        - `remaining_pct[j]`：逐 ECU 动态剩余资源。
        - `occupied_flag[j]`：逐 ECU 占用状态。
        - `valid_flag[j]`：逐 ECU 当前可行动作标记。
        - `remaining_svcs[t]`：未来任务需求序列。

- 当前 P5：`4N + 7 + M`
    - 在 PPO 标准 `4N+6+M` 基础上额外增加：
        - `lambda_norm`（1 维）：Lagrangian 乘子归一化值，表示当前约束惩罚强度。

## 当前维度展开（P3/P4/P6）

- `service_demand_norm`：1 维
- `ar`：1 维
- `remaining_usable_capacity_sum`：1 维
- `remaining_service_demand_sum`：1 维
- `remaining_usable_ecu_count`：1 维
- `remaining_services_count`：1 维
- 每台 ECU 的 `initial_cap_pct`：`N` 维
- 每台 ECU 的 `remaining_pct`：`N` 维
- 每台 ECU 的 `occupied_flag`：`N` 维
- 每台 ECU 的 `valid_flag`：`N` 维
- `remaining_svcs[t]`：`M` 维
- 总维度：`1 + 1 + 1 + 1 + 1 + 1 + N + N + N + N + M = 4N + 6 + M`

## English Table (Current Standard State)

| Description                                                     | Dimension        | DataType Description                                                       |
| --------------------------------------------------------------- | ---------------- | -------------------------------------------------------------------------- |
| Current service demand (`service_demand_norm`)                  | 1                | `float32`, normalized scalar in `[0, 1]`                                   |
| Current cumulative AR (`ar`)                                    | 1                | `float32`, normalized scalar, typically in `[0, 1]`                        |
| Remaining usable capacity sum (`remaining_usable_capacity_sum`) | 1                | `float32`, normalized scalar in `[0, 1]`                                   |
| Remaining service demand sum (`remaining_service_demand_sum`)   | 1                | `float32`, normalized scalar in `[0, 1]`                                   |
| Remaining usable ECU count (`remaining_usable_ecu_count`)       | 1                | `float32`, normalized scalar in `[0, 1]`                                   |
| Remaining services count (`remaining_services_count`)           | 1                | `float32`, normalized scalar in `[0, 1]`                                   |
| Initial capacity vector (`initial_cap_pct[j]`, `j=0..N-1`)      | N                | `float32` vector, each element normalized to `[0, 1]`                      |
| Remaining capacity vector (`remaining_pct[j]`, `j=0..N-1`)      | N                | `float32` vector; usually `[0, 1]`, may be `<0` in overload-allowed envs   |
| Occupancy flags (`occupied_flag[j]`, `j=0..N-1`)                | N                | binary flags stored as `float32` (`0.0` or `1.0`)                          |
| Valid action flags (`valid_flag[j]`, `j=0..N-1`)                | N                | binary flags stored as `float32` (`0.0` or `1.0`)                          |
| Remaining service sequence (`remaining_svcs[t]`, `t=0..M-1`)    | M                | `float32` vector, normalized to `[0, 1]`, placed positions padded with `0` |
| **Total**                                                       | **`4N + 6 + M`** | concatenated `np.float32` observation vector                               |

Notes:

- This table describes the current standard state used by P3/P4/P6.
- P5 uses the same state and appends one extra scalar `lambda_norm` (thus `4N + 7 + M`).
