# RL state 设计（P3-P6 PPO 标准）

当前 PPO 标准 observation 维度为 `4N + 6 + M`，顺序如下。

| 参数名                          | description                                                                                                    |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `service_demand_norm`           | 当前待部署服务的需求量（`requirement vms`），按 `max(initial_vms)` 归一化。                                    |
| `ar`                            | 当前已部署服务的平均资源利用率（Average Resource Utilization, `AR`）。                                         |
| `remaining_usable_capacity_sum` | 当前所有“未占用 ECU”的剩余可用容量总和（`capacity vms` 汇总），负值按 0 截断后再按 `sum(initial_vms)` 归一化。 |
| `remaining_service_demand_sum`  | 当前到回合结束的剩余服务需求总和（`all required vms of remaining services`），按 `sum(initial_vms)` 归一化。   |
| `remaining_usable_ecu_count`    | 当前剩余可用 ECU 数量（`sum of remaining ECUs`），按 `N` 归一化。                                              |
| `remaining_services_count`      | 当前剩余服务数量（`sum of remaining services`），按 `M` 归一化。                                               |
| `initial_cap_pct[j]`            | 第 `j` 台 ECU 的初始容量占比（容量向量），按 `max(initial_vms)` 归一化。                                       |
| `remaining_pct[j]`              | 第 `j` 台 ECU 的剩余容量占比（`RU` 相关逐 ECU 信息），按各自初始容量归一化。                                   |
| `occupied_flag[j]`              | 第 `j` 台 ECU 是否已分配（1=已占用，0=未占用）。                                                               |
| `valid_flag[j]`                 | 第 `j` 台 ECU 对“当前服务”是否可行动作（1=可选，0=不可选；与 action mask 一致）。                              |
| `remaining_svcs[t]`             | 从当前 step 到末尾的剩余服务需求序列（已按需求降序），已放置位置填 0，按 `max(initial_vms)` 归一化。           |

备注：

- 当前 `services` 在每个 episode 的 `reset` 时会按 `requirement` 降序排序（largest-demand-first）。
- 这套状态是“全局汇总 + 逐 ECU 细粒度 + 剩余需求序列”的组合，既保留可行动作判别能力，也提供未来信息。
- P3、P4、P6 直接使用这套标准状态。
- P5 使用这套标准状态，并额外附加一个 `lambda_norm` 标量，用于表达当前 Lagrangian 乘子。

维度增加

- 之前维度为 `N+2`（需求 + AR + 每台 ECU 的剩余容量比），现在增加了：
    - `remaining_service_demand_sum`：未来总剩余需求量，提供全局未来信息。
    - `remaining_usable_ecu_count` 和 `remaining_services_count`：提供当前资源和任务规模的全局统计。
    - 每台 ECU 的 `initial_cap_pct`、`occupied_flag`、`valid_flag`：提供逐 ECU 的详细状态和可行动作信息。
    - `remaining_svcs[t]`：未来剩余服务需求序列，提供更细粒度的未来信息。

之前维度数量为 `N+2`，现在增加了 `2 + 3N + M`，总维度为 `4N + 6 + M`。
之前的维度都是：
Dimension: N+2 (1 service demand + 1 AR + N ECU remaining capacities)

- `service_demand_norm`：1 维
- `ar`：1 维
- `remaining_usable_capacity_sum`：1 维
- 每台 ECU 的剩余容量比：`N` 维
  现在的维度是：
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
- 总维度：`1 + 1 + 1 + 1 + 1 + 1 + N + N + N + N + M = 4N + 6 + M`。
