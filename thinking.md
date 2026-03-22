# RL state 设计（当前 P4 实现）

当前 P4 的 observation 维度为 `4N + 6 + M`，顺序如下。

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
