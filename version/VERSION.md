# 版本历史索引

统一管理所有 git tag 对应的改动记录。版本号唯一真源见
[shared/version_config.py](../shared/version_config.py)::`CURRENT_VERSION`
——升级版本只改那一个文件，不在别处写死；这里只是历史变更记录，不是版本号来源。

有独立详细文档（`version/vX.Y.Z.md`）的版本点文件名可跳转；只有 git tag
注解、没有独立文档的版本，摘要直接写在这张表里，完整信息用
`git show vX.Y.Z` 或 `git tag -l --format='%(contents)' vX.Y.Z` 查看。

| 版本 | 日期 | 摘要 | 详细文档 |
|---|---|---|---|
| v0.0.1 | 2026-06-24 | 初始版本 | [v0.0.1_vs_v0.1.0.md](v0.0.1_vs_v0.1.0.md) |
| v0.1.0 | 2026-06-24 | 核心算法代码定型 | [v0.0.1_vs_v0.1.0.md](v0.0.1_vs_v0.1.0.md) |
| v0.1.1 | 2026-07-02 | dashboard (FastAPI+Vue) + N_ENVS统一 + shared/整合，在v0.1.0核心算法代码基础上 | — |
| v0.2.0 | 2026-07-02 | 场景/算法目录重构为 scenarios/ + shared/ 架构 | [v0.2.0.md](v0.2.0.md) |
| v0.2.1 | 2026-07-02 | 部署可靠性修复（backend --reload、deploy.sh） | — |
| v0.3.1 | 2026-07-03 | eq/gt/lt全场景feasibility-shaping实验 — lt场景success_rate上是负结果 | — |
| v0.3.2 | 2026-07-03 | 回滚v0.3.1的feasibility-shaping实验，退回v0.2.1的env baseline | — |
| v1.0.1 | 2026-07-09 | 熵系数bug修复（静态0.005对齐，无调度）exp_id=12ea950c | — |
| v1.0.2 | 2026-07-10 | 熵系数调度（前高后低，缓解PPO探索过早收敛）exp_id=9a75a253 | — |
| v1.0.3 | 2026-07-10 | 负优势样本剪枝（CMA-ES风格，权重0.1）exp_id=ac669003 | — |
| v1.0.4 | 2026-07-10 | 剪枝力度调优（权重0.3，更温和）exp_id=0c80d082 | — |
| v1.1.0 | 2026-07-10 | reward改为纯终局稀疏奖励，GAE负责反向信用分配；仅ppo_mask(部分场景ppo_lagrangian)在此奖励下success_rate=1.0，其余算法掉到0 | [v1.1.0.md](v1.1.0.md) |
| v1.2.0 | 2026-07-15 | 训练步数5M→2M；ppo_mask训练曲线补充valid_placed指标（原Services Placed曲线因env强制补齐机制恒等于M，不反映真实进展） | [v1.2.0.md](v1.2.0.md) |
| v1.2.1 | 2026-07-15 | AR指标排除失败episode，只在success episode上算ar_mean/ar_std（此前混合成功失败平均，与ILP的AR不是同一统计量） | [v1.2.1.md](v1.2.1.md) |
| v1.2.2 | 2026-07-15 | 评估阶段加best-of-N随机重采样（N=8），缓解在线不可回溯策略的成功率上限；lt/ppo_mask验证 success_rate 42.5%→65.0% | [v1.2.2.md](v1.2.2.md) |
| v1.2.3 | 2026-07-16 | 训练步数改为按(场景,算法)覆盖；lt场景valid_placed曲线2M步未收敛，恢复5M步，eq/gt维持2M | [v1.2.3.md](v1.2.3.md) |

## v1.0.x / v1.1.0 系列实验结果对比（lt/eq/gt × ppo_mask/ppo_lagrangian）

> 早于AR口径修正（v1.2.1）之前的历史数据——AR是成功+失败episode混合平均，
> 与v1.2.1之后的AR数字不可直接比较，仅作为超参演变的参考。

| 版本 | lt/ppo_mask AR (gap) | lt/ppo_lagrangian AR (gap) | eq/ppo_mask AR (gap) | gt/ppo_mask AR (gap) |
|---|---|---|---|---|
| v1.0.1 | 0.6362 (0.0851) | 0.6658 (0.0555) | 0.5281 (0.0040) | 0.5750 (0.0587) |
| v1.0.2 | 0.6302 (0.0910) | 0.6605 (0.0608) | 0.5289 (0.0032) | 0.5815 (0.0523) |
| v1.0.3 | 0.6310 (0.0902) | 0.6619 (0.0594) | 0.5282 (0.0039) | 0.5702 (0.0635) |
| v1.0.4 | 0.6358 (0.0855) | 0.6652 (0.0561) | 0.5286 (0.0035) | 0.5734 (0.0604) |

## 新建版本时的操作步骤

1. 改 [shared/version_config.py](../shared/version_config.py) 的 `CURRENT_VERSION`。
2. 写 `version/vX.Y.Z.md`（背景、改动、依据/实验结果、注意事项——参考 [v1.2.2.md](v1.2.2.md) 的格式），值得留档的版本才写；纯配置微调可以只打tag不写文档，但要把摘要补进这张表。
3. 在这张表里加一行索引。
4. `git add` + commit + `git tag -a vX.Y.Z -m "..."`。
5. `git push origin <branch> && git push origin --tags`（本地tag不会自动同步到远程，必须显式推送）。
