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
| v1.2.4 | 2026-07-21 | 新增实验进程看门狗；lt/ppo_mask多seed(42/1/2/3/4)诊断实验，success_rate均值≈0.640(区间0.575~0.675)，系统性瓶颈非seed方差；修复lt/ppo/ppo_opt/dqn/ddqn的TOTAL_STEPS未传scenario参数，SCENARIO_TOTAL_STEPS覆盖表对齐 | [v1.2.4.md](v1.2.4.md) |
| v2.0.0 | 2026-08-07 | **重大发现**：best-of-N的N=8从未调过参，N敏感性扫描(N=8→1024)显示success_rate从52.5%涨到95%，推翻此前"lt success_rate系统性瓶颈≈65%"的结论；40场景逐一定位2个真正卡死的高利用率难例 | [v2.0.0.md](v2.0.0.md) |
| v2.1.0 | 2026-08-07 | Self-Imitation/Expert Iteration微调（纯BC + BC+RL恢复两版）：核心假设不成立，6轮下来N=1单次成功率始终低于基线，负结果 | [v2.1.0.md](v2.1.0.md) |
| v2.2.0 | 2026-08-09 | **AR正式成为优化目标**：终局reward从二元(±M)改为零违反时M*AR、违反时-M，让梯度真正区分成功放置的质量高低；违规仍是硬约束，数学上保证"任何成功优于任何违规"；需要完整重训5M步，结果待补 | [v2.2.0.md](v2.2.0.md) |
| v2.3.0 | 2026-08-14 | **BC预训练永久停用**：隔离实验发现lt/ppo_mask去掉BC后success_rate 47.5%→70.0%(+22.5pp,同样5M步)，是目前发现的最大单一负面因素；state设计(bottleneck_risk特征)贡献较小(+5pp,仅500k步单次验证)，仍需继续研究，另开`add_states`分支跟进 | [v2.3.0.md](v2.3.0.md) |
| v2.4.0 | 2026-08-14 | `add_states`分支：bottleneck_risk特征推广到eq/gt/lt；发现eq/gt仍用v1.1.0旧二元reward(从未获得AR梯度信号)，移植v2.2.0的M*AR reward后AR gap从0.03~0.17压到0.004~0.05——目前影响最大的单一改动；同时发现一次"lt 5M步无特征对照"因stash/sleep时序问题被污染，需重新验证 | [v2.4.0.md](v2.4.0.md) |
| v2.5.0 | 2026-08-15 | `add_states`分支：reward-engineering路线收尾——5轮独立多seed实验(bottleneck shaping/权重对等+熵退火/梯度分级失败/FFD可行性shaping)全部零效应，success_rate 80~90%、AR 0.60~0.65是当前决策结构下的真实能力边界；否决了求解器masking方案(会让success_rate失去研究意义)；lt/eq/gt最终兜底数据完整跑一次，三场景数据严格分开存放 | [v2.5.0.md](v2.5.0.md) |
| v2.8.0 | 2026-08-22 | `add_states`分支：场景池200→2000(lt/eq/gt success_rate明显回升)；修复ILP缓存key不含场景内容的碰撞bug、gt冲突集合采样越界bug；继续5轮reward实验(失败惩罚降权/AR权重课程学习)均在5M步下暴露success/AR此消彼长，且发现"成功场景AR反而比失败场景低"；把分级reward推广到无masking的5个算法，success_rate从历史的0跳到27%~80%(5-seed确认非偶然)；累计10+轮实验收敛结论：AR差距~0.10~0.12是决策结构的结构性上限 | [v2.8.0.md](v2.8.0.md) |

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
