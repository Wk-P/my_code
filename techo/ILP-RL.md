# ILP - RL

## 一、理论讨论

### 预训练设想

1. ILP 预训练 RL
2. RL 最后类似 teacher - student 模型
3. RL 具备泛化能力，不需要再进行 ILP 环境设置
    - 比如之前还需要设置不同的函数对目标进行优化，constraints 需要很多算式，类似目前的 ar 优化。泛化后是否能够出现仅针对环境即可筛选出近似 ILP 的最优解，而且能够自动化直接构建目标 reward，完全省略 ILP 计算。（或许需要数学验证）

不知道以上是否能很好的得到结果

### 核心思路拆解

用 ILP 求得的最优解对 RL 做"预训练"，让 RL 从 teacher-student 模式出发，最终具备泛化能力：
不再需要针对每个环境重新设计目标函数/约束（如当前 ar 优化里手写的各种算式），
而是能直接从环境自动构建 reward，省略 ILP 计算这一步。

### 需要先厘清的问题

1. **"预训练"具体怎么做**
   - 方案 A：用 ILP 解出的最优分配 $x_{ij}$ 做行为克隆（behavior cloning，BC）初始化策略网络，再用 PPO/DQN 微调。
   - 方案 B：把 ILP 解作为 reward shaping 的势函数（potential-based shaping）。
   - **注意**：v0.3.1 已经尝试过 feasibility-potential reward shaping，结果是负向的，已在 v0.3.2 中 revert（见 commit c6c7ec1 / c61f8a5）。方案 B 需要先分析当时失败的原因，避免重复踩坑。
2. **泛化的边界**
   - RL 较容易泛化的是"同类结构、不同数值参数"（如同样是 lt 场景但换一批容量/需求采样）。
   - "跨结构泛化"（如 lt→gt，或引入新的冲突集类型）通常需要训练时覆盖更广的环境分布（domain randomization），这与当前 lt/eq/gt 分场景独立训练的做法存在冲突，需要重新设计训练环境采样策略。
3. **"自动构建 reward、省略 ILP"**
   - 如果要做到，本质上是学习一个通用的 reward model / 价值函数逼近，已经超出普通模仿学习范畴，接近 meta-RL / reward learning 方向。
   - 需要数学论证或至少一个可控小规模实验先验证可行性。

采用方案 A（behavior cloning）作为第一步落地方式，方案 B 的重新设计留待后续。

### 目标算法的选择

PPO baseline（无约束处理机制）不是优化目标，因为它本身没有 violation constraint，谈不上"预训练改善约束满足"。真正要做 BC 预训练对比的是五个**带约束处理机制**的算法，覆盖三类不同的约束处理哲学：

- 硬约束（action masking）：`ppo_mask`
- 软约束（自适应 λ 惩罚）：`ppo_lagrangian`
- 修复式（best-fit repair 启发式）：`ppo_opt`
- 无 mask、硬惩罚：`dqn`、`ddqn`（Double DQN）

覆盖 eq / gt / lt 三个场景，共 15 个 (scenario, algo) 组合。

### 下一步待验证方向

1. 扩大 BC 专家数据的场景覆盖，观察是否能缓解"BC 预训练可能让策略过拟合到贪心分配模式"的问题。
2. 尝试调整 BC epoch 数 / PPO fine-tune 步数比例。
3. 重新设计方案 B（reward shaping，结合 v0.3.1 失败经验），与方案 A 对比。

---

## 二、代码实现

### ILP 专家轨迹的构建（`shared/bc_pretrain.py`）

1. `ilp_expert_actions(caps, reqs, conflict_sets, sorted_desc)`：对场景跑 `shared.ilp_utils.solve_ilp`（非 Dinkelbach 单次 LP），得到最优分配 $x_{ij}$；按目标 env 的内部服务呈现顺序重排，转成逐步 expert action 序列。
   - **`sorted_desc` 参数**：并非所有 env 都按需求降序排列服务，三个场景 × ppo系/dqn系之间存在真实差异（eq 的 ppo 系降序、dqn 系不排序；gt 全部不排序；lt 全部降序）——不能假设统一常量。传错这个参数不会报错，只会让专家动作和 env 实际呈现的服务对不上号，静默拉低 `build_bc_dataset` 的可用场景比例（`mismatch_skipped` 升高）。
2. `build_bc_dataset()`：用 expert action 序列逐步 replay 目标 env，采集 `(obs, action_mask, action)` 三元组；只保留 expert 轨迹在 env 里能完整合法回放（零违规、mask 校验通过）的场景，防止 ILP 与 env 的约束编码出现细微不一致。
3. `pretrain_actor_critic()`：给 PPO / MaskablePPO 用，`-dist.log_prob(action).mean()` 作为 loss（`MaskableCategoricalDistribution` 自动处理 mask；无 mask 场景直接调用 `policy.get_distribution(obs)`）。
4. `pretrain_dqn()`：DQN/DDQN 的 Q 网络没有策略分布可言，改用 DQfD 风格的 **large-margin classification loss**：
   $$L = \max_a\big[Q(s,a) + \text{margin}\cdot\mathbb{1}(a\neq a_E)\big] - Q(s,a_E)$$
   逼迫专家动作的 Q 值比其余动作高出至少 margin（=0.8），预训练后把 online 网络权重同步给 target 网络，避免用未训练的随机 target 引导前几步 bootstrap。

### 每个算法目录下的 `run_all_bc.py`

复用对应 `run_all.py` 里的环境工厂 / 回调 / 评估 / 画图函数（`import run_all as RA`），只在 `model.learn()` 前插入 BC 预训练步骤，其余流程（ILP 求解、评估、画图、JSON/CSV 输出）与 baseline 完全一致，保证两者可比。

每个 `config.py` 新增：
```python
BC_EPOCHS     = 20
BC_BATCH_SIZE = 256
BC_LR         = 1e-3
BC_MARGIN     = 0.8   # 仅 DQN/DDQN
```

跨场景移植这些脚本时，除了 `sorted_desc` 之外还要核对两类隐藏差异，不能假设三个场景的 `run_all.py` 结构完全一致：
- `ppo_opt` 的 `run_episodes()` 返回字段名在 lt 和 eq/gt 之间不同（`viol_rates`/`cap_viols` vs `repair_rates`/`cap_violations`）。
- 部分场景的 PPO 构造参数不同（如 gt 的 `ppo_mask` 不传 `ent_coef`，而 eq/lt 传）。

### 看板前后端配套改动（`app/backend/main.py` + `app/frontend/`）

- `results.json` 里非算法结果的元数据键（`"bc"`、`"exp_id"`）必须加入 `_algo_key()` 的保留字段列表，否则会被误判成算法结果键，读取时崩溃——改 `results.json` schema 时要记得同步这张保留字段表。
- 每条结果记录暴露 `is_bc` / `display_algo` 字段（数据来源是 `results.json` 里 algo_key 是否以 `_bc` 结尾，不是猜目录名或分支名）。
- 同一批实验的 baseline 与 BC 结果在表格里用色点 + 徽章（`variant-dot` / `variant-badge`）区分，而不是仅靠算法名文字里的 `+bc` 后缀。
- `/api/progress`（实时训练进度）的进程识别正则同时匹配 `run_all.py` / `run_all_bc.py`；当命令行缺 `scenarios/` 前缀时会退回读取 `/proc/<pid>/cwd` 解析 scenario——这只是兜底，正常应该都走标准启动器，命令行天然带完整路径。

---

## 三、文件与目录结构

### 分支与代码

`pretrain` 分支相对 `main` 新增：

```
shared/bc_pretrain.py                          共享 BC 逻辑
scenarios/<eq|gt|lt>/<algo>/run_all_bc.py       每个算法的 BC 预训练+微调入口（15 个）
scripts/resume_scenario.sh                      扩展支持 <algo>_bc 后缀
scripts/start_experiment.sh                     同上（转发给 resume_scenario.sh）
```

`main` 分支保持"纯 RL 训练、优化目标只有 ar"不变，作为可随时回退的基线。

### 结果数据按分支物理隔离

`results/` 整个目录被 gitignore，切换 git 分支不会自动改变磁盘上已有的数据。为了让"BC 预训练"实验和"main 分支纯 RL 训练"两条线不互相污染，`shared/paths.py` 的 `results_dir()` 显式在路径里加入当前分支名：

```js
results/<git-branch>/<scenario>/<algo>/<run>/
```

`shared/paths.py`（训练脚本用）在模块加载时解析一次当前分支；`app/backend/main.py`（长期运行的看板服务）在每次请求时重新探测分支，这样另一个终端的 `git checkout` 不需要重启服务就能反映到看板上。

看板前端不强行"只看当前分支"，而是提供一个按真实分支切换的 tab（`ExperimentTree.vue`），每个 tab 对应一次 `/api/experiments?branch=<name>` 请求，读取该分支自己的 `results/<branch>/` 子树；详情页路由带上 `branch` 段（`#/run/<branch>/<scenario>/<algo>/<run>`），避免从非当前分支的 tab 钻进详情页时读错数据。

### exp_id：一批实验的归属标识

`shared/paths.py` 的 `resolve_exp_id()` 设计里，一个 `exp_id` 代表"一整批实验"（如 eq+gt+lt 全部算法共用一个 id），通过 `$EXP_ID` 环境变量在多个脚本进程间共享；不设置则各自随机生成。`run_all_bc.py` 的输出目录是 `<exp_id>_bc`（区别于 baseline 的 `<exp_id>`），因此目录名和 batch 归属不再是同一件事——`run_all_bc.py` 额外把真实 `exp_id` 显式写入 `results.json` 的 `"exp_id"` 字段，看板按这个字段分组，不是按目录名。

批量实验必须通过标准启动器共享 `EXP_ID`（`scripts/start_experiment.sh` 内部会对 eq/gt/lt 三个场景各调用一次 `resume_scenario.sh`，共用同一个 `EXP_ID`）：

```bash
scripts/start_experiment.sh bc-compare
```

`bc-compare` 是脚本提供的便捷参数，等价于把 5 个支持 BC 的算法（`ppo_mask`/`ppo_lagrangian`/`ppo_opt`/`dqn`/`ddqn`）的 baseline 与 `_bc` 变体都列出来，再加上不参与 BC 对比的 `ppo`，三场景共 33 个模型；也可以手写 algo 列表只跑一部分，例如：

```bash
scripts/start_experiment.sh ppo_mask ppo_mask_bc ppo_lagrangian ppo_lagrangian_bc dqn dqn_bc
```

