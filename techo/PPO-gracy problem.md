# 如何解决 PPO 中贪心带来的问题
在之前 v0.2.0 和 v0.1.1 的 LT 场景中，由于SVC数量多余ECU，前期选择SVC结束后后面的SVC几乎放不下，成功率大幅下降

因此为了解决 PPO 中贪心带来的问题，我们搜索查询多个论文，找出以下方式尝试

## 方法1：熵正则化 (Entropy Regularization)

这是最直接且原生于PPO的解决方案。

通过在损失函数中加入策略的熵惩罚项 $\mathcal{H}(\pi_\theta)$，强制要求网络输出的动作概率分布保持一定的“均匀性”，从而防止分布过早塌陷为确定性策略。

优化目标：
- $L^{PPO} = L^{CLIP} + c_1 L^{VF} - c_2 \mathcal{H}(\pi_\theta)$

支持论文：
- Proximal Policy Optimization Algorithms (Schulman et al., 2017)。

这篇PPO的开山之作本身就详细探讨了通过调整超参数 $c_2$（Entropy Coefficient）来缓解贪心收敛的必要性。

## 方法2：内在动机与好奇心机制 (Intrinsic Motivation & Curiosity)
在奖励稀疏或环境反馈延迟极高的场景中（例如长周期的金融量化回测阶段），外部奖励不足以指导策略的进化，导致PPO只能在初始随机动作中“贪心”地抓住偶尔出现的微小收益。为其引入“内在奖励”，可以鼓励智能体去主动访问未知的状态。

核心机制：

- ICM (Intrinsic Curiosity Module)：通过预测下一个状态的特征，将预测误差作为奖励。预测越不准，说明状态越新奇，内在奖励越高。

- RND (Random Network Distillation)：利用两个网络（固定随机网络和预测网络）处理同一状态，将两者的输出均方误差作为内在奖励，非常适合高维空间。

支持论文：

- Curiosity-driven Exploration by Self-supervised Prediction (Pathak et al., 2017) —— ICM的提出论文。

- Exploration by Random Network Distillation (Burda et al., 2018) —— RND的提出论文。


## 方法3：最大熵强化学习框架 (Maximum Entropy RL)
将最大化奖励和最大化策略熵结合为统一目标。虽然这是Soft Actor-Critic (SAC) 的核心思想，但目前已有大量研究将其思想融入PPO的变体中（如Soft-PPO），确保在整个训练生命周期内维持动态的探索性，而不是仅仅依赖静态的熵系数。

支持论文：
- Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (Haarnoja et al., 2018)。理解这篇论文从零构建最大熵框架的逻辑，对于改进PPO的贪心缺陷具有极高的参考价值。
