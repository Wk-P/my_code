<script setup>
import { ref, onMounted } from "vue";
import { getResults } from "../api.js";

const rows = ref([]);
const loading = ref(true);

const SCENARIO_ORDER = { lt: 0, eq: 1, gt: 2 };

onMounted(async () => {
  const all = await getResults("add_states");
  rows.value = all
    .filter((r) => r.algo === "ppo_mask")
    .sort((a, b) => (SCENARIO_ORDER[a.scenario] ?? 9) - (SCENARIO_ORDER[b.scenario] ?? 9));
  loading.value = false;
});

function fmtPct(x) {
  return x === null || x === undefined ? "—" : (x * 100).toFixed(1) + "%";
}
function fmt4(x) {
  return x === null || x === undefined ? "—" : Number(x).toFixed(4);
}
function fmtGap(ilp, rl) {
  if (ilp === null || ilp === undefined || rl === null || rl === undefined) return "—";
  return (ilp - rl).toFixed(4);
}
</script>

<template>
  <a class="thumb-link" href="#/">&larr; back to dashboard</a>

  <article class="paper">
    <header class="paper-header">
      <div class="paper-kicker">工作记录 · add_states 分支</div>
      <h1 class="paper-title">面向服务-ECU部署问题的强化学习：AR质量导向的Reward设计与能力边界</h1>
      <div class="paper-meta">2026年8月 · MaskablePPO (P4) · lt / eq / gt 三场景</div>
    </header>

    <section class="paper-abstract">
      <h2 class="paper-abstract-label">摘要</h2>
      <p>
        本文记录了在服务-ECU部署问题（ILP 建模，RL 用 action masking 保证零违反）上，
        围绕 <strong>success_rate 低下</strong>与 <strong>AR（资源利用率）逼近 ILP 最优解</strong>
        两个目标展开的一系列诊断与实验。核心发现有三点：
        (1) BC 预训练的作用因场景资源紧张程度而分化——对资源紧张场景（lt）有害，对资源宽松场景（eq/gt）有益；
        (2) eq/gt 场景此前从未把 AR 接入训练目标，移植 AR-proportional reward 后 AR gap 从
        0.03&nbsp;~&nbsp;0.17 压缩到 0.004&nbsp;~&nbsp;0.05，是目前发现的影响最大的单一改动；
        (3) 针对 lt 场景 success_rate 卡在 80%&nbsp;~&nbsp;90% 的问题，连续 5 轮独立的 reward-engineering
        尝试（势函数 shaping、权重再平衡、熵退火、梯度分级失败惩罚、可行性 shaping）均为零效应，
        表明这是当前"在线、单步、不可回溯"决策结构在纯 reward 学习框架下的真实能力边界，而非
        reward 设计未调优。
      </p>
    </section>

    <nav class="paper-toc">
      <strong>目录</strong>
      <ol>
        <li><a href="#sec-bg">问题背景</a></li>
        <li><a href="#sec-method">方法</a></li>
        <li><a href="#sec-exp1">实验一：BC 预训练的场景分化效应</a></li>
        <li><a href="#sec-exp2">实验二：Reward-Engineering 五轮消融</a></li>
        <li><a href="#sec-results">最终结果</a></li>
        <li><a href="#sec-discussion">讨论：能力边界与 Safe RL 的分野</a></li>
        <li><a href="#sec-conclusion">结论与展望</a></li>
      </ol>
    </nav>

    <section id="sec-bg">
      <h2>1. 问题背景</h2>
      <p>
        问题是把 <em>M</em> 个服务分配到 <em>N</em> 个 ECU 上，每个 ECU 有容量约束，服务之间有
        冲突集约束（同一冲突集内的服务不能共享 ECU）。三个场景对应资源供需的三种关系：
        <strong>lt</strong>（N&lt;M，资源紧张，ECU 少于服务）、<strong>eq</strong>（N=M，供需平衡）、
        <strong>gt</strong>（N&gt;M，资源宽松）。ILP（Dinkelbach 参数化子问题）给出全局最优解，
        作为 RL 逼近的上界。核心指标 AR（Average Resource Utilization）定义为已用 ECU 上
        "需求/容量"占比之和除以已用 ECU 数。
      </p>
      <p>
        RL 侧用 MaskablePPO（记为 P4），通过 action masking 让容量与冲突约束成为
        <strong>硬约束</strong>——非法动作直接从动作空间中排除，训练与评估全程违反率恒为 0。
        这使得"能不能把 M 个服务全部合法放完"（success_rate）和"放完之后利用率有多高"（AR）
        成为两个独立可测的目标。
      </p>
    </section>

    <section id="sec-method">
      <h2>2. 方法</h2>
      <h3>2.1 Reward 设计的演变</h3>
      <p>
        reward 设计经历了三个阶段：<strong>v1.1.0</strong> 纯终局稀疏奖励（成功 <code>+M</code>／
        违反 <code>-M</code>，不区分 AR 高低）→ <strong>v2.2.0</strong> AR 正式成为训练目标
        （成功 <code>M·AR</code>／违反 <code>-M</code>，lt 场景率先落地）→
        <strong>v2.4.0/v2.5.0</strong>（本文档主体）AR/违规权重对等（<code>M·(2·AR-1)</code>）+
        失败按 <code>valid_placed/M</code> 梯度分级，替代此前"任何失败都是同一个 <code>-M</code>"
        的 0/1-loss 式设计。
      </p>
      <h3>2.2 State 设计：bottleneck_risk</h3>
      <p>
        观测新增一维聚合特征 <code>bottleneck_risk</code>——剩余服务里
        <code>1/(valid_ecu_count+1)</code> 的均值，是一个连续的"死胡同临近"信号，让 policy
        不必再从逐服务的原始合法 ECU 计数里隐式推断风险。
      </p>
    </section>

    <section id="sec-exp1">
      <h2>3. 实验一：BC 预训练的场景分化效应</h2>
      <p>
        隔离实验（同样 5M 步，仅切换 BC 开关）显示：<strong>lt</strong> 场景去掉 BC 后
        success_rate 从 47.5% 涨到 70.0%（+22.5pp，AR 基本不变），是目前发现的最大单一负面因素；
        但同一实验搬到 <strong>eq/gt</strong>（success_rate 本就 100% 封顶）后，去掉 BC 反而让
        AR gap 从约 0.03 / 0.14 扩大到 0.076 / 0.173——BC 在资源宽松场景里其实是净正面的。
      </p>
      <table class="paper-table">
        <thead><tr><th>场景</th><th>配置</th><th>success_rate</th><th>AR gap</th></tr></thead>
        <tbody>
          <tr><td rowspan="2">lt</td><td>有 BC（原基线）</td><td>47.5%</td><td>0.089</td></tr>
          <tr><td>无 BC</td><td>70.0%</td><td>0.087</td></tr>
          <tr><td rowspan="2">eq</td><td>有 BC</td><td>100%</td><td>≈0.03</td></tr>
          <tr><td>无 BC</td><td>100%</td><td>0.076</td></tr>
          <tr><td rowspan="2">gt</td><td>有 BC</td><td>100%</td><td>≈0.14</td></tr>
          <tr><td>无 BC</td><td>100%</td><td>0.173</td></tr>
        </tbody>
      </table>
      <p class="paper-note">结论：BC 策略按场景资源紧张程度分化——lt 停用，eq/gt 保留（v2.3.0）。</p>
    </section>

    <section id="sec-exp2">
      <h2>4. 实验二：Reward-Engineering 五轮消融（lt 场景）</h2>
      <p>
        围绕 lt 场景 success_rate/AR 卡在一个区间不动的问题，连续做了 5 轮独立、有理论依据的
        reward 改动尝试，每轮均为 8&nbsp;~&nbsp;10 个随机种子的多 seed 验证（而非单次实验）。
      </p>
      <table class="paper-table">
        <thead><tr><th>#</th><th>改动</th><th>success_rate</th><th>AR mean</th><th>结论</th></tr></thead>
        <tbody>
          <tr><td>0</td><td>基线：M·AR reward，无熵退火</td><td>86.50% ± 5.68</td><td>0.6207 ± 0.0074</td><td>—</td></tr>
          <tr><td>1</td><td>bottleneck_risk 势函数 shaping（离散→连续，多组 β）</td><td>85~87%</td><td>0.6207~0.6208</td><td>零效应</td></tr>
          <tr><td>2</td><td>reward 拉伸为 M·(2·AR-1) + 熵系数退火 0.02→0.002</td><td>87.19% ± 6.19</td><td>0.6276 ± 0.0100</td><td>零效应</td></tr>
          <tr><td>3</td><td>失败惩罚按 valid_placed/M 梯度分级</td><td>85.62% ± 5.13</td><td>0.6210 ± 0.0132</td><td>零效应</td></tr>
          <tr><td>4</td><td>shaping 势函数换成 FFD 贪心可行性模拟</td><td>85.50% ± 4.53</td><td>0.6208 ± 0.0134</td><td>零效应</td></tr>
        </tbody>
      </table>
      <p class="paper-note">
        五种设计思路、信号强度跨度很大的机制，全部收敛到同一个区间（success_rate 80~90%，
        AR 0.60~0.65），差异均落在种子间噪声范围内（约 5~6pp）。
      </p>
    </section>

    <section id="sec-results">
      <h2>5. 最终结果</h2>
      <p>
        在当前分支状态（本文档记录的所有改动均已合入）下，对 lt/eq/gt 各完整跑一次标准训练
        （lt 5,000,000 步，eq/gt 各 2,000,000 步，均不使用 BC）。下表直接从
        <code>results/add_states/&lt;scenario&gt;/ppo_mask/</code> 最新一次运行结果实时读取。
      </p>
      <div v-if="loading" class="empty">加载中…</div>
      <table v-else class="paper-table">
        <thead>
          <tr><th>场景</th><th>N</th><th>M</th><th>ILP AR</th><th>RL AR</th><th>AR gap</th><th>success_rate</th><th>训练步数</th></tr>
        </thead>
        <tbody>
          <tr v-for="r in rows" :key="r.scenario">
            <td>{{ r.scenario }}</td>
            <td>{{ r.N }}</td>
            <td>{{ r.M }}</td>
            <td>{{ fmt4(r.ilp_ar) }}</td>
            <td>{{ fmt4(r.test_ar_mean) }}</td>
            <td>{{ fmtGap(r.ilp_ar, r.test_ar_mean) }}</td>
            <td>{{ fmtPct(r.test_success_rate) }}</td>
            <td>{{ r.train_steps?.toLocaleString() ?? "—" }}</td>
          </tr>
          <tr v-if="!rows.length"><td colspan="8" class="empty">暂无数据 —— 训练可能仍在进行中，完成后刷新本页即可看到最新结果。</td></tr>
        </tbody>
      </table>
    </section>

    <section id="sec-discussion">
      <h2>6. 讨论：能力边界与 Safe RL 的分野</h2>
      <p>
        capacity/conflict 约束能做成硬约束（masking），是因为它们是<strong>逐步、局部可验证</strong>
        的属性；而"能否把全部服务放完"（success）是<strong>关于整条轨迹的全局属性</strong>，
        无法用一步之内的判断来保证。理论上可以把 masking 扩展成"前瞻式可行性检查"（每一步验证
        剩余子问题是否还有解），让 success 也变成硬约束，但这样做的代价是
        <strong>success_rate 不再是"RL 学到了什么"的度量，而是"环境内嵌的求解器有多强"的度量</strong>
        ——与此前讨论并否决的"ILP 兜底"方案是同一类问题，只是换了个隐藏位置。
      </p>
      <p>
        用求解器信号做 <em>reward shaping</em>（而非 masking）符合"RL 自主学习"的原则——solver
        只在训练时提供更精确的教学信号，policy 仍需自己学会如何响应，部署时不依赖 solver。
        实验二的第 4 轮（FFD 可行性 shaping）正是这一原则下的实验，但结果依然是零效应，
        进一步支持"这是决策结构本身的能力边界"这一结论。
      </p>
    </section>

    <section id="sec-conclusion">
      <h2>7. 结论与展望</h2>
      <p>
        在"RL 完全自主通过 reward 学习"的框架内，lt 场景 80%&nbsp;~&nbsp;90% 的 success_rate 与
        0.60&nbsp;~&nbsp;0.65 的 AR，是当前在线、单步、不可回溯决策结构的真实能力边界，不是
        reward 设计未调优——这一结论建立在 5 轮独立多 seed 实验、而非单次结果之上。
        eq/gt 场景通过补齐 AR-reward 已经把 AR gap 压缩到 0.004&nbsp;~&nbsp;0.05，
        与 ILP 最优解已经非常接近。
      </p>
      <p>
        后续值得探索的方向包括：GRPO 风格的组内相对优势训练（同场景多次采样，组内比较而非跨场景平均）、
        curriculum learning（对高 demand/cap 比场景过采样）、以及训练时引入 lookahead/搜索
        （类 AlphaZero，搜索结果只用于生成训练目标，不在部署时依赖）。
      </p>
    </section>

    <footer class="paper-footer">
      详细版本记录见 <a href="#/versions">Version History</a>
      （<a href="#/versions/v2.3.0">v2.3.0</a>、
      <a href="#/versions/v2.4.0">v2.4.0</a>、
      <a href="#/versions/v2.5.0">v2.5.0</a>）。
    </footer>
  </article>
</template>
