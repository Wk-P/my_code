"""
用 PPO / DQN 训练 service placement 环境
• 5 个数量级：N ∈ { 10¹, 10², 10³, 10⁴, 10⁵ } ECU
• 自动检测 CUDA，支持 GPU 加速（device="auto"）
• 输出 3 张对比图（箱线图 / 训练曲线 / 最终AR对比）
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")                        # 无界面环境下也能保存图片
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from pathlib import Path

# 支持中文标签
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 把 env 目录加入 path ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))
from env import my_env, my_ecu, my_service

# ══════════════════════════════════════════════════════════════
#  ★ 配置区 —— 只改这里
# ══════════════════════════════════════════════════════════════

SCALES       = [10, 100, 1_000, 10_000, 100_000]  # N = ECU 数量（5 个数量级）
M_FIXED      = 100          # 每 episode 分配的服务数（固定，保证各规模可比较）
VMS_RANGE    = (10, 101)    # ECU VM 容量范围 [10, 100]（整数均匀抽样）
REQ_RANGE    = (1,  10)     # 服务 VM 需求范围 [1, 9]；max=9 < VMS min=10 → 永不超容
ALGO         = "PPO"        # "PPO" 或 "DQN"
TOTAL_STEPS  = 100_000      # 每个规模的训练总步数（100k/100steps ≈ 1000 episodes）
PPO_MAX_N    = 10_000       # N > 此值仅评估随机基线（action 空间过大，训练开销指数级增长）
EVAL_EPS     = 300          # 随机基线评估的 episode 数
DEVICE       = "auto"       # "auto"=自动选 CUDA/CPU | "cuda" | "cpu"
SEED         = 42
SMOOTH_W     = 30           # 训练曲线滑动平均窗口大小

OUTDIR = Path(__file__).parent / "results"
os.makedirs(OUTDIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  设备检测
# ══════════════════════════════════════════════════════════════

def resolve_device(cfg: str) -> str:
    """
    "auto" → 优先 CUDA（GPU），否则 CPU。
    PPO/DQN 策略网络的前向传播 / 反向传播均在此设备上运算。
    对于大 N（N=10000）：GPU 能显著加速最后一层 (128 → N) 的矩阵乘法。
    对于小 N（N=10~100）：CPU 反而更快（GPU数据传输开销 > 计算收益）。
    """
    if cfg != "auto":
        return cfg
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[CUDA ✓] {name}  显存 {mem:.1f} GB")
        return "cuda"
    print("[CPU  ] 未检测到 CUDA GPU，使用 CPU 训练")
    return "cpu"


# ══════════════════════════════════════════════════════════════
#  环境工厂
# ══════════════════════════════════════════════════════════════

def make_env(N: int, seed: int = 0) -> Monitor:
    """
    创建带 Monitor 包装的环境。
    Monitor 会自动记录每个 episode 的总 reward / 长度 / 时间，
    并注入 info["episode"] 供回调读取。
    因为 env 只在最后一步给 reward = AR（其余为 0），
    所以 episode_reward = final_AR，可直接作为学习曲线指标。
    """
    rng      = np.random.default_rng(seed)
    ecus     = [my_ecu(int(rng.integers(*VMS_RANGE))) for _ in range(N)]
    services = [my_service(i, int(rng.integers(*REQ_RANGE))) for i in range(M_FIXED)]
    return Monitor(my_env(ecus, services))


# ══════════════════════════════════════════════════════════════
#  回调：收集每 episode 的 final AR
# ══════════════════════════════════════════════════════════════

class ARCallback(BaseCallback):
    """
    PPO/DQN 在每个 step 结束时调用 _on_step()。
    当 Monitor 检测到 done=True 时，会在 info["episode"] 中写入
        {"r": total_reward, "l": episode_length, "t": wall_time}
    我们只需从这里读取 AR 即可。
    """

    def __init__(self):
        super().__init__()
        self.episode_ars:     list[float] = []
        self.timesteps_at_ep: list[int]   = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ar = float(info["episode"]["r"])       # = final AR
                self.episode_ars.append(ar)
                self.timesteps_at_ep.append(self.num_timesteps)
        return True  # 返回 False 会提前停止训练


# ══════════════════════════════════════════════════════════════
#  随机基线评估
# ══════════════════════════════════════════════════════════════

def eval_random(N: int, n_eps: int = EVAL_EPS) -> np.ndarray:
    """对给定规模 N，运行 n_eps 个完整 episode，记录每次的 final AR。"""
    env = make_env(N, seed=SEED)
    ars = []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done   = False
        ep_r   = 0.0
        while not done:
            obs, r, done, _, _ = env.step(env.action_space.sample())
            ep_r += r
        ars.append(ep_r)
    env.close()
    return np.array(ars, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
#  建立模型
# ══════════════════════════════════════════════════════════════

def build_model(algo: str, env: Monitor, device: str, N: int):
    """
    PPO：on-policy，策略梯度。适合连续控制和小规模离散动作空间。
    DQN：off-policy，Q-learning + replay buffer。对离散动作更样本高效。
    两者的最后一层均为 Linear(128, N)，N 很大时 CUDA 加速效果最明显。
    """
    if algo == "PPO":
        return PPO(
            "MlpPolicy", env,
            learning_rate = 3e-4,
            n_steps       = 512,        # 每次 rollout 收集 512 步再更新
            batch_size    = 64,
            n_epochs      = 10,         # 每批数据复用 10 次
            gamma         = 0.99,       # 折扣因子（episode 末才给 reward，需接近 1）
            policy_kwargs = dict(net_arch=[128, 128]),
            device        = device,
            verbose       = 0,
            seed          = SEED,
        )
    else:  # DQN
        return DQN(
            "MlpPolicy", env,
            learning_rate          = 1e-4,
            buffer_size            = 50_000,   # 经验回放池大小
            learning_starts        = 2_000,    # 先收集 2000 步再开始学
            batch_size             = 64,
            gamma                  = 0.99,
            train_freq             = 4,        # 每 4 步更新一次 Q 网络
            target_update_interval = 500,      # 每 500 步同步 target 网络
            exploration_fraction   = 0.3,      # 前 30% 步线性衰减 epsilon
            policy_kwargs          = dict(net_arch=[128, 128]),
            device                 = device,
            verbose                = 0,
            seed                   = SEED,
        )


# ══════════════════════════════════════════════════════════════
#  训练
# ══════════════════════════════════════════════════════════════

def train_agent(N: int, device: str) -> "ARCallback | None":
    if N > PPO_MAX_N:
        print(f"  ⚠  N={N:,}  超过 PPO_MAX_N={PPO_MAX_N:,}，跳过训练（只做随机基线）")
        return None

    env   = make_env(N, seed=SEED)
    cb    = ARCallback()
    model = build_model(ALGO, env, device, N)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb, progress_bar=False)
    elapsed = time.time() - t0

    n_ep    = len(cb.episode_ars)
    last_ar = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    rand_ar = np.mean(eval_random(N, n_eps=100))
    gain    = last_ar - rand_ar

    print(f"  耗时 {elapsed:.1f}s  |  episodes={n_ep}  "
          f"|  末50ep AR={last_ar:.4f}  |  vs 随机 {rand_ar:.4f}  "
          f"|  提升 {gain:+.4f}")
    env.close()
    return cb


# ══════════════════════════════════════════════════════════════
#  工具：滑动平均
# ══════════════════════════════════════════════════════════════

def moving_avg(arr: list | np.ndarray, w: int = SMOOTH_W):
    """返回长度为 len(arr)-w+1 的平滑数组，以及对应的起始偏移量。"""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr, 0
    sm = np.convolve(arr, np.ones(w) / w, mode="valid")
    return sm, w - 1           # sm[i] 对应原始 arr[i .. i+w-1] 的均值


# ══════════════════════════════════════════════════════════════
#  绘图（3 子图）
# ══════════════════════════════════════════════════════════════

def plot_results(results: dict, device: str):
    COLORS  = plt.cm.viridis(np.linspace(0.1, 0.85, len(SCALES)))
    XLABELS = [f"$10^{{{i+1}}}$" for i in range(len(SCALES))]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── ① 随机基线 AR 箱线图 ─────────────────────────────────────
    ax = axes[0]
    bp = ax.boxplot(
        [results[N]["rand"] for N in SCALES],
        labels=XLABELS,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.55),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=3, alpha=0.3),
    )
    # 在中位线上方标注均值
    for i, N in enumerate(SCALES):
        mean_val = float(np.mean(results[N]["rand"]))
        ax.text(i + 1, mean_val + 0.03, f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="navy")
    ax.set_title("① 随机策略 AR 分布（各规模）", fontsize=12)
    ax.set_xlabel("ECU 数量级（N）")
    ax.set_ylabel("AR（资源利用率）")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # ── ② PPO/DQN 训练曲线 ───────────────────────────────────────
    ax = axes[1]
    has_curve = False
    for i, N in enumerate(SCALES):
        cb = results[N]["cb"]
        if cb is None or len(cb.episode_ars) < SMOOTH_W + 5:
            ax.axhline(np.mean(results[N]["rand"]),
                       color=COLORS[i], linestyle="--", linewidth=1,
                       alpha=0.5, label=f"N={N:,}（仅随机基线）")
            continue
        has_curve = True
        sm, offset = moving_avg(cb.episode_ars, SMOOTH_W)
        ts = np.array(cb.timesteps_at_ep)[offset: offset + len(sm)]
        ax.plot(ts, sm, color=COLORS[i], label=f"N={N:,}", linewidth=2)

    ax.set_title(f"② {ALGO} 训练曲线（滑动均值 w={SMOOTH_W}）", fontsize=12)
    ax.set_xlabel("训练步数（总）")
    ax.set_ylabel("AR（episode reward）")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else str(int(x))))
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)

    # ── ③ 最终 AR 对比柱状图 ─────────────────────────────────────
    ax     = axes[2]
    x      = np.arange(len(SCALES))
    bw     = 0.32
    rand_m = [float(np.mean(results[N]["rand"])) for N in SCALES]
    ppo_m  = []
    for N in SCALES:
        cb = results[N]["cb"]
        if cb and len(cb.episode_ars) >= 50:
            ppo_m.append(float(np.mean(cb.episode_ars[-50:])))
        else:
            ppo_m.append(float("nan"))

    bars_r = ax.bar(x - bw / 2, rand_m, bw, label="随机基线",
                    color="steelblue", alpha=0.75)
    bars_p = ax.bar(x + bw / 2, ppo_m,  bw, label=ALGO,
                    color="coral",     alpha=0.85)

    # 在柱子上标注数值
    for bar, v in zip(bars_r, rand_m):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, color="steelblue")
    for bar, v in zip(bars_p, ppo_m):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, color="tomato")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "N/A",
                    ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS)
    ax.set_xlabel("ECU 数量级（N）")
    ax.set_ylabel("平均 AR（末 50 episodes）")
    ax.set_title(f"③ 随机 vs {ALGO}  最终 AR 对比", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 总标题
    fig.suptitle(
        f"Service Placement RL  ·  {ALGO}  ·  Device={device.upper()}"
        f"  ·  M={M_FIXED} services/episode  ·  {TOTAL_STEPS:,} steps/scale",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()

    path = os.path.join(OUTDIR, "training_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  图表已保存 → {path}")


# ══════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════

def main():
    device = resolve_device(DEVICE)
    print(f"\n算法={ALGO}  设备={device.upper()}  M={M_FIXED}  TOTAL_STEPS={TOTAL_STEPS:,}")
    print(f"ECU 容量范围={VMS_RANGE}  服务需求范围={REQ_RANGE}")
    print(f"{'═' * 60}\n")

    results = {}
    for i, N in enumerate(SCALES):
        print(f"[{i+1}/{len(SCALES)}]  N={N:,} ECU  M={M_FIXED} services")

        # ① 随机基线
        rand_ars = eval_random(N)
        print(f"  随机基线  mean={np.mean(rand_ars):.4f}  "
              f"std={np.std(rand_ars):.4f}  max={np.max(rand_ars):.4f}")

        # ② 训练 PPO / DQN
        cb = train_agent(N, device)
        results[N] = {"rand": rand_ars, "cb": cb}
        print()

    # ③ 绘图
    plot_results(results, device)
    print("全部完成！")


if __name__ == "__main__":
    main()
