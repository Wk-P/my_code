import gymnasium as gym
import numpy as np


class my_ecu:
    def __init__(self, vms):
        self.vms = vms


class my_service:
    def __init__(self, index, required_vms):
        self.index = index
        self.required_vms = required_vms


class my_env(gym.Env):
    """
    每个 episode 走 M 步（每步给服务选一个ECU）。
    中间步骤 reward = 0，最后一步 reward = 最终AR。
    目标：通过多次 episode 找到让 AR 最高的策略。
    """

    def __init__(self, ecus: list, services: list):
        self.ecus     = ecus
        self.services = services
        self.M = len(services)
        self.N = len(ecus)

        # 动作空间：选哪个 ECU（0 ~ N-1）
        self.action_space = gym.spaces.Discrete(self.N)

        # 观测空间：只有当前 AR，1维 [0, 1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.ar = 0.0
        self._current_step = 0

    # -----------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ar = 0.0
        self._current_step = 0
        return np.array([self.ar], dtype=np.float32), {}

    # -----------------------------------------------------------------
    def step(self, action: int):
        service = self.services[self._current_step]
        ecu     = self.ecus[action]

        if ecu.vms >= service.required_vms:
            ru = service.required_vms / ecu.vms
            # 增量均值更新 AR
            self.ar = (self.ar * self._current_step + ru) / (self._current_step + 1)
            self._current_step += 1

            last_step = (self._current_step >= self.M)
            # ★ 只在最后一步给 reward = 最终AR，中间步骤全为 0
            reward = float(self.ar) if last_step else 0.0
            done   = last_step
        else:
            # 容量不足 → 惩罚 -1，提前结束
            self._current_step += 1
            reward = -1.0
            done   = True

        obs = np.array([self.ar], dtype=np.float32)
        return obs, reward, done, False, {"ar": self.ar, "step": self._current_step}

    # -----------------------------------------------------------------
    def render(self):
        print(f"  Step {self._current_step}/{self.M} | AR = {self.ar:.4f}")


# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)

    N, M = 5, 8                                         # 5个ECU，8个服务
    ecus     = [my_ecu(np.random.randint(3, 10)) for _ in range(N)]
    services = [my_service(i, np.random.randint(1, 6)) for i in range(M)]

    print("ECU 容量  :", [e.vms for e in ecus])
    print("服务需求  :", [s.required_vms for s in services])
    print()

    env = my_env(ecus, services)

    # 跑多个 episode，记录每次的最终 AR
    n_episodes  = 20
    ar_history  = []
    best_ar     = 0.0
    best_actions = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_actions = []

        while not done:
            action = env.action_space.sample()      # 目前随机选
            ep_actions.append(action)
            obs, reward, done, _, info = env.step(action)

        final_ar = info["ar"]
        ar_history.append(final_ar)

        if final_ar > best_ar:
            best_ar      = final_ar
            best_actions = ep_actions[:]

        print(f"  ep {ep+1:3d} | actions={ep_actions} | AR={final_ar:.4f}"
              + (" ← best" if final_ar == best_ar else ""))

    print(f"\n{n_episodes} 个 episode 结果：")
    print(f"  平均 AR : {np.mean(ar_history):.4f}")
    print(f"  最高 AR : {best_ar:.4f}  actions={best_actions}")
