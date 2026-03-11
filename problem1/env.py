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
    Each episode runs M steps (each step assigns a service to an ECU).
    Intermediate steps give reward = 0; the final step gives reward = final AR.
    Goal: find a policy that maximizes AR over multiple episodes.
    """

    def __init__(self, ecus: list[my_ecu], services: list[my_service]):
        self.ecus     = ecus
        self.services = services
        self.M = len(services)
        self.N = len(ecus)

        # Action space: which ECU to select (0 ~ N-1)
        self.action_space = gym.spaces.Discrete(self.N)

        # Observation space extension: [current_service_demand, current_AR, ECU0_remaining%, ECU1_remaining%, ..., ECU(N-1)_remaining%]
        # Dimension: N+2 (1 service demand + 1 AR + N ECU remaining capacities)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.N + 2,), dtype=np.float32
        )

        # Save initial capacities for normalization and remaining capacity tracking
        self.initial_vms = np.array([e.vms for e in ecus], dtype=np.float32)
        self.remaining_vms = self.initial_vms.copy()
        
        self.ar = 0.0
        self._current_step = 0

    # -----------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.remaining_vms = self.initial_vms.copy()
        self.ar = 0.0
        self._current_step = 0
        return self._get_obs(), {}
    
    # -----------------------------------------------------------------
    def _get_obs(self):
        """Core improvement: return complete state information"""
        if self._current_step >= self.M:
            # Episode finished, set service demand to 0
            service_demand_norm = 0.0
        else:
            service = self.services[self._current_step]
            # Normalize: current service demand / max ECU initial capacity
            service_demand_norm = service.required_vms / np.max(self.initial_vms)
        
        # Remaining capacity percentage for each ECU (relative to initial capacity)
        remaining_pct = self.remaining_vms / (self.initial_vms + 1e-8)
        
        obs = np.concatenate([
            [service_demand_norm],  # Dimension 0: current service demand (normalized to [0,1])
            [self.ar],              # Dimension 1: current cumulative AR
            remaining_pct           # Dimension 2~N+1: remaining capacity percentage for each ECU
        ]).astype(np.float32)
        
        return obs

    # -----------------------------------------------------------------
    def step(self, action: int):
        service = self.services[self._current_step]
        ecu_remaining = self.remaining_vms[action]  # Use remaining capacity instead of initial capacity

        if ecu_remaining >= service.required_vms:
            # Successful placement
            ru = service.required_vms / self.initial_vms[action]  # RU calculated based on initial capacity
            
            # Update remaining capacity (key: let agent see dynamic changes)
            self.remaining_vms[action] -= service.required_vms
            
            # Incremental update for AR
            self.ar = (self.ar * self._current_step + ru) / (self._current_step + 1)
            self._current_step += 1

            last_step = (self._current_step >= self.M)
            # Original reward mechanism: only give AR at the last step (maintain backward compatibility)
            reward = float(self.ar) if last_step else 0.0
            done   = last_step
        else:
            # Insufficient capacity -> penalty and terminate
            self._current_step += 1
            reward = -1.0
            done   = True

        obs = self._get_obs()
        return obs, reward, done, False, {"ar": self.ar, "step": self._current_step}

    # -----------------------------------------------------------------
    def render(self):
        if self._current_step < self.M:
            svc = self.services[self._current_step]
            print(f"  Step {self._current_step}/{self.M} | Current service needs {svc.required_vms} VMs | AR = {self.ar:.4f}")
        else:
            print(f"  Episode finished | AR = {self.ar:.4f}")


# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)

    N, M = 5, 8                                         # 5 ECUs, 8 services
    ecus     = [my_ecu(np.random.randint(3, 10)) for _ in range(N)]
    services = [my_service(i, np.random.randint(1, 6)) for i in range(M)]

    print("ECU capacity :", [e.vms for e in ecus])
    print("Service demand:", [s.required_vms for s in services])
    print()

    env = my_env(ecus, services)

    # Test observation space dimension
    obs, _ = env.reset()
    print(f"✓ Observation shape: {obs.shape}  (expected: {N+2})")
    print(f"✓ Observation: {obs}")
    print(f"  [0]=service_demand_norm={obs[0]:.3f}, [1]=AR={obs[1]:.3f}, [2:]=ECU_remaining_pct\n")

    # Run multiple episodes and record the final AR of each
    n_episodes  = 20
    ar_history  = []
    best_ar     = 0.0
    best_actions = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_actions = []

        while not done:
            action = env.action_space.sample()      # random action for now
            ep_actions.append(action)
            obs, reward, done, _, info = env.step(action)

        final_ar = info["ar"]
        ar_history.append(final_ar)

        if final_ar > best_ar:
            best_ar      = final_ar
            best_actions = ep_actions[:]

        print(f"  ep {ep+1:3d} | actions={ep_actions} | AR={final_ar:.4f}"
              + (" <- best" if final_ar == best_ar else ""))

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean AR : {np.mean(ar_history):.4f}")
    print(f"  Best AR : {best_ar:.4f}  actions={best_actions}")
