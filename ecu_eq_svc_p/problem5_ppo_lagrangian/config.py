"""
Configuration for Problem 5: Lagrangian Constraint Relaxation.

P5 = PPO with adaptive Lagrangian multiplier λ.
    - No action masking; constraints are penalised, not blocked.
    - Episode always runs M steps; remaining_vms can go negative.
    - Reward per step: n_i / e_j - λ * c_t
    - λ is exposed in the observation (normalised by LAMBDA_MAX).
    - λ updated via dual ascent every LAMBDA_UPDATE_WINDOW episodes:
                λ ← clip(λ + LAMBDA_LR * (avg_viol_rate - LAMBDA_TARGET), 0, LAMBDA_MAX)
"""

from pathlib import Path
import sys
import os

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from training_steps_config import get_total_steps

# ── Scenario ──────────────────────────────────────────────────────────────────
YAML_CONFIG  = ROOT / ".." / "problem2_ilp" / "config" / "config_ecu_eq_svc.yaml"
SCENARIO_IDX = 0

with open(YAML_CONFIG) as f:
    import yaml
    cfg = yaml.safe_load(f)
    _all = cfg["scenarios"]
    N = len(_all[0]["ECUs"])
    M = len(_all[0]["SVCs"])
    SCENARIOS = [
        (
            [ecu["capacity"] for ecu in sc["ECUs"]],
            [svc["requirement"] for svc in sc["SVCs"]],
            sc.get("conflict_sets", []),
        )
        for sc in _all
    ]
    VMS_POOL = SCENARIOS[SCENARIO_IDX][0]
    REQ_POOL = SCENARIOS[SCENARIO_IDX][1]

# ── Training ──────────────────────────────────────────────────────────────────
TOTAL_STEPS = get_total_steps("problem5_ppo_lagrangian")
SEED        = int(os.environ.get("TRAIN_SEED", "42"))
# ── Train / Test split (80/20, deterministic) ────────────────────────────────
import random as _random
_rng = _random.Random(SEED)
_idxs = list(range(len(SCENARIOS)))
_rng.shuffle(_idxs)
_n_train = int(0.8 * len(SCENARIOS))
TRAIN_SCENARIOS = [SCENARIOS[i] for i in _idxs[:_n_train]]
TEST_SCENARIOS  = [SCENARIOS[i] for i in _idxs[_n_train:]]

DEVICE      = "cpu"
N_ENVS      = 40
TORCH_NUM_THREADS = 16       # 28 CPUs available; 16 threads for better BLAS throughput
PROGRESS_LOG_EVERY_STEPS = 200_000

# ── PPO hyperparameters ───────────────────────────────────────────────────────
PPO_LR         = 3e-4
PPO_N_STEPS     = 512    # collect multiple episodes per rollout to amortise SB3 overhead
PPO_BATCH_SIZE  = 256
PPO_N_EPOCHS   = 10
PPO_GAMMA      = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF   = 0.005   # entropy regularisation prevents premature convergence
PPO_NET_ARCH   = dict(pi=[256, 256], vf=[512, 512])

# ── Lagrangian multiplier (dual variable) ─────────────────────────────────────
LAMBDA_INIT          = 0.5    # meaningful initial penalty from the start
LAMBDA_LR            = 0.01   # larger step (window is also larger, signal is less noisy)
LAMBDA_TARGET        = 0.0    # zero conflict-violation objective
LAMBDA_MAX           = 5.0
LAMBDA_UPDATE_WINDOW = 200    # update λ every 200 episodes (≈5 eps/env with 40 envs)
LAMBDA_WARMUP_EPISODES = 0

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_EPS = len(TEST_SCENARIOS)
SMOOTH_W = 1000

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTDIR     = ROOT / "results"
MODEL_PATH = OUTDIR / "lagrange_ppo_model"
