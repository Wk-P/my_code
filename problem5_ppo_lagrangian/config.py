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

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from training_steps_config import get_total_steps

# ── Scenario ──────────────────────────────────────────────────────────────────
YAML_CONFIG  = ROOT / ".." / "problem2_ilp" / "config" / "config_20260305_183222.yaml"
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
        )
        for sc in _all
    ]
    VMS_POOL = SCENARIOS[SCENARIO_IDX][0]
    REQ_POOL = SCENARIOS[SCENARIO_IDX][1]

# ── Training ──────────────────────────────────────────────────────────────────
TOTAL_STEPS = get_total_steps("problem5_ppo_lagrangian")
SEED        = 42
DEVICE      = "cpu"
N_ENVS      = 40
SUBPROC_START_METHOD = "fork"
TORCH_NUM_THREADS = 4        # leave cores for SubprocVecEnv workers
PROGRESS_LOG_EVERY_STEPS = 200_000

# ── PPO hyperparameters ───────────────────────────────────────────────────────
PPO_LR         = 3e-4
PPO_N_STEPS    = 128          # shorter rollout → more frequent updates (suits 10-step episodes)
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS   = 10
PPO_GAMMA      = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_NET_ARCH   = dict(pi=[256, 256], vf=[512, 512])  # larger network for 43-dim obs

# ── Lagrangian multiplier (dual variable) ─────────────────────────────────────
LAMBDA_INIT          = 0.0    # initial λ value
LAMBDA_LR            = 0.0003   # slower dual-ascent to avoid AR collapse after warmup
LAMBDA_TARGET        = 0.0    # zero-violation objective
LAMBDA_MAX           = 2.0    # keep penalty scale comparable to per-step utilisation gain
LAMBDA_UPDATE_WINDOW = 20     # update λ every 20 episodes after warmup
LAMBDA_WARMUP_EPISODES = 20000 # longer unconstrained phase to learn high-AR structure first

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_EPS = 300
SMOOTH_W = 1000

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTDIR     = ROOT / "results"
MODEL_PATH = OUTDIR / "lagrange_ppo_model"
