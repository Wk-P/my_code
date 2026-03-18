"""
Configuration for Problem 5: Lagrangian Constraint Relaxation.

P5 = PPO with adaptive Lagrangian multiplier λ.
  - No action masking; constraints are penalised, not blocked.
  - Episode always runs M steps; remaining_vms can go negative.
  - Reward per step: ru/M - λ * c_t
  - λ updated via dual ascent every LAMBDA_UPDATE_WINDOW episodes:
        λ ← clip(λ + LAMBDA_LR * (avg_viol_rate - LAMBDA_TARGET), 0, LAMBDA_MAX)
"""

from pathlib import Path

ROOT = Path(__file__).parent

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
TOTAL_STEPS = 100_000
SEED        = 42
DEVICE      = "cpu"

# ── PPO hyperparameters ───────────────────────────────────────────────────────
PPO_LR         = 3e-4
PPO_N_STEPS    = 2048
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS   = 10
PPO_GAMMA      = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_NET_ARCH   = dict(pi=[128, 128], vf=[256, 256])

# ── Lagrangian multiplier (dual variable) ─────────────────────────────────────
LAMBDA_INIT          = 0.0    # initial λ value
LAMBDA_LR            = 0.05   # dual ascent step size per update
LAMBDA_TARGET        = 0.0    # target per-step violation rate (0 = zero violations)
LAMBDA_MAX           = 10.0   # upper clip to prevent explosion
LAMBDA_UPDATE_WINDOW = 30     # number of recent episodes to estimate violation rate

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_EPS = 300
SMOOTH_W = 30

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTDIR     = ROOT / "results"
MODEL_PATH = OUTDIR / "lagrange_ppo_model"
