"""
Hyperparameter & path configuration for Problem 4.
P4 = RL WITH constraint enforcement (Action Masking).
Edit ONLY this file to change problem size, training length, etc.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from training_steps_config import get_total_steps

# ── Scenario source (same YAML config file as problem2_ilp) ──────────────
YAML_CONFIG  = ROOT / ".." / "problem2_ilp" / "config" / "config_20260305_183222.yaml"
SCENARIO_IDX = 0   # 0-indexed: 0 = Scenario 1, 1 = Scenario 2 ...

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
TOTAL_STEPS = get_total_steps("problem4_ppo_mask")
SEED        = 42
DEVICE      = "auto"
N_ENVS      = 40
SUBPROC_START_METHOD = "fork"
TORCH_NUM_THREADS = 4        # leave cores for SubprocVecEnv workers
PROGRESS_LOG_EVERY_STEPS = 200_000

# ── MaskablePPO hyperparameters ───────────────────────────────────────────────
PPO_LR          = 3e-4
PPO_N_STEPS     = 256
PPO_BATCH_SIZE  = 128
PPO_N_EPOCHS    = 10
PPO_GAMMA       = 0.999
PPO_GAE_LAMBDA  = 0.95
PPO_CLIP_RANGE  = 0.2
# Larger network to process richer observation space (3N+2 dims)
PPO_NET_ARCH    = dict(pi=[256, 256], vf=[256, 256])

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_EPS  = 300
SMOOTH_W  = 1000

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTDIR     = ROOT / "results"
MODEL_PATH = OUTDIR / "maskppo_p4_model"
