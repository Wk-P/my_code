"""
Hyperparameter & path configuration for Problem 4.
P4 = RL WITH constraint enforcement (Action Masking).
Edit ONLY this file to change problem size, training length, etc.
"""

from pathlib import Path

ROOT = Path(__file__).parent

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
TOTAL_STEPS = 100_000
SEED        = 42
DEVICE      = "auto"

# ── MaskablePPO hyperparameters ───────────────────────────────────────────────
PPO_LR          = 3e-4
PPO_N_STEPS     = 1024
PPO_BATCH_SIZE  = 128
PPO_N_EPOCHS    = 10
PPO_GAMMA       = 0.999
PPO_GAE_LAMBDA  = 0.95
PPO_CLIP_RANGE  = 0.2
PPO_NET_ARCH    = dict(pi=[128, 128], vf=[256, 256])

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_EPS  = 300
SMOOTH_W  = 20

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTDIR     = ROOT / "results"
MODEL_PATH = OUTDIR / "maskppo_p4_model"
