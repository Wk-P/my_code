"""
Hyperparameter & path configuration for DQN.
DQN = RL WITHOUT action masking; violations → reward=-1, episode terminates.
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
TOTAL_STEPS = get_total_steps("problem_dqn")
SEED        = 42
DEVICE      = "auto"
N_ENVS      = 12
SUBPROC_START_METHOD = "fork"
TORCH_NUM_THREADS = 4        # leave cores for SubprocVecEnv workers
PROGRESS_LOG_EVERY_STEPS = 200_000

# ── DQN hyperparameters ──────────────────────────────────────────────────────────────────
DQN_LR                    = 1e-3
DQN_BUFFER_SIZE           = 100_000
DQN_LEARNING_STARTS       = 2_000
DQN_BATCH_SIZE            = 64
DQN_TAU                   = 1.0
DQN_GAMMA                 = 0.99
DQN_TRAIN_FREQ            = 4
DQN_GRADIENT_STEPS        = 1
DQN_TARGET_UPDATE         = 500
DQN_EXPLORATION_FRACTION  = 0.5
DQN_EXPLORATION_FINAL_EPS = 0.05
DQN_NET_ARCH              = [128, 128]

# ── Evaluation ──────────────────────────────────────────────────────────────────
EVAL_EPS  = 300
SMOOTH_W  = 1000

# ── Paths ───────────────────────────────────────────────────────────────────────
OUTDIR     = ROOT / "results"
MODEL_PATH = OUTDIR / "dqn_model"
