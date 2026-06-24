"""
Hyperparameter & path configuration for DDQN.
DDQN = Double DQN — RL WITHOUT action masking; violations → reward=-1, episode terminates.
Uses two Q-networks to reduce overestimation bias compared to standard DQN.
Edit ONLY this file to change problem size, training length, etc.
"""

from pathlib import Path
import sys
import os

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from training_steps_config import get_total_steps

# ── Scenario source (same YAML config file as problem2_ilp) ──────────────
YAML_CONFIG  = ROOT / ".." / "problem2_ilp" / "config" / "config_ecu_gt_svc.yaml"
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
            sc.get("conflict_sets", []),
        )
        for sc in _all
    ]
    VMS_POOL = SCENARIOS[SCENARIO_IDX][0]
    REQ_POOL = SCENARIOS[SCENARIO_IDX][1]

# ── Training ──────────────────────────────────────────────────────────────────
TOTAL_STEPS = get_total_steps("problem_ddqn")
SEED        = int(os.environ.get("TRAIN_SEED", "42"))
# ── Train / Test split (80/20, deterministic) ────────────────────────────────
import random as _random
_rng = _random.Random(SEED)
_idxs = list(range(len(SCENARIOS)))
_rng.shuffle(_idxs)
_n_train = int(0.8 * len(SCENARIOS))
TRAIN_SCENARIOS = [SCENARIOS[i] for i in _idxs[:_n_train]]
TEST_SCENARIOS  = [SCENARIOS[i] for i in _idxs[_n_train:]]

DEVICE      = "auto"
N_ENVS      = 12
TORCH_NUM_THREADS = 8        # DummyVecEnv: more threads for CPU inference
PROGRESS_LOG_EVERY_STEPS = 200_000

# ── DDQN hyperparameters ──────────────────────────────────────────────────────────────────
DDQN_LR                    = 1e-3
DDQN_BUFFER_SIZE           = 100_000
DDQN_LEARNING_STARTS       = 2_000
DDQN_BATCH_SIZE            = 64
DDQN_TAU                   = 1.0
DDQN_GAMMA                 = 0.99
DDQN_TRAIN_FREQ            = 4
DDQN_GRADIENT_STEPS        = 1
DDQN_TARGET_UPDATE         = 500
DDQN_EXPLORATION_FRACTION  = 0.5
DDQN_EXPLORATION_FINAL_EPS = 0.0
DDQN_NET_ARCH              = [128, 128]

# ── Evaluation ──────────────────────────────────────────────────────────────────
EVAL_EPS  = len(TEST_SCENARIOS)
SMOOTH_W  = 1000

# ── Paths ───────────────────────────────────────────────────────────────────────
OUTDIR     = ROOT / "results"
MODEL_PATH = OUTDIR / "ddqn_model"
