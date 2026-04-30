"""
Hyperparameter & path configuration for DDQN.
DDQN = Double DQN — RL WITHOUT action masking; violations → reward=-1, episode terminates.
Uses two Q-networks to reduce overestimation bias compared to standard DQN.
Edit ONLY this file to change problem size, training length, etc.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from training_steps_config import get_total_steps

# ── Scenario source (same YAML config file as problem2_ilp) ──────────────
YAML_CONFIG  = ROOT / ".." / "problem2_ilp" / "config" / "config_ecu_lt_svc.yaml"
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
    # Only scenarios marked feasible — used for evaluation metrics
    FEASIBLE_SCENARIOS = [
        s for s, sc in zip(SCENARIOS, _all) if sc.get("feasible", True)
    ]
    VMS_POOL = SCENARIOS[SCENARIO_IDX][0]
    REQ_POOL = SCENARIOS[SCENARIO_IDX][1]

# ── Training ──────────────────────────────────────────────────────────────────
TOTAL_STEPS = get_total_steps("problem_ddqn")
SEED        = 42
# ── Train / Test split (80/20 of feasible scenarios, deterministic) ──────────
import random as _random
_rng = _random.Random(SEED)
_idxs = list(range(len(FEASIBLE_SCENARIOS)))
_rng.shuffle(_idxs)
_n_train = int(0.8 * len(FEASIBLE_SCENARIOS))
TRAIN_SCENARIOS = [FEASIBLE_SCENARIOS[i] for i in _idxs[:_n_train]]
TEST_SCENARIOS  = [FEASIBLE_SCENARIOS[i] for i in _idxs[_n_train:]]

DEVICE      = "auto"
N_ENVS      = 12
SUBPROC_START_METHOD = "fork"
TORCH_NUM_THREADS = 4        # leave cores for SubprocVecEnv workers
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
DDQN_EXPLORATION_FINAL_EPS = 0.05
DDQN_NET_ARCH              = [128, 128]

# ── Evaluation ──────────────────────────────────────────────────────────────────
EVAL_EPS  = 300
SMOOTH_W  = 1000

# ── Paths ───────────────────────────────────────────────────────────────────────
OUTDIR     = ROOT / "results"
MODEL_PATH = OUTDIR / "ddqn_model"
