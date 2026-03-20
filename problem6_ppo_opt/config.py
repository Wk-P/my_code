"""
Hyperparameter & path configuration for Problem 6.
Edit ONLY this file to change problem size, training length, etc.
"""

from pathlib import Path

ROOT = Path(__file__).parent

# ── Scenario source (same YAML config file as problem2_ilp) ──────────────
YAML_CONFIG  = ROOT / ".." / "problem2_ilp" / "config" / "config_20260305_183222.yaml"
SCENARIO_IDX = 0   # 0-indexed: 0 = Scenario 1, 1 = Scenario 2 ...
# N and M are inferred automatically from the YAML file

with open(YAML_CONFIG) as f:
    import yaml
    cfg = yaml.safe_load(f)
    _all = cfg["scenarios"]
    N = len(_all[0]["ECUs"])          # 所有 200 个 scenario 均为相同规模
    M = len(_all[0]["SVCs"])

    # 全部 200 个 scenario，每项为 (caps_list, reqs_list)
    SCENARIOS = [
        (
            [ecu["capacity"] for ecu in sc["ECUs"]],
            [svc["requirement"] for svc in sc["SVCs"]],
        )
        for sc in _all
    ]
    # 单 scenario 兼容接口（向后兼容）
    VMS_POOL = SCENARIOS[SCENARIO_IDX][0]
    REQ_POOL = SCENARIOS[SCENARIO_IDX][1]

# ── Training ──────────────────────────────────────────────────────────────────
TOTAL_STEPS = 10_000_000
SEED        = 42
DEVICE      = "auto"    # "auto" -> use CUDA if available, else CPU
N_ENVS      = 40
SUBPROC_START_METHOD = "fork"
TORCH_NUM_THREADS = 4        # leave cores for SubprocVecEnv workers
PROGRESS_LOG_EVERY_STEPS = 200_000

# ── PPO hyperparameters ───────────────────────────────────────────────────────
PPO_LR          = 3e-4
PPO_N_STEPS     = 256   # rollout length (steps collected before each update)
PPO_BATCH_SIZE  = 128
PPO_N_EPOCHS    = 10
PPO_GAMMA       = 0.999  # high gamma: final-step reward must propagate M steps back
PPO_GAE_LAMBDA  = 0.95
PPO_CLIP_RANGE  = 0.2
PPO_NET_ARCH    = [256, 256]

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_EPS  = 300
SMOOTH_W  = 1000

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTDIR     = ROOT / "results"
MODEL_PATH = OUTDIR / "ppo_p6_model"
