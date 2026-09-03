"""
Hyperparameter & path configuration for Problem 4.
P4 = RL WITH constraint enforcement (Action Masking).
Edit ONLY this file to change problem size, training length, etc.
"""

from pathlib import Path
import sys
import os

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from shared.training_steps_config import get_total_steps

# ── Scenario source (same YAML config file as problem2_ilp) ──────────────
YAML_CONFIG  = ROOT / ".." / "ilp" / "config" / "config_ecu_lt_svc.yaml"
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
TOTAL_STEPS = get_total_steps("ppo_mask", scenario=ROOT.parent.name)
SEED        = int(os.environ.get("TRAIN_SEED", "42"))
# ── Train / Test split (80/20 of feasible scenarios, deterministic) ──────────
import random as _random
_rng = _random.Random(SEED)
_idxs = list(range(len(FEASIBLE_SCENARIOS)))
_rng.shuffle(_idxs)
_n_train_default = int(0.8 * len(FEASIBLE_SCENARIOS))
# TRAIN_SCENARIO_COUNT lets an experiment shrink the training set to study
# whether more training scenarios helps, while TEST_SCENARIOS (the last 20%
# of the SEED-shuffled pool) stays fixed regardless, so results across
# different counts stay comparable against the same held-out set.
_n_train = int(os.environ.get("TRAIN_SCENARIO_COUNT", _n_train_default))
_n_train = min(_n_train, _n_train_default)
TRAIN_SCENARIOS = [FEASIBLE_SCENARIOS[i] for i in _idxs[:_n_train]]
TEST_SCENARIOS  = [FEASIBLE_SCENARIOS[i] for i in _idxs[_n_train_default:]]

DEVICE      = "auto"
N_ENVS      = 40
TORCH_NUM_THREADS = 10       # shared host: keep total concurrent demand ~30 cores
PROGRESS_LOG_EVERY_STEPS = 200_000

# ── MaskablePPO hyperparameters ───────────────────────────────────────────────
PPO_LR          = 3e-4
PPO_N_STEPS     = 512    # collect multiple episodes per rollout to amortise SB3 overhead
PPO_BATCH_SIZE  = 256
PPO_N_EPOCHS    = 10
PPO_GAMMA       = 0.99
PPO_GAE_LAMBDA  = 0.95
PPO_CLIP_RANGE  = 0.2
# v1.0.1: entropy schedule replaces the static PPO_ENT_COEF=0.005 constant —
# PPO-CMA (Hamalainen et al. 2020) shows a fixed/small entropy coefficient lets
# exploration collapse prematurely; anneal from high to low instead.
# All three overridable via env vars so scripts/run_paper_verification.sh can
# drive v1.0.1..v1.0.4 without editing this file per run.
#
# v1.0.1..v2.3.x default was INIT=FINAL=0.005 -- i.e. entropy_at() was a
# no-op the whole time, NOT actually annealing despite the mechanism being
# wired up. v1.0.2 DID test a real schedule (0.02->0.002) but under the old
# pre-v2.2.0 binary +-M reward (AR wasn't part of the objective yet), came
# out slightly worse on AR, and the no-op default was kept ever since.
# v2.4.0 (add_states): AR is now reward-central (v2.2.0's M*ar, and the
# M*(2*ar-1) rescale that gives it equal weight to the violation penalty),
# so premature exploration collapse matters differently than it did in
# v1.0.2's test -- reusing v1.0.2's validated 0.02->0.002 values as the
# default here to test the schedule under the current reward for the first
# time, not repeating the old (differently-scoped) experiment.
PPO_ENT_COEF_INIT  = float(os.environ.get("ENT_COEF_INIT", "0.02"))
PPO_ENT_COEF_FINAL = float(os.environ.get("ENT_COEF_FINAL", "0.002"))
# v1.0.3/v1.0.4: down-weight negative-advantage samples in the policy loss
# (CMA-ES-style selection). 1.0 = vanilla PPO (no pruning).
ADV_PRUNE_WEIGHT   = float(os.environ.get("ADV_PRUNE_WEIGHT", "1.0"))
# v2.4.0 (add_states): potential-based shaping weight on the bottleneck_risk
# state feature (see P4Env docstring / shared/adaptive_ppo.py::beta_at).
# Linearly annealed init->final over training like the entropy schedule,
# but decaying (not rising) -- see beta_at()'s docstring for why a constant
# beta measured worse than beta=0.0 in an ablation. Both 0.0 by default
# (exact no-op, preserves pre-v2.4.0 behaviour).
BOTTLENECK_SHAPING_WEIGHT_INIT  = float(os.environ.get("BOTTLENECK_SHAPING_WEIGHT_INIT", "0.0"))
BOTTLENECK_SHAPING_WEIGHT_FINAL = float(os.environ.get("BOTTLENECK_SHAPING_WEIGHT_FINAL", "0.0"))
# v2.7.0: weight on AR quality within the success-branch reward (P4Env.step,
# see shared/adaptive_ppo.py::ar_weight_at). Annealed init->final RISING
# (opposite direction from bottleneck shaping) -- 0.0 = pure "did you
# complete the placement" signal, 1.0 = original M*(2*ar-1) formula.
# Default 0.0->1.0: early training ignores AR entirely (pure success
# signal), full AR-quality gradient only kicks in once training is mostly
# done. Set both to 1.0 to reproduce the pre-v2.7.0 behaviour exactly.
AR_WEIGHT_INIT  = float(os.environ.get("AR_WEIGHT_INIT", "0.0"))
AR_WEIGHT_FINAL = float(os.environ.get("AR_WEIGHT_FINAL", "1.0"))
# Fraction of TOTAL_STEPS over which the AR-weight ramp completes (reaches
# AR_WEIGHT_FINAL and holds there for the rest of training), instead of
# stretching the ramp across the entire run. 1.0 = old behaviour (ramp
# finishes right as training ends, leaving ~no budget for the reactivated
# AR gradient to actually improve packing quality). 0.5 = reach full AR
# weight at the halfway point, giving the back half of training to act on
# the AR signal instead of only touching it in the last few percent.
AR_WEIGHT_RAMP_FRACTION = float(os.environ.get("AR_WEIGHT_RAMP_FRACTION", "1.0"))
# Larger network to process richer observation space (3N+2 dims)
PPO_NET_ARCH    = dict(pi=[256, 256], vf=[256, 256])

# ── Behavior-cloning pretraining (ILP expert warm-start) ──────────────────────
BC_EPOCHS     = 20
BC_BATCH_SIZE = 256
BC_LR         = 1e-3

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_EPS  = len(TEST_SCENARIOS)
SMOOTH_W  = 1000
# best-of-N stochastic re-rolls at eval time: an online/no-backtrack
# policy commits to one irrevocable pass per attempt, so re-sampling N
# independent stochastic rollouts per test scenario and keeping the best
# (success first, then most services validly placed, then highest AR)
# sidesteps that ceiling without touching training.
EVAL_BEST_OF_N = 8

# ── Paths ─────────────────────────────────────────────────────────────────────
from shared.paths import results_dir
OUTDIR     = results_dir(ROOT.parent.name, "ppo_mask")
