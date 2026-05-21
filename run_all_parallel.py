"""
Launch a full experiment across all 3 groups (lt / eq / gt) in parallel,
sharing a single run_id so plot_metrics.py can identify this batch.

Usage:
    python run_all_parallel.py --seed 1 2 3 --episodes 50000
    python run_all_parallel.py --seed 10 11 --episodes 100000 --groups lt eq
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).parent
PYTHON = sys.executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",     type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--groups",   nargs="+", default=["lt", "eq", "gt"],
                        choices=["lt", "eq", "gt"])
    parser.add_argument("--run-id",   type=str, default=None,
                        help="Override auto-generated run_id")
    args = parser.parse_args()

    run_id = args.run_id or (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(4).hex()
    )
    log_dir = Path("/tmp")

    print(f"run_id  : {run_id}")
    print(f"groups  : {args.groups}")
    print(f"seeds   : {args.seed}")
    print(f"episodes: {args.episodes:,}")
    print()

    procs: list = []
    for group in args.groups:
        log_path = log_dir / f"run_{group}_{run_id}.log"
        cmd = [
            PYTHON, "-u", str(ROOT / "run_experiment.py"),
            "--group", group,
            "--seed", *[str(s) for s in args.seed],
            "--episodes", str(args.episodes),
            "--run-id", run_id,
        ]
        log_f = open(log_path, "w")
        proc  = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((group, proc, log_path, log_f))
        print(f"  [{group}] PID {proc.pid}  log → {log_path}")

    print(f"\nAll {len(procs)} groups launched. Waiting for completion...\n")

    for group, proc, log_path, log_f in procs:
        rc = proc.wait()
        log_f.close()
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  [{group}] {status}")

    print(f"\nAll done. run_id={run_id}")
    print("Run:  python scripts/plot_metrics.py  to generate figures.")


if __name__ == "__main__":
    main()
