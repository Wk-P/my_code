"""shared/version_config.py — single source of truth for the "current" paper
version tag.

Every place that used to hardcode a version string (shared/paths.py's
default VERSION, scripts/run_*.sh's PAPER_VERSION) should import
CURRENT_VERSION from here instead. Bump CURRENT_VERSION once per release;
$PAPER_VERSION still overrides it for ad-hoc/historical runs (e.g. replaying
an old ablation tag) without editing this file.
"""

CURRENT_VERSION = "1.2.4"
