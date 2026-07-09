"""
Email notification helper for long-running background training jobs.

Configuration lives in a git-ignored `.env.email` file at the repo root
(see `.env.email.example` for the template), one KEY=VALUE per line:
    NOTIFY_EMAIL_TO     recipient address (required to actually send anything)
    NOTIFY_SMTP_HOST    default: smtp.gmail.com
    NOTIFY_SMTP_PORT    default: 465 (SMTPS)
    NOTIFY_SMTP_USER    sender login (e.g. your Gmail address)
    NOTIFY_SMTP_PASS    sender password — for Gmail this must be a 16-char
                        "App Password" (myaccount.google.com/apppasswords),
                        NOT your normal login password (Gmail rejects those
                        for SMTP since 2FA is required for app passwords).

Real environment variables (if already set) take precedence over the file,
so CI/production can still override without touching `.env.email`.

If NOTIFY_EMAIL_TO / NOTIFY_SMTP_USER / NOTIFY_SMTP_PASS are not all
resolved, send_notification() silently no-ops (prints a note) so training
scripts never fail just because email isn't configured.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env.email"
_KEYS = (
    "NOTIFY_EMAIL_TO", "NOTIFY_SMTP_HOST", "NOTIFY_SMTP_PORT",
    "NOTIFY_SMTP_USER", "NOTIFY_SMTP_PASS",
)


def _load_env_file() -> dict[str, str]:
    """Parse KEY=VALUE lines from .env.email (missing file -> {})."""
    values: dict[str, str] = {}
    if not _ENV_FILE.exists():
        return values
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key in _KEYS:
            values[key] = val
    return values


def _config() -> dict[str, str | None]:
    """Real env vars win; fall back to .env.email; missing -> None."""
    file_values = _load_env_file()
    return {key: os.environ.get(key, file_values.get(key)) for key in _KEYS}


def send_notification(subject: str, body: str) -> bool:
    cfg = _config()
    to_addr   = cfg["NOTIFY_EMAIL_TO"]
    smtp_user = cfg["NOTIFY_SMTP_USER"]
    smtp_pass = cfg["NOTIFY_SMTP_PASS"]

    if not (to_addr and smtp_user and smtp_pass):
        print("[notify] NOTIFY_EMAIL_TO / NOTIFY_SMTP_USER / NOTIFY_SMTP_PASS "
              f"not all set (checked env vars and {_ENV_FILE}) — skipping email.")
        return False

    host = cfg["NOTIFY_SMTP_HOST"] or "smtp.gmail.com"
    port = int(cfg["NOTIFY_SMTP_PORT"] or "465")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = to_addr
    msg.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"[notify] email sent to {to_addr}")
        return True
    except Exception as e:
        print(f"[notify] failed to send email: {e}")
        return False


if __name__ == "__main__":
    send_notification("[test] shared.notify", "This is a test notification.")
