"""Create the Hugging Face webhook that re-runs the evaluation template job.

The webhook watches the submissions repository (pull requests and content updates)
and triggers ``--job-id``, a job created once with ``hf_jobs_eval.py``. Hugging Face
re-runs the template with ``WEBHOOK_PAYLOAD``, ``WEBHOOK_REPO_ID`` and ``WEBHOOK_ID``
set in the environment.

Usage::

    python scripts/hf_webhook_setup.py --watch-repo wakewordworld/submissions \
        --job-id <job id> [--secret ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def build_payload(watch_repo: str, job_id: str, *, secret: str | None) -> dict[str, Any]:
    """Arguments for ``huggingface_hub.create_webhook``."""
    return {
        "job_id": job_id,
        "watched": [{"type": "dataset", "name": watch_repo}],
        "domains": ["repo", "discussion"],
        "secret": secret,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--watch-repo", required=True, help="Dataset repo receiving submissions.")
    ap.add_argument("--job-id", required=True, help="Template job to re-trigger.")
    ap.add_argument("--secret", default=None, help="Webhook secret (optional).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    payload = build_payload(args.watch_repo, args.job_id, secret=args.secret)
    if args.dry_run:
        printable = {**payload, "secret": "***" if payload["secret"] else None}
        print(json.dumps(printable, indent=2))
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set (use --dry-run)", file=sys.stderr)
        return 2
    import huggingface_hub as hf

    info = hf.create_webhook(token=token, **payload)
    print(json.dumps({"id": getattr(info, "id", None), "url": getattr(info, "url", None)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
