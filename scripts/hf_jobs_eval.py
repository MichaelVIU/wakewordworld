"""Launch verified evaluation runs on Hugging Face Jobs.

One job per (engine, manifest version). The job image is the engine's container pushed
to a registry; the private dataset repository (public manifest, sealed split and chunk
audio) is mounted read-only at ``/data`` through a Jobs volume, and results are pushed to
a results dataset repository from inside the job.

Usage::

    python scripts/hf_jobs_eval.py --engine openwakeword --engine vosk \
        --manifest-version 0.1.0 --image-prefix ghcr.io/michaelviu/wakewordworld \
        --image-tag 0.1.0 --dataset-repo wakewordworld/benchmark-internal \
        --results-repo wakewordworld/results [--dry-run]

Requires ``HF_TOKEN`` unless ``--dry-run`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_FLAVOR = "cpu-upgrade"
DEFAULT_TIMEOUT = "6h"


@dataclass
class JobSpec:
    """Everything needed to call ``huggingface_hub.run_job``."""

    image: str
    command: list[str]
    flavor: str
    timeout: str
    name: str
    env: dict[str, str] = field(default_factory=dict)
    secrets: list[str] = field(default_factory=list)
    volumes: list[dict[str, str]] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)


def build_job_spec(
    engine: str,
    *,
    manifest_version: str,
    image_prefix: str,
    image_tag: str,
    dataset_repo: str,
    results_repo: str,
    flavor: str = DEFAULT_FLAVOR,
    timeout: str = DEFAULT_TIMEOUT,
    n_boot: int = 1000,
) -> JobSpec:
    """Compose the job for one engine."""
    manifest = f"/data/manifests/{manifest_version}"
    script = " && ".join(
        [
            f"wakewordworld eval run {engine} --manifest {manifest} --n-boot {n_boot} "
            "--data-root /data --out /results",
            f"hf upload {results_repo} /results results/{manifest_version}/{engine} "
            "--repo-type dataset --commit-message "
            f"'eval {engine} on {manifest_version}'",
        ]
    )
    return JobSpec(
        image=f"{image_prefix}-{engine}:{image_tag}",
        command=["bash", "-lc", script],
        flavor=flavor,
        timeout=timeout,
        name=f"wakewordworld-eval-{engine}-{manifest_version}".replace(".", "-"),
        env={"WWW_DATA_ROOT": "/data", "PYTHONUNBUFFERED": "1"},
        secrets=["HF_TOKEN"],
        volumes=[{"type": "dataset", "source": dataset_repo, "mount_path": "/data"}],
        labels={"project": "wakewordworld", "engine": engine, "manifest": manifest_version},
    )


def launch(spec: JobSpec, *, token: str, namespace: str | None) -> dict[str, Any]:
    """Submit one job; returns a small dict describing it."""
    import huggingface_hub as hf

    volume_cls = getattr(hf, "Volume", None)
    volumes = None
    if volume_cls is not None:
        volumes = [volume_cls(**v) for v in spec.volumes]
    else:  # pragma: no cover - older clients: download inside the job instead
        spec.command = [
            "bash",
            "-lc",
            f"hf download {spec.volumes[0]['source']} --repo-type dataset --local-dir /data && "
            + spec.command[-1],
        ]
    info = hf.run_job(
        image=spec.image,
        command=spec.command,
        env=spec.env,
        secrets={s: os.environ[s] for s in spec.secrets if s in os.environ},
        flavor=spec.flavor,
        timeout=spec.timeout,
        name=spec.name,
        labels=spec.labels,
        volumes=volumes,
        namespace=namespace,
        token=token,
    )
    return {"id": getattr(info, "id", None), "url": getattr(info, "url", None), "name": spec.name}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--engine", action="append", required=True, help="Engine id (repeatable).")
    ap.add_argument("--manifest-version", required=True)
    ap.add_argument("--image-prefix", required=True, help="e.g. ghcr.io/<owner>/wakewordworld")
    ap.add_argument("--image-tag", required=True)
    ap.add_argument(
        "--dataset-repo", required=True, help="Private dataset repo with manifests and audio."
    )
    ap.add_argument("--results-repo", required=True)
    ap.add_argument("--flavor", default=DEFAULT_FLAVOR)
    ap.add_argument("--timeout", default=DEFAULT_TIMEOUT)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--namespace", default=None, help="Organisation billed for the jobs.")
    ap.add_argument(
        "--dry-run", action="store_true", help="Print job specs as JSON, submit nothing."
    )
    args = ap.parse_args(argv)

    specs = [
        build_job_spec(
            e,
            manifest_version=args.manifest_version,
            image_prefix=args.image_prefix,
            image_tag=args.image_tag,
            dataset_repo=args.dataset_repo,
            results_repo=args.results_repo,
            flavor=args.flavor,
            timeout=args.timeout,
            n_boot=args.n_boot,
        )
        for e in args.engine
    ]
    if args.dry_run:
        print(json.dumps([asdict(s) for s in specs], indent=2))
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set (use --dry-run to inspect job specs)", file=sys.stderr)
        return 2
    for spec in specs:
        print(json.dumps(launch(spec, token=token, namespace=args.namespace)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
