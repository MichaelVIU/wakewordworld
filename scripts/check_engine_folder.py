"""Validate an ``engines/<id>/`` submission folder.

Checks ``engine.yaml`` keys, the disclosure headings and that the Dockerfile switches to
a non-root user. Usage: ``python scripts/check_engine_folder.py engines/<id> [...]``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REQUIRED_HEADINGS = ("Training data", "Benchmark overlap", "Commercial status")
CUSTOM_WORD_METHODS = {"none", "text", "training", "enrolment"}


def check_engine_yaml(path: Path) -> list[str]:
    """Return problems found in ``engine.yaml``."""
    problems: list[str] = []
    if not path.exists():
        return [f"{path}: missing"]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return [f"{path}: top level must be a mapping"]
    for key in ("id", "name", "homepage", "licence", "version", "models", "wake_words"):
        if key not in data:
            problems.append(f"{path}: missing key {key!r}")
    lic = data.get("licence")
    if not isinstance(lic, dict) or "runtime" not in lic or "models" not in lic:
        problems.append(f"{path}: licence must have 'runtime' and 'models'")
    models = data.get("models")
    if not isinstance(models, list):
        problems.append(f"{path}: models must be a list")
    else:
        for i, m in enumerate(models):
            if not isinstance(m, dict) or "sha256" not in m:
                problems.append(f"{path}: models[{i}] needs a sha256 (or 'unpinned')")
            elif m["sha256"] != "unpinned" and not re.fullmatch(r"[0-9a-f]{64}", str(m["sha256"])):
                problems.append(
                    f"{path}: models[{i}].sha256 is neither a hex digest nor 'unpinned'"
                )
    if data.get("custom_word_method") not in CUSTOM_WORD_METHODS:
        problems.append(f"{path}: custom_word_method must be one of {sorted(CUSTOM_WORD_METHODS)}")
    if not isinstance(data.get("on_device"), bool):
        problems.append(f"{path}: on_device must be a boolean")
    if not isinstance(data.get("wake_words"), list) or not data.get("wake_words"):
        problems.append(f"{path}: wake_words must be a non-empty list")
    if data.get("id") and data["id"] != path.parent.name:
        problems.append(f"{path}: id {data['id']!r} must equal folder name {path.parent.name!r}")
    return problems


def check_disclosure(path: Path) -> list[str]:
    """Return problems found in ``DISCLOSURE.md``."""
    if not path.exists():
        return [f"{path}: missing"]
    text = path.read_text(encoding="utf-8")
    return [
        f"{path}: missing heading '{h}'"
        for h in REQUIRED_HEADINGS
        if not re.search(rf"^#+\s*{re.escape(h)}\b", text, flags=re.MULTILINE | re.IGNORECASE)
    ]


def check_dockerfile(path: Path) -> list[str]:
    """Return problems found in the ``Dockerfile``."""
    if not path.exists():
        return [f"{path}: missing"]
    users = re.findall(r"^\s*USER\s+(\S+)", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if not users:
        return [f"{path}: no USER instruction (must run as non-root)"]
    if users[-1].split(":")[0] in {"root", "0"}:
        return [f"{path}: final USER is root"]
    return []


def check_folder(folder: Path) -> list[str]:
    """All checks for one engine folder."""
    return (
        check_engine_yaml(folder / "engine.yaml")
        + check_disclosure(folder / "DISCLOSURE.md")
        + check_dockerfile(folder / "Dockerfile")
    )


def main(argv: list[str]) -> int:
    """CLI entry point."""
    folders = [Path(a) for a in argv] or sorted(p for p in Path("engines").iterdir() if p.is_dir())
    problems: list[str] = []
    for f in folders:
        problems.extend(check_folder(f))
    for p in problems:
        print(p, file=sys.stderr)
    print(f"engine folder check: {len(folders)} folder(s), {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
