"""CI gate: public manifests must not leak tier B audio.

Checks every ``manifests/**/chunks.jsonl``:
- tier B rows have no ``audio_sha256``;
- tier A rows reference a source whose spec has licence evidence;
- no row has tier ``forbidden``.
"""

from __future__ import annotations

import json
import sys

from wakewordworld.licences import LicenceTier
from wakewordworld.sources.spec import load_source_specs
from wakewordworld.util.paths import repo_root


def main() -> int:
    root = repo_root()
    specs = {s.id: s for s in load_source_specs(root / "sources")}
    problems: list[str] = []
    for path in sorted((root / "manifests").glob("**/chunks.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                tier = row.get("licence_tier")
                where = f"{path.relative_to(root)}:{lineno}"
                if tier == LicenceTier.FORBIDDEN:
                    problems.append(f"{where}: forbidden tier in public manifest")
                if tier == LicenceTier.B and row.get("audio_sha256"):
                    problems.append(f"{where}: tier B row carries an audio checksum")
                if tier == LicenceTier.A:
                    spec = specs.get(row.get("source_id", ""))
                    if spec is None:
                        problems.append(f"{where}: unknown source {row.get('source_id')!r}")
                    elif spec.licence.evidence is None:
                        problems.append(f"{where}: tier A row from source without evidence")
    for p in problems:
        print(p, file=sys.stderr)
    print(f"licence gate: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
