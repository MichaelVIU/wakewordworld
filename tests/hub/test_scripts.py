from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_hf_jobs_eval_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load("hf_jobs_eval")
    rc = mod.main(
        [
            "--engine",
            "openwakeword",
            "--engine",
            "vosk",
            "--manifest-version",
            "0.1.0",
            "--image-prefix",
            "ghcr.io/o/wakewordworld",
            "--image-tag",
            "0.1.0",
            "--dataset-repo",
            "o/internal",
            "--results-repo",
            "o/results",
            "--dry-run",
        ]
    )
    assert rc == 0
    specs = json.loads(capsys.readouterr().out)
    assert [s["image"] for s in specs] == [
        "ghcr.io/o/wakewordworld-openwakeword:0.1.0",
        "ghcr.io/o/wakewordworld-vosk:0.1.0",
    ]
    assert specs[0]["flavor"] == "cpu-upgrade"
    assert specs[0]["volumes"][0] == {
        "type": "dataset",
        "source": "o/internal",
        "mount_path": "/data",
    }
    assert "eval run openwakeword --manifest /data/manifests/0.1.0" in specs[0]["command"][-1]
    assert "hf upload o/results" in specs[0]["command"][-1]


def test_hf_jobs_eval_launch_uses_fake_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load("hf_jobs_eval")
    calls: list[dict[str, object]] = []
    fake = types.ModuleType("huggingface_hub")

    class Volume:
        def __init__(self, **kw: object) -> None:
            self.kw = kw

    def run_job(**kw: object) -> types.SimpleNamespace:
        calls.append(kw)
        return types.SimpleNamespace(id="job1", url="https://hf/jobs/job1")

    fake.Volume = Volume  # type: ignore[attr-defined]
    fake.run_job = run_job  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setenv("HF_TOKEN", "tok")
    spec = mod.build_job_spec(
        "vosk",
        manifest_version="0.1.0",
        image_prefix="p",
        image_tag="t",
        dataset_repo="o/internal",
        results_repo="o/results",
    )
    out = mod.launch(spec, token="tok", namespace=None)
    assert out["id"] == "job1"
    assert calls[0]["flavor"] == "cpu-upgrade"
    assert calls[0]["secrets"] == {"HF_TOKEN": "tok"}
    assert isinstance(calls[0]["volumes"][0], Volume)  # type: ignore[index]


def test_hf_jobs_eval_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load("hf_jobs_eval")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc = mod.main(
        [
            "--engine",
            "x",
            "--manifest-version",
            "1",
            "--image-prefix",
            "p",
            "--image-tag",
            "t",
            "--dataset-repo",
            "d",
            "--results-repo",
            "r",
        ]
    )
    assert rc == 2


def test_webhook_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load("hf_webhook_setup")
    rc = mod.main(["--watch-repo", "o/subs", "--job-id", "j1", "--secret", "s", "--dry-run"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "j1"
    assert payload["watched"] == [{"type": "dataset", "name": "o/subs"}]
    assert payload["secret"] == "***"


def _good_engine_folder(root: Path) -> Path:
    d = root / "engines" / "toy"
    d.mkdir(parents=True)
    (d / "engine.yaml").write_text(
        "id: toy\nname: Toy\nhomepage: https://example.org\n"
        "licence:\n  runtime: Apache-2.0\n  models: CC-BY-4.0\n"
        "version: '1.0'\nmodels:\n  - name: a\n    sha256: unpinned\n  - name: b\n    sha256: "
        + "a" * 64
        + "\nwake_words: [alexa]\ncustom_word_method: training\non_device: true\n"
    )
    (d / "DISCLOSURE.md").write_text(
        "# Toy\n\n## Training data\nx\n\n## Benchmark overlap\nnone\n\n## Commercial status\nfree\n"
    )
    (d / "Dockerfile").write_text(
        'FROM python:3.12-slim\nRUN useradd -m www\nUSER www\nENTRYPOINT ["wakewordworld"]\n'
    )
    return d


def test_check_engine_folder(tmp_path: Path) -> None:
    mod = _load("check_engine_folder")
    good = _good_engine_folder(tmp_path)
    assert mod.check_folder(good) == []
    (good / "Dockerfile").write_text("FROM python:3.12-slim\nUSER root\n")
    (good / "DISCLOSURE.md").write_text("# Toy\n\n## Training data\nx\n")
    y = good / "engine.yaml"
    y.write_text(
        y.read_text()
        .replace("on_device: true", "on_device: yes please")
        .replace("sha256: unpinned", "sha256: nope")
    )
    problems = mod.check_folder(good)
    joined = "\n".join(problems)
    assert "final USER is root" in joined
    assert "Benchmark overlap" in joined
    assert "on_device must be a boolean" in joined
    assert "neither a hex digest nor 'unpinned'" in joined
    assert mod.check_folder(tmp_path / "engines" / "missing")  # three missing files
