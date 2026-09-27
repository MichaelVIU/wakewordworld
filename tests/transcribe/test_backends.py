from __future__ import annotations

import builtins
import sys

import pytest

from wakewordworld.transcribe.backends import BackendUnavailable, check_backends, get_backend


def _block(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if any(name == n or name.startswith(n + ".") for n in names):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for n in names:
        sys.modules.pop(n, None)


def test_faster_whisper_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _block(monkeypatch, "faster_whisper")
    with pytest.raises(BackendUnavailable, match="faster-whisper"):
        get_backend("faster-whisper")


def test_parakeet_missing_or_wrong_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    _block(monkeypatch, "parakeet_mlx")
    with pytest.raises(BackendUnavailable, match=r"parakeet-mlx|macOS"):
        get_backend("parakeet-mlx")


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown ASR backend"):
        get_backend("whisperx")


def test_check_backends_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    _block(monkeypatch, "faster_whisper", "parakeet_mlx")
    report = check_backends()
    assert set(report) == {"faster-whisper", "parakeet-mlx"}
    assert all(isinstance(v, str) for v in report.values())
