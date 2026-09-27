"""sherpa-onnx open-vocabulary keyword spotting adapter (https://k2-fsa.github.io/sherpa/onnx/kws/).

A streaming zipformer transducer is decoded with keyword-constrained beam search.
Keywords are token sequences (BPE pieces for the English GigaSpeech model) with an
optional boosting score, an acoustic threshold and a label. The spotter emits a hit
event with the keyword's label; there is no graded per-frame score, so this is a
boolean-style engine: the phrase scores ``1.0`` on the harness frame in which a hit
was decoded and the harness sweeps ``keywords_threshold`` to obtain a curve.

Config:
    ``model_dir``: folder with ``tokens.txt``, ``bpe.model``, encoder/decoder/joiner
    ONNX files; or ``model_name`` (default ``sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01``)
    to download and extract the release archive into the benchmark cache.
    ``keywords``: list of phrases (converted with ``sherpa_onnx.text2token``, which
    needs ``sentencepiece``); or ``keywords_file`` in sherpa's native format.
    ``keywords_threshold`` (default 0.25), ``keywords_score`` (1.0),
    ``num_trailing_blanks`` (1), ``int8`` (True), ``num_threads`` (1).
"""

from __future__ import annotations

import re
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wakewordworld.engines.base import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    Capabilities,
    Engine,
    EngineInfo,
    EngineUnavailableError,
    engine_registry,
)
from wakewordworld.engines.models import download_file, hash_models, models_dir
from wakewordworld.util.paths import DataRoot

__all__ = ["SherpaKwsEngine", "create", "ensure_sherpa_model"]

_INSTALL_HINT = "install with: uv sync --extra engines-sherpa  (package: sherpa-onnx)"
DEFAULT_MODEL = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
MODEL_URLS: dict[str, tuple[str, str | None]] = {
    DEFAULT_MODEL: (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
        f"{DEFAULT_MODEL}.tar.bz2",
        "f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a",
    ),
}
SWEEP_THRESHOLDS: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)
_LANG_RE = re.compile(r"-(zh-en|wenetspeech|gigaspeech)-")
_LANG_MAP = {"gigaspeech": ("en",), "wenetspeech": ("zh",), "zh-en": ("zh", "en")}


def ensure_sherpa_model(
    model_name: str = DEFAULT_MODEL, *, data_root: DataRoot | None = None
) -> tuple[Path, str]:
    """Download and extract a sherpa-onnx KWS release archive; returns (dir, sha256)."""
    target = models_dir("sherpa_kws", data_root)
    folder = target / model_name
    if folder.is_dir() and (folder / "tokens.txt").exists():
        archive = target / f"{model_name}.tar.bz2"
        if not archive.exists():
            return folder, "unpinned-preinstalled"
    try:
        url, expected = MODEL_URLS[model_name]
    except KeyError as exc:
        msg = f"unknown sherpa KWS model {model_name!r}; known: {sorted(MODEL_URLS)}"
        raise ValueError(msg) from exc
    archive = target / f"{model_name}.tar.bz2"
    digest = download_file(url, archive, expected_sha256=expected)
    if not (folder / "tokens.txt").exists():
        with tarfile.open(archive, "r:bz2") as tf:
            members = [
                m
                for m in tf.getmembers()
                if not (m.name.startswith("/") or ".." in Path(m.name).parts)
            ]
            tf.extractall(target, members=members, filter="data")
    return folder, digest


def _pick(folder: Path, prefix: str, int8: bool) -> Path:
    cands = sorted(folder.glob(f"{prefix}*.onnx"))
    quant = [c for c in cands if c.name.endswith(".int8.onnx")]
    plain = [c for c in cands if not c.name.endswith(".int8.onnx")]
    chosen = (quant or plain) if int8 else (plain or quant)
    if not chosen:
        msg = f"no {prefix}*.onnx in {folder}"
        raise FileNotFoundError(msg)
    return chosen[0]


def _label(phrase: str) -> str:
    return re.sub(r"\s+", "_", phrase.strip().lower())


class SherpaKwsEngine:
    """Streaming adapter around ``sherpa_onnx.KeywordSpotter``."""

    def __init__(
        self,
        keywords: list[str] | None = None,
        keywords_file: str | None = None,
        model_dir: str | None = None,
        model_name: str = DEFAULT_MODEL,
        keywords_threshold: float = 0.25,
        keywords_score: float = 1.0,
        num_trailing_blanks: int = 1,
        int8: bool = True,
        num_threads: int = 1,
        max_active_paths: int = 4,
    ) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - exercised via fake module tests
            raise EngineUnavailableError(f"sherpa_onnx not importable; {_INSTALL_HINT}") from exc
        if not keywords and not keywords_file:
            msg = "sherpa_kws engine needs `keywords` (phrases) or a `keywords_file`"
            raise ValueError(msg)
        archive_hash = "unpinned"
        if model_dir:
            folder = Path(model_dir)
            name = folder.name
        else:
            folder, archive_hash = ensure_sherpa_model(model_name)
            name = model_name
        tokens = folder / "tokens.txt"
        encoder = _pick(folder, "encoder", int8)
        decoder = _pick(folder, "decoder", int8)
        joiner = _pick(folder, "joiner", int8)

        if keywords_file:
            kw_path = Path(keywords_file)
            labels = _labels_from_file(kw_path)
            self._phrase_by_label = {lbl: lbl.replace("_", " ") for lbl in labels}
        else:
            assert keywords is not None
            bpe = folder / "bpe.model"
            try:
                pieces = sherpa_onnx.text2token(
                    [k.upper() for k in keywords],
                    tokens=str(tokens),
                    tokens_type="bpe",
                    bpe_model=str(bpe),
                )
            except ImportError as exc:
                msg = (
                    "converting phrases to BPE tokens needs `sentencepiece` and `pypinyin` "
                    "(pip install sentencepiece pypinyin) or pass a prepared keywords_file"
                )
                raise EngineUnavailableError(msg) from exc
            self._phrase_by_label = {_label(k): k for k in keywords}
            lines = [
                " ".join(str(t) for t in toks)
                + f" :{keywords_score} #{keywords_threshold} @{_label(k)}"
                for k, toks in zip(keywords, pieces, strict=True)
            ]
            kw_path = models_dir("sherpa_kws") / f"keywords-{abs(hash(tuple(lines)))}.txt"
            kw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._wake_words = tuple(self._phrase_by_label)
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            keywords_file=str(kw_path),
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            max_active_paths=max_active_paths,
            keywords_score=keywords_score,
            keywords_threshold=keywords_threshold,
            num_trailing_blanks=num_trailing_blanks,
            provider="cpu",
        )
        self._stream = self._spotter.create_stream()
        m = _LANG_RE.search(name)
        self._info = EngineInfo(
            engine_id="sherpa_kws",
            version=_dist_version("sherpa-onnx"),
            wake_words=self._wake_words,
            model_hashes={
                f"{name}.tar.bz2": archive_hash,
                **{
                    f"{name}/{k}": v
                    for k, v in hash_models(
                        {"encoder": encoder, "decoder": decoder, "joiner": joiner, "tokens": tokens}
                    ).items()
                },
                "keywords": hash_models({"k": kw_path}).get("k", "unpinned"),
            },
            config={
                "model": name,
                "keywords": list(self._phrase_by_label.values()),
                "keywords_threshold": keywords_threshold,
                "keywords_score": keywords_score,
                "num_trailing_blanks": num_trailing_blanks,
                "int8": int8,
            },
            capabilities=Capabilities(
                continuous_scores=False,
                sweep_param="keywords_threshold",
                sweep_values=SWEEP_THRESHOLDS,
                custom_words="text",
                languages=_LANG_MAP.get(m.group(1), ()) if m else (),
            ),
        )

    @property
    def info(self) -> EngineInfo:
        """Identity and capabilities."""
        return self._info

    def reset(self) -> None:
        """Start a fresh online stream."""
        self._stream = self._spotter.create_stream()

    def process(self, frame: NDArray[np.int16]) -> Mapping[str, float]:
        """Feed one frame; 1.0 for each keyword whose hit was decoded during it."""
        if frame.shape[0] != FRAME_SAMPLES:
            msg = f"expected {FRAME_SAMPLES} samples, got {frame.shape[0]}"
            raise ValueError(msg)
        out = dict.fromkeys(self._wake_words, 0.0)
        self._stream.accept_waveform(SAMPLE_RATE, frame.astype(np.float32) / 32768.0)
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = str(self._spotter.get_result(self._stream) or "").strip()
            if result:
                label = _label(result)
                if label in out:
                    out[label] = 1.0
                self._spotter.reset_stream(self._stream)
        return out

    def close(self) -> None:
        """Drop the stream and spotter."""
        self._stream = None
        self._spotter = None


def _labels_from_file(path: Path) -> list[str]:
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        at = [p for p in line.split() if p.startswith("@")]
        labels.append(
            at[0][1:] if at else _label(" ".join(line.split(":")[0].split("#")[0].split()))
        )
    return labels


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


@engine_registry.register("sherpa_kws")
def create(**config: Any) -> Engine:
    """Factory used by the registry."""
    return SherpaKwsEngine(**config)
