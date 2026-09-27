"""Forced alignment of reference transcripts to audio.

Two strategies:

* :func:`align_reference` uses torchaudio's ``MMS_FA`` pipeline (wav2vec2 trained on
  1,100+ languages, CTC forced alignment). Its tokenizer covers lower-case ``a``-``z``
  and the apostrophe, so text is romanised for the aligner only: lower-cased,
  punctuation removed, accents stripped via NFKD, ``ß`` -> ``ss``. Word texts keep
  their original spelling. Each reference segment is aligned inside its own time
  window; segments longer than 60 s are split at punctuation first.
* :func:`align_uniform` spreads the words evenly across the segment. It is a
  fallback for environments without PyTorch; word timings are then only accurate to
  the utterance level and the transcript origin stays ``"reference"``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from wakewordworld.sources.spec import Language
from wakewordworld.transcribe.backends.base import BackendUnavailable
from wakewordworld.transcribe.schema import Segment, Word

__all__ = ["align_reference", "align_uniform", "romanise", "split_long_segments", "tokenize"]

MAX_SEGMENT_S = 60.0
_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")
_TOKEN_RE = re.compile("[^\\W_]+(?:['\u2019\\-][^\\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Split text into word tokens, keeping internal apostrophes and hyphens."""
    return _TOKEN_RE.findall(text)


def romanise(word: str) -> str:
    """Reduce a word to the MMS_FA alphabet (a-z and apostrophe)."""
    w = word.lower().replace("ß", "ss").replace("æ", "ae").replace("œ", "oe").replace("ø", "o")
    w = unicodedata.normalize("NFKD", w)
    w = "".join(ch for ch in w if not unicodedata.combining(ch))
    w = w.replace("\u2019", "'")
    return re.sub(r"[^a-z']", "", w)


def split_long_segments(segments: list[Segment], max_s: float = MAX_SEGMENT_S) -> list[Segment]:
    """Split segments longer than ``max_s`` at sentence punctuation, proportionally."""
    out: list[Segment] = []
    for seg in segments:
        dur = seg.end - seg.start
        if dur <= max_s:
            out.append(seg)
            continue
        pieces = [p for p in _SPLIT_RE.split(seg.text) if p.strip()]
        if len(pieces) < 2:
            out.append(seg)
            continue
        total_chars = sum(len(p) for p in pieces)
        cursor = seg.start
        for piece in pieces:
            share = dur * len(piece) / total_chars
            out.append(
                Segment(
                    start=cursor,
                    end=min(seg.end, cursor + share),
                    text=piece.strip(),
                    speaker=seg.speaker,
                    is_music=seg.is_music,
                )
            )
            cursor += share
    return out


def align_uniform(segments: list[Segment]) -> list[Segment]:
    """Distribute words evenly over each segment (no acoustic evidence)."""
    out: list[Segment] = []
    for seg in segments:
        tokens = tokenize(seg.text)
        if not tokens:
            out.append(seg)
            continue
        step = (seg.end - seg.start) / len(tokens)
        words = [
            Word(
                text=tok,
                start=seg.start + i * step,
                end=seg.start + (i + 1) * step,
                confidence=None,
                speaker=seg.speaker,
            )
            for i, tok in enumerate(tokens)
        ]
        out.append(seg.model_copy(update={"words": words}))
    return out


@dataclass
class _Mms:
    bundle: Any
    model: Any
    tokenizer: Any
    aligner: Any
    torch: Any
    sample_rate: int


_MMS: _Mms | None = None


def _load_mms() -> _Mms:
    global _MMS
    if _MMS is not None:
        return _MMS
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA
    except ImportError as exc:
        msg = "forced alignment needs torch and torchaudio; install with: uv sync --extra align"
        raise BackendUnavailable(msg) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MMS_FA.get_model(with_star=False).to(device).eval()
    _MMS = _Mms(
        bundle=MMS_FA,
        model=model,
        tokenizer=MMS_FA.get_tokenizer(),
        aligner=MMS_FA.get_aligner(),
        torch=torch,
        sample_rate=int(MMS_FA.sample_rate),
    )
    del torchaudio
    return _MMS


def align_reference(
    audio_path: Path, segments: list[Segment], language: Language | None = None
) -> list[Segment]:
    """Attach word timings to reference segments with MMS_FA.

    Args:
        audio_path: 16 kHz mono file.
        segments: Utterances with ``start``/``end``/``text``.
        language: Unused by MMS_FA (romanised input), kept for API symmetry.

    Raises:
        BackendUnavailable: When torch/torchaudio are missing.
    """
    del language
    mms = _load_mms()
    info = sf.info(str(audio_path))
    if info.samplerate != mms.sample_rate:
        msg = f"{audio_path.name}: expected {mms.sample_rate} Hz, got {info.samplerate}"
        raise ValueError(msg)
    out: list[Segment] = []
    with sf.SoundFile(str(audio_path)) as fh:
        for seg in split_long_segments(segments):
            tokens = tokenize(seg.text)
            roman = [romanise(t) for t in tokens]
            keep = [i for i, r in enumerate(roman) if r]
            if not keep:
                out.append(seg)
                continue
            start_frame = int(seg.start * info.samplerate)
            n_frames = max(int((seg.end - seg.start) * info.samplerate), 1)
            fh.seek(min(start_frame, max(info.frames - 1, 0)))
            audio = fh.read(n_frames, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if audio.size < info.samplerate // 10:
                out.append(align_uniform([seg])[0])
                continue
            try:
                spans = _align_window(mms, audio, [roman[i] for i in keep])
            except Exception:
                out.append(align_uniform([seg])[0])
                continue
            words: list[Word] = []
            frame_s = (audio.size / info.samplerate) / max(spans["n_frames"], 1)
            span_iter = iter(spans["words"])
            last_end = seg.start
            for i, tok in enumerate(tokens):
                if i in keep:
                    s_frame, e_frame, score = next(span_iter)
                    w_start = seg.start + s_frame * frame_s
                    w_end = seg.start + max(e_frame, s_frame + 1) * frame_s
                else:
                    w_start, w_end, score = last_end, last_end, None
                words.append(
                    Word(
                        text=tok,
                        start=w_start,
                        end=max(w_end, w_start),
                        confidence=score,
                        speaker=seg.speaker,
                    )
                )
                last_end = words[-1].end
            out.append(seg.model_copy(update={"words": words}))
    return out


def _align_window(mms: _Mms, audio: np.ndarray, roman_words: list[str]) -> dict[str, Any]:
    torch = mms.torch
    device = next(mms.model.parameters()).device
    waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).to(device)
    with torch.inference_mode():
        emission, _ = mms.model(waveform)
    emission = emission[0].cpu()
    token_spans = mms.aligner(emission, mms.tokenizer(roman_words))
    words: list[tuple[int, int, float | None]] = []
    for spans in token_spans:
        score = float(sum(s.score * (s.end - s.start) for s in spans)) / max(
            sum(s.end - s.start for s in spans), 1
        )
        words.append((int(spans[0].start), int(spans[-1].end), min(1.0, max(0.0, score))))
    return {"words": words, "n_frames": int(emission.shape[0])}
