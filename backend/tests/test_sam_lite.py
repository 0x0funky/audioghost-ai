"""
Unit tests for the model-free parts of the SAM-Audio Lite engine.

Run from the repo root:  python -m pytest backend/tests -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers.sam_lite import (  # noqa: E402
    QUALITY_STEPS,
    crossfade_window,
    localize_anchors,
    model_repo_id,
    normalize_anchors,
    overlap_add,
    plan_chunks,
)

SR = 48_000
HOP = 1_920


def test_short_audio_is_one_padded_chunk():
    plan = plan_chunks(num_samples=10 * SR + 7, max_chunk=25 * SR, overlap=2 * SR, hop=HOP)
    assert plan.starts == (0,)
    assert plan.length % HOP == 0
    assert plan.length >= 10 * SR + 7
    assert plan.length - (10 * SR + 7) < HOP
    assert plan.overlap == 0


@pytest.mark.parametrize("seconds", [25.01, 26, 47, 60, 61.3, 265.6, 600])
def test_long_audio_plan_covers_everything_with_equal_chunks(seconds):
    n = int(seconds * SR)
    plan = plan_chunks(n, 25 * SR, 2 * SR, HOP)
    assert plan.num_chunks >= 2
    assert plan.length <= 25 * SR
    assert plan.length % HOP == 0
    assert all(s % HOP == 0 for s in plan.starts)
    assert plan.padded_length >= n
    # every neighbouring pair overlaps by exactly `overlap`
    for a, b in zip(plan.starts, plan.starts[1:]):
        assert a + plan.length - b == plan.overlap
    # no needless extra chunk: dropping one could not cover the audio
    n_minus = plan.num_chunks - 1
    assert n_minus * (25 * SR - plan.overlap) + plan.overlap < n
    # padding is small (< one hop per chunk)
    assert plan.padded_length - n < HOP * plan.num_chunks


def test_overlap_is_capped_for_small_chunks():
    plan = plan_chunks(60 * SR, 5 * SR, 4 * SR, HOP)
    assert plan.overlap <= 2.5 * SR


def test_crossfades_are_complementary():
    fade = 2 * SR
    left = crossfade_window(25 * SR, 0, fade)
    right = crossfade_window(25 * SR, fade, 0)
    total = left[-fade:] + right[:fade]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)
    assert left[0] == 1 and right[-1] == 1


@pytest.mark.parametrize("seconds", [3.3, 25, 61.3, 130])
def test_overlap_add_reconstructs_identical_chunks(seconds):
    n = int(seconds * SR)
    plan = plan_chunks(n, 25 * SR, 2 * SR, HOP)
    signal = torch.randn(2, plan.padded_length)
    chunks = [signal[:, s:s + plan.length] for s in plan.starts]
    rebuilt = overlap_add(chunks, plan)
    assert rebuilt.shape == signal.shape
    assert torch.allclose(rebuilt, signal, atol=1e-5)


def test_localize_anchors_clips_to_chunk():
    anchors = [("+", 10.0, 30.0), ("+", 50.0, 51.0)]
    assert localize_anchors(anchors, 0.0, 23.0) == [("+", 10.0, 23.0)]
    assert localize_anchors(anchors, 21.0, 44.0) == [("+", 0.0, 9.0)]
    assert localize_anchors(anchors, 31.0, 49.0) == []
    assert localize_anchors(None, 0.0, 10.0) == []


def test_normalize_anchors_accepts_legacy_and_flat_forms():
    legacy = [[["+", 1.0, 4.5]]]
    flat = [["+", 1.0, 4.5], ["-", 6, 7]]
    assert normalize_anchors(legacy) == [("+", 1.0, 4.5)]
    assert normalize_anchors(flat) == [("+", 1.0, 4.5), ("-", 6.0, 7.0)]
    assert normalize_anchors([["?", 1, 2], ["+", 3, 3], "junk"]) == []
    assert normalize_anchors(None) == []


def test_model_repo_ids():
    assert model_repo_id("base") == "facebook/sam-audio-base"
    assert model_repo_id("large", "tv") == "facebook/sam-audio-large-tv"
    with pytest.raises(ValueError):
        model_repo_id("huge")


def test_quality_presets_are_ordered():
    assert QUALITY_STEPS["fast"] < QUALITY_STEPS["balanced"] < QUALITY_STEPS["high"]
    assert QUALITY_STEPS["balanced"] == 16  # Meta's default (midpoint, step 2/32)


def test_stem_filenames_are_readable_and_safe():
    from workers.audio_io import stem_filename

    assert stem_filename("My Song (live).mp3", "singing voice", "ghost", "wav") == "My Song (live) [singing voice].wav"
    assert stem_filename("clip.mp4", "crowd noise", "clean", "mp3") == "clip [no crowd noise].mp3"
    assert stem_filename(None, "", "original", "flac") == "audio [original].flac"
    name = stem_filename("訪談 錄音.m4a", 'music: "loud"? <now>', "ghost", "wav")
    assert name == "訪談 錄音 [music loud now].wav"
    assert not any(c in name for c in '\\/:*?"<>|')


def test_xformers_stand_in_when_missing(monkeypatch):
    import importlib.util

    from workers import compat

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: None if name == "xformers" else real_find_spec(name, *a, **k),
    )
    saved = {k: v for k, v in sys.modules.items() if k == "xformers" or k.startswith("xformers.")}
    for key in saved:
        del sys.modules[key]
    try:
        assert compat.install_xformers_stub() is True
        from xformers.ops import AttentionBias, fmha  # what perception_models imports

        assert isinstance(AttentionBias, type)
        with pytest.raises(RuntimeError):
            fmha.memory_efficient_attention(None, None, None)
        assert compat.install_xformers_stub() is False  # idempotent
    finally:
        for key in [k for k in sys.modules if k == "xformers" or k.startswith("xformers.")]:
            del sys.modules[key]
        sys.modules.update(saved)
