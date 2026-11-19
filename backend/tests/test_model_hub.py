"""
Where the SAM-Audio weights come from (no Hugging Face account for AudioGhost's
copies), the SAM License gate, and the license API. No model or network needed.

Run from the repo root:  python -m pytest backend/tests -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model_license  # noqa: E402
from workers import model_files  # noqa: E402


@pytest.fixture
def cache(monkeypatch):
    """A fake local cache: the set of (repo, filename) pairs on this computer."""
    files = set()
    monkeypatch.setattr(model_files, "cached", lambda repo, name: f"/cache/{repo}/{name}" if (repo, name) in files else None)
    return files


def test_meta_checkpoint_in_the_cache_is_used_as_is(cache):
    assert model_files.model_repo_id("large") == "audioghostai/sam-audio-large"
    cache.add(("facebook/sam-audio-large", "checkpoint.pt"))
    assert model_files.model_repo_id("large") == "facebook/sam-audio-large"
    assert model_files.model_repo_id("large", "tv") == "facebook/sam-audio-large-tv"
    with pytest.raises(ValueError):
        model_files.model_repo_id("huge")


def test_each_precision_loads_its_own_file():
    ours = "audioghostai/sam-audio-base"
    assert model_files.weight_files(ours, "fp16") == [
        (ours, "config.json"), (ours, "model.fp16.safetensors"), ("audioghostai/sam-audio-shared", "audio_codec.safetensors")]
    assert model_files.weight_files(ours, "bf16")[1] == (ours, "model.fp32.safetensors")
    assert model_files.weight_files(ours, "fp32")[1] == (ours, "model.fp32.safetensors")
    meta = "facebook/sam-audio-base"
    assert model_files.weight_files(meta) == [(meta, "config.json"), (meta, "checkpoint.pt")]
    assert model_files.needs_token(meta) and not model_files.needs_token(ours)


def test_what_a_separation_still_needs(cache):
    need = model_files.missing_requirement
    # Nothing downloaded yet: AudioGhost's copy needs the license, no token.
    assert need("small", "default", "fp16", token=None, license_accepted=False) == "license"
    assert need("small", "default", "fp16", token=None, license_accepted=True) is None
    # Meta's -tv checkpoints need a token, the license doesn't matter.
    assert need("small", "tv", "fp16", token=None, license_accepted=True) == "token"
    assert need("small", "tv", "fp16", token="hf_x", license_accepted=False) is None
    # Already on this computer: nothing at all.
    cache.update({("audioghostai/sam-audio-small", "config.json"), ("audioghostai/sam-audio-small", "model.fp16.safetensors"),
                  ("audioghostai/sam-audio-shared", "audio_codec.safetensors")})
    assert need("small", "default", "fp16", token=None, license_accepted=False) is None
    # ...but full FP32 needs the fp32 file, which isn't there.
    assert need("small", "default", "fp32", token=None, license_accepted=False) == "license"


def test_local_model_dir_is_used_without_download(tmp_path, monkeypatch):
    folder = tmp_path / "sam-audio-small"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    monkeypatch.setattr(model_files, "LOCAL_MODEL_DIR", str(tmp_path))
    assert model_files.cached("audioghostai/sam-audio-small", "config.json") == str(folder / "config.json")


def test_license_is_accepted_once_per_version(tmp_path, monkeypatch):
    monkeypatch.setattr(model_license, "ACCEPTED_FILE", tmp_path / "accepted.json")
    assert not model_license.accepted()
    model_license.accept()
    assert model_license.accepted()
    monkeypatch.setattr(model_license, "VERSION", "2099-01-01")  # a new license text asks again
    assert not model_license.accepted()
    assert model_license.text().startswith("SAM License")


def test_license_api_and_separate_gate(tmp_path, monkeypatch):
    import io

    import numpy as np
    import soundfile as sf
    from fastapi.testclient import TestClient

    import api.sessions as api_sessions
    import main
    import sessions

    monkeypatch.setattr(sessions, "SESSION_DIR", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()
    monkeypatch.setattr(model_license, "ACCEPTED_FILE", tmp_path / "accepted.json")
    import api.auth as api_auth

    monkeypatch.setattr(api_sessions, "read_hf_token", lambda: None)
    monkeypatch.setattr(api_auth, "get_saved_token", lambda: None)  # no real token, no network
    monkeypatch.setattr(model_files, "cached", lambda repo, name: None)  # nothing downloaded
    client = TestClient(main.app)

    info = client.get("/api/models/license").json()
    assert info["accepted"] is False and info["text"].startswith("SAM License") and info["version"]

    buf = io.BytesIO()
    sf.write(buf, np.zeros((48000, 1), dtype=np.float32), 48000, format="WAV")
    sid = client.post("/api/sessions", files={"file": ("a.wav", buf.getvalue(), "audio/wav")}).json()["id"]
    body = {"prompts": [{"text": "speech"}], "model_size": "small", "quality": "fast", "variant": "default"}
    res = client.post(f"/api/sessions/{sid}/separate", json=body)
    assert res.status_code == 409 and "license" in res.json()["detail"].lower()
    res = client.post(f"/api/sessions/{sid}/separate", json={**body, "variant": "tv"})
    assert res.status_code == 401 and "hugging face" in res.json()["detail"].lower()

    assert client.post("/api/models/license/accept").json()["accepted"] is True
    assert client.get("/api/models/license").json()["accepted"] is True
    assert client.get("/api/auth/status").json()["license_accepted"] is True
