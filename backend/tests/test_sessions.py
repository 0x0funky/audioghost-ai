"""
Sessions: storage, "everything else", mixing, exports and the HTTP API.
Synthetic audio only; no model, GPU or Redis needed.

Run from the repo root:  python -m pytest backend/tests -q
"""
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sessions  # noqa: E402
from workers import queue_state  # noqa: E402

SR = sessions.SAMPLE_RATE


def tone(freq, seconds=3.0, amp=0.2):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSION_DIR", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()
    monkeypatch.setattr(queue_state, "_client", None)
    monkeypatch.setattr(queue_state, "_redis", lambda: (_ for _ in ()).throw(ConnectionError("no redis")))
    return tmp_path


def make_session(store, channels=2, name="My Song.mp3"):
    a, b, c = tone(220), tone(330), tone(550)
    mix = a + b + c
    data = np.stack([mix, mix * 0.5], axis=1) if channels == 2 else mix[:, None]
    upload = store / "upload.wav"
    sf.write(str(upload), data, SR, subtype="FLOAT")
    session = sessions.create(upload, name, is_video=False)
    return session, (a, b, c)


def read(path):
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return data


def finish(session, tid, mono_signal):
    sessions.finish_track(session["id"], tid, torch.from_numpy(mono_signal[None, :]), SR, elapsed=1.0)


def test_create_decodes_and_writes_overview(store):
    session, _ = make_session(store)
    folder = sessions.session_dir(session["id"])
    assert (folder / "original.wav").exists()
    assert session["source"]["channels"] == 2
    assert session["source"]["filename"] == "My Song.mp3"
    assert abs(session["source"]["duration"] - 3.0) < 0.01
    peaks = json.loads((folder / "original.peaks.json").read_text())
    assert 400 <= len(peaks["peaks"]) <= 6000
    assert 0.3 < max(peaks["peaks"]) <= 1.0


def test_rest_is_original_minus_tracks_mono_track_on_stereo_source(store):
    session, (a, b, c) = make_session(store)
    sid = session["id"]
    ids = sessions.add_tracks(sid, [{"text": "low tone"}, {"text": "mid tone"}], {"model_size": "small"}, "job-1")
    finish(session, ids[0], a)
    finish(session, ids[1], b)
    data = sessions.load(sid)
    assert [t["status"] for t in data["tracks"]] == ["done", "done"]
    assert data["rest"]["rev"] == 2
    rest = read(sessions.session_dir(sid) / data["rest"]["file"])
    original = read(sessions.session_dir(sid) / "original.wav")
    expected = original - (a + b)[:, None]  # mono tracks are subtracted from both channels
    np.testing.assert_allclose(rest, expected, atol=1e-5)
    # The old revision's files are gone.
    assert not (sessions.session_dir(sid) / "rest.1.wav").exists()


def test_all_tracks_at_unity_rebuild_the_original(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    ids = sessions.add_tracks(sid, [{"text": "low"}, {"text": "mid"}], {}, "job-1")
    finish(session, ids[0], a)
    finish(session, ids[1], b)
    gains = {ids[0]: 1.0, ids[1]: 1.0, "rest": 1.0}
    path, name = sessions.export_mix(sid, gains, "wav")
    original = read(sessions.session_dir(sid) / "original.wav")
    np.testing.assert_allclose(read(path), original, atol=2e-6)  # 24-bit export
    assert name == "My Song [mix].wav"


def test_mix_labels(store):
    session, (a, b, c) = make_session(store)
    sid = session["id"]
    ids = sessions.add_tracks(sid, [{"text": "vocals"}, {"text": "drums"}], {}, "job-1")
    finish(session, ids[0], a)
    finish(session, ids[1], b)
    data = sessions.load(sid)
    assert sessions.mix_label(data, {ids[0]: 1.0}) == "vocals"
    assert sessions.mix_label(data, {ids[1]: 1.0, "rest": 1.0}) == "no vocals"
    assert sessions.mix_label(data, {ids[0]: 0.5, ids[1]: 1.0, "rest": 1.0}) == "mix"


def test_muted_everything_is_an_error(store):
    session, (a, _, _) = make_session(store)
    ids = sessions.add_tracks(session["id"], [{"text": "x"}], {}, "job-1")
    finish(session, ids[0], a)
    with pytest.raises(ValueError):
        sessions.export_mix(session["id"], {ids[0]: 0.0, "rest": 0.0}, "wav")


def test_export_tracks_zip_names(store):
    session, (a, b, _) = make_session(store)
    sid = session["id"]
    ids = sessions.add_tracks(sid, [{"text": "vocals"}, {"text": "vocals"}], {}, "job-1")
    finish(session, ids[0], a)
    finish(session, ids[1], b)
    path, name = sessions.export_tracks(sid, [ids[0], ids[1], "rest"], "flac")
    assert name == "My Song [tracks].zip"
    with zipfile.ZipFile(path) as zf:
        assert sorted(zf.namelist()) == [
            "My Song [everything else].flac", "My Song [vocals 2].flac", "My Song [vocals].flac",
        ]


def test_video_export_needs_a_video(store):
    session, (a, _, _) = make_session(store)
    ids = sessions.add_tracks(session["id"], [{"text": "x"}], {}, "job-1")
    finish(session, ids[0], a)
    with pytest.raises(ValueError):
        sessions.export_video(session["id"], {ids[0]: 1.0})


def test_drop_track_updates_rest(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    ids = sessions.add_tracks(sid, [{"text": "low"}, {"text": "mid"}], {}, "job-1")
    finish(session, ids[0], a)
    finish(session, ids[1], b)
    sessions.drop_track(sid, ids[1])
    sessions.rebuild_rest(sid)
    data = sessions.load(sid)
    assert [t["id"] for t in data["tracks"]] == [ids[0]]
    rest = read(sessions.session_dir(sid) / data["rest"]["file"])[:, 0]
    np.testing.assert_allclose(rest, b + c, atol=1e-5)
    assert not (sessions.session_dir(sid) / f"{ids[1]}.wav").exists()


def test_finish_after_removal_discards_audio(store):
    session, (a, _, _) = make_session(store)
    sid = session["id"]
    [tid] = sessions.add_tracks(sid, [{"text": "x"}], {}, "job-1")
    sessions.drop_track(sid, tid)
    assert sessions.finish_track(sid, tid, torch.from_numpy(a[None]), SR, 1.0) is False
    assert not (sessions.session_dir(sid) / f"{tid}.wav").exists()


def test_list_and_cleanup(store):
    s1, _ = make_session(store, name="old.wav")
    s2, _ = make_session(store, name="new.wav")
    listed = sessions.list_sessions()
    assert {s["id"] for s in listed} == {s1["id"], s2["id"]}
    sessions.update(s1["id"], lambda d: None)
    data = sessions.load(s1["id"])
    data["updated_at"] = time.time() - 48 * 3600
    sessions._write(s1["id"], data)
    assert sessions.cleanup(24) == 1
    assert [s["id"] for s in sessions.list_sessions()] == [s2["id"]]


def test_ids_cannot_escape_the_session_folder(store):
    for bad in ("..", "../x", "0123", "ABCDEF0123456789", "0123456789abcdef/.."):
        with pytest.raises(sessions.SessionNotFound):
            sessions.session_dir(bad)
    for name in ("session.json", "../original.wav", "original.wav/..", "t123.wav", "exports"):
        assert not sessions.SERVABLE.match(name)
    for name in ("original.wav", "t0123abcd.wav", "rest.3.peaks.json", "source.mp4"):
        assert sessions.SERVABLE.match(name)


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(store, monkeypatch):
    from fastapi.testclient import TestClient

    import api.sessions as api_sessions
    import main
    from workers import tasks

    queued = []
    monkeypatch.setattr(api_sessions, "read_hf_token", lambda: "hf_test")
    monkeypatch.setattr(api_sessions.queue_state, "enqueue", lambda job: None)
    monkeypatch.setattr(api_sessions.queue_state, "jobs_ahead", lambda job: 0)
    monkeypatch.setattr(
        tasks.separate_session_task, "apply_async",
        lambda args, task_id: queued.append((args, task_id)),
    )
    test_client = TestClient(main.app)
    test_client.queued = queued
    return test_client


def upload(client, seconds=2.0):
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone(440, seconds), tone(660, seconds)], axis=1), SR, format="WAV")
    return client.post("/api/sessions", files={"file": ("clip.wav", buf.getvalue(), "audio/wav")})


def test_api_upload_separate_and_remove(client):
    res = upload(client)
    assert res.status_code == 201, res.text
    session = res.json()
    sid = session["id"]
    assert session["source"]["audio_url"].endswith("/files/original.wav")
    assert session["source"]["video_url"] is None

    res = client.post(f"/api/sessions/{sid}/separate", json={
        "prompts": [{"text": "  violin  ", "anchors": [["+", 0.5, 1.0], ["-", 1.9, 9.0], ["+", 1, 1.01]]},
                    {"text": "piano"}],
        "model_size": "small", "quality": "fast",
    })
    assert res.status_code == 202, res.text
    body = res.json()
    assert len(client.queued) == 1
    args, task_id = client.queued[0]
    assert task_id == body["job_id"] and args[1] == body["track_ids"]
    tracks = body["session"]["tracks"]
    assert [t["prompt"] for t in tracks] == ["violin", "piano"]
    assert tracks[0]["anchors"] == [["+", 0.5, 1.0], ["-", 1.9, 2.0]]  # clamped; too short dropped
    assert all(t["status"] == "queued" and "job_id" not in t for t in tracks)

    res = client.delete(f"/api/sessions/{sid}/tracks/{body['track_ids'][0]}")
    assert res.json() == {"removed": body["track_ids"][0]}
    assert [t["prompt"] for t in client.get(f"/api/sessions/{sid}").json()["tracks"]] == ["piano"]

    assert client.get("/api/sessions").json()["sessions"][0]["id"] == sid
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_api_rejects_bad_requests(client):
    sid = upload(client).json()["id"]
    assert client.post(f"/api/sessions/{sid}/separate", json={"prompts": []}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/separate",
                       json={"prompts": [{"text": "x"}], "model_size": "huge"}).status_code == 400
    assert client.get(f"/api/sessions/{sid}/files/session.json").status_code == 404
    assert client.get("/api/sessions/zzzz/files/original.wav").status_code == 404
    bad = client.post("/api/sessions", files={"file": ("x.mp3", b"not audio", "audio/mpeg")})
    assert bad.status_code == 400


def test_api_files_support_range_requests(client):
    sid = upload(client).json()["id"]
    res = client.get(f"/api/sessions/{sid}/files/original.wav", headers={"Range": "bytes=0-99"})
    assert res.status_code == 206 and len(res.content) == 100
    assert "immutable" in res.headers["cache-control"]


def test_api_export_mix(client):
    sid = upload(client).json()["id"]
    [tid] = sessions.add_tracks(sid, [{"text": "high"}], {}, "job-1")
    sessions.finish_track(sid, tid, torch.from_numpy(tone(660, 2.0)[None]), SR, 1.0)
    res = client.post(f"/api/sessions/{sid}/export",
                      json={"kind": "mix", "format": "flac", "gains": {tid: 0.0, "rest": 1.0}})
    assert res.status_code == 200, res.text
    assert "clip%20%5Bno%20high%5D.flac" in res.headers["content-disposition"] \
        or 'clip [no high].flac' in res.headers["content-disposition"]
    assert res.content[:4] == b"fLaC"
    res = client.post(f"/api/sessions/{sid}/export", json={"kind": "video", "gains": {"rest": 1.0}})
    assert res.status_code == 400
