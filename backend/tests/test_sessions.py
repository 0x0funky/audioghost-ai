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


def as_audio(x):
    x = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(x[None, :] if x.ndim == 1 else x)


def finish(session, tid, target, without, rest=None, chain_from="current"):
    """
    Finish a take with given model outputs, the way the worker does: a chain step (``rest``) only when
    the take's sound is new to a chain that already holds another sound. Returns True unless the track
    was removed meanwhile.
    """
    sid = session["id"]
    data = sessions.load(sid)
    step = sessions.chain_need(data, sessions.find_track(data, tid)) == "step" if sessions.find_track(data, tid) else False
    if chain_from == "current":
        chain_from = data.get("rest") if step else None
    outcome = sessions.finish_track(sid, tid, as_audio(target), as_audio(without), SR, 1.0,
                                    rest=as_audio(rest) if step and rest is not None else None, chain_from=chain_from)
    return outcome != "gone"


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


def test_finish_publishes_track_without_and_chain(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    folder = sessions.session_dir(sid)
    ids = sessions.add_tracks(sid, [{"text": "low"}, {"text": "mid"}], {}, "job-1")
    assert finish(session, ids[0], a, b + c)
    data = sessions.load(sid)
    assert data["rest"]["rev"] == 1 and data["rest"]["tracks"] == [ids[0]]
    assert data["tracks"][0]["without"] == f"{ids[0]}.without.wav"
    assert data["rest"]["file"] == f"{ids[0]}.without.wav"  # one sound: its own "without", no copy
    np.testing.assert_allclose(read(folder / f"{ids[0]}.without.wav")[:, 0], b + c, atol=1e-6)
    assert finish(session, ids[1], b, a + c, c)  # the chain step removed "mid" from "everything but low"
    data = sessions.load(sid)
    assert [t["status"] for t in data["tracks"]] == ["done", "done"]
    assert data["rest"]["rev"] == 2 and data["rest"]["tracks"] == ids
    assert data["rest"]["file"] == "rest.2.wav"
    np.testing.assert_allclose(read(folder / data["rest"]["file"])[:, 0], c, atol=1e-6)
    assert (folder / f"{ids[0]}.without.wav").exists()  # still that track's own "without"


def test_unity_mix_plays_the_original(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    ids = sessions.add_tracks(sid, [{"text": "low"}, {"text": "mid"}], {}, "job-1")
    finish(session, ids[0], a, b + c, b + c)
    finish(session, ids[1], b, a + c, c)
    path, name = sessions.export_mix(sid, {ids[0]: 1.0, ids[1]: 1.0, "rest": 1.0}, "wav")
    assert sf.info(str(path)).subtype == "PCM_24"  # not the float working file
    np.testing.assert_allclose(read(path), read(sessions.session_dir(sid) / "original.wav"), atol=2e-6)
    assert name == "My Song [mix].wav"


def test_mix_plan_plays_model_output(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    ids = sessions.add_tracks(sid, [{"text": "low"}, {"text": "mid"}], {}, "job-1")
    finish(session, ids[0], a, b + c, b + c)
    finish(session, ids[1], b, a + c, c)
    data = sessions.load(sid)

    def plan(**changes):
        gains = {ids[0]: 1.0, ids[1]: 1.0, "rest": 1.0}
        gains.update({ids[int(k[1])] if k in ("t0", "t1") else k: v for k, v in changes.items()})
        return [(path.name, round(g, 3)) for path, g in sessions.mix_plan(data, gains)]

    assert plan() == [("original.wav", 1.0)]
    assert plan(t0=0.0) == [(f"{ids[0]}.without.wav", 1.0)]
    assert plan(t1=0.5) == [(f"{ids[1]}.without.wav", 1.0), (f"{ids[1]}.wav", 0.5)]
    assert plan(t0=0.0, t1=0.5) == [(data["rest"]["file"], 1.0), (f"{ids[1]}.wav", 0.5)]
    assert plan(rest=0.0) == [(f"{ids[0]}.wav", 1.0), (f"{ids[1]}.wav", 1.0)]
    # Muting one sound plays exactly the model's own "without".
    path, _ = sessions.export_mix(sid, {ids[0]: 0.0, ids[1]: 1.0, "rest": 1.0}, "wav")
    np.testing.assert_allclose(read(path)[:, 0], b + c, atol=2e-5)


def test_mix_labels(store):
    session, (a, b, c) = make_session(store)
    sid = session["id"]
    ids = sessions.add_tracks(sid, [{"text": "vocals"}, {"text": "drums"}], {}, "job-1")
    finish(session, ids[0], a, b + c, b + c)
    finish(session, ids[1], b, a + c, c)
    data = sessions.load(sid)
    assert sessions.mix_label(data, {ids[0]: 1.0}) == "vocals"
    assert sessions.mix_label(data, {ids[1]: 1.0, "rest": 1.0}) == "no vocals"
    assert sessions.mix_label(data, {ids[0]: 0.5, ids[1]: 1.0, "rest": 1.0}) == "mix"


def test_muted_everything_is_an_error(store):
    session, (a, b, c) = make_session(store)
    ids = sessions.add_tracks(session["id"], [{"text": "x"}], {}, "job-1")
    finish(session, ids[0], a, b + c, b + c)
    with pytest.raises(ValueError):
        sessions.export_mix(session["id"], {ids[0]: 0.0, "rest": 0.0}, "wav")


def test_export_tracks_zip_names(store):
    session, (a, b, c) = make_session(store)
    sid = session["id"]
    ids = sessions.add_tracks(sid, [{"text": "vocals"}, {"text": "vocals"}], {}, "job-1")
    finish(session, ids[0], a, b + c, b + c)
    finish(session, ids[1], b, a + c, c)
    path, name = sessions.export_tracks(sid, [ids[0], ids[1], "rest"], "flac")
    assert name == "My Song [tracks].zip"
    with zipfile.ZipFile(path) as zf:
        assert sorted(zf.namelist()) == [
            "My Song [everything else].flac", "My Song [vocals 2].flac", "My Song [vocals].flac",
        ]


def test_video_export_needs_a_video(store):
    session, (a, b, c) = make_session(store)
    ids = sessions.add_tracks(session["id"], [{"text": "x"}], {}, "job-1")
    finish(session, ids[0], a, b + c, b + c)
    with pytest.raises(ValueError):
        sessions.export_video(session["id"], {ids[0]: 1.0})


def test_removing_a_done_track_puts_its_sound_back(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    folder = sessions.session_dir(sid)
    ids = sessions.add_tracks(sid, [{"text": "low"}, {"text": "mid"}], {}, "job-1")
    finish(session, ids[0], a, b + c, b + c)
    finish(session, ids[1], b, a + c, c)
    sessions.remove_done_track(sid, ids[0])
    data = sessions.load(sid)
    assert [t["id"] for t in data["tracks"]] == [ids[1]]
    assert data["rest"]["tracks"] == [ids[1]]
    # One sound left: "everything else" is exactly that sound's own "without" again.
    assert data["rest"]["file"] == f"{ids[1]}.without.wav"
    np.testing.assert_allclose(read(folder / data["rest"]["file"])[:, 0], a + c, atol=1e-5)
    for leftover in (f"{ids[0]}.wav", f"{ids[0]}.without.wav", f"{ids[0]}.deleted.wav", "rest.2.wav"):
        assert not (folder / leftover).exists()
    sessions.remove_done_track(sid, ids[1])  # nothing separated any more: no "everything else"
    assert sessions.load(sid)["rest"] is None


def test_everything_else_never_reuses_a_file_name(store):
    """
    The files are served as immutable, so a name must never come back with other
    audio. Separate, remove, separate again: "everything else" used to be
    rest.1.wav both times, and the app kept playing the first one from its cache.
    """
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    folder = sessions.session_dir(sid)
    seen = []
    for attempt in range(3):
        tid = sessions.add_tracks(sid, [{"text": "baby crying"}], {}, f"job-{attempt}")[0]
        finish(session, tid, a, b + c)
        rest = sessions.load(sid)["rest"]
        assert rest["file"] == f"{tid}.without.wav"  # a single sound: a name of its own take
        seen.append((rest["rev"], rest["file"]))
        sessions.remove_done_track(sid, tid)  # nothing left: "everything else" goes away
        assert sessions.load(sid)["rest"] is None
    assert [rev for rev, _ in seen] == [1, 2, 3] and len({name for _, name in seen}) == 3

    # The chain gets a new name every time too, after the highest revision so far.
    ids = sessions.add_tracks(sid, [{"text": "low"}, {"text": "mid"}, {"text": "high"}], {}, "job-9")
    finish(session, ids[0], a, b + c)
    finish(session, ids[1], b, a + c, c)
    assert sessions.load(sid)["rest"]["file"] == "rest.5.wav"
    finish(session, ids[2], c, a + b, np.zeros_like(c))
    assert sessions.load(sid)["rest"]["file"] == "rest.6.wav"
    sessions.remove_done_track(sid, ids[0])  # its sound goes back in: everything else + the track
    assert sessions.load(sid)["rest"]["file"] == "rest.7.wav"
    np.testing.assert_allclose(read(folder / "rest.7.wav")[:, 0], a, atol=1e-5)
    assert sorted(p.name for p in folder.glob("rest.*.wav")) == ["rest.7.wav"]


def test_finish_after_removal_discards_audio(store):
    session, (a, _, _) = make_session(store)
    sid = session["id"]
    [tid] = sessions.add_tracks(sid, [{"text": "x"}], {}, "job-1")
    sessions.drop_track(sid, tid)
    assert finish(session, tid, a, a, a) is False
    for leftover in (f"{tid}.wav", f"{tid}.without.wav"):
        assert not (sessions.session_dir(sid) / leftover).exists()


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


def test_mono_track_on_stereo_source_gets_its_stereo_image(store):
    a, b = tone(220), tone(330)
    upload = store / "panned.wav"
    sf.write(str(upload), np.stack([a + b, b], axis=1), SR, subtype="FLOAT")  # a hard left, b centre
    session = sessions.create(upload, "panned.wav", is_video=False)
    sid = session["id"]
    ids = sessions.add_tracks(sid, [{"text": "left"}, {"text": "centre"}], {}, "job-1")
    finish(session, ids[0], 0.5 * a, b, b)  # what mono separation finds in the downmix
    finish(session, ids[1], b, 0.5 * a, np.zeros_like(a))
    folder = sessions.session_dir(sid)
    np.testing.assert_allclose(read(folder / f"{ids[0]}.wav"), np.stack([a, 0 * a], axis=1), atol=1e-3)
    assert read(folder / f"{ids[1]}.wav").shape[1] == 1  # centred: stays mono


def test_interrupted_jobs_fail_their_tracks(store):
    session, _ = make_session(store, channels=1)
    sid = session["id"]
    crashed = sessions.add_tracks(sid, [{"text": "a"}, {"text": "b"}], {}, "job-crashed")
    waiting = sessions.add_tracks(sid, [{"text": "c"}], {}, "job-waiting")
    sessions.set_track(sid, crashed[0], status="running", progress=0.4)
    assert sessions.recover_interrupted() == 2
    tracks = {t["id"]: t for t in sessions.load(sid)["tracks"]}
    assert [tracks[t]["status"] for t in crashed] == ["failed", "failed"]
    assert tracks[crashed[1]]["error"] == sessions.INTERRUPTED
    assert tracks[waiting[0]]["status"] == "queued"  # its job is still in the queue
    # A job that was loading its model (nothing running yet) is known from Redis.
    assert sessions.recover_interrupted({"job-waiting"}) == 1


def test_cleanup_removes_abandoned_running_sessions(store):
    session, _ = make_session(store, channels=1)
    sid = session["id"]
    [tid] = sessions.add_tracks(sid, [{"text": "a"}], {}, "job-1")
    sessions.set_track(sid, tid, status="running")
    data = sessions.load(sid)
    data["updated_at"] = time.time() - 48 * 3600
    sessions._write(sid, data)
    assert sessions.cleanup(24) == 1
    assert sessions.list_sessions() == []


def test_chain_step_puts_back_a_track_removed_meanwhile(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    folder = sessions.session_dir(sid)
    ids = sessions.add_tracks(sid, [{"text": "low"}, {"text": "mid"}, {"text": "high"}], {}, "job-1")
    finish(session, ids[0], a, b + c)
    finish(session, ids[1], b, a + c, c)
    chain_from = dict(sessions.load(sid)["rest"])  # the "high" step starts from "everything but low and mid"
    sessions.set_track(sid, ids[2], status="running")
    sessions.remove_done_track(sid, ids[0])  # removed while "high" runs
    assert (folder / f"{ids[0]}.deleted.wav").exists()
    outcome = sessions.finish_track(sid, ids[2], as_audio(c), as_audio(a + b), SR, 1.0,
                                    rest=as_audio(np.zeros_like(c)), chain_from=chain_from)
    assert outcome == "done"
    data = sessions.load(sid)
    assert data["rest"]["tracks"] == [ids[1], ids[2]]
    np.testing.assert_allclose(read(folder / data["rest"]["file"])[:, 0], a, atol=1e-5)  # "low" is back in
    sessions.purge_deleted(sid)
    assert not (folder / f"{ids[0]}.deleted.wav").exists()


def test_cleanup_if_due_is_throttled(store, monkeypatch):
    runs = []
    monkeypatch.setattr(sessions, "cleanup", lambda hours: runs.append(hours) or 0)
    monkeypatch.setattr(sessions, "_last_cleanup", {"time": None})
    sessions.cleanup_if_due(24)
    sessions.cleanup_if_due(24)
    assert runs == [24]


def test_ids_cannot_escape_the_session_folder(store):
    for bad in ("..", "../x", "0123", "ABCDEF0123456789", "0123456789abcdef/.."):
        with pytest.raises(sessions.SessionNotFound):
            sessions.session_dir(bad)
    for name in ("session.json", "../original.wav", "original.wav/..", "t123.wav", "exports",
                 "t0123abcd.deleted.wav", "rest-build-1234abcd.wav"):
        assert not sessions.SERVABLE.match(name)
    for name in ("original.wav", "t0123abcd.wav", "t0123abcd.without.wav", "t0123abcd.without.peaks.json",
                 "rest.3.peaks.json", "source.mp4"):
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
    # Whatever models this machine has cached: the license counts as accepted.
    monkeypatch.setattr(api_sessions.model_license, "accepted", lambda: True)
    monkeypatch.setattr(api_sessions.queue_state, "enqueue", lambda job: None)
    monkeypatch.setattr(api_sessions.queue_state, "jobs_ahead", lambda job: 0)
    for task in (tasks.separate_session_task, tasks.rebuild_rest_task):
        monkeypatch.setattr(task, "apply_async", lambda args, task_id: queued.append((args, task_id)))
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
    assert args[2]["cleanup"] is False  # "Clean up leftovers" is off unless asked for

    res = client.delete(f"/api/sessions/{sid}/tracks/{body['track_ids'][0]}")
    assert res.json() == {"removed": body["track_ids"][0]}
    assert [t["prompt"] for t in client.get(f"/api/sessions/{sid}").json()["tracks"]] == ["piano"]

    assert client.get("/api/sessions").json()["sessions"][0]["id"] == sid
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_api_passes_clean_up_leftovers_to_the_worker(client):
    sid = upload(client).json()["id"]
    res = client.post(f"/api/sessions/{sid}/separate", json={"prompts": [{"text": "baby crying"}], "cleanup": True})
    assert res.status_code == 202, res.text
    args, _ = client.queued[0]
    assert args[2]["cleanup"] is True
    assert res.json()["session"]["tracks"][0]["settings"]["cleanup"] is True


def test_api_removes_a_running_track_nobody_is_running(client, monkeypatch):
    sid = upload(client).json()["id"]
    [tid] = sessions.add_tracks(sid, [{"text": "x"}], {}, "job-1")
    sessions.set_track(sid, tid, status="running")
    import api.sessions as api_sessions

    # A live worker running this job: cooperative cancel.
    monkeypatch.setattr(api_sessions.queue_state, "worker_state", lambda: "ready")
    monkeypatch.setattr(api_sessions.queue_state, "running_task", lambda: "job-1")
    assert client.delete(f"/api/sessions/{sid}/tracks/{tid}").json() == {"cancelling": tid}
    # The worker stopped (no heartbeat): removed right away.
    monkeypatch.setattr(api_sessions.queue_state, "worker_state", lambda: None)
    assert client.delete(f"/api/sessions/{sid}/tracks/{tid}").json() == {"removed": tid}
    assert client.get(f"/api/sessions/{sid}").json()["tracks"] == []


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
    finish({"id": sid}, tid, tone(660, 2.0), tone(440, 2.0))
    track = client.get(f"/api/sessions/{sid}").json()["tracks"][0]
    assert track["without_url"].endswith(f"/files/{tid}.without.wav")
    assert client.get(track["without_url"]).status_code == 200
    res = client.post(f"/api/sessions/{sid}/export",
                      json={"kind": "mix", "format": "flac", "gains": {tid: 0.0, "rest": 1.0}})
    assert res.status_code == 200, res.text
    assert "clip%20%5Bno%20high%5D.flac" in res.headers["content-disposition"] \
        or 'clip [no high].flac' in res.headers["content-disposition"]
    assert res.content[:4] == b"fLaC"
    res = client.post(f"/api/sessions/{sid}/export", json={"kind": "video", "gains": {"rest": 1.0}})
    assert res.status_code == 400
