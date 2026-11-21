"""
Batches: one recipe (which sounds to separate, how to mix them, what to export)
applied to many files.

A batch is a small JSON file in BATCH_DIR. Each of its files is an ordinary
session, separated through the normal queue, so every result can be opened and
adjusted like any other. The batch only remembers the recipe and which session
belongs to which file, and turns the recipe's mix into gains for each session.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import sessions
from settings import OUTPUT_DIR
from workers import queue_state

BATCH_DIR = Path(os.environ.get("AUDIOGHOST_BATCH_DIR", OUTPUT_DIR / "batches"))
_BATCH_ID = re.compile(r"^b[0-9a-f]{12}$")


class BatchNotFound(KeyError):
    pass


def _path(bid: str) -> Path:
    if not _BATCH_ID.match(bid or ""):
        raise BatchNotFound(bid)
    return BATCH_DIR / f"{bid}.json"


def load(bid: str) -> dict:
    try:
        return json.loads(_path(bid).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BatchNotFound(bid) from None


def _write(data: dict) -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(data["id"])
    tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    sessions._replace(tmp, path)


def create(name: str, recipe: dict) -> dict:
    """``recipe``: ``{"prompts": [{"text", "gain", "seed"}], "rest_gain", "settings", "kind", "format"}``."""
    now = time.time()
    data = {"id": "b" + uuid.uuid4().hex[:12], "name": name.strip()[:120], "created_at": now, "updated_at": now,
            "recipe": recipe, "items": []}
    _write(data)
    return data


def add_item(bid: str, session_id: str, filename: str, sounds: List[str]) -> dict:
    """Record a file of the batch: its session and, in recipe order, the sounds queued for it."""
    with queue_state.redis_lock(f"batch:{bid}"):
        data = load(bid)
        data["items"].append({"session": session_id, "filename": filename, "sounds": list(sounds)})
        data["updated_at"] = time.time()
        _write(data)
        return data


def delete(bid: str, with_sessions: bool = False) -> None:
    data = load(bid)
    if with_sessions:
        for item in data["items"]:
            try:
                sessions.delete(item["session"])
            except sessions.SessionNotFound:
                pass
    _path(bid).unlink(missing_ok=True)


def list_batches(limit: int = 50) -> List[dict]:
    out = []
    if not BATCH_DIR.exists():
        return out
    for path in BATCH_DIR.glob("b*.json"):
        try:
            out.append(present(json.loads(path.read_text(encoding="utf-8")), brief=True))
        except Exception:  # noqa: BLE001 - half-written or foreign file
            continue
    out.sort(key=lambda b: b["created_at"], reverse=True)
    return out[:limit]


def _item_state(item: dict) -> dict:
    """Where a file of the batch stands, from its session's tracks."""
    base = {"session": item["session"], "filename": item["filename"]}
    try:
        data = sessions.load(item["session"])
    except sessions.SessionNotFound:
        return dict(base, status="missing", progress=0.0, message="", error="This file's session was deleted.")
    takes = {sessions.sound_of(t): t for t in sessions.active_takes(data)}
    states = []
    for sound in item["sounds"]:
        rows = [t for t in data["tracks"] if sessions.sound_of(t) == sound]
        if sound in takes:
            states.append(("done", 1.0, "", None))
        elif any(t["status"] == "running" for t in rows):
            track = next(t for t in rows if t["status"] == "running")
            states.append(("running", float(track.get("progress") or 0.0), track.get("message") or "", None))
        elif any(t["status"] == "queued" for t in rows):
            track = next(t for t in rows if t["status"] == "queued")
            states.append(("queued", 0.0, track.get("message") or "", None))
        elif rows:
            states.append(("failed", 0.0, "", rows[-1].get("error") or "Separation failed"))
        else:
            states.append(("failed", 0.0, "", "This sound was removed from the session."))
    kinds = [s[0] for s in states]
    progress = sum(s[1] for s in states) / max(1, len(states))
    if "failed" in kinds:
        status, error = "failed", next(s[3] for s in states if s[0] == "failed")
    elif all(kind == "done" for kind in kinds):
        status, error = "done", None
    elif "running" in kinds or "done" in kinds:
        status, error = "running", None
    else:
        status, error = "queued", None
    message = next((s[2] for s in states if s[0] == "running" and s[2]), "")
    return dict(base, status=status, progress=round(progress, 3), message=message, error=error,
                duration=data["source"]["duration"], is_video=data["source"]["is_video"])


def present(data: dict, brief: bool = False) -> dict:
    items = [_item_state(item) for item in data["items"]]
    counts: Dict[str, int] = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    out = {"id": data["id"], "name": data["name"], "created_at": data["created_at"],
           "updated_at": data["updated_at"], "recipe": data["recipe"], "counts": counts, "files": len(items)}
    if not brief:
        out["items"] = items
    return out


def gains_for(batch: dict, session_id: str) -> Optional[Dict[str, float]]:
    """The recipe's mix as gains for one of the batch's sessions; None if a sound is not finished."""
    item = next((i for i in batch["items"] if i["session"] == session_id), None)
    if item is None:
        return None
    data = sessions.load(session_id)
    takes = {sessions.sound_of(t): t for t in sessions.active_takes(data)}
    gains = {"rest": float(batch["recipe"].get("rest_gain", 1.0))}
    for sound, prompt in zip(item["sounds"], batch["recipe"]["prompts"]):
        if sound not in takes:
            return None
        gains[takes[sound]["id"]] = float(prompt.get("gain", 1.0))
    for sound, take in takes.items():  # sounds added to the session later play as they are
        gains.setdefault(take["id"], 1.0)
    return gains


def export_item(batch: dict, session_id: str) -> tuple:
    """One file's result, as the recipe says: ``(path, download name)``."""
    gains = gains_for(batch, session_id)
    if gains is None:
        raise ValueError("This file is not finished yet.")
    recipe = batch["recipe"]
    data = sessions.load(session_id)
    kind, fmt = recipe.get("kind", "mix"), recipe.get("format", "wav")
    if kind == "video" and data["source"]["is_video"]:
        return sessions.export_video(session_id, gains)
    if kind == "tracks":
        keys = [take["id"] for take in sessions.active_takes(data)] + ["rest"]
        return sessions.export_tracks(session_id, keys, fmt)
    return sessions.export_mix(session_id, gains, fmt)


def export_all(bid: str) -> tuple:
    """A ZIP with every finished file's result: ``(path, download name)``."""
    batch = load(bid)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 600
    for old in BATCH_DIR.glob("export-*.zip"):  # keep the last few minutes of downloads only
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass
    entries = []
    used = set()
    for item in batch["items"]:
        try:
            path, name = export_item(batch, item["session"])
        except (ValueError, sessions.SessionNotFound):
            continue
        stem, ext = os.path.splitext(name)
        suffix = 2
        while name in used:  # two files with the same name
            name = f"{stem} ({suffix}){ext}"
            suffix += 1
        used.add(name)
        entries.append((path, name))
    if not entries:
        raise ValueError("No file of this batch is finished yet.")
    zip_path = BATCH_DIR / f"export-{uuid.uuid4().hex[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path, name in entries:
            zf.write(path, arcname=name)
    label = sessions._tag(batch["name"]) if batch["name"] else "batch"
    return zip_path, f"AudioGhost {label} ({len(entries)} files).zip"
