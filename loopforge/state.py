from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path


class Store:
    def __init__(self, directory: str):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = root / "sessions.sqlite3"
        self.db = sqlite3.connect(path)
        os.chmod(path, 0o600)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, workspace TEXT NOT NULL, updated REAL NOT NULL, data TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, time REAL NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL)")
        self.db.commit()

    def new(self, workspace: str) -> dict:
        session = {"id": uuid.uuid4().hex, "workspace": workspace, "inputs": [], "turns": [],
                   "memory": "", "changed": [], "shell_used": False, "pending": None, "status": "idle"}
        self.save(session)
        return session

    def get(self, session_id: str, workspace: str) -> dict:
        row = self.db.execute("SELECT workspace, data FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            raise ValueError("Unknown session")
        if row[0] != workspace:
            raise ValueError("Session belongs to another workspace")
        return json.loads(row[1])

    def save(self, session: dict):
        with self.db:
            self.db.execute("INSERT INTO sessions VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated=excluded.updated,data=excluded.data",
                            (session["id"], session["workspace"], time.time(), json.dumps(session, ensure_ascii=False)))

    def event(self, session_id: str, kind: str, data: dict):
        with self.db:
            self.db.execute("INSERT INTO events(session,time,kind,data) VALUES(?,?,?,?)",
                            (session_id, time.time(), kind, json.dumps(data, ensure_ascii=False)))

    def recent(self, workspace: str):
        rows = self.db.execute("SELECT id, updated, data FROM sessions WHERE workspace=? ORDER BY updated DESC LIMIT 20", (workspace,)).fetchall()
        return [{"id": r[0], "updated": r[1], "status": json.loads(r[2]).get("status")} for r in rows]

    def close(self):
        self.db.close()


@asynccontextmanager
async def workspace_lock(workspace: str):
    # Stable across processes and state directories for the same Unix account.
    lock_root = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/loopforge-{os.getuid()}")) / "loopforge-locks"
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = lock_root / (hashlib.sha256(workspace.encode()).hexdigest() + ".lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.05)
        yield
    finally:
        os.close(fd)
