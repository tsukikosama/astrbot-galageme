"""SQLite persistence core for the relationship memory plugin."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

UTC8 = timezone(timedelta(hours=8))
SCORE_FIELDS = ("affection", "trust", "familiarity", "closeness")
TAG_CATEGORIES = {"interest", "preference", "communication", "boundary", "impression"}


def _tag_fingerprint(name: str) -> str:
    return "sha256:" + hashlib.sha256(name.strip().casefold().encode("utf-8")).hexdigest()


def _clip(value: Any, length: int, default: str = "") -> str:
    return str(value if value is not None else default)[:length]


def _stamp(ts: float | None = None) -> str:
    return (
        datetime.fromtimestamp(ts, UTC8).isoformat(timespec="seconds")
        if ts is not None
        else datetime.now(UTC8).isoformat(timespec="seconds")
    )


def _parse_iso(value: str) -> float:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return dt.timestamp()


class Store:
    """Thread-safe SQLite store. Mutations use BEGIN IMMEDIATE for serialization."""

    def __init__(self, path: str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=30000")
        if self.path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.RLock()
        self._schema()

    def _schema(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS profiles(
          key TEXT PRIMARY KEY, group_id TEXT NOT NULL, user_id TEXT NOT NULL, umo TEXT NOT NULL,
          display_name TEXT NOT NULL DEFAULT '',
          provider_id TEXT NOT NULL DEFAULT '', persona TEXT NOT NULL DEFAULT '',
          memory_enabled INTEGER NOT NULL DEFAULT 1, proactive_enabled INTEGER NOT NULL DEFAULT 1,
          affection INTEGER NOT NULL DEFAULT 20, trust INTEGER NOT NULL DEFAULT 10,
          familiarity INTEGER NOT NULL DEFAULT 0, closeness INTEGER NOT NULL DEFAULT 0,
          style TEXT NOT NULL DEFAULT '', emotion TEXT NOT NULL DEFAULT '', emotion_updated_at REAL NOT NULL DEFAULT 0,
          last_interaction REAL NOT NULL DEFAULT 0, revision INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS messages(
          key TEXT NOT NULL, message_id TEXT NOT NULL, PRIMARY KEY(key,message_id)
        );
        CREATE TABLE IF NOT EXISTS score_inputs(
          key TEXT NOT NULL, fingerprint TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS score_inputs_recent ON score_inputs(key,fingerprint,created_at);
        CREATE TABLE IF NOT EXISTS turns(
          id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, message_id TEXT NOT NULL,
          user_text TEXT NOT NULL, assistant_text TEXT NOT NULL, created_at REAL NOT NULL,
          summarized INTEGER NOT NULL DEFAULT 0, UNIQUE(key,message_id)
        );
        CREATE INDEX IF NOT EXISTS turns_recent ON turns(key,created_at DESC);
        CREATE TABLE IF NOT EXISTS memories(
          id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, content TEXT NOT NULL,
          kind TEXT NOT NULL DEFAULT 'fact', public INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
          source_ids TEXT NOT NULL DEFAULT '[]', source_type TEXT NOT NULL DEFAULT 'turn'
        );
        CREATE INDEX IF NOT EXISTS memories_key ON memories(key,id DESC);
        CREATE TABLE IF NOT EXISTS user_tags(
          id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, name TEXT NOT NULL,
          category TEXT NOT NULL, source TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 1,
          evidence TEXT NOT NULL DEFAULT '', public INTEGER NOT NULL DEFAULT 0,
          source_ids TEXT NOT NULL DEFAULT '[]', created_at REAL NOT NULL, updated_at REAL NOT NULL,
          UNIQUE(key,name)
        );
        CREATE INDEX IF NOT EXISTS user_tags_key ON user_tags(key,name);
        CREATE TABLE IF NOT EXISTS tag_tombstones(
          key TEXT NOT NULL, name TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY(key,name)
        );
        CREATE TABLE IF NOT EXISTS summaries(
          key TEXT PRIMARY KEY, revision INTEGER NOT NULL, content TEXT NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS daily_history(
          key TEXT NOT NULL, day TEXT NOT NULL, first_turn INTEGER NOT NULL,
          last_turn INTEGER NOT NULL, content TEXT NOT NULL, updated_at REAL NOT NULL,
          PRIMARY KEY(key,day,first_turn,last_turn)
        );
        CREATE TABLE IF NOT EXISTS retired_memories(
          key TEXT NOT NULL, content TEXT NOT NULL, PRIMARY KEY(key,content)
        );
        CREATE TABLE IF NOT EXISTS followups(
          id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, content TEXT NOT NULL,
          due_at REAL NOT NULL, public INTEGER NOT NULL DEFAULT 0, finished INTEGER NOT NULL DEFAULT 0,
          created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS daily_deltas(
          key TEXT NOT NULL, day TEXT NOT NULL, field TEXT NOT NULL, positive INTEGER NOT NULL DEFAULT 0,
          negative INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(key,day,field)
        );
        CREATE TABLE IF NOT EXISTS audit(
          id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, created_at REAL NOT NULL,
          deltas TEXT NOT NULL, reason TEXT NOT NULL, old_values TEXT NOT NULL, new_values TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS audit_key ON audit(key,id DESC);
        CREATE TABLE IF NOT EXISTS proactive_jobs(
          id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, group_id TEXT NOT NULL, group_scope TEXT NOT NULL DEFAULT '',
          opportunity TEXT NOT NULL, created_at REAL NOT NULL, day TEXT NOT NULL,
          status TEXT NOT NULL, text TEXT NOT NULL DEFAULT '', UNIQUE(key,opportunity)
        );
        CREATE INDEX IF NOT EXISTS jobs_group_day ON proactive_jobs(group_id,day,status);
        """)
        self._migrate_schema()
        # A process restart makes delivery outcome unknowable. Keep its budget reservation.
        self.db.execute("UPDATE proactive_jobs SET status='unknown' WHERE status='reserved'")

    def _migrate_schema(self) -> None:
        additions = {
            "profiles": [
                ("user_persona", "TEXT NOT NULL DEFAULT ''"),
                ("emotion_updated_at", "REAL NOT NULL DEFAULT 0"),
                ("display_name", "TEXT NOT NULL DEFAULT ''"),
            ],
            "memories": [
                ("source_ids", "TEXT NOT NULL DEFAULT '[]'"),
                ("source_type", "TEXT NOT NULL DEFAULT 'turn'"),
            ],
            "proactive_jobs": [("group_scope", "TEXT NOT NULL DEFAULT ''")],
        }
        for table, columns in additions.items():
            existing = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for name, definition in columns:
                if name not in existing:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        self.db.execute("UPDATE proactive_jobs SET group_scope=group_id WHERE group_scope=''")
        self.db.execute("UPDATE profiles SET last_interaction=0 WHERE last_interaction IS NULL")
        # Suppression markers need equality matching, not the forgotten label text.
        for row in self.db.execute("SELECT key,name,created_at FROM tag_tombstones").fetchall():
            if re.fullmatch(r"sha256:[0-9a-f]{64}", row["name"]):
                continue
            self.db.execute(
                "INSERT OR IGNORE INTO tag_tombstones(key,name,created_at) VALUES(?,?,?)",
                (row["key"], _tag_fingerprint(row["name"]), row["created_at"]),
            )
            self.db.execute(
                "DELETE FROM tag_tombstones WHERE key=? AND name=?", (row["key"], row["name"])
            )

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def _transaction(self):
        return _Transaction(self)

    def ensure_profile(
        self,
        key: str,
        group_id: str,
        user_id: str,
        umo: str,
        provider_id: str = "",
        persona: str = "",
        display_name: str = "",
    ) -> dict:
        key = str(key)
        with self._transaction():
            old = self.db.execute("SELECT persona FROM profiles WHERE key=?", (key,)).fetchone()
            next_persona = _clip(persona, 16000) if persona else (old["persona"] if old else "")
            next_provider = _clip(provider_id, 128) if provider_id else ""
            if old:
                self.db.execute(
                    """UPDATE profiles SET group_id=?,user_id=?,umo=?,
                  provider_id=CASE WHEN ?<>'' THEN ? ELSE provider_id END,persona=?,
                  display_name=CASE WHEN ?<>'' THEN ? ELSE display_name END,
                  revision=revision+? WHERE key=?""",
                    (
                        _clip(group_id, 128),
                        _clip(user_id, 128),
                        _clip(umo, 256),
                        next_provider,
                        next_provider,
                        next_persona,
                        _clip(display_name, 120),
                        _clip(display_name, 120),
                        int(next_persona != old["persona"]),
                        key,
                    ),
                )
            else:
                self.db.execute(
                    "INSERT INTO profiles(key,group_id,user_id,umo,provider_id,persona,display_name,last_interaction) VALUES(?,?,?,?,?,?,?,0)",
                    (
                        key,
                        _clip(group_id, 128),
                        _clip(user_id, 128),
                        _clip(umo, 256),
                        next_provider,
                        next_persona,
                        _clip(display_name, 120),
                    ),
                )
        return self.profile(key)

    @staticmethod
    def _profile(row: sqlite3.Row | None) -> dict:
        if row is None:
            return {}
        d = dict(row)
        d["memory_enabled"] = bool(d["memory_enabled"])
        d["proactive_enabled"] = bool(d["proactive_enabled"])
        return d

    def profile(self, key: str) -> dict:
        row = self.db.execute("SELECT * FROM profiles WHERE key=?", (key,)).fetchone()
        return self._profile(row)

    def profiles(self) -> list[dict]:
        return [self._profile(r) for r in self.db.execute("SELECT * FROM profiles ORDER BY key")]

    def dashboard_profiles(
        self,
        query: str = "",
        group_id: str = "",
        offset: int = 0,
        limit: int = 20,
        keys: list[str] | None = None,
    ) -> dict:
        if keys is not None and not keys:
            return {"items": [], "total": 0, "groups": []}
        where = []
        params: list[Any] = []
        if keys is not None:
            where.append("p.key IN (SELECT value FROM json_each(?))")
            params.append(json.dumps(keys, ensure_ascii=False))
        if group_id:
            where.append("p.group_id=?")
            params.append(_clip(group_id, 128))
        if query:
            literal = (
                _clip(query, 200).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            pattern = f"%{literal}%"
            where.append(
                "(p.user_id LIKE ? ESCAPE '\\' OR p.display_name LIKE ? ESCAPE '\\' OR EXISTS(SELECT 1 FROM user_tags t WHERE t.key=p.key AND t.name LIKE ? ESCAPE '\\'))"
            )
            params.extend((pattern, pattern, pattern))
        clause = " WHERE " + " AND ".join(where) if where else ""
        groups_where = ""
        groups_params: list[Any] = []
        if keys is not None:
            groups_where = " WHERE key IN (SELECT value FROM json_each(?))"
            groups_params.append(json.dumps(keys, ensure_ascii=False))
        groups = [
            row[0]
            for row in self.db.execute(
                "SELECT DISTINCT group_id FROM profiles" + groups_where + " ORDER BY group_id",
                groups_params,
            ).fetchall()
        ]
        total = self.db.execute("SELECT COUNT(*) FROM profiles p" + clause, params).fetchone()[0]
        safe_fields = (
            "p.key,p.group_id,p.user_id,p.display_name,p.memory_enabled,p.proactive_enabled,"
            "p.affection,p.trust,p.familiarity,p.closeness,p.last_interaction,p.revision"
        )
        page_params = [*params, max(0, min(int(limit), 50)), max(0, int(offset))]
        rows = self.db.execute(
            f"SELECT {safe_fields},(SELECT COUNT(*) FROM user_tags t WHERE t.key=p.key) AS tag_count,"
            "(SELECT COUNT(*) FROM memories m WHERE m.key=p.key) AS memory_count "
            "FROM profiles p" + clause + " ORDER BY p.group_id,p.user_id,p.key LIMIT ? OFFSET ?",
            page_params,
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["memory_enabled"] = bool(item["memory_enabled"])
            item["proactive_enabled"] = bool(item["proactive_enabled"])
            items.append(item)
        return {"items": items, "total": int(total), "groups": groups}

    def dashboard_detail(self, key: str) -> dict | None:
        profile = self.profile(key)
        if not profile:
            return None
        profile.pop("persona", None)
        memory_rows = self.db.execute(
            "SELECT id,content,kind,public,source_ids,source_type,created_at FROM memories WHERE key=? ORDER BY id DESC LIMIT 100",
            (key,),
        ).fetchall()
        summaries = self.db.execute(
            "SELECT day,first_turn,last_turn,content,updated_at FROM daily_history WHERE key=? ORDER BY day DESC,updated_at DESC LIMIT 20",
            (key,),
        ).fetchall()
        jobs = self.db.execute(
            "SELECT id,opportunity,created_at,status,text FROM proactive_jobs WHERE key=? ORDER BY created_at DESC,id DESC LIMIT 20",
            (key,),
        ).fetchall()
        return {
            "profile": profile,
            "tags": self.tags(key),
            "memories": [self._memory_dict(row) for row in memory_rows],
            "daily_summaries": [
                dict(row, updated_at=_stamp(row["updated_at"])) for row in summaries
            ],
            "changes": self.audit(key, 20),
            "proactive": [dict(row, created_at=_stamp(row["created_at"])) for row in jobs],
        }

    def set_preferences(
        self, key: str, memory_enabled: bool | None = None, proactive_enabled: bool | None = None
    ) -> None:
        with self._transaction():
            p = self.db.execute(
                "SELECT memory_enabled,proactive_enabled FROM profiles WHERE key=?", (key,)
            ).fetchone()
            if not p:
                return
            m = p["memory_enabled"] if memory_enabled is None else int(memory_enabled)
            pr = p["proactive_enabled"] if proactive_enabled is None else int(proactive_enabled)
            if m != p["memory_enabled"] or pr != p["proactive_enabled"]:
                self.db.execute(
                    "UPDATE profiles SET memory_enabled=?,proactive_enabled=?,revision=revision+1 WHERE key=?",
                    (m, pr, key),
                )

    def recent(self, key: str, limit: int = 8) -> list[dict]:
        p = self.profile(key)
        if not p or not p["memory_enabled"]:
            return []
        rows = self.db.execute(
            "SELECT id,user_text,assistant_text,created_at FROM turns WHERE key=? ORDER BY created_at DESC,id DESC LIMIT ?",
            (key, max(0, min(int(limit), 100))),
        ).fetchall()
        return [dict(r, created_at=_stamp(r["created_at"])) for r in reversed(rows)]

    def memories(self, key: str, query: str = "", limit: int = 5) -> list[dict]:
        p = self.profile(key)
        if not p or not p["memory_enabled"]:
            return []
        lim = max(0, min(int(limit), 100))
        if not query:
            rows = self.db.execute(
                "SELECT id,content,kind,public,source_ids,source_type,created_at FROM memories WHERE key=? ORDER BY id DESC LIMIT ?",
                (key, lim),
            ).fetchall()
            return [self._memory_dict(r) for r in rows]
        terms = self._query_terms(query)
        if not terms:
            return []
        rows = self.db.execute(
            "SELECT id,content,kind,public,source_ids,source_type,created_at FROM memories WHERE key=? ORDER BY id DESC LIMIT 200",
            (key,),
        ).fetchall()
        ranked = []
        for row in rows:
            overlap = terms & self._query_terms(row["content"])
            if overlap:
                ranked.append((len(overlap), row["id"], row))
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [self._memory_dict(item[2]) for item in ranked[:lim]]

    @staticmethod
    def _query_terms(value: str) -> set[str]:
        text = _clip(value, 1000).lower()
        words = set(re.findall(r"[a-z0-9_]+", text))
        cjk = "".join(re.findall(r"[\u3400-\u9fff]", text))
        if len(cjk) == 1:
            words.add(cjk)
        else:
            words.update(cjk[i : i + 2] for i in range(len(cjk) - 1))
        return words

    @staticmethod
    def _memory_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["public"] = bool(d["public"])
        d["source_ids"] = json.loads(d.get("source_ids") or "[]")
        return d

    def commit_turn(
        self,
        key: str,
        message_id: str,
        user_text: str,
        assistant_text: str,
        analysis: dict,
        now: float | None = None,
    ) -> bool:
        now = now if now is not None else time.time()
        mid = _clip(message_id, 256)
        fingerprint = hashlib.sha256(_clip(user_text, 12000).encode("utf-8")).hexdigest()
        with self._transaction():
            p = self.db.execute("SELECT * FROM profiles WHERE key=?", (key,)).fetchone()
            if not p:
                return False
            try:
                self.db.execute("INSERT INTO messages(key,message_id) VALUES(?,?)", (key, mid))
            except sqlite3.IntegrityError:
                return False
            old = {f: p[f] for f in SCORE_FIELDS}
            same_recent = self.db.execute(
                "SELECT 1 FROM score_inputs WHERE key=? AND fingerprint=? AND created_at>=? LIMIT 1",
                (key, fingerprint, now - 3600),
            ).fetchone()
            self.db.execute(
                "INSERT INTO score_inputs(key,fingerprint,created_at) VALUES(?,?,?)",
                (key, fingerprint, now),
            )
            if p["memory_enabled"]:
                self.db.execute(
                    "INSERT INTO turns(key,message_id,user_text,assistant_text,created_at) VALUES(?,?,?,?,?)",
                    (key, mid, _clip(user_text, 12000), _clip(assistant_text, 12000), now),
                )
            try:
                confidence = float(analysis.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = 0
            deltas = {}
            try:
                persona_confidence = float(analysis.get("persona_confidence", confidence))
            except (TypeError, ValueError):
                persona_confidence = 0
            if p["memory_enabled"] and persona_confidence >= 0.8:
                user_persona = _clip(analysis.get("user_persona"), 4000).strip()
                if user_persona:
                    self.db.execute(
                        "UPDATE profiles SET user_persona=? WHERE key=?", (user_persona, key)
                    )
            if p["memory_enabled"] and confidence >= 0.7:
                keep_private = self._replace_memories(key, analysis.get("replace_memory_ids") or [])
                for mem in (analysis.get("memories") or [])[:20]:
                    self._insert_memory(key, mem, now, [mid], "turn", keep_private)
                style = _clip(analysis.get("style"), 800)
                emotion = _clip(analysis.get("emotion"), 100)
                if style:
                    self.db.execute("UPDATE profiles SET style=? WHERE key=?", (style, key))
                if emotion and emotion != p["emotion"]:
                    self.db.execute(
                        "UPDATE profiles SET emotion=?,emotion_updated_at=? WHERE key=?",
                        (emotion, now, key),
                    )
                for item in (analysis.get("followups") or [])[:10]:
                    try:
                        due = _parse_iso(_clip(item.get("due_at", ""), 80))
                    except (ValueError, TypeError):
                        continue
                    content = _clip(item.get("content"), 1000).strip()
                    if (
                        not content
                        or self.db.execute(
                            "SELECT 1 FROM followups WHERE key=? AND content=? AND due_at=?",
                            (key, content, due),
                        ).fetchone()
                    ):
                        continue
                    self.db.execute(
                        "INSERT INTO followups(key,content,due_at,public,finished,created_at) VALUES(?,?,?,?,0,?)",
                        (key, content, due, int(bool(item.get("public"))), now),
                    )
                self._store_tags(key, analysis.get("tags"), now, "model", [mid], 6)
            if confidence >= 0.7:
                if not same_recent:
                    raw = analysis.get("deltas") or {}
                    day = _stamp(now)[:10]
                    for field in SCORE_FIELDS:
                        try:
                            requested = max(-3, min(3, int(raw.get(field, 0))))
                        except (TypeError, ValueError):
                            requested = 0
                        if field == "familiarity":
                            requested = max(0, requested)
                        row = self.db.execute(
                            "SELECT positive,negative FROM daily_deltas WHERE key=? AND day=? AND field=?",
                            (key, day, field),
                        ).fetchone()
                        pos, neg = (row[0], row[1]) if row else (0, 0)
                        if requested > 0:
                            applied = min(requested, 8 - pos)
                            if applied > 0:
                                self.db.execute(
                                    "INSERT INTO daily_deltas(key,day,field,positive,negative) VALUES(?,?,?,?,0) ON CONFLICT(key,day,field) DO UPDATE SET positive=positive+excluded.positive",
                                    (key, day, field, applied),
                                )
                        elif requested < 0:
                            applied = -min(-requested, 10 - neg)
                            if applied < 0:
                                self.db.execute(
                                    "INSERT INTO daily_deltas(key,day,field,positive,negative) VALUES(?,?,?,0,?) ON CONFLICT(key,day,field) DO UPDATE SET negative=negative+excluded.negative",
                                    (key, day, field, -applied),
                                )
                        else:
                            applied = 0
                        deltas[field] = applied
            new = {f: max(0, min(100, old[f] + deltas.get(f, 0))) for f in SCORE_FIELDS}
            changed = any(new[f] != old[f] for f in SCORE_FIELDS)
            self.db.execute(
                "UPDATE profiles SET affection=?,trust=?,familiarity=?,closeness=?,last_interaction=?,revision=revision+1 WHERE key=?",
                (*(new[f] for f in SCORE_FIELDS), now, key),
            )
            if changed:
                self.db.execute(
                    "INSERT INTO audit(key,created_at,deltas,reason,old_values,new_values) VALUES(?,?,?,?,?,?)",
                    (
                        key,
                        now,
                        json.dumps(deltas, ensure_ascii=False),
                        _clip(analysis.get("reason"), 500) if p["memory_enabled"] else "",
                        json.dumps(old),
                        json.dumps(new),
                    ),
                )
        return True

    def _replace_memories(self, key: str, ids: list[int]) -> bool:
        keep_private = False
        for value in ids[:100]:
            try:
                memory_id = int(value)
            except (TypeError, ValueError):
                continue
            row = self.db.execute(
                "SELECT public,content FROM memories WHERE key=? AND id=?", (key, memory_id)
            ).fetchone()
            if row and not row["public"]:
                keep_private = True
            if row:
                self.db.execute(
                    "INSERT OR IGNORE INTO retired_memories(key,content) VALUES(?,?)",
                    (key, row["content"]),
                )
            self.db.execute("DELETE FROM memories WHERE key=? AND id=?", (key, memory_id))
        return keep_private

    def _insert_memory(
        self,
        key: str,
        mem: dict,
        now: float,
        source_ids: list[Any] | None = None,
        source_type: str = "turn",
        force_private: bool = False,
    ) -> None:
        content = _clip(mem.get("content"), 1000).strip()
        if content:
            if (
                source_type == "summary"
                and self.db.execute(
                    "SELECT 1 FROM retired_memories WHERE key=? AND content=?", (key, content)
                ).fetchone()
            ):
                return
            if source_type == "turn":
                self.db.execute(
                    "DELETE FROM retired_memories WHERE key=? AND content=?", (key, content)
                )
            incoming = [_clip(x, 256) for x in (source_ids or [])]
            existing = self.db.execute(
                "SELECT id,public,source_ids FROM memories WHERE key=? AND content=?",
                (key, content),
            ).fetchone()
            if existing:
                merged = list(dict.fromkeys(json.loads(existing["source_ids"] or "[]") + incoming))[
                    -100:
                ]
                self.db.execute(
                    "UPDATE memories SET public=MIN(public,?),source_ids=? WHERE id=?",
                    (
                        int(bool(mem.get("public"))) and not force_private,
                        json.dumps(merged, ensure_ascii=False),
                        existing["id"],
                    ),
                )
            else:
                public = int(bool(mem.get("public"))) and not force_private
                self.db.execute(
                    "INSERT INTO memories(key,content,kind,public,created_at,source_ids,source_type) VALUES(?,?,?,?,?,?,?)",
                    (
                        key,
                        content,
                        _clip(mem.get("kind", "fact"), 40),
                        int(public),
                        now,
                        json.dumps(incoming, ensure_ascii=False),
                        _clip(source_type, 20),
                    ),
                )

    def _store_tags(
        self,
        key: str,
        items: Any,
        now: float,
        source: str,
        source_ids: list[Any],
        limit: int,
    ) -> None:
        if source not in {"model", "summary"} or not isinstance(items, list):
            return
        ids = [_clip(x, 256) for x in source_ids][:100]
        saved = 0
        for item in items:
            if saved >= limit:
                break
            if not isinstance(item, dict):
                continue
            if not isinstance(item.get("name"), str):
                continue
            name = _clip(item.get("name"), 25).strip()
            category = item.get("category")
            confidence = item.get("confidence")
            if type(confidence) not in (int, float):
                continue
            try:
                numeric_confidence = float(confidence)
            except (OverflowError, ValueError):
                continue
            if (
                not name
                or len(name) > 24
                or not isinstance(category, str)
                or category not in TAG_CATEGORIES
                or not math.isfinite(numeric_confidence)
                or numeric_confidence < 0.8
            ):
                continue
            if self.db.execute(
                "SELECT 1 FROM tag_tombstones WHERE key=? AND name=?", (key, _tag_fingerprint(name))
            ).fetchone():
                continue
            existing = self.db.execute(
                "SELECT id,source,public,source_ids FROM user_tags WHERE key=? AND name=?",
                (key, name),
            ).fetchone()
            if existing and existing["source"] == "manual":
                continue
            if existing:
                merged = list(dict.fromkeys(json.loads(existing["source_ids"] or "[]") + ids))[
                    -100:
                ]
                self.db.execute(
                    "UPDATE user_tags SET category=?,source=?,confidence=?,evidence=?,public=MIN(public,?),source_ids=?,updated_at=? WHERE id=?",
                    (
                        category,
                        source,
                        min(1.0, numeric_confidence),
                        _clip(item.get("evidence"), 200),
                        int(item.get("public") is True),
                        json.dumps(merged, ensure_ascii=False),
                        now,
                        existing["id"],
                    ),
                )
                saved += 1
            else:
                count = self.db.execute(
                    "SELECT COUNT(*) FROM user_tags WHERE key=?", (key,)
                ).fetchone()[0]
                if count >= 30:
                    continue
                self.db.execute(
                    "INSERT INTO user_tags(key,name,category,source,confidence,evidence,public,source_ids,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        key,
                        name,
                        category,
                        source,
                        min(1.0, numeric_confidence),
                        _clip(item.get("evidence"), 200),
                        int(item.get("public") is True),
                        json.dumps(ids, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                saved += 1

    @staticmethod
    def _tag_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["public"] = bool(d["public"])
        d["source_ids"] = json.loads(d.get("source_ids") or "[]")
        return d

    def tags(self, key: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT id,name,category,source,confidence,evidence,public,source_ids,updated_at FROM user_tags WHERE key=? ORDER BY category,name,id",
            (key,),
        ).fetchall()
        return [self._tag_dict(row) for row in rows]

    def update_tag(
        self,
        key: str,
        name: str,
        category: str = "preference",
        public: bool = False,
        expected_revision: int | None = None,
    ) -> bool:
        normalized = name.strip() if isinstance(name, str) else ""
        if not normalized or len(normalized) > 24 or any(ord(c) < 32 for c in normalized):
            raise ValueError("tag name must contain 1 to 24 characters")
        if not isinstance(category, str) or category not in TAG_CATEGORIES:
            raise ValueError("invalid tag category")
        if not isinstance(public, bool):
            raise ValueError("public must be bool")
        now = time.time()
        with self._transaction():
            profile = self.db.execute(
                "SELECT revision FROM profiles WHERE key=?", (key,)
            ).fetchone()
            if not profile or (
                expected_revision is not None and profile["revision"] != expected_revision
            ):
                return False
            existing = self.db.execute(
                "SELECT id FROM user_tags WHERE key=? AND name=?", (key, normalized)
            ).fetchone()
            if existing:
                self.db.execute(
                    "UPDATE user_tags SET category=?,source='manual',confidence=1,evidence='手动添加或编辑',public=?,source_ids='[]',updated_at=? WHERE id=?",
                    (category, int(public), now, existing["id"]),
                )
            else:
                count = self.db.execute(
                    "SELECT COUNT(*) FROM user_tags WHERE key=?", (key,)
                ).fetchone()[0]
                if count >= 30:
                    raise ValueError("tag limit reached")
                self.db.execute(
                    "INSERT INTO user_tags(key,name,category,source,confidence,evidence,public,source_ids,created_at,updated_at) VALUES(?,?,?,'manual',1,'手动添加',?,'[]',?,?)",
                    (key, normalized, category, int(public), now, now),
                )
            self.db.execute(
                "DELETE FROM tag_tombstones WHERE key=? AND name=?",
                (key, _tag_fingerprint(normalized)),
            )
            self.db.execute("UPDATE profiles SET revision=revision+1 WHERE key=?", (key,))
        return True

    def delete_tag(self, key: str, tag_id: int, expected_revision: int | None = None) -> bool:
        with self._transaction():
            profile = self.db.execute(
                "SELECT revision FROM profiles WHERE key=?", (key,)
            ).fetchone()
            if not profile or (
                expected_revision is not None and profile["revision"] != expected_revision
            ):
                return False
            row = self.db.execute(
                "SELECT name FROM user_tags WHERE key=? AND id=?", (key, tag_id)
            ).fetchone()
            if not row:
                return False
            self.db.execute(
                "INSERT INTO tag_tombstones(key,name,created_at) VALUES(?,?,?) ON CONFLICT(key,name) DO UPDATE SET created_at=excluded.created_at",
                (key, _tag_fingerprint(row["name"]), time.time()),
            )
            self.db.execute("DELETE FROM user_tags WHERE key=? AND id=?", (key, tag_id))
            self.db.execute("UPDATE profiles SET revision=revision+1 WHERE key=?", (key,))
        return True

    def forget(self, key: str, query: str = "") -> int:
        with self._transaction():
            if query:
                literal = (
                    _clip(query, 200).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                )
                cur = self.db.execute(
                    "DELETE FROM memories WHERE key=? AND content LIKE ? ESCAPE '\\'",
                    (key, "%" + literal + "%"),
                )
                count = cur.rowcount
            else:
                count = self.db.execute(
                    "SELECT COUNT(*) FROM memories WHERE key=?", (key,)
                ).fetchone()[0]
                self.db.execute("DELETE FROM memories WHERE key=?", (key,))
            if query:
                tag_query = (
                    _clip(query, 200).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                )
                tag_rows = self.db.execute(
                    "SELECT id,name,source FROM user_tags WHERE key=? AND (source<>'manual' OR name LIKE ? ESCAPE '\\')",
                    (key, "%" + tag_query + "%"),
                ).fetchall()
            else:
                tag_rows = self.db.execute(
                    "SELECT id,name,source FROM user_tags WHERE key=?", (key,)
                ).fetchall()
            for tag in tag_rows:
                self.db.execute(
                    "INSERT INTO tag_tombstones(key,name,created_at) VALUES(?,?,?) ON CONFLICT(key,name) DO UPDATE SET created_at=excluded.created_at",
                    (key, _tag_fingerprint(tag["name"]), time.time()),
                )
                self.db.execute("DELETE FROM user_tags WHERE key=? AND id=?", (key, tag["id"]))
            self._clear_context(key)
            self.db.execute("DELETE FROM retired_memories WHERE key=?", (key,))
            self.db.execute(
                "UPDATE profiles SET user_persona='',style='',emotion='',emotion_updated_at=0,revision=revision+1 WHERE key=?",
                (key,),
            )
            self.db.execute("UPDATE audit SET reason='' WHERE key=?", (key,))
            self.db.execute("UPDATE proactive_jobs SET text='' WHERE key=?", (key,))
        return count

    def _clear_context(self, key: str) -> None:
        self.db.execute("DELETE FROM turns WHERE key=?", (key,))
        self.db.execute("DELETE FROM summaries WHERE key=?", (key,))
        self.db.execute("DELETE FROM daily_history WHERE key=?", (key,))
        self.db.execute("DELETE FROM followups WHERE key=?", (key,))

    def pending_summaries(
        self, before: float, limit: int = 20, keys: list[str] | None = None
    ) -> list[dict]:
        if keys is not None and not keys:
            return []
        key_filter = (
            " AND p.key IN (" + ",".join("?" for _ in keys) + ")" if keys is not None else ""
        )
        parameters = (before, *(keys or []), max(0, min(int(limit), 1000)))
        rows = self.db.execute(
            "SELECT p.key,p.revision,p.style FROM profiles p WHERE p.memory_enabled=1 AND EXISTS(SELECT 1 FROM turns t WHERE t.key=p.key AND t.summarized=0 AND t.created_at<=?)"
            + key_filter
            + " ORDER BY p.key LIMIT ?",
            parameters,
        ).fetchall()
        out = []
        for r in rows:
            first = self.db.execute(
                "SELECT MIN(created_at) FROM turns WHERE key=? AND summarized=0 AND created_at<=?",
                (r["key"], before),
            ).fetchone()[0]
            day_end = (
                datetime.fromtimestamp(first, UTC8).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                + timedelta(days=1)
            ).timestamp()
            turns = self.db.execute(
                "SELECT id,user_text,assistant_text,created_at FROM turns WHERE key=? AND summarized=0 AND created_at<=? AND created_at<? ORDER BY id LIMIT 30",
                (r["key"], before, day_end),
            ).fetchall()
            mems = self.db.execute(
                "SELECT id,content,kind,public,created_at FROM memories WHERE key=? ORDER BY id DESC LIMIT 50",
                (r["key"],),
            ).fetchall()
            out.append(
                {
                    "key": r["key"],
                    "revision": r["revision"],
                    "turns": [dict(t, created_at=_stamp(t["created_at"])) for t in turns],
                    "memories": [dict(m, public=bool(m["public"])) for m in mems],
                    "style": r["style"],
                }
            )
        return out

    def apply_summary(self, key: str, revision: int, turn_ids: list[int], summary: dict) -> bool:
        if not turn_ids:
            return False
        with self._transaction():
            p = self.db.execute(
                "SELECT revision,memory_enabled FROM profiles WHERE key=?", (key,)
            ).fetchone()
            if not p or not p["memory_enabled"] or p["revision"] != revision:
                return False
            ids = tuple(int(i) for i in turn_ids[:30])
            placeholders = ",".join("?" for _ in ids)
            found = self.db.execute(
                f"SELECT COUNT(*) FROM turns WHERE key=? AND id IN ({placeholders}) AND summarized=0",
                (key, *ids),
            ).fetchone()[0]
            if found != len(set(ids)):
                return False
            now = time.time()
            first_time = self.db.execute(
                f"SELECT MIN(created_at) FROM turns WHERE key=? AND id IN ({placeholders})",
                (key, *ids),
            ).fetchone()[0]
            keep_private = self._replace_memories(key, summary.get("replace_memory_ids") or [])
            source_ids = [str(i) for i in ids]
            for mem in (summary.get("memories") or [])[:20]:
                self._insert_memory(key, mem, now, source_ids, "summary", keep_private)
            self._store_tags(key, summary.get("tags"), now, "summary", source_ids, 6)
            self.db.execute(
                f"UPDATE turns SET summarized=1 WHERE key=? AND id IN ({placeholders})", (key, *ids)
            )
            content = _clip(summary.get("summary"), 4000)
            self.db.execute(
                "INSERT INTO summaries(key,revision,content,updated_at) VALUES(?,?,?,?) ON CONFLICT(key) DO UPDATE SET revision=excluded.revision,content=excluded.content,updated_at=excluded.updated_at",
                (key, revision, content, now),
            )
            self.db.execute(
                "INSERT INTO daily_history(key,day,first_turn,last_turn,content,updated_at) VALUES(?,?,?,?,?,?)",
                (key, _stamp(first_time)[:10], min(ids), max(ids), content, now),
            )
            style = _clip(summary.get("style"), 800)
            self.db.execute(
                "UPDATE profiles SET style=CASE WHEN ?<>'' THEN ? ELSE style END,revision=revision+1 WHERE key=?",
                (style, style, key),
            )
        return True

    def followups(self, key: str, now: float) -> list[dict]:
        rows = self.db.execute(
            "SELECT id,content,due_at,public FROM followups WHERE key=? AND public=1 AND finished=0 AND due_at<=? AND due_at>=? ORDER BY due_at,id",
            (key, now, now - 7 * 86400),
        ).fetchall()
        return [dict(r, due_at=_stamp(r["due_at"]), public=bool(r["public"])) for r in rows]

    def finish_followup(self, id: int) -> None:
        with self._transaction():
            self.db.execute("UPDATE followups SET finished=1 WHERE id=?", (id,))

    def reserve_proactive(
        self,
        key: str,
        opportunity: str,
        now: float,
        probability: float,
        draw: float,
        group_limit: int = 3,
        user_limit: int = 1,
        cooldown_hours: float = 12,
        group_interval_minutes: float = 60,
    ) -> int | None:
        day = _stamp(now)[:10]
        opp = _clip(opportunity, 256)
        with self._transaction():
            p = self.db.execute(
                "SELECT group_id,proactive_enabled FROM profiles WHERE key=?", (key,)
            ).fetchone()
            if not p:
                return None
            if self.db.execute(
                "SELECT 1 FROM proactive_jobs WHERE key=? AND opportunity=?", (key, opp)
            ).fetchone():
                return None
            group_scope = self._group_scope(key, p["group_id"])
            allowed = bool(p["proactive_enabled"])
            active = "('reserved','sent','unknown')"
            user_count = self.db.execute(
                f"SELECT COUNT(*) FROM proactive_jobs WHERE key=? AND day=? AND status IN {active}",
                (key, day),
            ).fetchone()[0]
            group_count = self.db.execute(
                f"SELECT COUNT(*) FROM proactive_jobs WHERE group_scope=? AND day=? AND status IN {active}",
                (group_scope, day),
            ).fetchone()[0]
            last_user = self.db.execute(
                f"SELECT MAX(created_at) FROM proactive_jobs WHERE key=? AND status IN {active}",
                (key,),
            ).fetchone()[0]
            last_group = self.db.execute(
                f"SELECT MAX(created_at) FROM proactive_jobs WHERE group_scope=? AND status IN {active}",
                (group_scope,),
            ).fetchone()[0]
            if user_count >= max(0, user_limit) or group_count >= max(0, group_limit):
                allowed = False
            if last_user is not None and now - last_user < cooldown_hours * 3600:
                allowed = False
            if last_group is not None and now - last_group < group_interval_minutes * 60:
                allowed = False
            if not allowed:
                return None
            success = max(0.0, min(1.0, float(probability))) > max(0.0, min(1.0, float(draw)))
            status = "reserved" if success else "missed"
            cur = self.db.execute(
                "INSERT INTO proactive_jobs(key,group_id,group_scope,opportunity,created_at,day,status) VALUES(?,?,?,?,?,?,?)",
                (key, p["group_id"], group_scope, opp, now, day, status),
            )
            return int(cur.lastrowid) if success else None

    @staticmethod
    def _group_scope(key: str, fallback_group_id: str) -> str:
        try:
            parts = json.loads(key)
            if isinstance(parts, list) and len(parts) >= 4:
                return json.dumps(parts[:-1], ensure_ascii=False, separators=(",", ":"))
        except (ValueError, TypeError):
            pass
        return _clip(fallback_group_id, 128)

    def finish_proactive(self, job_id: int, status: str, text: str = "") -> None:
        if status not in ("sent", "failed", "unknown"):
            raise ValueError("status must be sent, failed, or unknown")
        with self._transaction():
            row = self.db.execute(
                "SELECT status FROM proactive_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row or row["status"] != "reserved":
                return
            self.db.execute(
                "UPDATE proactive_jobs SET status=?,text=? WHERE id=?",
                (status, _clip(text, 2000), job_id),
            )

    def proactive_stats(self, key: str, now: float) -> dict:
        p = self.db.execute("SELECT last_interaction FROM profiles WHERE key=?", (key,)).fetchone()
        if not p:
            return {"unanswered": 0, "last_texts": []}
        last_interaction = p["last_interaction"] or 0
        unanswered = self.db.execute(
            "SELECT COUNT(*) FROM proactive_jobs WHERE key=? AND status IN ('sent','unknown') AND created_at>?",
            (key, last_interaction),
        ).fetchone()[0]
        texts = self.db.execute(
            "SELECT text FROM proactive_jobs WHERE key=? AND status='sent' AND text<>'' ORDER BY created_at DESC,id DESC LIMIT 5",
            (key,),
        ).fetchall()
        return {"unanswered": int(unanswered), "last_texts": [r["text"] for r in texts]}

    def prune(self, before: float) -> int:
        with self._transaction():
            a = self.db.execute("DELETE FROM turns WHERE created_at<?", (before,)).rowcount
            self.db.execute("DELETE FROM score_inputs WHERE created_at<?", (before - 3600,))
            b = self.db.execute(
                "DELETE FROM proactive_jobs WHERE created_at<? AND status IN ('failed','missed')",
                (before,),
            ).rowcount
        return a + b

    def audit(self, key: str, limit: int = 10) -> list[dict]:
        rows = self.db.execute(
            "SELECT id,created_at,deltas,reason,old_values,new_values FROM audit WHERE key=? ORDER BY id DESC LIMIT ?",
            (key, max(0, min(int(limit), 100))),
        ).fetchall()
        return [
            dict(
                r,
                created_at=_stamp(r["created_at"]),
                deltas=json.loads(r["deltas"]),
                old_values=json.loads(r["old_values"]),
                new_values=json.loads(r["new_values"]),
            )
            for r in rows
        ]

    def reset_relationship(self, key: str) -> None:
        with self._transaction():
            self._clear_context(key)
            self.db.execute(
                "UPDATE profiles SET affection=20,trust=10,familiarity=0,closeness=0,user_persona='',style='',emotion='',emotion_updated_at=0,revision=revision+1 WHERE key=?",
                (key,),
            )
            self.db.execute("DELETE FROM audit WHERE key=?", (key,))
            self.db.execute("DELETE FROM daily_deltas WHERE key=?", (key,))
            self.db.execute("DELETE FROM memories WHERE key=? AND kind='relationship'", (key,))
            impressions = self.db.execute(
                "SELECT id,name FROM user_tags WHERE key=? AND category='impression'", (key,)
            ).fetchall()
            for tag in impressions:
                self.db.execute(
                    "INSERT INTO tag_tombstones(key,name,created_at) VALUES(?,?,?) ON CONFLICT(key,name) DO UPDATE SET created_at=excluded.created_at",
                    (key, _tag_fingerprint(tag["name"]), time.time()),
                )
                self.db.execute("DELETE FROM user_tags WHERE key=? AND id=?", (key, tag["id"]))


class _Transaction:
    def __init__(self, store: Store):
        self.store = store

    def __enter__(self):
        self.store._lock.acquire()
        self.store.db.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, typ, value, tb):
        try:
            self.store.db.execute("ROLLBACK" if typ else "COMMIT")
        finally:
            self.store._lock.release()
