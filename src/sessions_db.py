import sqlite3
import uuid
import json
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sessions.db"
)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_items (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                brand TEXT NOT NULL,
                title TEXT,
                category TEXT,
                features TEXT,
                image_ref TEXT,
                suggested_low REAL,
                suggested_high REAL,
                final_price REAL,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)


def create_session(name: str) -> str:
    session_id = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, name, status, created_at) VALUES (?, ?, 'active', ?)",
            (session_id, name, now)
        )
    return session_id


def get_sessions() -> list:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT s.id, s.name, s.status, s.created_at,
                   COUNT(si.id) AS item_count
            FROM sessions s
            LEFT JOIN session_items si ON s.id = si.session_id
            GROUP BY s.id
            ORDER BY s.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def get_session_items(session_id: str) -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM session_items WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,)
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        if d.get("features"):
            try:
                d["features"] = json.loads(d["features"])
            except Exception:
                d["features"] = {}
        result.append(d)
    return result


def add_item(
    session_id: str,
    brand: str,
    title: str,
    category: str,
    features: dict,
    image_ref: str,
    suggested_low: float,
    suggested_high: float,
    final_price: float = None,
) -> str:
    item_id = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO session_items
                (id, session_id, brand, title, category, features,
                 image_ref, suggested_low, suggested_high, final_price, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            item_id, session_id, brand, title, category,
            json.dumps(features), image_ref,
            suggested_low, suggested_high, final_price, now
        ))
    return item_id


def update_item(item_id: str, final_price=None, status: str = None):
    cols, vals = [], []
    if final_price is not None:
        cols.append("final_price = ?")
        vals.append(float(final_price))
    if status is not None:
        cols.append("status = ?")
        vals.append(status)
    if not cols:
        return
    vals.append(item_id)
    with get_db() as conn:
        conn.execute(f"UPDATE session_items SET {', '.join(cols)} WHERE id = ?", vals)


def update_session_status(session_id: str, status: str):
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET status = ? WHERE id = ?", (status, session_id)
        )


def get_items_by_ids(item_ids: list) -> list:
    if not item_ids:
        return []
    placeholders = ",".join("?" * len(item_ids))
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM session_items WHERE id IN ({placeholders})", item_ids
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        if d.get("features"):
            try:
                d["features"] = json.loads(d["features"])
            except Exception:
                d["features"] = {}
        result.append(d)
    return result


def count_sessions_today(date_prefix: str) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE created_at LIKE ?",
            (date_prefix + "%",)
        ).fetchone()
    return row[0] if row else 0
