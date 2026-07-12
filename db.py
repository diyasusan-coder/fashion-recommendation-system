"""
Lightweight SQLite persistence for the "Lookbook" history feature.
No ORM needed — this is a small, presentation-scale app.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "lookbook.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            image_path TEXT,
            garment TEXT,
            color_name TEXT,
            color_hex TEXT,
            occasion TEXT,
            season TEXT,
            suggestion TEXT,
            pairing_images TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_entry(image_path, garment, color_name, color_hex, occasion, season, suggestion, pairing_images):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """
        INSERT INTO history
        (created_at, image_path, garment, color_name, color_hex, occasion, season, suggestion, pairing_images)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            image_path,
            garment,
            color_name,
            color_hex,
            occasion,
            season,
            suggestion,
            json.dumps(pairing_images),
        ),
    )
    conn.commit()
    entry_id = cur.lastrowid
    conn.close()
    return entry_id


def get_history(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    result = []
    for row in rows:
        d = dict(row)
        d["pairing_images"] = json.loads(d["pairing_images"] or "[]")
        result.append(d)
    return result


def delete_entry(entry_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()
