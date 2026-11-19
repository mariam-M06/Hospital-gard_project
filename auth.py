"""Authentication: usernames + bcrypt-hashed passwords stored in a local SQLite file.

FR-1.3: passwords are never stored in plain text.
"""
import os
import sqlite3
from datetime import datetime

import bcrypt

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


def _connect(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
               username      TEXT PRIMARY KEY,
               password_hash BLOB NOT NULL,
               role          TEXT NOT NULL DEFAULT 'staff',
               created_at    TEXT NOT NULL
           )"""
    )
    return conn


def add_user(username: str, password: str, role: str = "staff", db_path: str = DB_PATH) -> None:
    """Create a user, or replace the password of an existing one."""
    username = username.strip().lower()
    if not username or len(password) < 6:
        raise ValueError("Username is required and the password must have at least 6 characters.")
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (username, hashed, role, datetime.now().isoformat(timespec="seconds")),
        )


def verify_user(username: str, password: str, db_path: str = DB_PATH):
    """Return {'username', 'role'} if the credentials are correct, otherwise None."""
    username = username.strip().lower()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT password_hash, role FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        # Hash anyway so the response time does not reveal whether the user exists.
        bcrypt.checkpw(b"dummy", bcrypt.hashpw(b"dummy", bcrypt.gensalt()))
        return None
    stored_hash, role = row
    if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        return {"username": username, "role": role}
    return None


def user_count(db_path: str = DB_PATH) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
