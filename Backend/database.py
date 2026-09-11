import os
import re
import hashlib
import secrets
import sqlite3
from typing import Optional, Tuple, Any

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgrespassword@localhost:5432/recruitment_db"
)
SQLITE_DB_NAME = "recruitment.db"

# Try importing psycopg2
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import psycopg2.extensions
    import numpy as np

    psycopg2.extensions.register_adapter(np.float64, lambda val: psycopg2.extensions.AsIs(float(val)))
    psycopg2.extensions.register_adapter(np.float32, lambda val: psycopg2.extensions.AsIs(float(val)))
    psycopg2.extensions.register_adapter(np.int64, lambda val: psycopg2.extensions.AsIs(int(val)))
    psycopg2.extensions.register_adapter(np.int32, lambda val: psycopg2.extensions.AsIs(int(val)))
    PSYCOPG2_AVAILABLE = True
except (ImportError, Exception):
    PSYCOPG2_AVAILABLE = False


def _clean_param(p: Any) -> Any:
    if hasattr(p, "item"):
        val = p.item()
        if isinstance(val, float):
            return float(val)
        if isinstance(val, int):
            return int(val)
        return val
    return p


class PostgresCursorWrapper:
    def __init__(self, cursor):
        self.cursor = cursor
        self.lastrowid = None

    def execute(self, query: str, params: Any = None):
        # Convert SQLite style ? placeholders to Postgres %s
        clean_query = query.replace("?", "%s")
        # Replace CURRENT_TIMESTAMP functions or SQLite specifics if any
        clean_query = re.sub(r"\bAUTOINCREMENT\b", "", clean_query, flags=re.IGNORECASE)

        # For INSERT statements without RETURNING, append RETURNING id to capture lastrowid (except user_sessions)
        is_insert = clean_query.strip().upper().startswith("INSERT INTO")
        has_returning = "RETURNING" in clean_query.upper()
        is_session_table = "USER_SESSIONS" in clean_query.upper()

        if is_insert and not has_returning and not is_session_table:
            clean_query = clean_query.rstrip("; ") + " RETURNING id;"

        if params is not None:
            if isinstance(params, (list, tuple)):
                cleaned_params = tuple(_clean_param(p) for p in params)
                self.cursor.execute(clean_query, cleaned_params)
            else:
                self.cursor.execute(clean_query, _clean_param(params))
        else:
            self.cursor.execute(clean_query)

        if is_insert and not is_session_table:
            try:
                row = self.cursor.fetchone()
                if row and "id" in row:
                    self.lastrowid = row["id"]
            except Exception:
                pass

        return self

    def fetchone(self):
        try:
            return self.cursor.fetchone()
        except Exception:
            return None

    def fetchall(self):
        try:
            return self.cursor.fetchall()
        except Exception:
            return []

    @property
    def rowcount(self):
        return self.cursor.rowcount

    def close(self):
        self.cursor.close()


class PostgresConnectionWrapper:
    def __init__(self, raw_connection):
        self.connection = raw_connection

    def cursor(self):
        return PostgresCursorWrapper(self.connection.cursor(cursor_factory=RealDictCursor))

    def execute(self, query: str, params: Any = None):
        cur = self.cursor()
        cur.execute(query, params)
        return cur

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def close(self):
        self.connection.close()


def get_db_connection():
    """
    Returns a connection to Docker PostgreSQL database if available,
    otherwise falls back gracefully to local SQLite.
    """
    if PSYCOPG2_AVAILABLE and DATABASE_URL:
        try:
            raw_conn = psycopg2.connect(DATABASE_URL)
            return PostgresConnectionWrapper(raw_conn)
        except Exception:
            pass

    # Fallback to SQLite
    connection = sqlite3.connect(SQLITE_DB_NAME)
    connection.row_factory = sqlite3.Row
    return connection


def is_postgres() -> bool:
    conn = get_db_connection()
    is_pg = isinstance(conn, PostgresConnectionWrapper)
    conn.close()
    return is_pg


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash a password using PBKDF2-HMAC-SHA256 with a unique cryptographic salt."""
    if not salt:
        salt = secrets.token_hex(16)
    hash_bytes = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    )
    return hash_bytes.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a plain password against the stored hash and salt."""
    computed_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(computed_hash, stored_hash)


def _migrate_sqlite_to_postgres(pg_conn):
    """If PostgreSQL is fresh and SQLite exists, migrate data automatically."""
    if not os.path.exists(SQLITE_DB_NAME):
        return

    try:
        sqlite_conn = sqlite3.connect(SQLITE_DB_NAME)
        sqlite_conn.row_factory = sqlite3.Row
        
        # Check if users exist in Postgres
        pg_users = pg_conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()
        if pg_users and pg_users["count"] > 0:
            sqlite_conn.close()
            return

        # Migrate users
        sqlite_users = sqlite_conn.execute("SELECT * FROM users").fetchall()
        for u in sqlite_users:
            pg_conn.execute(
                "INSERT INTO users (id, name, email, password_hash, salt, role) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
                (u["id"], u["name"], u["email"], u["password_hash"], u["salt"], u["role"])
            )

        # Migrate candidates
        sqlite_candidates = sqlite_conn.execute("SELECT * FROM candidates").fetchall()
        for c in sqlite_candidates:
            c_keys = c.keys()
            phone = c["phone"] if "phone" in c_keys else ""
            user_id = c["user_id"] if "user_id" in c_keys else 1
            pg_conn.execute(
                """INSERT INTO candidates 
                   (id, name, email, phone, resume_filename, resume_text, embedding, skills, experience, summary, uploaded_at, status, user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING""",
                (c["id"], c["name"], c["email"], phone, c["resume_filename"], c["resume_text"], 
                 c["embedding"], c["skills"], c["experience"], c["summary"], c["uploaded_at"], c["status"], user_id)
            )

        # Migrate jobs
        sqlite_jobs = sqlite_conn.execute("SELECT * FROM jobs").fetchall()
        for j in sqlite_jobs:
            j_keys = j.keys()
            user_id = j["user_id"] if "user_id" in j_keys else 1
            pg_conn.execute(
                "INSERT INTO jobs (id, title, description, embedding, skills, keywords, user_id) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
                (j["id"], j["title"], j["description"], j["embedding"], j["skills"], j["keywords"], user_id)
            )

        # Migrate sessions
        try:
            sqlite_sessions = sqlite_conn.execute("SELECT * FROM user_sessions").fetchall()
            for s in sqlite_sessions:
                pg_conn.execute(
                    "INSERT INTO user_sessions (token, user_id, created_at) VALUES (?, ?, ?) ON CONFLICT (token) DO NOTHING",
                    (s["token"], s["user_id"], s["created_at"])
                )
        except Exception:
            pass

        # Migrate candidate matches
        try:
            sqlite_matches = sqlite_conn.execute("SELECT * FROM candidate_matches").fetchall()
            for m in sqlite_matches:
                pg_conn.execute(
                    """INSERT INTO candidate_matches 
                       (id, job_id, candidate_id, match_score, semantic_score, skill_score, keyword_score, matched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING""",
                    (m["id"], m["job_id"], m["candidate_id"], m["match_score"], m["semantic_score"], m["skill_score"], m["keyword_score"], m["matched_at"])
                )
        except Exception:
            pass

        # Reset serial sequences in PostgreSQL
        for table in ["users", "candidates", "jobs", "candidate_matches"]:
            try:
                pg_conn.execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), COALESCE(MAX(id), 1)) FROM {table}")
            except Exception:
                pass

        pg_conn.commit()
        sqlite_conn.close()
    except Exception as e:
        print(f"Notice: SQLite to Postgres migration skipped or encountered non-fatal notice: {e}")


def create_tables():
    conn = get_db_connection()

    if isinstance(conn, PostgresConnectionWrapper):
        # PostgreSQL Schema
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'recruiter',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT DEFAULT '',
                resume_filename TEXT NOT NULL,
                resume_text TEXT NOT NULL,
                embedding TEXT,
                skills TEXT NOT NULL DEFAULT '[]',
                experience REAL NOT NULL DEFAULT 0,
                summary TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'New',
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding TEXT,
                skills TEXT,
                keywords TEXT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidate_matches (
                id SERIAL PRIMARY KEY,
                job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
                match_score REAL NOT NULL,
                semantic_score REAL NOT NULL,
                skill_score REAL NOT NULL,
                keyword_score REAL NOT NULL,
                matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(job_id, candidate_id)
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS interview_questions (
                id SERIAL PRIMARY KEY,
                job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                answer TEXT NOT NULL
            );
        """)

        conn.commit()

        # Migrate data from SQLite if fresh Postgres
        _migrate_sqlite_to_postgres(conn)

        # Seed default recruiter account if no users exist
        users_count = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()
        if not users_count or users_count["count"] == 0:
            demo_pwd_hash, demo_salt = hash_password("admin123")
            conn.execute(
                "INSERT INTO users (name, email, password_hash, salt, role) VALUES (?, ?, ?, ?, ?)",
                ("Admin Recruiter", "admin@example.com", demo_pwd_hash, demo_salt, "recruiter")
            )
            conn.commit()

        conn.close()
        return

    # SQLite Schema (Fallback)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'recruiter',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("SELECT COUNT(*) AS count FROM users")
    if cursor.fetchone()["count"] == 0:
        demo_pwd_hash, demo_salt = hash_password("admin123")
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, salt, role)
            VALUES (?, ?, ?, ?, ?)
        """, ("Admin Recruiter", "admin@example.com", demo_pwd_hash, demo_salt, "recruiter"))

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            resume_filename TEXT NOT NULL,
            resume_text TEXT NOT NULL,
            embedding TEXT,
            status TEXT NOT NULL DEFAULT 'New',
            skills TEXT NOT NULL DEFAULT '[]',
            experience REAL NOT NULL DEFAULT 0,
            summary TEXT,
            uploaded_at TEXT,
            phone TEXT DEFAULT '',
            user_id INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            embedding TEXT,
            skills TEXT,
            keywords TEXT,
            user_id INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidate_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            candidate_id INTEGER NOT NULL,
            match_score REAL NOT NULL,
            semantic_score REAL NOT NULL,
            skill_score REAL NOT NULL,
            keyword_score REAL NOT NULL,
            matched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(job_id, candidate_id),
            FOREIGN KEY (job_id) REFERENCES jobs(id),
            FOREIGN KEY (candidate_id) REFERENCES candidates(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interview_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES jobs(id)
        )
    """)

    conn.commit()
    conn.close()
