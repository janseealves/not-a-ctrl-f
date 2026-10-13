"""Conexão SQLite, schema versionado e helpers mínimos."""

import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from not_a_ctrl_f.shared.config import EMBED_DIM

SCHEMA_VERSION = 1

# float16 = 2 bytes por dimensão; o CHECK pega vetor de dtype/dimensão errados.
VECTOR_BYTES = EMBED_DIM * 2

_SCHEMA_V1: tuple[str, ...] = (
    """
    CREATE TABLE meta(
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE files(
      id            INTEGER PRIMARY KEY,
      path          TEXT NOT NULL,
      path_key      TEXT NOT NULL UNIQUE,
      name          TEXT NOT NULL,
      ext           TEXT NOT NULL,
      size          INTEGER NOT NULL,
      mtime_ns      INTEGER NOT NULL,
      file_ref      INTEGER,
      volume_serial INTEGER,
      state         TEXT NOT NULL DEFAULT 'pending'
                    CHECK (state IN ('pending','indexing','done','skipped','failed')),
      error         TEXT,
      info          TEXT CHECK (info IS NULL OR json_valid(info)),  -- páginas, pixels, duração...
      attempts      INTEGER NOT NULL DEFAULT 0,
      indexed_at    REAL,
      created_at    REAL NOT NULL DEFAULT (unixepoch('subsec'))
    )
    """,
    "CREATE INDEX files_ref ON files(volume_serial, file_ref)",
    "CREATE INDEX files_state ON files(state)",
    f"""
    CREATE TABLE chunks(
      id        INTEGER PRIMARY KEY,
      file_id   INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
      -- name: nome + pastas | text: trecho de documento | page: texto de página de PDF
      -- page_image: página renderizada (scan/slide) | image | ocr: texto lido de imagem
      -- audio: janela de áudio | frame: quadro de vídeo
      kind      TEXT NOT NULL CHECK (kind IN
                  ('name','text','page','page_image','image','ocr','audio','frame')),
      -- modalidade do que foi embeddado; 'none' quando não há vetor
      modality  TEXT NOT NULL DEFAULT 'none'
                CHECK (modality IN ('none','text','image','audio','video')),
      seq       INTEGER NOT NULL,
      loc       REAL,                    -- página, offset de caractere ou segundo
      loc_end   REAL,
      text      TEXT,                    -- vai pro FTS; NULL em chunk só visual/sonoro
      vec       BLOB CHECK (vec IS NULL OR length(vec) = {VECTOR_BYTES}),
      CHECK ((vec IS NULL) = (modality = 'none')),
      UNIQUE(file_id, seq)
    )
    """,
    """
    CREATE VIRTUAL TABLE chunks_fts USING fts5(
      text,
      content='chunks',
      content_rowid='id',
      tokenize="unicode61 remove_diacritics 2",
      prefix='2 3'
    )
    """,
    """
    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
    """
    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END
    """,
    """
    CREATE TRIGGER chunks_au AFTER UPDATE OF text ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
      INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
)


def _migrate_to_v1(conn: sqlite3.Connection) -> None:
    for stmt in _SCHEMA_V1:
        conn.execute(stmt)


# Versão de destino -> função que migra da versão anterior para ela.
_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    1: _migrate_to_v1,
}


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _user_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection) -> None:
    version = _user_version(conn)
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Índice criado por versão mais nova (schema {version} > {SCHEMA_VERSION})"
        )
    if version == SCHEMA_VERSION:
        return
    with transaction(conn):
        # Relê já com o lock de escrita: outra conexão pode ter migrado antes.
        version = _user_version(conn)
        for target in range(version + 1, SCHEMA_VERSION + 1):
            _MIGRATIONS[target](conn)
        # PRAGMA não aceita parâmetros; o valor é um int interno.
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def connect(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
    path_str = str(path)
    if readonly:
        uri = Path(path_str).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=True)
    else:
        if path_str != ":memory:":
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path_str, isolation_level=None, check_same_thread=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA temp_store = MEMORY")
        if not readonly:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            migrate(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str | None) -> None:
    if value is None:
        conn.execute("DELETE FROM meta WHERE key = ?", (key,))
    else:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
