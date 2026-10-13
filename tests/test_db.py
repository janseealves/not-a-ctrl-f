import sqlite3

import pytest

from not_a_ctrl_f.shared import db


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "sub" / "index.db")
    yield c
    c.close()


def add_file(conn, path="C:\\docs\\a.txt"):
    cur = conn.execute(
        "INSERT INTO files(path, path_key, name, ext, size, mtime_ns) VALUES (?,?,?,?,?,?)",
        (path, db.path_key(path), "a.txt", "txt", 1, 1),
    )
    return cur.lastrowid


def add_chunk(conn, file_id, text, seq=0):
    cur = conn.execute(
        "INSERT INTO chunks(file_id, kind, seq, text) VALUES (?, 'text', ?, ?)",
        (file_id, seq, text),
    )
    return cur.lastrowid


def match(conn, q):
    rows = conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (q,))
    return [r[0] for r in rows]


def test_connect_creates_db(tmp_path):
    p = tmp_path / "nested" / "index.db"
    c = db.connect(p)
    try:
        assert p.exists()
        assert c.execute("PRAGMA user_version").fetchone()[0] == 1
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        c.close()


def test_reopen_idempotent(tmp_path):
    p = tmp_path / "index.db"
    c = db.connect(p)
    fid = add_file(c)
    c.close()
    c = db.connect(p)
    try:
        assert c.execute("SELECT id FROM files").fetchone()[0] == fid
    finally:
        c.close()


def test_newer_version_rejected(tmp_path):
    p = tmp_path / "index.db"
    raw = sqlite3.connect(p)
    raw.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
    raw.close()
    with pytest.raises(RuntimeError):
        db.connect(p)


def test_fts_match(conn):
    cid = add_chunk(conn, add_file(conn), "Relatório anual de vendas")
    assert match(conn, "relatorio") == [cid]
    assert match(conn, "rel*") == [cid]


def test_fts_update(conn):
    fid = add_file(conn)
    cid = add_chunk(conn, fid, "Relatório anual")
    conn.execute("UPDATE chunks SET text = 'contrato mensal' WHERE id = ?", (cid,))
    assert match(conn, "relatorio") == []
    assert match(conn, "contrato") == [cid]


def test_delete_file_cascades_to_fts(conn):
    fid = add_file(conn)
    add_chunk(conn, fid, "Relatório anual")
    conn.execute("DELETE FROM files WHERE id = ?", (fid,))
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert match(conn, "relatorio") == []


def test_path_key_and_unique(conn):
    assert db.path_key(r"C:\Users\Foo\A.TXT") == db.path_key("c:/users/foo/a.txt")
    add_file(conn, r"C:\Users\Foo\A.TXT")
    with pytest.raises(sqlite3.IntegrityError):
        add_file(conn, r"C:\Users\Foo\a.txt")


def test_transaction_rollback(conn):
    with pytest.raises(ValueError):
        with db.transaction(conn):
            add_file(conn)
            raise ValueError("boom")
    assert conn.execute("SELECT count(*) FROM files").fetchone()[0] == 0
    with db.transaction(conn):
        add_file(conn)
    assert conn.execute("SELECT count(*) FROM files").fetchone()[0] == 1


def test_readonly_rejects_writes(tmp_path):
    p = tmp_path / "index.db"
    db.connect(p).close()
    ro = db.connect(p, readonly=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("INSERT INTO meta(key, value) VALUES ('a', 'b')")
    finally:
        ro.close()


def test_meta(conn):
    assert db.get_meta(conn, "k") is None
    db.set_meta(conn, "k", "v1")
    assert db.get_meta(conn, "k") == "v1"
    db.set_meta(conn, "k", "v2")
    assert db.get_meta(conn, "k") == "v2"
    db.set_meta(conn, "k", None)
    assert db.get_meta(conn, "k") is None


def add_vec_chunk(conn, file_id, seq, kind, modality, vec, text=None):
    cur = conn.execute(
        "INSERT INTO chunks(file_id, kind, modality, seq, text, vec) VALUES (?,?,?,?,?,?)",
        (file_id, kind, modality, seq, text, vec),
    )
    return cur.lastrowid


def test_image_chunk_without_text(conn):
    fid = add_file(conn)
    add_vec_chunk(conn, fid, 0, "image", "image", b"\0" * db.VECTOR_BYTES)
    add_vec_chunk(conn, fid, 1, "ocr", "none", None, text="Nota fiscal")
    assert match(conn, "nota") != []
    conn.execute("DELETE FROM files WHERE id = ?", (fid,))
    assert match(conn, "nota") == []


def test_vector_size_check(conn):
    fid = add_file(conn)
    with pytest.raises(sqlite3.IntegrityError):  # float32 por engano
        add_vec_chunk(conn, fid, 0, "text", "text", b"\0" * (db.VECTOR_BYTES * 2), "x")


def test_vector_modality_consistency(conn):
    fid = add_file(conn)
    with pytest.raises(sqlite3.IntegrityError):  # vetor sem modalidade
        add_vec_chunk(conn, fid, 0, "text", "none", b"\0" * db.VECTOR_BYTES, "x")
    with pytest.raises(sqlite3.IntegrityError):  # modalidade sem vetor
        add_vec_chunk(conn, fid, 1, "image", "image", None)


def test_info_must_be_json(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO files(path, path_key, name, ext, size, mtime_ns, info) "
            "VALUES ('x','x','x','',0,0,'not json')"
        )


def test_state_check(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO files(path, path_key, name, ext, size, mtime_ns, state) "
            "VALUES ('x','x','x','',0,0,'bogus')"
        )
