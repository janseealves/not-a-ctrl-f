import hashlib
import json
import time

import numpy as np
import pytest
from PIL import Image

from not_a_ctrl_f.processing import indexer, sync
from not_a_ctrl_f.shared import db
from not_a_ctrl_f.shared.embedder import Embedding, EmbedderError


# ---------- apoio (também usado por test_jobs.py) ----------


def _vec(key: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).normal(size=768).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """Mesma interface pública do Embedder, com vetores determinísticos e ganchos de teste."""

    def __init__(self, fp="fake", *, fail_marker=None, fail_times=0, on_call=None, delay=0.0):
        self._fp = fp
        self.fail_marker = fail_marker  # item cujo texto contém isto vem com vector None
        self.fail_times = fail_times  # EmbedderError nas N primeiras chamadas
        self.on_call = on_call  # callback executado durante cada chamada
        self.delay = delay
        self.calls: list[tuple[str, list, dict]] = []

    @property
    def fingerprint(self) -> str:
        return self._fp

    def _enter(self, kind, items, **kw):
        self.calls.append((kind, list(items), kw))
        if self.delay:
            time.sleep(self.delay)
        if self.on_call:
            self.on_call()
        if self.fail_times > 0:
            self.fail_times -= 1
            raise EmbedderError("servidor fora do ar")

    def embed_query(self, text):
        return _vec("q:" + text)

    def embed_documents(self, items):
        self._enter("text", items)
        return [
            Embedding(None, "recusado")
            if self.fail_marker and self.fail_marker in t
            else Embedding(_vec(f"{title}|{t}"))
            for t, title in items
        ]

    def embed_images(self, images, budget=140):
        self._enter("image", images, budget=budget)
        return [Embedding(_vec(hashlib.sha256(b).hexdigest())) for b in images]

    def embed_audio(self, clips):
        self._enter("audio", clips)
        return [Embedding(_vec(hashlib.sha256(b).hexdigest())) for b in clips]


def mk(path, text="conteudo de teste"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def mk_png(path, side=512, color=(200, 30, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (side, side), color).save(path)
    return path


def add(conn, path, root):
    """Insere/atualiza a linha como o watcher faria."""
    sync.apply_path(conn, path, roots=[root])
    return conn.execute("SELECT * FROM files WHERE path_key=?", (db.path_key(path),)).fetchone()


def claim(conn, fid):
    conn.execute("UPDATE files SET state='indexing' WHERE id=?", (fid,))
    return conn.execute("SELECT * FROM files WHERE id=?", (fid,)).fetchone()


def chunks(conn, fid):
    return conn.execute("SELECT * FROM chunks WHERE file_id=? ORDER BY seq", (fid,)).fetchall()


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    r.mkdir()
    return r


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "idx.db")
    yield c
    c.close()


# ---------- build_rows ----------


def test_build_rows_texto(root):
    f = mk(root / "nota.txt", "o gato subiu no telhado")
    emb = FakeEmbedder()
    rows, info = indexer.build_rows(f, emb, home=root)
    assert rows[0][:3] == ("name", "none", 0) and rows[0][6] is None
    kind, modality, seq, _loc, _end, text, vec = rows[1]
    assert (kind, modality, seq, text) == ("text", "text", 1, "o gato subiu no telhado")
    assert isinstance(vec, bytes) and len(vec) == 1536
    assert emb.calls == [("text", [("o gato subiu no telhado", "nota.txt")], {})]
    assert info["chunks"] == 1 and "embed_errors" not in info


def test_build_rows_imagem(root):
    f = mk_png(root / "foto.png", 512)
    emb = FakeEmbedder()
    rows, info = indexer.build_rows(f, emb, home=root)
    assert [r[0] for r in rows] == ["name", "image"]
    assert rows[1][1] == "image" and len(rows[1][6]) == 1536 and rows[1][5] is None
    assert emb.calls[0][0] == "image" and emb.calls[0][2] == {"budget": 140}
    assert info["width"] == 512


def test_build_rows_falha_de_item(root):
    f = mk(root / "a.txt", "texto com RUIM dentro")
    rows, info = indexer.build_rows(f, FakeEmbedder(fail_marker="RUIM"), home=root)
    assert rows[1][:3] == ("text", "none", 1) and rows[1][6] is None
    assert rows[1][5] == "texto com RUIM dentro"  # o texto continua para o FTS
    assert info["embed_errors"] == 1 and info["embed_error_sample"] == "recusado"


# ---------- write_rows ----------


def test_write_rows(root, conn):
    f = mk(root / "a.txt", "palavra raríssima aqui")
    r = claim(conn, add(conn, f, root)["id"])
    rows, info = indexer.build_rows(f, FakeEmbedder(), home=root)

    assert indexer.write_rows(conn, r["id"], r["size"], r["mtime_ns"], rows, info)
    after = conn.execute("SELECT * FROM files WHERE id=?", (r["id"],)).fetchone()
    assert after["state"] == "done" and after["error"] is None and after["indexed_at"]
    assert json.loads(after["info"])["chunks"] == 1
    assert len(chunks(conn, r["id"])) == 2
    hit = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'rarissima'"
    ).fetchall()
    assert len(hit) == 1

    # segunda escrita substitui, sem duplicar nem violar UNIQUE(file_id, seq)
    conn.execute("UPDATE files SET state='indexing' WHERE id=?", (r["id"],))
    assert indexer.write_rows(conn, r["id"], r["size"], r["mtime_ns"], rows, info)
    assert len(chunks(conn, r["id"])) == 2
    assert conn.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'rarissima'").fetchone()[0] == 1


def test_write_rows_descarta_se_mudou_ou_sumiu(root, conn):
    f = mk(root / "a.txt", "um texto")
    r = claim(conn, add(conn, f, root)["id"])
    rows, info = indexer.build_rows(f, FakeEmbedder(), home=root)

    assert not indexer.write_rows(conn, r["id"], r["size"] + 1, r["mtime_ns"], rows, info)
    assert not indexer.write_rows(conn, r["id"], r["size"], r["mtime_ns"] + 1, rows, info)
    conn.execute("UPDATE files SET state='pending' WHERE id=?", (r["id"],))  # watcher invalidou
    assert not indexer.write_rows(conn, r["id"], r["size"], r["mtime_ns"], rows, info)
    assert chunks(conn, r["id"]) == []
    assert conn.execute("SELECT state FROM files WHERE id=?", (r["id"],)).fetchone()[0] == "pending"

    conn.execute("DELETE FROM files WHERE id=?", (r["id"],))
    assert not indexer.write_rows(conn, r["id"], r["size"], r["mtime_ns"], rows, info)
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


# ---------- ensure_fingerprint ----------


def test_ensure_fingerprint(root, conn):
    assert indexer.ensure_fingerprint(conn, "a") is False  # índice novo: só grava
    assert db.get_meta(conn, indexer.FINGERPRINT_KEY) == "a"
    assert indexer.ensure_fingerprint(conn, "a") is False

    ids = {}
    for name, state in [("d", "done"), ("f", "failed"), ("s", "skipped")]:
        r = add(conn, mk(root / f"{name}.txt", name), root)
        conn.execute("UPDATE files SET state=?, attempts=2, error='x' WHERE id=?", (state, r["id"]))
        ids[state] = r["id"]

    assert indexer.ensure_fingerprint(conn, "b") is True
    assert db.get_meta(conn, indexer.FINGERPRINT_KEY) == "b"
    got = {r["id"]: r for r in conn.execute("SELECT * FROM files")}
    for state in ("done", "failed"):
        assert got[ids[state]]["state"] == "pending"
        assert got[ids[state]]["attempts"] == 0 and got[ids[state]]["error"] is None
    assert got[ids["skipped"]]["state"] == "skipped"
