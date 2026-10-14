import threading
import time

import numpy as np
import pytest

from not_a_ctrl_f.query.search import MARK_END, MARK_START, Searcher, index_status
from not_a_ctrl_f.shared import db
from not_a_ctrl_f.shared.embedder import EmbedderError, to_blob


def e(i, j=None, a=1.0):
    """Vetor unitário: e_i, ou mistura a*e_i + sqrt(1-a²)*e_j."""
    v = np.zeros(768, np.float32)
    v[i] = a
    if j is not None:
        v[j] = (1 - a * a) ** 0.5
    return v


class QEmbedder:
    def __init__(self, vec=None, error=None):
        self.vec = vec
        self.error = error
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        if self.error:
            raise EmbedderError(self.error)
        return self.vec


def add(conn, name, chunks=(), ext="txt", state="done"):
    """chunks: iterável de (texto|None, vetor|None)."""
    cur = conn.execute(
        "INSERT INTO files(path, path_key, name, ext, size, mtime_ns, state, indexed_at) "
        "VALUES (?,?,?,?,0,0,?,?)",
        (f"/x/{name}.{ext}", f"/x/{name}.{ext}", f"{name}.{ext}", ext, state, time.time()),
    )
    fid = cur.lastrowid
    put_chunks(conn, fid, chunks)
    return fid


def put_chunks(conn, fid, chunks):
    for seq, (text, vec) in enumerate(chunks):
        conn.execute(
            "INSERT INTO chunks(file_id, kind, modality, seq, loc, text, vec) VALUES (?,?,?,?,?,?,?)",
            (fid, "text", "none" if vec is None else "text", seq, float(seq), text,
             None if vec is None else to_blob(vec)),
        )


@pytest.fixture
def path(tmp_path):
    return tmp_path / "idx.db"


@pytest.fixture
def conn(path):
    c = db.connect(path)
    yield c
    c.close()


@pytest.fixture
def make(path):
    made = []

    def _make(embedder=None, **kw):
        s = Searcher(path, embedder, **kw)
        made.append(s)
        return s

    yield _make
    for s in made:
        s.close()


def names(res):
    return [h.name for h in res.hits]


def test_palavra_chave(conn, make):
    add(conn, "a", [("O relatório anual de vendas", None)])
    add(conn, "b", [("receita de bolo", None)])
    s = make()
    res = s.search("vendas")
    assert names(res) == ["a.txt"]
    h = res.hits[0]
    assert MARK_START in h.snippet and MARK_END in h.snippet
    assert h.matched == ("keyword",) and h.kind == "text"
    assert res.semantic is False and res.error is None and res.took_ms > 0
    assert names(s.search("relatorio")) == ["a.txt"]  # sem acento casa com acento
    assert names(s.search("relat")) == ["a.txt"]  # prefixo
    assert names(s.search("bolo receita")) == ["b.txt"]


def test_fallback_and_para_or(conn, make):
    add(conn, "a", [("gatos e cachorros", None)])
    add(conn, "b", [("pássaros", None)])
    s = make()
    assert set(names(s.search("gatos pássaros"))) == {"a.txt", "b.txt"}
    assert names(s.search("gatos zebra")) == ["a.txt"]


@pytest.mark.parametrize("q", ['"', "-foo", "AND", "a:b", "NEAR(", "", "!!!", "foo^ bar*", "'; DROP"])
def test_entradas_hostis(conn, make, q):
    add(conn, "a", [("foo bar", None)])
    emb = QEmbedder(e(0))
    res = make(emb).search(q)
    assert isinstance(res.hits, list)
    if not any(ch.isalnum() for ch in q):
        assert res.hits == [] and emb.calls == 0


def test_semantica(conn, make):
    add(conn, "longe", [("alfa", e(2))])
    add(conn, "perto", [("beta", e(0))])
    add(conn, "meio", [("gama", e(0, 1, 0.6))])
    res = make(QEmbedder(e(0))).search("zzz")
    assert names(res) == ["perto.txt", "meio.txt", "longe.txt"]
    assert res.semantic is True and res.error is None
    assert all(h.matched == ("semantic",) for h in res.hits)
    assert res.hits[0].snippet == "beta" and res.hits[0].kind == "text"


def test_sem_semantica(conn, make):
    add(conn, "a", [("palavra", e(0))])
    emb = QEmbedder(e(0))
    r1 = make(emb).search("palavra", keyword_only=True)
    r2 = make(None).search("palavra")
    for r in (r1, r2):
        assert r.semantic is False and r.error is None
        assert names(r) == ["a.txt"] and r.hits[0].matched == ("keyword",)
    assert emb.calls == 0


def test_embedder_falha(conn, make):
    add(conn, "a", [("palavra", e(0))])
    res = make(QEmbedder(error="fora do ar")).search("palavra")
    assert res.semantic is False and res.error == "fora do ar"
    assert names(res) == ["a.txt"]


def test_fusao(conn, make):
    add(conn, "ambos", [("tema comum", e(0, 1, 0.9))])
    add(conn, "so_kw", [("tema tema tema comum", None)])
    add(conn, "so_sem", [("outra coisa", e(0))])
    res = make(QEmbedder(e(0))).search("tema")
    by = {h.name: h for h in res.hits}
    assert res.hits[0].name == "ambos.txt"
    assert by["ambos.txt"].matched == ("keyword", "semantic")
    assert by["so_kw.txt"].matched == ("keyword",)
    assert by["so_sem.txt"].matched == ("semantic",)
    assert by["ambos.txt"].score > by["so_sem.txt"].score
    assert len(make(QEmbedder(e(0))).search("tema", limit=1).hits) == 1


def test_fusao_matched_semantico_so_em_uma(conn, make):
    add(conn, "a", [("tema", None)])
    add(conn, "b", [("nada", e(0))])
    res = make(QEmbedder(e(0))).search("tema")
    by = {h.name: h for h in res.hits}
    assert by["a.txt"].matched == ("keyword",) and by["b.txt"].matched == ("semantic",)


def test_filtro_tipo(conn, make):
    add(conn, "d", [("tema", e(0))], ext="txt")
    add(conn, "i", [("tema", e(0))], ext="png")
    add(conn, "s", [("tema", e(0))], ext="mp3")
    s = make(QEmbedder(e(0)))
    assert names(s.search("tema", type="image")) == ["i.png"]
    assert names(s.search("tema", type="audio")) == ["s.mp3"]
    assert names(s.search("tema", type="doc")) == ["d.txt"]
    assert names(s.search("zzz", type="image")) == ["i.png"]  # só semântica
    assert len(s.search("tema").hits) == 3
    with pytest.raises(ValueError):
        s.search("tema", type="video")


def test_cache_incremental(path, conn, make):
    f1 = add(conn, "um", [("a", e(1))])
    s = make(QEmbedder(e(0)))
    assert names(s.search("zzz")) == ["um.txt"]

    other = db.connect(path)
    f2 = add(other, "dois", [("b", e(0))])
    assert names(s.search("zzz")) == ["dois.txt", "um.txt"]

    other.execute("DELETE FROM files WHERE id=?", (f2,))
    assert names(s.search("zzz")) == ["um.txt"]

    # regravar: vetor novo + indexed_at novo
    other.execute("DELETE FROM chunks WHERE file_id=?", (f1,))
    put_chunks(other, f1, [("a", e(0))])
    other.execute("UPDATE files SET indexed_at=indexed_at+1 WHERE id=?", (f1,))
    add(other, "tres", [("c", e(0, 1, 0.5))])
    res = s.search("zzz")
    assert names(res) == ["um.txt", "tres.txt"]
    other.close()


def test_arquivo_sem_vetor_ou_sem_indexed_at(conn, make):
    add(conn, "sem", [("tema", None)])
    conn.execute(
        "INSERT INTO files(path, path_key, name, ext, size, mtime_ns) VALUES ('p','p','n.txt','txt',0,0)"
    )
    res = make(QEmbedder(e(0))).search("zzz")
    assert res.hits == [] and res.semantic is True


def test_max_vector_files(conn, make):
    for i in range(5):
        add(conn, f"f{i}", [("x", e(0, 1, 0.9 - i * 0.1))])
    res = make(QEmbedder(e(0)), max_vector_files=2).search("zzz")
    assert names(res) == ["f0.txt", "f1.txt"]


def test_threads(path, conn, make):
    for i in range(20):
        add(conn, f"f{i}", [(f"tema numero{i}", e(i % 5, (i + 1) % 5, 0.8))])
    s = make(QEmbedder(e(0)))
    errors = []
    stop = threading.Event()

    def writer():
        w = db.connect(path)
        n = 0
        try:
            while not stop.is_set():
                add(w, f"w{n}", [(f"tema w{n}", e(n % 7))])
                n += 1
                time.sleep(0.005)
        except Exception as ex:
            errors.append(ex)
        finally:
            w.close()

    def reader():
        try:
            for _ in range(30):
                assert s.search("tema").hits
                s.search("numero3", type="doc")
        except Exception as ex:
            errors.append(ex)

    wt = threading.Thread(target=writer)
    wt.start()
    ts = [threading.Thread(target=reader) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    stop.set()
    wt.join()
    assert errors == []


def test_index_status(conn):
    st = index_status(conn)
    assert st == {"files": 0, "by_state": dict.fromkeys(
        ("pending", "indexing", "done", "skipped", "failed"), 0), "chunks": 0}
    add(conn, "a", [("x", None), ("y", e(0))])
    add(conn, "b", state="failed")
    add(conn, "c", state="done")
    st = index_status(conn)
    assert st["files"] == 3 and st["chunks"] == 2
    assert st["by_state"]["done"] == 2 and st["by_state"]["failed"] == 1
    assert st["by_state"]["pending"] == 0


def test_warm(conn, make):
    add(conn, "a", [("x", e(0))])
    emb = QEmbedder(e(0))
    s = make(emb)
    s.warm()
    assert emb.calls == 0
    assert len(s._vectors._mat) == 1
