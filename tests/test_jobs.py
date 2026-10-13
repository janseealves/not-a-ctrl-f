import logging
import os
import threading
import time

import pytest
from test_indexer import FakeEmbedder, add, chunks, mk, mk_png

from not_a_ctrl_f.processing import indexer, sync
from not_a_ctrl_f.processing.jobs import Worker
from not_a_ctrl_f.processing.tracking import ONLINE_ERROR
from not_a_ctrl_f.processing.watcher import Watcher
from not_a_ctrl_f.shared import config, db
from not_a_ctrl_f.shared import embedder as embedder_module


def wait_until(cond, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    r.mkdir()
    return r


@pytest.fixture
def dbp(tmp_path):
    return tmp_path / "jobs.db"


@pytest.fixture
def conn(dbp):
    c = db.connect(dbp)
    yield c
    c.close()


@pytest.fixture
def workers():
    ws: list[Worker] = []
    yield ws
    for w in ws:
        w.stop()


@pytest.fixture
def start(dbp, root, workers):
    def _start(emb, **kw):
        kw.setdefault("idle_wait", 60.0)
        kw.setdefault("retry_delay", 0.2)
        kw.setdefault("embedder_retry", 0.2)
        w = Worker(dbp, emb, home=root, **kw)
        workers.append(w)
        return w.start()

    return _start


def file_row(conn, path):
    return conn.execute("SELECT * FROM files WHERE path_key=?", (db.path_key(path),)).fetchone()


def state(conn, path):
    r = file_row(conn, path)
    return r["state"] if r else None


def fts(conn, word):
    return conn.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (word,)
    ).fetchone()[0]


def worker_threads():
    return [t for t in threading.enumerate() if t.name == "nacf-worker"]


# 1
def test_processa_txt_e_png_e_notify(conn, root, start):
    txt = mk(root / "a.txt", "relatorio sobre girafas")
    png = mk_png(root / "b.png")
    add(conn, txt, root)
    add(conn, png, root)
    w = start(FakeEmbedder())
    assert wait_until(lambda: state(conn, txt) == "done" and state(conn, png) == "done")
    assert [(c["kind"], c["modality"]) for c in chunks(conn, file_row(conn, txt)["id"])] == [
        ("name", "none"), ("text", "text")]
    assert [(c["kind"], c["modality"]) for c in chunks(conn, file_row(conn, png)["id"])] == [
        ("name", "none"), ("image", "image")]
    assert fts(conn, "girafas") == 1
    assert w.wait_idle(10)

    novo = mk(root / "c.txt", "texto sobre elefantes")
    add(conn, novo, root)
    t0 = time.monotonic()
    w.notify()  # idle_wait=60: só o notify explica a rapidez
    assert wait_until(lambda: state(conn, novo) == "done", 5)
    assert time.monotonic() - t0 < 5
    assert fts(conn, "elefantes") == 1


# 2
def test_imagem_pequena_vira_skipped(conn, root, start):
    f = mk_png(root / "mini.png", 50)
    add(conn, f, root)
    start(FakeEmbedder())
    assert wait_until(lambda: state(conn, f) == "skipped")
    r = file_row(conn, f)
    assert "pequena" in r["error"]
    assert [c["kind"] for c in chunks(conn, r["id"])] == ["name"]


# 3
def test_docx_corrompido_vira_failed(conn, root, start):
    f = root / "ruim.docx"
    f.write_bytes(b"isto nao e um zip")
    add(conn, f, root)
    start(FakeEmbedder())
    assert wait_until(lambda: state(conn, f) == "failed")
    r = file_row(conn, f)
    assert r["error"] and [c["kind"] for c in chunks(conn, r["id"])] == ["name"]


# 4
def test_arquivo_muda_durante_embedding(dbp, conn, root, start):
    f = mk(root / "a.txt", "conteudo antigo")
    add(conn, f, root)
    fired = []

    def mudar():
        if fired:
            return
        fired.append(1)
        f.write_text("conteudo novo com a palavra zebra", encoding="utf-8")
        cc = db.connect(dbp)
        try:
            add(cc, f, root)
        finally:
            cc.close()

    emb = FakeEmbedder(on_call=mudar)
    start(emb)
    assert wait_until(lambda: state(conn, f) == "done")
    assert len(emb.calls) == 2  # a primeira escrita foi descartada e o arquivo reprocessado
    assert fts(conn, "zebra") == 1
    assert fts(conn, "antigo") == 0


# 5
def test_arquivo_apagado_durante_embedding(dbp, conn, root, start):
    f = mk(root / "a.txt", "vai sumir")
    add(conn, f, root)

    def apagar():
        if f.exists():
            f.unlink()
            cc = db.connect(dbp)
            try:
                sync.apply_path(cc, f, roots=[root])
            finally:
                cc.close()

    w = start(FakeEmbedder(on_call=apagar))
    assert wait_until(lambda: file_row(conn, f) is None)
    assert w.wait_idle(10)
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    g = mk(root / "b.txt", "continua vivo")
    add(conn, g, root)
    w.notify()
    assert wait_until(lambda: state(conn, g) == "done")


# 6
def test_permission_error_com_retentativa(conn, root, start, monkeypatch):
    f = mk(root / "a.txt", "arquivo travado no word")
    add(conn, f, root)
    real, n = indexer.build_rows, []

    def instavel(path, emb, **kw):
        n.append(1)
        if len(n) <= 2:
            raise PermissionError("em uso")
        return real(path, emb, **kw)

    monkeypatch.setattr(indexer, "build_rows", instavel)
    start(FakeEmbedder())
    assert wait_until(lambda: state(conn, f) == "done")
    assert file_row(conn, f)["attempts"] == 2


def test_permission_error_esgota_tentativas(conn, root, start, monkeypatch):
    f = mk(root / "a.txt", "sempre travado")
    add(conn, f, root)

    def sempre(path, emb, **kw):
        raise PermissionError("em uso")

    monkeypatch.setattr(indexer, "build_rows", sempre)
    start(FakeEmbedder(), max_attempts=3)
    assert wait_until(lambda: state(conn, f) == "failed")
    r = file_row(conn, f)
    assert r["attempts"] == 3 and "em uso" in r["error"]
    assert [c["kind"] for c in chunks(conn, r["id"])] == ["name"]


# 7
def test_embedder_fora_do_ar_nao_conta_tentativa(conn, root, start):
    f = mk(root / "a.txt", "servidor cai e volta")
    add(conn, f, root)
    emb = FakeEmbedder(fail_times=2)
    start(emb, max_attempts=2)
    assert wait_until(lambda: state(conn, f) == "done")
    assert file_row(conn, f)["attempts"] == 0
    assert len(emb.calls) == 3


# 8
def test_fingerprint_mudou_reprocessa(conn, root, start):
    f = mk(root / "a.txt", "indexado duas vezes")
    add(conn, f, root)
    w = start(FakeEmbedder("a"))
    assert wait_until(lambda: state(conn, f) == "done")
    first = file_row(conn, f)["indexed_at"]
    w.stop()

    emb_b = FakeEmbedder("b")
    start(emb_b)
    assert wait_until(lambda: len(emb_b.calls) >= 1 and state(conn, f) == "done")
    assert file_row(conn, f)["indexed_at"] > first
    assert db.get_meta(conn, indexer.FINGERPRINT_KEY) == "b"


# 9
def test_recupera_linha_indexing(conn, root, start):
    f = mk(root / "a.txt", "crash no meio")
    r = add(conn, f, root)
    conn.execute("UPDATE files SET state='indexing' WHERE id=?", (r["id"],))
    start(FakeEmbedder())
    assert wait_until(lambda: state(conn, f) == "done")


# 10
def test_online_ganha_so_o_chunk_name(conn, root, start):
    f = mk(root / "nuvem.txt", "so online")
    r = add(conn, f, root)
    conn.execute("UPDATE files SET state='skipped', error=? WHERE id=?", (ONLINE_ERROR, r["id"]))
    emb = FakeEmbedder()
    start(emb)
    assert wait_until(lambda: len(chunks(conn, r["id"])) == 1)
    c = chunks(conn, r["id"])[0]
    assert c["kind"] == "name" and "nuvem.txt" in c["text"] and c["vec"] is None
    assert state(conn, f) == "skipped" and emb.calls == []


# 11
def test_stop_com_fila_cheia(conn, root, start):
    files = [mk(root / f"f{i:02}.txt", f"texto numero {i}") for i in range(40)]
    for f in files:
        add(conn, f, root)
    w = start(FakeEmbedder(delay=0.05))
    assert wait_until(lambda: conn.execute(
        "SELECT count(*) FROM files WHERE state='done'").fetchone()[0] >= 1)
    t0 = time.monotonic()
    w.stop(timeout=10)
    assert time.monotonic() - t0 < 5
    assert conn.execute("SELECT count(*) FROM files WHERE state='indexing'").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM files WHERE state='pending'").fetchone()[0] > 0
    assert not worker_threads()
    w.stop()  # idempotente


# 12
def test_watcher_e_worker_juntos(dbp, conn, root, start, caplog):
    caplog.set_level(logging.WARNING)
    w = start(FakeEmbedder())
    with Watcher(dbp, [root], debounce=0.3, rescan_interval=None, on_pending=w.notify):
        f = mk(root / "novo.txt", "primeira versao com a palavra abacate")
        assert wait_until(lambda: file_row(conn, f) and state(conn, f) == "done")
        assert fts(conn, "abacate") == 1
        first = file_row(conn, f)["indexed_at"]

        time.sleep(0.05)
        f.write_text("segunda versao com a palavra manga e bem maior", encoding="utf-8")
        assert wait_until(lambda: fts(conn, "manga") == 1 and state(conn, f) == "done")
        assert fts(conn, "abacate") == 0
        assert file_row(conn, f)["indexed_at"] > first

        lote = [mk(root / f"l{i:02}.txt", f"arquivo do lote {i}") for i in range(15)]
        assert wait_until(lambda: all(state(conn, p) == "done" for p in lote), 20)
    assert not [r for r in caplog.records if "locked" in r.getMessage().lower()
                or (r.exc_info and "locked" in str(r.exc_info[1]).lower())]
    assert w.wait_idle(10)


# 14
def test_renomeado_durante_o_processamento(dbp, conn, root, start):
    f = mk(root / "antes.txt", "conteudo qualquer")
    g = root / "depois.txt"
    add(conn, f, root)
    fired = []

    def renomear():
        if fired:
            return
        fired.append(1)
        os.rename(f, g)
        cc = db.connect(dbp)
        try:
            sync.apply_move(cc, f, g, home=root, roots=[root])
        finally:
            cc.close()

    emb = FakeEmbedder(on_call=renomear)
    start(emb)
    assert wait_until(lambda: file_row(conn, g) and state(conn, g) == "done"
                      and "depois.txt" in chunks(conn, file_row(conn, g)["id"])[0]["text"])
    assert fts(conn, "antes") == 0
    assert len(emb.calls) == 1  # só o name foi corrigido; nada foi embeddado de novo


# 13
_REAL = (
    config.llama_server_path().exists()
    and (config.models_dir() / config.TEXT_MODEL_FILE).exists()
    and (config.models_dir() / config.MEDIA_MODEL_FILE).exists()
)


@pytest.mark.integration
@pytest.mark.skipif(not _REAL, reason="llama-server/modelos não instalados")
def test_integracao_real(conn, root, start):
    txt = mk(root / "cao.txt", "O cachorro late alto no quintal toda vez que o carteiro passa na rua.")
    png = mk_png(root / "azul.png", 512, (10, 40, 200))
    add(conn, txt, root)
    add(conn, png, root)
    with embedder_module.LlamaServer() as srv:
        emb = embedder_module.Embedder(srv.base_url)
        w = start(emb)
        try:
            assert wait_until(
                lambda: state(conn, txt) == "done" and state(conn, png) == "done", 180)
        finally:
            w.stop()
        vecs = {}
        for path, kind in ((txt, "text"), (png, "image")):
            (c,) = conn.execute(
                "SELECT vec FROM chunks WHERE file_id=? AND kind=?",
                (file_row(conn, path)["id"], kind)).fetchall()
            assert len(c["vec"]) == 1536
            vecs[kind] = embedder_module.from_blob(c["vec"])
        q = emb.embed_query("cachorro latindo")
        assert float(q @ vecs["text"]) > float(q @ vecs["image"])
