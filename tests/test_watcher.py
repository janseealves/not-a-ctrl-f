import os
import shutil
import threading
import time

import pytest

from not_a_ctrl_f.processing.watcher import Watcher
from not_a_ctrl_f.shared import db


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    r.mkdir()
    return r


def mk(path, text="conteudo"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def row(conn, path):
    return conn.execute("SELECT * FROM files WHERE path_key = ?", (db.path_key(path),)).fetchone()


def count(conn):
    return conn.execute("SELECT count(*) FROM files").fetchone()[0]


def wait_until(cond, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


# ---------- Watcher ao vivo ----------

DEBOUNCE = 0.3


@pytest.fixture
def dbp(tmp_path):
    return tmp_path / "live.db"


def make_watcher(dbp, root, **kw):
    kw.setdefault("debounce", DEBOUNCE)
    kw.setdefault("rescan_interval", None)
    return Watcher(dbp, [root], **kw)


def writer_threads():
    return [t for t in threading.enumerate() if t.name == "nacf-watcher-writer"]


def test_watcher_criar_editar_apagar(dbp, root):
    pre = mk(root / "pre.txt", "ja existia")
    calls = []
    with make_watcher(dbp, root, on_pending=lambda: calls.append(1)) as w:
        c = db.connect(dbp)
        try:
            assert row(c, pre) is not None  # scan inicial é síncrono
            n0 = len(calls)
            assert n0 >= 1

            f = mk(root / "novo.txt", "oi")
            assert wait_until(lambda: row(c, f) is not None)
            assert row(c, f)["state"] == "pending"
            assert wait_until(lambda: len(calls) > n0)

            c.execute("UPDATE files SET state='done', attempts=1 WHERE path_key=?", (db.path_key(f),))
            old_mtime = row(c, f)["mtime_ns"]
            time.sleep(0.05)
            f.write_text("oi, agora maior", encoding="utf-8")
            assert wait_until(lambda: row(c, f)["mtime_ns"] != old_mtime)
            r = row(c, f)
            assert r["state"] == "pending" and r["attempts"] == 0 and r["size"] == 15

            f.unlink()
            assert wait_until(lambda: row(c, f) is None)
            assert w.wait_idle(10)
            assert row(c, pre) is not None
        finally:
            c.close()
    assert not writer_threads()


def test_watcher_rename_mover_pasta(dbp, root):
    f = mk(root / "a.txt", "dados")
    sub = root / "sub"
    sub.mkdir()
    with make_watcher(dbp, root) as w:
        c = db.connect(dbp)
        try:
            fid = row(c, f)["id"]

            f2 = root / "b.txt"
            os.rename(f, f2)
            assert wait_until(lambda: row(c, f2) is not None)
            assert row(c, f2)["id"] == fid and row(c, f) is None

            f3 = sub / "b.txt"  # o Windows entrega como deleted + created
            os.rename(f2, f3)
            assert wait_until(lambda: row(c, f3) is not None and row(c, f2) is None)
            assert row(c, f3)["id"] == fid

            sub2 = root / "sub2"
            os.rename(sub, sub2)
            f4 = sub2 / "b.txt"
            assert wait_until(lambda: row(c, f4) is not None)
            assert row(c, f4)["id"] == fid and row(c, f3) is None
            assert row(c, f4)["path"] == str(f4)

            shutil.rmtree(sub2)
            assert wait_until(lambda: count(c) == 0)
            assert w.wait_idle(10)
        finally:
            c.close()


def test_watcher_pasta_movida_de_fora(dbp, root, tmp_path):
    fora = tmp_path / "fora"
    mk(fora / "d" / "a.txt")
    mk(fora / "d" / "sub" / "b.txt")
    with make_watcher(dbp, root):
        c = db.connect(dbp)
        try:
            os.rename(fora / "d", root / "d")
            assert wait_until(lambda: count(c) == 2)
        finally:
            c.close()


def test_watcher_salvar_do_word(dbp, root):
    doc = mk(root / "relatorio.docx", "versao 1")
    with make_watcher(dbp, root) as w:
        c = db.connect(dbp)
        try:
            r0 = row(c, doc)
            c.execute("UPDATE files SET state='done' WHERE id=?", (r0["id"],))
            mk(root / "~WRD0000.tmp", "versao 2, bem maior")
            os.replace(doc, root / "~WRL0001.tmp")
            os.replace(root / "~WRD0000.tmp", doc)
            os.remove(root / "~WRL0001.tmp")

            assert wait_until(lambda: row(c, doc)["size"] == len("versao 2, bem maior"))
            assert w.wait_idle(10)
            r = row(c, doc)
            assert r["id"] == r0["id"] and r["state"] == "pending"
            assert count(c) == 1
        finally:
            c.close()


def test_watcher_stop_idempotente_e_rescan(dbp, root):
    f = mk(root / "a.txt")
    w = Watcher(dbp, [root], debounce=DEBOUNCE, rescan_interval=0.5).start()
    c = db.connect(dbp)
    try:
        c.execute("DELETE FROM files")  # o rescan periódico deve repor
        assert wait_until(lambda: row(c, f) is not None)
    finally:
        c.close()
    w.stop()
    w.stop()
    assert not writer_threads()


def test_watcher_on_pending_com_erro_nao_derruba(dbp, root):
    def boom():
        raise RuntimeError("falha")

    with make_watcher(dbp, root, on_pending=boom):
        c = db.connect(dbp)
        try:
            f = mk(root / "a.txt")
            assert wait_until(lambda: row(c, f) is not None)
            g = mk(root / "b.txt")
            assert wait_until(lambda: row(c, g) is not None)
        finally:
            c.close()
