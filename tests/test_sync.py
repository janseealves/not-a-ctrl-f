import ctypes
import os
import threading
import time
from types import SimpleNamespace

import pytest

from not_a_ctrl_f.processing import extractors, sync
from not_a_ctrl_f.processing.sync import (
    apply_move,
    apply_path,
    move_tree,
    remove_tree,
    scan,
)
from not_a_ctrl_f.processing.tracking import should_track
from not_a_ctrl_f.shared import config, db

windows_only = pytest.mark.skipif(os.name != "nt", reason="atributos do Windows")


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


def fts(conn, q):
    return [r[0] for r in conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (q,))]


# ---------- should_track ----------

def test_should_track_basico(root):
    roots = [root]
    assert should_track(mk(root / "a.txt"), roots=roots)
    assert not should_track(mk(root / "a.exe"), roots=roots)
    for name in ("~$doc.docx", "~WRL0001.tmp", "x.crdownload", "desktop.ini", "x.txt.part"):
        assert not should_track(mk(root / name), roots=roots), name
    for d in ("node_modules", ".git", ".hidden", "__pycache__"):
        assert not should_track(mk(root / d / "a.txt"), roots=roots), d
    assert should_track(mk(root / "docs" / "sub" / "a.txt"), roots=roots)


def test_should_track_fora_dos_roots_e_data_dir(root, tmp_path, monkeypatch):
    outro = mk(tmp_path / "outro" / "a.txt")
    assert not should_track(outro, roots=[root])
    monkeypatch.setenv("LOCALAPPDATA", str(root))
    inside = mk(config.data_dir() / "a.txt")
    assert not should_track(inside, roots=[root])


@windows_only
def test_should_track_atributos(root):
    p = mk(root / "oculto.txt")
    assert should_track(p, roots=[root])
    assert ctypes.windll.kernel32.SetFileAttributesW(str(p), 0x2)
    assert not should_track(p, roots=[root])


# ---------- apply_path ----------

def test_apply_path_ciclo(conn, root):
    p = mk(root / "a.txt", "um")
    assert apply_path(conn, p, roots=[root]) == "added"
    r = row(conn, p)
    assert r["state"] == "pending" and r["size"] == 2 and r["mtime_ns"] > 0
    assert r["file_ref"] and r["file_ref"] != 0 and r["name"] == "a.txt" and r["ext"] == "txt"
    assert apply_path(conn, p, roots=[root]) == "unchanged"

    conn.execute("UPDATE files SET state='done', attempts=2, error='x' WHERE id=?", (r["id"],))
    p.write_text("conteudo maior", encoding="utf-8")
    assert apply_path(conn, p, roots=[root]) == "changed"
    r2 = row(conn, p)
    assert r2["id"] == r["id"] and r2["state"] == "pending"
    assert r2["attempts"] == 0 and r2["error"] is None and r2["size"] == 14

    p.unlink()
    assert apply_path(conn, p, roots=[root]) == "removed"
    assert row(conn, p) is None
    assert apply_path(conn, p, roots=[root]) == "ignored"


def test_apply_path_ignora_nao_rastreavel(conn, root):
    assert apply_path(conn, mk(root / "a.exe"), roots=[root]) == "ignored"
    assert apply_path(conn, root, roots=[root]) == "ignored"  # pasta
    p = mk(root / "b.txt")
    assert apply_path(conn, p, roots=[root]) == "added"
    # passa a ser ignorado (agora fora de qualquer root): a linha some
    assert apply_path(conn, p, roots=[root / "outra"]) == "removed"


def test_apply_path_so_online(conn, root, monkeypatch):
    p = mk(root / "nuvem.txt")
    real = os.stat

    def fake(path, *a, **kw):
        st = real(path, *a, **kw)
        if os.fspath(path) == str(p):
            return SimpleNamespace(
                st_mode=st.st_mode, st_size=st.st_size, st_mtime_ns=st.st_mtime_ns,
                st_ino=st.st_ino, st_dev=st.st_dev, st_file_attributes=0x400000,
            )
        return st

    monkeypatch.setattr(sync.os, "stat", fake)
    assert apply_path(conn, p, roots=[root]) == "added"
    r = row(conn, p)
    assert r["state"] == "skipped" and r["error"] == "disponível só online"
    assert apply_path(conn, p, roots=[root]) == "unchanged"
    monkeypatch.setattr(sync.os, "stat", real)  # baixou: vira pending
    assert apply_path(conn, p, roots=[root]) == "changed"
    r = row(conn, p)
    assert r["state"] == "pending" and r["error"] is None


def test_rename_por_identidade(conn, root):
    old = mk(root / "alpha.txt")
    assert apply_path(conn, old, roots=[root]) == "added"
    fid = row(conn, old)["id"]
    conn.execute("INSERT INTO chunks(file_id, kind, seq, text) VALUES (?, 'name', 0, 'alpha.txt')", (fid,))
    conn.execute("INSERT INTO chunks(file_id, kind, seq, text) VALUES (?, 'text', 1, 'corpo zebra')", (fid,))
    new = root / "bravo.txt"
    os.rename(old, new)

    assert apply_path(conn, new, roots=[root]) == "moved"
    r = row(conn, new)
    assert r["id"] == fid and r["path"] == str(new) and r["name"] == "bravo.txt"
    assert row(conn, old) is None
    texts = {c["kind"]: c["text"] for c in conn.execute("SELECT kind, text FROM chunks WHERE file_id=?", (fid,))}
    assert texts["name"] == extractors.name_text(new, None)
    assert texts["text"] == "corpo zebra"
    cid = conn.execute("SELECT id FROM chunks WHERE file_id=? AND kind='name'", (fid,)).fetchone()[0]
    assert fts(conn, "bravo") == [cid]
    assert fts(conn, "alpha") == []
    assert len(fts(conn, "zebra")) == 1
    assert apply_path(conn, old, roots=[root]) == "ignored"


def test_apply_move_sobrescrita(conn, root):
    a, b = mk(root / "a.txt", "aaa"), mk(root / "b.txt", "bbbbbb")
    apply_path(conn, a, roots=[root])
    apply_path(conn, b, roots=[root])
    id_a, id_b = row(conn, a)["id"], row(conn, b)["id"]
    conn.execute("INSERT INTO chunks(file_id, kind, seq, text) VALUES (?, 'text', 1, 'texto do b')", (id_b,))
    os.replace(a, b)

    assert apply_move(conn, a, b, roots=[root]) == "moved"
    assert row(conn, a) is None
    assert row(conn, b)["id"] == id_a
    assert conn.execute("SELECT count(*) FROM files WHERE id=?", (id_b,)).fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM chunks WHERE file_id=?", (id_b,)).fetchone()[0] == 0
    assert count(conn) == 1


def test_apply_move_mudando_extensao_para_nao_suportada(conn, root):
    a = mk(root / "a.txt")
    apply_path(conn, a, roots=[root])
    b = root / "a.bak"
    os.rename(a, b)
    assert apply_move(conn, a, b, roots=[root]) == "ignored"
    assert count(conn) == 0


def test_apply_move_so_caixa(conn, root):
    a = mk(root / "doc.txt")
    apply_path(conn, a, roots=[root])
    fid = row(conn, a)["id"]
    b = root / "DOC.txt"
    os.rename(a, b)
    assert apply_move(conn, a, b, roots=[root]) == "moved"
    r = row(conn, b)
    assert r["id"] == fid and r["name"] == "DOC.txt"


def test_move_tree_e_remove_tree(conn, root):
    d = root / "pasta"
    files = [mk(d / "a.txt"), mk(d / "sub" / "b.txt", "bb"), mk(d / "sub" / "c.txt", "ccc")]
    for f in files:
        apply_path(conn, f, roots=[root])
    # irmão com prefixo parecido não pode ser tocado
    irmao = mk(root / "pasta2" / "z.txt")
    apply_path(conn, irmao, roots=[root])
    ids = {f.name: row(conn, f)["id"] for f in files}
    for name, fid in ids.items():
        conn.execute("INSERT INTO chunks(file_id, kind, seq, text) VALUES (?, 'name', 0, ?)", (fid, name))

    novo = root / "renomeada"
    os.rename(d, novo)
    assert move_tree(conn, d, novo, roots=[root]) == 3
    for f in files:
        nf = novo / f.relative_to(d)
        r = row(conn, nf)
        assert r["id"] == ids[f.name] and r["path"] == str(nf)
        assert row(conn, f) is None
        txt = conn.execute("SELECT text FROM chunks WHERE file_id=? AND kind='name'", (r["id"],)).fetchone()[0]
        assert txt == extractors.name_text(nf, None)
    assert row(conn, irmao) is not None
    assert len(fts(conn, "renomeada")) == 3

    assert remove_tree(conn, novo) == 3
    assert count(conn) == 1


def test_move_tree_para_pasta_ignorada_remove(conn, root):
    d = root / "pasta"
    f = mk(d / "a.txt")
    apply_path(conn, f, roots=[root])
    ign = root / "node_modules"
    os.rename(d, ign)
    assert move_tree(conn, d, ign, roots=[root]) == 1
    assert count(conn) == 0


# ---------- scan ----------

def test_scan(conn, root):
    a, b, c = mk(root / "a.txt", "a"), mk(root / "sub" / "b.md", "bb"), mk(root / "sub" / "c.txt", "ccc")
    mk(root / "x.exe")
    mk(root / "node_modules" / "n.txt")
    mk(root / ".git" / "g.txt")
    r1 = scan(conn, [root])
    assert r1["added"] == 3 and r1["errors"] == 0
    assert count(conn) == 3
    assert {r["name"] for r in conn.execute("SELECT name FROM files")} == {"a.txt", "b.md", "c.txt"}

    r2 = scan(conn, [root])
    assert r2["unchanged"] == 3 and r2["added"] == r2["changed"] == r2["removed"] == r2["moved"] == 0

    id_c = row(conn, c)["id"]
    a.unlink()
    b.write_text("bastante maior", encoding="utf-8")
    c2 = root / "sub" / "renomeado.txt"
    os.rename(c, c2)
    r3 = scan(conn, [root])
    assert (r3["removed"], r3["changed"], r3["moved"]) == (1, 1, 1)
    assert r3["added"] == 0 and r3["unchanged"] == 0
    assert row(conn, c2)["id"] == id_c
    assert count(conn) == 2

    assert scan(conn, [])["removed"] == 2
    assert count(conn) == 0


def test_scan_lote_pequeno_e_root_removido_da_lista(conn, tmp_path):
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    for i in range(7):
        mk(r1 / f"f{i}.txt", str(i))
    mk(r2 / "g.txt")
    assert scan(conn, [r1, r2], batch=3)["added"] == 8
    res = scan(conn, [r1], batch=3)
    assert res["removed"] == 1 and res["unchanged"] == 7
    assert count(conn) == 7


def test_scan_preserva_linhas_de_pasta_ausente_ilegivel(conn, root, monkeypatch):
    f = mk(root / "sub" / "a.txt")
    scan(conn, [root])
    real = os.scandir

    def boom(path=None):
        if path is not None and os.fspath(path).endswith("sub"):
            raise PermissionError(13, "negado", os.fspath(path))
        return real(path) if path is not None else real()

    monkeypatch.setattr(os, "scandir", boom)
    res = scan(conn, [root])
    assert res["errors"] == 1 and res["removed"] == 0
    assert row(conn, f) is not None


# ---------- config.watched_dirs ----------

def test_watched_dirs_override(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("NACF_WATCH_DIRS", os.pathsep.join([str(a), str(b), str(tmp_path / "nao_existe")]))
    assert config.watched_dirs() == [a, b]
    # pasta dentro de outra é removida
    (a / "dentro").mkdir()
    monkeypatch.setenv("NACF_WATCH_DIRS", os.pathsep.join([str(a), str(a / "dentro"), str(a)]))
    assert config.watched_dirs() == [a]


@windows_only
def test_watched_dirs_padrao(monkeypatch):
    monkeypatch.delenv("NACF_WATCH_DIRS", raising=False)
    dirs = config.watched_dirs()
    assert dirs and all(d.is_dir() for d in dirs)
    assert len({os.path.normcase(os.path.abspath(d)) for d in dirs}) == len(dirs)


def test_scan_nao_segura_lock_de_escrita(tmp_path, root, monkeypatch):
    """Outra conexão (ex.: o worker) consegue gravar enquanto um scan grande roda."""
    dbp = tmp_path / "lock.db"
    c = db.connect(dbp)
    for i in range(3000):
        mk(root / f"d{i % 30}" / f"f{i}.txt", str(i))
    scan(c, [root])
    mk(root / "d15" / "f1515.txt", "mudou, e bastante")  # 1 alterado no meio

    real_stat = os.stat

    def slow_stat(path, *a, **kw):  # walk > 1 s, mais que o busy_timeout do outro lado
        if str(path).startswith(str(root)):
            time.sleep(0.0006)
        return real_stat(path, *a, **kw)

    monkeypatch.setattr(sync.os, "stat", slow_stat)
    errors, done = [], threading.Event()

    def run():
        sc = db.connect(dbp)
        try:
            res = scan(sc, [root], batch=50)
            assert res["changed"] == 1 and res["unchanged"] == 2999
        except Exception as e:
            errors.append(e)
        finally:
            sc.close()
            done.set()

    t = threading.Thread(target=run)
    t.start()
    w = db.connect(dbp)
    w.execute("PRAGMA busy_timeout = 1000")
    tentativas = 0
    try:
        while not done.is_set():
            tentativas += 1
            w.execute("BEGIN IMMEDIATE")  # levantaria "database is locked" se o scan segurasse o lock
            w.execute("COMMIT")
            time.sleep(0.005)
    finally:
        w.close()
        t.join()
        c.close()
    assert not errors and tentativas > 0


def test_extractors_exporta_name_text():
    assert extractors.name_text is not None
