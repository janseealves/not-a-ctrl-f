import io
import time

import httpx
import pytest
from PIL import Image

from not_a_ctrl_f.query import api as api_mod
from not_a_ctrl_f.query.api import ApiServer
from not_a_ctrl_f.query.search import MARK_END, MARK_START, Hit, SearchResult
from not_a_ctrl_f.shared import db


class StubSearcher:
    def __init__(self):
        self.calls = []

    def search(self, q, *, limit=30, type=None, keyword_only=False):
        self.calls.append((q, limit, type))
        hit = Hit(
            file_id=1, path=r"C:\docs\relatorio.txt", name="relatorio.txt", ext="txt", state="done",
            score=0.5, kind="text", loc=3.0, loc_end=9.0, matched=("keyword", "semantic"),
            snippet=f"<script>alert(1)</script> {MARK_START}achado{MARK_END} & fim",
        )
        return SearchResult(hits=[hit], semantic=True, error=None, took_ms=1.5)


def insert(conn, path, ext, state="done"):
    return conn.execute(
        "INSERT INTO files(path, path_key, name, ext, size, mtime_ns, state) VALUES (?,?,?,?,?,?,?)",
        (str(path), db.path_key(path), path.name, ext, 1, 1, state),
    ).lastrowid


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod, "index_status", lambda conn: {
        "files": conn.execute("SELECT count(*) FROM files").fetchone()[0],
        "by_state": {"done": 1}, "chunks": 0,
    })
    db_path = tmp_path / "idx.db"
    conn = db.connect(db_path)
    txt = tmp_path / "nota.txt"
    txt.write_text("oi")
    png = tmp_path / "foto.png"
    Image.new("RGB", (800, 400), (10, 120, 200)).save(png)
    gone = tmp_path / "sumiu.txt"
    gone.write_text("x")
    ids = {"txt": insert(conn, txt, "txt"), "png": insert(conn, png, "png"),
           "gone": insert(conn, gone, "txt")}
    gone.unlink()
    conn.close()

    opened, revealed = [], []
    srv = ApiServer(
        StubSearcher(), db_path, status_extra=lambda: {"embedder": "ok", "scanning": False},
        opener=opened.append, revealer=revealed.append,
    ).start()
    client = httpx.Client(base_url=srv.url, timeout=10)
    e = type("Env", (), {})()
    e.srv, e.client, e.ids, e.opened, e.revealed = srv, client, ids, opened, revealed
    e.txt, e.png = txt, png
    e.auth = {"X-NACF-Token": srv.token}
    yield e
    client.close()
    srv.stop()


def post(env, body, headers=None, **kw):
    h = {**env.auth, "Content-Type": "application/json", **(headers or {})}
    return env.client.post("/api/open", headers=h, **kw)


# 1
def test_raiz_e_ping(env):
    r = env.client.get("/")
    assert r.status_code == 404
    assert "access-control-allow-origin" not in r.headers
    r = env.client.get("/api/ping")
    assert r.status_code == 200 and r.json() == {"app": "not-a-ctrl-f"}


def test_host_invalido_em_toda_rota(env):
    port = env.srv.port
    for path in ("/", "/api/ping"):
        r = env.client.get(path, headers={"Host": f"evil.com:{port}"})
        assert r.status_code == 403
    assert env.client.get("/api/ping", headers={"Host": f"localhost:{port}"}).status_code == 200
    assert env.client.get("/api/ping", headers={"Host": "127.0.0.1"}).status_code == 403


# 2
def test_search(env):
    c = env.client
    assert c.get("/api/search?q=oi").status_code == 403
    assert c.get("/api/search?q=oi", headers={"X-NACF-Token": "errado"}).status_code == 403
    assert c.get("/api/search?q=oi", headers={"X-NACF-Token": bytes([110, 227, 111])}).status_code == 403
    assert c.get("/api/search?q=oi&t=" + env.srv.token).status_code == 403  # ?t= só vale em thumb
    r = c.get("/api/search?q=oi", headers={"Host": f"evil.com:{env.srv.port}", **env.auth})
    assert r.status_code == 403

    r = c.get("/api/search", params={"q": "oi", "type": "doc", "limit": "5"}, headers=env.auth)
    assert r.status_code == 200
    data = r.json()
    assert data["semantic"] is True and data["error"] is None and data["took_ms"] == 1.5
    h = data["hits"][0]
    assert h["folder"] == r"C:\docs" and h["type"] == "doc" and h["matched"] == ["keyword", "semantic"]
    assert "<script>" not in h["snippet_html"]
    assert "&lt;script&gt;" in h["snippet_html"]
    assert "<mark>achado</mark>" in h["snippet_html"] and "&amp; fim" in h["snippet_html"]
    assert env.srv.searcher.calls[-1] == ("oi", 5, "doc")

    assert c.get("/api/search?q=oi", headers=env.auth).status_code == 200
    assert env.srv.searcher.calls[-1] == ("oi", 30, None)
    for bad in ("0", "101", "abc", "-1"):
        assert c.get("/api/search", params={"q": "oi", "limit": bad}, headers=env.auth).status_code == 400
    assert c.get("/api/search", params={"q": "oi", "type": "video"}, headers=env.auth).status_code == 400


# 3
def test_open(env):
    r = post(env, None, json={"file_id": env.ids["txt"]})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert env.opened == [str(env.txt)] and env.revealed == []
    r = post(env, None, json={"file_id": env.ids["txt"], "reveal": True})
    assert r.status_code == 200
    assert env.revealed == [str(env.txt)] and env.opened == [str(env.txt)]


def test_open_nao_encontrado(env):
    assert post(env, None, json={"file_id": 9999}).status_code == 404
    assert post(env, None, json={"file_id": env.ids["gone"]}).status_code == 404
    assert env.opened == [] and env.revealed == []


def test_open_nao_abre_tipo_desconhecido(env, tmp_path):
    exe = tmp_path / "x.exe"
    exe.write_text("x")
    conn = db.connect(env.srv.db_path)
    fid = insert(conn, exe, "exe")
    conn.close()
    assert post(env, None, json={"file_id": fid}).status_code == 404
    assert env.opened == []


def test_open_validacoes(env):
    ok = {"file_id": env.ids["txt"]}
    # Origin estranho
    r = post(env, None, json=ok, headers={"Origin": "http://evil.com"})
    assert r.status_code == 403
    r = post(env, None, json=ok, headers={"Origin": f"http://localhost:{env.srv.port}"})
    assert r.status_code == 200
    # sem Content-Type json -> 415
    r = env.client.post("/api/open", headers=env.auth, content=b'{"file_id": 1}')
    assert r.status_code == 415
    r = env.client.post("/api/open", headers={**env.auth, "Content-Type": "text/plain"},
                        content=b'{"file_id": 1}')
    assert r.status_code == 415
    # JSON inválido / tipos errados
    hdr = {**env.auth, "Content-Type": "application/json"}
    assert env.client.post("/api/open", headers=hdr, content=b"{nao json").status_code == 400
    assert env.client.post("/api/open", headers=hdr, content=b"[1]").status_code == 400
    for body in ({"file_id": "1"}, {"file_id": True}, {"file_id": 1.5}, {}, {"file_id": 1, "reveal": "sim"}):
        assert post(env, None, json=body).status_code == 400
    # path vindo do cliente é ignorado
    r = post(env, None, json={"file_id": env.ids["txt"], "path": r"C:\Windows\notepad.exe"})
    assert r.status_code == 200 and env.opened[-1] == str(env.txt)
    # método e token
    assert env.client.get("/api/open", headers=env.auth).status_code == 405
    r = env.client.post("/api/open", headers={"Content-Type": "application/json"}, json=ok)
    assert r.status_code == 403
    # corpo grande
    r = env.client.post("/api/open", headers=hdr, content=b" " * (64 * 1024 + 1))
    assert r.status_code == 413
    assert env.client.put("/api/open", headers=hdr, content=b"{}").status_code == 405


def test_revealer_padrao_recusa_aspas(monkeypatch):
    called = []
    monkeypatch.setattr(api_mod.subprocess, "Popen", lambda *a, **k: called.append(a))
    with pytest.raises(ValueError):
        api_mod.default_revealer('C:\\a"b.txt')
    api_mod.default_revealer(r"C:\a b\c.txt")
    assert called == [(r'explorer /select,"C:\a b\c.txt"',)]


# 4
def test_thumb(env):
    c, i = env.client, env.ids
    r = c.get(f"/api/thumb/{i['png']}", headers=env.auth)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["cache-control"] == "private, max-age=300"
    img = Image.open(io.BytesIO(r.content))
    assert img.format == "JPEG" and max(img.size) == 320
    r = c.get(f"/api/thumb/{i['png']}?t={env.srv.token}")
    assert r.status_code == 200
    assert c.get(f"/api/thumb/{i['png']}?t=errado").status_code == 403
    assert c.get(f"/api/thumb/{i['png']}").status_code == 403
    assert c.get(f"/api/thumb/{i['txt']}", headers=env.auth).status_code == 404
    assert c.get("/api/thumb/9999", headers=env.auth).status_code == 404
    assert c.post(f"/api/thumb/{i['png']}", headers=env.auth, json={}).status_code == 405


def test_thumb_imagem_corrompida(env, tmp_path):
    bad = tmp_path / "ruim.png"
    bad.write_bytes(b"nao e png")
    conn = db.connect(env.srv.db_path)
    fid = insert(conn, bad, "png")
    conn.close()
    assert env.client.get(f"/api/thumb/{fid}", headers=env.auth).status_code == 404


# 5
def test_status(env):
    assert env.client.get("/api/status").status_code == 403
    d = env.client.get("/api/status", headers=env.auth).json()
    assert d["app"] == "not-a-ctrl-f" and d["index"]["files"] == 3
    assert d["embedder"] == "ok" and d["scanning"] is False


def test_rota_inexistente(env):
    assert env.client.get("/api/nada", headers=env.auth).status_code == 404
    assert env.client.get("/nada").status_code == 404


# 6
def test_stop_idempotente_e_porta_ocupada(tmp_path):
    db.connect(tmp_path / "i.db").close()
    srv = ApiServer(StubSearcher(), tmp_path / "i.db").start()
    try:
        with pytest.raises(OSError):
            ApiServer(StubSearcher(), tmp_path / "i.db", port=srv.port).start()
    finally:
        srv.stop()
        srv.stop()


# 7
def _shutdown_srv(tmp_path, calls):
    db.connect(tmp_path / "s.db").close()
    return ApiServer(StubSearcher(), tmp_path / "s.db", on_shutdown=lambda: calls.append(1)).start()


def test_shutdown(tmp_path):
    calls = []
    srv = _shutdown_srv(tmp_path, calls)
    try:
        hdr = {"X-NACF-Token": srv.token, "Content-Type": "application/json"}
        with httpx.Client(base_url=srv.url, timeout=10) as c:
            r = c.post("/api/shutdown", headers=hdr, json={})
            assert r.status_code == 200 and r.json() == {"ok": True}
            for _ in range(100):
                if calls:
                    break
                time.sleep(0.05)
            assert calls == [1]
            # corpo diferente de {} e método errado
            assert c.post("/api/shutdown", headers=hdr, json={"x": 1}).status_code == 400
            assert c.get("/api/shutdown", headers=hdr).status_code == 405
    finally:
        srv.stop()


def test_shutdown_recusado(tmp_path):
    calls = []
    srv = _shutdown_srv(tmp_path, calls)
    try:
        hdr = {"X-NACF-Token": srv.token, "Content-Type": "application/json"}
        with httpx.Client(base_url=srv.url, timeout=10) as c:
            no_token = {"Content-Type": "application/json"}
            assert c.post("/api/shutdown", headers=no_token, json={}).status_code == 403
            r = c.post("/api/shutdown", headers={**hdr, "Origin": "http://evil.com"}, json={})
            assert r.status_code == 403
            assert c.post("/api/shutdown", headers={"X-NACF-Token": srv.token}, content=b"{}").status_code == 415
        time.sleep(0.2)
        assert calls == []
    finally:
        srv.stop()


def test_shutdown_sem_callback(env):
    r = env.client.post("/api/shutdown", headers={**env.auth, "Content-Type": "application/json"}, json={})
    assert r.status_code == 404
