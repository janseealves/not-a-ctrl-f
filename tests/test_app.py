import logging
import os
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from test_indexer import FakeEmbedder

from not_a_ctrl_f import app as app_mod
from not_a_ctrl_f.app import App, instance_file, ping, read_instance, setup_logging
from not_a_ctrl_f.processing import sync
from not_a_ctrl_f.shared import config, db
from not_a_ctrl_f.shared.embedder import EmbedderError


class FakeEmb(FakeEmbedder):
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url
        self.closed = False

    def close(self):
        self.closed = True


class FakeLlama:
    def __init__(self, cfg, **kw):
        self.cfg, self.kw = cfg, kw
        self.base_url = "http://127.0.0.1:9"
        self.closed = False
        cfg.instances.append(self)

    def start(self):
        if self.cfg.delay:
            time.sleep(self.cfg.delay)
        if self.cfg.fail_times > 0:
            self.cfg.fail_times -= 1
            raise EmbedderError("falha simulada")
        return self

    def close(self):
        self.closed = True


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k in ("DB_PATH", "MODELS_DIR", "LLAMA_SERVER", "PORT"):
        monkeypatch.delenv(f"NACF_{k}", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    watch = tmp_path / "watch"
    watch.mkdir()
    monkeypatch.setenv("NACF_WATCH_DIRS", str(watch))
    cfg = SimpleNamespace(instances=[], fail_times=0, delay=0.0, embedders=[])
    monkeypatch.setattr(app_mod, "LlamaServer", lambda **kw: FakeLlama(cfg, **kw))

    def make_emb(base_url):
        emb = FakeEmb(base_url)
        cfg.embedders.append(emb)
        return emb

    monkeypatch.setattr(app_mod, "Embedder", make_emb)
    monkeypatch.setattr(app_mod, "missing_components", lambda: [])
    cfg.watch = watch
    before = {t for t in threading.enumerate() if t.name.startswith("nacf-")}
    cfg.new_threads = lambda: [
        t for t in threading.enumerate() if t.name.startswith("nacf-") and t not in before
    ]
    cfg.running = []
    yield cfg
    for a, _, _ in cfg.running:
        a.request_stop()
    for _, t, _ in cfg.running:
        t.join(20)


def launch(env, **kw):
    """Roda App.run() numa thread; devolve (app, thread, resultado)."""
    kw.setdefault("port", 0)
    a = App(**kw)
    result = {}
    t = threading.Thread(target=lambda: result.update(code=a.run()), name="test-app")
    t.start()
    env.running.append((a, t, result))
    return a, t, result


def wait_for(cond, timeout=15.0, what="condição"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError(f"timeout esperando {what}")


def up(a):
    return wait_for(lambda: a.api, what="api")


def status(a):
    r = httpx.get(a.api.url + "api/status", headers={"X-NACF-Token": a.api.token}, timeout=5)
    return r.json()


def search(a, q):
    r = httpx.get(a.api.url + "api/search", params={"q": q},
                  headers={"X-NACF-Token": a.api.token}, timeout=10)
    assert r.status_code == 200
    return r.json()


def shutdown(a, t, result):
    r = httpx.post(a.api.url + "api/shutdown", json={},
                   headers={"X-NACF-Token": a.api.token}, timeout=5)
    assert r.status_code == 200
    t.join(15)
    assert not t.is_alive() and result["code"] == 0


# 1
def test_ciclo_completo(env):
    a, t, result = launch(env)
    api = up(a)
    info = wait_for(read_instance, what="instance.json")
    assert info["port"] == api.port and info["token"] == api.token and info["url"] == api.url
    assert info["pid"] == os.getpid()
    assert ping(api.port)

    wait_for(lambda: status(a)["embedder"] == "ok" and status(a)["scanning"] is False,
             what="embedder ok e varredura concluída")
    st = status(a)
    assert st["watch_dirs"] == [str(env.watch)] and st["version"]
    assert [i.kw for i in env.instances] == [{}]  # primeira tentativa: GPU (padrão)

    (env.watch / "relatorio_unico.txt").write_text("banana azul", encoding="utf-8")
    res = wait_for(lambda: (r := search(a, "azul"))["hits"] and r, what="arquivo na busca")
    assert res["semantic"] is True
    assert res["hits"][0]["name"] == "relatorio_unico.txt"

    shutdown(a, t, result)
    assert read_instance() is None and not instance_file().exists()
    assert env.new_threads() == []
    assert len(env.embedders) == 2 and all(e.closed for e in env.embedders)
    assert env.instances[0].closed


# 2
def test_gpu_falha_usa_cpu(env):
    env.fail_times = 1
    a, t, result = launch(env)
    up(a)
    wait_for(lambda: status(a)["embedder"] == "ok", what="embedder ok")
    assert [i.kw for i in env.instances] == [{}, {"gpu_layers": 0}]
    assert env.instances[0].closed and not env.instances[1].closed
    shutdown(a, t, result)
    assert env.instances[1].closed and env.new_threads() == []


# 3
def test_llama_indisponivel_busca_por_palavra_chave(env):
    env.fail_times = 99
    # arquivo já indexado: a busca por palavra-chave deve achá-lo sem embedder
    f = env.watch / "gato.txt"
    f.write_text("gato preto", encoding="utf-8")
    conn = db.connect(config.default_db_path())
    sync.scan(conn, [env.watch])
    conn.execute("UPDATE files SET state='done', indexed_at=1")
    fid = conn.execute("SELECT id FROM files").fetchone()[0]
    conn.execute(
        "INSERT INTO chunks(file_id, kind, modality, seq, text) VALUES (?, 'text', 'none', 0, 'gato preto')",
        (fid,),
    )
    conn.close()

    a, t, result = launch(env)
    up(a)
    wait_for(lambda: status(a)["embedder"].startswith("indisponível"), what="embedder indisponível")
    assert status(a)["embedder"] == "indisponível: falha simulada"
    wait_for(lambda: status(a)["scanning"] is False, what="varredura")
    res = search(a, "gato")
    assert res["semantic"] is False and "falha simulada" in res["error"]
    assert [h["name"] for h in res["hits"]] == ["gato.txt"]
    assert [i.kw for i in env.instances] == [{}, {"gpu_layers": 0}]
    assert env.embedders == []
    shutdown(a, t, result)
    assert env.new_threads() == [] and read_instance() is None


# 4
def test_instancia_existente(env):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    a, t, result = launch(env, port=port)
    assert up(a).port == port
    wait_for(read_instance, what="instance.json")

    t0 = time.monotonic()
    second = App(port=port)
    assert second.run() == 0
    assert time.monotonic() - t0 < 5
    assert second.api is None
    assert read_instance()["pid"] == os.getpid() and read_instance()["port"] == port
    shutdown(a, t, result)


# 5
def test_componentes_faltando(env, monkeypatch):
    monkeypatch.setattr(app_mod, "missing_components", lambda: ["llama-server.exe"])
    installed = []
    monkeypatch.setattr(app_mod, "install_all", lambda **kw: installed.append(kw))
    assert App(port=0, install=False).run() == 1
    assert installed == [] and not instance_file().exists()

    def boom(**kw):
        raise app_mod.InstallError("sem rede")

    monkeypatch.setattr(app_mod, "install_all", boom)
    assert App(port=0).run() == 1


def test_porta_em_uso(env, caplog):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        with caplog.at_level(logging.ERROR, logger="not_a_ctrl_f.app"):
            assert App(port=port).run() == 1
    assert "NACF_PORT" in caplog.text and not instance_file().exists()


# 6
def test_parar_durante_a_subida(env):
    env.delay = 1.5
    a, t, result = launch(env)
    up(a)
    assert status(a)["embedder"] == "iniciando"
    a.request_stop()
    t.join(15)
    assert not t.is_alive() and result["code"] == 0
    assert env.new_threads() == []
    assert env.instances[0].closed and env.embedders == []
    assert read_instance() is None


# utilitários
def test_read_instance_ausente_e_ilegivel(env):
    assert read_instance() is None
    instance_file().parent.mkdir(parents=True, exist_ok=True)
    instance_file().write_text("{lixo", encoding="utf-8")
    assert read_instance() is None


def test_ping_porta_fechada():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert ping(port, timeout=0.5) is False


def test_setup_logging_idempotente(env):
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    try:
        p1 = setup_logging(console=False)
        n = len(root.handlers)
        assert setup_logging(console=False) == p1 and len(root.handlers) == n
        assert p1 == config.data_dir() / "logs" / "app.log" and n == len(before) + 1
        logging.getLogger("x").info("oi")
        for h in root.handlers:
            h.flush()
        assert "oi" in p1.read_text(encoding="utf-8")
    finally:
        for h in root.handlers[:]:
            if h not in before:
                root.removeHandler(h)
                h.close()
        root.setLevel(level)
