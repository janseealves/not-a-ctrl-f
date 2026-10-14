import json
import subprocess
import sys
import time
import types

import httpx
import pytest

from not_a_ctrl_f import installer
from not_a_ctrl_f.query.search import MARK_END, MARK_START
from not_a_ctrl_f.shared import config, db
from not_a_ctrl_f.shared.embedder import EmbedderError

from not_a_ctrl_f import cli


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    for k in ("NACF_DB_PATH", "NACF_MODELS_DIR", "NACF_LLAMA_SERVER", "NACF_WATCH_DIRS", "NACF_PORT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(installer, "missing_components", lambda: [])
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    return tmp_path


@pytest.fixture
def fake_app(monkeypatch):
    app = types.ModuleType("not_a_ctrl_f.app")
    app.instance = None
    app.alive = False
    app.log = []

    app.read_instance = lambda: app.instance
    app.ping = lambda port, timeout=1.0: app.alive
    app.setup_logging = lambda *, console=True: app.log.append(("logging", console))

    class App:
        def __init__(self, *, port=None, install=True):
            app.log.append(("App", port))

        def run(self):
            app.log.append(("run",))
            return 0

    app.App = App
    monkeypatch.setitem(sys.modules, "not_a_ctrl_f.app", app)
    monkeypatch.setattr(config, "ui_port", lambda: 47615, raising=False)
    return app


INST = {"pid": 1, "port": 5555, "token": "tok", "url": "http://127.0.0.1:5555"}


# --- despacho ------------------------------------------------------------------------------


def test_sem_subcomando_roda_run(fake_app):
    assert cli.main([]) == 0
    assert fake_app.log == [("logging", True), ("App", None), ("run",)]


def test_run_flags(fake_app):
    assert cli.main(["run", "--port", "1234"]) == 0
    assert ("App", 1234) in fake_app.log


def test_flag_sem_subcomando_vira_run(fake_app):
    cli.main(["--port", "99"])
    assert ("App", 99) in fake_app.log


def test_parser_subcomandos():
    p = cli.build_parser()
    assert p.parse_args(["setup", "--backend", "cpu", "--force", "--verify"]).backend == "cpu"
    a = p.parse_args(["search", "a", "b", "--type", "doc", "--limit", "3", "--keyword-only", "--json"])
    assert (a.query, a.type, a.limit, a.keyword_only, a.json) == (["a", "b"], "doc", 3, True, True)
    assert p.parse_args(["autostart", "status"]).action == "status"
    assert p.parse_args(["start"]).fn is cli.cmd_start
    assert p.parse_args(["stop"]).fn is cli.cmd_stop
    assert p.parse_args(["status"]).fn is cli.cmd_status


def test_help_e_versao(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    assert "search" in capsys.readouterr().out


# --- start / stop --------------------------------------------------------------------------


def test_start_com_instancia_rodando(fake_app, monkeypatch, capsys):
    fake_app.instance, fake_app.alive = INST, True
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("não deveria iniciar"))
    assert cli.main(["start"]) == 0
    assert f"já está rodando em {INST['url']}" in capsys.readouterr().out


def test_start_instala_e_dispara_processo(fake_app, monkeypatch):
    calls = []
    monkeypatch.setattr(installer, "missing_components", lambda: ["llama-server.exe"])
    monkeypatch.setattr(installer, "install_all", lambda *a, **k: calls.append(k))

    spawned = {}

    def popen(cmd, **kw):
        spawned["cmd"], spawned["kw"] = cmd, kw
        fake_app.alive = True

    monkeypatch.setattr(subprocess, "Popen", popen)
    assert cli.main(["start"]) == 0
    assert calls and calls[0]["progress"] is installer.console_progress
    cmd = spawned["cmd"]
    assert cmd[1:] == ["-m", "not_a_ctrl_f", "run"]
    assert cmd[0].lower().endswith(("pythonw.exe", "python.exe", "python", "python3"))
    kw = spawned["kw"]
    assert kw["creationflags"] == 0x00000008 | 0x00000200 | 0x08000000
    assert kw["stdin"] == kw["stdout"] == kw["stderr"] == subprocess.DEVNULL
    assert kw["close_fds"] is True


def test_start_erro_de_instalacao(fake_app, monkeypatch, capsys):
    monkeypatch.setattr(installer, "missing_components", lambda: ["x"])

    def boom(*a, **k):
        raise installer.InstallError("sem rede")

    monkeypatch.setattr(installer, "install_all", boom)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("não deveria iniciar"))
    assert cli.main(["start"]) == 1
    assert "sem rede" in capsys.readouterr().err


def test_start_timeout_mostra_log(fake_app, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_START_TIMEOUT", 0.0)
    assert cli.main(["start"]) == 1
    assert "app.log" in capsys.readouterr().err


def test_stop_sem_instancia(fake_app, capsys):
    assert cli.main(["stop"]) == 0
    assert "não está rodando" in capsys.readouterr().out


def test_stop_envia_shutdown(fake_app, monkeypatch, capsys):
    fake_app.instance, fake_app.alive = INST, True
    sent = {}

    def post(url, **kw):
        sent["url"], sent["kw"] = url, kw
        fake_app.alive = False

    monkeypatch.setattr(httpx, "post", post)
    assert cli.main(["stop"]) == 0
    assert sent["url"] == "http://127.0.0.1:5555/api/shutdown"
    assert sent["kw"]["headers"]["X-NACF-Token"] == "tok" and sent["kw"]["json"] == {}


# --- setup ---------------------------------------------------------------------------------


def test_setup_erro(monkeypatch, capsys):
    def boom(*a, **k):
        raise installer.InstallError("disco cheio")

    monkeypatch.setattr(installer, "install_all", boom)
    assert cli.main(["setup"]) == 1
    assert "disco cheio" in capsys.readouterr().err


def test_setup_ok(monkeypatch, capsys):
    got = {}
    monkeypatch.setattr(installer, "install_all", lambda *a, **k: got.update(a=a, k=k))
    assert cli.main(["setup", "--backend", "vulkan", "--force", "--verify"]) == 0
    assert got["a"] == ("vulkan",)
    assert got["k"]["force"] and got["k"]["verify"]
    assert "llama-server" in capsys.readouterr().out


# --- search --------------------------------------------------------------------------------


def _populate():
    conn = db.connect(config.default_db_path())
    cur = conn.execute(
        "INSERT INTO files(path, path_key, name, ext, size, mtime_ns, state, indexed_at) "
        "VALUES (?,?,?,?,0,0,'done',?)",
        ("C:/docs/relatorio.pdf", "c:/docs/relatorio.pdf", "relatorio.pdf", "pdf", time.time()),
    )
    conn.execute(
        "INSERT INTO chunks(file_id, kind, modality, seq, loc, text) VALUES (?,?,?,?,?,?)",
        (cur.lastrowid, "page", "none", 0, 3.0, "O relatório anual de vendas cresceu"),
    )
    conn.close()


def test_search_sem_banco(fake_app, capsys):
    assert cli.main(["search", "x", "--keyword-only"]) == 1
    assert "índice vazio" in capsys.readouterr().err


def test_search_local_texto(fake_app, capsys):
    _populate()
    assert cli.main(["search", "vendas", "--keyword-only"]) == 0
    out = capsys.readouterr().out
    assert "1. relatorio.pdf  [página 3]" in out
    assert "[vendas]" in out
    assert "docs" in out
    assert "semântica: não (--keyword-only)" in out


def test_search_local_json(fake_app, capsys):
    _populate()
    assert cli.main(["search", "vendas", "--keyword-only", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 1
    h = data[0]
    assert h["name"] == "relatorio.pdf" and h["type"] == "doc" and h["kind"] == "page"
    assert h["snippet"] == "O relatório anual de vendas cresceu" or "vendas" in h["snippet"]
    assert MARK_START not in h["snippet"] and "snippet_html" not in h


def test_search_local_embedder_falha_cai_para_palavra_chave(fake_app, monkeypatch, capsys):
    from not_a_ctrl_f.shared import embedder

    class Bad:
        def __init__(self, **kw):
            pass

        def start(self):
            raise EmbedderError("llama-server ausente")

    monkeypatch.setattr(embedder, "LlamaServer", Bad)
    _populate()
    assert cli.main(["search", "vendas"]) == 0
    cap = capsys.readouterr()
    assert "llama-server ausente" in cap.err
    assert "semântica: não (llama-server ausente)" in cap.out
    assert "relatorio.pdf" in cap.out


def test_search_usa_instancia(fake_app, monkeypatch, capsys):
    fake_app.instance, fake_app.alive = INST, True
    seen = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"semantic": True, "error": None, "took_ms": 1, "hits": [{
                "file_id": 1, "path": "C:/m/aula.mp3", "name": "aula.mp3", "folder": "C:/m",
                "ext": "mp3", "type": "audio", "state": "done", "score": 1.0, "kind": "audio",
                "loc": 75.0, "loc_end": 105.0, "matched": ["semantic"],
                "snippet_html": "um &lt;b&gt; <mark>teste</mark>",
            }]}

    def get(url, **kw):
        seen["url"], seen["kw"] = url, kw
        return Resp()

    monkeypatch.setattr(httpx, "get", get)
    assert cli.main(["search", "teste", "--type", "audio", "--limit", "5"]) == 0
    assert seen["url"] == "http://127.0.0.1:5555/api/search"
    assert seen["kw"]["params"] == {"q": "teste", "limit": 5, "type": "audio"}
    assert seen["kw"]["headers"]["X-NACF-Token"] == "tok"
    out = capsys.readouterr().out
    assert "1. aula.mp3  [01:15–01:45]" in out
    assert "um <b> [teste]" in out
    assert "semântica: sim" in out


def test_snippet_da_api():
    s = cli._snippet_from_html("&lt;b&gt; <mark>x</mark> <i>y</i>")
    assert s == f"<b> {MARK_START}x{MARK_END} y"
    assert cli._render_snippet(s, tty=False) == "<b> [x] y"
    assert cli._render_snippet(s, tty=True) == "<b> \x1b[1mx\x1b[22m y"
    assert cli._snippet_from_html(None) is None


# --- status --------------------------------------------------------------------------------


def test_status(fake_app, capsys):
    _populate()
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "instalado" in out and "não está rodando" in out
    assert "1 arquivos" in out and "1 trechos" in out


def test_status_sem_banco_e_componentes_faltando(fake_app, monkeypatch, capsys):
    monkeypatch.setattr(installer, "missing_components", lambda: ["llama-server.exe"])
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "falta llama-server.exe" in out and "ainda não existe" in out


# --- autostart -----------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="registro do Windows")
def test_autostart(monkeypatch, capsys):
    import winreg

    base = r"Software\not-a-ctrl-f-test"
    monkeypatch.setattr(cli, "_RUN_KEY", base + r"\Run")
    try:
        assert cli.main(["autostart", "status"]) == 0
        assert capsys.readouterr().out.strip() == "desativado"
        assert cli.main(["autostart", "on"]) == 0
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, cli._RUN_KEY) as k:
            value, _ = winreg.QueryValueEx(k, cli._RUN_VALUE)
        assert value.endswith(" -m not_a_ctrl_f run") and value.startswith('"')
        capsys.readouterr()
        assert cli.main(["autostart", "status"]) == 0
        assert "ativado" in capsys.readouterr().out
        assert cli.main(["autostart", "off"]) == 0
        assert cli.main(["autostart", "off"]) == 0  # idempotente
        capsys.readouterr()
        cli.main(["autostart", "status"])
        assert capsys.readouterr().out.strip() == "desativado"
    finally:
        for sub in (base + r"\Run", base):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, sub)
            except OSError:
                pass

