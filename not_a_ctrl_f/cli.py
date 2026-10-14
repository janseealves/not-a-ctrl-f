"""Linha de comando: run, start, stop, setup, search, status, autostart."""

import argparse
import html
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

from not_a_ctrl_f import __version__, installer
from not_a_ctrl_f.shared import config

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_RUN_VALUE = "not-a-ctrl-f"

_DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

_START_TIMEOUT = 90.0
_STOP_TIMEOUT = 30.0

def _app():
    return importlib.import_module("not_a_ctrl_f.app")


def _port() -> int:
    return getattr(config, "ui_port", lambda: 47615)()


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _pythonw() -> str:
    w = Path(sys.executable).with_name("pythonw.exe")
    return str(w if w.exists() else sys.executable)


def _running() -> dict | None:
    """A instância em execução (read_instance) se responder ao ping."""
    app = _app()
    inst = app.read_instance()
    if inst and app.ping(int(inst["port"])):
        return inst
    return None


# --- run / start / stop --------------------------------------------------------------------


def cmd_run(args) -> int:
    app = _app()
    app.setup_logging(console=True)
    return app.App(port=args.port).run()


def cmd_start(args) -> int:
    inst = _running()
    if inst:
        print(f"já está rodando em {inst['url']}")
        return 0
    if installer.missing_components():
        print("Instalando componentes que faltam (só na primeira vez)...")
        try:
            installer.install_all(progress=installer.console_progress)
        except installer.InstallError as e:
            _err(f"erro na instalação: {e}")
            return 1
    cmd = [_pythonw(), "-m", "not_a_ctrl_f", "run"]
    subprocess.Popen(
        cmd,
        creationflags=_DETACHED_PROCESS | _NEW_PROCESS_GROUP | _NO_WINDOW,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True,
    )  # fmt: skip
    app = _app()
    port = _port()
    print("Iniciando...")
    deadline = time.monotonic() + _START_TIMEOUT
    while True:
        if app.ping(port):
            print(f"pronto: http://127.0.0.1:{port}")
            return 0
        if time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    _err(f"não respondeu em {_START_TIMEOUT:.0f}s; veja o log: {config.data_dir() / 'logs' / 'app.log'}")
    return 1


def cmd_stop(args) -> int:
    inst = _running()
    if not inst:
        print("não está rodando")
        return 0
    port = int(inst["port"])
    try:
        httpx.post(
            f"http://127.0.0.1:{port}/api/shutdown", json={},
            headers={"X-NACF-Token": str(inst["token"])}, timeout=10.0,
        )
    except httpx.HTTPError as e:
        _err(f"falha ao pedir o encerramento: {e}")
        return 1
    app = _app()
    deadline = time.monotonic() + _STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not app.ping(port):
            print("encerrado")
            return 0
        time.sleep(0.3)
    _err(f"ainda respondendo após {_STOP_TIMEOUT:.0f}s")
    return 1


# --- setup ---------------------------------------------------------------------------------


def cmd_setup(args) -> int:
    try:
        installer.install_all(
            args.backend, force=args.force, verify=args.verify,
            progress=installer.console_progress,
        )
    except installer.InstallError as e:
        _err(f"erro na instalação: {e}")
        return 1
    marker = installer._read_marker(config.llama_server_path().parent) or {}
    print("Instalação concluída.")
    print(f"  backend:      {marker.get('backend', 'manual')}")
    print(f"  llama-server: {config.llama_server_path()}")
    print(f"  modelos:      {config.models_dir()}")
    return 0


# --- search --------------------------------------------------------------------------------


def _snippet_from_html(s: str | None) -> str | None:
    """snippet_html da API -> texto com MARK_START/MARK_END (sem outras tags, entidades resolvidas)."""
    from not_a_ctrl_f.query.search import MARK_END, MARK_START

    if s is None:
        return None
    s = s.replace("<mark>", MARK_START).replace("</mark>", MARK_END)
    s = re.sub(r"<[^>]*>", "", s)
    return html.unescape(s)


def _render_snippet(s: str, tty: bool) -> str:
    from not_a_ctrl_f.query.search import MARK_END, MARK_START

    if tty:
        return s.replace(MARK_START, "\x1b[1m").replace(MARK_END, "\x1b[22m")
    return s.replace(MARK_START, "[").replace(MARK_END, "]")


def _plain(s: str | None) -> str | None:
    from not_a_ctrl_f.query.search import MARK_END, MARK_START

    return None if s is None else s.replace(MARK_START, "").replace(MARK_END, "")


def _hit_dict(h) -> dict:
    from not_a_ctrl_f.query.search import type_of

    return {
        "file_id": h.file_id, "path": h.path, "name": h.name, "folder": os.path.dirname(h.path),
        "ext": h.ext, "type": type_of(h.ext), "state": h.state, "score": h.score,
        "kind": h.kind, "loc": h.loc, "loc_end": h.loc_end, "matched": list(h.matched),
        "snippet": h.snippet,
    }  # fmt: skip


def _mmss(sec: float) -> str:
    sec = max(0, round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _location(h: dict) -> str:
    loc = h.get("loc")
    if loc is None:
        return ""
    if h.get("kind") in ("page", "page_image"):
        return f"página {round(loc)}"
    if h.get("kind") == "audio":
        end = h.get("loc_end")
        return f"{_mmss(loc)}–{_mmss(end)}" if end is not None else _mmss(loc)
    return ""


def _search_api(inst: dict, args) -> dict:
    params = {"q": " ".join(args.query), "limit": args.limit}
    if args.type:
        params["type"] = args.type
    r = httpx.get(
        f"http://127.0.0.1:{int(inst['port'])}/api/search", params=params,
        headers={"X-NACF-Token": str(inst["token"])}, timeout=60.0,
    )
    r.raise_for_status()
    data = r.json()
    hits = []
    for h in data.get("hits", []):
        h = dict(h)
        h["snippet"] = _snippet_from_html(h.pop("snippet_html", None))
        hits.append(h)
    return {"semantic": bool(data.get("semantic")), "error": data.get("error"), "hits": hits}


def _search_local(args) -> dict | None:
    from not_a_ctrl_f.query.search import Searcher
    from not_a_ctrl_f.shared.embedder import Embedder, EmbedderError, LlamaServer

    db_path = config.default_db_path()
    if not Path(db_path).is_file():
        _err("índice vazio; rode `not-a-ctrl-f start`")
        return None
    server = embedder = None
    note = "--keyword-only" if args.keyword_only else None
    try:
        if not args.keyword_only:
            try:
                server = LlamaServer(text_only=True).start()
                embedder = Embedder(server.base_url, text_only=True)
            except EmbedderError as e:
                note = (str(e).splitlines() or ["embedder indisponível"])[0]
                _err(f"aviso: busca semântica indisponível ({note}); usando só palavra-chave")
        searcher = Searcher(db_path, embedder)
        try:
            res = searcher.search(
                " ".join(args.query), limit=args.limit, type=args.type,
                keyword_only=args.keyword_only,
            )
        finally:
            searcher.close()
    finally:
        if embedder is not None:
            embedder.close()
        if server is not None:
            server.close()
    return {
        "semantic": res.semantic, "error": res.error or note,
        "hits": [_hit_dict(h) for h in res.hits],
    }


def cmd_search(args) -> int:
    data = None
    inst = _running()
    if inst:
        try:
            data = _search_api(inst, args)
        except (httpx.HTTPError, ValueError) as e:
            _err(f"aviso: a instância não respondeu à busca ({e}); usando o índice local")
    if data is None:
        data = _search_local(args)
        if data is None:
            return 1
    if args.json:
        out = [{**h, "snippet": _plain(h.get("snippet"))} for h in data["hits"]]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    tty = sys.stdout.isatty()
    dim, reset = ("\x1b[2m", "\x1b[0m") if tty else ("", "")
    if not data["hits"]:
        print("nenhum resultado")
    for i, h in enumerate(data["hits"], 1):
        loc = _location(h)
        print(f"{i}. {h['name']}" + (f"  [{loc}]" if loc else ""))
        print(f"   {dim}{h.get('folder') or os.path.dirname(h['path'])}{reset}")
        if h.get("snippet"):
            print(f"   {_render_snippet(h['snippet'], tty)}")
    why = "sim" if data["semantic"] else "não" + (f" ({data['error']})" if data["error"] else "")
    print(f"semântica: {why}")
    return 0


# --- status --------------------------------------------------------------------------------


def cmd_status(args) -> int:
    print(f"versão:   {__version__}")
    print(f"dados:    {config.data_dir()}")
    db_path = Path(config.default_db_path())
    print(f"banco:    {db_path}" + ("" if db_path.is_file() else " (ainda não existe)"))
    dirs = config.watched_dirs()
    print("pastas monitoradas:" + ("" if dirs else " nenhuma"))
    for d in dirs:
        print(f"  {d}")
    missing = installer.missing_components()
    print("componentes: " + (", ".join(f"falta {m}" for m in missing) if missing else "instalado"))
    inst = _running()
    print("instância: " + (f"rodando em {inst['url']}" if inst else "não está rodando"))
    if db_path.is_file():
        from not_a_ctrl_f.query.search import index_status
        from not_a_ctrl_f.shared import db

        try:
            conn = db.connect(db_path, readonly=True)
            try:
                st = index_status(conn)
            finally:
                conn.close()
        except Exception as e:
            _err(f"não foi possível ler o índice: {e}")
            return 1
        by = ", ".join(f"{k}: {v}" for k, v in st["by_state"].items() if v)
        print(f"índice:   {st['files']} arquivos" + (f" ({by})" if by else "") + f", {st['chunks']} trechos")
    return 0


# --- autostart --------------------------------------------------------------------


def _autostart_command() -> str:
    return f'"{_pythonw()}" -m not_a_ctrl_f run'


def cmd_autostart(args) -> int:
    if sys.platform != "win32":
        _err("só disponível no Windows")
        return 1
    import winreg

    try:
        if args.action == "on":
            with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
                winreg.SetValueEx(k, _RUN_VALUE, 0, winreg.REG_SZ, _autostart_command())
            print("início automático ativado")
        elif args.action == "off":
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
                    winreg.DeleteValue(k, _RUN_VALUE)
            except FileNotFoundError:
                pass
            print("início automático desativado")
        else:
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as k:
                    value, _ = winreg.QueryValueEx(k, _RUN_VALUE)
                print(f"ativado: {value}")
            except FileNotFoundError:
                print("desativado")
    except OSError as e:
        _err(f"erro no registro: {e}")
        return 1
    return 0


# --- parser --------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="not-a-ctrl-f", description="Busca local nos seus arquivos.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="comando")

    s = sub.add_parser("run", help="roda o app neste terminal (padrão)")
    s.add_argument("--port", type=int, help="porta da API local")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("start", help="inicia em segundo plano")
    s.set_defaults(fn=cmd_start)

    s = sub.add_parser("stop", help="encerra a instância em segundo plano")
    s.set_defaults(fn=cmd_stop)

    s = sub.add_parser("setup", help="baixa llama.cpp e modelos")
    s.add_argument("--backend", choices=installer.BACKENDS, default="auto")
    s.add_argument("--force", action="store_true", help="baixa de novo")
    s.add_argument("--verify", action="store_true", help="confere o sha256 dos modelos")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("search", help="busca no índice")
    s.add_argument("query", nargs="+")
    s.add_argument("--type", choices=["doc", "image", "audio"])
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--keyword-only", action="store_true", help="sem busca semântica")
    s.add_argument("--json", action="store_true", help="saída em JSON")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("status", help="mostra o estado da instalação e do índice")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("autostart", help="iniciar com o Windows")
    s.add_argument("action", choices=["on", "off", "status"])
    s.set_defaults(fn=cmd_autostart)

    return p


def main(argv: list[str] | None = None) -> int:
    for stream in ("stdout", "stderr"):  # pythonw não tem console
        if getattr(sys, stream) is None:
            setattr(sys, stream, open(os.devnull, "w", encoding="utf-8"))
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--version")):
        argv.insert(0, "run")
    args = build_parser().parse_args(argv)
    if getattr(args, "limit", 1) < 1:
        _err("--limit deve ser >= 1")
        return 2
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130
