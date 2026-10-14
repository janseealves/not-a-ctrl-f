"""Servidor HTTP local: API de busca consumida por clientes como a CLI (o token fica em instance.json).

Segurança: só escuta no host dado (127.0.0.1), valida o header Host (anti DNS rebinding),
exige token em /api/* (exceto /api/ping), valida Origin/Content-Type nos POST e nunca emite
CORS. O caminho de /api/open e /api/thumb vem sempre do banco, nunca do cliente.
"""

import hmac
import html
import io
import json
import logging
import os
import re
import secrets
import subprocess
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageOps

from not_a_ctrl_f.query.search import (
    MARK_END,
    MARK_START,
    TYPE_EXTS,
    Hit,
    index_status,
    type_of,
)
from not_a_ctrl_f.shared import db

log = logging.getLogger(__name__)

APP_NAME = "not-a-ctrl-f"
MAX_BODY = 64 * 1024
DRAIN_MAX = 1024 * 1024  # corpo grande demais: descarta até aqui para o cliente ler o 413
DEFAULT_LIMIT, MAX_LIMIT = 30, 100
THUMB_SIDE = 320

_THUMB_RE = re.compile(r"/api/thumb/(\d{1,18})")


def default_opener(path: str) -> None:
    os.startfile(path)  # type: ignore[attr-defined]  # só Windows


def default_revealer(path: str) -> None:
    if '"' in path:
        raise ValueError("caminho com aspas")
    subprocess.Popen(f'explorer /select,"{path}"')


def _hit_json(h: Hit) -> dict:
    snippet = None
    if h.snippet is not None:
        snippet = html.escape(h.snippet).replace(MARK_START, "<mark>").replace(MARK_END, "</mark>")
    return {
        "file_id": h.file_id,
        "path": h.path,
        "name": h.name,
        "folder": os.path.dirname(h.path),
        "ext": h.ext,
        "type": type_of(h.ext),
        "state": h.state,
        "score": h.score,
        "kind": h.kind,
        "loc": h.loc,
        "loc_end": h.loc_end,
        "matched": list(h.matched),
        "snippet_html": snippet,
    }


def _make_thumb(path: str) -> bytes:
    with Image.open(path) as img:
        img.draft("RGB", (THUMB_SIDE, THUMB_SIDE))  # JPEG decodifica já reduzido
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((THUMB_SIDE, THUMB_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


class _HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # no Windows, SO_REUSEADDR deixaria outro processo dividir a porta

    def __init__(self, addr, handler, api: "ApiServer") -> None:
        self.api = api
        super().__init__(addr, handler)

    def handle_error(self, request, client_address) -> None:
        log.debug("erro na conexão de %s", client_address, exc_info=True)


class _Handler(BaseHTTPRequestHandler):
    server_version = "nacf"
    sys_version = ""
    timeout = 15  # conexão parada não prende a thread

    server: _Server

    def log_message(self, format, *args) -> None:
        log.debug("%s - %s", self.address_string(), format % args)

    # --- respostas -------------------------------------------------------------------------

    def _send(self, status: int, body: bytes, ctype: str, headers: dict[str, str] | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, data: dict, headers: dict[str, str] | None = None) -> None:
        h = {"Cache-Control": "no-store", **(headers or {})}
        self._send(status, json.dumps(data).encode(), "application/json; charset=utf-8", h)

    def send_error(self, code, message=None, explain=None) -> None:  # erros do http.server em JSON
        self._json(code, {"error": message or self.responses.get(code, ("erro",))[0]})
        self.close_connection = True

    # --- validações ------------------------------------------------------------------------

    def _origins(self) -> tuple[str, str]:
        port = self.server.server_address[1]
        return f"127.0.0.1:{port}", f"localhost:{port}"

    def _check_host(self) -> None:
        if self.headers.get("Host") not in self._origins():
            raise _HttpError(403, "host não permitido")

    def _check_token(self, query: dict[str, list[str]], *, allow_query: bool) -> None:
        given = self.headers.get("X-NACF-Token")
        if given is None and allow_query:
            given = (query.get("t") or [None])[0]
        expected = self.server.api.token
        if given is None or not hmac.compare_digest(given.encode(), expected.encode()):
            raise _HttpError(403, "token inválido")

    def _check_post(self) -> bytes:
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{h}" for h in self._origins()}:
            raise _HttpError(403, "origem não permitida")
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise _HttpError(415, "Content-Type deve ser application/json")
        if self.headers.get("Transfer-Encoding"):
            raise _HttpError(400, "Content-Length obrigatório")
        if self._length is None:
            raise _HttpError(400, "Content-Length obrigatório")
        if self._length > MAX_BODY:
            raise _HttpError(413, "corpo grande demais")
        return self._body

    # --- despacho --------------------------------------------------------------------------

    def _drain_body(self) -> None:
        """Lê (limitado) o corpo antes de responder: fechar com bytes não lidos faz o Windows
        mandar RST e o cliente perde a resposta. Corpo enorme: só fecha a conexão."""
        self._length, self._body = None, b""
        raw = self.headers.get("Content-Length", "")
        if self.headers.get("Transfer-Encoding") or not raw.isascii() or not raw.isdigit():
            self.close_connection = True
            return
        self._length = int(raw)
        if self._length <= DRAIN_MAX:
            self._body = self.rfile.read(self._length)
        else:
            self.close_connection = True

    def _dispatch(self) -> None:
        try:
            self._drain_body()
            self._route()
        except _HttpError as e:
            self._json(e.status, {"error": e.message})
        except Exception:
            log.exception("erro em %s %s", self.command, urlsplit(self.path).path)
            self._json(500, {"error": "erro interno"})

    do_GET = do_POST = do_HEAD = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _dispatch

    def _route(self) -> None:
        self._check_host()
        url = urlsplit(self.path)
        path, query = url.path, parse_qs(url.query)
        method = self.command

        def allow(*methods: str) -> None:
            if method not in methods:
                raise _HttpError(405, "método não permitido")

        if not path.startswith("/api/"):
            raise _HttpError(404, "não encontrado")
        if path == "/api/ping":
            allow("GET")
            return self._json(200, {"app": APP_NAME})

        is_thumb = _THUMB_RE.fullmatch(path) is not None
        self._check_token(query, allow_query=is_thumb and method == "GET")
        if path == "/api/search":
            allow("GET")
            return self._search(query)
        if path == "/api/status":
            allow("GET")
            return self._status()
        if path == "/api/open":
            allow("POST")
            return self._open(self._check_post())
        if path == "/api/shutdown":
            allow("POST")
            return self._shutdown(self._check_post())
        if is_thumb:
            allow("GET")
            return self._thumb(int(path.rsplit("/", 1)[1]))
        raise _HttpError(404, "não encontrado")

    # --- endpoints -------------------------------------------------------------------------

    def _search(self, query: dict[str, list[str]]) -> None:
        q = (query.get("q") or [""])[0]
        type_ = (query.get("type") or [""])[0] or None
        if type_ == "all":
            type_ = None
        if type_ is not None and type_ not in TYPE_EXTS:
            raise _HttpError(400, "type inválido")
        raw = (query.get("limit") or [str(DEFAULT_LIMIT)])[0]
        try:
            limit = int(raw)
        except ValueError:
            raise _HttpError(400, "limit inválido") from None
        if not 1 <= limit <= MAX_LIMIT:
            raise _HttpError(400, f"limit deve estar entre 1 e {MAX_LIMIT}")
        if not q.strip():
            return self._json(200, {"semantic": False, "error": None, "took_ms": 0.0, "hits": []})
        try:
            res = self.server.api.searcher.search(q, limit=limit, type=type_, keyword_only=False)
        except ValueError as e:
            raise _HttpError(400, str(e)) from None
        self._json(200, {
            "semantic": res.semantic,
            "error": res.error,
            "took_ms": res.took_ms,
            "hits": [_hit_json(h) for h in res.hits],
        })

    def _status(self) -> None:
        api = self.server.api
        conn = db.connect(api.db_path)
        try:
            index = index_status(conn)
        finally:
            conn.close()
        extra = {}
        if api.status_extra is not None:
            try:
                extra = api.status_extra()
            except Exception:
                log.exception("status_extra falhou")
        self._json(200, {"app": APP_NAME, "index": index, **extra})

    def _file_row(self, file_id: int):
        conn = db.connect(self.server.api.db_path)
        try:
            return conn.execute("SELECT path, ext FROM files WHERE id = ?", (file_id,)).fetchone()
        finally:
            conn.close()

    def _open(self, raw: bytes) -> None:
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise _HttpError(400, "JSON inválido") from None
        if not isinstance(data, dict):
            raise _HttpError(400, "JSON deve ser um objeto")
        file_id, reveal = data.get("file_id"), data.get("reveal", False)
        if type(file_id) is not int or type(reveal) is not bool:
            raise _HttpError(400, "file_id (int) e reveal (bool) obrigatórios")
        row = self._file_row(file_id)
        if row is None or not os.path.isfile(row["path"]):
            raise _HttpError(404, "arquivo não encontrado")
        api = self.server.api
        try:
            if reveal:
                api.revealer(row["path"])
            elif type_of(row["ext"]) is None:  # nunca executa tipos que não indexamos
                raise _HttpError(404, "tipo de arquivo não pode ser aberto")
            else:
                api.opener(row["path"])
        except ValueError as e:
            raise _HttpError(400, str(e)) from None
        self._json(200, {"ok": True})

    def _shutdown(self, raw: bytes) -> None:
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise _HttpError(400, "JSON inválido") from None
        if data != {}:
            raise _HttpError(400, "corpo deve ser {}")
        callback = self.server.api.on_shutdown
        if callback is None:
            raise _HttpError(404, "não encontrado")
        self._json(200, {"ok": True})
        # depois da resposta, e fora desta thread: parar o servidor daqui travaria o serve_forever
        threading.Thread(target=callback, name="api-shutdown", daemon=True).start()

    def _thumb(self, file_id: int) -> None:
        row = self._file_row(file_id)
        if row is None or type_of(row["ext"]) != "image" or not os.path.isfile(row["path"]):
            raise _HttpError(404, "sem miniatura")
        try:
            body = _make_thumb(row["path"])
        except Exception:
            log.debug("miniatura falhou para %s", row["path"], exc_info=True)
            raise _HttpError(404, "sem miniatura") from None
        self._send(200, body, "image/jpeg", {"Cache-Control": "private, max-age=300"})


class ApiServer:
    def __init__(self, searcher, db_path, *, host: str = "127.0.0.1", port: int = 0,
                 status_extra: Callable[[], dict] | None = None,
                 opener: Callable[[str], None] | None = None,
                 revealer: Callable[[str], None] | None = None,
                 on_shutdown: Callable[[], None] | None = None) -> None:
        self.searcher = searcher
        self.db_path = db_path
        self.host = host
        self.port = port
        self.status_extra = status_extra
        self.opener = opener or default_opener
        self.revealer = revealer or default_revealer
        self.on_shutdown = on_shutdown
        self.token = secrets.token_urlsafe(24)
        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> "ApiServer":
        if self._httpd is not None:
            return self
        self._httpd = _Server((self.host, self.port), _Handler, self)  # OSError se porta ocupada
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, name="api-http",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        self._httpd = self._thread = None
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        if thread is not None:
            thread.join()
