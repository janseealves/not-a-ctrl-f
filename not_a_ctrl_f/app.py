"""Aplicação: orquestra banco, llama-server, watcher, worker e a API local.

A API sobe primeiro (só palavra-chave); embedder, worker e varredura inicial chegam depois,
em threads de fundo. Encerrar (Ctrl+C, /api/shutdown, request_stop) desfaz tudo na ordem inversa.
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
from importlib import metadata
from pathlib import Path

import httpx

from not_a_ctrl_f.installer import InstallError, console_progress, install_all, missing_components
from not_a_ctrl_f.processing.jobs import Worker
from not_a_ctrl_f.processing.watcher import Watcher
from not_a_ctrl_f.query.api import ApiServer
from not_a_ctrl_f.query.search import Searcher
from not_a_ctrl_f.shared import config, db
from not_a_ctrl_f.shared.embedder import Embedder, EmbedderError, LlamaServer

log = logging.getLogger(__name__)

APP_ID = "not-a-ctrl-f"
INSTANCE_FILE_NAME = "instance.json"
_LOG_MARK = "_nacf_handler"
_JOIN_TIMEOUT = 30.0


# --- instância única / log ---------------------------------------------------


def instance_file() -> Path:
    return config.data_dir() / INSTANCE_FILE_NAME


def read_instance() -> dict | None:
    """{"pid", "port", "token", "url"} da instância registrada, ou None (ausente/ilegível)."""
    try:
        data = json.loads(instance_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_instance(data: dict) -> None:
    path = instance_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def _remove_instance(token: str) -> None:
    """Apaga instance.json só se ainda for o nosso (mesmo pid e token)."""
    info = read_instance()
    if info is not None and info.get("pid") == os.getpid() and info.get("token") == token:
        instance_file().unlink(missing_ok=True)


def ping(port: int, timeout: float = 1.0) -> bool:
    """True se há um not-a-ctrl-f respondendo em 127.0.0.1:port."""
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            r = client.get(f"http://127.0.0.1:{port}/api/ping")
        return r.status_code == 200 and r.json() == {"app": APP_ID}
    except Exception:
        return False


def setup_logging(*, console: bool = True) -> Path:
    """Log em data_dir()/logs/app.log (5 MB x 3) e, se houver stderr (pythonw não tem), no console."""
    root = logging.getLogger()
    for h in root.handlers:
        if getattr(h, _LOG_MARK, False) and isinstance(h, logging.handlers.RotatingFileHandler):
            return Path(h.baseFilename)
    path = config.data_dir() / "logs" / "app.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(
            path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
    ]
    if console and sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    for h in handlers:
        h.setFormatter(fmt)
        setattr(h, _LOG_MARK, True)
        root.addHandler(h)
    root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # uma linha por requisição ao llama-server
    return path


# --- embedder ----------------------------------------------------------------


def _short(msg: object, limit: int = 200) -> str:
    lines = str(msg).strip().splitlines()
    return (lines[0].strip() if lines else "erro desconhecido")[:limit]


class _EmbedderProxy:
    """Entregue ao Searcher na subida: falha com EmbedderError até o embedder real chegar."""

    def __init__(self) -> None:
        self._embedder = None
        self._message = "servidor de embeddings iniciando"

    def set(self, embedder) -> None:
        self._embedder = embedder

    def unavailable(self, message: str) -> None:
        self._message = message

    def embed_query(self, text: str):
        embedder = self._embedder
        if embedder is None:
            raise EmbedderError(self._message)
        return embedder.embed_query(text)


# --- app ---------------------------------------------------------------------


class App:
    def __init__(self, *, port: int | None = None, install: bool = True):
        self._port = port
        self._install = install
        self.api: ApiServer | None = None

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._proxy = _EmbedderProxy()
        self._searcher: Searcher | None = None
        self._llama = None
        self._embedders: list = []
        self._worker: Worker | None = None
        self._watcher: Watcher | None = None
        self._threads: list[threading.Thread] = []
        self._boot_thread: threading.Thread | None = None
        self._embedder_status = "iniciando"
        self._scanning = False
        self._watch_dirs: list[Path] = []

    # --- API pública ---

    def request_stop(self) -> None:
        self._stop.set()

    def status_extra(self) -> dict:
        try:
            version = metadata.version(APP_ID)
        except metadata.PackageNotFoundError:
            version = "dev"
        return {
            "version": version,
            "embedder": self._embedder_status,
            "scanning": self._scanning,
            "watch_dirs": [str(p) for p in self._watch_dirs],
        }

    def run(self) -> int:
        try:
            return self._run()
        finally:
            self._teardown()

    # --- subida ---

    def _run(self) -> int:
        port = config.ui_port() if self._port is None else self._port
        if port != 0 and ping(port):
            log.info("já existe uma instância em http://127.0.0.1:%d/", port)
            return 0

        missing = missing_components()
        if missing:
            if not self._install:
                log.error("faltam componentes: %s", ", ".join(missing))
                return 1
            log.info("instalando: %s", ", ".join(missing))
            try:
                install_all(progress=console_progress if sys.stderr is not None else None)
            except InstallError as e:
                log.error("falha na instalação: %s", e)
                return 1

        db_path = config.default_db_path()
        try:
            db.connect(db_path).close()  # cria/migra
        except Exception:
            log.exception("falha ao abrir o banco %s", db_path)
            return 1
        self._db_path = db_path
        self._watch_dirs = config.watched_dirs()
        self._scanning = True

        self._searcher = Searcher(db_path, embedder=self._proxy)
        api = ApiServer(
            self._searcher, db_path, port=port, status_extra=self.status_extra,
            on_shutdown=self.request_stop,
        )
        try:
            api.start()
        except OSError as e:
            if port != 0 and ping(port):  # corrida com outra instância subindo ao mesmo tempo
                log.info("já existe uma instância em http://127.0.0.1:%d/", port)
                return 0
            log.error("porta %d em uso; defina outra com NACF_PORT (%s)", port, e)
            return 1
        self.api = api
        _write_instance({"pid": os.getpid(), "port": api.port, "token": api.token, "url": api.url})
        log.info("servidor pronto em %s", api.url)

        self._spawn("nacf-app-boot", self._boot_embedder, boot=True)
        self._spawn("nacf-app-watcher", self._boot_watcher)
        self._spawn("nacf-app-warm", self._warm)

        try:
            while not self._stop.wait(0.5):
                pass
        except KeyboardInterrupt:
            log.info("interrompido pelo usuário")
            self.request_stop()
        return 0

    def _spawn(self, name: str, target, *, boot: bool = False) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(t)
        if boot:
            self._boot_thread = t
        t.start()

    def _boot_embedder(self) -> None:
        try:
            self._start_embedder()
        except Exception as e:
            log.exception("falha ao subir o embedder")
            self._set_unavailable(_short(e))

    def _set_unavailable(self, msg: str) -> None:
        self._proxy.unavailable(f"servidor de embeddings indisponível: {msg}")
        self._embedder_status = f"indisponível: {msg}"

    def _start_embedder(self) -> None:
        llama = None
        error: Exception | None = None
        for gpu_layers in (None, 0):
            if self._stop.is_set():
                return
            candidate = LlamaServer() if gpu_layers is None else LlamaServer(gpu_layers=gpu_layers)
            self._llama = candidate
            try:
                candidate.start()
            except EmbedderError as e:
                candidate.close()
                error = e
                if gpu_layers is None:
                    log.warning("GPU falhou, usando CPU: %s", _short(e))
                continue
            except Exception as e:
                candidate.close()
                if self._stop.is_set():  # close() de outra thread no meio do start()
                    return
                error = e
                break
            llama = candidate
            break
        if self._stop.is_set():
            if llama is not None:
                llama.close()
            return
        if llama is None:
            msg = _short(error) if error else "erro desconhecido"
            log.error("embedder indisponível: %s (a busca por palavra-chave segue funcionando)", msg)
            self._set_unavailable(msg)
            return

        log.info("servidor de embeddings pronto em %s", llama.base_url)
        for_worker, for_search = Embedder(llama.base_url), Embedder(llama.base_url)
        self._embedders = [for_worker, for_search]
        self._proxy.set(for_search)
        self._embedder_status = "ok"
        worker = Worker(self._db_path, for_worker, home=Path.home())
        with self._lock:
            if self._stop.is_set():
                return
            self._worker = worker
        worker.start()
        log.info("worker de indexação iniciado")

    def _notify_worker(self) -> None:
        worker = self._worker
        if worker is not None:
            worker.notify()

    def _boot_watcher(self) -> None:
        try:
            watcher = Watcher(
                self._db_path, self._watch_dirs, home=Path.home(), on_pending=self._notify_worker
            )
            with self._lock:
                if self._stop.is_set():
                    return
                self._watcher = watcher
            log.info("varredura inicial: %s", ", ".join(map(str, self._watch_dirs)) or "(sem pastas)")
            watcher.start()
            log.info("varredura inicial concluída")
        except Exception:
            log.exception("falha ao iniciar o watcher")
            return
        finally:
            self._scanning = False
        self._notify_worker()

    def _warm(self) -> None:
        try:
            if self._searcher is not None:
                self._searcher.warm()
        except Exception:
            log.exception("falha ao carregar o cache de vetores")

    # --- encerramento ---

    def _step(self, what: str, fn) -> None:
        try:
            fn()
        except Exception:
            log.exception("erro ao encerrar: %s", what)

    def _teardown(self) -> None:
        self._stop.set()
        if self._threads:
            log.info("encerrando")
        if self.api is not None:
            self._step("api", self.api.stop)
        # a subida do llama-server pode estar em andamento: derruba para destravar o start()
        boot = self._boot_thread
        if boot is not None and boot.is_alive() and self._llama is not None:
            self._step("llama-server (subida)", self._llama.close)
        if boot is not None:
            boot.join(_JOIN_TIMEOUT)
        with self._lock:
            watcher, worker = self._watcher, self._worker
        if watcher is not None:
            self._step("watcher", watcher.stop)
        if worker is not None:
            self._step("worker", worker.stop)
        for e in self._embedders:
            self._step("embedder", e.close)
        if self._llama is not None:
            self._step("llama-server", self._llama.close)
        for t in self._threads:
            t.join(_JOIN_TIMEOUT)
        if self._searcher is not None:
            self._step("searcher", self._searcher.close)
        if self.api is not None:
            self._step("instance.json", lambda: _remove_instance(self.api.token))
