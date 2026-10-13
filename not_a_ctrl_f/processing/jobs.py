"""Worker de indexação: thread que consome `files.state='pending'`.

Nunca segura transação durante extração/embedding. O watcher invalida o trabalho em curso
voltando a linha para 'pending' (ou apagando-a); `indexer.write_rows` percebe e não escreve.
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from not_a_ctrl_f.processing import indexer
from not_a_ctrl_f.processing.extractors import ExtractError, SkipFile
from not_a_ctrl_f.processing.tracking import ONLINE_ERROR
from not_a_ctrl_f.shared import db
from not_a_ctrl_f.shared.embedder import EmbedderError

log = logging.getLogger(__name__)

_MAX_ERR = 300
_NAME_BATCH = 200
_DB_RETRY = 5.0  # espera antes de tentar abrir o banco de novo

_CLAIM = (
    "UPDATE files SET state='indexing' WHERE id = ("
    " SELECT id FROM files WHERE state='pending' AND id NOT IN (SELECT value FROM json_each(?))"
    " ORDER BY mtime_ns DESC LIMIT 1)"
    " RETURNING id, path, size, mtime_ns, attempts"
)
_SAME = "id=? AND state='indexing' AND size=? AND mtime_ns=?"


def _short(msg: object) -> str:
    return " ".join(str(msg).split())[:_MAX_ERR]


class Worker:
    def __init__(self, db_path: Path, embedder, *, home: Path | None = None, max_attempts: int = 3,
                 retry_delay: float = 60.0, idle_wait: float = 30.0,
                 embedder_retry: float = 30.0) -> None:
        self._db_path = Path(db_path)
        self._embedder = embedder
        self._home = home
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay
        self._idle_wait = idle_wait
        self._embedder_retry = embedder_retry

        self._stop_event = threading.Event()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._lock = threading.Lock()  # notify x decisão de "ocioso"
        self._thread: threading.Thread | None = None
        # só a thread do worker mexe nestes
        self._backoff: dict[int, float] = {}  # id -> instante (monotonic) em que volta a ser elegível
        self._pause_until = 0.0
        self._current: sqlite3.Row | None = None

    # --- API ---

    def start(self) -> "Worker":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="nacf-worker")
            self._thread.start()
        return self

    def stop(self, timeout: float = 30.0) -> None:
        self._stop_event.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                log.warning("worker não terminou em %.0fs", timeout)

    def notify(self) -> None:
        with self._lock:
            self._idle.clear()
            self._wake.set()

    def wait_idle(self, timeout: float) -> bool:
        """True quando não há pending elegível (fora de backoff) nem arquivo em processamento."""
        return self._idle.wait(timeout)

    def __enter__(self) -> "Worker":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- thread ---

    def _run(self) -> None:
        conn: sqlite3.Connection | None = None
        try:
            while not self._stop_event.is_set():
                try:
                    conn = db.connect(self._db_path)
                    self._startup(conn)
                    break
                except Exception:
                    log.exception("worker: falha ao iniciar")
                    if conn is not None:
                        conn.close()
                        conn = None
                    self._stop_event.wait(_DB_RETRY)
            while conn is not None and not self._stop_event.is_set():
                try:
                    self._cycle(conn)
                except Exception:
                    log.exception("worker: erro inesperado no ciclo")
                    self._release(conn)
                    self._stop_event.wait(1.0)
        finally:
            if conn is not None:
                self._release(conn)
                conn.close()
            self._idle.set()  # não deixa wait_idle pendurado depois de parar

    def _startup(self, conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE files SET state='pending' WHERE state='indexing'")  # crash anterior
        if indexer.ensure_fingerprint(conn, self._embedder.fingerprint):
            log.info("worker: embedder mudou, arquivos voltaram para a fila")
        self._name_only_online(conn)

    def _cycle(self, conn: sqlite3.Connection) -> None:
        now = time.monotonic()
        if now < self._pause_until:  # embedder fora do ar
            self._stop_event.wait(self._pause_until - now)
            return
        self._wake.clear()
        if self._stop_event.is_set():
            return
        claim = self._claim(conn, now)
        if claim is not None:
            self._idle.clear()
            self._current = claim
            self._process(conn, claim)
            self._current = None
            return
        self._name_only_online(conn)
        with self._lock:
            if not self._wake.is_set():
                self._idle.set()
        timeout = self._idle_wait
        if self._backoff:
            timeout = min(timeout, max(0.0, min(self._backoff.values()) - time.monotonic()))
        self._wake.wait(timeout)

    # --- fila ---

    def _claim(self, conn: sqlite3.Connection, now: float) -> sqlite3.Row | None:
        self._backoff = {i: t for i, t in self._backoff.items() if t > now}
        rows = conn.execute(_CLAIM, (json.dumps(list(self._backoff)),)).fetchall()
        return rows[0] if rows else None

    def _requeue(self, conn: sqlite3.Connection, claim: sqlite3.Row, *, count: bool = False,
                 error: str | None = None, backoff: bool = True) -> None:
        """Volta para 'pending' se o arquivo ainda é o do claim."""
        conn.execute(
            "UPDATE files SET state='pending', attempts=attempts+?, error=COALESCE(?, error)"
            f" WHERE {_SAME}",
            (int(count), error, claim["id"], claim["size"], claim["mtime_ns"]),
        )
        if backoff:
            self._backoff[claim["id"]] = time.monotonic() + self._retry_delay

    def _release(self, conn: sqlite3.Connection) -> None:
        """Devolve à fila o que este worker deixou em 'indexing'."""
        claim, self._current = self._current, None
        if claim is None:
            return
        try:
            self._requeue(conn, claim, backoff=False)
        except sqlite3.Error:
            log.exception("worker: não consegui devolver o arquivo %s à fila", claim["id"])

    def _name_only_online(self, conn: sqlite3.Connection) -> None:
        """Arquivos só-online ganham o chunk 'name' (sem ler nem embeddar o conteúdo)."""
        while not self._stop_event.is_set():
            rows = conn.execute(
                "SELECT id, path FROM files f WHERE state='skipped' AND error=?"
                " AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.file_id=f.id AND c.kind='name')"
                " LIMIT ?", (ONLINE_ERROR, _NAME_BATCH),
            ).fetchall()
            if not rows:
                return
            texts = [(r["id"], indexer.name_row(Path(r["path"]), home=self._home)[5]) for r in rows]
            added = 0
            with db.transaction(conn):
                for fid, text in texts:
                    added += conn.execute(
                        "INSERT INTO chunks(file_id, kind, modality, seq, text)"
                        " SELECT ?, 'name', 'none', 0, ? WHERE"
                        " EXISTS (SELECT 1 FROM files WHERE id=? AND state='skipped' AND error=?)"
                        " AND NOT EXISTS (SELECT 1 FROM chunks WHERE file_id=? AND seq=0)",
                        (fid, text, fid, ONLINE_ERROR, fid),
                    ).rowcount
            if not added:
                return

    # --- um arquivo ---

    def _process(self, conn: sqlite3.Connection, claim: sqlite3.Row) -> None:
        path = Path(claim["path"])
        try:
            try:
                rows, info = indexer.build_rows(path, self._embedder, home=self._home)
            except SkipFile as e:
                self._finish_name_only(conn, claim, "skipped", _short(e))
            except ExtractError as e:
                self._finish_name_only(conn, claim, "failed", _short(e))
            except FileNotFoundError:
                self._requeue(conn, claim)  # o watcher vai remover a linha
            except OSError as e:  # arquivo travado, sem permissão...
                attempts = claim["attempts"] + 1
                if attempts >= self._max_attempts:
                    self._finish_name_only(conn, claim, "failed", _short(e), attempts=attempts)
                else:
                    self._requeue(conn, claim, count=True, error=_short(e))
            except EmbedderError as e:
                log.warning("worker: embedder indisponível (%s); pausando %.0fs", e,
                            self._embedder_retry)
                self._requeue(conn, claim, backoff=False)
                self._pause_until = time.monotonic() + self._embedder_retry
            except Exception as e:
                log.exception("worker: erro ao indexar %s", path)
                self._finish_name_only(conn, claim, "failed", _short(f"{type(e).__name__}: {e}"))
            else:
                self._write(conn, claim, rows, info, "done", None)
        except sqlite3.Error:
            log.exception("worker: erro de banco com %s", path)
            self._release(conn)  # _current ainda é o claim
            self._backoff[claim["id"]] = time.monotonic() + self._retry_delay

    def _finish_name_only(self, conn: sqlite3.Connection, claim: sqlite3.Row, state: str,
                          error: str, *, attempts: int | None = None) -> None:
        row = indexer.name_row(Path(claim["path"]), home=self._home)
        self._write(conn, claim, [row], {}, state, error, attempts)

    def _write(self, conn: sqlite3.Connection, claim: sqlite3.Row, rows: list[tuple], info: dict,
               state: str, error: str | None, attempts: int | None = None) -> None:
        if not indexer.write_rows(conn, claim["id"], claim["size"], claim["mtime_ns"], rows, info,
                                  state=state, error=error, attempts=attempts):
            log.debug("worker: %s mudou ou sumiu durante o processamento; descartado", claim["path"])
            return
        # renomeado no meio do caminho: só o chunk 'name' tem o caminho velho; o resto vale
        row = conn.execute("SELECT path FROM files WHERE id=?", (claim["id"],)).fetchone()
        if row is not None and row["path"] != claim["path"]:
            conn.execute(
                "UPDATE chunks SET text=? WHERE file_id=? AND kind='name'",
                (indexer.name_row(Path(row["path"]), home=self._home)[5], claim["id"]),
            )
