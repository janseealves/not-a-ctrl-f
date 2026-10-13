"""Watcher ao vivo: mantém `files` sincronizada via watchdog.

Uma única thread de escrita (com a própria conexão) faz debounce por caminho e aplica
as operações de `sync`. O handler do watchdog só enfileira.
"""

import logging
import os
import queue
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from not_a_ctrl_f.processing.sync import (
    apply_move,
    apply_path,
    move_tree,
    remove_tree,
    scan,
)
from not_a_ctrl_f.processing.tracking import dir_ok, name_ok
from not_a_ctrl_f.shared import db

try:  # backend do Windows (ReadDirectoryChangesW); a subclasse detecta perda de eventos
    from watchdog.observers.read_directory_changes import WindowsApiEmitter
    from watchdog.observers.winapi import (
        WinAPINativeEvent,
        _parse_event_buffer,
        read_directory_changes,
    )
except ImportError:  # pragma: no cover
    WindowsApiEmitter = None

log = logging.getLogger(__name__)

# ---------- watcher ao vivo ----------

_WAKE = object()  # acorda a thread de escrita
_CHUNK = 200  # caminhos por transação no ciclo de escrita


class _Handler(FileSystemEventHandler):
    """Só enfileira (tipo, src, dest, é_pasta, timestamp); nada de banco nem stat aqui."""

    def __init__(self, watcher: "Watcher") -> None:
        self._w = watcher

    def on_any_event(self, event: FileSystemEvent) -> None:
        kind = event.event_type
        is_dir = event.is_directory
        if kind not in ("created", "modified", "deleted", "moved"):
            return
        if is_dir and kind in ("created", "modified"):
            return  # o conteúdo chega em eventos de arquivo
        src = os.fsdecode(event.src_path)
        dest = os.fsdecode(getattr(event, "dest_path", "") or "")
        if kind == "moved" and src and getattr(event, "is_synthetic", False):
            return  # sub-evento de pasta movida: move_tree já cobre
        roots = self._w._roots
        if is_dir:
            keep = (src and dir_ok(src, roots)) or (dest and dir_ok(dest, roots))
        elif kind == "deleted":
            # o Windows reporta pasta apagada como arquivo apagado
            keep = name_ok(src, roots) or dir_ok(src, roots)
        else:
            keep = (src and name_ok(src, roots)) or (dest and name_ok(dest, roots))
        if keep:
            self._w._submit((kind, src, dest, is_dir, time.monotonic()))


def _make_emitter_class(on_lost: Callable[[], None]):
    """Emissor do Windows que avisa quando o buffer estoura (leitura volta com 0 bytes).

    Depende de internos do watchdog 6.0.0 (`read_directory_changes`, `_parse_event_buffer`,
    `WinAPINativeEvent`), por isso a versão está fixada no pyproject: a API pública não
    expõe overflow (devolve lista vazia), e sem isso perderíamos eventos em silêncio.
    """

    class Emitter(WindowsApiEmitter):
        def _read_events(self):
            if not self._whandle:
                return []
            try:
                buf, n = read_directory_changes(
                    self._whandle, self.watch.path, recursive=self.watch.is_recursive
                )
            except OSError:
                log.exception("erro lendo mudanças de %s", self.watch.path)
                on_lost()
                time.sleep(1.0)
                return []
            if n == 0:
                if self.should_keep_running():
                    on_lost()
                return []
            return [WinAPINativeEvent(a, p) for a, p in _parse_event_buffer(buf, n)]

    return Emitter


class Watcher:
    def __init__(self, db_path: Path, roots: list[Path], *, home: Path | None = None,
                 debounce: float = 1.5, rescan_interval: float | None = 1800.0,
                 on_pending: Callable[[], None] | None = None) -> None:
        self._db_path = Path(db_path)
        self._roots = [Path(os.path.abspath(r)) for r in roots]
        self._home = home
        self._debounce = debounce
        self._rescan_interval = rescan_interval
        self._on_pending = on_pending

        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        # tudo abaixo é protegido por _lock
        self._inflight = 0  # eventos na fila ainda não absorvidos
        self._dirty: dict[str, list] = {}  # path_key -> [instante do último evento, Path]
        self._moves: list[list[Path]] = []  # [src, dest]
        self._immediate: list[tuple] = []  # ops de pasta, sem debounce
        self._processing = False
        self._scan_requested = False

        self._next_rescan: float | None = None
        self._stop_event = threading.Event()
        self._ready = threading.Event()  # scan inicial terminou
        self._start_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._observer: BaseObserver | None = None
        self._stopped = False

    # --- ciclo de vida ---

    def start(self) -> "Watcher":
        if self._thread is not None:
            raise RuntimeError("Watcher já iniciado")
        # o observador sobe antes do scan para não perder o que mudar durante ele;
        # os eventos ficam na fila até a thread de escrita terminar o scan.
        self._observer = self._make_observer()
        handler = _Handler(self)
        for root in self._roots:
            try:
                self._observer.schedule(handler, str(root), recursive=True)
            except OSError:
                log.warning("pasta não monitorada (inacessível): %s", root)
        self._observer.start()
        self._thread = threading.Thread(target=self._run, name="nacf-watcher-writer", daemon=False)
        self._thread.start()
        self._ready.wait()
        if self._start_error is not None:
            self.stop()
            raise self._start_error
        return self

    def stop(self, timeout: float = 10.0) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout)
            except Exception:
                log.exception("erro ao parar o observer")
        self._stop_event.set()
        self._q.put(_WAKE)
        if self._thread is not None:
            self._thread.join(timeout)

    def wait_idle(self, timeout: float) -> bool:
        """Para testes: True quando fila e debounce esvaziaram e nada está sendo escrito."""
        end = time.monotonic() + timeout
        while True:
            with self._lock:
                idle = not (self._inflight or self._dirty or self._moves or self._immediate
                            or self._processing or self._scan_requested)
            if idle and self._ready.is_set():
                return True
            if time.monotonic() >= end:
                return False
            time.sleep(0.02)

    def __enter__(self) -> "Watcher":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # --- observer / fila ---

    def _make_observer(self) -> BaseObserver:
        if WindowsApiEmitter is None:  # pragma: no cover
            return Observer()
        cls = _make_emitter_class(self._request_scan)
        return BaseObserver(
            lambda q, watch, *, timeout, event_filter=None: cls(
                q, watch, timeout=timeout, event_filter=event_filter
            )
        )

    def _submit(self, item: tuple) -> None:
        with self._lock:
            self._inflight += 1
        self._q.put(item)

    def _request_scan(self) -> None:
        """Eventos podem ter sido perdidos: agenda varredura completa."""
        with self._lock:
            self._scan_requested = True
        self._q.put(_WAKE)

    def _ingest(self, item: Any) -> None:
        if item is _WAKE:
            return
        kind, src, dest, is_dir, ts = item
        with self._lock:
            self._inflight -= 1
            if is_dir:
                if kind == "moved" and src:
                    self._immediate.append(("move_tree", Path(src), Path(dest)))
                elif kind == "deleted":
                    self._immediate.append(("remove_tree", Path(src)))
            elif kind == "moved" and src:
                s, d = Path(src), Path(dest)
                sk = db.path_key(s)
                for m in reversed(self._moves):  # A->B seguido de B->C vira A->C
                    if db.path_key(m[1]) == sk:
                        m[1] = d
                        break
                else:
                    self._moves.append([s, d])
                self._dirty[sk] = [ts, s]
                self._dirty[db.path_key(d)] = [ts, d]
            else:
                p = Path(dest if kind == "moved" else src)  # move sem origem = criado
                self._dirty[db.path_key(p)] = [ts, p]

    # --- thread de escrita ---

    def _run(self) -> None:
        try:
            conn = db.connect(self._db_path)
        except BaseException as e:
            self._start_error = e
            self._ready.set()
            return
        try:
            self._full_scan(conn)
            self._ready.set()
            while not self._stop_event.is_set():
                try:
                    self._wait_and_ingest()
                    if self._rescan_due():
                        self._full_scan(conn)
                    self._cycle(conn)
                except Exception:
                    log.exception("erro no ciclo do watcher")
                    time.sleep(0.1)
            try:
                self._cycle(conn)  # o que já estava pronto
            except Exception:
                log.exception("erro no ciclo final do watcher")
        finally:
            self._ready.set()
            conn.close()

    def _rescan_due(self) -> bool:
        with self._lock:
            requested = self._scan_requested
        return requested or (self._next_rescan is not None and time.monotonic() >= self._next_rescan)

    def _wait_and_ingest(self) -> None:
        now = time.monotonic()
        with self._lock:
            due = [e[0] + self._debounce for e in self._dirty.values()]
        wait = min(due) - now if due else 1.0
        if self._next_rescan is not None:
            wait = min(wait, self._next_rescan - now)
        try:
            item = self._q.get(timeout=min(max(wait, 0.01), 1.0))
        except queue.Empty:
            return
        self._ingest(item)
        while True:
            try:
                self._ingest(self._q.get_nowait())
            except queue.Empty:
                return

    def _full_scan(self, conn: sqlite3.Connection) -> None:
        with self._lock:
            self._processing = True
            self._scan_requested = False
        try:
            counts = scan(conn, self._roots, home=self._home)
            log.info("scan: %s", counts)
            if counts["added"] or counts["changed"] or counts["moved"]:
                self._notify()
        except Exception:
            log.exception("erro na varredura")
        finally:
            if self._rescan_interval is not None:
                self._next_rescan = time.monotonic() + self._rescan_interval
            with self._lock:
                self._processing = False

    def _guard(self, conn: sqlite3.Connection, fn: Callable[[], Any]) -> Any:
        """Uma operação com SAVEPOINT: se falhar, desfaz só ela, loga e segue."""
        conn.execute("SAVEPOINT op")
        try:
            result = fn()
        except Exception:
            log.exception("falha numa operação do watcher")
            conn.execute("ROLLBACK TO op")
            conn.execute("RELEASE op")
            return None
        conn.execute("RELEASE op")
        return result

    def _cycle(self, conn: sqlite3.Connection) -> None:
        now = time.monotonic()
        deb = self._debounce
        with self._lock:
            ready_moves, kept, blocked = [], [], set()
            for m in self._moves:
                ks = [db.path_key(m[0]), db.path_key(m[1])]
                if all(k in self._dirty and now - self._dirty[k][0] >= deb for k in ks):
                    ready_moves.append(m)
                else:
                    kept.append(m)
                    blocked.update(ks)
            self._moves = kept
            ready = [(k, e[1]) for k, e in self._dirty.items()
                     if now - e[0] >= deb and k not in blocked]
            for k, _ in ready:
                del self._dirty[k]
            immediate, self._immediate = self._immediate, []
            if not (ready_moves or ready or immediate):
                return
            self._processing = True
        try:
            # quem existe vai primeiro: assim "apagou A + criou B" vira rename (mesmo id)
            paths = sorted((p for _, p in ready), key=lambda p: not os.path.lexists(p))
            pending = False
            # transações curtas: um lote grande de eventos vira vários commits
            chunks = [(immediate, ready_moves, [])]
            chunks += [([], [], paths[i:i + _CHUNK]) for i in range(0, len(paths), _CHUNK)]
            chunks = [c for c in chunks if any(c)]
            for n, (imm, mvs, pts) in enumerate(chunks):
                try:
                    with db.transaction(conn):
                        pending |= self._apply(conn, imm, mvs, pts)
                except Exception:
                    log.exception("transação do watcher falhou; reagendando")
                    for imm2, mvs2, pts2 in chunks[n:]:
                        self._requeue(mvs2, pts2, imm2)
                    return
            if pending:
                self._notify()
        finally:
            with self._lock:
                self._processing = False

    def _apply(self, conn: sqlite3.Connection, immediate: list, moves: list, paths: list[Path]) -> bool:
        """Aplica um lote dentro da transação aberta. True se algo ficou pending."""
        pending = False
        for op in immediate:
            if op[0] == "move_tree":
                self._guard(conn, lambda op=op: move_tree(
                    conn, op[1], op[2], home=self._home, roots=self._roots))
            else:
                self._guard(conn, lambda op=op: remove_tree(conn, op[1]))
        for src, dest in moves:
            r = self._guard(conn, lambda s=src, d=dest: apply_move(
                conn, s, d, home=self._home, roots=self._roots))
            pending |= r in ("added", "changed") or (r == "moved" and self._is_pending(conn, dest))
        for p in paths:
            r = self._guard(conn, lambda p=p: apply_path(
                conn, p, home=self._home, roots=self._roots))
            if r == "ignored" and not os.path.lexists(p):
                # pasta apagada chega como arquivo apagado
                self._guard(conn, lambda p=p: remove_tree(conn, p))
            pending |= r in ("added", "changed") or (r == "moved" and self._is_pending(conn, p))
        return pending

    @staticmethod
    def _is_pending(conn: sqlite3.Connection, path: Path) -> bool:
        row = conn.execute("SELECT state FROM files WHERE path_key = ?", (db.path_key(path),)).fetchone()
        return bool(row) and row["state"] == "pending"

    def _requeue(self, moves: list, paths: list[Path], immediate: list) -> None:
        now = time.monotonic()
        with self._lock:
            self._moves = moves + self._moves
            self._immediate = immediate + self._immediate
            for m in moves:
                for p in m:
                    self._dirty.setdefault(db.path_key(p), [now, p])
            for p in paths:
                self._dirty.setdefault(db.path_key(p), [now, p])

    def _notify(self) -> None:
        if self._on_pending is None:
            return
        try:
            self._on_pending()
        except Exception:
            log.exception("on_pending falhou")
