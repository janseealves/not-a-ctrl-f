"""Operações no banco que reconciliam `files` com o disco.

Só mexe em `files` (e no texto do chunk `name` num rename). Não extrai nem embedda:
quem processa as linhas `state='pending'` é o worker. Nenhuma função aqui abre
transação, exceto `scan`, que grava em lotes curtos.
"""

import os
import sqlite3
import stat
import time
from pathlib import Path

from not_a_ctrl_f.processing import extractors
from not_a_ctrl_f.processing.tracking import (
    ATTR_IGNORED,
    ATTR_ONLINE,
    ONLINE_ERROR,
    data_dir_key,
    dir_ok,
    file_name_ok,
    is_excluded_dir,
    root_key,
    should_track,
)
from not_a_ctrl_f.shared import db

_ERROR_INVALID_NAME = 123
_COUNT_KEYS = ("added", "changed", "unchanged", "removed", "moved", "skipped_online", "errors")

# ---------- helpers de disco/linha ----------

def _signed(n: int) -> int:
    """SQLite guarda inteiros de 64 bits com sinal; o file index do NTFS pode passar disso."""
    return n - (1 << 64) if n >= 1 << 63 else n


def _stat_file(path: str | Path) -> os.stat_result | None:
    """stat de um arquivo regular; None se não existe ou não é arquivo."""
    try:
        st = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as e:
        if getattr(e, "winerror", 0) == _ERROR_INVALID_NAME:
            return None
        raise
    return st if stat.S_ISREG(st.st_mode) else None


def _fields(st: os.stat_result) -> tuple[int, int, int | None, int]:
    """(size, mtime_ns, file_ref, volume_serial). Só os.stat de verdade traz st_ino/st_dev."""
    return st.st_size, st.st_mtime_ns, _signed(st.st_ino) or None, _signed(st.st_dev)


def _online(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & ATTR_ONLINE)


def _state(online: bool) -> tuple[str, str | None]:
    return ("skipped", ONLINE_ERROR) if online else ("pending", None)


def _ext(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def _insert(conn: sqlite3.Connection, path: Path, key: str, st: os.stat_result) -> int:
    size, mtime, ref, vol = _fields(st)
    state, error = _state(_online(st))
    cur = conn.execute(
        "INSERT INTO files(path, path_key, name, ext, size, mtime_ns, file_ref, volume_serial,"
        " state, error) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (str(path), key, path.name, _ext(path), size, mtime, ref, vol, state, error),
    )
    return cur.lastrowid


def _update(conn: sqlite3.Connection, file_id: int, path: Path, st: os.stat_result) -> None:
    """Arquivo mudou: volta para a fila (ou skipped se só online)."""
    size, mtime, ref, vol = _fields(st)
    state, error = _state(_online(st))
    conn.execute(
        "UPDATE files SET path=?, name=?, ext=?, size=?, mtime_ns=?, file_ref=?, volume_serial=?,"
        " state=?, error=?, attempts=0, indexed_at=NULL WHERE id=?",
        (str(path), path.name, _ext(path), size, mtime, ref, vol, state, error, file_id),
    )


def _move_row(conn: sqlite3.Connection, file_id: int, new_path: Path, home: Path | None) -> None:
    """Mesmo arquivo, outro caminho: mantém id, state, chunks e vetores."""
    conn.execute(
        "UPDATE files SET path=?, path_key=?, name=?, ext=? WHERE id=?",
        (str(new_path), db.path_key(new_path), new_path.name, _ext(new_path), file_id),
    )
    # os triggers atualizam o FTS
    conn.execute(
        "UPDATE chunks SET text=? WHERE file_id=? AND kind='name'",
        (extractors.name_text(new_path, home), file_id),
    )


# ---------- operações em um caminho ----------

def apply_path(conn: sqlite3.Connection, path: str | Path, *, home: Path | None = None,
               roots: list[Path] | None = None) -> str:
    """Reconcilia UM caminho com o disco. Devolve added|changed|unchanged|removed|moved|ignored."""
    path = Path(os.path.abspath(path))
    key = db.path_key(path)
    row = conn.execute(
        "SELECT id, size, mtime_ns, file_ref, state, error FROM files WHERE path_key = ?", (key,)
    ).fetchone()
    st = _stat_file(path)
    if st is None or not should_track(path, roots=roots, st=st):
        if row:
            conn.execute("DELETE FROM files WHERE id = ?", (row["id"],))
            return "removed"
        return "ignored"

    size, mtime, ref, vol = _fields(st)
    online = _online(st)
    if row:
        was_online = row["state"] == "skipped" and row["error"] == ONLINE_ERROR
        if (row["size"], row["mtime_ns"], row["file_ref"]) == (size, mtime, ref) \
                and was_online == online:
            return "unchanged"
        _update(conn, row["id"], path, st)
        return "changed"

    if ref:  # rename detectado por identidade
        old_rows = conn.execute(
            "SELECT id, path, size, mtime_ns FROM files WHERE volume_serial = ? AND file_ref = ?",
            (vol, ref),
        ).fetchall()
        for old in old_rows:
            if old["size"] == size and old["mtime_ns"] == mtime and not os.path.exists(old["path"]):
                _move_row(conn, old["id"], path, home)
                return "moved"
    _insert(conn, path, key, st)
    return "added"


def apply_move(conn: sqlite3.Connection, src: str | Path, dest: str | Path, *,
               home: Path | None = None, roots: list[Path] | None = None) -> str:
    """Evento de rename/move de arquivo. Preserva o id quando dá."""
    src = Path(os.path.abspath(src))
    dest = Path(os.path.abspath(dest))
    skey, dkey = db.path_key(src), db.path_key(dest)
    row = conn.execute(
        "SELECT id, size, mtime_ns FROM files WHERE path_key = ?", (skey,)
    ).fetchone()
    st = _stat_file(dest)
    # skey == dkey: só mudou a caixa do nome (src "existe" porque o NTFS ignora a caixa)
    if row and st is not None and should_track(dest, roots=roots, st=st) \
            and (skey == dkey or _stat_file(src) is None):
        if skey != dkey:
            conn.execute("DELETE FROM files WHERE path_key = ?", (dkey,))  # sobrescrita
        _move_row(conn, row["id"], dest, home)
        if (st.st_size, st.st_mtime_ns) != (row["size"], row["mtime_ns"]):
            _update(conn, row["id"], dest, st)
        return "moved"
    apply_path(conn, src, home=home, roots=roots)
    return apply_path(conn, dest, home=home, roots=roots)


def _tree_prefix(folder: str | Path) -> str:
    return db.path_key(folder).rstrip(os.sep) + os.sep


_IN_TREE = "path_key >= ? AND path_key < ? || char(1114111)"


def remove_tree(conn: sqlite3.Connection, folder: str | Path) -> int:
    """Pasta apagada: remove todas as linhas sob ela."""
    prefix = _tree_prefix(folder)
    return conn.execute(f"DELETE FROM files WHERE {_IN_TREE}", (prefix, prefix)).rowcount


def move_tree(conn: sqlite3.Connection, src_dir: str | Path, dest_dir: str | Path, *,
              home: Path | None = None, roots: list[Path] | None = None) -> int:
    """Pasta renomeada/movida: reescreve path, path_key e o chunk name de todas as linhas sob ela."""
    src_dir = Path(os.path.abspath(src_dir))
    dest_dir = Path(os.path.abspath(dest_dir))
    if not dir_ok(dest_dir, roots):
        return remove_tree(conn, src_dir)
    prefix = _tree_prefix(src_dir)
    rows = conn.execute(
        f"SELECT id, path FROM files WHERE {_IN_TREE}", (prefix, prefix)
    ).fetchall()
    for r in rows:
        new = Path(os.path.join(dest_dir, r["path"][len(prefix):]))
        # linha velha no destino (sobrescrita) cede lugar
        conn.execute("DELETE FROM files WHERE path_key = ? AND id <> ?", (db.path_key(new), r["id"]))
        _move_row(conn, r["id"], new, home)
    return len(rows)


# ---------- varredura ----------

_BATCH_MAX_AGE = 0.5  # segundos: nenhuma transação do scan dura mais que isso


class _Batch:
    """Transações curtas: abre na primeira escrita; fecha com `size` operações ou _BATCH_MAX_AGE s.
    Um walk sem mudanças não segura o lock de escrita."""

    def __init__(self, conn: sqlite3.Connection, size: int) -> None:
        self._conn = conn
        self._size = size
        self._open = False
        self._ops = 0
        self._t0 = 0.0

    def begin(self) -> None:
        """Chamar antes de cada escrita."""
        if not self._open:
            self._conn.execute("BEGIN IMMEDIATE")
            self._open = True
            self._ops = 0
            self._t0 = time.monotonic()

    def done(self) -> None:
        """Chamar depois de cada escrita."""
        self._ops += 1
        if self._ops >= self._size:
            self.commit()

    def poll(self) -> None:
        """Chamar a cada arquivo visto: fecha a transação velha mesmo sem novas escritas."""
        if self._open and time.monotonic() - self._t0 >= _BATCH_MAX_AGE:
            self.commit()

    def commit(self) -> None:
        if self._open:
            self._conn.execute("COMMIT")
            self._open = False

    def rollback(self) -> None:
        if self._open:
            self._conn.execute("ROLLBACK")
            self._open = False


def scan(conn: sqlite3.Connection, roots: list[Path], *, home: Path | None = None,
         batch: int = 500) -> dict[str, int]:
    """Reconciliação completa disco x banco. Grava em transações curtas (até `batch` operações ou ~0,5 s), abertas só na primeira escrita."""
    counts = dict.fromkeys(_COUNT_KEYS, 0)
    roots = [Path(os.path.abspath(r)) for r in roots]
    root_keys = [root_key(r) for r in roots]
    data_key = data_dir_key()

    # key -> (id, size, mtime_ns, file_ref, state, error)
    known: dict[str, tuple] = {}
    by_ref: dict[tuple[int, int], list[str]] = {}
    for r in conn.execute(
        "SELECT id, path_key, size, mtime_ns, file_ref, volume_serial, state, error FROM files"
    ):
        known[r["path_key"]] = (r["id"], r["size"], r["mtime_ns"], r["file_ref"], r["state"], r["error"])
        if r["file_ref"]:
            by_ref.setdefault((r["volume_serial"], r["file_ref"]), []).append(r["path_key"])

    seen: set[str] = set()
    bad_dirs: list[str] = []  # pastas que não deu para listar: não apagar o que está sob elas
    deferred: list[tuple[str, Path, os.stat_result]] = []  # "novos" que podem ser renames
    tx = _Batch(conn, batch)

    def onerror(e: OSError) -> None:
        counts["errors"] += 1
        d = root_key(e.filename) if e.filename else ""
        # pasta que sumiu (FileNotFoundError) fora do root: as linhas dela podem ser apagadas
        if d and not (isinstance(e, FileNotFoundError) and d not in root_keys):
            bad_dirs.append(d)

    try:
        for root in roots:
            for dirpath, dirs, files in os.walk(root, onerror=onerror, followlinks=False):
                dirs[:] = [
                    d for d in dirs
                    if not is_excluded_dir(d)
                    and os.path.normcase(os.path.join(dirpath, d)) != data_key
                ]
                for name in files:
                    tx.poll()
                    if not file_name_ok(name):
                        continue
                    path = Path(dirpath, name)
                    key = os.path.normcase(str(path))
                    try:
                        st = os.stat(path)
                    except FileNotFoundError:
                        continue
                    except OSError:
                        counts["errors"] += 1
                        seen.add(key)  # existe, mas não deu para ler: preserva a linha
                        continue
                    if not stat.S_ISREG(st.st_mode) or getattr(st, "st_file_attributes", 0) & ATTR_IGNORED:
                        continue
                    seen.add(key)
                    online = _online(st)
                    counts["skipped_online"] += online
                    size, mtime, ref, vol = _fields(st)
                    k = known.get(key)
                    if k:
                        fid, k_size, k_mtime, k_ref, k_state, k_error = k
                        was_online = k_state == "skipped" and k_error == ONLINE_ERROR
                        if (k_size, k_mtime, k_ref) == (size, mtime, ref) and was_online == online:
                            counts["unchanged"] += 1
                        else:
                            tx.begin()
                            _update(conn, fid, path, st)
                            counts["changed"] += 1
                            tx.done()
                    elif ref and (vol, ref) in by_ref:
                        deferred.append((key, path, st))
                    else:
                        tx.begin()
                        _insert(conn, path, key, st)
                        counts["added"] += 1
                        tx.done()

        # "novos" com a identidade de uma linha que sumiu do disco = arquivo movido
        unseen = {k for k in known if k not in seen}
        for key, path, st in deferred:
            size, mtime, ref, vol = _fields(st)
            match = next(
                (o for o in by_ref[(vol, ref)]
                 if o in unseen and known[o][1] == size and known[o][2] == mtime),
                None,
            )
            tx.begin()
            if match:
                unseen.discard(match)
                _move_row(conn, known[match][0], path, home)
                counts["moved"] += 1
            else:
                _insert(conn, path, key, st)
                counts["added"] += 1
            tx.done()

        # o que não foi visto: apaga, exceto sob pasta que não pôde ser listada
        for key in unseen:
            in_root = any(key.startswith(rk + os.sep) for rk in root_keys)
            if in_root and any(key.startswith(bd + os.sep) for bd in bad_dirs):
                continue
            tx.poll()
            tx.begin()
            conn.execute("DELETE FROM files WHERE id = ?", (known[key][0],))
            counts["removed"] += 1
            tx.done()
        tx.commit()
    except BaseException:
        tx.rollback()
        raise
    return counts
