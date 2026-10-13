"""Quais arquivos e pastas entram no índice (filtros puros, sem banco)."""

import os
from pathlib import Path

from not_a_ctrl_f.processing import extractors
from not_a_ctrl_f.shared import config

ONLINE_ERROR = "disponível só online"

_EXCLUDED_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".cache", ".idea", ".vscode", "$recycle.bin", "appdata",
    "system volume information",
})
_IGNORED_NAMES = frozenset({"desktop.ini", "thumbs.db"})
_IGNORED_PREFIXES = ("~$", ".~lock.", "~wr")
_IGNORED_SUFFIXES = (".tmp", ".temp", ".crdownload", ".part", ".partial", ".download")

ATTR_IGNORED = 0x2 | 0x4 | 0x100  # HIDDEN | SYSTEM | TEMPORARY
ATTR_ONLINE = 0x400000 | 0x1000  # RECALL_ON_DATA_ACCESS | OFFLINE



# ---------- filtragem ----------

def is_excluded_dir(name: str) -> bool:
    n = name.lower()
    return n.startswith(".") or n in _EXCLUDED_DIRS


def file_name_ok(name: str) -> bool:
    """Só pelo nome: extensão suportada e fora dos temporários conhecidos."""
    n = name.lower()
    if n in _IGNORED_NAMES or n.startswith(_IGNORED_PREFIXES) or n.endswith(_IGNORED_SUFFIXES):
        return False
    return os.path.splitext(n)[1].lstrip(".") in extractors.SUPPORTED_EXTENSIONS


def root_key(root: str | Path) -> str:
    return os.path.normcase(os.path.abspath(root)).rstrip(os.sep)


def data_dir_key() -> str:
    return root_key(config.data_dir())


def _rel_parts(path: str | Path, roots: list[Path] | None) -> list[str] | None:
    """Componentes do caminho abaixo do root que o contém (ou do drive, sem roots).
    None se estiver fora de todos os roots ou dentro do data_dir."""
    p = os.path.abspath(path)
    k = os.path.normcase(p)
    if k.startswith(data_dir_key() + os.sep):
        return None
    if roots is None:
        rel = os.path.splitdrive(p)[1]
    else:
        for r in roots:
            rk = root_key(r)
            if k.startswith(rk + os.sep):
                rel = p[len(rk) + 1:]
                break
        else:
            return None
    return [x for x in rel.split(os.sep) if x]


def name_ok(path: str | Path, roots: list[Path] | None = None) -> bool:
    """should_track sem olhar atributos (sem stat)."""
    parts = _rel_parts(path, roots)
    if not parts:
        return False
    return not any(is_excluded_dir(d) for d in parts[:-1]) and file_name_ok(parts[-1])


def dir_ok(path: str | Path, roots: list[Path] | None = None) -> bool:
    """A pasta (e seus ancestrais abaixo do root) não é ignorada?"""
    parts = _rel_parts(path, roots)
    return bool(parts) and not any(is_excluded_dir(d) for d in parts)


def should_track(path: str | Path, *, roots: list[Path] | None = None,
                 st: os.stat_result | None = None) -> bool:
    """O arquivo entra no índice? `roots` restringe a checagem de pastas ignoradas ao que está
    abaixo do root (sem roots, vale o caminho todo: um root dentro de AppData seria ignorado)."""
    if not name_ok(path, roots):
        return False
    if st is None:
        try:
            st = os.stat(path)
        except OSError:
            return False
    return not getattr(st, "st_file_attributes", 0) & ATTR_IGNORED
