"""Busca híbrida: palavra-chave (FTS5/bm25) + semântica (vetores), fundidas por RRF, uma linha por arquivo."""

import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from not_a_ctrl_f.processing.extractors import AudioExtractor, DocumentExtractor, ImageExtractor
from not_a_ctrl_f.shared import db
from not_a_ctrl_f.shared.config import EMBED_DIM
from not_a_ctrl_f.shared.embedder import EmbedderError

# marcadores de destaque no snippet (a API troca por <mark>); caracteres da área privada U+E000/U+E001
MARK_START = ""
MARK_END = ""

TYPE_EXTS: dict[str, frozenset[str]] = {
    "doc": frozenset(DocumentExtractor.extensions),
    "image": frozenset(ImageExtractor.extensions),
    "audio": frozenset(AudioExtractor.extensions),
}


def type_of(ext: str) -> str | None:
    for name, exts in TYPE_EXTS.items():
        if ext in exts:
            return name
    return None


@dataclass(slots=True)
class Hit:
    file_id: int
    path: str
    name: str
    ext: str
    state: str
    score: float
    kind: str | None = None  # kind do melhor chunk
    loc: float | None = None
    loc_end: float | None = None
    snippet: str | None = None  # texto com MARK_START/MARK_END em volta dos termos
    matched: tuple[str, ...] = ()  # subconjunto de ("keyword", "semantic")


@dataclass(slots=True)
class SearchResult:
    hits: list[Hit] = field(default_factory=list)
    semantic: bool = False  # a busca vetorial foi usada
    error: str | None = None  # por que a busca vetorial não foi usada (embedder falhou)
    took_ms: float = 0.0


_STATES = ("pending", "indexing", "done", "skipped", "failed")
_BLOCK = 16384  # linhas convertidas para float32 por vez
_FTS_LIMIT = 500
_META_BATCH = 200


def _fts_queries(query: str) -> list[str]:
    """Consultas FTS5 seguras (AND e, havendo mais de um termo, OR); vazio se não há termos."""
    tokens = re.findall(r"\w+", query)
    if not tokens:
        return []
    terms = [f'"{t}"' for t in tokens]
    if len(tokens[-1]) >= 2:
        terms[-1] += "*"
    queries = [" ".join(terms)]
    if len(terms) > 1:
        queries.append(" OR ".join(terms))
    return queries


def _ext_filter(type: str | None) -> tuple[str, list[str]]:
    if type is None:
        return "", []
    exts = sorted(TYPE_EXTS[type])
    return f" AND f.ext IN ({','.join('?' * len(exts))})", exts


class _VectorIndex:
    """Vetores de todos os chunks em memória, atualizados de forma incremental por indexed_at."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._version: int | None = None
        # file_id -> (indexed_at, chunk_ids int64 (n,), matriz float16 (n, dim))
        self._files: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
        self._mat = np.empty((0, EMBED_DIM), np.float16)
        self._chunk_ids = np.empty(0, np.int64)
        self._file_ids = np.empty(0, np.int64)
        self._starts = np.empty(0, np.int64)  # início de cada arquivo em _mat

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                self._version = None

    def _refresh(self) -> None:
        conn = self._connect()
        version = conn.execute("PRAGMA data_version").fetchone()[0]
        if version == self._version:
            return
        changed = False
        conn.execute("BEGIN")
        try:
            rows = conn.execute(
                "SELECT id, indexed_at FROM files WHERE indexed_at IS NOT NULL"
            ).fetchall()
            live = {r["id"] for r in rows}
            for fid in [f for f in self._files if f not in live]:
                del self._files[fid]
                changed = True
            for r in rows:
                cached = self._files.get(r["id"])
                if cached is not None and cached[0] == r["indexed_at"]:
                    continue
                chunks = conn.execute(
                    "SELECT id, vec FROM chunks WHERE file_id=? AND vec IS NOT NULL ORDER BY seq",
                    (r["id"],),
                ).fetchall()
                if chunks:
                    mat = np.frombuffer(b"".join(c["vec"] for c in chunks), "<f2")
                    mat = mat.reshape(-1, EMBED_DIM)
                    ids = np.array([c["id"] for c in chunks], np.int64)
                else:
                    mat = np.empty((0, EMBED_DIM), np.float16)
                    ids = np.empty(0, np.int64)
                self._files[r["id"]] = (r["indexed_at"], ids, mat)
                changed = True
        finally:
            conn.execute("COMMIT")
        self._version = version
        if changed:
            self._rebuild()

    def _rebuild(self) -> None:
        items = [(fid, e) for fid, e in self._files.items() if len(e[1])]
        if not items:
            self._mat = np.empty((0, EMBED_DIM), np.float16)
            self._chunk_ids = np.empty(0, np.int64)
            self._file_ids = np.empty(0, np.int64)
            self._starts = np.empty(0, np.int64)
            return
        counts = np.array([len(e[1]) for _, e in items], np.int64)
        self._file_ids = np.array([fid for fid, _ in items], np.int64)
        self._starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
        self._chunk_ids = np.concatenate([e[1] for _, e in items])
        self._mat = np.concatenate([e[2] for _, e in items])

    def rank(self, q: np.ndarray):
        """Arquivos por similaridade decrescente: (lista de (file_id, pos), scores, pos -> chunk_id)."""
        with self._lock:
            self._refresh()
            mat, chunk_ids, starts, file_ids = (
                self._mat, self._chunk_ids, self._starts, self._file_ids)
            n = len(mat)
            if n == 0:
                return [], [], None
            q = np.asarray(q, np.float32)
            scores = np.empty(n, np.float32)
            for i in range(0, n, _BLOCK):
                scores[i:i + _BLOCK] = mat[i:i + _BLOCK].astype(np.float32) @ q
            fscores = np.maximum.reduceat(scores, starts)
            order = np.argsort(-fscores, kind="stable")
        ends = np.append(starts[1:], n)

        def best_chunk(pos: int) -> int:
            a, b = starts[pos], ends[pos]
            return int(chunk_ids[a + int(np.argmax(scores[a:b]))])

        return ([(int(file_ids[o]), int(o)) for o in order],
                [float(fscores[o]) for o in order], best_chunk)


class Searcher:
    def __init__(self, db_path: str | Path, embedder=None, *, max_vector_files: int = 200,
                 rrf_k: int = 60) -> None:
        self._db_path = Path(db_path)
        self._embedder = embedder
        self._max_vector_files = max_vector_files
        self._rrf_k = rrf_k
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._vectors = _VectorIndex(db_path)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = db.connect(self._db_path)
            self._local.conn = conn
            with self._conns_lock:
                self._conns.append(conn)
        return conn

    def search(self, query: str, *, limit: int = 30, type: str | None = None,
               keyword_only: bool = False) -> SearchResult:
        if type is not None and type not in TYPE_EXTS:
            raise ValueError(f"tipo inválido: {type!r}")
        t0 = time.perf_counter()
        result = SearchResult()
        if not _fts_queries(query):
            result.took_ms = (time.perf_counter() - t0) * 1000
            return result
        conn = self._conn()

        kw = self._keyword(conn, query, type)  # file_id -> Row, na ordem do ranking
        sem: dict[int, tuple[float, int]] = {}  # file_id -> (score, chunk_id)
        meta: dict[int, sqlite3.Row] = {}
        if self._embedder is not None and not keyword_only:
            try:
                q = self._embedder.embed_query(query)
            except EmbedderError as e:
                result.error = str(e)
            else:
                result.semantic = True
                sem, meta = self._semantic(conn, q, type)

        scores: dict[int, float] = {}
        for ranking in (list(kw), list(sem)):
            for rank, fid in enumerate(ranking, 1):
                scores[fid] = scores.get(fid, 0.0) + 1.0 / (self._rrf_k + rank)

        meta.update(self._meta(conn, [f for f in scores if f not in meta]))
        ordered = sorted((f for f in scores if f in meta),
                         key=lambda f: (-scores[f], meta[f]["name"]))[:limit]

        # chunks semânticos só para quem não veio da palavra-chave
        need = [sem[f][1] for f in ordered if f not in kw and f in sem]
        chunk_rows: dict[int, sqlite3.Row] = {}
        if need:
            chunk_rows = {
                r["id"]: r for r in conn.execute(
                    "SELECT id, kind, loc, loc_end, text FROM chunks "
                    f"WHERE id IN ({','.join('?' * len(need))})", need,
                )
            }

        for fid in ordered:
            m = meta[fid]
            hit = Hit(fid, m["path"], m["name"], m["ext"], m["state"], scores[fid])
            matched = []
            if fid in kw:
                r = kw[fid]
                hit.kind, hit.loc, hit.loc_end, hit.snippet = (
                    r["kind"], r["loc"], r["loc_end"], r["snip"])
                matched.append("keyword")
            if fid in sem:
                matched.append("semantic")
                c = chunk_rows.get(sem[fid][1])
                if fid not in kw and c is not None:
                    hit.kind, hit.loc, hit.loc_end = c["kind"], c["loc"], c["loc_end"]
                    if c["text"] is not None:
                        hit.snippet = " ".join(c["text"].split())[:240]
            hit.matched = tuple(matched)
            result.hits.append(hit)
        result.took_ms = (time.perf_counter() - t0) * 1000
        return result

    def _keyword(self, conn, query: str, type: str | None) -> dict[int, sqlite3.Row]:
        flt, params = _ext_filter(type)
        sql = (
            "SELECT c.id, c.file_id, c.kind, c.loc, c.loc_end, bm25(chunks_fts) AS r, "
            f"snippet(chunks_fts, 0, '{MARK_START}', '{MARK_END}', '…', 16) AS snip "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "JOIN files f ON f.id = c.file_id "
            f"WHERE chunks_fts MATCH ?{flt} ORDER BY r LIMIT {_FTS_LIMIT}"
        )
        for fts in _fts_queries(query):
            rows = conn.execute(sql, [fts, *params]).fetchall()
            if rows:
                best: dict[int, sqlite3.Row] = {}
                for r in rows:
                    best.setdefault(r["file_id"], r)  # já ordenado por bm25
                return best
        return {}

    def _semantic(self, conn, q, type: str | None):
        order, fscores, best_chunk = self._vectors.rank(q)
        flt, params = _ext_filter(type)
        sem: dict[int, tuple[float, int]] = {}
        meta: dict[int, sqlite3.Row] = {}
        for i in range(0, len(order), _META_BATCH):
            batch = order[i:i + _META_BATCH]
            ids = [fid for fid, _ in batch]
            found = {
                r["id"]: r for r in conn.execute(
                    "SELECT f.id, f.path, f.name, f.ext, f.state FROM files f "
                    f"WHERE f.id IN ({','.join('?' * len(ids))}){flt}", [*ids, *params],
                )
            }
            for j, (fid, pos) in enumerate(batch):
                if fid in found:
                    sem[fid] = (fscores[i + j], best_chunk(pos))
                    meta[fid] = found[fid]
                    if len(sem) >= self._max_vector_files:
                        return sem, meta
        return sem, meta

    @staticmethod
    def _meta(conn, ids: list[int]) -> dict[int, sqlite3.Row]:
        out: dict[int, sqlite3.Row] = {}
        for i in range(0, len(ids), _META_BATCH):
            batch = ids[i:i + _META_BATCH]
            for r in conn.execute(
                "SELECT id, path, name, ext, state FROM files "
                f"WHERE id IN ({','.join('?' * len(batch))})", batch,
            ):
                out[r["id"]] = r
        return out

    def warm(self) -> None:
        """Carrega o cache vetorial sem buscar nada."""
        with self._vectors._lock:
            self._vectors._refresh()

    def close(self) -> None:
        self._vectors.close()
        with self._conns_lock:
            conns, self._conns = self._conns, []
        for c in conns:
            try:
                c.close()
            except sqlite3.Error:
                pass  # conexão de outra thread: o GC fecha
        self._local = threading.local()


def index_status(conn) -> dict:
    by_state = dict.fromkeys(_STATES, 0)
    for state, n in conn.execute("SELECT state, count(*) FROM files GROUP BY state"):
        by_state[state] = n
    return {
        "files": sum(by_state.values()),
        "by_state": by_state,
        "chunks": conn.execute("SELECT count(*) FROM chunks").fetchone()[0],
    }
