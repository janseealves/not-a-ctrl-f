"""Indexa UM arquivo: extrai, embedda e grava os chunks. Sem threads.

`build_rows` é a parte lenta e não toca no banco; `write_rows` é uma transação curta
que só grava se o arquivo ainda é o mesmo do momento do claim.
"""

import json
import sqlite3
import time
from pathlib import Path

from not_a_ctrl_f.processing import extractors
from not_a_ctrl_f.shared import db
from not_a_ctrl_f.shared import embedder as embedder_module
from not_a_ctrl_f.shared.embedder import Embedding, EmbedderError

FINGERPRINT_KEY = "embed_fingerprint"
DEFAULT_BUDGET = 140  # tokens de visão quando a peça não informa

_INSERT = (
    "INSERT INTO chunks(file_id, kind, modality, seq, loc, loc_end, text, vec)"
    " VALUES (?,?,?,?,?,?,?,?)"
)


def ensure_fingerprint(conn: sqlite3.Connection, fingerprint: str) -> bool:
    """Embedder mudou: o que já foi indexado (done/failed) volta para a fila. True se reiniciou."""
    with db.transaction(conn):
        current = db.get_meta(conn, FINGERPRINT_KEY)
        if current == fingerprint:
            return False
        if current is not None:
            conn.execute(
                "UPDATE files SET state='pending', attempts=0, error=NULL"
                " WHERE state IN ('done','failed')"
            )
        db.set_meta(conn, FINGERPRINT_KEY, fingerprint)
    return current is not None


def name_row(path: Path, *, home: Path | None = None) -> tuple:
    return ("name", "none", 0, None, None, extractors.name_text(path, home), None)


def _embed_pieces(pieces: list, embedder) -> dict[int, Embedding]:
    """Índice da peça -> Embedding, um lote por tipo (e por orçamento, nas imagens)."""
    text_idx = [i for i, p in enumerate(pieces) if p.embed_as == "text"]
    audio_idx = [i for i, p in enumerate(pieces) if p.embed_as == "audio"]
    by_budget: dict[int, list[int]] = {}
    for i, p in enumerate(pieces):
        if p.embed_as == "image":
            by_budget.setdefault(p.budget or DEFAULT_BUDGET, []).append(i)

    jobs: list[tuple[list[int], list[Embedding]]] = []
    if text_idx:
        res = embedder.embed_documents([(pieces[i].text or "", pieces[i].title) for i in text_idx])
        jobs.append((text_idx, res))
    for budget, idx in by_budget.items():
        jobs.append((idx, embedder.embed_images([pieces[i].media for i in idx], budget=budget)))
    if audio_idx:
        jobs.append((audio_idx, embedder.embed_audio([pieces[i].media for i in audio_idx])))

    out: dict[int, Embedding] = {}
    for idx, res in jobs:
        if len(res) != len(idx):
            raise EmbedderError(f"embedder devolveu {len(res)} vetores para {len(idx)} peças")
        out.update(zip(idx, res))
    return out


def build_rows(path: Path, embedder, *, home: Path | None = None) -> tuple[list[tuple], dict]:
    """Extrai e embedda. Linhas: (kind, modality, seq, loc, loc_end, text, vec_blob_ou_None)."""
    extraction = extractors.extract(path, home=home)
    pieces = extraction.pieces
    embeddings = _embed_pieces(pieces, embedder)

    rows: list[tuple] = []
    errors = 0
    sample: str | None = None
    for i, p in enumerate(pieces):
        modality, blob = "none", None
        if p.embed_as is not None:
            emb = embeddings[i]
            err = emb.error or "sem vetor"
            if emb.vector is not None:
                try:
                    blob, modality = embedder_module.to_blob(emb.vector), p.embed_as
                except ValueError as e:
                    err = str(e)
            if blob is None:
                errors += 1
                sample = sample or err
        rows.append((p.kind, modality, p.seq, p.loc, p.loc_end, p.text, blob))

    info = dict(extraction.info)
    if errors:
        info["embed_errors"] = errors
        info["embed_error_sample"] = sample
    return rows, info


def write_rows(conn: sqlite3.Connection, file_id: int, size: int, mtime_ns: int,
               rows: list[tuple], info: dict, *, state: str = "done",
               error: str | None = None, attempts: int | None = None) -> bool:
    """Substitui os chunks do arquivo, se ele ainda está 'indexing' com o mesmo size/mtime.
    False (sem escrever nada) quando mudou ou sumiu: o watcher cuida do resto."""
    with db.transaction(conn):
        still = conn.execute(
            "SELECT attempts FROM files WHERE id=? AND state='indexing' AND size=? AND mtime_ns=?",
            (file_id, size, mtime_ns),
        ).fetchone()
        if still is None:
            return False
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        conn.executemany(_INSERT, [(file_id, *r) for r in rows])
        conn.execute(
            "UPDATE files SET state=?, error=?, info=?, attempts=?, indexed_at=? WHERE id=?",
            (state, error, json.dumps(info, ensure_ascii=False, default=str),
             still["attempts"] if attempts is None else attempts, time.time(), file_id),
        )
    return True
