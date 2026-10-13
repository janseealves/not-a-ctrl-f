"""Base dos extratores: tipos comuns, limpeza/chunking de texto e peça 'name'."""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
MAX_CHUNKS = 2000


class SkipFile(Exception):
    """O arquivo não entra no índice; str(e) é o motivo curto."""


class ExtractError(Exception):
    """Tipo suportado, mas o arquivo não pôde ser lido (corrompido)."""


@dataclass(slots=True)
class Piece:
    kind: str
    seq: int
    text: str | None = None
    loc: float | None = None
    loc_end: float | None = None
    embed_as: str | None = None  # 'text' | 'image' | 'audio' | None (só palavra-chave)
    media: bytes | None = None  # PNG ou WAV
    budget: int | None = None  # tokens de visão
    title: str | None = None


@dataclass(slots=True)
class Extraction:
    pieces: list[Piece]
    info: dict[str, object]


# ---------- texto ----------

_SPACES = re.compile(r"[ \t\u00a0]+")
_BREAKS = re.compile(r"\n(?: ?\n){2,}")


def _clean(text: str) -> str:
    return _BREAKS.sub("\n\n", _SPACES.sub(" ", text)).strip()


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    limit: int = MAX_CHUNKS,
) -> list[tuple[int, int, str]]:
    """Divide em pedaços com sobreposição; offsets são no texto limpo."""
    clean = _clean(text)
    n = len(clean)
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < n and len(chunks) < limit:
        end = min(start + size, n)
        if end < n:
            # corta no último espaço da metade final do pedaço
            for i in range(end - 1, start + size // 2, -1):
                if clean[i].isspace():
                    end = i
                    break
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end >= n:
            break
        nxt = max(end - overlap, start)
        if nxt <= start:
            nxt = end
        while nxt < end and not clean[nxt].isspace():  # começa em fronteira de palavra
            nxt += 1
        start = nxt if nxt < end else end
    return chunks


# ---------- extrator base ----------

def name_text(path: Path, home: str | Path | None) -> str:
    full = Path(os.path.abspath(path))
    base = Path(os.path.abspath(home if home is not None else Path.home()))
    try:
        folders = full.parent.relative_to(base).parts
    except ValueError:
        folders = full.parent.parts[1:] if full.parent.anchor else full.parent.parts
    return " ".join([full.name, *folders])


class BaseExtractor(ABC):
    extensions: ClassVar[frozenset[str]]  # minúsculas, sem ponto

    def extract(self, path: Path, *, home: str | Path | None = None) -> Extraction:
        """Peça 'name' (seq 0) + as peças do corpo, numeradas a partir de 1."""
        body = self.read(path)
        head = Piece("name", 0, name_text(path, home))
        return Extraction([head, *body.pieces], body.info)

    @abstractmethod
    def read(self, path: Path) -> Extraction:
        """Só o corpo do arquivo: peças com seq a partir de 1."""
