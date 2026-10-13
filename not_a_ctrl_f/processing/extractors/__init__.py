"""Extrai de UM arquivo os pedaços (pieces) que viram linhas de `chunks`.

Não embedda, não acessa o banco: só lê o arquivo e devolve texto/mídia.
Erros de SO (permissão, arquivo travado) propagam para o indexador decidir.
"""

from pathlib import Path

from .audio import AUDIO_HOP, AUDIO_WINDOW, MAX_AUDIO_WINDOWS, AudioExtractor
from .base import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNKS,
    BaseExtractor,
    ExtractError,
    Extraction,
    Piece,
    SkipFile,
    _clean,  # privado, mas os testes o acessam como `extractors._clean`
    chunk_text,
)
from .document import (
    MAX_PAGES,
    MAX_TEXT_BYTES,
    MIN_PAGE_TEXT,
    PAGE_BUDGET,
    PAGE_RENDER_SIDE,
    PAGE_TEXT_SPLIT,
    DocumentExtractor,
)
from .image import MAX_IMAGE_SIDE, MIN_IMAGE_SIDE, PHOTO_BUDGET, ImageExtractor

EXTRACTORS: tuple[BaseExtractor, ...] = (
    DocumentExtractor(),
    ImageExtractor(),
    AudioExtractor(),
)

_BY_EXT: dict[str, BaseExtractor] = {}
for _ex in EXTRACTORS:
    for _ext in _ex.extensions:
        if _ext in _BY_EXT:
            raise RuntimeError(f"extensão .{_ext} pertence a mais de um extrator")
        _BY_EXT[_ext] = _ex

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_BY_EXT)


def _ext_of(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def extractor_for(path: str | Path) -> BaseExtractor | None:
    return _BY_EXT.get(_ext_of(Path(path)))


def extract(path: str | Path, *, home: str | Path | None = None) -> Extraction:
    p = Path(path)
    extractor = extractor_for(p)
    if extractor is None:
        raise SkipFile(f"tipo não suportado: .{_ext_of(p)}")
    return extractor.extract(p, home=home)
