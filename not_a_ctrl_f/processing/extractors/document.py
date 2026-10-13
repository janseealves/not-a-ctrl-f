"""Extrator de documentos: texto puro, HTML, DOCX, ODT e PDF."""

import io
import zipfile
import zlib
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

import pypdfium2 as pdfium

from .base import (
    BaseExtractor,
    ExtractError,
    Extraction,
    Piece,
    SkipFile,
    _clean,
    chunk_text,
)

MAX_PAGES = 500
MIN_PAGE_TEXT = 200
PAGE_TEXT_SPLIT = 4000
PAGE_BUDGET = 280
MAX_TEXT_BYTES = 20 * 1024 * 1024

# PAGE_BUDGET (280 tokens) usa ~912 px de altura numa A4: renderizar maior só gasta memória.
PAGE_RENDER_SIDE = 1024
OLE_MAGIC = b"\xd0\xcf\x11\xe0"

TEXT_EXTS = frozenset("txt md markdown csv tsv log json yaml yml ini toml rst tex xml".split())
HTML_EXTS = frozenset({"html", "htm"})

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_T = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"


def _text_pieces(text: str, title: str) -> Extraction:
    clean = _clean(text)
    if not clean:
        raise SkipFile("vazio")
    chunks = chunk_text(clean)
    pieces = [
        Piece("text", 1 + i, chunk, start, end, "text", title=title)
        for i, (start, end, chunk) in enumerate(chunks)
    ]
    return Extraction(pieces, {"chars": len(clean), "chunks": len(chunks)})


def _decode(data: bytes) -> str:
    if b"\x00" in data[:8192]:
        raise SkipFile("binário")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_TEXT_BYTES:
        raise SkipFile("arquivo de texto grande demais")
    return _decode(path.read_bytes())


class _HtmlText(HTMLParser):
    _BLOCK = frozenset("p div br li h1 h2 h3 h4 h5 h6 tr section article".split())
    _IGNORED = frozenset({"script", "style", "head"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._depth = 0  # >0 dentro de script/style/head

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._IGNORED:
            self._depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORED:
            self._depth = max(0, self._depth - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._depth:
            self.parts.append(data)


def _html_to_text(html: str) -> str:
    parser = _HtmlText()
    parser.feed(html)
    parser.close()
    return "".join(parser.parts)


# ---------- DOCX / ODT ----------

def _read_zip_member(path: Path, member: str) -> bytes:
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(0)
        try:
            with zipfile.ZipFile(f) as z:
                try:
                    return z.read(member)
                except KeyError:
                    raise ExtractError(f"{member} ausente") from None
        except zipfile.BadZipFile:
            if head == OLE_MAGIC:
                raise SkipFile("protegido por senha") from None
            raise ExtractError("arquivo zip inválido") from None
        except (zlib.error, NotImplementedError, RuntimeError, EOFError) as e:
            raise ExtractError(f"zip ilegível: {e}") from None


def _parse_xml(data: bytes) -> ET.Element:
    try:
        return ET.fromstring(data)
    except ET.ParseError as e:
        raise ExtractError(f"xml inválido: {e}") from None


def _docx_text(path: Path) -> str:
    root = _parse_xml(_read_zip_member(path, "word/document.xml"))
    lines = []
    for p in root.iter(f"{_W}p"):
        buf = []
        for el in p.iter():
            if el.tag == f"{_W}t":
                buf.append(el.text or "")
            elif el.tag == f"{_W}tab":
                buf.append("\t")
            elif el.tag in (f"{_W}br", f"{_W}cr"):
                buf.append("\n")
        lines.append("".join(buf))
    return "\n".join(lines)


def _odt_inline(el: ET.Element, buf: list[str]) -> None:
    buf.append(el.text or "")
    for child in el:
        if child.tag == f"{_T}s":
            buf.append(" " * int(child.get(f"{_T}c", "1")))
        elif child.tag == f"{_T}tab":
            buf.append("\t")
        elif child.tag == f"{_T}line-break":
            buf.append("\n")
        else:
            _odt_inline(child, buf)
        buf.append(child.tail or "")


def _odt_text(path: Path) -> str:
    root = _parse_xml(_read_zip_member(path, "content.xml"))
    lines = []
    for el in root.iter():
        if el.tag in (f"{_T}p", f"{_T}h"):
            buf: list[str] = []
            _odt_inline(el, buf)
            lines.append("".join(buf))
    return "\n".join(lines)


# ---------- PDF ----------

def _extract_pdf(path: Path, title: str) -> Extraction:
    try:
        doc = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as e:
        if "password" in str(e).lower() or getattr(e, "err_code", None) == 4:
            raise SkipFile("protegido por senha") from None
        raise ExtractError(f"PDF ilegível: {e}") from None
    pieces: list[Piece] = []
    seq = 1
    try:
        total = len(doc)
        if total == 0:
            raise SkipFile("vazio")
        for i in range(min(total, MAX_PAGES)):
            num = float(i + 1)
            page = doc[i]
            try:
                try:
                    textpage = page.get_textpage()
                    try:
                        text = _clean(textpage.get_text_bounded())
                    finally:
                        textpage.close()
                    if len(text) >= MIN_PAGE_TEXT:
                        parts = (
                            [text]
                            if len(text) <= PAGE_TEXT_SPLIT
                            else [c for _, _, c in chunk_text(text)]
                        )
                        for part in parts:
                            pieces.append(
                                Piece("page", seq, part, num, num, "text", title=title)
                            )
                            seq += 1
                    else:
                        w, h = page.get_size()
                        scale = PAGE_RENDER_SIDE / max(w, h, 1.0)
                        img = page.render(scale=scale).to_pil().convert("RGB")
                        buf = io.BytesIO()
                        img.save(buf, "PNG")
                        pieces.append(
                            Piece(
                                "page_image", seq, text or None, num, num,
                                "image", buf.getvalue(), PAGE_BUDGET,
                            )
                        )
                        seq += 1
                except pdfium.PdfiumError as e:
                    raise ExtractError(f"página {i + 1} ilegível: {e}") from None
            finally:
                page.close()
    finally:
        doc.close()
    return Extraction(pieces, {"pages": total})


# ---------- extrator ----------

class DocumentExtractor(BaseExtractor):
    extensions = TEXT_EXTS | HTML_EXTS | {"docx", "odt", "pdf"}

    def read(self, path: Path) -> Extraction:
        ext = path.suffix.lower().lstrip(".")
        name = path.name
        if ext in TEXT_EXTS:
            return _text_pieces(_read_text(path), name)
        if ext in HTML_EXTS:
            return _text_pieces(_html_to_text(_read_text(path)), name)
        if ext == "docx":
            return _text_pieces(_docx_text(path), name)
        if ext == "odt":
            return _text_pieces(_odt_text(path), name)
        return _extract_pdf(path, name)
