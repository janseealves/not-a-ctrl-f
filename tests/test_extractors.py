import io
import zipfile
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
import soundfile as sf
from PIL import Image

from not_a_ctrl_f.processing import extractors as ex
from not_a_ctrl_f.processing.extractors import (
    ExtractError,
    SkipFile,
    chunk_text,
    extract,
)


def run(path: Path, home: Path):
    r = extract(path, home=home)
    assert [p.seq for p in r.pieces] == list(range(len(r.pieces)))
    return r


# ---------- chunk_text ----------

def test_chunk_curto():
    assert chunk_text("olá mundo") == [(0, 9, "olá mundo")]


def test_chunk_longo():
    words = [f"palavra{i}" for i in range(2000)]
    text = " ".join(words)
    assert len(text) > 10_000
    clean = ex._clean(text)
    chunks = chunk_text(text, size=500, overlap=100)
    assert len(chunks) > 5
    for (s, e, c), nxt in zip(chunks, chunks[1:] + [None]):
        assert len(c) <= 500
        assert clean[s:e].strip() == c
        assert s == 0 or clean[s - 1].isspace() or clean[s].isspace()
        assert e == len(clean) or clean[e].isspace()
        if nxt:
            assert nxt[0] < e  # sobreposição
            assert nxt[0] > s
    assert chunks[0][0] == 0 and chunks[-1][1] == len(clean)


def test_chunk_palavra_gigante():
    text = "x" * 5000
    chunks = chunk_text(text)
    assert chunks
    assert chunks[0][0] == 0 and chunks[-1][1] == 5000


def test_chunk_limit():
    text = " ".join(["abc"] * 5000)
    assert len(chunk_text(text, size=100, overlap=10, limit=3)) == 3


# ---------- name ----------

def test_name_piece(tmp_path):
    d = tmp_path / "Documents" / "Contratos"
    d.mkdir(parents=True)
    f = d / "aluguel_2024.txt"
    f.write_text("conteúdo")
    r = run(f, tmp_path)
    p = r.pieces[0]
    assert (p.kind, p.seq, p.embed_as) == ("name", 0, None)
    assert p.text == "aluguel_2024.txt Documents Contratos"


def test_name_fora_do_home(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("oi")
    home = tmp_path / "outra"
    home.mkdir()
    text = extract(f, home=home).pieces[0].text
    assert text.startswith("a.txt ")
    assert ":" not in text


# ---------- texto ----------

def test_txt_bom_e_cp1252(tmp_path):
    a = tmp_path / "a.txt"
    a.write_bytes("relatório ação".encode("utf-8-sig"))
    b = tmp_path / "b.txt"
    b.write_bytes("relatório ação".encode("cp1252"))
    for f in (a, b):
        r = run(f, tmp_path)
        assert r.pieces[1].text == "relatório ação"
        assert r.pieces[1].kind == "text" and r.pieces[1].embed_as == "text"
        assert r.pieces[1].title == f.name
        assert r.info == {"chars": 14, "chunks": 1}


def test_txt_binario_e_vazio(tmp_path):
    a = tmp_path / "a.txt"
    a.write_bytes(b"abc\x00def")
    with pytest.raises(SkipFile, match="binário"):
        extract(a, home=tmp_path)
    b = tmp_path / "b.txt"
    b.write_text("  \n ")
    with pytest.raises(SkipFile, match="vazio"):
        extract(b, home=tmp_path)


def test_html(tmp_path):
    f = tmp_path / "p.html"
    f.write_text(
        "<html><head><title>T</title><style>p{}</style></head><body>"
        "<script>var x = 1;</script><!-- nota -->"
        "<p>Tom &amp; Jerry</p><div>caf&#233;</div></body></html>",
        encoding="utf-8",
    )
    r = run(f, tmp_path)
    text = r.pieces[1].text
    assert [ln for ln in text.split("\n") if ln] == ["Tom & Jerry", "café"]
    assert "var" not in text and "nota" not in text


def test_html_malformado(tmp_path):
    f = tmp_path / "m.html"
    f.write_text("<p>a<p>b<div>c", encoding="utf-8")
    text = run(f, tmp_path).pieces[1].text
    assert [ln for ln in text.split("\n") if ln] == ["a", "b", "c"]


def test_html_inline_na_mesma_linha(tmp_path):
    f = tmp_path / "i.html"
    f.write_text(
        "<p>Tom <b>&amp;</b> <i>Jerry</i> e <span>Spike</span> <a href='x'>link</a></p>",
        encoding="utf-8",
    )
    text = run(f, tmp_path).pieces[1].text
    assert [ln for ln in text.split("\n") if ln] == ["Tom & Jerry e Spike link"]


# ---------- docx / odt ----------

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
T = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"


def test_docx(tmp_path):
    f = tmp_path / "d.docx"
    xml = (
        f'<w:document xmlns:w="{W}"><w:body>'
        "<w:p><w:r><w:t>primeiro</w:t><w:tab/><w:t>trecho</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>segundo</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", xml)
    r = run(f, tmp_path)
    assert r.pieces[1].text == "primeiro trecho\nsegundo"  # tab vira espaço na limpeza


def test_odt(tmp_path):
    f = tmp_path / "d.odt"
    xml = (
        f'<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        f'xmlns:text="{T}"><office:body><office:text>'
        '<text:p>a<text:span>b</text:span><text:s text:c="3"/>c</text:p>'
        "<text:h>título</text:h>"
        "</office:text></office:body></office:document-content>"
    )
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("content.xml", xml)
    r = run(f, tmp_path)
    assert r.pieces[1].text == "ab c\ntítulo"


def test_docx_invalido(tmp_path):
    a = tmp_path / "a.docx"
    a.write_bytes(b"isto nao e zip")
    with pytest.raises(ExtractError):
        extract(a, home=tmp_path)
    b = tmp_path / "b.docx"
    b.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 100)
    with pytest.raises(SkipFile, match="senha"):
        extract(b, home=tmp_path)


# ---------- pdf ----------

def make_text_pdf(path: Path, text: str) -> None:
    lines = [text[i : i + 60] for i in range(0, len(text), 60)]
    ops = "BT /F1 12 Tf 14 TL 20 750 Td " + " T* ".join(f"({ln}) Tj" for ln in lines) + " ET"
    stream = ops.encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path.write_bytes(bytes(out))


def test_pdf_scan(tmp_path):
    f = tmp_path / "scan.pdf"
    Image.new("RGB", (400, 300), "white").save(f, "PDF")
    r = run(f, tmp_path)
    assert r.info == {"pages": 1}
    p = r.pieces[1]
    assert (p.kind, p.embed_as, p.budget, p.loc, p.loc_end) == ("page_image", "image", 280, 1, 1)
    with Image.open(io.BytesIO(p.media)) as im:
        assert im.format == "PNG"
        assert abs(max(im.size) - ex.PAGE_RENDER_SIDE) <= 4


def test_pdf_texto(tmp_path):
    f = tmp_path / "t.pdf"
    make_text_pdf(f, "A raposa marrom pula sobre o cachorro preguiçoso. " * 8)
    r = run(f, tmp_path)
    assert r.info == {"pages": 1}
    p = r.pieces[1]
    assert (p.kind, p.embed_as, p.loc) == ("page", "text", 1)
    assert "raposa marrom pula" in p.text
    assert p.title == "t.pdf"


def test_pdf_invalido(tmp_path):
    f = tmp_path / "x.pdf"
    f.write_bytes(b"nao e pdf")
    with pytest.raises(ExtractError):
        extract(f, home=tmp_path)


# ---------- imagem ----------

def test_imagem_pequena(tmp_path):
    f = tmp_path / "p.png"
    Image.new("RGB", (100, 100)).save(f)
    with pytest.raises(SkipFile, match="pequena"):
        extract(f, home=tmp_path)


def test_imagem_rgba_grande(tmp_path):
    f = tmp_path / "g.png"
    Image.new("RGBA", (3000, 1000), (255, 0, 0, 0)).save(f)
    r = run(f, tmp_path)
    p = r.pieces[1]
    assert (p.kind, p.embed_as, p.budget, p.text) == ("image", "image", 140, None)
    with Image.open(io.BytesIO(p.media)) as im:
        assert im.format == "PNG" and im.mode == "RGB"
        assert max(im.size) == 2048
        assert im.getpixel((0, 0)) == (255, 255, 255)  # alpha sobre branco
    assert r.info["width"] == 3000 and r.info["height"] == 1000


def test_imagem_exif_rotacao(tmp_path):
    f = tmp_path / "r.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (600, 300), "blue").save(f, exif=exif)
    r = run(f, tmp_path)
    assert r.info["width"] == 300 and r.info["height"] == 600


def test_imagem_corrompida(tmp_path):
    f = tmp_path / "c.png"
    f.write_bytes(b"lixo")
    with pytest.raises(ExtractError):
        extract(f, home=tmp_path)


# ---------- áudio ----------

def make_wav(path: Path, seconds: float, sr: int = 8000, channels: int = 2) -> None:
    t = np.arange(int(seconds * sr)) / sr
    mono = 0.3 * np.sin(2 * np.pi * 440 * t)
    data = np.stack([mono] * channels, axis=1)
    sf.write(path, data, sr)


def test_audio_70s(tmp_path):
    f = tmp_path / "a.wav"
    make_wav(f, 70)
    r = run(f, tmp_path)
    assert r.info["duration"] == pytest.approx(70)
    assert r.info["sample_rate"] == 8000 and r.info["channels"] == 2
    wins = [(p.loc, p.loc_end) for p in r.pieces[1:]]
    assert wins == [(0, 30), (25, 55), (50, 70)]
    for p in r.pieces[1:]:
        assert (p.kind, p.embed_as, p.text) == ("audio", "audio", None)
        data, sr = sf.read(io.BytesIO(p.media))
        assert data.ndim == 1 and sr == 8000
        assert len(data) / sr == pytest.approx(p.loc_end - p.loc, abs=0.01)


def test_audio_10s(tmp_path):
    f = tmp_path / "a.wav"
    make_wav(f, 10, channels=1)
    r = run(f, tmp_path)
    assert [(p.loc, p.loc_end) for p in r.pieces[1:]] == [(0, 10)]


# ---------- geral ----------

def test_extensao_nao_suportada(tmp_path):
    f = tmp_path / "a.exe"
    f.write_bytes(b"MZ")
    with pytest.raises(SkipFile, match="não suportado"):
        extract(f, home=tmp_path)


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract(tmp_path / "nada.txt", home=tmp_path)


def test_sem_handle_vazado(tmp_path):
    arquivos = []
    f = tmp_path / "scan.pdf"
    Image.new("RGB", (400, 300), "white").save(f, "PDF")
    arquivos.append(f)
    g = tmp_path / "a.wav"
    make_wav(g, 5)
    arquivos.append(g)
    h = tmp_path / "i.png"
    Image.new("RGB", (300, 300)).save(h)
    arquivos.append(h)
    for a in arquivos:
        extract(a, home=tmp_path)
        a.unlink()  # no Windows falha se o handle ficou aberto


# ---------- registro de extratores ----------

def test_extractor_for():
    assert isinstance(ex.extractor_for("a.PDF"), ex.DocumentExtractor)
    assert isinstance(ex.extractor_for("a.jpg"), ex.ImageExtractor)
    assert isinstance(ex.extractor_for("a.flac"), ex.AudioExtractor)
    assert ex.extractor_for("a.exe") is None


def test_supported_extensions_uniao_sem_intersecao():
    sets = [e.extensions for e in ex.EXTRACTORS]
    assert len(sets) == 3
    assert ex.SUPPORTED_EXTENSIONS == frozenset().union(*sets)
    assert sum(len(s) for s in sets) == len(ex.SUPPORTED_EXTENSIONS)


def test_base_extractor_abstrata():
    with pytest.raises(TypeError):
        ex.BaseExtractor()
