import base64
import hashlib
import io
import json
import socket
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from not_a_ctrl_f.shared import config, embedder
from not_a_ctrl_f.shared.embedder import (
    Embedder,
    EmbedderError,
    LlamaServer,
    build_server_args,
    document_prompt,
    from_blob,
    image_target_size,
    prepare_image,
    query_prompt,
    to_blob,
)


def _vec(key: str) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return (np.random.default_rng(seed).normal(size=768) * 5 + 1).tolist()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive

    def setup(self):
        super().setup()
        self.server.connections += 1

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        srv = self.server
        srv.bodies.append(body)
        keys = [x if isinstance(x, str) else json.dumps(x) for x in body["input"]]
        if any("FALHA" in k for k in keys):
            out = json.dumps({"error": {"message": "input recusado"}}).encode()
            self.send_response(500)
        else:
            data = [{"index": i, "embedding": _vec(k)} for i, k in enumerate(keys)]
            if srv.reverse:
                data.reverse()
            out = json.dumps({"data": data}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def fake():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.bodies = []
    srv.connections = 0
    srv.reverse = False
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _png(w, h, mode="RGBA"):
    buf = io.BytesIO()
    Image.new(mode, (w, h), (200, 30, 30, 255)[: len(mode)]).save(buf, format="PNG")
    return buf.getvalue()


def _wav(seconds=1.0, rate=16000):
    t = np.arange(int(seconds * rate)) / rate
    pcm = (np.sin(2 * np.pi * 440 * t) * 12000).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _unit(key: str) -> np.ndarray:
    v = np.asarray(_vec(key), dtype=np.float32)
    return v / np.linalg.norm(v)


def test_reuso_de_conexao(fake):
    with Embedder(fake.url) as emb:
        for i in range(20):
            emb.embed_query(f"q{i}")
    assert len(fake.bodies) == 20
    assert fake.connections <= 2


# 1
def test_prompts():
    assert query_prompt("x") == "task: search result | query: x"
    assert document_prompt("t", None) == "title: none | text: t"
    assert document_prompt("t", "tit") == "title: tit | text: t"


# 2
def test_image_target_size():
    assert image_target_size(1000, 1000, 280) == (768, 768)
    # factor = sqrt(1260*256/307200) ~ 1.0247 -> 491.9/48 = 10 ; 655.8/48 = 13
    assert image_target_size(480, 640, 140) == (480, 624)
    for h, w, b in [(480, 640, 140), (1000, 1000, 280), (333, 777, 70), (4000, 3000, 1120)]:
        th, tw = image_target_size(h, w, b)
        assert th % 48 == 0 and tw % 48 == 0
        assert (th // 16) * (tw // 16) <= b * 9


def test_image_target_size_lado_zero():
    # th == 0: altura vira 48 e largura é limitada por max_side (630//9*48 = 3360)
    assert image_target_size(10, 10000, 70) == (48, 3360)
    assert image_target_size(10000, 10, 70) == (3360, 48)
    assert image_target_size(0, 10, 70) is None


def test_image_target_size_minuscula():
    # o algoritmo da referência amplia imagens pequenas até o orçamento (não devolve None)
    th, tw = image_target_size(1, 1, 70)
    assert th > 0 and th % 48 == 0 and tw % 48 == 0


# 3
def test_prepare_image():
    out = prepare_image(_png(1000, 800), 140)
    img = Image.open(io.BytesIO(out))
    assert img.format == "PNG" and img.mode == "RGB"
    th, tw = image_target_size(800, 1000, 140)
    assert img.size == (tw, th)
    with pytest.raises(ValueError):
        prepare_image(_png(10, 10), 100)
    with pytest.raises(EmbedderError):
        prepare_image(b"lixo", 140)


# 4
def test_blob_roundtrip():
    v = np.random.default_rng(1).normal(size=768).astype(np.float32)
    v /= np.linalg.norm(v)
    blob = to_blob(v)
    assert len(blob) == 1536
    back = from_blob(blob)
    assert back.dtype == np.float32 and back.shape == (768,)
    assert np.abs(back - v).max() < 1e-3
    with pytest.raises(ValueError):
        to_blob(np.zeros(100, dtype=np.float32))


def test_fingerprint():
    fp = embedder.fingerprint()
    assert config.MODEL_REVISION[:7] in fp and config.LLAMA_CPP_TAG in fp and "prompts1" in fp
    assert embedder.fingerprint(True) != fp


# 5
def test_embed_documents_lotes_e_ordem(fake):
    fake.reverse = True
    emb = Embedder(fake.url)
    items = [(f"texto {i}", None if i % 2 else f"t{i}") for i in range(20)]
    res = emb.embed_documents(items)
    assert [len(b["input"]) for b in fake.bodies] == [16, 4]
    assert len(res) == 20
    for i, r in enumerate(res):
        assert r.error is None
        assert abs(np.linalg.norm(r.vector) - 1) < 1e-5
        assert np.allclose(r.vector, _unit(document_prompt(*items[i])), atol=1e-5)
    assert fake.bodies[0]["input"][0] == "title: t0 | text: texto 0"
    assert fake.bodies[0]["input"][1] == "title: none | text: texto 1"


# 6
def test_item_com_falha_isolado(fake):
    emb = Embedder(fake.url)
    items = [("a", None), ("b", None), ("FALHA", None), ("d", None), ("e", None)]
    res = emb.embed_documents(items)
    assert res[2].vector is None and res[2].error and len(res[2].error) <= 300
    assert all(r.vector is not None and r.error is None for i, r in enumerate(res) if i != 2)


# 7
def test_embed_images_e_audio(fake):
    emb = Embedder(fake.url)
    res = emb.embed_images([_png(1000, 800), b"nao e imagem"], budget=140)
    assert res[0].vector is not None
    assert res[1].vector is None and res[1].error
    assert len(fake.bodies) == 1 and len(fake.bodies[0]["input"]) == 1  # a ruim não vai ao servidor
    url = fake.bodies[0]["input"][0]["content"][0]["image_url"]["url"]
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    img = Image.open(io.BytesIO(base64.b64decode(url[len(prefix):])))
    th, tw = image_target_size(800, 1000, 140)
    assert img.size == (tw, th)

    fake.bodies.clear()
    res = emb.embed_audio([_wav(), _wav(0.5), _wav(0.7)])
    assert [len(b["input"]) for b in fake.bodies] == [2, 1]
    part = fake.bodies[0]["input"][0]["content"][0]
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "wav"
    assert all(r.vector is not None for r in res)


# 8
def test_text_only_recusa_midia(fake):
    emb = Embedder(fake.url, text_only=True)
    with pytest.raises(EmbedderError):
        emb.embed_images([_png(10, 10)])
    with pytest.raises(EmbedderError):
        emb.embed_audio([_wav()])


# 9
def test_servidor_desligado():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(EmbedderError):
        Embedder(f"http://127.0.0.1:{port}", timeout=2).embed_query("oi")


# 10
def test_build_server_args(tmp_path):
    full = build_server_args(
        tmp_path / "s.exe", tmp_path, port=1234, text_only=False, gpu_layers=999, threads=None
    )
    assert "--mmproj" in full and "--embeddings" in full
    assert full[full.index("--pooling") + 1] == "mean"
    assert full[full.index("--image-max-tokens") + 1] == "1120"
    assert "-t" not in full
    txt = build_server_args(
        tmp_path / "s.exe", tmp_path, port=1234, text_only=True, gpu_layers=0, threads=4
    )
    assert "--mmproj" not in txt and not any(a.startswith("--image") for a in txt)
    assert txt[txt.index("-t") + 1] == "4"


# 11
def test_start_sem_binario(tmp_path):
    missing = tmp_path / "nao-existe" / "llama-server.exe"
    with pytest.raises(EmbedderError, match="nao-existe"):
        LlamaServer(server_path=missing, models_dir=tmp_path).start()


# 12
def test_concorrencia(fake):
    emb = Embedder(fake.url)
    results = {}

    def work(i):
        results[i] = emb.embed_query(f"consulta {i}")

    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for i in range(4):
        assert np.allclose(results[i], _unit(query_prompt(f"consulta {i}")), atol=1e-5)


# 13
_REAL = (
    config.llama_server_path().exists()
    and (config.models_dir() / config.TEXT_MODEL_FILE).exists()
    and (config.models_dir() / config.MEDIA_MODEL_FILE).exists()
)


@pytest.mark.integration
@pytest.mark.skipif(not _REAL, reason="llama-server/modelos não instalados")
def test_integracao_real():
    with LlamaServer() as srv:
        emb = Embedder(srv.base_url)
        q = emb.embed_query("cachorro latindo")
        d1, d2 = emb.embed_documents(
            [("o cão late alto no quintal", None), ("nota fiscal de energia elétrica", None)]
        )
        assert float(q @ d1.vector) > float(q @ d2.vector)
        (im,) = emb.embed_images([_png(512, 512, "RGB")])
        (au,) = emb.embed_audio([_wav(3.0)])
        for r in (im, au):
            assert r.vector.shape == (768,) and abs(np.linalg.norm(r.vector) - 1) < 1e-3


def _pid_alive(pid: int) -> bool:
    import ctypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = ctypes.c_uint32()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        return code.value == 259  # STILL_ACTIVE
    finally:
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        k32.CloseHandle(h)


@pytest.mark.skipif(sys.platform != "win32", reason="Job Object é do Windows")
def test_kill_on_close_job_mata_filho_quando_o_pai_morre():
    import subprocess
    import time

    script = (
        "import os, subprocess, sys\n"
        "from not_a_ctrl_f.shared.embedder import _kill_on_close_job\n"
        "d = subprocess.DEVNULL\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
        " stdin=d, stdout=d, stderr=d)\n"
        "assert _kill_on_close_job(p) is not None\n"
        "print(p.pid, flush=True)\n"
        "os._exit(0)\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    pid = int(out.stdout.strip())
    deadline = time.monotonic() + 5
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    alive = _pid_alive(pid)
    if alive:  # não deixa lixo
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    assert not alive
