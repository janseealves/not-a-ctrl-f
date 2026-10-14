import hashlib
import io
import json
import subprocess
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from not_a_ctrl_f import installer
from not_a_ctrl_f.installer import Asset, InstallError
from not_a_ctrl_f.shared import config


class Srv:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.requests: list[tuple[str, str | None]] = []
        self.cut_once: set[str] = set()
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                srv.requests.append((self.path, rng))
                data = srv.files.get(self.path)
                if data is None:
                    self.send_error(404)
                    return
                start, code = 0, 200
                if rng:
                    start, code = int(rng.split("=")[1].rstrip("-")), 206
                body = data[start:]
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.path in srv.cut_once:
                    srv.cut_once.discard(self.path)
                    self.wfile.write(body[: len(body) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def add(self, path: str, data: bytes) -> Asset:
        self.files["/" + path] = data
        return Asset(path, hashlib.sha256(data).hexdigest(), len(data))

    def count(self) -> int:
        return len(self.requests)


def make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in members.items():
            z.writestr(n, d)
    return buf.getvalue()


@pytest.fixture
def srv():
    s = Srv()
    yield s
    s.httpd.shutdown()
    s.httpd.server_close()


@pytest.fixture(autouse=True)
def limpo(tmp_path, monkeypatch):
    for k in ("DB_PATH", "MODELS_DIR", "LLAMA_SERVER", "WATCH_DIRS"):
        monkeypatch.delenv(f"NACF_{k}", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    return tmp_path


@pytest.fixture
def env(srv, monkeypatch):
    monkeypatch.setattr(installer, "llama_url", lambda n: f"{srv.base}/{n}")
    monkeypatch.setattr(installer, "model_url", lambda n: f"{srv.base}/{n}")
    monkeypatch.setattr(installer, "detect_backend", lambda: "cpu")
    return srv


def set_llama(monkeypatch, srv, **zips):
    """zips: backend -> lista de dicts de membros (um zip por dict)."""
    assets = {}
    for backend, lst in zips.items():
        assets[backend] = tuple(srv.add(f"{backend}{i}.zip", make_zip(m)) for i, m in enumerate(lst))
    monkeypatch.setattr(installer, "LLAMA_ASSETS", assets)


# 1
def test_download_ok(srv, tmp_path):
    a = srv.add("f.bin", b"x" * 5000)
    dest = tmp_path / "sub" / "f.bin"
    assert installer.download(f"{srv.base}/f.bin", dest, a.sha256, a.size) == dest
    assert dest.read_bytes() == b"x" * 5000
    assert not dest.with_name("f.bin.part").exists()


def test_download_progress(srv, tmp_path):
    a = srv.add("f.bin", b"x" * 100)
    calls = []
    installer.download(f"{srv.base}/f.bin", tmp_path / "f", a.sha256, a.size,
                       progress=lambda d, t: calls.append((d, t)))  # fmt: skip
    assert calls[-1] == (100, 100)


def test_download_sha_errado(srv, tmp_path):
    a = srv.add("f.bin", b"abc" * 100)
    dest = tmp_path / "f.bin"
    with pytest.raises(InstallError, match="sha256"):
        installer.download(f"{srv.base}/f.bin", dest, "0" * 64, a.size)
    assert not dest.exists()
    assert not (tmp_path / "f.bin.part").exists()


def test_download_404(srv, tmp_path):
    with pytest.raises(InstallError, match="404"):
        installer.download(f"{srv.base}/nada", tmp_path / "n", "0" * 64)


# 2
def test_download_retoma_part(srv, tmp_path):
    data = bytes(range(256)) * 40
    a = srv.add("f.bin", data)
    (tmp_path / "f.bin.part").write_bytes(data[:3000])
    installer.download(f"{srv.base}/f.bin", tmp_path / "f.bin", a.sha256, a.size)
    assert (tmp_path / "f.bin").read_bytes() == data
    assert srv.requests == [("/f.bin", "bytes=3000-")]


# 3
def test_download_conexao_cortada(srv, tmp_path):
    data = bytes(range(256)) * 400
    a = srv.add("f.bin", data)
    srv.cut_once.add("/f.bin")
    installer.download(f"{srv.base}/f.bin", tmp_path / "f.bin", a.sha256, a.size)
    assert (tmp_path / "f.bin").read_bytes() == data
    assert srv.count() == 2
    assert srv.requests[1][1] is not None and srv.requests[1][1].startswith("bytes=")


# 4
def test_install_llama_achata_marca_e_idempotente(env, monkeypatch):
    set_llama(monkeypatch, env, cpu=[{"llama-b1/llama-server.exe": b"EXE", "llama-b1/a.dll": b"A"}])
    exe = installer.install_llama("cpu")
    assert exe == config.llama_server_path()
    assert exe.read_bytes() == b"EXE"
    assert (exe.parent / "a.dll").read_bytes() == b"A"
    assert json.loads((exe.parent / installer.MARKER).read_text()) == {
        "tag": config.LLAMA_CPP_TAG, "backend": "cpu",
    }  # fmt: skip
    assert not (config.data_dir() / "downloads" / "cpu0.zip").exists()
    n = env.count()
    installer.install_llama("cpu")
    installer.install_llama("auto")
    assert env.count() == n
    installer.install_llama("cpu", force=True)
    assert env.count() == n + 1


def test_install_llama_sem_topo_e_cuda_junta(env, monkeypatch):
    set_llama(
        monkeypatch, env,
        cuda=[{"llama-server.exe": b"EXE"}, {"cudart64.dll": b"RT"}],
    )  # fmt: skip
    exe = installer.install_llama("cuda")
    assert exe.read_bytes() == b"EXE"
    assert (exe.parent / "cudart64.dll").read_bytes() == b"RT"


def test_install_llama_auto_usa_detect(env, monkeypatch):
    set_llama(monkeypatch, env, cpu=[{"llama-server.exe": b"CPU"}], cuda=[{"llama-server.exe": b"CUDA"}])
    monkeypatch.setattr(installer, "detect_backend", lambda: "cuda")
    assert installer.install_llama().read_bytes() == b"CUDA"
    n = env.count()
    installer.install_llama("auto")
    assert env.count() == n
    installer.install_llama("cpu")  # backend explícito diferente: reinstala
    assert env.count() == n + 1
    assert config.llama_server_path().read_bytes() == b"CPU"


def test_install_llama_backend_invalido():
    with pytest.raises(ValueError):
        installer.install_llama("metal")


def test_install_llama_tag_diferente_reinstala(env, monkeypatch):
    set_llama(monkeypatch, env, cpu=[{"llama-server.exe": b"NOVO"}])
    exe = config.llama_server_path()
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"VELHO")
    (exe.parent / installer.MARKER).write_text(json.dumps({"tag": "b1", "backend": "cpu"}))
    installer.install_llama("cpu")
    assert exe.read_bytes() == b"NOVO"
    assert not exe.parent.with_name(exe.parent.name + ".old").exists()


def test_install_llama_exe_sem_marcador_respeitado(env, monkeypatch):
    set_llama(monkeypatch, env, cpu=[{"llama-server.exe": b"NOVO"}])
    exe = config.llama_server_path()
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MANUAL")
    assert installer.install_llama("cpu") == exe
    assert env.count() == 0
    assert exe.read_bytes() == b"MANUAL"


# 5
@pytest.mark.parametrize("evil", ["../evil.dll", "/abs/evil.dll", "C:/evil.dll", "ok/../../evil.dll"])
def test_install_llama_zip_inseguro(env, monkeypatch, evil, tmp_path):
    set_llama(monkeypatch, env, cpu=[{"llama-server.exe": b"X", evil: b"E"}])
    exe = config.llama_server_path()
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"VELHO")
    (exe.parent / installer.MARKER).write_text(json.dumps({"tag": "velha", "backend": "cpu"}))
    with pytest.raises(InstallError, match="inseguro"):
        installer.install_llama("cpu")
    assert exe.read_bytes() == b"VELHO"
    assert not exe.parent.with_name(exe.parent.name + ".new").exists()
    assert not (tmp_path / "evil.dll").exists()


# 6
def test_install_llama_sem_exe(env, monkeypatch):
    set_llama(monkeypatch, env, cpu=[{"x/outro.dll": b"X"}])
    with pytest.raises(InstallError, match="llama-server.exe"):
        installer.install_llama("cpu")


# 7
def set_models(monkeypatch, srv):
    assets = (srv.add("m1.gguf", b"a" * 3000), srv.add("m2.gguf", b"b" * 4000))
    monkeypatch.setattr(installer, "MODEL_ASSETS", assets)
    return assets


def test_install_models_pula_e_verify(env, monkeypatch):
    m1, m2 = set_models(monkeypatch, env)
    mdir = installer.install_models()
    assert mdir == config.models_dir()
    assert (mdir / "m1.gguf").read_bytes() == b"a" * 3000
    assert env.count() == 2
    installer.install_models()
    assert env.count() == 2
    (mdir / "m2.gguf").write_bytes(b"c" * 4000)  # corrompido, mesmo tamanho
    installer.install_models()
    assert env.count() == 2
    installer.install_models(verify=True)
    assert env.count() == 3
    assert (mdir / "m2.gguf").read_bytes() == b"b" * 4000
    installer.install_models(force=True)
    assert env.count() == 5


# 8
def test_missing_components(env, monkeypatch):
    set_models(monkeypatch, env)
    set_llama(monkeypatch, env, cpu=[{"llama-server.exe": b"EXE"}])
    assert len(installer.missing_components()) == 3
    installer.install_all("cpu")
    assert installer.missing_components() == []


def test_install_all_progress(env, monkeypatch):
    set_models(monkeypatch, env)
    set_llama(monkeypatch, env, cpu=[{"llama-server.exe": b"EXE"}])
    labels = []

    def fabrica(label):
        labels.append(label)
        return lambda d, t: None

    installer.install_all("cpu", progress=fabrica)
    assert labels == ["cpu0.zip", "m1.gguf", "m2.gguf"]


# 9
def _run(stdout="", rc=0, exc=None):
    def run(*a, **k):
        if exc:
            raise exc
        return subprocess.CompletedProcess(a, rc, stdout, "")

    return run


def test_detect_backend(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _run("GPU 0: NVIDIA RTX (UUID: x)\n"))
    assert installer.detect_backend() == "cuda"
    monkeypatch.setattr(subprocess, "run", _run("nada\n"))
    assert installer.detect_backend() == "cpu"
    monkeypatch.setattr(subprocess, "run", _run("GPU 0", rc=1))
    assert installer.detect_backend() == "cpu"
    monkeypatch.setattr(subprocess, "run", _run(exc=FileNotFoundError()))
    assert installer.detect_backend() == "cpu"
    monkeypatch.setattr(subprocess, "run", _run(exc=subprocess.TimeoutExpired("nvidia-smi", 10)))
    assert installer.detect_backend() == "cpu"


def test_console_progress(capsys):
    cb = installer.console_progress("x")
    cb(5 * 2**20, 10 * 2**20)
    cb(10 * 2**20, 10 * 2**20)
    err = capsys.readouterr().err
    assert "50% 5/10 MB" in err and err.endswith("\n")
