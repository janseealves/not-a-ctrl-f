"""Embeddings do EmbeddingGemma 2 via llama-server (subprocesso + HTTP local)."""

import base64
import io
import json
import math
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from not_a_ctrl_f.shared import config

BUDGETS = (70, 140, 280, 560, 1120)
PROMPTS_VERSION = 1  # mude se os prompts mudarem: entra no fingerprint

_PATCH = 16
_POOLING = 3
_MAX_ERR = 300


class EmbedderError(Exception):
    """Falha do servidor/cliente que não é culpa de um item específico."""


@dataclass(slots=True)
class Embedding:
    vector: np.ndarray | None  # float32 (768,), L2-normalizado
    error: str | None = None


class _ItemError(Exception):
    """Servidor recusou a requisição (HTTP 4xx/5xx)."""


# --- prompts -----------------------------------------------------------------


def query_prompt(text: str) -> str:
    return f"task: search result | query: {text}"


def document_prompt(text: str, title: str | None = None) -> str:
    return f"title: {title or 'none'} | text: {text}"


# --- imagens -----------------------------------------------------------------


def image_target_size(height: int, width: int, budget: int) -> tuple[int, int] | None:
    """(h, w) final para caber no orçamento de soft tokens; lados múltiplos de 48."""
    if height <= 0 or width <= 0:
        return None
    max_patches = budget * _POOLING * _POOLING
    factor = math.sqrt(max_patches * _PATCH * _PATCH / (height * width))
    side = _POOLING * _PATCH
    th = math.floor(factor * height / side) * side
    tw = math.floor(factor * width / side) * side
    if th == 0 and tw == 0:
        return None
    max_side = (max_patches // (_POOLING * _POOLING)) * side
    if th == 0:
        th = side
        tw = min(width // height * side, max_side)
    elif tw == 0:
        tw = side
        th = min(height // width * side, max_side)
    return th, tw


def prepare_image(data: bytes, budget: int) -> bytes:
    """Abre, converte p/ RGB, redimensiona ao orçamento e devolve PNG."""
    if budget not in BUDGETS:
        raise ValueError(f"budget inválido: {budget} (válidos: {BUDGETS})")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            size = image_target_size(img.height, img.width, budget)
            if size is None:
                raise EmbedderError("imagem fina demais")
            th, tw = size
            img = img.resize((tw, th), Image.Resampling.BICUBIC)
            out = io.BytesIO()
            img.save(out, format="PNG")
            return out.getvalue()
    except EmbedderError:
        raise
    except Exception as e:
        raise EmbedderError(f"imagem ilegível: {e}") from e


# --- vetores -----------------------------------------------------------------


def to_blob(vector: np.ndarray) -> bytes:
    v = np.asarray(vector)
    if v.shape != (config.EMBED_DIM,):
        raise ValueError(f"shape {v.shape}, esperado ({config.EMBED_DIM},)")
    return v.astype("<f2").tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f2").astype(np.float32)


def fingerprint(text_only: bool = False) -> str:
    fp = (
        f"google/embeddinggemma-2@{config.MODEL_REVISION[:7]} q8_0 "
        f"llama.cpp-{config.LLAMA_CPP_TAG} mean dim{config.EMBED_DIM} f16 "
        f"prompts{PROMPTS_VERSION}"
    )
    return fp + " text-only" if text_only else fp


# --- processo llama-server ---------------------------------------------------


def build_server_args(
    server: Path,
    models: Path,
    *,
    port: int,
    text_only: bool,
    gpu_layers: int,
    threads: int | None,
) -> list[str]:
    args = [str(server), "-m", str(models / config.TEXT_MODEL_FILE)]
    if not text_only:
        args += ["--mmproj", str(models / config.MEDIA_MODEL_FILE)]
    args += [
        "--embeddings", "--pooling", "mean",
        "--host", "127.0.0.1", "--port", str(port),
        "-c", "8192", "-np", "4", "--kv-unified",
        "-b", "2048", "-ub", "2048", "-ngl", str(gpu_layers),
    ]  # fmt: skip
    if not text_only:
        # imagens já chegam no tamanho final: o servidor não deve reescalar
        args += ["--image-min-tokens", "1", "--image-max-tokens", "1120"]
    args.append("--no-webui")
    if threads is not None:
        args += ["-t", str(threads)]
    return args


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    """Processo llama-server. start() sobe e espera /health; close() encerra."""

    def __init__(
        self,
        *,
        server_path: Path | None = None,
        models_dir: Path | None = None,
        text_only: bool = False,
        port: int | None = None,
        gpu_layers: int = 999,
        threads: int | None = None,
        startup_timeout: float = 180.0,
    ):
        self.server_path = Path(server_path) if server_path else config.llama_server_path()
        self.models_dir = Path(models_dir) if models_dir else config.models_dir()
        self.text_only = text_only
        self.port = port
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.startup_timeout = startup_timeout
        self.base_url = ""
        self.log_path = config.data_dir() / "logs" / "llama-server.log"
        self._proc: subprocess.Popen | None = None
        self._log = None

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def _check_files(self) -> None:
        if not self.server_path.is_file():
            raise EmbedderError(
                f"llama-server não encontrado em {self.server_path}. "
                f"Coloque o llama-server.exe do release {config.LLAMA_CPP_TAG} (com as DLLs) nesse caminho."
            )
        files = [config.TEXT_MODEL_FILE]
        if not self.text_only:
            files.append(config.MEDIA_MODEL_FILE)
        for name in files:
            if not (self.models_dir / name).is_file():
                raise EmbedderError(
                    f"modelo ausente: {self.models_dir / name}. "
                    f"Baixe {name} de {config.MODEL_REPO} para {self.models_dir}."
                )

    def _log_tail(self, n: int = 20) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    def start(self) -> "LlamaServer":
        if self._proc is not None:
            return self
        self._check_files()
        port = self.port or _free_port()
        args = build_server_args(
            self.server_path, self.models_dir, port=port, text_only=self.text_only,
            gpu_layers=self.gpu_layers, threads=self.threads,
        )  # fmt: skip
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        try:
            self._proc = subprocess.Popen(
                args, stdout=self._log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, creationflags=flags,
            )  # fmt: skip
        except OSError as e:
            self._close_log()
            raise EmbedderError(f"falha ao iniciar {self.server_path}: {e}") from e
        self.base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + self.startup_timeout
        while True:
            if self._proc.poll() is not None:
                code = self._proc.returncode
                self.close()
                raise EmbedderError(f"llama-server encerrou (código {code}):\n{self._log_tail()}")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except OSError:
                pass  # ainda carregando (503) ou porta ainda fechada (URLError/HTTPError são OSError)
            if time.monotonic() >= deadline:
                self.close()
                raise EmbedderError(
                    f"llama-server não ficou pronto em {self.startup_timeout:.0f}s:\n{self._log_tail()}"
                )
            time.sleep(0.25)

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._close_log()


# --- cliente -----------------------------------------------------------------


def _short(msg: str) -> str:
    return " ".join(msg.split())[:_MAX_ERR]


class Embedder:
    """Cliente HTTP do llama-server. Thread-safe (Lock em volta de cada requisição)."""

    def __init__(
        self,
        base_url: str,
        *,
        text_only: bool = False,
        timeout: float = 300.0,
        batch_size: int = 16,
    ):
        self.base_url = base_url.rstrip("/")
        self.text_only = text_only
        self.timeout = timeout
        self.batch_size = batch_size
        self._lock = threading.Lock()

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.text_only)

    def _post(self, inputs: list) -> list[np.ndarray]:
        """Uma requisição. _ItemError em HTTP de erro; EmbedderError no resto."""
        body = json.dumps({"input": inputs, "encoding_format": "float"}).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/embeddings", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )  # fmt: skip
        try:
            with self._lock, urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = ""
            try:
                err = json.loads(e.read()).get("error")
                msg = err.get("message", "") if isinstance(err, dict) else str(err)
            except Exception:
                pass
            raise _ItemError(_short(msg or f"HTTP {e.code}")) from e
        except OSError as e:  # inclui URLError, timeout e conexão recusada
            raise EmbedderError(f"servidor de embeddings inacessível: {e}") from e
        except ValueError as e:
            raise EmbedderError(f"resposta inválida do servidor: {e}") from e

        try:
            data = sorted(payload["data"], key=lambda d: d["index"])
            raw = [d["embedding"] for d in data]
        except (KeyError, TypeError) as e:
            raise EmbedderError(f"resposta inválida do servidor: {e!r}") from e
        if len(raw) != len(inputs):
            raise EmbedderError(f"servidor devolveu {len(raw)} vetores para {len(inputs)} entradas")
        out = []
        for emb in raw:
            v = np.asarray(emb, dtype=np.float32)
            if v.shape != (config.EMBED_DIM,):
                raise EmbedderError(f"dimensão {v.shape}, esperado ({config.EMBED_DIM},)")
            out.append(v)
        return out

    @staticmethod
    def _normalize(v: np.ndarray) -> Embedding:
        norm = float(np.linalg.norm(v))
        if not math.isfinite(norm) or norm == 0.0:
            return Embedding(None, "vetor inválido (nulo ou não finito)")
        return Embedding((v / norm).astype(np.float32))

    def _embed(self, inputs: list, batch: int) -> list[Embedding]:
        out: list[Embedding] = []
        for i in range(0, len(inputs), batch):
            chunk = inputs[i : i + batch]
            try:
                vecs: list = self._post(chunk)
            except _ItemError:
                vecs = []
                for item in chunk:  # lote falhou: isola o item culpado
                    try:
                        vecs.append(self._post([item])[0])
                    except _ItemError as e:
                        vecs.append(str(e))
            for v in vecs:
                out.append(Embedding(None, v) if isinstance(v, str) else self._normalize(v))
        return out

    def embed_query(self, text: str) -> np.ndarray:
        try:
            vecs = self._post([query_prompt(text)])
        except _ItemError as e:
            raise EmbedderError(f"consulta recusada: {e}") from e
        emb = self._normalize(vecs[0])
        if emb.vector is None:
            raise EmbedderError(emb.error or "vetor inválido")
        return emb.vector

    def embed_documents(self, items: list[tuple[str, str | None]]) -> list[Embedding]:
        return self._embed([document_prompt(t, title) for t, title in items], self.batch_size)

    def embed_images(self, images: list[bytes], budget: int = 140) -> list[Embedding]:
        if self.text_only:
            raise EmbedderError("embedder em modo somente texto: sem suporte a imagens")
        if budget not in BUDGETS:
            raise ValueError(f"budget inválido: {budget} (válidos: {BUDGETS})")
        results: list[Embedding | None] = [None] * len(images)
        inputs, idx = [], []
        for i, data in enumerate(images):
            try:
                png = prepare_image(data, budget)
            except EmbedderError as e:
                results[i] = Embedding(None, _short(str(e)))
                continue
            b64 = base64.b64encode(png).decode()
            url = f"data:image/png;base64,{b64}"
            inputs.append({"content": [{"type": "image_url", "image_url": {"url": url}}]})
            idx.append(i)
        for i, emb in zip(idx, self._embed(inputs, min(self.batch_size, 4))):
            results[i] = emb
        return results  # type: ignore[return-value]

    def embed_audio(self, clips: list[bytes]) -> list[Embedding]:
        if self.text_only:
            raise EmbedderError("embedder em modo somente texto: sem suporte a áudio")
        inputs = [
            {
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": base64.b64encode(c).decode(), "format": "wav"},
                    }
                ]
            }
            for c in clips
        ]
        return self._embed(inputs, min(self.batch_size, 2))
