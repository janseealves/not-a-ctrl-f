"""Instalador do llama.cpp e dos modelos GGUF: download verificável (sha256), retomável e idempotente."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import httpx

from not_a_ctrl_f.shared import config

MARKER = "nacf-install.json"
_CHUNK = 1 << 20


class InstallError(Exception):
    pass


class Asset(NamedTuple):
    name: str
    sha256: str
    size: int


_TAG = config.LLAMA_CPP_TAG

# Hashes e tamanhos valem para a tag b11461.
LLAMA_ASSETS: dict[str, tuple[Asset, ...]] = {
    "cpu": (
        Asset(f"llama-{_TAG}-bin-win-cpu-x64.zip",
              "e283199134e74405a3684cfaa3911708ca7d051a8c2a8dcb10b3dfa1678913c3", 19437105),
    ),
    "cuda": (
        Asset(f"llama-{_TAG}-bin-win-cuda-12.4-x64.zip",
              "bce20c3417ffe3f4d81b837c3c0b0a77f409d43afe0012870b89ad6a0c79b021", 264528471),
        Asset("cudart-llama-bin-win-cuda-12.4-x64.zip",
              "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6", 391443627),
    ),
    "vulkan": (
        Asset(f"llama-{_TAG}-bin-win-vulkan-x64.zip",
              "6b9ca8ad3e43ac3e166d1997875709e4547214bb5bf1c2ebfdd17a2350a8380c", 33377638),
    ),
}  # fmt: skip

MODEL_ASSETS: tuple[Asset, ...] = (
    Asset(config.TEXT_MODEL_FILE,
          "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135", 309855456),
    Asset(config.MEDIA_MODEL_FILE,
          "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f", 554821024),
)  # fmt: skip

BACKENDS = ("auto", "cuda", "vulkan", "cpu")

Progress = Callable[[str], Callable[[int, int | None], None]]


def llama_url(name: str) -> str:
    return f"https://github.com/ggml-org/llama.cpp/releases/download/{config.LLAMA_CPP_TAG}/{name}"


def model_url(name: str) -> str:
    return f"https://huggingface.co/{config.MODEL_REPO}/resolve/{config.MODEL_REVISION}/{name}"


def detect_backend() -> str:
    """"cuda" se há GPU NVIDIA (nvidia-smi -L); senão "cpu". Nunca lança."""
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        r = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10, creationflags=flags
        )
        if r.returncode == 0 and "GPU" in (r.stdout or ""):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _model_ok(asset: Asset, verify: bool = False) -> bool:
    p = config.models_dir() / asset.name
    try:
        if not p.is_file() or p.stat().st_size != asset.size:
            return False
    except OSError:
        return False
    return not verify or _sha256_file(p) == asset.sha256


def missing_components() -> list[str]:
    """O que falta instalar (existência/tamanho, sem hash). [] = tudo pronto."""
    out = []
    if not config.llama_server_path().is_file():
        out.append("llama-server.exe")
    for a in MODEL_ASSETS:
        if not _model_ok(a):
            out.append(f"modelo {a.name}")
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def download(
    url: str,
    dest: Path,
    sha256: str,
    size: int | None = None,
    *,
    progress: Callable[[int, int | None], None] | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Baixa url para dest (via dest.part, retomável), conferindo tamanho e sha256.

    progress aqui é o callback (feito, total); as funções install_* recebem a fábrica.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    own = client is None
    if own:
        client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={"User-Agent": "not-a-ctrl-f"},
        )
    try:
        failures = 0
        restarted = False
        while True:
            offset = part.stat().st_size if part.exists() else 0
            if size is not None and offset > size:
                _unlink(part)
                offset = 0
            if size is not None and offset == size:
                break
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with client.stream("GET", url, headers=headers) as r:
                    if r.status_code == 416 and offset and not restarted:
                        restarted = True
                        _unlink(part)
                        continue
                    if r.status_code >= 400:
                        raise InstallError(f"falha no download (HTTP {r.status_code}): {url}")
                    if r.status_code == 206 and offset:
                        mode = "ab"
                    else:
                        mode, offset = "wb", 0
                    total = size
                    if total is None and (cl := r.headers.get("content-length")):
                        total = offset + int(cl)
                    done = offset
                    with open(part, mode) as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
                            done += len(chunk)
                            if progress:
                                progress(done, total)
                break
            except httpx.TransportError as e:
                failures += 1
                if failures >= 3:
                    raise InstallError(f"falha de rede ao baixar {url}: {e}") from e
    finally:
        if own:
            client.close()
    got = part.stat().st_size
    if size is not None and got != size:
        _unlink(part)
        raise InstallError(f"tamanho incorreto em {url}: esperado {size}, recebido {got}")
    if _sha256_file(part) != sha256:
        _unlink(part)
        raise InstallError(f"sha256 não confere em {url}; download descartado")
    os.replace(part, dest)
    return dest


def _free_bytes(path: Path) -> int:
    p = path.resolve()
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def _check_space(path: Path, need: float, what: str) -> None:
    free = _free_bytes(path)
    if free < need:
        raise InstallError(
            f"espaço em disco insuficiente para {what}: precisa de {need / 2**20:.0f} MB, "
            f"livres {free / 2**20:.0f} MB em {path}"
        )


def _safe_parts(name: str) -> list[str]:
    n = name.replace("\\", "/")
    if n.startswith("/") or re.match(r"^[A-Za-z]:", n):
        raise InstallError(f"zip com caminho inseguro: {name}")
    parts = [p for p in PurePosixPath(n).parts if p not in ("", ".")]
    if ".." in parts:
        raise InstallError(f"zip com caminho inseguro: {name}")
    return parts


def _extract(zips: list[Path], out: Path) -> None:
    for z in zips:
        try:
            zf = zipfile.ZipFile(z)
        except zipfile.BadZipFile as e:
            raise InstallError(f"zip inválido: {z.name}") from e
        with zf:
            members = [(i, _safe_parts(i.filename)) for i in zf.infolist()]
            members = [(i, p) for i, p in members if p]
            tops = {p[0] for _, p in members}
            strip = (
                len(tops) == 1
                and all(len(p) > 1 or i.is_dir() for i, p in members)
            )
            root = out.resolve()
            for info, parts in members:
                rel = parts[1:] if strip else parts
                if not rel:
                    continue
                target = out.joinpath(*rel)
                if not target.resolve().is_relative_to(root):
                    raise InstallError(f"zip com caminho inseguro: {info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def _read_marker(target: Path) -> dict | None:
    try:
        m = json.loads((target / MARKER).read_text(encoding="utf-8"))
        return m if isinstance(m, dict) else None
    except (OSError, ValueError):
        return None


def install_llama(backend: str = "auto", *, force: bool = False, progress: Progress | None = None) -> Path:
    """Instala o llama.cpp em llama_server_path().parent. Devolve o caminho do llama-server.exe."""
    if backend not in BACKENDS:
        raise ValueError(f"backend inválido: {backend!r} (use um de {', '.join(BACKENDS)})")
    exe = config.llama_server_path()
    target = exe.parent
    marker = _read_marker(target)
    if not force and exe.is_file():
        if marker is None:
            return exe  # instalação manual: respeitar
        if marker.get("tag") == config.LLAMA_CPP_TAG and backend in ("auto", marker.get("backend")):
            return exe
    chosen = detect_backend() if backend == "auto" else backend
    assets = LLAMA_ASSETS[chosen]
    _check_space(target.parent, 2.5 * sum(a.size for a in assets), "o llama.cpp")

    dl_dir = config.data_dir() / "downloads"
    new = target.with_name(target.name + ".new")
    old = target.with_name(target.name + ".old")
    zips = [
        download(llama_url(a.name), dl_dir / a.name, a.sha256, a.size,
                 progress=progress(a.name) if progress else None)
        for a in assets
    ]  # fmt: skip
    try:
        shutil.rmtree(new, ignore_errors=True)
        new.mkdir(parents=True)
        _extract(zips, new)
        if not (new / exe.name).is_file():
            raise InstallError(f"{exe.name} não encontrado nos arquivos baixados")
        (new / MARKER).write_text(
            json.dumps({"tag": config.LLAMA_CPP_TAG, "backend": chosen}), encoding="utf-8"
        )
        had_old = target.exists()
        try:
            if had_old:
                shutil.rmtree(old, ignore_errors=True)
                os.rename(target, old)
            try:
                os.rename(new, target)
            except OSError:
                if had_old:
                    os.rename(old, target)
                raise
        except OSError as e:
            raise InstallError(
                f"não foi possível substituir {target} ({e}): "
                "feche o not-a-ctrl-f (llama-server em uso) e tente de novo"
            ) from e
        if had_old:
            shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(new, ignore_errors=True)
        raise
    for z in zips:
        _unlink(z)
    return exe


def install_models(
    *, force: bool = False, verify: bool = False, progress: Progress | None = None
) -> Path:
    """Baixa os GGUF que faltam (ou estão corrompidos) em models_dir()."""
    mdir = config.models_dir()
    todo = [a for a in MODEL_ASSETS if force or not _model_ok(a, verify)]
    if todo:
        mdir.mkdir(parents=True, exist_ok=True)
        _check_space(mdir, 1.1 * sum(a.size for a in todo), "os modelos")
    for a in todo:
        download(model_url(a.name), mdir / a.name, a.sha256, a.size,
                 progress=progress(a.name) if progress else None)  # fmt: skip
    return mdir


def install_all(
    backend: str = "auto", *, force: bool = False, verify: bool = False, progress: Progress | None = None
) -> None:
    install_llama(backend, force=force, progress=progress)
    install_models(force=force, verify=verify, progress=progress)


def console_progress(label: str) -> Callable[[int, int | None], None]:
    """Callback que imprime 'label  45% 123/270 MB' em stderr, na mesma linha."""
    last = [0.0]

    def cb(done: int, total: int | None) -> None:
        now = time.monotonic()
        finished = total is not None and done >= total
        if not finished and now - last[0] < 0.2:
            return
        last[0] = now
        mb = done / 2**20
        if total:
            text = f"{label}  {done * 100 // total}% {mb:.0f}/{total / 2**20:.0f} MB"
        else:
            text = f"{label}  {mb:.0f} MB"
        sys.stderr.write("\r" + text + ("\n" if finished else ""))
        sys.stderr.flush()

    return cb
