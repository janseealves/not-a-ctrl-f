"""Configuração e caminhos da aplicação."""

import os
from pathlib import Path
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

APP_NAME = "not-a-ctrl-f"

# EmbeddingGemma 2: espaço único para texto, imagem, áudio e vídeo.
# Mudar qualquer um destes invalida o índice (entra no fingerprint).
EMBED_DIM = 768
VECTOR_DTYPE = "float16"  # little-endian, L2-normalizado

# GGUF Q8_0 da ggml-org numa revisão fixa. Nunca F16: as ativações do modelo estouram em float16.
MODEL_REPO = "ggml-org/embeddinggemma-2-GGUF"
MODEL_REVISION = "bfcd298762cc34d0357ece5ebdd31791a3a374d8"
TEXT_MODEL_FILE = "embeddinggemma-2-Q8_0.gguf"         # 310 MB: texto (basta para consultas)
MEDIA_MODEL_FILE = "mmproj-embeddinggemma-2-Q8_0.gguf"  # 555 MB: encoders de imagem e áudio

# llama.cpp com EmbeddingGemma 2 (texto + visão + áudio) a partir de b11452.
LLAMA_CPP_TAG = "b11461"


def data_dir() -> Path:
    """Diretório de dados do app. Não é criado aqui."""
    local = os.getenv("LOCALAPPDATA")
    if local:
        return Path(local) / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


class Settings(BaseSettings):
    """Configuração do usuário: init > env (NACF_*) > data_dir()/config.toml > padrões."""

    model_config = SettingsConfigDict(env_prefix="NACF_", env_ignore_empty=True, extra="ignore")

    db_path: Path | None = None
    models_dir: Path | None = None
    llama_server: Path | None = None
    # no env: separadas por os.pathsep (não JSON); no toml: lista
    watch_dirs: Annotated[list[Path] | None, NoDecode] = None

    @field_validator("watch_dirs", mode="before")
    @classmethod
    def _split_watch_dirs(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [p.strip() for p in v.split(os.pathsep) if p.strip()]
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml = TomlConfigSettingsSource(settings_cls, toml_file=data_dir() / "config.toml")
        return init_settings, env_settings, toml


def get_settings() -> Settings:
    """Sem cache: é barato e os testes mudam o ambiente."""
    return Settings()


def default_db_path() -> Path:
    return get_settings().db_path or data_dir() / "index.db"


def models_dir() -> Path:
    """Onde ficam os dois GGUF (TEXT_MODEL_FILE e MEDIA_MODEL_FILE)."""
    return get_settings().models_dir or data_dir() / "models"


def llama_server_path() -> Path:
    """O llama-server.exe do release LLAMA_CPP_TAG (zip win-cuda ou win-cpu, com as DLLs ao lado)."""
    return get_settings().llama_server or data_dir() / "llama.cpp" / "llama-server.exe"


# (KNOWNFOLDERID, nome em inglês sob ~ se a API falhar)
_KNOWN_FOLDERS: tuple[tuple[str, str], ...] = (
    ("{FDD39AD0-238F-46AF-ADB4-6C85480369C7}", "Documents"),
    ("{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}", "Desktop"),
    ("{374DE290-123F-4565-9164-39C4925E467B}", "Downloads"),
    ("{33E28130-4E1E-4676-835A-98395C3BC3BB}", "Pictures"),
    ("{4BD8D571-6D19-48D3-BE97-422220080E43}", "Music"),
    ("{18989B1D-99B5-455B-841C-AB7C74E4DDFC}", "Videos"),
)


def _known_folder(folder_id: str) -> Path | None:
    """Resolve uma pasta conhecida do Windows (respeita redirecionamento do OneDrive)."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        u = uuid.UUID(folder_id)
        guid = GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        ole32 = ctypes.WinDLL("ole32")
        shell32.SHGetKnownFolderPath.argtypes = [
            ctypes.POINTER(GUID), wintypes.DWORD, wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p),
        ]
        shell32.SHGetKnownFolderPath.restype = ctypes.HRESULT
        ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        ptr = ctypes.c_void_p()
        try:
            shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(ptr))
            return Path(ctypes.wstring_at(ptr.value)) if ptr.value else None
        finally:
            ole32.CoTaskMemFree(ptr)
    except Exception:
        return None


def watched_dirs() -> list[Path]:
    """Pastas monitoradas. watch_dirs (NACF_WATCH_DIRS ou config.toml) ou as pastas conhecidas do usuário que existirem."""
    override = get_settings().watch_dirs
    if override:
        candidates = override
    else:
        candidates = [
            _known_folder(fid) or Path.home() / fallback for fid, fallback in _KNOWN_FOLDERS
        ]
    found: dict[str, Path] = {}
    for p in candidates:
        if p.is_dir():
            found.setdefault(os.path.normcase(os.path.abspath(p)), p)
    # remove pastas dentro de outra da lista
    keys = list(found)
    return [
        found[k]
        for k in keys
        if not any(o != k and k.startswith(o.rstrip(os.sep) + os.sep) for o in keys)
    ]
