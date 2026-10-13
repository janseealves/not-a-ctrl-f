"""Configuração e caminhos da aplicação."""

import os
from pathlib import Path

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
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


def default_db_path() -> Path:
    override = os.environ.get("NACF_DB_PATH")
    if override:
        return Path(override)
    return data_dir() / "index.db"


def models_dir() -> Path:
    """Onde ficam os dois GGUF (TEXT_MODEL_FILE e MEDIA_MODEL_FILE)."""
    override = os.environ.get("NACF_MODELS_DIR")
    if override:
        return Path(override)
    return data_dir() / "models"


def llama_server_path() -> Path:
    """O llama-server.exe do release LLAMA_CPP_TAG (zip win-cuda ou win-cpu, com as DLLs ao lado)."""
    override = os.environ.get("NACF_LLAMA_SERVER")
    if override:
        return Path(override)
    return data_dir() / "llama.cpp" / "llama-server.exe"
