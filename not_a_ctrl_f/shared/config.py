"""Configuração e caminhos da aplicação."""

import os
from pathlib import Path

APP_NAME = "not-a-ctrl-f"

# EmbeddingGemma 2: espaço único para texto, imagem, áudio e vídeo.
# Mudar qualquer um destes invalida o índice (entra no fingerprint).
EMBED_DIM = 768
VECTOR_DTYPE = "float16"  # little-endian, L2-normalizado


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
