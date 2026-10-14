import os
from pathlib import Path

import pytest

from not_a_ctrl_f.shared import config


@pytest.fixture(autouse=True)
def limpo(tmp_path, monkeypatch):
    for k in ("DB_PATH", "MODELS_DIR", "LLAMA_SERVER", "WATCH_DIRS", "PORT"):
        monkeypatch.delenv(f"NACF_{k}", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    return tmp_path


def test_defaults(limpo):
    base = limpo / config.APP_NAME
    assert config.default_db_path() == base / "index.db"
    assert config.models_dir() == base / "models"
    assert config.llama_server_path() == base / "llama.cpp" / "llama-server.exe"
    assert config.get_settings().watch_dirs is None


def test_env_sobrescreve(limpo, monkeypatch):
    monkeypatch.setenv("NACF_DB_PATH", str(limpo / "x.db"))
    monkeypatch.setenv("NACF_MODELS_DIR", str(limpo / "m"))
    monkeypatch.setenv("NACF_LLAMA_SERVER", str(limpo / "ls.exe"))
    assert config.default_db_path() == limpo / "x.db"
    assert config.models_dir() == limpo / "m"
    assert config.llama_server_path() == limpo / "ls.exe"


def test_watch_dirs_env_separado_por_pathsep(monkeypatch):
    monkeypatch.setenv("NACF_WATCH_DIRS", f"X{os.pathsep} Y {os.pathsep}")
    assert config.get_settings().watch_dirs == [Path("X"), Path("Y")]


def _toml(limpo, text):
    d = limpo / config.APP_NAME
    d.mkdir()
    (d / "config.toml").write_text(text, encoding="utf-8")


def test_config_toml(limpo):
    a, b = limpo / "a", limpo / "b"
    a.mkdir()
    b.mkdir()
    _toml(limpo, f'watch_dirs = ["{a.as_posix()}", "{b.as_posix()}"]\nmodels_dir = "{limpo.as_posix()}/mm"\n')
    assert config.get_settings().watch_dirs == [a, b]
    assert config.models_dir() == limpo / "mm"
    assert config.watched_dirs() == [a, b]


def test_env_vence_toml(limpo, monkeypatch):
    _toml(limpo, f'models_dir = "{limpo.as_posix()}/mm"\nwatch_dirs = ["{limpo.as_posix()}/a"]\n')
    monkeypatch.setenv("NACF_MODELS_DIR", str(limpo / "env"))
    monkeypatch.setenv("NACF_WATCH_DIRS", "X")
    assert config.models_dir() == limpo / "env"
    assert config.get_settings().watch_dirs == [Path("X")]


def test_ui_port(limpo, monkeypatch):
    assert config.ui_port() == 47615
    monkeypatch.setenv("NACF_PORT", "50000")
    assert config.ui_port() == 50000
    monkeypatch.delenv("NACF_PORT")
    _toml(limpo, "port = 51000\n")
    assert config.ui_port() == 51000
