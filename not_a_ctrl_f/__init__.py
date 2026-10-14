"""Busca local por palavra-chave e semântica nos seus arquivos."""

import sys
from importlib import metadata

try:
    __version__ = metadata.version("not-a-ctrl-f")
except metadata.PackageNotFoundError:
    __version__ = "dev"


def main() -> None:
    from not_a_ctrl_f.cli import main as _main

    sys.exit(_main())
