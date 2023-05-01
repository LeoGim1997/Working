from __future__ import annotations
from pathlib import Path


class Constants:
    @property
    def ressourcePath(self):
        return Path(__file__).parents[1] / "resources"
