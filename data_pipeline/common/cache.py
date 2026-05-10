from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


class CacheStore:
    def __init__(self, root: str | Path = "cache/mainline_data") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def build_path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        path = self.root / namespace / f"{digest}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load_frame(self, namespace: str, key: str) -> pd.DataFrame | None:
        path = self.build_path(namespace, key)
        if not path.exists():
            return None
        return pd.read_pickle(path)

    def save_frame(self, namespace: str, key: str, frame: pd.DataFrame) -> Path:
        path = self.build_path(namespace, key)
        frame.to_pickle(path)
        return path

    def save_object(self, namespace: str, key: str, payload: Any) -> Path:
        path = self.build_path(namespace, key)
        pd.to_pickle(payload, path)
        return path

    def load_object(self, namespace: str, key: str) -> Any | None:
        path = self.build_path(namespace, key)
        if not path.exists():
            return None
        return pd.read_pickle(path)
