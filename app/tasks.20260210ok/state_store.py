from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Iterable

@dataclass
class StateStore:
    scanned_path: Path
    failed_path: Path
    log: object

    def load_scanned(self) -> Set[str]:
        return self._load_set(self.scanned_path)

    def load_failed(self) -> Set[str]:
        return self._load_set(self.failed_path)

    def save_scanned(self, items: Iterable[str]) -> None:
        self._save_set(self.scanned_path, items)

    def save_failed(self, items: Iterable[str]) -> None:
        self._save_set(self.failed_path, items)

    def _load_set(self, path: Path) -> Set[str]:
        if not path.exists():
            return set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return set(obj)
            if isinstance(obj, dict):
                # tolerate older dict format: keys are symbols
                return set(obj.keys())
            return set()
        except Exception as e:
            self.log.warning(f"读取状态文件失败: {path} err={e}")
            return set()

    def _save_set(self, path: Path, items: Iterable[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sorted(set(items)), f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log.warning(f"写入状态文件失败: {path} err={e}")
