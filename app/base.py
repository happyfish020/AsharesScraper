from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.context import RunnerConfig

class Task(Protocol):
    name: str
    def run(self, ctx: "RunContext") -> None: ...

@dataclass
class RunContext:
    config: "RunnerConfig"
    engine: object
    log: object
