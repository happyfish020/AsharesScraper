from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SectorRotationSnapshotTask:
    name: str = 'SectorRotationSnapshotTask_Deprecated'

    def run(self, ctx) -> None:
        raise RuntimeError('rotation_snapshot_source.py is deprecated. Use app/tasks/rotation_sector_snapshot_task.py')
