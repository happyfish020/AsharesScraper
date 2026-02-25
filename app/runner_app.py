from __future__ import annotations

import shutil
from typing import List

from app.base import RunContext, Task
from app.context import RunnerConfig

class RunnerApp:
    def __init__(self, ctx: RunContext, tasks: List[Task]):
        self.ctx = ctx
        self.tasks = tasks

    def run(self) -> None:
        cfg = self.ctx.config
        if cfg.refresh_state:
            self._clear_state_files()
        self.ctx.log.info(f"启动 A股数据加载，回溯天数: {cfg.look_back_days} | start={cfg.start_date} end={cfg.end_date}")
        for i, task in enumerate(self.tasks, start=1):
            self.ctx.log.info(f"[TASK {i}/{len(self.tasks)}] {task.name} - START")
            task.run(self.ctx)
            self.ctx.log.info(f"[TASK {i}/{len(self.tasks)}] {task.name} - DONE")
        self.ctx.log.info("本次运行完成")


    def _clear_state_files(self) -> None:
        """Clear ONLY the state files relevant to the selected tasks in this run."""
        cfg = self.ctx.config
        folder_path = cfg.state_dir

     
    
        for item in folder_path.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item) # shutil.rmtree works with Path objects
                self.ctx.log.info(f"[REFRESH] cleared state file: {item}")
            except Exception as e:
                self.ctx.log.warn (f'Failed to delete {item}. Reason: {e}')
       

    def _clear_state_files_1(self) -> None:
        """Clear ONLY the state files relevant to the selected tasks in this run."""
        cfg = self.ctx.config

        state_files = []
        for t in self.tasks:
            getter = getattr(t, "get_state_files", None)
            if callable(getter):
                try:
                    state_files.extend(getter(cfg) or [])
                except Exception as e:
                    self.ctx.log.warning(f"获取任务状态文件失败: task={getattr(t,'name',type(t))} err={e}")

        # Backward compatibility: if no task exposes state files, fall back to the original two.
        if not state_files:
            state_files = [cfg.scanned_file, cfg.failed_file]

        # De-dup
        uniq = []
        seen = set()
        for p in state_files:
            try:
                from pathlib import Path
                pp = p if isinstance(p, Path) else Path(str(p))
            except Exception:
                continue
            if str(pp) in seen:
                continue
            seen.add(str(pp))
            uniq.append(pp)

        for p in uniq:
            try:
                if p.exists():
                    p.unlink()
                    self.ctx.log.info(f"[REFRESH] cleared state file: {p}")
            except Exception as e:
                self.ctx.log.warning(f"清空状态文件失败: {p} err={e}")
