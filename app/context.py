from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import os,sys
@dataclass
class RunnerConfig:
    # window
    look_back_days: int = 20
    start_date: Optional[str] = None   # YYYYMMDD
    end_date: Optional[str] = None     # YYYYMMDD

    # universe
    manual_stock_symbols: List[str] = field(default_factory=list)
    index_symbols: List[str] = field(default_factory=list)
    base_index: str = "sh000001"

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".." ))
    

    # state
    state_dir: Path = Path("state")
    #scanned_file: Path = Path("state/scanned.json")
    #failed_file: Path = Path("state/failed.json") 

    # per-task state files (keep separate; refresh only clears selected task states)
    stock_scanned_file: Path = Path(os.path.join(root_dir, r"state\\STOCK_scanned.json"))
    stock_failed_file: Path = Path(os.path.join(root_dir, r"state\\STOCK_failed.json"))
    etf_scanned_file: Path = Path(os.path.join(root_dir, r"state\\ETF_scanned.json"))
    etf_failed_file: Path = Path(os.path.join(root_dir, r"state\\ETF_failed.json"))
    opt_scanned_file: Path = Path(os.path.join(root_dir, r"state\\OPT_scanned.json"))
    opt_failed_file: Path = Path(os.path.join(root_dir, r"state\\OPT_failed.json"))

    
    
    scanned_file: Path =  Path(os.path.join(root_dir, r"state\scanned.json"))
    failed_file: Path =  Path(os.path.join(root_dir, r"state\failed.json") )
 
    state_flush_every: int = 3

    # audit
    #audit_output_dir: Path = Path("audit_reports")
    audit_reports_dir = os.path.join(root_dir, "audit_reports")

    # runtime flags
    refresh_state: bool = False

    def finalize_dates(self) -> None:
        """Fill start/end if missing using look_back_days, using local date()."""
        if self.start_date is None:
            self.start_date = (date.today() - timedelta(days=self.look_back_days)).strftime("%Y%m%d")
        if self.end_date is None:
            self.end_date = date.today().strftime("%Y%m%d")