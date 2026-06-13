"""Static audit: locate remaining cn_ga_mainline_radar_daily references in AShareScraper code.

Usage:
  python scripts/audit_radar_consumers.py --root . --strict
"""
from __future__ import annotations
import argparse
from pathlib import Path

ALLOW = {
    "scripts/audit_radar_consumers.py",
    "docs/MAINLINE_MARKET_PULSE_UNIFIED_ALPHA_FACT_REWIRE_P0I.md",
}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    root = Path(args.root).resolve()
    hits: list[tuple[str,int,str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".sql", ".md", ".bat"}:
            continue
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "cn_ga_mainline_radar_daily" not in text:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "cn_ga_mainline_radar_daily" in line:
                hits.append((rel, i, line.strip()))
    real_hits = [h for h in hits if h[0] not in ALLOW]
    print("[RADAR CONSUMER AUDIT]", {"hits": len(hits), "non_allowlisted_hits": len(real_hits)})
    for rel, i, line in real_hits[:200]:
        print(f"{rel}:{i}: {line}")
    if args.strict and real_hits:
        raise SystemExit("[RADAR CONSUMER AUDIT FAILED] remaining non-allowlisted references found")
    print("[RADAR CONSUMER AUDIT PASS]")

if __name__ == "__main__":
    main()
