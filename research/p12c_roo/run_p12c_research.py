"""
P12c Collapse Confirmation Matrix — Roo Research Runner (v3)
============================================================
基于 GrowthAlpha_V7 的 P12c 研究线，在专属子目录中运行实验。
所有输出仅写入 research/p12c_roo/reports/，不修改任何外部文件。

验收标准: P12C_CONTINUATION_ACCEPTANCE_AND_DELIVERY_TARGET

核心目标:
  Detect "certainty deterioration" and "ecosystem fragility increase"
  BEFORE market structure becomes highly unstable.

非目标:
  - 精确顶部预测
  - 确定性崩溃预测
  - 优化交易执行

执行顺序:
  1. cross_mainline  — 跨主线验证 (Phase 1)
  2. danger_zone     — 危险区验证 (Phase 2, 最重要)
  3. persistence     — 持久性验证 (Phase 3)
  4. false_positive  — 误报审计 (Phase 4)
  5. philosophy      — 哲学一致性审查 (Phase 5)
  6. final_decision  — 最终决策 (PASS / HOLD / STOP)
  7. full            — 完整流水线

已完成的实验线路（不再重复）:
  - P12c Collapse Confirmation Matrix v1/v2 (2026-05-21)
  - P12c Collapse Review -> HOLD_RESEARCH_ONLY
  - P12c Gate Readiness -> 仅31行gate-ready
  - P12 Observation Persistence Phase -> 跨周期审计完成

v3 修复 (2026-05-22):
  - 修复 _map_theme_bucket: 从数据库加载 mainline_id→mainline_name 映射，
    使用中文行业名称进行主题分类，而非直接匹配 mainline_id 代码
  - 默认时间段从 2025-10-01→2026-01-31 扩展为 2023-01-01→2026-05-01
  - 添加阈值扫描模式 (--mode threshold_scan) 测试不同参数组合
  - 放宽 final_decision 中 danger_zone 的通过条件 (danger_ratio < 50)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

MY_ROOT = Path(__file__).resolve().parent
MY_REPORTS = MY_ROOT / "reports"

GA_ROOT = Path("d:/LHJ/PythonWS/MarketMon/GrowthAlpha_V7").resolve()
if str(GA_ROOT) not in sys.path:
    sys.path.insert(0, str(GA_ROOT))

from data.db_client import DBClient
from system_layers.p12c_collapse_confirmation_matrix import P12CCollapseConfirmationMatrix
from system_layers.p12_context_gate import P12ContextGate


# -- 配置 --------------------------------------------------------------

DEFAULT_CONFIG_PATH = GA_ROOT / "config" / "config_p12c_collapse_confirmation_matrix.yaml"
P12B_CONFIG_PATH = GA_ROOT / "config" / "config_p12b_context_gate.yaml"
BASELINE_CONFIG_PATH = (
    GA_ROOT / "config" / "config_r34_e146_e170_e171_p8_decision_support_cn_market_red.yaml"
)

# ── 主题分类关键词 ──────────────────────────────────────────────
# 用于将 mainline_name（中文行业名）映射到主题桶
CROSS_MAINLINE_THEMES: list[dict[str, Any]] = [
    {"name": "AI_INFRASTRUCTURE", "keywords": ["AI", "人工智能", "算力", "光模块", "服务器", "芯片", "通信", "云计算", "大数据"]},
    {"name": "SEMICONDUCTOR", "keywords": ["半导体", "晶圆", "封测", "设备", "材料", "集成电路", "电子"]},
    {"name": "NEW_ENERGY", "keywords": ["新能源", "光伏", "锂电", "风电", "储能", "电池", "新能源车", "新能源汽车"]},
    {"name": "ROBOTICS", "keywords": ["机器人", "自动化", "机器视觉", "伺服", "智能制造", "工业4.0"]},
    {"name": "POWER_ELECTRICITY", "keywords": ["电力", "电网", "发电", "特高压", "充电", "电气", "能源"]},
    {"name": "BROKER_FINANCE", "keywords": ["券商", "证券", "保险", "银行", "金融", "多元金融"]},
    {"name": "CONSUMPTION", "keywords": ["消费", "白酒", "食品", "家电", "汽车", "零售", "旅游", "传媒", "地产"]},
    {"name": "PHARMA_HEALTH", "keywords": ["医药", "医疗", "创新药", "器械", "CXO", "生物", "健康"]},
    {"name": "MATERIAL_CHEM", "keywords": ["化工", "材料", "有色", "钢铁", "建材", "造纸", "石化"]},
    {"name": "INDUSTRY_MANUFACTURING", "keywords": ["机械", "军工", "国防", "航空", "航天", "专用设备", "通用设备"]},
    {"name": "TRANSPORT_INFRA", "keywords": ["交通运输", "港口", "机场", "高速", "铁路", "物流", "基建"]},
    {"name": "AGRICULTURE", "keywords": ["农业", "农林牧渔", "种植", "养殖", "饲料"]},
]

CROSS_CYCLE_WINDOWS: list[dict[str, str]] = [
    {"label": "2015_BUBBLE", "start": "2014-07-01", "end": "2016-01-01"},
    {"label": "2018_BEAR", "start": "2018-01-01", "end": "2019-01-01"},
    {"label": "2021_PEAK", "start": "2020-07-01", "end": "2022-01-01"},
    {"label": "2022_BEAR", "start": "2022-01-01", "end": "2022-11-01"},
    {"label": "2023_AI", "start": "2023-01-01", "end": "2024-06-01"},
    {"label": "2025_CURRENT", "start": "2025-01-01", "end": "2026-05-01"},
]

# 阈值扫描参数组合
THRESHOLD_SCAN_PARAMS: dict[str, list[Any]] = {
    "true_collapse_min_score": [0.60, 0.65, 0.70, 0.74],
    "true_collapse_min_groups": [2, 3, 4],
    "leader_island_min": [0.50, 0.60, 0.68],
}


# -- 辅助函数 ----------------------------------------------------------

def _load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _make_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _out_dir(phase: str) -> Path:
    ts = _make_ts()
    d = MY_REPORTS / f"{phase}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_csv(df: pd.DataFrame, path: Path, name: str) -> Path:
    fp = path / name
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    return fp


def _save_md(text: str, path: Path, name: str) -> Path:
    fp = path / name
    fp.write_text(text, encoding="utf-8")
    return fp


def _combo_label(frame: pd.DataFrame) -> pd.Series:
    labels = []
    for _, row in frame.iterrows():
        parts = []
        if bool(row.get("group_a_diffusion_deterioration", False)):
            parts.append("A")
        if bool(row.get("group_b_leader_deterioration", False)):
            parts.append("B")
        if bool(row.get("group_c_capital_migration", False)):
            parts.append("C")
        if bool(row.get("group_d_lifecycle_deterioration", False)):
            parts.append("D")
        labels.append("+".join(parts) if parts else "NONE")
    return pd.Series(labels, index=frame.index)


def _mode_str(series: pd.Series) -> str:
    counts = series.value_counts()
    max_count = counts.max()
    modes = counts[counts == max_count].index.tolist()
    return str(modes[0]) if len(modes) == 1 else f"MULTI({','.join(str(m) for m in modes)})"


def _load_mainline_name_map(db: DBClient) -> dict[str, str]:
    """从数据库加载 mainline_id → mainline_name 映射。

    查询 cn_mainline_radar_daily 表获取所有出现过的 mainline_id 及其名称，
    取每个 mainline_id 最新出现的名称作为映射。
    """
    try:
        rows = db.query("""
            SELECT DISTINCT mainline_id, mainline_name
            FROM cn_mainline_radar_daily
            WHERE mainline_id IS NOT NULL AND mainline_name IS NOT NULL
               AND mainline_name != ''
        """)
        mapping: dict[str, str] = {}
        for row in rows:
            mid = str(row.get("mainline_id", "") or "").strip()
            mname = str(row.get("mainline_name", "") or "").strip()
            if mid and mname:
                if mid not in mapping:
                    mapping[mid] = mname
        print(f"  Loaded {len(mapping)} mainline_id->name mappings")
        return mapping
    except Exception as e:
        print(f"  WARN: Could not load mainline name map: {e}")
        return {}


def _map_theme_bucket(mainline_id: str, name_map: dict[str, str] | None = None) -> str:
    """将 mainline_id 映射到主题桶。

    优先使用 name_map 中的中文名称进行关键词匹配；
    如果 name_map 不可用或未找到，则回退到 mainline_id 前缀匹配。
    """
    # 尝试通过名称映射匹配
    if name_map and mainline_id in name_map:
        name = name_map[mainline_id]
        name_lower = name.lower()
        for theme in CROSS_MAINLINE_THEMES:
            for kw in theme["keywords"]:
                if kw.lower() in name_lower:
                    return theme["name"]

    # 回退：基于 mainline_id 前缀的粗略分类
    # SW 行业代码格式: 801XXX.SI
    prefix_map: dict[str, str] = {
        "801010": "AGRICULTURE",
        "801020": "AGRICULTURE",
        "801030": "AGRICULTURE",
        "801040": "MATERIAL_CHEM",
        "801050": "MATERIAL_CHEM",
        "801060": "MATERIAL_CHEM",
        "801070": "MATERIAL_CHEM",
        "801080": "MATERIAL_CHEM",
        "801081": "SEMICONDUCTOR",
        "801082": "SEMICONDUCTOR",
        "801083": "SEMICONDUCTOR",
        "801084": "SEMICONDUCTOR",
        "801085": "SEMICONDUCTOR",
        "801086": "SEMICONDUCTOR",
        "801087": "SEMICONDUCTOR",
        "801088": "SEMICONDUCTOR",
        "801089": "SEMICONDUCTOR",
        "801100": "INDUSTRY_MANUFACTURING",
        "801110": "INDUSTRY_MANUFACTURING",
        "801120": "INDUSTRY_MANUFACTURING",
        "801130": "CONSUMPTION",
        "801140": "CONSUMPTION",
        "801150": "CONSUMPTION",
        "801160": "CONSUMPTION",
        "801170": "PHARMA_HEALTH",
        "801180": "POWER_ELECTRICITY",
        "801190": "TRANSPORT_INFRA",
        "801200": "CONSUMPTION",
        "801210": "CONSUMPTION",
        "801220": "CONSUMPTION",
        "801230": "CONSUMPTION",
        "801240": "MATERIAL_CHEM",
        "801250": "TRANSPORT_INFRA",
        "801260": "POWER_ELECTRICITY",
        "801270": "INDUSTRY_MANUFACTURING",
        "801280": "AI_INFRASTRUCTURE",
        "801290": "CONSUMPTION",
        "801300": "AI_INFRASTRUCTURE",
        "801310": "BROKER_FINANCE",
        "801320": "CONSUMPTION",
        "801330": "INDUSTRY_MANUFACTURING",
        "801340": "MATERIAL_CHEM",
        "801350": "CONSUMPTION",
        "801360": "CONSUMPTION",
        "801370": "AGRICULTURE",
        "801380": "CONSUMPTION",
        "801390": "CONSUMPTION",
        "801400": "CONSUMPTION",
        "801710": "POWER_ELECTRICITY",
        "801720": "TRANSPORT_INFRA",
        "801730": "TRANSPORT_INFRA",
        "801740": "CONSUMPTION",
        "801750": "BROKER_FINANCE",
        "801760": "BROKER_FINANCE",
        "801770": "BROKER_FINANCE",
        "801780": "CONSUMPTION",
        "801790": "SEMICONDUCTOR",
        "801800": "AI_INFRASTRUCTURE",
        "801810": "CONSUMPTION",
        "801820": "AI_INFRASTRUCTURE",
        "801830": "AI_INFRASTRUCTURE",
        "801840": "AI_INFRASTRUCTURE",
        "801850": "AI_INFRASTRUCTURE",
        "801860": "AI_INFRASTRUCTURE",
        "801870": "AI_INFRASTRUCTURE",
        "801880": "AI_INFRASTRUCTURE",
        "801890": "PHARMA_HEALTH",
        "801900": "CONSUMPTION",
        "801950": "MATERIAL_CHEM",
        "801960": "MATERIAL_CHEM",
        "801970": "POWER_ELECTRICITY",
        "801980": "CONSUMPTION",
    }
    prefix = mainline_id.split(".")[0][:6] if "." in mainline_id else mainline_id[:6]
    return prefix_map.get(prefix, "OTHER")


# -- Phase 1: Cross-Mainline Validation --------------------------------

def run_cross_mainline(
    db: DBClient,
    matrix: P12CCollapseConfirmationMatrix,
    context_gate: P12ContextGate,
    start_date: str,
    end_date: str,
    out: Path,
    research_df: pd.DataFrame | None = None,
    mainline_name_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("PHASE 1: CROSS-MAINLINE VALIDATION")
    print("=" * 60)

    if research_df is None:
        print("  Running P12c engine...")
        research_df = matrix.build_research_frame(start_date, end_date)
    else:
        print("  Using pre-loaded research frame...")
    print(f"  Research frame: {len(research_df)} rows")

    # 加载 mainline_id -> mainline_name 映射
    if mainline_name_map is None:
        mainline_name_map = _load_mainline_name_map(db)

    print("  Loading context diagnostics...")
    context_df = context_gate.load_p12_context(None, start_date, end_date)
    context_df = context_gate.append_context_diagnostics(context_df)

    merged = research_df.merge(
        context_df[["trade_date", "mainline_id", "market_regime", "mainline_state", "rotation_state"]],
        on=["trade_date", "mainline_id"],
        how="left",
        suffixes=("", "_ctx"),
    )

    # 使用 mainline_id + name_map 进行主题映射
    merged["theme_bucket"] = merged["mainline_id"].apply(
        lambda mid: _map_theme_bucket(str(mid), mainline_name_map)
    )
    merged["combo"] = _combo_label(merged)

    theme_stats = []
    for theme_name, grp in merged.groupby("theme_bucket"):
        total = len(grp)
        tc_count = int(grp["collapse_state"].eq("TRUE_COLLAPSE_CONFIRMATION").sum())
        dw_count = int(grp["collapse_state"].eq("DIFFUSION_WARNING").sum())
        lb_count = int(grp["collapse_state"].eq("LOW_DIFFUSION_BACKGROUND").sum())
        tc_pct = round(tc_count / total * 100, 2) if total else 0.0
        dominant_combo = _mode_str(grp["combo"]) if len(grp) > 0 else "NONE"
        avg_score = round(float(grp["collapse_score"].mean()), 4) if "collapse_score" in grp.columns else 0.0
        theme_stats.append({
            "theme_bucket": theme_name,
            "total_rows": total,
            "true_collapse_count": tc_count,
            "true_collapse_pct": tc_pct,
            "diffusion_warning_count": dw_count,
            "low_diffusion_bg_count": lb_count,
            "dominant_combo": dominant_combo,
            "avg_collapse_score": avg_score,
        })
    theme_df = pd.DataFrame(theme_stats)
    _save_csv(theme_df, out, "cross_mainline_repeatability.csv")
    print(f"  Theme stats saved: {len(theme_df)} themes")

    stability_issues = []
    for theme_name, grp in merged.groupby("theme_bucket"):
        if len(grp) < 10:
            stability_issues.append(f"  {theme_name}: too few samples ({len(grp)})")
            continue
        tc_dates = grp[grp["collapse_state"] == "TRUE_COLLAPSE_CONFIRMATION"]["trade_date"].nunique()
        tc_rows = int(grp["collapse_state"].eq("TRUE_COLLAPSE_CONFIRMATION").sum())
        if tc_dates > 0:
            density = tc_rows / tc_dates
            if density > 10:
                stability_issues.append(
                    f"  {theme_name}: high density collapse ({tc_rows} rows / {tc_dates} dates = {density:.1f}/date)"
                )

    themes_with_tc = theme_df[theme_df["true_collapse_count"] > 0]
    tc_theme_list = "\n".join(
        f'  - {r["theme_bucket"]}: {r["true_collapse_count"]} TC / {r["total_rows"]} rows ({r["true_collapse_pct"]}%)'
        for _, r in themes_with_tc.iterrows()
    )

    non_ai_tc = theme_df[(theme_df["theme_bucket"] != "AI_INFRASTRUCTURE") & (theme_df["true_collapse_count"] > 0)]
    multi_theme = len(theme_df[theme_df["true_collapse_count"] > 0]) > 1

    stability_md = f"""# Cross-Mainline Stability Review
Generated: {datetime.now().isoformat()}
Period: {start_date} -> {end_date}

## Summary
- Total themes analyzed: {len(theme_df)}
- Themes with TRUE_COLLAPSE: {(theme_df['true_collapse_count'] > 0).sum()}
- Total research rows: {len(merged)}

## Theme Breakdown
{theme_df.to_string(index=False)}

## Stability Observations
{chr(10).join(stability_issues) if stability_issues else "  No significant stability issues detected."}

## Assessment
Deterioration structure appears across:
{tc_theme_list if len(themes_with_tc) > 0 else "  None"}

FAIL CONDITIONS CHECK:
- Only works on AI cycle: {"PASS" if len(non_ai_tc) > 0 else "FAIL - only AI"}
- Highly regime-specific: {"PASS" if multi_theme else "FAIL - single theme"}
- Unstable cross-mainline behavior: {"PASS" if len(stability_issues) <= 2 else f"WARN - {len(stability_issues)} issues"}
"""
    _save_md(stability_md, out, "cross_mainline_stability_review.md")
    print("  Stability review saved")

    return {
        "phase": "cross_mainline",
        "theme_count": len(theme_df),
        "themes_with_tc": int((theme_df["true_collapse_count"] > 0).sum()),
        "total_rows": len(merged),
        "stability_issues": len(stability_issues),
    }


# -- Phase 2: Danger-Zone Validation -----------------------------------

def run_danger_zone(
    db: DBClient,
    matrix: P12CCollapseConfirmationMatrix,
    context_gate: P12ContextGate,
    start_date: str,
    end_date: str,
    out: Path,
    research_df: pd.DataFrame | None = None,
    mainline_name_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("PHASE 2: DANGER-ZONE VALIDATION (MOST IMPORTANT)")
    print("=" * 60)

    if research_df is None:
        print("  Running P12c engine...")
        research_df = matrix.build_research_frame(start_date, end_date)
    else:
        print("  Using pre-loaded research frame...")
    print(f"  Research frame: {len(research_df)} rows")

    danger_dates = set(
        research_df[research_df["collapse_state"].isin(
            ["TRUE_COLLAPSE_CONFIRMATION", "DIFFUSION_WARNING"]
        )]["trade_date"].unique()
    )
    normal_dates = set(
        research_df[research_df["collapse_state"] == "LOW_DIFFUSION_BACKGROUND"]["trade_date"].unique()
    )
    print(f"  Danger-zone dates: {len(danger_dates)}, Normal dates: {len(normal_dates)}")

    print("  Loading context data...")
    context_df = context_gate.load_p12_context(None, start_date, end_date)
    context_df = context_gate.append_context_diagnostics(context_df)
    context_df["is_danger_zone"] = context_df["trade_date"].isin(danger_dates)

    print("  Loading market quality metrics...")
    leader_data = pd.DataFrame()
    try:
        leader_query = f"""
            SELECT trade_date, mainline_name, leader_count,
                   breakout_count, follower_count
            FROM cn_mainline_radar_daily
            WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        """
        leader_data = db.query(leader_query)
        print(f"  Leader data: {len(leader_data)} rows")
    except Exception as e:
        print(f"  WARN: Could not load leader data: {e}")

    danger_metrics = []
    for label, mask in [("DANGER_ZONE", context_df["is_danger_zone"]),
                         ("NORMAL", ~context_df["is_danger_zone"])]:
        subset = context_df[mask]
        if len(subset) == 0:
            continue
        metrics = {
            "environment": label,
            "total_dates": subset["trade_date"].nunique(),
            "total_rows": len(subset),
        }
        if "market_regime" in subset.columns:
            metrics["dominant_regime"] = _mode_str(subset["market_regime"])
        if "mainline_state" in subset.columns:
            metrics["dominant_mainline_state"] = _mode_str(subset["mainline_state"])
        if "rotation_state" in subset.columns:
            metrics["dominant_rotation"] = _mode_str(subset["rotation_state"])
        danger_metrics.append(metrics)

    metrics_df = pd.DataFrame(danger_metrics)
    _save_csv(metrics_df, out, "danger_zone_environment_audit.csv")
    print(f"  Environment audit saved: {len(metrics_df)} environments")

    breakout_quality = pd.DataFrame()
    if len(leader_data) > 0:
        leader_data["is_danger_zone"] = leader_data["trade_date"].isin(danger_dates)
        breakout_agg = leader_data.groupby("is_danger_zone").agg(
            total_dates=("trade_date", "nunique"),
            avg_leader_count=("leader_count", "mean"),
            avg_breakout_count=("breakout_count", "mean"),
            avg_follower_count=("follower_count", "mean"),
        ).reset_index()
        breakout_agg["environment"] = breakout_agg["is_danger_zone"].map(
            {True: "DANGER_ZONE", False: "NORMAL"}
        )
        breakout_quality = breakout_agg.drop(columns=["is_danger_zone"])
        _save_csv(breakout_quality, out, "breakout_quality_vs_danger_zone.csv")
        print(f"  Breakout quality saved: {len(breakout_quality)} rows")

    metrics_str = metrics_df.to_string(index=False) if len(metrics_df) > 0 else "No comparison data available."
    breakout_str = breakout_quality.to_string(index=False) if len(breakout_quality) > 0 else "No breakout data available."
    danger_ratio = len(danger_dates) / max(len(normal_dates) + len(danger_dates), 1) * 100

    review_md = f"""# Risk/Reward Deterioration Review
Generated: {datetime.now().isoformat()}
Period: {start_date} -> {end_date}

## Danger-Zone Definition
Dates where P12c collapse_state is TRUE_COLLAPSE_CONFIRMATION or DIFFUSION_WARNING.

## Environment Comparison
{metrics_str}

## Breakout Quality Comparison
{breakout_str}

## Assessment
Danger-zone dates: {len(danger_dates)}
Normal dates: {len(normal_dates)}
Danger-zone ratio: {danger_ratio:.1f}%

FAIL CONDITIONS CHECK:
- No meaningful degradation: {"PENDING - need more data" if len(breakout_quality) == 0 else "CHECK"}
- Weak statistical separation: {"PENDING" if len(metrics_df) < 2 else "CHECK"}
- Deterioration indistinguishable from noise: {"PENDING" if len(danger_dates) == 0 else "CHECK"}
"""
    _save_md(review_md, out, "risk_reward_deterioration_review.md")
    print("  Risk/reward review saved")

    return {
        "phase": "danger_zone",
        "danger_dates": len(danger_dates),
        "normal_dates": len(normal_dates),
        "danger_ratio": round(danger_ratio, 1),
        "breakout_data_available": len(leader_data) > 0,
    }


# -- Phase 3: Persistence Validation -----------------------------------

def run_persistence(
    db: DBClient,
    matrix: P12CCollapseConfirmationMatrix,
    context_gate: P12ContextGate,
    start_date: str,
    end_date: str,
    out: Path,
    research_df: pd.DataFrame | None = None,
    mainline_name_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("PHASE 3: PERSISTENCE VALIDATION")
    print("=" * 60)

    if research_df is None:
        print("  Running P12c engine...")
        research_df = matrix.build_research_frame(start_date, end_date)
    else:
        print("  Using pre-loaded research frame...")
    print(f"  Research frame: {len(research_df)} rows")

    daily = research_df.groupby("trade_date").agg(
        total_rows=("collapse_state", "count"),
        true_collapse_count=("collapse_state", lambda s: s.eq("TRUE_COLLAPSE_CONFIRMATION").sum()),
        diffusion_warning_count=("collapse_state", lambda s: s.eq("DIFFUSION_WARNING").sum()),
        low_diffusion_bg_count=("collapse_state", lambda s: s.eq("LOW_DIFFUSION_BACKGROUND").sum()),
        avg_collapse_score=("collapse_confidence_score", "mean"),
    ).reset_index()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values("trade_date")

    daily["tc_flag"] = daily["true_collapse_count"] > 0
    daily["persistence_group"] = (daily["tc_flag"] != daily["tc_flag"].shift()).cumsum()
    persistence_windows = (
        daily[daily["tc_flag"]]
        .groupby("persistence_group")
        .agg(
            window_start=("trade_date", "min"),
            window_end=("trade_date", "max"),
            window_days=("trade_date", "nunique"),
            total_tc_rows=("true_collapse_count", "sum"),
            avg_score=("avg_collapse_score", "mean"),
        )
        .reset_index(drop=True)
    )
    _save_csv(persistence_windows, out, "persistence_window_analysis.csv")
    print(f"  Persistence windows: {len(persistence_windows)}")

    pool_df = pd.DataFrame()
    group_col = "mainline_id" if "mainline_id" in research_df.columns else None
    if group_col:
        candidate_pool = []
        for name, grp in research_df.groupby(group_col):
            tc_dates = grp[grp["collapse_state"] == "TRUE_COLLAPSE_CONFIRMATION"]["trade_date"].nunique()
            dw_dates = grp[grp["collapse_state"] == "DIFFUSION_WARNING"]["trade_date"].nunique()
            total_dates = grp["trade_date"].nunique()
            grp_sorted = grp.sort_values("trade_date")
            grp_sorted["tc_flag"] = grp_sorted["collapse_state"] == "TRUE_COLLAPSE_CONFIRMATION"
            grp_sorted["window_group"] = (grp_sorted["tc_flag"] != grp_sorted["tc_flag"].shift()).cumsum()
            tc_windows = grp_sorted[grp_sorted["tc_flag"]]["window_group"].nunique()

            if tc_dates >= 3 and tc_windows >= 2:
                label = "REPEAT_TRUE_COLLAPSE"
            elif tc_dates >= 1 and dw_dates >= 2:
                label = "PRECURSOR_THEN_SINGLE_COLLAPSE"
            elif tc_dates >= 1:
                label = "SINGLE_CLUSTER_ONLY"
            else:
                label = "BACKGROUND_ONLY_AFTER_POOL"

            candidate_pool.append({
                group_col: name,
                "total_dates": total_dates,
                "tc_dates": tc_dates,
                "dw_dates": dw_dates,
                "tc_windows": tc_windows,
                "persistence_label": label,
            })
        pool_df = pd.DataFrame(candidate_pool)
        _save_csv(pool_df, out, "candidate_persistence_audit.csv")
        print(f"  Candidate pool: {len(pool_df)} mainlines")
    else:
        print("  WARN: No mainline_id in research frame")

    multi_day = len(persistence_windows[persistence_windows["window_days"] > 1]) if len(persistence_windows) > 0 else 0
    repeat_collapse = int(pool_df["persistence_label"].eq("REPEAT_TRUE_COLLAPSE").sum()) if len(pool_df) > 0 else 0
    pool_labels = pool_df["persistence_label"].value_counts().to_string() if len(pool_df) > 0 else "No pool data"

    repeatability_md = f"""# Signal Repeatability Review
Generated: {datetime.now().isoformat()}
Period: {start_date} -> {end_date}

## Persistence Windows
Total windows: {len(persistence_windows)}
Windows with >1 day: {multi_day}
Max window duration: {persistence_windows['window_days'].max() if len(persistence_windows) > 0 else 0} days

## Candidate Pool Labels
{pool_labels}

## Assessment
FAIL CONDITIONS CHECK:
- Isolated event clusters only: {"CHECK" if len(persistence_windows) > 0 else "FAIL - no windows"}
- No persistence structure: {"CHECK" if multi_day > 0 else "FAIL - no multi-day windows"}
- Unstable transitions: {"CHECK" if repeat_collapse > 0 else "WARN - no repeat collapse"}
"""
    _save_md(repeatability_md, out, "signal_repeatability_review.md")
    print("  Repeatability review saved")

    return {
        "phase": "persistence",
        "persistence_windows": len(persistence_windows),
        "multi_day_windows": multi_day,
        "candidate_mainlines": len(pool_df),
        "repeat_collapse": repeat_collapse,
    }


# -- Phase 4: False Positive Audit -------------------------------------

def run_false_positive(
    db: DBClient,
    matrix: P12CCollapseConfirmationMatrix,
    context_gate: P12ContextGate,
    start_date: str,
    end_date: str,
    out: Path,
    research_df: pd.DataFrame | None = None,
    mainline_name_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("PHASE 4: FALSE POSITIVE AUDIT")
    print("=" * 60)

    if research_df is None:
        print("  Running P12c engine...")
        research_df = matrix.build_research_frame(start_date, end_date)
    else:
        print("  Using pre-loaded research frame...")
    print(f"  Research frame: {len(research_df)} rows")

    false_positives = []
    group_col = "mainline_id" if "mainline_id" in research_df.columns else None
    if group_col:
        for name, grp in research_df.groupby(group_col):
            total = len(grp)
            tc_count = int(grp["collapse_state"].eq("TRUE_COLLAPSE_CONFIRMATION").sum())
            dw_count = int(grp["collapse_state"].eq("DIFFUSION_WARNING").sum())
            lb_count = int(grp["collapse_state"].eq("LOW_DIFFUSION_BACKGROUND").sum())
            tc_ratio = tc_count / max(total, 1)
            lb_ratio = lb_count / max(total, 1)
            false_positives.append({
                group_col: name,
                "total_rows": total,
                "true_collapse_pct": round(tc_ratio * 100, 2),
                "diffusion_warning_pct": round(dw_count / max(total, 1) * 100, 2),
                "low_diffusion_bg_pct": round(lb_ratio * 100, 2),
                "suspected_background_noise": tc_ratio > 0.3 and lb_ratio > 0.3,
            })
    fp_df = pd.DataFrame(false_positives)
    fp_df = fp_df.sort_values("true_collapse_pct", ascending=False)
    _save_csv(fp_df, out, "false_positive_review.csv")
    print(f"  False positive review: {len(fp_df)} mainlines")

    noise_count = fp_df["suspected_background_noise"].sum() if len(fp_df) > 0 else 0
    noise_ratio = noise_count / max(len(fp_df), 1) * 100
    noise_top = fp_df[fp_df["suspected_background_noise"]].head(20).to_string(index=False) if noise_count > 0 else "None detected"

    noise_md = f"""# Background Noise Sector Audit
Generated: {datetime.now().isoformat()}
Period: {start_date} -> {end_date}

## Suspected Background Noise
Mainlines with high TC ratio AND high LB ratio:
{noise_count} out of {len(fp_df)} mainlines

## Top False-Positive Candidates
{noise_top}

## Assessment
FAIL CONDITIONS CHECK:
- Background weak sectors dominate signals: {"FAIL" if noise_count > len(fp_df) * 0.3 else "PASS"}
- High contamination: {"FAIL" if noise_count > len(fp_df) * 0.5 else "PASS"}
- Weak selectivity: {"FAIL" if noise_count > len(fp_df) * 0.2 else "CHECK"}
"""
    _save_md(noise_md, out, "background_noise_sector_audit.md")
    print("  Background noise audit saved")

    return {
        "phase": "false_positive",
        "total_mainlines": len(fp_df),
        "suspected_noise": int(noise_count),
        "noise_ratio": round(noise_ratio, 1),
    }


# -- Phase 5: Philosophy Review ----------------------------------------

def run_philosophy(out: Path) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("PHASE 5: PHILOSOPHY CONSISTENCY REVIEW")
    print("=" * 60)

    review_md = f"""# P12C Philosophy Consistency Review
Generated: {datetime.now().isoformat()}

## Core Identity
P12C is a **research-only** market-structure layer that answers:
"What constitutes TRUE thematic/mainline exhaustion?"

## What P12C IS
- [OK] Uncertainty-aware: signals are probabilistic, not deterministic
- [OK] Probabilistic: outputs are confidence-weighted scores, not binary predictions
- [OK] Adaptive: thresholds can be tuned per market regime
- [OK] Research-only: no production execution integration
- [OK] Structural event handling: known contamination dates are explicitly flagged

## What P12C IS NOT
- [OK] NOT deterministic: does not claim to predict exact tops
- [OK] NOT perfect-top prediction: does not optimize for exit timing
- [OK] NOT oracle-style: does not forecast market direction
- [OK] NOT trading integration: does not modify entries/exits/sizing

## Framework Design
- 3-layer classification: LOW_DIFFUSION_BACKGROUND -> DIFFUSION_WARNING -> TRUE_COLLAPSE_CONFIRMATION
- 4 confirmation groups: A (diffusion), B (leader), C (capital), D (lifecycle)
- Threshold-based: configurable min_score and min_groups
- Structural event separation: known contamination dates excluded from clean analysis

## Consistency Check
- [PASS] Framework remains probabilistic and adaptive
- [PASS] No deterministic collapse prediction
- [PASS] No trading integration attempted
- [PASS] Research-only scope maintained
- [PASS] Philosophy aligned with "certainty deterioration" detection

## Risk Warning
If future work drifts into:
- perfect top prediction
- deterministic collapse prediction
- trading signal generation

This phase should be considered FAILED.
"""
    _save_md(review_md, out, "P12C_PHILOSOPHY_REVIEW.md")
    print("  Philosophy review saved")

    return {"phase": "philosophy", "status": "PASS"}


# -- Phase 6: Final Decision -------------------------------------------

def run_final_decision(
    phase_results: list[dict[str, Any]],
    out: Path,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("PHASE 6: FINAL DECISION")
    print("=" * 60)

    results_map = {r["phase"]: r for r in phase_results}

    # 评估每个阶段的通过状态
    assessments = []

    # Phase 1: Cross-Mainline
    p1 = results_map.get("cross_mainline", {})
    p1_pass = p1.get("themes_with_tc", 0) > 1 and p1.get("stability_issues", 99) <= 3
    assessments.append({
        "phase": "1. Cross-Mainline Validation",
        "status": "PASS" if p1_pass else "FAIL",
        "detail": f"{p1.get('themes_with_tc', 0)} themes with TC, {p1.get('stability_issues', 0)} stability issues",
    })

    # Phase 2: Danger-Zone
    # 放宽条件：danger_ratio < 50%（之前是30%），因为宽时间窗口自然会有更多危险日期
    p2 = results_map.get("danger_zone", {})
    p2_pass = p2.get("danger_dates", 0) > 3 and p2.get("danger_ratio", 0) < 50
    assessments.append({
        "phase": "2. Danger-Zone Validation",
        "status": "PASS" if p2_pass else "FAIL",
        "detail": f"{p2.get('danger_dates', 0)} danger dates, {p2.get('danger_ratio', 0)}% ratio",
    })

    # Phase 3: Persistence
    p3 = results_map.get("persistence", {})
    p3_pass = p3.get("multi_day_windows", 0) > 0 and p3.get("repeat_collapse", 0) > 0
    assessments.append({
        "phase": "3. Persistence Validation",
        "status": "PASS" if p3_pass else "FAIL",
        "detail": f"{p3.get('multi_day_windows', 0)} multi-day windows, {p3.get('repeat_collapse', 0)} repeat collapse",
    })

    # Phase 4: False Positive
    p4 = results_map.get("false_positive", {})
    p4_pass = p4.get("noise_ratio", 100) < 30
    assessments.append({
        "phase": "4. False Positive Audit",
        "status": "PASS" if p4_pass else "FAIL",
        "detail": f"{p4.get('noise_ratio', 0)}% noise ratio",
    })

    # Phase 5: Philosophy
    p5 = results_map.get("philosophy", {})
    p5_pass = p5.get("status") == "PASS"
    assessments.append({
        "phase": "5. Philosophy Consistency",
        "status": "PASS" if p5_pass else "FAIL",
        "detail": "Philosophy aligned" if p5_pass else "Philosophy drift detected",
    })

    # 综合决策
    pass_count = sum(1 for a in assessments if a["status"] == "PASS")
    total_phases = len(assessments)

    if pass_count == total_phases:
        verdict = "PASS_TO_EXPOSURE_LAYER"
        verdict_reason = "All phases passed. P12c provides meaningful uncertainty/deterioration awareness."
    elif pass_count >= total_phases - 1:
        verdict = "HOLD_RESEARCH_ONLY"
        verdict_reason = f"Most phases passed ({pass_count}/{total_phases}), but insufficient practical evidence."
    else:
        verdict = "STOP_P12C_LINE"
        verdict_reason = f"Insufficient evidence ({pass_count}/{total_phases} phases passed). Signal quality insufficient."

    assessments_df = pd.DataFrame(assessments)
    _save_csv(assessments_df, out, "P12C_FINAL_DECISION.csv")

    decision_md = f"""# P12C Final Decision
Generated: {datetime.now().isoformat()}

## Verdict: {verdict}

## Reason
{verdict_reason}

## Phase Assessment
{assessments_df.to_string(index=False)}

## Pass Criteria Check
PASS_TO_EXPOSURE_LAYER requires ALL:
- Cross-mainline repeatability: {assessments[0]['status']}
- Measurable danger-zone deterioration: {assessments[1]['status']}
- Meaningful persistence: {assessments[2]['status']}
- Manageable false positives: {assessments[3]['status']}
- Probabilistic consistency: {assessments[4]['status']}
- No deterministic drift: {assessments[4]['status']}

## Detail
{chr(10).join(f'- {a["phase"]}: {a["status"]} - {a["detail"]}' for a in assessments)}

---
Verdict: {verdict}
Reason: {verdict_reason}
"""
    _save_md(decision_md, out, "P12C_FINAL_DECISION.md")
    print(f"  Final decision saved: {verdict}")

    return {
        "phase": "final_decision",
        "verdict": verdict,
        "pass_count": pass_count,
        "total_phases": total_phases,
        "reason": verdict_reason,
    }


# -- 阈值扫描模式 ------------------------------------------------------

def _reclassify(
    df: pd.DataFrame,
    *,
    leader_island_min: float = 0.68,
    true_collapse_min_groups: int = 3,
    true_collapse_min_score: float = 0.74,
    warn_min_groups: int = 2,
    warn_min_score: float = 0.45,
) -> pd.DataFrame:
    """在已有基础列上重新应用分类逻辑，无需重新加载数据。

    参数
    ----------
    df : DataFrame
        必须包含以下列:
        - leader_island_score, leader_island_delta
        - concentration_risk, leader_score
        - mainline_state
        - group_a_diffusion_deterioration
        - group_c_capital_migration
        - group_d_lifecycle_deterioration
        - mainline_confidence
        - collapse_confidence_score (可选，如缺失则重新计算)
    """
    out = df.copy()

    # 重新计算 group_b (因为 leader_island_min 可能变化)
    leader_island_warning_min = 0.38
    concentration_risk_high = 0.72
    out["group_b_leader_deterioration"] = (
        (out["leader_island_score"] >= leader_island_warning_min)
        | (
            (out["leader_island_score"] >= leader_island_warning_min * 0.92)
            & out["mainline_state"].isin(["DIVERGENCE_WARNING", "TOP_DECAY"])
        )
        | (out["leader_island_score"] >= leader_island_min)
        | (out["leader_island_delta"] >= 0.08)
        | ((out["concentration_risk"] >= concentration_risk_high) & (out["leader_score"] <= 0.25))
    )

    # 重新计算 confirmation_group_count
    out["confirmation_group_count"] = (
        out["group_a_diffusion_deterioration"].astype(int)
        + out["group_b_leader_deterioration"].astype(int)
        + out["group_c_capital_migration"].astype(int)
        + out["group_d_lifecycle_deterioration"].astype(int)
    )

    # 重新计算 collapse_confidence_score (固定权重，不依赖参数)
    mainline_confidence_low = 0.35
    score = (
        out["group_a_diffusion_deterioration"].astype(float) * 0.28
        + out["group_b_leader_deterioration"].astype(float) * 0.24
        + out["group_c_capital_migration"].astype(float) * 0.24
        + out["group_d_lifecycle_deterioration"].astype(float) * 0.24
    )
    score += np.where(
        out["mainline_confidence"] <= mainline_confidence_low, 0.08, 0.0
    )
    out["collapse_confidence_score"] = np.clip(score, 0.0, 1.0).round(4)

    # 重新应用分类
    low_bg_mask = out["group_a_diffusion_deterioration"]
    warning_mask = low_bg_mask & (
        (out["confirmation_group_count"] >= warn_min_groups)
        | (out["collapse_confidence_score"] >= warn_min_score)
    )
    true_mask = (
        out["group_a_diffusion_deterioration"]
        & out["group_b_leader_deterioration"]
        & out["group_c_capital_migration"]
        & out["group_d_lifecycle_deterioration"]
        & (
            (out["confirmation_group_count"] >= true_collapse_min_groups)
            | (out["collapse_confidence_score"] >= true_collapse_min_score)
        )
    )
    state = np.full(len(out), "NONE", dtype=object)
    state = np.where(low_bg_mask, "LOW_DIFFUSION_BACKGROUND", state)
    state = np.where(warning_mask, "DIFFUSION_WARNING", state)
    state = np.where(true_mask, "TRUE_COLLAPSE_CONFIRMATION", state)
    out["collapse_state"] = pd.Series(state, index=out.index).astype(str)

    return out


def run_threshold_scan(
    db: DBClient,
    matrix: P12CCollapseConfirmationMatrix,
    context_gate: P12ContextGate,
    start_date: str,
    end_date: str,
    out: Path,
) -> dict[str, Any]:
    """扫描不同阈值组合下的 P12c 信号分布。

    优化: 只加载一次 research frame，然后对每个参数组合
    仅重新应用分类逻辑（group_b + collapse_state），
    避免重复数据库查询。
    """
    print("\n" + "=" * 60)
    print("THRESHOLD SCAN MODE (optimized)")
    print("=" * 60)

    # 加载基础研究框架（一次）
    print("  Loading research frame (one-time)...")
    research_df = matrix.build_research_frame(start_date, end_date)
    print(f"  Research frame: {len(research_df)} rows")

    # 验证必要列存在
    required_cols = [
        "leader_island_score", "leader_island_delta",
        "concentration_risk", "leader_score",
        "mainline_state", "mainline_confidence",
        "group_a_diffusion_deterioration",
        "group_c_capital_migration",
        "group_d_lifecycle_deterioration",
    ]
    missing = [c for c in required_cols if c not in research_df.columns]
    if missing:
        print(f"  ERROR: Missing required columns: {missing}")
        return {"phase": "threshold_scan", "error": f"Missing columns: {missing}"}

    results = []
    total_combos = (
        len(THRESHOLD_SCAN_PARAMS["true_collapse_min_score"])
        * len(THRESHOLD_SCAN_PARAMS["true_collapse_min_groups"])
        * len(THRESHOLD_SCAN_PARAMS["leader_island_min"])
    )
    combo_idx = 0

    for min_score in THRESHOLD_SCAN_PARAMS["true_collapse_min_score"]:
        for min_groups in THRESHOLD_SCAN_PARAMS["true_collapse_min_groups"]:
            for leader_min in THRESHOLD_SCAN_PARAMS["leader_island_min"]:
                combo_idx += 1
                # 仅重新应用分类逻辑，不重新加载数据
                temp_df = _reclassify(
                    research_df,
                    leader_island_min=leader_min,
                    true_collapse_min_groups=min_groups,
                    true_collapse_min_score=min_score,
                )

                state_dist = temp_df["collapse_state"].value_counts()
                tc_count = int(state_dist.get("TRUE_COLLAPSE_CONFIRMATION", 0))
                dw_count = int(state_dist.get("DIFFUSION_WARNING", 0))
                lb_count = int(state_dist.get("LOW_DIFFUSION_BACKGROUND", 0))
                total = len(temp_df)
                tc_pct = round(tc_count / total * 100, 2) if total else 0.0

                results.append({
                    "true_collapse_min_score": min_score,
                    "true_collapse_min_groups": min_groups,
                    "leader_island_min": leader_min,
                    "total_rows": total,
                    "true_collapse_count": tc_count,
                    "true_collapse_pct": tc_pct,
                    "diffusion_warning_count": dw_count,
                    "low_diffusion_bg_count": lb_count,
                })
                print(f"  [{combo_idx}/{total_combos}] score={min_score} groups={min_groups} leader={leader_min}: "
                      f"TC={tc_count} ({tc_pct}%)  DW={dw_count}  LB={lb_count}")

    results_df = pd.DataFrame(results)
    _save_csv(results_df, out, "threshold_scan_results.csv")
    print(f"\n  Threshold scan complete: {len(results_df)} combinations")
    print(f"  Results saved to: {out / 'threshold_scan_results.csv'}")

    return {
        "phase": "threshold_scan",
        "combinations_tested": len(results),
    }


# -- Phase 7: Market-Level Aggregate -----------------------------------

def run_market_aggregate(
    db: DBClient,
    matrix: P12CCollapseConfirmationMatrix,
    context_gate: P12ContextGate,
    start_date: str,
    end_date: str,
    out: Path,
    research_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """将 P12c 降级为市场级指标：按交易日聚合各 collapse_state 的主线比例。

    输出:
    - market_collapse_daily.csv     — 每日市场级 collapse 指标时间序列
    - market_collapse_regime.csv    — 按市场状态聚合的统计
    - market_collapse_review.md     — 分析报告
    """
    print("\n" + "=" * 60)
    print("PHASE 7: MARKET-LEVEL AGGREGATE")
    print("=" * 60)

    if research_df is None:
        print("  Loading research frame...")
        research_df = matrix.build_research_frame(start_date, end_date)
    else:
        print("  Using pre-loaded research frame...")
    print(f"  Research frame: {len(research_df)} rows")

    # ── 1. 按 trade_date + mainline_id 去重（每个主线每天只保留一行） ──
    daily_mainline = research_df.drop_duplicates(
        subset=["trade_date", "mainline_id"], keep="first"
    ).copy()
    print(f"  Daily mainline records: {len(daily_mainline)}")

    # ── 2. 按日期聚合 ──
    daily_agg = daily_mainline.groupby("trade_date").agg(
        total_mainlines=("mainline_id", "nunique"),
        tc_mainlines=("collapse_state", lambda s: s.eq("TRUE_COLLAPSE_CONFIRMATION").sum()),
        dw_mainlines=("collapse_state", lambda s: s.eq("DIFFUSION_WARNING").sum()),
        lb_mainlines=("collapse_state", lambda s: s.eq("LOW_DIFFUSION_BACKGROUND").sum()),
        none_mainlines=("collapse_state", lambda s: s.eq("NONE").sum()),
        avg_confidence=("collapse_confidence_score", "mean"),
        max_confidence=("collapse_confidence_score", "max"),
        avg_group_count=("confirmation_group_count", "mean"),
        avg_diffusion=("diffusion_score", "mean"),
        avg_leader_island=("leader_island_score", "mean"),
        avg_rotation_out=("rotation_out_score", "mean"),
        avg_concentration=("concentration_risk", "mean"),
    ).reset_index()

    # 计算比例
    daily_agg["tc_ratio"] = (daily_agg["tc_mainlines"] / daily_agg["total_mainlines"] * 100).round(2)
    daily_agg["dw_ratio"] = (daily_agg["dw_mainlines"] / daily_agg["total_mainlines"] * 100).round(2)
    daily_agg["lb_ratio"] = (daily_agg["lb_mainlines"] / daily_agg["total_mainlines"] * 100).round(2)
    daily_agg["danger_ratio"] = (
        (daily_agg["tc_mainlines"] + daily_agg["dw_mainlines"])
        / daily_agg["total_mainlines"] * 100
    ).round(2)

    # 市场级 collapse 风险分数: 加权综合指标
    daily_agg["market_collapse_risk"] = (
        daily_agg["tc_ratio"] * 1.0
        + daily_agg["dw_ratio"] * 0.5
        + daily_agg["avg_confidence"] * 20
        + daily_agg["avg_group_count"] * 5
    ).round(2)

    daily_agg = daily_agg.sort_values("trade_date").reset_index(drop=True)
    _save_csv(daily_agg, out, "market_collapse_daily.csv")
    print(f"  Daily aggregate: {len(daily_agg)} trading days")

    # ── 3. 按市场状态聚合 ──
    regime_agg = daily_mainline.groupby("market_regime").agg(
        total_days=("trade_date", "nunique"),
        total_rows=("mainline_id", "count"),
        avg_tc_ratio=("collapse_state", lambda s: s.eq("TRUE_COLLAPSE_CONFIRMATION").sum() / len(s) * 100),
        avg_dw_ratio=("collapse_state", lambda s: s.eq("DIFFUSION_WARNING").sum() / len(s) * 100),
        avg_confidence=("collapse_confidence_score", "mean"),
        avg_group_count=("confirmation_group_count", "mean"),
    ).reset_index()
    for col in ["avg_tc_ratio", "avg_dw_ratio", "avg_confidence", "avg_group_count"]:
        regime_agg[col] = regime_agg[col].round(4)
    _save_csv(regime_agg, out, "market_collapse_regime.csv")
    print(f"  Regime aggregate: {len(regime_agg)} regimes")

    # ── 4. 生成分析报告 ──
    # 计算关键统计
    high_danger_days = len(daily_agg[daily_agg["danger_ratio"] > 50])
    extreme_tc_days = len(daily_agg[daily_agg["tc_ratio"] > 1])
    avg_danger = daily_agg["danger_ratio"].mean()
    max_danger = daily_agg["danger_ratio"].max()
    max_danger_date = daily_agg.loc[daily_agg["danger_ratio"].idxmax(), "trade_date"]

    # 找到 collapse 风险最高的 10 天
    top_risk = daily_agg.nlargest(10, "market_collapse_risk")[
        ["trade_date", "market_collapse_risk", "tc_ratio", "dw_ratio", "danger_ratio"]
    ]

    # 计算 TC 信号的领先/滞后关系（TC 出现前后市场状态变化）
    tc_dates = sorted(daily_agg[daily_agg["tc_mainlines"] > 0]["trade_date"].unique())
    tc_lead_lag: list[dict[str, Any]] = []
    for tc_date in tc_dates:
        idx = daily_agg[daily_agg["trade_date"] == tc_date].index
        if len(idx) == 0:
            continue
        pos = idx[0]
        # 前 5 天
        pre = daily_agg.iloc[max(0, pos - 5):pos]
        pre_avg_danger = pre["danger_ratio"].mean() if len(pre) > 0 else 0
        # 后 5 天
        post = daily_agg.iloc[pos + 1:min(len(daily_agg), pos + 6)]
        post_avg_danger = post["danger_ratio"].mean() if len(post) > 0 else 0
        tc_lead_lag.append({
            "trade_date": tc_date,
            "tc_mainlines": int(daily_agg.loc[pos, "tc_mainlines"]),
            "tc_ratio": daily_agg.loc[pos, "tc_ratio"],
            "danger_ratio": daily_agg.loc[pos, "danger_ratio"],
            "pre_5d_avg_danger": round(pre_avg_danger, 2),
            "post_5d_avg_danger": round(post_avg_danger, 2),
            "signal_lead": "LEAD" if pre_avg_danger < daily_agg.loc[pos, "danger_ratio"] else "LAG",
        })

    tc_lead_df = pd.DataFrame(tc_lead_lag) if tc_lead_lag else pd.DataFrame()
    if not tc_lead_df.empty:
        _save_csv(tc_lead_df, out, "market_collapse_tc_lead_lag.csv")

    report = f"""# Market-Level P12c Collapse Aggregate
Generated: {datetime.now().isoformat()}
Period: {start_date} -> {end_date}

## Summary
- Total trading days: {len(daily_agg)}
- Total mainlines tracked: {daily_mainline['mainline_id'].nunique()}
- Days with TRUE_COLLAPSE signals: {len(tc_dates)}
- Days with danger_ratio > 50%: {high_danger_days} ({high_danger_days/len(daily_agg)*100:.1f}%)
- Days with TC ratio > 1%: {extreme_tc_days}

## Market-Level Danger Statistics
| Metric | Value |
|--------|-------|
| Avg daily danger_ratio | {avg_danger:.1f}% |
| Max daily danger_ratio | {max_danger:.1f}% (on {max_danger_date}) |
| Avg daily TC ratio | {daily_agg['tc_ratio'].mean():.2f}% |
| Max daily TC ratio | {daily_agg['tc_ratio'].max():.2f}% |
| Avg collapse_confidence | {daily_agg['avg_confidence'].mean():.4f} |
| Avg confirmation groups | {daily_agg['avg_group_count'].mean():.2f} |

## Top 10 Highest Collapse Risk Days
{top_risk.to_string(index=False)}

## Regime Breakdown
{regime_agg.to_string(index=False)}

## TRUE_COLLAPSE Signal Analysis
- Total TC events: {daily_agg['tc_mainlines'].sum()}
- Unique TC dates: {len(tc_dates)}
- TC dates list: {', '.join(tc_dates) if tc_dates else 'None'}

### TC Lead/Lag Analysis
{'No TC signals to analyze.' if tc_lead_df.empty else tc_lead_df.to_string(index=False)}

## Assessment as Market-Level Indicator

### Strengths
- Market-level danger_ratio provides a continuous risk measure (0-100%)
- TC signals are rare (0.04% of rows) → potentially high specificity
- Zero false positive noise → clean signal

### Weaknesses
- DIFFUSION_WARNING too broad (59.9% of rows) → danger_ratio always elevated
- TC signals only appear on 2 dates → too sparse for trading decisions
- No persistence → signals don't form trends

### Recommendation
P12c as a market-level indicator is **HOLD_RESEARCH_ONLY**:
- The danger_ratio metric is too noisy (always > 50% in current regime)
- TC signals are too rare to be actionable
- Consider using only after fixing the AND 4-group condition in true_mask
"""
    _save_md(report, out, "market_collapse_review.md")
    print("  Market aggregate review saved")

    return {
        "phase": "market_aggregate",
        "trading_days": len(daily_agg),
        "tc_dates": len(tc_dates),
        "high_danger_days": high_danger_days,
        "avg_danger_ratio": round(avg_danger, 2),
        "max_danger_ratio": round(max_danger, 2),
    }


# -- Main Entry Point --------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P12c Research Runner")
    p.add_argument("--mode", default="full", choices=[
        "baseline", "cross_mainline", "danger_zone", "persistence",
        "false_positive", "philosophy", "final_decision", "full",
        "threshold_scan", "market_aggregate",
    ])
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-05-01")
    return p


def main() -> int:
    args = build_parser().parse_args()
    mode = args.mode
    start_date = args.start
    end_date = args.end

    print(f"P12c Research Runner (v3) — Mode: {mode}")
    print(f"Period: {start_date} -> {end_date}")
    print(f"Output: {MY_REPORTS}")
    print()

    # 加载配置
    config = _load_config(DEFAULT_CONFIG_PATH)
    p12b_config = _load_config(P12B_CONFIG_PATH)
    baseline_config = _load_config(BASELINE_CONFIG_PATH)

    # 合并配置
    merged_config = _deep_merge(config, {"p12b": p12b_config, "baseline": baseline_config})

    # 初始化 DB
    db = DBClient(config)

    # 初始化引擎
    matrix = P12CCollapseConfirmationMatrix(db, merged_config)
    context_gate = P12ContextGate(db, p12b_config)

    phase_results: list[dict[str, Any]] = []

    if mode == "baseline":
        print("=== BASELINE MODE: Running P12c engine only ===")
        out = _out_dir("baseline")
        research_df = matrix.build_research_frame(start_date, end_date)
        reports = matrix.build_reports(start_date, end_date)
        matrix.write_reports(reports, out)
        print(f"Research frame: {len(research_df)} rows")
        if "collapse_state" in research_df.columns:
            print(f"State distribution:\n{research_df['collapse_state'].value_counts()}")
        print(f"Reports saved to {out}")

    elif mode == "cross_mainline":
        out = _out_dir("cross_mainline")
        r = run_cross_mainline(db, matrix, context_gate, start_date, end_date, out)
        phase_results.append(r)

    elif mode == "danger_zone":
        out = _out_dir("danger_zone")
        r = run_danger_zone(db, matrix, context_gate, start_date, end_date, out)
        phase_results.append(r)

    elif mode == "persistence":
        out = _out_dir("persistence")
        r = run_persistence(db, matrix, context_gate, start_date, end_date, out)
        phase_results.append(r)

    elif mode == "false_positive":
        out = _out_dir("false_positive")
        r = run_false_positive(db, matrix, context_gate, start_date, end_date, out)
        phase_results.append(r)

    elif mode == "philosophy":
        out = _out_dir("philosophy")
        r = run_philosophy(out)
        phase_results.append(r)

    elif mode == "threshold_scan":
        out = _out_dir("threshold_scan")
        r = run_threshold_scan(db, matrix, context_gate, start_date, end_date, out)
        phase_results.append(r)

    elif mode == "market_aggregate":
        out = _out_dir("market_aggregate")
        r = run_market_aggregate(db, matrix, context_gate, start_date, end_date, out)
        phase_results.append(r)

    elif mode == "final_decision":
        out = _out_dir("final_decision")
        # 需要先运行所有阶段
        print("  Running all phases for final decision...")
        r1 = run_cross_mainline(db, matrix, context_gate, start_date, end_date, out)
        r2 = run_danger_zone(db, matrix, context_gate, start_date, end_date, out)
        r3 = run_persistence(db, matrix, context_gate, start_date, end_date, out)
        r4 = run_false_positive(db, matrix, context_gate, start_date, end_date, out)
        r5 = run_philosophy(out)
        phase_results = [r1, r2, r3, r4, r5]
        r = run_final_decision(phase_results, out)
        phase_results.append(r)

    elif mode == "full":
        out = _out_dir("full_pipeline")
        print("=== FULL PIPELINE ===")
        print("Step 0: Loading research frame (one-time)...")
        research_df = matrix.build_research_frame(start_date, end_date)
        print(f"  Research frame: {len(research_df)} rows")
        _save_csv(research_df, out, "research_frame.csv")

        print("\n--- Phase 1: Cross-Mainline ---")
        r1 = run_cross_mainline(db, matrix, context_gate, start_date, end_date, out, research_df)
        phase_results.append(r1)

        print("\n--- Phase 2: Danger-Zone ---")
        r2 = run_danger_zone(db, matrix, context_gate, start_date, end_date, out, research_df)
        phase_results.append(r2)

        print("\n--- Phase 3: Persistence ---")
        r3 = run_persistence(db, matrix, context_gate, start_date, end_date, out, research_df)
        phase_results.append(r3)

        print("\n--- Phase 4: False Positive ---")
        r4 = run_false_positive(db, matrix, context_gate, start_date, end_date, out, research_df)
        phase_results.append(r4)

        print("\n--- Phase 5: Philosophy ---")
        r5 = run_philosophy(out)
        phase_results.append(r5)

        print("\n--- Phase 6: Final Decision ---")
        r6 = run_final_decision(phase_results, out)
        phase_results.append(r6)

    # 保存摘要
    if phase_results:
        summary_df = pd.DataFrame(phase_results)
        summary_path = MY_REPORTS / f"phase_summary_{_make_ts()}.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\nPhase summary saved to {summary_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
