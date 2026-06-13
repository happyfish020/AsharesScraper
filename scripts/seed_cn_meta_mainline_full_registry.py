"""Seed/extend GrowthAlpha cn_meta mainline registry.

This is the system-owned MAINLINE_ID layer.  External source codes such as
801080.SI / 850xxx.SI / BKxxxx must be mapped into this layer before any
GrowthAlpha consumer or fact builder uses them.

This script intentionally DOES NOT create/use cn_canonical_* tables and does
not read cn_ga_mainline_radar_daily.

Usage:
  python scripts/seed_cn_meta_mainline_full_registry.py --dry-run --strict
  python scripts/seed_cn_meta_mainline_full_registry.py --apply --strict
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL


@dataclass(frozen=True)
class MainlineSeed:
    mainline_id: str
    mainline_name: str
    mainline_alias: str | None
    category: str
    mainline_group: str
    display_order: int


@dataclass(frozen=True)
class SourceMapSeed:
    source_system: str
    source_level: str
    source_code: str
    source_name: str
    mainline_id: str
    source_type: str
    confidence: float
    display_order: int


@dataclass(frozen=True)
class SublineSeed:
    subline_id: str
    mainline_id: str
    subline_name: str
    subline_alias: str | None
    display_order: int


# A. Whole-market SW-L1 universe.  IDs are system-owned, not source codes.
SW_L1_MAINLINES: list[MainlineSeed] = [
    MainlineSeed("CN_AGRI_FORESTRY", "农林牧渔", "农业,林业,牧业,渔业,农林牧渔", "MARKET_SECTOR", "SW_L1", 1001),
    MainlineSeed("CN_BASIC_CHEMICALS", "基础化工", "化工,基础化工,化学品", "MARKET_SECTOR", "SW_L1", 1003),
    MainlineSeed("CN_STEEL", "钢铁", "钢铁,普钢,特钢", "MARKET_SECTOR", "SW_L1", 1004),
    MainlineSeed("CN_NONFERROUS_METALS", "有色金属", "有色,有色金属,工业金属,贵金属,小金属", "MARKET_SECTOR", "SW_L1", 1005),
    MainlineSeed("CN_ELECTRONICS", "电子", "电子,半导体,消费电子,元件", "MARKET_SECTOR", "SW_L1", 1008),
    MainlineSeed("CN_HOME_APPLIANCES", "家用电器", "家电,白电,黑电,小家电", "MARKET_SECTOR", "SW_L1", 1011),
    MainlineSeed("CN_FOOD_BEVERAGE", "食品饮料", "食品饮料,白酒,啤酒,调味品", "MARKET_SECTOR", "SW_L1", 1012),
    MainlineSeed("CN_TEXTILE_APPAREL", "纺织服饰", "纺织,服装,服饰", "MARKET_SECTOR", "SW_L1", 1013),
    MainlineSeed("CN_LIGHT_MANUFACTURING", "轻工制造", "轻工,造纸,包装,家居", "MARKET_SECTOR", "SW_L1", 1014),
    MainlineSeed("CN_PHARMA_BIO", "医药生物", "医药,生物医药,创新药,医疗器械", "MARKET_SECTOR", "SW_L1", 1015),
    MainlineSeed("CN_UTILITIES", "公用事业", "公用事业,电力,燃气,水务", "MARKET_SECTOR", "SW_L1", 1016),
    MainlineSeed("CN_TRANSPORTATION", "交通运输", "交通运输,航运,航空,铁路,物流", "MARKET_SECTOR", "SW_L1", 1017),
    MainlineSeed("CN_REAL_ESTATE", "房地产", "房地产,地产,物业", "MARKET_SECTOR", "SW_L1", 1018),
    MainlineSeed("CN_RETAIL", "商贸零售", "商贸,零售,电商,贸易", "MARKET_SECTOR", "SW_L1", 1020),
    MainlineSeed("CN_SOCIAL_SERVICES", "社会服务", "社会服务,旅游,酒店,教育", "MARKET_SECTOR", "SW_L1", 1021),
    MainlineSeed("CN_CONGLOMERATE", "综合", "综合", "MARKET_SECTOR", "SW_L1", 1023),
    MainlineSeed("CN_BUILDING_MATERIALS", "建筑材料", "建材,水泥,玻璃,消费建材", "MARKET_SECTOR", "SW_L1", 1710),
    MainlineSeed("CN_CONSTRUCTION_DECORATION", "建筑装饰", "建筑,基建,装饰,工程", "MARKET_SECTOR", "SW_L1", 1720),
    MainlineSeed("CN_POWER_EQUIPMENT_SECTOR", "电力设备", "电力设备,新能源设备,光伏,风电,电池", "MARKET_SECTOR", "SW_L1", 1730),
    MainlineSeed("CN_DEFENSE", "国防军工", "军工,国防军工,航空装备,航天装备", "MARKET_SECTOR", "SW_L1", 1740),
    MainlineSeed("CN_COMPUTER", "计算机", "计算机,软件,IT服务,信创", "MARKET_SECTOR", "SW_L1", 1750),
    MainlineSeed("CN_MEDIA", "传媒", "传媒,游戏,影视,出版", "MARKET_SECTOR", "SW_L1", 1760),
    MainlineSeed("CN_COMMUNICATION", "通信", "通信,通信设备,光通信,运营商", "MARKET_SECTOR", "SW_L1", 1770),
    MainlineSeed("CN_BANK", "银行", "银行,国有行,股份行,城商行", "MARKET_SECTOR", "SW_L1", 1780),
    MainlineSeed("CN_NONBANK_FINANCE", "非银金融", "券商,保险,多元金融,非银金融", "MARKET_SECTOR", "SW_L1", 1790),
    MainlineSeed("CN_AUTO", "汽车", "汽车,整车,零部件,智能汽车", "MARKET_SECTOR", "SW_L1", 1880),
    MainlineSeed("CN_MACHINERY", "机械设备", "机械设备,工控,工业母机,机器人", "MARKET_SECTOR", "SW_L1", 1890),
    MainlineSeed("CN_COAL", "煤炭", "煤炭,焦煤,动力煤", "MARKET_SECTOR", "SW_L1", 1950),
    MainlineSeed("CN_PETROCHEMICAL", "石油石化", "石油石化,油气,炼化", "MARKET_SECTOR", "SW_L1", 1960),
    MainlineSeed("CN_ENVIRONMENTAL", "环保", "环保,固废,水处理,环境治理", "MARKET_SECTOR", "SW_L1", 1970),
    MainlineSeed("CN_BEAUTY_CARE", "美容护理", "美容护理,化妆品,医美", "MARKET_SECTOR", "SW_L1", 1980),
]

SW_L1_SOURCE_MAP: list[SourceMapSeed] = [
    SourceMapSeed("SW", "L1", "801010.SI", "农林牧渔", "CN_AGRI_FORESTRY", "INDUSTRY_INDEX", 1.0, 1001),
    SourceMapSeed("SW", "L1", "801030.SI", "基础化工", "CN_BASIC_CHEMICALS", "INDUSTRY_INDEX", 1.0, 1003),
    SourceMapSeed("SW", "L1", "801040.SI", "钢铁", "CN_STEEL", "INDUSTRY_INDEX", 1.0, 1004),
    SourceMapSeed("SW", "L1", "801050.SI", "有色金属", "CN_NONFERROUS_METALS", "INDUSTRY_INDEX", 1.0, 1005),
    SourceMapSeed("SW", "L1", "801080.SI", "电子", "CN_ELECTRONICS", "INDUSTRY_INDEX", 1.0, 1008),
    SourceMapSeed("SW", "L1", "801110.SI", "家用电器", "CN_HOME_APPLIANCES", "INDUSTRY_INDEX", 1.0, 1011),
    SourceMapSeed("SW", "L1", "801120.SI", "食品饮料", "CN_FOOD_BEVERAGE", "INDUSTRY_INDEX", 1.0, 1012),
    SourceMapSeed("SW", "L1", "801130.SI", "纺织服饰", "CN_TEXTILE_APPAREL", "INDUSTRY_INDEX", 1.0, 1013),
    SourceMapSeed("SW", "L1", "801140.SI", "轻工制造", "CN_LIGHT_MANUFACTURING", "INDUSTRY_INDEX", 1.0, 1014),
    SourceMapSeed("SW", "L1", "801150.SI", "医药生物", "CN_PHARMA_BIO", "INDUSTRY_INDEX", 1.0, 1015),
    SourceMapSeed("SW", "L1", "801160.SI", "公用事业", "CN_UTILITIES", "INDUSTRY_INDEX", 1.0, 1016),
    SourceMapSeed("SW", "L1", "801170.SI", "交通运输", "CN_TRANSPORTATION", "INDUSTRY_INDEX", 1.0, 1017),
    SourceMapSeed("SW", "L1", "801180.SI", "房地产", "CN_REAL_ESTATE", "INDUSTRY_INDEX", 1.0, 1018),
    SourceMapSeed("SW", "L1", "801200.SI", "商贸零售", "CN_RETAIL", "INDUSTRY_INDEX", 1.0, 1020),
    SourceMapSeed("SW", "L1", "801210.SI", "社会服务", "CN_SOCIAL_SERVICES", "INDUSTRY_INDEX", 1.0, 1021),
    SourceMapSeed("SW", "L1", "801230.SI", "综合", "CN_CONGLOMERATE", "INDUSTRY_INDEX", 1.0, 1023),
    SourceMapSeed("SW", "L1", "801710.SI", "建筑材料", "CN_BUILDING_MATERIALS", "INDUSTRY_INDEX", 1.0, 1710),
    SourceMapSeed("SW", "L1", "801720.SI", "建筑装饰", "CN_CONSTRUCTION_DECORATION", "INDUSTRY_INDEX", 1.0, 1720),
    SourceMapSeed("SW", "L1", "801730.SI", "电力设备", "CN_POWER_EQUIPMENT_SECTOR", "INDUSTRY_INDEX", 1.0, 1730),
    SourceMapSeed("SW", "L1", "801740.SI", "国防军工", "CN_DEFENSE", "INDUSTRY_INDEX", 1.0, 1740),
    SourceMapSeed("SW", "L1", "801750.SI", "计算机", "CN_COMPUTER", "INDUSTRY_INDEX", 1.0, 1750),
    SourceMapSeed("SW", "L1", "801760.SI", "传媒", "CN_MEDIA", "INDUSTRY_INDEX", 1.0, 1760),
    SourceMapSeed("SW", "L1", "801770.SI", "通信", "CN_COMMUNICATION", "INDUSTRY_INDEX", 1.0, 1770),
    SourceMapSeed("SW", "L1", "801780.SI", "银行", "CN_BANK", "INDUSTRY_INDEX", 1.0, 1780),
    SourceMapSeed("SW", "L1", "801790.SI", "非银金融", "CN_NONBANK_FINANCE", "INDUSTRY_INDEX", 1.0, 1790),
    SourceMapSeed("SW", "L1", "801880.SI", "汽车", "CN_AUTO", "INDUSTRY_INDEX", 1.0, 1880),
    SourceMapSeed("SW", "L1", "801890.SI", "机械设备", "CN_MACHINERY", "INDUSTRY_INDEX", 1.0, 1890),
    SourceMapSeed("SW", "L1", "801950.SI", "煤炭", "CN_COAL", "INDUSTRY_INDEX", 1.0, 1950),
    SourceMapSeed("SW", "L1", "801960.SI", "石油石化", "CN_PETROCHEMICAL", "INDUSTRY_INDEX", 1.0, 1960),
    SourceMapSeed("SW", "L1", "801970.SI", "环保", "CN_ENVIRONMENTAL", "INDUSTRY_INDEX", 1.0, 1970),
    SourceMapSeed("SW", "L1", "801980.SI", "美容护理", "CN_BEAUTY_CARE", "INDUSTRY_INDEX", 1.0, 1980),
]

# B. Strategic GrowthAlpha mainlines.  Existing rows are preserved and idempotently updated.
STRATEGIC_MAINLINES: list[MainlineSeed] = [
    MainlineSeed("OPTICAL_COMMS", "光通信", "通信/光通信,光模块,CPO,光芯片,光器件", "CORE_DESTINATION", "AI", 1),
    MainlineSeed("PCB_ELECTRONICS", "PCB电子", "PCB铜连接/电子,AI服务器PCB,高速PCB,铜连接", "CORE_DESTINATION", "AI", 2),
    MainlineSeed("AI_COMPUTE", "AI算力", "AI算力,GPU,AI芯片,国产算力,算力服务器", "CORE_DESTINATION", "AI", 3),
    MainlineSeed("AI_POWER_INFRA", "AI电力基础设施", "电力设备/AI电力,AI电源,液冷,数据中心电力", "EMERGING", "AI", 4),
    MainlineSeed("AI_DEEPER_INFRA", "AI deeper infra", "先进封装,玻璃基板,CXL,PCIe,AI存储,AI材料", "EMERGING", "AI", 5),
    MainlineSeed("AI_CLOUD_CORE", "AI cloud core", "云计算,数据中心,云服务,AI云", "EMERGING", "AI", 6),
    MainlineSeed("SEMICONDUCTOR_EQP", "半导体设备", "电子/半导体设备,刻蚀,薄膜沉积,CMP,检测量测", "CORE_DESTINATION", "电子", 7),
    MainlineSeed("MECHANICAL_EQP", "机械设备", "机械设备/工控,工业母机,高端装备", "CORE_DESTINATION", "机械设备", 8),
    MainlineSeed("POWER_EQP", "电力设备", "电力设备/新能源,储能,电网,电力电子", "CORE_DESTINATION", "电力设备", 9),
    MainlineSeed("ROBOTICS_AUTOMATION", "机器人自动化", "机器人,工业机器人,自动化,工业自动化,智能制造,人形机器人,具身智能", "EMERGING", "AI", 80),
    MainlineSeed("AI_AGENT_APPLICATION", "AI Agent应用", "AI Agent,智能体应用,企业Agent,办公Agent", "EMERGING", "AI", 120),
    MainlineSeed("NUCLEAR_FUSION", "核聚变", "可控核聚变,聚变能源,聚变材料,聚变电源", "EMERGING", "ENERGY", 220),
]

STRATEGIC_SUBLINES: list[SublineSeed] = [
    SublineSeed("OPTICAL_MODULE", "OPTICAL_COMMS", "光模块", "光模块,800G,1.6T,高速光模块", 10),
    SublineSeed("OPTICAL_CHIP", "OPTICAL_COMMS", "光芯片", "光芯片,激光器,探测器,硅光", 20),
    SublineSeed("CPO_OPTICAL_INTERCONNECT", "OPTICAL_COMMS", "CPO光互连", "CPO,硅光,CPO光互连,光互连", 30),
    SublineSeed("OPTICAL_FIBER_CABLE", "OPTICAL_COMMS", "光纤光缆", "光纤,光缆,光棒,海缆,通信光纤", 40),
    SublineSeed("AI_PCB", "PCB_ELECTRONICS", "AI服务器PCB", "AI服务器PCB,高速PCB,HDI,高多层PCB", 10),
    SublineSeed("CCL_MATERIAL", "PCB_ELECTRONICS", "覆铜板材料", "覆铜板,CCL,高频高速覆铜板,电子布,电子纱", 20),
    SublineSeed("COPPER_INTERCONNECT", "PCB_ELECTRONICS", "铜连接", "铜连接,高速连接器,高速线缆,高速互连", 30),
    SublineSeed("AI_ACCELERATOR_CHIP", "AI_COMPUTE", "AI加速芯片", "AI芯片,GPU,ASIC,AI加速卡,AI算力芯片,国产算力", 10),
    SublineSeed("AI_COOLING", "AI_POWER_INFRA", "AI液冷散热", "液冷,浸没式冷却,散热,数据中心冷却,服务器散热", 10),
    SublineSeed("AI_POWER_SUPPLY", "AI_POWER_INFRA", "AI电源", "服务器电源,UPS,HVDC,AI电源,数据中心电源", 20),
    SublineSeed("POWER_DISTRIBUTION", "AI_POWER_INFRA", "配电与电网支撑", "配电,电网设备,变压器,智能电网,电力基础设施", 30),
    SublineSeed("ADVANCED_PACKAGING", "AI_DEEPER_INFRA", "先进封装", "先进封装,Chiplet,2.5D封装,3D封装,HBM封装,CoWoS", 10),
    SublineSeed("GLASS_SUBSTRATE", "AI_DEEPER_INFRA", "玻璃基板", "玻璃基板,TGV,Glass Core Substrate,玻璃核心基板", 20),
    SublineSeed("AI_STORAGE_MEMORY", "AI_DEEPER_INFRA", "AI存储", "HBM,DDR5,存储芯片,企业级SSD,AI存储", 30),
    SublineSeed("AI_MATERIALS", "AI_DEEPER_INFRA", "AI材料", "电子布,电子纱,覆铜板材料,AI材料,低介电材料,封装材料", 40),
    SublineSeed("CXL_PCIE_INTERCONNECT", "AI_DEEPER_INFRA", "CXL/PCIe互连", "CXL,PCIe,高速互连,交换芯片,高速连接", 50),
    SublineSeed("SEMI_LITHOGRAPHY", "SEMICONDUCTOR_EQP", "光刻相关设备", "光刻机,涂胶显影,光刻胶配套设备", 10),
    SublineSeed("SEMI_ETCH", "SEMICONDUCTOR_EQP", "刻蚀设备", "刻蚀,等离子刻蚀,介质刻蚀", 20),
    SublineSeed("SEMI_DEPOSITION", "SEMICONDUCTOR_EQP", "薄膜沉积设备", "PVD,CVD,ALD,薄膜沉积", 30),
    SublineSeed("SEMI_CMP_CLEAN", "SEMICONDUCTOR_EQP", "CMP/清洗设备", "CMP,清洗设备,抛光,湿法设备", 40),
    SublineSeed("SEMI_TEST_METROLOGY", "SEMICONDUCTOR_EQP", "检测量测设备", "检测设备,量测设备,测试设备,ATE", 50),
    SublineSeed("ROBOT_INDUSTRIAL", "ROBOTICS_AUTOMATION", "工业机器人", "机器人本体,六轴机器人,工业机器人,机器人集成", 10),
    SublineSeed("MOTION_CONTROL", "ROBOTICS_AUTOMATION", "运动控制", "伺服,控制器,PLC,变频器,工业控制,运动控制", 20),
    SublineSeed("ROBOT_REDUCER", "ROBOTICS_AUTOMATION", "减速器", "RV减速器,谐波减速器,精密减速器", 30),
    SublineSeed("MACHINE_VISION", "ROBOTICS_AUTOMATION", "机器视觉", "工业视觉,视觉检测,机器视觉,工业相机", 40),
    SublineSeed("HUMANOID_ROBOT", "ROBOTICS_AUTOMATION", "人形机器人", "人形机器人,具身智能,Tesla Optimus,机器人执行器", 50),
    SublineSeed("ROBOT_COMPONENTS", "ROBOTICS_AUTOMATION", "机器人核心部件", "丝杠,传感器,执行器,关节模组,空心杯电机,精密零部件", 60),
    SublineSeed("AI_ROBOT_SOFTWARE", "ROBOTICS_AUTOMATION", "AI机器人软件", "机器人大模型,具身智能,机器人算法,机器人软件,AI控制", 70),
    SublineSeed("AGENT_FRAMEWORK", "AI_AGENT_APPLICATION", "Agent框架", "agent framework,智能体框架", 10),
    SublineSeed("AGENT_ENTERPRISE", "AI_AGENT_APPLICATION", "企业Agent", "enterprise agent,企业智能体", 20),
    SublineSeed("AGENT_OFFICE", "AI_AGENT_APPLICATION", "办公Agent", "office agent,办公智能体", 30),
    SublineSeed("FUSION_MATERIAL", "NUCLEAR_FUSION", "聚变材料", "fusion material,聚变材料", 10),
    SublineSeed("FUSION_MAGNET", "NUCLEAR_FUSION", "聚变磁体", "fusion magnet,聚变磁体", 20),
    SublineSeed("FUSION_POWER_SYSTEM", "NUCLEAR_FUSION", "聚变电源系统", "fusion power system,聚变电源系统", 30),
    SublineSeed("FUSION_CONTROL_SYSTEM", "NUCLEAR_FUSION", "聚变控制系统", "fusion control system,聚变控制系统", 40),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Seed full cn_meta mainline/subline/source mapping registry")
    p.add_argument("--db-name", default="cn_market_red")
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="cn_opr_red")
    p.add_argument("--db-password", default=None)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p


def build_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = URL.create("mysql+pymysql", username=user, password=password, host=host, port=port, database=db, query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True, future=True)


def ensure_source_map_schema(engine: Engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS cn_meta_source_mainline_map (
        source_system VARCHAR(32) NOT NULL,
        source_level VARCHAR(32) NOT NULL,
        source_code VARCHAR(64) NOT NULL,
        source_name VARCHAR(128) NULL,
        source_type VARCHAR(64) NOT NULL DEFAULT 'UNKNOWN',
        mainline_id VARCHAR(64) NOT NULL,
        mapping_confidence DECIMAL(6,4) NOT NULL DEFAULT 1.0000,
        is_primary_mapping TINYINT NOT NULL DEFAULT 1,
        is_active TINYINT NOT NULL DEFAULT 1,
        display_order INT NOT NULL DEFAULT 0,
        effective_start_date DATE NOT NULL DEFAULT '2010-01-01',
        effective_end_date DATE NULL,
        note VARCHAR(512) NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (source_system, source_level, source_code, mainline_id, effective_start_date),
        KEY idx_cmsmm_mainline (mainline_id, is_active),
        KEY idx_cmsmm_source (source_system, source_level, source_code, is_active)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def upsert_mainlines(engine: Engine, rows: list[MainlineSeed]) -> int:
    sql = text("""
        INSERT INTO cn_meta_mainline_registry
            (mainline_id, mainline_name, mainline_alias, category, mainline_group,
             is_active, display_order, created_at, updated_at)
        VALUES
            (:mainline_id, :mainline_name, :mainline_alias, :category, :mainline_group,
             1, :display_order, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON DUPLICATE KEY UPDATE
            mainline_name=VALUES(mainline_name),
            mainline_alias=VALUES(mainline_alias),
            category=VALUES(category),
            mainline_group=VALUES(mainline_group),
            is_active=1,
            display_order=VALUES(display_order),
            updated_at=CURRENT_TIMESTAMP
    """)
    payload = [r.__dict__ for r in rows]
    with engine.begin() as conn:
        conn.execute(sql, payload)
    return len(payload)


def upsert_sublines(engine: Engine, rows: list[SublineSeed]) -> int:
    sql = text("""
        INSERT INTO cn_meta_subline_registry
            (subline_id, mainline_id, subline_name, subline_alias,
             is_active, display_order, created_at, updated_at)
        VALUES
            (:subline_id, :mainline_id, :subline_name, :subline_alias,
             1, :display_order, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON DUPLICATE KEY UPDATE
            mainline_id=VALUES(mainline_id),
            subline_name=VALUES(subline_name),
            subline_alias=VALUES(subline_alias),
            is_active=1,
            display_order=VALUES(display_order),
            updated_at=CURRENT_TIMESTAMP
    """)
    payload = [r.__dict__ for r in rows]
    with engine.begin() as conn:
        conn.execute(sql, payload)
    return len(payload)


def upsert_source_maps(engine: Engine, rows: list[SourceMapSeed]) -> int:
    sql = text("""
        INSERT INTO cn_meta_source_mainline_map
            (source_system, source_level, source_code, source_name, source_type,
             mainline_id, mapping_confidence, is_primary_mapping, is_active,
             display_order, effective_start_date, effective_end_date, note,
             created_at, updated_at)
        VALUES
            (:source_system, :source_level, :source_code, :source_name, :source_type,
             :mainline_id, :confidence, 1, 1,
             :display_order, '2010-01-01', NULL, 'seeded by P0G1 metadata full registry',
             CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON DUPLICATE KEY UPDATE
            source_name=VALUES(source_name),
            source_type=VALUES(source_type),
            mapping_confidence=VALUES(mapping_confidence),
            is_primary_mapping=1,
            is_active=1,
            display_order=VALUES(display_order),
            note=VALUES(note),
            updated_at=CURRENT_TIMESTAMP
    """)
    payload = [r.__dict__ for r in rows]
    with engine.begin() as conn:
        conn.execute(sql, payload)
    return len(payload)


def count_rows(engine: Engine) -> dict[str, Any]:
    sql = text("""
        SELECT
          (SELECT COUNT(*) FROM cn_meta_mainline_registry WHERE is_active=1) AS active_mainlines,
          (SELECT COUNT(*) FROM cn_meta_mainline_registry WHERE is_active=1 AND category='MARKET_SECTOR') AS active_market_sectors,
          (SELECT COUNT(*) FROM cn_meta_subline_registry WHERE is_active=1) AS active_sublines,
          (SELECT COUNT(*) FROM cn_meta_source_mainline_map WHERE is_active=1) AS active_source_maps
    """)
    with engine.connect() as conn:
        return dict(conn.execute(sql).mappings().first() or {})


def main() -> None:
    args = build_parser().parse_args()
    if args.apply and args.dry_run:
        raise SystemExit("Use only one of --apply or --dry-run")
    if not args.apply and not args.dry_run:
        raise SystemExit("Specify --dry-run to preview or --apply to write")
    password = args.db_password if args.db_password is not None else os.getenv("MYSQL_PASSWORD", "")
    engine = build_engine(args.db_host, args.db_port, args.db_user, password, args.db_name)

    mainlines = SW_L1_MAINLINES + STRATEGIC_MAINLINES
    sublines = STRATEGIC_SUBLINES
    source_maps = SW_L1_SOURCE_MAP
    print("[META SEED PLAN]", {"mainlines": len(mainlines), "sublines": len(sublines), "source_maps": len(source_maps)})

    if args.dry_run:
        print("[META SEED DRY-RUN] no write")
        return

    ensure_source_map_schema(engine)
    n_main = upsert_mainlines(engine, mainlines)
    n_sub = upsert_sublines(engine, sublines)
    n_map = upsert_source_maps(engine, source_maps)
    summary = count_rows(engine)
    print("[META SEED WROTE]", {"mainlines_upserted": n_main, "sublines_upserted": n_sub, "source_maps_upserted": n_map, **summary})

    if args.strict:
        if int(summary.get("active_market_sectors") or 0) < 31:
            raise SystemExit("[META SEED FAILED] active MARKET_SECTOR mainlines < 31")
        if int(summary.get("active_source_maps") or 0) < 31:
            raise SystemExit("[META SEED FAILED] active source maps < 31")
        if int(summary.get("active_sublines") or 0) < 35:
            raise SystemExit("[META SEED FAILED] active sublines < 35")
    print("[META SEED PASS]")


if __name__ == "__main__":
    main()
