-- MySQL dump 10.13  Distrib 8.0.43, for Win64 (x86_64)
--
-- Host: localhost    Database: cn_market
-- ------------------------------------------------------
-- Server version	9.4.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `cn_baseline_decision_t`
--

DROP TABLE IF EXISTS `cn_baseline_decision_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_baseline_decision_t` (
  `RUN_ID` varchar(128) COLLATE utf8mb4_unicode_ci NOT NULL,
  `BASELINE_KEY` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `BASELINE_RUN_ID` varchar(128) COLLATE utf8mb4_unicode_ci NOT NULL,
  `DECISION` varchar(16) COLLATE utf8mb4_unicode_ci NOT NULL,
  `REASON_CODE` varchar(4000) COLLATE utf8mb4_unicode_ci NOT NULL,
  `METRICS_JSON` longtext COLLATE utf8mb4_unicode_ci,
  `COMPARE_ASOF` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `CREATED_AT` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `UPDATED_AT` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `CREATED_BY` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  PRIMARY KEY (`RUN_ID`,`BASELINE_KEY`),
  KEY `IDX_CN_BASELINE_DECISION_ASOF` (`COMPARE_ASOF`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_baseline_registry_t`
--

DROP TABLE IF EXISTS `cn_baseline_registry_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_baseline_registry_t` (
  `BASELINE_KEY` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `RUN_ID` varchar(128) COLLATE utf8mb4_unicode_ci NOT NULL,
  `VERSION_TAG` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `IS_ACTIVE` int NOT NULL DEFAULT '0',
  `BASELINE_DESC` varchar(400) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `UPDATED_AT` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`BASELINE_KEY`),
  UNIQUE KEY `UK_CN_BASELINE_REGISTRY_RUN` (`RUN_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_board_concept_cons`
--

DROP TABLE IF EXISTS `cn_board_concept_cons`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_board_concept_cons` (
  `CONCEPT_ID` varchar(40) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SYMBOL` varchar(10) COLLATE utf8mb4_unicode_ci NOT NULL,
  `EXCHANGE` varchar(10) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ASOF_DATE` date NOT NULL,
  `SOURCE` varchar(30) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`CONCEPT_ID`,`SYMBOL`,`ASOF_DATE`),
  KEY `IDX_CONCEPT_CONS_CONCEPT` (`CONCEPT_ID`),
  KEY `IDX_CONCEPT_CONS_SYMBOL` (`SYMBOL`),
  KEY `IDX_CONSCONS_ID_ASOF` (`CONCEPT_ID`,`ASOF_DATE`),
  KEY `idx_con_cons_symbol_asof` (`SYMBOL`,`ASOF_DATE`),
  KEY `idx_con_cons_asof_symbol_concept` (`ASOF_DATE`,`SYMBOL`,`CONCEPT_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_board_concept_cons_current`
--

DROP TABLE IF EXISTS `cn_board_concept_cons_current`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_board_concept_cons_current` (
  `concept_id` int unsigned NOT NULL,
  `symbol` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '股票代码',
  `asof_date` date NOT NULL,
  PRIMARY KEY (`concept_id`,`symbol`),
  KEY `idx_symbol` (`symbol`),
  KEY `idx_concept` (`concept_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary view structure for view `cn_board_concept_eod_agg_v`
--

DROP TABLE IF EXISTS `cn_board_concept_eod_agg_v`;
/*!50001 DROP VIEW IF EXISTS `cn_board_concept_eod_agg_v`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `cn_board_concept_eod_agg_v` AS SELECT 
 1 AS `sector_type`,
 1 AS `sector_id`,
 1 AS `trade_date`,
 1 AS `members`,
 1 AS `amount_sum`,
 1 AS `avg_ret`,
 1 AS `median_ret`,
 1 AS `eqw_ret`,
 1 AS `up_ratio`*/;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `cn_board_concept_master`
--

DROP TABLE IF EXISTS `cn_board_concept_master`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_board_concept_master` (
  `CONCEPT_ID` varchar(40) COLLATE utf8mb4_unicode_ci NOT NULL,
  `CONCEPT_NAME` varchar(80) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `PROVIDER` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT 'EASTMONEY',
  `ASOF_DATE` date NOT NULL,
  `SOURCE` varchar(30) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` datetime DEFAULT CURRENT_TIMESTAMP,
  `RAW_JSON` longtext COLLATE utf8mb4_unicode_ci,
  PRIMARY KEY (`CONCEPT_ID`,`ASOF_DATE`),
  KEY `IDX_CONCEPT_ASOF` (`ASOF_DATE`),
  KEY `IDX_CONCEPT_NAME` (`CONCEPT_NAME`),
  KEY `IX_SECTOR_MASTER_MAP_V_KEY2` (`CONCEPT_ID`),
  KEY `idx_con_master_concept_asof` (`CONCEPT_ID`,`ASOF_DATE`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_board_concept_member_hist`
--

DROP TABLE IF EXISTS `cn_board_concept_member_hist`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_board_concept_member_hist` (
  `concept_id` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `symbol` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `valid_from` date NOT NULL,
  `valid_to` date DEFAULT NULL,
  `source` varchar(32) NOT NULL,
  `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`concept_id`,`symbol`,`valid_from`),
  KEY `idx_cbcmh_symbol_valid` (`symbol`,`valid_from`,`valid_to`),
  KEY `idx_cbcmh_concept_valid` (`concept_id`,`valid_from`,`valid_to`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_board_concept_member_stg`
--

DROP TABLE IF EXISTS `cn_board_concept_member_stg`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_board_concept_member_stg` (
  `asof_date` date NOT NULL,
  `concept_id` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `symbol` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `source` varchar(32) NOT NULL,
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`asof_date`,`concept_id`,`symbol`),
  KEY `idx_cbcms_asof_symbol` (`asof_date`,`symbol`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_board_industry_cons`
--

DROP TABLE IF EXISTS `cn_board_industry_cons`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_board_industry_cons` (
  `BOARD_ID` varchar(40) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SYMBOL` varchar(10) COLLATE utf8mb4_unicode_ci NOT NULL,
  `EXCHANGE` varchar(10) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ASOF_DATE` date NOT NULL,
  `SOURCE` varchar(30) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`BOARD_ID`,`SYMBOL`,`ASOF_DATE`),
  KEY `IDX_INDCONS_ID_ASOF` (`BOARD_ID`,`ASOF_DATE`),
  KEY `IDX_INDCONS_ID_SYMBOL` (`BOARD_ID`,`SYMBOL`),
  KEY `IDX_INDUSTRY_CONS_BOARD` (`BOARD_ID`),
  KEY `IDX_INDUSTRY_CONS_SYMBOL` (`SYMBOL`),
  KEY `idx_ind_cons_symbol_asof` (`SYMBOL`,`ASOF_DATE`),
  KEY `idx_ind_cons_asof_symbol_board` (`ASOF_DATE`,`SYMBOL`,`BOARD_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary view structure for view `cn_board_industry_eod_agg_v`
--

DROP TABLE IF EXISTS `cn_board_industry_eod_agg_v`;
/*!50001 DROP VIEW IF EXISTS `cn_board_industry_eod_agg_v`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `cn_board_industry_eod_agg_v` AS SELECT 
 1 AS `sector_type`,
 1 AS `sector_id`,
 1 AS `trade_date`,
 1 AS `members`,
 1 AS `amount_sum`,
 1 AS `avg_ret`,
 1 AS `median_ret`,
 1 AS `eqw_ret`,
 1 AS `up_ratio`*/;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `cn_board_industry_master`
--

DROP TABLE IF EXISTS `cn_board_industry_master`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_board_industry_master` (
  `BOARD_ID` varchar(40) COLLATE utf8mb4_unicode_ci NOT NULL,
  `BOARD_NAME` varchar(80) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `PROVIDER` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT 'EASTMONEY',
  `ASOF_DATE` date NOT NULL,
  `SOURCE` varchar(30) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` datetime DEFAULT CURRENT_TIMESTAMP,
  `RAW_JSON` longtext COLLATE utf8mb4_unicode_ci,
  PRIMARY KEY (`BOARD_ID`,`ASOF_DATE`),
  KEY `IDX_INDUSTRY_ASOF` (`ASOF_DATE`),
  KEY `IDX_INDUSTRY_NAME` (`BOARD_NAME`),
  KEY `IX_SECTOR_MASTER_MAP_V_KEY` (`BOARD_ID`),
  KEY `idx_ind_master_board_asof` (`BOARD_ID`,`ASOF_DATE`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_board_industry_member_hist`
--

DROP TABLE IF EXISTS `cn_board_industry_member_hist`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_board_industry_member_hist` (
  `board_id` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `symbol` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `valid_from` date NOT NULL,
  `valid_to` date DEFAULT NULL,
  `source` varchar(32) NOT NULL,
  `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`board_id`,`symbol`,`valid_from`),
  KEY `idx_cbimh_symbol_valid` (`symbol`,`valid_from`,`valid_to`),
  KEY `idx_cbimh_board_valid` (`board_id`,`valid_from`,`valid_to`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_board_industry_member_stg`
--

DROP TABLE IF EXISTS `cn_board_industry_member_stg`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_board_industry_member_stg` (
  `asof_date` date NOT NULL,
  `board_id` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `symbol` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `source` varchar(32) NOT NULL,
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`asof_date`,`board_id`,`symbol`),
  KEY `idx_cbims_asof_symbol` (`asof_date`,`symbol`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_board_member_map_d`
--

DROP TABLE IF EXISTS `cn_board_member_map_d`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_board_member_map_d` (
  `trade_date` date NOT NULL,
  `sector_type` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `sector_id` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `symbol` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  PRIMARY KEY (`trade_date`,`sector_type`,`sector_id`,`symbol`),
  KEY `idx_cbmmd_symbol_date` (`symbol`,`trade_date`),
  KEY `idx_cbmmd_type_sector_date` (`sector_type`,`sector_id`,`trade_date`),
  KEY `idx_cbmmd_date_type_symbol_sector` (`trade_date`,`sector_type`,`symbol`,`sector_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_fund_etf_hist_em`
--

DROP TABLE IF EXISTS `cn_fund_etf_hist_em`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_fund_etf_hist_em` (
  `CODE` varchar(50) COLLATE utf8mb4_unicode_ci NOT NULL,
  `DATA_DATE` date NOT NULL,
  `OPEN_PRICE` double DEFAULT NULL,
  `CLOSE_PRICE` double DEFAULT NULL,
  `HIGH_PRICE` double DEFAULT NULL,
  `LOW_PRICE` double DEFAULT NULL,
  `VOLUME` bigint DEFAULT NULL,
  `AMOUNT` bigint DEFAULT NULL,
  `AMPLITUDE` double DEFAULT NULL,
  `CHANGE_PCT` double DEFAULT NULL,
  `CHANGE_AMOUNT` double DEFAULT NULL,
  `TURNOVER_RATE` double DEFAULT NULL,
  `ADJUST_TYPE` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `SOURCE` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`CODE`,`DATA_DATE`),
  KEY `cn_fund_etf_hist_em_CODE_IDX` (`CODE`) USING BTREE,
  KEY `cn_fund_etf_hist_em_DATA_DATE_IDX` (`DATA_DATE`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_fut_index_his`
--

DROP TABLE IF EXISTS `cn_fut_index_his`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_fut_index_his` (
  `TRADE_DATE` date NOT NULL,
  `SYMBOL` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL,
  `MAIN_CONTRACT` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `OPEN_PRICE` double DEFAULT NULL,
  `HIGH_PRICE` double DEFAULT NULL,
  `LOW_PRICE` double DEFAULT NULL,
  `CLOSE_PRICE` double DEFAULT NULL,
  `SETTLE_PRICE` double DEFAULT NULL,
  `PRE_SETTLE` double DEFAULT NULL,
  `VOLUME` bigint DEFAULT NULL,
  `TURNOVER` double DEFAULT NULL,
  `OPEN_INTEREST` bigint DEFAULT NULL,
  `VARIETY` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `SOURCE` varchar(30) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`TRADE_DATE`,`SYMBOL`),
  KEY `cn_fut_index_his_TRADE_DATE_IDX` (`TRADE_DATE`) USING BTREE,
  KEY `cn_fut_index_his_SYMBOL_IDX` (`SYMBOL`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_index_daily_price`
--

DROP TABLE IF EXISTS `cn_index_daily_price`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_index_daily_price` (
  `INDEX_CODE` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `TRADE_DATE` date NOT NULL,
  `OPEN` double DEFAULT NULL,
  `CLOSE` double DEFAULT NULL,
  `HIGH` double DEFAULT NULL,
  `LOW` double DEFAULT NULL,
  `VOLUME` bigint DEFAULT NULL,
  `AMOUNT` bigint DEFAULT NULL,
  `SOURCE` varchar(30) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `PRE_CLOSE` double DEFAULT NULL,
  `CHG_PCT` double DEFAULT NULL,
  PRIMARY KEY (`INDEX_CODE`,`TRADE_DATE`),
  KEY `cn_index_daily_price_INDEX_CODE_IDX` (`INDEX_CODE`) USING BTREE,
  KEY `cn_index_daily_price_TRADE_DATE_IDX` (`TRADE_DATE`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_option_sse_daily`
--

DROP TABLE IF EXISTS `cn_option_sse_daily`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_option_sse_daily` (
  `CONTRACT_CODE` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `UNDERLYING_CODE` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `EXPIRY_MONTH` varchar(6) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `DATA_DATE` date NOT NULL,
  `OPEN_PRICE` double DEFAULT NULL,
  `HIGH_PRICE` double DEFAULT NULL,
  `LOW_PRICE` double DEFAULT NULL,
  `CLOSE_PRICE` double DEFAULT NULL,
  `VOLUME` bigint DEFAULT NULL,
  `SOURCE` varchar(30) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`CONTRACT_CODE`,`DATA_DATE`),
  KEY `cn_option_sse_daily_UNDERLYING_CODE_IDX` (`UNDERLYING_CODE`) USING BTREE,
  KEY `cn_option_sse_daily_CONTRACT_CODE_IDX` (`CONTRACT_CODE`) USING BTREE,
  KEY `cn_option_sse_daily_DATA_DATE_IDX` (`DATA_DATE`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_rotation_entry_snap_t`
--

DROP TABLE IF EXISTS `cn_rotation_entry_snap_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_rotation_entry_snap_t` (
  `RUN_ID` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `TRADE_DATE` date NOT NULL,
  `SECTOR_TYPE` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_ID` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_NAME` varchar(200) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ENTRY_RANK` bigint DEFAULT NULL,
  `ENTRY_CNT` bigint DEFAULT NULL,
  `WEIGHT_SUGGESTED` decimal(18,8) DEFAULT NULL,
  `SIGNAL_SCORE` decimal(38,16) DEFAULT NULL,
  `ENERGY_SCORE` decimal(38,16) DEFAULT NULL,
  `ENERGY_PCT` decimal(38,16) DEFAULT NULL,
  `ENERGY_TIER` varchar(16) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `STATE` varchar(32) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `TRANSITION` varchar(64) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `SOURCE_JSON` longtext COLLATE utf8mb4_unicode_ci,
  `CREATED_AT` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`RUN_ID`,`TRADE_DATE`,`SECTOR_TYPE`,`SECTOR_ID`),
  KEY `IDX_ROT_ENTRY_SNAP_Q1` (`RUN_ID`,`TRADE_DATE`,`ENERGY_TIER`,`ENERGY_PCT`,`SIGNAL_SCORE`,`ENTRY_RANK`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_rotation_exit_snap_t`
--

DROP TABLE IF EXISTS `cn_rotation_exit_snap_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_rotation_exit_snap_t` (
  `RUN_ID` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `TRADE_DATE` date NOT NULL,
  `EXEC_EXIT_DATE` date DEFAULT NULL,
  `SECTOR_TYPE` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_ID` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_NAME` varchar(200) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `STATE` varchar(32) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `TRANSITION` varchar(64) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ENTRY_RANK` bigint DEFAULT NULL,
  `SIGNAL_SCORE` decimal(38,16) DEFAULT NULL,
  `ENTER_SIGNAL_DATE` date DEFAULT NULL,
  `EXEC_ENTER_DATE` date DEFAULT NULL,
  `HOLD_DAYS` bigint DEFAULT NULL,
  `MIN_HOLD_DAYS` bigint DEFAULT NULL,
  `EXIT_EXEC_STATUS` varchar(16) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `SOURCE_JSON` longtext COLLATE utf8mb4_unicode_ci,
  `CREATED_AT` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`RUN_ID`,`TRADE_DATE`,`SECTOR_TYPE`,`SECTOR_ID`),
  KEY `IDX_ROT_EXIT_SNAP_Q1` (`RUN_ID`,`TRADE_DATE`,`EXIT_EXEC_STATUS`,`EXEC_EXIT_DATE`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_rotation_holding_snap_t`
--

DROP TABLE IF EXISTS `cn_rotation_holding_snap_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_rotation_holding_snap_t` (
  `RUN_ID` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `TRADE_DATE` date NOT NULL,
  `SECTOR_TYPE` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_ID` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_NAME` varchar(200) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ENTER_SIGNAL_DATE` date DEFAULT NULL,
  `EXEC_ENTER_DATE` date DEFAULT NULL,
  `HOLD_DAYS` bigint DEFAULT NULL,
  `MIN_HOLD_DAYS` bigint DEFAULT NULL,
  `EXIT_SIGNAL_TODAY` int DEFAULT NULL,
  `EXIT_TRANSITION` varchar(64) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `EXIT_EXEC_STATUS` varchar(16) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `NEXT_EXIT_ELIGIBLE_DATE` date DEFAULT NULL,
  `SOURCE_JSON` longtext COLLATE utf8mb4_unicode_ci,
  `CREATED_AT` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`RUN_ID`,`TRADE_DATE`,`SECTOR_TYPE`,`SECTOR_ID`),
  KEY `IDX_ROT_HOLD_SNAP_Q1` (`RUN_ID`,`TRADE_DATE`,`EXIT_EXEC_STATUS`,`SECTOR_TYPE`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_sector_energy_snap_t`
--

DROP TABLE IF EXISTS `cn_sector_energy_snap_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_sector_energy_snap_t` (
  `TRADE_DATE` date NOT NULL,
  `SECTOR_TYPE` varchar(16) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_ID` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `ENERGY_SCORE` double DEFAULT NULL,
  `ENERGY_PCT` double DEFAULT NULL,
  `CREATED_AT` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`TRADE_DATE`,`SECTOR_TYPE`,`SECTOR_ID`),
  KEY `IX_CN_SECTOR_ENERGY_SNAP_T_DT` (`TRADE_DATE`),
  KEY `IX_SECTOR_ENERGY_SNAP_DT` (`TRADE_DATE`,`SECTOR_TYPE`),
  KEY `IX_SRENG_DT_TYPE_PCT_SCORE` (`TRADE_DATE`,`SECTOR_TYPE`,`ENERGY_PCT`,`ENERGY_SCORE`,`SECTOR_ID`),
  KEY `idx_energy_snap_key` (`TRADE_DATE`,`SECTOR_TYPE`,`SECTOR_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary view structure for view `cn_sector_energy_v`
--

DROP TABLE IF EXISTS `cn_sector_energy_v`;
/*!50001 DROP VIEW IF EXISTS `cn_sector_energy_v`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `cn_sector_energy_v` AS SELECT 
 1 AS `trade_date`,
 1 AS `sector_type`,
 1 AS `sector_id`,
 1 AS `energy_score`,
 1 AS `energy_pct`*/;
SET character_set_client = @saved_cs_client;

--
-- Temporary view structure for view `cn_sector_eod_agg_v`
--

DROP TABLE IF EXISTS `cn_sector_eod_agg_v`;
/*!50001 DROP VIEW IF EXISTS `cn_sector_eod_agg_v`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `cn_sector_eod_agg_v` AS SELECT 
 1 AS `SECTOR_TYPE`,
 1 AS `SECTOR_ID`,
 1 AS `TRADE_DATE`,
 1 AS `MEMBERS`,
 1 AS `AMOUNT_SUM`,
 1 AS `AVG_RET`,
 1 AS `MEDIAN_RET`,
 1 AS `EQW_RET`,
 1 AS `UP_RATIO`*/;
SET character_set_client = @saved_cs_client;

--
-- Temporary view structure for view `cn_sector_eod_feature_v`
--

DROP TABLE IF EXISTS `cn_sector_eod_feature_v`;
/*!50001 DROP VIEW IF EXISTS `cn_sector_eod_feature_v`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `cn_sector_eod_feature_v` AS SELECT 
 1 AS `sector_type`,
 1 AS `sector_id`,
 1 AS `trade_date`,
 1 AS `members`,
 1 AS `amount_sum`,
 1 AS `avg_ret`,
 1 AS `median_ret`,
 1 AS `eqw_ret`,
 1 AS `up_ratio`,
 1 AS `cov_5`,
 1 AS `cov_10`,
 1 AS `cov_20`,
 1 AS `amt_ma20`,
 1 AS `amt_impulse`,
 1 AS `up_ma5`,
 1 AS `up_slope`,
 1 AS `mom_5`,
 1 AS `mom_10`,
 1 AS `mom_20`*/;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `cn_sector_eod_hist_t`
--

DROP TABLE IF EXISTS `cn_sector_eod_hist_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_sector_eod_hist_t` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `trade_date` date NOT NULL,
  `sector_type` varchar(16) COLLATE utf8mb4_general_ci NOT NULL COMMENT 'INDUSTRY|CONCEPT',
  `sector_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `members` int DEFAULT NULL,
  `amount_sum` decimal(20,2) DEFAULT NULL,
  `sector_close` decimal(20,6) DEFAULT NULL,
  `sector_ma20` decimal(20,6) DEFAULT NULL,
  `sector_ret20` decimal(20,8) DEFAULT NULL,
  `up_ratio` decimal(10,6) DEFAULT NULL,
  `score` decimal(20,8) DEFAULT NULL COMMENT 'optional: strength score for ranking',
  `rank_pct` decimal(10,6) DEFAULT NULL,
  `cond1_trend` tinyint(1) DEFAULT NULL,
  `cond2_rank` tinyint(1) DEFAULT NULL,
  `cond3_breadth` tinyint(1) DEFAULT NULL,
  `cond_count` tinyint unsigned DEFAULT NULL,
  `sector_pass` tinyint(1) NOT NULL DEFAULT '0',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_sector_day` (`trade_date`,`sector_type`,`sector_id`),
  KEY `idx_trade_date` (`trade_date`),
  KEY `idx_sector_day` (`sector_type`,`sector_id`,`trade_date`),
  KEY `idx_sector_pass_day` (`trade_date`,`sector_pass`)
) ENGINE=InnoDB AUTO_INCREMENT=4112571 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_sector_rot_baseline_t`
--

DROP TABLE IF EXISTS `cn_sector_rot_baseline_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_sector_rot_baseline_t` (
  `BASELINE_KEY` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `RUN_ID` varchar(128) COLLATE utf8mb4_unicode_ci NOT NULL,
  `VERSION_TAG` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `IS_ACTIVE` int NOT NULL DEFAULT '0',
  `NOTES` varchar(400) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `UPDATED_AT` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`BASELINE_KEY`),
  UNIQUE KEY `UK_CN_SECTOR_ROT_BASELINE_RUN` (`RUN_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_sector_rot_bt_daily_t`
--

DROP TABLE IF EXISTS `cn_sector_rot_bt_daily_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_sector_rot_bt_daily_t` (
  `TRADE_DATE` date NOT NULL,
  `RUN_ID` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `N_POS` int NOT NULL,
  `K_USED` int NOT NULL,
  `PORT_RET_1` decimal(18,10) DEFAULT NULL,
  `TURNOVER` decimal(18,10) DEFAULT NULL,
  `COST` decimal(18,10) DEFAULT NULL,
  `NET_RET` decimal(18,10) DEFAULT NULL,
  `NAV` decimal(18,10) DEFAULT NULL,
  `EXPOSED_FLAG` int NOT NULL,
  `CREATED_AT` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`RUN_ID`,`TRADE_DATE`),
  KEY `IX_SECTOR_ROT_BT_DT` (`TRADE_DATE`),
  KEY `idx_bt_run_trade` (`RUN_ID`,`TRADE_DATE`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_sector_rot_bt_pos_t`
--

DROP TABLE IF EXISTS `cn_sector_rot_bt_pos_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_sector_rot_bt_pos_t` (
  `TRADE_DATE` date DEFAULT NULL,
  `SECTOR_TYPE` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `SECTOR_ID` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `SECTOR_NAME` varchar(200) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ENTRY_SCORE` bigint DEFAULT NULL,
  `N_POS` bigint DEFAULT NULL,
  `WEIGHT` bigint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `cn_sector_rot_pos_daily_t`
--

DROP TABLE IF EXISTS `cn_sector_rot_pos_daily_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_sector_rot_pos_daily_t` (
  `TRADE_DATE` date NOT NULL,
  `SECTOR_TYPE` varchar(16) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_ID` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_NAME` varchar(128) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `W` decimal(12,8) NOT NULL,
  `W_RAW` decimal(12,8) DEFAULT NULL,
  `WEIGHT_MODE` varchar(16) COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'EQW',
  `ACTION` varchar(16) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `STATE` varchar(16) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ENERGY_SCORE` decimal(18,6) DEFAULT NULL,
  `ENERGY_PCT` decimal(12,8) DEFAULT NULL,
  `ENTRY_RANK` int DEFAULT NULL,
  `HOLD_DAYS` int NOT NULL DEFAULT '1',
  `MIN_HOLD` int NOT NULL,
  `REBALANCE_FREQ` int NOT NULL,
  `IS_REBALANCE_DAY` int NOT NULL DEFAULT '0',
  `CAN_TRADE_TODAY` int NOT NULL DEFAULT '0',
  `EXIT_FLAG` int NOT NULL DEFAULT '0',
  `EXIT_REASON` varchar(64) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ENERGY_BELOW_CNT` int NOT NULL DEFAULT '0',
  `DOWNGRADE_CNT` int NOT NULL DEFAULT '0',
  `RUN_ID` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `PARAMS_HASH` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `CREATED_AT` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `EXIT_ALLOWED_FLAG` int NOT NULL DEFAULT '0',
  `EXIT_TRIGGERED_FLAG` int NOT NULL DEFAULT '0',
  `EXIT_EXECUTED_FLAG` int NOT NULL DEFAULT '0',
  PRIMARY KEY (`RUN_ID`,`TRADE_DATE`,`SECTOR_TYPE`,`SECTOR_ID`),
  KEY `IX_ROT_POS_DAILY_DT` (`TRADE_DATE`),
  KEY `IX_ROT_POS_DAILY_RUN_DT` (`RUN_ID`,`TRADE_DATE`),
  KEY `IX_ROT_POS_DAILY_RUN_SEC` (`RUN_ID`,`SECTOR_TYPE`,`SECTOR_ID`,`TRADE_DATE`),
  KEY `IX_SRPOS_DT_TYPE_ID` (`TRADE_DATE`,`SECTOR_TYPE`,`SECTOR_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary view structure for view `cn_sector_rotation_named_v`
--

DROP TABLE IF EXISTS `cn_sector_rotation_named_v`;
/*!50001 DROP VIEW IF EXISTS `cn_sector_rotation_named_v`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `cn_sector_rotation_named_v` AS SELECT 
 1 AS `trade_date`,
 1 AS `trading_date`,
 1 AS `sector_type`,
 1 AS `sector_id`,
 1 AS `sector_name`,
 1 AS `state`,
 1 AS `score`,
 1 AS `members`,
 1 AS `cov_5`,
 1 AS `cov_10`,
 1 AS `cov_20`,
 1 AS `mom_5`,
 1 AS `mom_10`,
 1 AS `mom_20`,
 1 AS `amt_impulse`,
 1 AS `up_ratio`,
 1 AS `up_ma5`,
 1 AS `up_slope`*/;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `cn_sector_rotation_ranked_t`
--

DROP TABLE IF EXISTS `cn_sector_rotation_ranked_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_sector_rotation_ranked_t` (
  `TRADE_DATE` date NOT NULL,
  `SECTOR_TYPE` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_ID` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_NAME` varchar(200) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `STATE` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `TIER` varchar(10) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `THEME_GROUP` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `THEME_RANK` decimal(38,10) DEFAULT NULL,
  `SCORE` decimal(38,10) DEFAULT NULL,
  `CONFIRM_STREAK` decimal(38,10) DEFAULT NULL,
  `AMT_IMPULSE` decimal(38,10) DEFAULT NULL,
  `UP_MA5` decimal(38,10) DEFAULT NULL,
  `UP_RATIO` decimal(38,10) DEFAULT NULL,
  `CREATED_AT` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`TRADE_DATE`,`SECTOR_TYPE`,`SECTOR_ID`),
  KEY `idx_rank_trade_sector` (`TRADE_DATE`,`SECTOR_TYPE`,`SECTOR_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary view structure for view `cn_sector_rotation_ranked_v_too_slow`
--

DROP TABLE IF EXISTS `cn_sector_rotation_ranked_v_too_slow`;
/*!50001 DROP VIEW IF EXISTS `cn_sector_rotation_ranked_v_too_slow`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `cn_sector_rotation_ranked_v_too_slow` AS SELECT 
 1 AS `trade_date`,
 1 AS `trading_date`,
 1 AS `sector_type`,
 1 AS `sector_id`,
 1 AS `sector_name`,
 1 AS `theme_group`,
 1 AS `state`,
 1 AS `confirm_streak`,
 1 AS `tier`,
 1 AS `tier_pri`,
 1 AS `score`,
 1 AS `members`,
 1 AS `cov_5`,
 1 AS `cov_10`,
 1 AS `cov_20`,
 1 AS `mom_5`,
 1 AS `mom_10`,
 1 AS `mom_20`,
 1 AS `amt_impulse`,
 1 AS `up_ratio`,
 1 AS `up_ma5`,
 1 AS `up_slope`,
 1 AS `theme_rank`,
 1 AS `theme_flag`*/;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `cn_sector_rotation_signal_t`
--

DROP TABLE IF EXISTS `cn_sector_rotation_signal_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_sector_rotation_signal_t` (
  `SIGNAL_DATE` date NOT NULL,
  `SECTOR_TYPE` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_ID` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL,
  `SECTOR_NAME` varchar(200) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ACTION` varchar(10) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `ENTRY_RANK` decimal(38,10) DEFAULT NULL,
  `ENTRY_CNT` decimal(38,10) DEFAULT NULL,
  `WEIGHT_SUGGESTED` decimal(38,10) DEFAULT NULL,
  `SCORE` decimal(38,10) DEFAULT NULL,
  `STATE` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `TRANSITION` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`SIGNAL_DATE`,`SECTOR_TYPE`,`SECTOR_ID`),
  KEY `IDX_SECTOR_ROT_SIGNAL_DT` (`SIGNAL_DATE`,`ACTION`,`ENTRY_RANK`),
  KEY `IX_SRSIG_DT_TYPE_STATE_ACT` (`SIGNAL_DATE`,`SECTOR_TYPE`,`STATE`,`ACTION`,`SECTOR_ID`),
  KEY `idx_signal_date_action` (`SIGNAL_DATE`,`ACTION`,`SECTOR_TYPE`,`SECTOR_ID`),
  KEY `idx_signal_sector_date` (`SECTOR_TYPE`,`SECTOR_ID`,`SIGNAL_DATE`),
  KEY `idx_signal_date_sector` (`SIGNAL_DATE`,`SECTOR_TYPE`,`SECTOR_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary view structure for view `cn_sector_rotation_state_v`
--

DROP TABLE IF EXISTS `cn_sector_rotation_state_v`;
/*!50001 DROP VIEW IF EXISTS `cn_sector_rotation_state_v`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `cn_sector_rotation_state_v` AS SELECT 
 1 AS `sector_type`,
 1 AS `sector_id`,
 1 AS `trade_date`,
 1 AS `members`,
 1 AS `amount_sum`,
 1 AS `eqw_ret`,
 1 AS `up_ratio`,
 1 AS `mom_5`,
 1 AS `mom_10`,
 1 AS `mom_20`,
 1 AS `amt_impulse`,
 1 AS `up_ma5`,
 1 AS `up_slope`,
 1 AS `cov_5`,
 1 AS `cov_10`,
 1 AS `cov_20`,
 1 AS `state`,
 1 AS `score`*/;
SET character_set_client = @saved_cs_client;

--
-- Temporary view structure for view `cn_stock_active_universe_v`
--

DROP TABLE IF EXISTS `cn_stock_active_universe_v`;
/*!50001 DROP VIEW IF EXISTS `cn_stock_active_universe_v`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `cn_stock_active_universe_v` AS SELECT 
 1 AS `symbol`,
 1 AS `is_active`,
 1 AS `inactive_reason`,
 1 AS `first_trade_date`,
 1 AS `last_trade_date`,
 1 AS `recent_trade_days`,
 1 AS `updated_at`*/;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `cn_stock_daily_price`
--

DROP TABLE IF EXISTS `cn_stock_daily_price`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_stock_daily_price` (
  `SYMBOL` varchar(10) COLLATE utf8mb4_unicode_ci NOT NULL,
  `TRADE_DATE` date NOT NULL,
  `OPEN` double DEFAULT NULL,
  `CLOSE` double DEFAULT NULL,
  `PRE_CLOSE` double DEFAULT NULL,
  `HIGH` double DEFAULT NULL,
  `LOW` double DEFAULT NULL,
  `VOLUME` bigint DEFAULT NULL,
  `AMOUNT` bigint DEFAULT NULL,
  `AMPLITUDE` double DEFAULT NULL,
  `CHG_PCT` double DEFAULT NULL,
  `CHANGE` double DEFAULT NULL,
  `TURNOVER_RATE` double DEFAULT NULL,
  `SOURCE` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `WINDOW_START` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `CREATED_AT` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `EXCHANGE` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `NAME` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`SYMBOL`,`TRADE_DATE`),
  KEY `cn_stock_daily_price_SYMBOL_IDX` (`SYMBOL`) USING BTREE,
  KEY `cn_stock_daily_price_TRADE_DATE_IDX` (`TRADE_DATE`) USING BTREE,
  KEY `idx_stock_trade_symbol` (`TRADE_DATE`,`SYMBOL`),
  KEY `idx_date_symbol_amount_close` (`TRADE_DATE`,`SYMBOL`,`AMOUNT`,`CLOSE`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary view structure for view `cn_stock_daily_price_active_v`
--

DROP TABLE IF EXISTS `cn_stock_daily_price_active_v`;
/*!50001 DROP VIEW IF EXISTS `cn_stock_daily_price_active_v`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `cn_stock_daily_price_active_v` AS SELECT 
 1 AS `SYMBOL`,
 1 AS `TRADE_DATE`,
 1 AS `OPEN`,
 1 AS `CLOSE`,
 1 AS `PRE_CLOSE`,
 1 AS `HIGH`,
 1 AS `LOW`,
 1 AS `VOLUME`,
 1 AS `AMOUNT`,
 1 AS `AMPLITUDE`,
 1 AS `CHG_PCT`,
 1 AS `CHANGE`,
 1 AS `TURNOVER_RATE`,
 1 AS `SOURCE`,
 1 AS `WINDOW_START`,
 1 AS `CREATED_AT`,
 1 AS `EXCHANGE`,
 1 AS `NAME`*/;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `cn_stock_universe_status_t`
--

DROP TABLE IF EXISTS `cn_stock_universe_status_t`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cn_stock_universe_status_t` (
  `symbol` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL,
  `is_active` tinyint(1) NOT NULL DEFAULT '1',
  `inactive_reason` varchar(64) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `first_trade_date` date DEFAULT NULL,
  `last_trade_date` date DEFAULT NULL,
  `recent_trade_days` int NOT NULL DEFAULT '0',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`symbol`),
  KEY `idx_csus_active` (`is_active`),
  KEY `idx_csus_last_trade` (`last_trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary view structure for view `v_rotation_entry_exec_v1`
--

DROP TABLE IF EXISTS `v_rotation_entry_exec_v1`;
/*!50001 DROP VIEW IF EXISTS `v_rotation_entry_exec_v1`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `v_rotation_entry_exec_v1` AS SELECT 
 1 AS `p_run_id`,
 1 AS `signal_date`,
 1 AS `exec_enter_date`,
 1 AS `sector_type`,
 1 AS `sector_id`,
 1 AS `sector_name`,
 1 AS `entry_rank`,
 1 AS `entry_cnt`,
 1 AS `weight_suggested`,
 1 AS `signal_score`,
 1 AS `energy_score`,
 1 AS `energy_pct`,
 1 AS `energy_tier`,
 1 AS `state`,
 1 AS `transition`*/;
SET character_set_client = @saved_cs_client;

--
-- Temporary view structure for view `v_rotation_exit_exec_v1`
--

DROP TABLE IF EXISTS `v_rotation_exit_exec_v1`;
/*!50001 DROP VIEW IF EXISTS `v_rotation_exit_exec_v1`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `v_rotation_exit_exec_v1` AS SELECT 
 1 AS `p_run_id`,
 1 AS `signal_date`,
 1 AS `exec_exit_date`,
 1 AS `sector_type`,
 1 AS `sector_id`,
 1 AS `sector_name`,
 1 AS `state`,
 1 AS `transition`,
 1 AS `entry_rank`,
 1 AS `signal_score`,
 1 AS `enter_signal_date`,
 1 AS `exec_enter_date`,
 1 AS `exit_exec_status`*/;
SET character_set_client = @saved_cs_client;

--
-- Dumping routines for database 'cn_market'
--
/*!50003 DROP PROCEDURE IF EXISTS `SP_BACKFILL_ROT_BT_FROM_PRICE` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `SP_BACKFILL_ROT_BT_FROM_PRICE`(
    IN p_run_id VARCHAR(64),
    IN p_end_date DATE,
    IN p_force TINYINT
)
proc: BEGIN
    DECLARE v_run_id VARCHAR(64);
    DECLARE v_end_date DATE;
    DECLARE v_min_px_date DATE;
    DECLARE v_last_bt_date DATE;
    DECLARE v_start_date DATE;
    DECLARE v_base_nav DECIMAL(18,10);

    SET v_run_id = COALESCE(NULLIF(p_run_id, ''), 'SR_LIVE_DEFAULT');
    SET v_end_date = COALESCE(p_end_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));
    SET v_min_px_date = (SELECT MIN(trade_date) FROM cn_stock_daily_price);

    IF v_end_date IS NULL OR v_min_px_date IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'SP_BACKFILL_ROT_BT_FROM_PRICE: price calendar is empty';
    END IF;

    SET v_last_bt_date = (
        SELECT MAX(b.trade_date)
        FROM cn_sector_rot_bt_daily_t b
        WHERE b.run_id = v_run_id
          AND b.trade_date <= v_end_date
    );

    IF IFNULL(p_force, 0) = 1 THEN
        SET v_start_date = COALESCE(v_last_bt_date, v_min_px_date);
        DELETE FROM cn_sector_rot_bt_daily_t
        WHERE run_id = v_run_id
          AND trade_date BETWEEN v_start_date AND v_end_date;
        SET v_base_nav = (
            SELECT b.nav
            FROM cn_sector_rot_bt_daily_t b
            WHERE b.run_id = v_run_id
              AND b.trade_date < v_start_date
              AND b.nav IS NOT NULL
            ORDER BY b.trade_date DESC
            LIMIT 1
        );
    ELSE
        IF v_last_bt_date IS NULL THEN
            SET v_start_date = v_min_px_date;
            SET v_base_nav = 1.0000000000;
        ELSE
            SET v_start_date = DATE_ADD(v_last_bt_date, INTERVAL 1 DAY);
            SET v_base_nav = (
                SELECT b.nav
                FROM cn_sector_rot_bt_daily_t b
                WHERE b.run_id = v_run_id
                  AND b.trade_date = v_last_bt_date
                  AND b.nav IS NOT NULL
                LIMIT 1
            );
            IF v_base_nav IS NULL THEN
                SET v_base_nav = (
                    SELECT b.nav
                    FROM cn_sector_rot_bt_daily_t b
                    WHERE b.run_id = v_run_id
                      AND b.trade_date <= v_last_bt_date
                      AND b.nav IS NOT NULL
                    ORDER BY b.trade_date DESC
                    LIMIT 1
                );
            END IF;
        END IF;
    END IF;

    IF v_start_date > v_end_date THEN
        LEAVE proc;
    END IF;

    SET v_base_nav = COALESCE(v_base_nav, 1.0000000000);

    INSERT INTO cn_sector_rot_bt_daily_t (
        trade_date, run_id, n_pos, k_used, port_ret_1, turnover, cost, net_ret, nav, exposed_flag, created_at
    )
    SELECT
        d.trade_date,
        v_run_id,
        IFNULL(pz.n_pos, 0) AS n_pos,
        IFNULL(sg.k_used, 0) AS k_used,
        CAST(0 AS DECIMAL(18,10)) AS port_ret_1,
        CAST(0 AS DECIMAL(18,10)) AS turnover,
        CAST(0 AS DECIMAL(18,10)) AS cost,
        CAST(0 AS DECIMAL(18,10)) AS net_ret,
        v_base_nav AS nav,
        CASE WHEN IFNULL(pz.n_pos, 0) > 0 THEN 1 ELSE 0 END AS exposed_flag,
        NOW()
    FROM (
        SELECT DISTINCT p.trade_date
        FROM cn_stock_daily_price p
        WHERE p.trade_date BETWEEN v_start_date AND v_end_date
    ) d
    LEFT JOIN (
        SELECT trade_date, COUNT(*) AS n_pos
        FROM cn_sector_rot_pos_daily_t
        WHERE run_id = v_run_id
          AND w > 0
          AND trade_date BETWEEN v_start_date AND v_end_date
        GROUP BY trade_date
    ) pz
      ON pz.trade_date = d.trade_date
    LEFT JOIN (
        SELECT signal_date AS trade_date, MAX(CAST(IFNULL(entry_cnt, 0) AS SIGNED)) AS k_used
        FROM cn_sector_rotation_signal_t
        WHERE signal_date BETWEEN v_start_date AND v_end_date
          AND action = 'ENTER'
        GROUP BY signal_date
    ) sg
      ON sg.trade_date = d.trade_date
    WHERE NOT EXISTS (
        SELECT 1
        FROM cn_sector_rot_bt_daily_t b
        WHERE b.run_id = v_run_id
          AND b.trade_date = d.trade_date
    )
    ORDER BY d.trade_date;
END proc ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `sp_backfill_sector_eod_hist_monthly` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `sp_backfill_sector_eod_hist_monthly`(
    IN p_start_date DATE,
    IN p_end_date DATE,
    IN p_top_pct DECIMAL(10, 6),
    IN p_breadth_min DECIMAL(10, 6)
)
BEGIN
    DECLARE v_cur_start DATE;
    DECLARE v_cur_end DATE;
    DECLARE v_call_failed TINYINT DEFAULT 0;
    DECLARE v_errmsg TEXT DEFAULT NULL;

    DECLARE CONTINUE HANDLER FOR SQLSTATE '45000'
    BEGIN
        SET v_call_failed = 1;
        GET DIAGNOSTICS CONDITION 1 v_errmsg = MESSAGE_TEXT;
    END;

    IF p_start_date IS NULL OR p_end_date IS NULL OR p_start_date > p_end_date THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'invalid date range';
    END IF;

    SET v_cur_start = DATE_FORMAT(p_start_date, '%Y-%m-01');

    WHILE v_cur_start <= p_end_date DO
        SET v_cur_end = LAST_DAY(v_cur_start);
        IF v_cur_end > p_end_date THEN
            SET v_cur_end = p_end_date;
        END IF;

        SET v_call_failed = 0;
        SET v_errmsg = NULL;
        CALL sp_refresh_sector_eod_hist(v_cur_start, v_cur_end, p_top_pct, p_breadth_min);

        IF v_call_failed = 1 THEN
            SELECT
                v_cur_start AS batch_from,
                v_cur_end AS batch_to,
                'SKIPPED' AS batch_status,
                COALESCE(v_errmsg, 'source empty') AS detail,
                NULL AS rows_before,
                NULL AS rows_after,
                NULL AS rows_delta;
        ELSE
            SELECT
                v_cur_start AS batch_from,
                v_cur_end AS batch_to,
                'OK' AS batch_status,
                '' AS detail,
                NULL AS rows_before,
                NULL AS rows_after,
                NULL AS rows_delta;
        END IF;

        SET v_cur_start = DATE_ADD(v_cur_end, INTERVAL 1 DAY);
    END WHILE;
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `sp_build_board_member_map` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `sp_build_board_member_map`(
    IN p_start_date DATE,
    IN p_end_date DATE
)
proc: BEGIN
    DECLARE v_start DATE;
    DECLARE v_end DATE;

    SET v_start = COALESCE(p_start_date, (SELECT MIN(trade_date) FROM cn_stock_daily_price));
    SET v_end = COALESCE(p_end_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));

    IF v_start IS NULL OR v_end IS NULL OR v_start > v_end THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'sp_build_board_member_map: invalid date range';
    END IF;

    DELETE FROM cn_board_member_map_d
    WHERE trade_date BETWEEN v_start AND v_end;

    INSERT IGNORE INTO cn_board_member_map_d (trade_date, sector_type, sector_id, symbol)
    SELECT d.trade_date, 'CONCEPT', h.concept_id, h.symbol
    FROM (
        SELECT DISTINCT trade_date
        FROM cn_stock_daily_price
        WHERE trade_date BETWEEN v_start AND v_end
    ) d
    JOIN cn_board_concept_member_hist h
      ON d.trade_date >= h.valid_from
     AND d.trade_date <= COALESCE(h.valid_to, '9999-12-31');

    INSERT IGNORE INTO cn_board_member_map_d (trade_date, sector_type, sector_id, symbol)
    SELECT d.trade_date, 'INDUSTRY', h.board_id, h.symbol
    FROM (
        SELECT DISTINCT trade_date
        FROM cn_stock_daily_price
        WHERE trade_date BETWEEN v_start AND v_end
    ) d
    JOIN cn_board_industry_member_hist h
      ON d.trade_date >= h.valid_from
     AND d.trade_date <= COALESCE(h.valid_to, '9999-12-31');
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE`(
    IN p_trade_date DATE
)
proc: BEGIN
    DECLARE v_trade_date DATE;
    SET v_trade_date = COALESCE(p_trade_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));

    IF v_trade_date IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE: trade_date is NULL';
    END IF;

    INSERT INTO cn_sector_rotation_ranked_t (
        trade_date, sector_type, sector_id, sector_name,
        state, tier, theme_group, theme_rank, score, confirm_streak,
        amt_impulse, up_ma5, up_ratio, created_at
    )
    SELECT
        t.trade_date,
        t.sector_type,
        t.sector_id,
        t.sector_name,
        t.state,
        t.tier,
        t.theme_group,
        t.theme_rank,
        t.score,
        t.confirm_streak,
        t.amt_impulse,
        t.up_ma5,
        t.up_ratio,
        NOW()
    FROM cn_sector_rotation_transition_v t
    WHERE t.trade_date = v_trade_date
    ON DUPLICATE KEY UPDATE
        sector_name = VALUES(sector_name),
        state = VALUES(state),
        tier = VALUES(tier),
        theme_group = VALUES(theme_group),
        theme_rank = VALUES(theme_rank),
        score = VALUES(score),
        confirm_streak = VALUES(confirm_streak),
        amt_impulse = VALUES(amt_impulse),
        up_ma5 = VALUES(up_ma5),
        up_ratio = VALUES(up_ratio),
        created_at = NOW();
END proc ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `SP_BUILD_SECTOR_ROTATION_RANKED_LATEST` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `SP_BUILD_SECTOR_ROTATION_RANKED_LATEST`()
proc: BEGIN
    CALL SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE(NULL);
END proc ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE`(
    IN p_trade_date DATE
)
proc: BEGIN
    DECLARE v_trade_date DATE;
    SET v_trade_date = COALESCE(p_trade_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));

    IF v_trade_date IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE: trade_date is NULL';
    END IF;

    INSERT INTO cn_sector_rotation_signal_t (
        signal_date, sector_type, sector_id, sector_name, action,
        entry_rank, entry_cnt, weight_suggested, score, state, transition, created_at
    )
    WITH sig0 AS (
        SELECT
            t.trade_date AS signal_date,
            t.sector_type,
            t.sector_id,
            t.sector_name,
            CASE
                WHEN t.transition IN ('IGNITE_TO_CONFIRM', 'DIRECT_CONFIRM')
                     AND t.theme_rank = 1
                     AND t.tier = 'T1'
                     AND t.up_ma5 >= 0.52
                     AND t.amt_impulse >= 1.10 THEN 'ENTER'
                WHEN t.transition IN ('CONFIRM_TO_FADE', 'FADE_TO_OFF', 'CONFIRM_TO_OFF', 'T1_TO_T2', 'T2_TO_T3') THEN 'EXIT'
                ELSE 'WATCH'
            END AS action,
            t.score,
            t.state,
            t.transition
        FROM cn_sector_rotation_transition_v t
        WHERE t.trade_date = v_trade_date
    ),
    sig1 AS (
        SELECT
            s.*,
            CASE WHEN s.action = 'ENTER'
                 THEN ROW_NUMBER() OVER (PARTITION BY s.signal_date ORDER BY s.score DESC, s.sector_type, s.sector_id)
                 ELSE NULL
            END AS entry_rank
        FROM sig0 s
    ),
    sig2 AS (
        SELECT
            s.*,
            SUM(CASE WHEN s.action = 'ENTER' THEN 1 ELSE 0 END) OVER (PARTITION BY s.signal_date) AS entry_cnt
        FROM sig1 s
    )
    SELECT
        s.signal_date,
        s.sector_type,
        s.sector_id,
        s.sector_name,
        s.action,
        s.entry_rank,
        CASE WHEN s.action = 'ENTER' THEN s.entry_cnt ELSE NULL END AS entry_cnt,
        CASE WHEN s.action = 'ENTER' AND s.entry_cnt > 0 THEN 1.0 / s.entry_cnt ELSE NULL END AS weight_suggested,
        s.score,
        s.state,
        s.transition,
        NOW()
    FROM sig2 s
    ON DUPLICATE KEY UPDATE
        sector_name = VALUES(sector_name),
        action = VALUES(action),
        entry_rank = VALUES(entry_rank),
        entry_cnt = VALUES(entry_cnt),
        weight_suggested = VALUES(weight_suggested),
        score = VALUES(score),
        state = VALUES(state),
        transition = VALUES(transition),
        created_at = NOW();
END proc ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `SP_BUILD_SECTOR_ROTATION_SIGNAL_LATEST` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `SP_BUILD_SECTOR_ROTATION_SIGNAL_LATEST`()
proc: BEGIN
    CALL SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE(NULL);
END proc ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `sp_rebuild_rotation_20200101_20260228` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `sp_rebuild_rotation_20200101_20260228`()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE v_dt DATE;

    DECLARE cur CURSOR FOR
        SELECT DISTINCT trade_date
        FROM cn_stock_daily_price
        WHERE trade_date BETWEEN '2020-01-01' AND '2026-02-28'
        ORDER BY trade_date;

    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    OPEN cur;
    read_loop: LOOP
        FETCH cur INTO v_dt;
        IF done = 1 THEN
            LEAVE read_loop;
        END IF;

        CALL SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE(v_dt);
        CALL SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE(v_dt);

        -- 如果这次也要重建回测链路，取消下面两行注释
        -- CALL SP_BACKFILL_ROT_BT_FROM_PRICE('SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS', v_dt, 0);
    END LOOP;
    CLOSE cur;

    -- 如果上面补了 bt，再执行 NAV 修复
    -- CALL SP_REPAIR_ROT_BT_NAV('SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS', 1.0000000000);
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `sp_refresh_board_member_hist` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `sp_refresh_board_member_hist`(
    IN p_asof_date DATE,
    IN p_source VARCHAR(32),
    IN p_apply_concept TINYINT,
    IN p_apply_industry TINYINT
)
proc: BEGIN
    DECLARE v_asof DATE;
    DECLARE v_source VARCHAR(32);
    DECLARE v_concept_rows BIGINT DEFAULT 0;
    DECLARE v_industry_rows BIGINT DEFAULT 0;

    SET v_asof = COALESCE(p_asof_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));
    SET v_source = COALESCE(NULLIF(p_source, ''), 'tushare');

    IF v_asof IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'sp_refresh_board_member_hist: asof_date is NULL';
    END IF;

    IF IFNULL(p_apply_concept, 1) = 1 THEN
        SELECT COUNT(*) INTO v_concept_rows
        FROM cn_board_concept_member_stg
        WHERE asof_date = v_asof;

        IF v_concept_rows = 0 THEN
            SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'sp_refresh_board_member_hist: concept staging is empty for asof_date';
        END IF;

        UPDATE cn_board_concept_member_hist h
        LEFT JOIN cn_board_concept_member_stg s
          ON s.asof_date = v_asof
         AND s.concept_id = h.concept_id
         AND s.symbol = h.symbol
        SET h.valid_to = DATE_SUB(v_asof, INTERVAL 1 DAY),
            h.source = v_source
        WHERE h.valid_to IS NULL
          AND s.concept_id IS NULL;

        INSERT INTO cn_board_concept_member_hist (concept_id, symbol, valid_from, valid_to, source)
        SELECT s.concept_id, s.symbol, v_asof, NULL, v_source
        FROM cn_board_concept_member_stg s
        LEFT JOIN cn_board_concept_member_hist h
          ON h.concept_id = s.concept_id
         AND h.symbol = s.symbol
         AND h.valid_to IS NULL
        WHERE s.asof_date = v_asof
          AND h.concept_id IS NULL;
    END IF;

    IF IFNULL(p_apply_industry, 1) = 1 THEN
        SELECT COUNT(*) INTO v_industry_rows
        FROM cn_board_industry_member_stg
        WHERE asof_date = v_asof;

        IF v_industry_rows = 0 THEN
            SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'sp_refresh_board_member_hist: industry staging is empty for asof_date';
        END IF;

        UPDATE cn_board_industry_member_hist h
        LEFT JOIN cn_board_industry_member_stg s
          ON s.asof_date = v_asof
         AND s.board_id = h.board_id
         AND s.symbol = h.symbol
        SET h.valid_to = DATE_SUB(v_asof, INTERVAL 1 DAY),
            h.source = v_source
        WHERE h.valid_to IS NULL
          AND s.board_id IS NULL;

        INSERT INTO cn_board_industry_member_hist (board_id, symbol, valid_from, valid_to, source)
        SELECT s.board_id, s.symbol, v_asof, NULL, v_source
        FROM cn_board_industry_member_stg s
        LEFT JOIN cn_board_industry_member_hist h
          ON h.board_id = s.board_id
         AND h.symbol = s.symbol
         AND h.valid_to IS NULL
        WHERE s.asof_date = v_asof
          AND h.board_id IS NULL;
    END IF;
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `SP_REFRESH_ROTATION_SNAP_ALL` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `SP_REFRESH_ROTATION_SNAP_ALL`(
    IN p_run_id VARCHAR(64),
    IN p_trade_date DATE,
    IN p_force TINYINT
)
proc: BEGIN
    DECLARE v_trade_date DATE;
    DECLARE v_run_id VARCHAR(64);
    DECLARE v_has_entry BIGINT DEFAULT 0;
    DECLARE v_has_holding BIGINT DEFAULT 0;
    DECLARE v_has_exit BIGINT DEFAULT 0;

    SET v_run_id = COALESCE(NULLIF(p_run_id, ''), 'SR_LIVE_DEFAULT');
    SET v_trade_date = COALESCE(p_trade_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));

    IF v_trade_date IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'SP_REFRESH_ROTATION_SNAP_ALL: trade_date is NULL';
    END IF;

    IF IFNULL(p_force, 0) = 1 THEN
        DELETE FROM cn_rotation_entry_snap_t WHERE run_id = v_run_id AND trade_date = v_trade_date;
        DELETE FROM cn_rotation_holding_snap_t WHERE run_id = v_run_id AND trade_date = v_trade_date;
        DELETE FROM cn_rotation_exit_snap_t WHERE run_id = v_run_id AND trade_date = v_trade_date;
    END IF;

    INSERT INTO cn_rotation_entry_snap_t (
        run_id, trade_date, sector_type, sector_id, sector_name,
        entry_rank, entry_cnt, weight_suggested, signal_score,
        energy_score, energy_pct, energy_tier, state, transition,
        source_json, created_at
    )
    SELECT
        v_run_id,
        s.signal_date,
        s.sector_type,
        s.sector_id,
        s.sector_name,
        CAST(s.entry_rank AS SIGNED),
        CAST(s.entry_cnt AS SIGNED),
        CAST(s.weight_suggested AS DECIMAL(18,8)),
        CAST(s.score AS DECIMAL(38,16)),
        CAST(e.energy_score AS DECIMAL(38,16)),
        CAST(e.energy_pct AS DECIMAL(38,16)),
        CASE
            WHEN e.energy_pct >= 0.80 THEN 'T1'
            WHEN e.energy_pct >= 0.50 THEN 'T2'
            WHEN e.energy_pct IS NULL THEN NULL
            ELSE 'T3'
        END,
        s.state,
        s.transition,
        JSON_OBJECT('source', 'SP_REFRESH_ROTATION_SNAP_ALL', 'trade_date', CAST(v_trade_date AS CHAR)),
        NOW(6)
    FROM cn_sector_rotation_signal_t s
    LEFT JOIN cn_sector_energy_v e
      ON e.trade_date = s.signal_date
     AND e.sector_type = s.sector_type
     AND e.sector_id = s.sector_id
    WHERE s.signal_date = v_trade_date
      AND s.action = 'ENTER'
    ON DUPLICATE KEY UPDATE
        sector_name = VALUES(sector_name),
        entry_rank = VALUES(entry_rank),
        entry_cnt = VALUES(entry_cnt),
        weight_suggested = VALUES(weight_suggested),
        signal_score = VALUES(signal_score),
        energy_score = VALUES(energy_score),
        energy_pct = VALUES(energy_pct),
        energy_tier = VALUES(energy_tier),
        state = VALUES(state),
        transition = VALUES(transition),
        source_json = VALUES(source_json),
        created_at = NOW(6);

    SELECT COUNT(*) INTO v_has_entry
    FROM cn_rotation_entry_snap_t
    WHERE run_id = v_run_id AND trade_date = v_trade_date;

    IF v_has_entry = 0 THEN
        INSERT INTO cn_rotation_entry_snap_t (
            run_id, trade_date, sector_type, sector_id, sector_name,
            entry_rank, entry_cnt, weight_suggested, signal_score,
            energy_score, energy_pct, energy_tier, state, transition,
            source_json, created_at
        )
        VALUES (
            v_run_id, v_trade_date, 'ALL', '-1', 'NO_ENTRY_TODAY',
            NULL, 0, NULL, NULL,
            NULL, NULL, NULL, NULL, NULL,
            JSON_OBJECT('summary', 'NO_ENTRY_TODAY'),
            NOW(6)
        )
        ON DUPLICATE KEY UPDATE
            sector_name = VALUES(sector_name),
            entry_cnt = VALUES(entry_cnt),
            source_json = VALUES(source_json),
            created_at = NOW(6);
    END IF;

    INSERT INTO cn_rotation_holding_snap_t (
        run_id, trade_date, sector_type, sector_id, sector_name,
        enter_signal_date, exec_enter_date, hold_days, min_hold_days,
        exit_signal_today, exit_transition, exit_exec_status, next_exit_eligible_date,
        source_json, created_at
    )
    SELECT
        p.run_id,
        p.trade_date,
        p.sector_type,
        p.sector_id,
        p.sector_name,
        NULL,
        NULL,
        CAST(p.hold_days AS SIGNED),
        CAST(p.min_hold AS SIGNED),
        CAST(IFNULL(p.exit_flag, 0) AS SIGNED),
        p.exit_reason,
        CASE WHEN IFNULL(p.exit_flag, 0) = 1 THEN 'PENDING' ELSE 'HOLD' END,
        DATE_ADD(p.trade_date, INTERVAL GREATEST(IFNULL(p.min_hold, 0) - IFNULL(p.hold_days, 0), 0) DAY),
        JSON_OBJECT('source', 'CN_SECTOR_ROT_POS_DAILY_T'),
        NOW(6)
    FROM cn_sector_rot_pos_daily_t p
    WHERE p.run_id = v_run_id
      AND p.trade_date = v_trade_date
      AND p.w > 0
    ON DUPLICATE KEY UPDATE
        sector_name = VALUES(sector_name),
        hold_days = VALUES(hold_days),
        min_hold_days = VALUES(min_hold_days),
        exit_signal_today = VALUES(exit_signal_today),
        exit_transition = VALUES(exit_transition),
        exit_exec_status = VALUES(exit_exec_status),
        next_exit_eligible_date = VALUES(next_exit_eligible_date),
        source_json = VALUES(source_json),
        created_at = NOW(6);

    SELECT COUNT(*) INTO v_has_holding
    FROM cn_rotation_holding_snap_t
    WHERE run_id = v_run_id AND trade_date = v_trade_date;

    IF v_has_holding = 0 THEN
        INSERT INTO cn_rotation_holding_snap_t (
            run_id, trade_date, sector_type, sector_id, sector_name,
            enter_signal_date, exec_enter_date, hold_days, min_hold_days,
            exit_signal_today, exit_transition, exit_exec_status, next_exit_eligible_date,
            source_json, created_at
        )
        VALUES (
            v_run_id, v_trade_date, 'ALL', '-1', 'NO_HOLDING_TODAY',
            NULL, NULL, 0, 0,
            0, NULL, 'NO_HOLDING', NULL,
            JSON_OBJECT('summary', 'NO_HOLDING_TODAY'),
            NOW(6)
        )
        ON DUPLICATE KEY UPDATE
            sector_name = VALUES(sector_name),
            hold_days = VALUES(hold_days),
            min_hold_days = VALUES(min_hold_days),
            exit_exec_status = VALUES(exit_exec_status),
            source_json = VALUES(source_json),
            created_at = NOW(6);
    END IF;

    INSERT INTO cn_rotation_exit_snap_t (
        run_id, trade_date, exec_exit_date, sector_type, sector_id, sector_name,
        state, transition, entry_rank, signal_score,
        enter_signal_date, exec_enter_date, hold_days, min_hold_days,
        exit_exec_status, source_json, created_at
    )
    SELECT
        v_run_id,
        s.signal_date,
        s.signal_date,
        s.sector_type,
        s.sector_id,
        s.sector_name,
        s.state,
        s.transition,
        CAST(s.entry_rank AS SIGNED),
        CAST(s.score AS DECIMAL(38,16)),
        NULL,
        NULL,
        CAST(IFNULL(p.hold_days, 0) AS SIGNED),
        CAST(IFNULL(p.min_hold, 0) AS SIGNED),
        'EXIT_TODAY',
        JSON_OBJECT('source', 'SP_REFRESH_ROTATION_SNAP_ALL', 'action', 'EXIT'),
        NOW(6)
    FROM cn_sector_rotation_signal_t s
    LEFT JOIN cn_sector_rot_pos_daily_t p
      ON p.trade_date = s.signal_date
     AND p.run_id = v_run_id
     AND p.sector_type = s.sector_type
     AND p.sector_id = s.sector_id
    WHERE s.signal_date = v_trade_date
      AND s.action = 'EXIT'
    ON DUPLICATE KEY UPDATE
        sector_name = VALUES(sector_name),
        state = VALUES(state),
        transition = VALUES(transition),
        entry_rank = VALUES(entry_rank),
        signal_score = VALUES(signal_score),
        hold_days = VALUES(hold_days),
        min_hold_days = VALUES(min_hold_days),
        exit_exec_status = VALUES(exit_exec_status),
        source_json = VALUES(source_json),
        created_at = NOW(6);

    SELECT COUNT(*) INTO v_has_exit
    FROM cn_rotation_exit_snap_t
    WHERE run_id = v_run_id AND trade_date = v_trade_date;

    IF v_has_exit = 0 THEN
        INSERT INTO cn_rotation_exit_snap_t (
            run_id, trade_date, exec_exit_date, sector_type, sector_id, sector_name,
            state, transition, entry_rank, signal_score,
            enter_signal_date, exec_enter_date, hold_days, min_hold_days,
            exit_exec_status, source_json, created_at
        )
        VALUES (
            v_run_id, v_trade_date, NULL, 'ALL', '-1', 'NO_EXIT_TODAY',
            NULL, NULL, NULL, NULL,
            NULL, NULL, 0, 0,
            'NO_EXIT',
            JSON_OBJECT('summary', 'NO_EXIT_TODAY'),
            NOW(6)
        )
        ON DUPLICATE KEY UPDATE
            sector_name = VALUES(sector_name),
            hold_days = VALUES(hold_days),
            min_hold_days = VALUES(min_hold_days),
            exit_exec_status = VALUES(exit_exec_status),
            source_json = VALUES(source_json),
            created_at = NOW(6);
    END IF;
END proc ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `sp_refresh_sector_eod_hist` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `sp_refresh_sector_eod_hist`(
    IN p_from DATE,
    IN p_to DATE,
    IN p_top_pct DECIMAL(10, 6),
    IN p_breadth_min DECIMAL(10, 6)
)
BEGIN
    IF p_from IS NULL OR p_to IS NULL OR p_from > p_to THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'invalid date range';
    END IF;

    DELETE FROM cn_sector_eod_hist_t
    WHERE trade_date BETWEEN p_from AND p_to;

    DROP TEMPORARY TABLE IF EXISTS tmp_price;
    CREATE TEMPORARY TABLE tmp_price (
        trade_date DATE NOT NULL,
        symbol VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
        amount DOUBLE NULL,
        pre_close DOUBLE NULL,
        close DOUBLE NULL,
        chg DOUBLE NULL,
        chg_pct DOUBLE NULL,
        KEY idx_tp_date_symbol (trade_date, symbol),
        KEY idx_tp_symbol_date (symbol, trade_date)
    ) ENGINE=InnoDB;

    INSERT INTO tmp_price (trade_date, symbol, amount, pre_close, close, chg, chg_pct)
    SELECT p.trade_date, p.symbol, p.amount, p.pre_close, p.close, p.change, p.chg_pct
    FROM cn_stock_daily_price_active_v p
    WHERE p.trade_date BETWEEN p_from AND p_to;

    DROP TEMPORARY TABLE IF EXISTS tmp_map;
    CREATE TEMPORARY TABLE tmp_map (
        trade_date DATE NOT NULL,
        sector_type VARCHAR(16) NOT NULL,
        sector_id VARCHAR(64) NOT NULL,
        symbol VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
        KEY idx_tm_date_symbol_type_sector (trade_date, symbol, sector_type, sector_id),
        KEY idx_tm_symbol_date (symbol, trade_date, sector_type, sector_id)
    ) ENGINE=InnoDB;

    INSERT INTO tmp_map (trade_date, sector_type, sector_id, symbol)
    SELECT m.trade_date, m.sector_type, m.sector_id, (m.symbol COLLATE utf8mb4_unicode_ci)
    FROM cn_board_member_map_d m
    WHERE m.trade_date BETWEEN p_from AND p_to
      AND m.sector_type IN ('INDUSTRY', 'CONCEPT');

    DROP TEMPORARY TABLE IF EXISTS tmp_sector_eod_raw;
    CREATE TEMPORARY TABLE tmp_sector_eod_raw (
        trade_date DATE NOT NULL,
        sector_type VARCHAR(16) NOT NULL,
        sector_id VARCHAR(64) NOT NULL,
        members INT NOT NULL,
        amount_sum DOUBLE NULL,
        eqw_ret DOUBLE NULL,
        up_ratio DOUBLE NULL,
        KEY idx_tmp1 (trade_date, sector_type, sector_id),
        KEY idx_tmp2 (sector_type, sector_id, trade_date)
    ) ENGINE=InnoDB;

    INSERT INTO tmp_sector_eod_raw (trade_date, sector_type, sector_id, members, amount_sum, eqw_ret, up_ratio)
    SELECT z.trade_date, z.sector_type, z.sector_id,
           COUNT(*) AS members,
           SUM(IFNULL(z.amount, 0)) AS amount_sum,
           AVG(z.ret_eff) AS eqw_ret,
           AVG(CASE WHEN z.ret_eff > 0 THEN 1 ELSE 0 END) AS up_ratio
    FROM (
        SELECT p.trade_date, m.sector_type, m.sector_id, p.amount,
               CASE
                   WHEN COALESCE(p.pre_close, p.close - p.chg) IS NOT NULL
                        AND COALESCE(p.pre_close, p.close - p.chg) <> 0
                       THEN (p.close / COALESCE(p.pre_close, p.close - p.chg)) - 1
                   WHEN p.chg_pct IS NOT NULL
                       THEN CASE WHEN ABS(p.chg_pct) > 1 THEN p.chg_pct / 100 ELSE p.chg_pct END
                   ELSE NULL
               END AS ret_eff
        FROM tmp_price p
        JOIN tmp_map m ON m.trade_date = p.trade_date AND m.symbol = p.symbol
    ) z
    GROUP BY z.trade_date, z.sector_type, z.sector_id;

    IF NOT EXISTS (SELECT 1 FROM tmp_sector_eod_raw LIMIT 1) THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = 'no source rows found for requested date range (check price/mapping tables)';
    END IF;

    INSERT INTO cn_sector_eod_hist_t (
        trade_date, sector_type, sector_id,
        members, amount_sum,
        sector_close, sector_ma20, sector_ret20, up_ratio,
        score, rank_pct,
        cond1_trend, cond2_rank, cond3_breadth, cond_count, sector_pass
    )
    WITH base AS (
        SELECT r.trade_date, r.sector_type, r.sector_id, r.members, r.amount_sum,
               NULL AS sector_close, NULL AS sector_ma20, NULL AS sector_ret20,
               CAST(CASE WHEN r.up_ratio IS NULL OR r.up_ratio <> r.up_ratio THEN NULL WHEN r.up_ratio < 0 THEN 0 WHEN r.up_ratio > 1 THEN 1 ELSE r.up_ratio END AS DECIMAL(10, 6)) AS up_ratio,
               CAST(CASE WHEN r.eqw_ret IS NULL OR r.eqw_ret <> r.eqw_ret THEN 0 WHEN r.eqw_ret > 99999 THEN 99999 WHEN r.eqw_ret < -99999 THEN -99999 ELSE r.eqw_ret END * 100 AS DECIMAL(20, 8)) AS score,
               CASE WHEN IFNULL(r.eqw_ret, 0) > 0 THEN 1 ELSE 0 END AS cond1_trend,
               CASE WHEN IFNULL(r.up_ratio, 0) > p_breadth_min THEN 1 ELSE 0 END AS cond3_breadth
        FROM tmp_sector_eod_raw r
    ), ranked AS (
        SELECT b.*, CAST(CASE WHEN pr IS NULL THEN 1 WHEN pr < 0 THEN 0 WHEN pr > 1 THEN 1 ELSE pr END AS DECIMAL(10,6)) AS rank_pct
        FROM (
            SELECT b.*, PERCENT_RANK() OVER (PARTITION BY b.trade_date ORDER BY b.score DESC) AS pr
            FROM base b
        ) b
    )
    SELECT
        trade_date, sector_type, sector_id, members, amount_sum,
        sector_close, sector_ma20, sector_ret20, up_ratio,
        CAST(CASE WHEN score > 999999999999.99999999 THEN 999999999999.99999999 WHEN score < -999999999999.99999999 THEN -999999999999.99999999 ELSE score END AS DECIMAL(20,8)) AS score,
        rank_pct,
        cond1_trend,
        CASE WHEN rank_pct <= p_top_pct THEN 1 ELSE 0 END AS cond2_rank,
        cond3_breadth,
        cond1_trend + (CASE WHEN rank_pct <= p_top_pct THEN 1 ELSE 0 END) + cond3_breadth AS cond_count,
        CASE WHEN cond1_trend + (CASE WHEN rank_pct <= p_top_pct THEN 1 ELSE 0 END) + cond3_breadth >= 2 THEN 1 ELSE 0 END AS sector_pass
    FROM ranked;
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `sp_refresh_stock_universe_status` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `sp_refresh_stock_universe_status`(
    IN p_asof_date DATE,
    IN p_recent_days INT,
    IN p_min_trade_days INT
)
proc: BEGIN
    DECLARE v_asof DATE;
    DECLARE v_recent_days INT;
    DECLARE v_min_trade_days INT;

    SET v_asof = COALESCE(p_asof_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));
    SET v_recent_days = GREATEST(IFNULL(p_recent_days, 30), 1);
    SET v_min_trade_days = GREATEST(IFNULL(p_min_trade_days, 1), 1);

    IF v_asof IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'sp_refresh_stock_universe_status: price calendar empty';
    END IF;

    INSERT INTO cn_stock_universe_status_t (
        symbol, is_active, inactive_reason, first_trade_date, last_trade_date, recent_trade_days, updated_at
    )
    SELECT
        x.symbol,
        CASE
            WHEN x.last_trade_date >= DATE_SUB(v_asof, INTERVAL v_recent_days DAY)
                 AND x.recent_trade_days >= v_min_trade_days THEN 1
            ELSE 0
        END AS is_active,
        CASE
            WHEN x.last_trade_date >= DATE_SUB(v_asof, INTERVAL v_recent_days DAY)
                 AND x.recent_trade_days >= v_min_trade_days THEN NULL
            ELSE 'LONG_INACTIVE_OR_DELISTED'
        END AS inactive_reason,
        x.first_trade_date,
        x.last_trade_date,
        x.recent_trade_days,
        NOW()
    FROM (
        SELECT
            p.symbol,
            MIN(p.trade_date) AS first_trade_date,
            MAX(p.trade_date) AS last_trade_date,
            COUNT(DISTINCT CASE
                WHEN p.trade_date BETWEEN DATE_SUB(v_asof, INTERVAL v_recent_days DAY) AND v_asof
                THEN p.trade_date
                ELSE NULL
            END) AS recent_trade_days
        FROM cn_stock_daily_price p
        GROUP BY p.symbol
    ) x
    ON DUPLICATE KEY UPDATE
        is_active = VALUES(is_active),
        inactive_reason = VALUES(inactive_reason),
        first_trade_date = VALUES(first_trade_date),
        last_trade_date = VALUES(last_trade_date),
        recent_trade_days = VALUES(recent_trade_days),
        updated_at = NOW();
END proc ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `SP_REPAIR_ROT_BT_NAV` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `SP_REPAIR_ROT_BT_NAV`(
    IN p_run_id VARCHAR(128),
    IN p_default_nav DECIMAL(18,10)
)
proc: BEGIN
    DECLARE v_run_id VARCHAR(128);
    DECLARE v_default_nav DECIMAL(18,10);

    SET v_run_id = NULLIF(p_run_id, '');
    SET v_default_nav = COALESCE(p_default_nav, 1.0000000000);

    DROP TEMPORARY TABLE IF EXISTS tmp_bt_nav_fix;
    CREATE TEMPORARY TABLE tmp_bt_nav_fix (
        run_id VARCHAR(64) NOT NULL,
        trade_date DATE NOT NULL,
        nav_new DECIMAL(18,10) NOT NULL,
        PRIMARY KEY (run_id, trade_date)
    ) ENGINE=MEMORY;

    INSERT INTO tmp_bt_nav_fix (run_id, trade_date, nav_new)
    SELECT
        b.run_id,
        b.trade_date,
        COALESCE((
            SELECT b2.nav
            FROM cn_sector_rot_bt_daily_t b2
            WHERE b2.run_id = b.run_id
              AND b2.trade_date < b.trade_date
              AND b2.nav IS NOT NULL
            ORDER BY b2.trade_date DESC
            LIMIT 1
        ), v_default_nav) AS nav_new
    FROM cn_sector_rot_bt_daily_t b
    WHERE b.nav IS NULL
      AND (v_run_id IS NULL OR b.run_id = v_run_id);

    UPDATE cn_sector_rot_bt_daily_t b
    JOIN tmp_bt_nav_fix t
      ON t.run_id = b.run_id
     AND t.trade_date = b.trade_date
    SET b.nav = t.nav_new
    WHERE b.nav IS NULL;

    DROP TEMPORARY TABLE IF EXISTS tmp_bt_nav_fix;
END proc ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `SP_ROTATION_DAILY_REFRESH` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `SP_ROTATION_DAILY_REFRESH`(
    IN p_run_id VARCHAR(64),
    IN p_trade_date DATE,
    IN p_force TINYINT,
    IN p_refresh_energy TINYINT
)
proc: BEGIN
    DECLARE v_trade_date DATE;
    DECLARE v_run_id VARCHAR(64);

    SET v_run_id = COALESCE(NULLIF(p_run_id, ''), 'SR_LIVE_DEFAULT');
    SET v_trade_date = COALESCE(p_trade_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));

    IF v_trade_date IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'SP_ROTATION_DAILY_REFRESH: trade_date is NULL';
    END IF;

    IF IFNULL(p_refresh_energy, 1) = 1 THEN
        DO 1;
    END IF;

    -- Incremental sector index-level daily facts
    CALL sp_refresh_sector_eod_hist(v_trade_date, v_trade_date, 0.30, 0.60);

    -- Incremental rotation signal/ranked for exact trade date
    CALL SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE(v_trade_date);
    CALL SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE(v_trade_date);

    CALL SP_BACKFILL_ROT_BT_FROM_PRICE(v_run_id, v_trade_date, IFNULL(p_force, 0));
    CALL SP_REPAIR_ROT_BT_NAV(v_run_id, 1.0000000000);
    CALL SP_REFRESH_ROTATION_SNAP_ALL(v_run_id, v_trade_date, IFNULL(p_force, 0));
END proc ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `SP_VALIDATE_AGAINST_BASELINE` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `SP_VALIDATE_AGAINST_BASELINE`(
    IN p_run_id VARCHAR(128),
    IN p_baseline_id VARCHAR(64),
    IN p_min_alpha DECIMAL(18,10)
)
proc: BEGIN
    DECLARE v_run_id VARCHAR(128);
    DECLARE v_baseline_key VARCHAR(64);
    DECLARE v_baseline_run_id VARCHAR(128);
    DECLARE v_asof DATE;
    DECLARE v_nav_run DECIMAL(18,10);
    DECLARE v_nav_base DECIMAL(18,10);
    DECLARE v_alpha DECIMAL(18,10);
    DECLARE v_decision VARCHAR(16);
    DECLARE v_reason VARCHAR(4000);
    DECLARE v_min_alpha DECIMAL(18,10);

    SET v_run_id = COALESCE(NULLIF(p_run_id, ''), 'SR_LIVE_DEFAULT');
    SET v_baseline_key = COALESCE(NULLIF(p_baseline_id, ''), 'DEFAULT_BASELINE');
    SET v_min_alpha = COALESCE(p_min_alpha, 0);

    SELECT r.run_id
    INTO v_baseline_run_id
    FROM cn_baseline_registry_t r
    WHERE r.baseline_key = v_baseline_key
      AND IFNULL(r.is_active, 1) = 1
    LIMIT 1;

    IF v_baseline_run_id IS NULL THEN
        SELECT b.run_id
        INTO v_baseline_run_id
        FROM cn_sector_rot_baseline_t b
        WHERE b.baseline_key = v_baseline_key
          AND IFNULL(b.is_active, 1) = 1
        LIMIT 1;
    END IF;

    IF v_baseline_run_id IS NULL THEN
        SET v_decision = 'FAIL';
        SET v_reason = CONCAT('BASELINE_NOT_FOUND:', v_baseline_key);
        SET v_asof = CURRENT_DATE();
        SET v_nav_run = NULL;
        SET v_nav_base = NULL;
        SET v_alpha = NULL;
    ELSE
        SET v_asof = (
            SELECT LEAST(
                COALESCE((SELECT MAX(trade_date) FROM cn_sector_rot_bt_daily_t WHERE run_id = v_run_id), CURRENT_DATE()),
                COALESCE((SELECT MAX(trade_date) FROM cn_sector_rot_bt_daily_t WHERE run_id = v_baseline_run_id), CURRENT_DATE())
            )
        );

        SELECT b.nav
        INTO v_nav_run
        FROM cn_sector_rot_bt_daily_t b
        WHERE b.run_id = v_run_id
          AND b.trade_date = v_asof
        LIMIT 1;

        SELECT b.nav
        INTO v_nav_base
        FROM cn_sector_rot_bt_daily_t b
        WHERE b.run_id = v_baseline_run_id
          AND b.trade_date = v_asof
        LIMIT 1;

        IF v_nav_run IS NULL THEN
            SET v_decision = 'FAIL';
            SET v_reason = CONCAT('RUN_NAV_MISSING@', CAST(v_asof AS CHAR));
            SET v_alpha = NULL;
        ELSEIF v_nav_base IS NULL OR v_nav_base = 0 THEN
            SET v_decision = 'FAIL';
            SET v_reason = CONCAT('BASE_NAV_MISSING_OR_ZERO@', CAST(v_asof AS CHAR));
            SET v_alpha = NULL;
        ELSE
            SET v_alpha = (v_nav_run / v_nav_base) - 1;
            IF v_alpha >= v_min_alpha THEN
                SET v_decision = 'PASS';
                SET v_reason = 'ALPHA_OK';
            ELSE
                SET v_decision = 'FAIL';
                SET v_reason = 'ALPHA_BELOW_THRESHOLD';
            END IF;
        END IF;
    END IF;

    INSERT INTO cn_baseline_decision_t (
        run_id, baseline_key, baseline_run_id, decision, reason_code,
        metrics_json, compare_asof, created_at, updated_at, created_by
    )
    VALUES (
        v_run_id,
        v_baseline_key,
        COALESCE(v_baseline_run_id, ''),
        v_decision,
        v_reason,
        JSON_OBJECT(
            'alpha', v_alpha,
            'min_alpha', v_min_alpha,
            'nav_run', v_nav_run,
            'nav_baseline', v_nav_base,
            'asof', CAST(v_asof AS CHAR)
        ),
        NOW(6),
        NOW(6),
        NOW(6),
        'SP_VALIDATE_AGAINST_BASELINE'
    )
    ON DUPLICATE KEY UPDATE
        baseline_run_id = VALUES(baseline_run_id),
        decision = VALUES(decision),
        reason_code = VALUES(reason_code),
        metrics_json = VALUES(metrics_json),
        compare_asof = NOW(6),
        updated_at = NOW(6),
        created_by = 'SP_VALIDATE_AGAINST_BASELINE';

    SELECT
        v_run_id AS run_id,
        v_baseline_key AS baseline_key,
        v_baseline_run_id AS baseline_run_id,
        v_decision AS decision,
        v_reason AS reason_code,
        v_alpha AS alpha,
        v_min_alpha AS min_alpha,
        v_nav_run AS nav_run,
        v_nav_base AS nav_baseline,
        v_asof AS asof_date;
END proc ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `SP_VALIDATE_SECTOR_ROT_RUN` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_0900_ai_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`cn_opr`@`localhost` PROCEDURE `SP_VALIDATE_SECTOR_ROT_RUN`(
    IN p_run_id VARCHAR(128)
)
proc: BEGIN
    DECLARE v_run_id VARCHAR(128);
    DECLARE v_rows BIGINT DEFAULT 0;
    DECLARE v_dt_min DATE;
    DECLARE v_dt_max DATE;
    DECLARE v_nav_min DECIMAL(18,10);
    DECLARE v_nav_max DECIMAL(18,10);
    DECLARE v_nav_last DECIMAL(18,10);
    DECLARE v_nav_null BIGINT DEFAULT 0;
    DECLARE v_neg_nav BIGINT DEFAULT 0;
    DECLARE v_sig_days BIGINT DEFAULT 0;
    DECLARE v_bt_days BIGINT DEFAULT 0;
    DECLARE v_status VARCHAR(16);

    SET v_run_id = COALESCE(NULLIF(p_run_id, ''), 'SR_LIVE_DEFAULT');

    SELECT
        COUNT(*),
        MIN(trade_date),
        MAX(trade_date),
        MIN(nav),
        MAX(nav),
        SUM(CASE WHEN nav IS NULL THEN 1 ELSE 0 END),
        SUM(CASE WHEN IFNULL(nav, 0) <= 0 THEN 1 ELSE 0 END)
    INTO
        v_rows, v_dt_min, v_dt_max, v_nav_min, v_nav_max, v_nav_null, v_neg_nav
    FROM cn_sector_rot_bt_daily_t
    WHERE run_id = v_run_id;

    SELECT b.nav
    INTO v_nav_last
    FROM cn_sector_rot_bt_daily_t b
    WHERE b.run_id = v_run_id
    ORDER BY b.trade_date DESC
    LIMIT 1;

    SELECT COUNT(DISTINCT signal_date)
    INTO v_sig_days
    FROM cn_sector_rotation_signal_t;

    SET v_bt_days = IFNULL(v_rows, 0);

    IF v_rows = 0 THEN
        SET v_status = 'FAIL';
    ELSEIF IFNULL(v_nav_null, 0) > 0 OR IFNULL(v_neg_nav, 0) > 0 THEN
        SET v_status = 'FAIL';
    ELSE
        SET v_status = 'PASS';
    END IF;

    SELECT
        v_run_id AS run_id,
        v_status AS validation_status,
        v_rows AS bt_rows,
        v_dt_min AS bt_start_date,
        v_dt_max AS bt_end_date,
        v_nav_last AS nav_last,
        v_nav_min AS nav_min,
        v_nav_max AS nav_max,
        v_nav_null AS nav_null_rows,
        v_neg_nav AS nav_nonpositive_rows,
        v_sig_days AS signal_days,
        v_bt_days AS bt_days,
        CASE
            WHEN v_status = 'PASS' THEN 'OK'
            WHEN v_rows = 0 THEN 'BT_EMPTY'
            WHEN IFNULL(v_nav_null, 0) > 0 THEN 'BT_NAV_NULL'
            ELSE 'BT_NAV_NONPOSITIVE'
        END AS reason_code;
END proc ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;

--
-- Final view structure for view `cn_board_concept_eod_agg_v`
--

/*!50001 DROP VIEW IF EXISTS `cn_board_concept_eod_agg_v`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `cn_board_concept_eod_agg_v` AS select 'CONCEPT' AS `sector_type`,`m`.`sector_id` AS `sector_id`,`p`.`TRADE_DATE` AS `trade_date`,count(0) AS `members`,sum(ifnull(`p`.`AMOUNT`,0)) AS `amount_sum`,avg((case when ((coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) is not null) and (coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) <> 0)) then ((`p`.`CLOSE` / coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`))) - 1) when (`p`.`CHG_PCT` is not null) then (case when (abs(`p`.`CHG_PCT`) > 1) then (`p`.`CHG_PCT` / 100) else `p`.`CHG_PCT` end) else NULL end)) AS `avg_ret`,avg((case when ((coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) is not null) and (coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) <> 0)) then ((`p`.`CLOSE` / coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`))) - 1) when (`p`.`CHG_PCT` is not null) then (case when (abs(`p`.`CHG_PCT`) > 1) then (`p`.`CHG_PCT` / 100) else `p`.`CHG_PCT` end) else NULL end)) AS `median_ret`,avg((case when ((coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) is not null) and (coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) <> 0)) then ((`p`.`CLOSE` / coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`))) - 1) when (`p`.`CHG_PCT` is not null) then (case when (abs(`p`.`CHG_PCT`) > 1) then (`p`.`CHG_PCT` / 100) else `p`.`CHG_PCT` end) else NULL end)) AS `eqw_ret`,avg((case when ((case when ((coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) is not null) and (coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) <> 0)) then ((`p`.`CLOSE` / coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`))) - 1) when (`p`.`CHG_PCT` is not null) then (case when (abs(`p`.`CHG_PCT`) > 1) then (`p`.`CHG_PCT` / 100) else `p`.`CHG_PCT` end) else NULL end) is null) then NULL when ((case when ((coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) is not null) and (coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) <> 0)) then ((`p`.`CLOSE` / coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`))) - 1) when (`p`.`CHG_PCT` is not null) then (case when (abs(`p`.`CHG_PCT`) > 1) then (`p`.`CHG_PCT` / 100) else `p`.`CHG_PCT` end) else NULL end) > 0) then 1 else 0 end)) AS `up_ratio` from (`cn_board_member_map_d` `m` join `cn_stock_daily_price_active_v` `p` on(((`p`.`TRADE_DATE` = `m`.`trade_date`) and (`p`.`SYMBOL` = (`m`.`symbol` collate utf8mb4_unicode_ci))))) where (`m`.`sector_type` = 'CONCEPT') group by `m`.`sector_id`,`p`.`TRADE_DATE` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `cn_board_industry_eod_agg_v`
--

/*!50001 DROP VIEW IF EXISTS `cn_board_industry_eod_agg_v`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `cn_board_industry_eod_agg_v` AS select 'INDUSTRY' AS `sector_type`,`m`.`sector_id` AS `sector_id`,`p`.`TRADE_DATE` AS `trade_date`,count(0) AS `members`,sum(ifnull(`p`.`AMOUNT`,0)) AS `amount_sum`,avg((case when ((coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) is not null) and (coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) <> 0)) then ((`p`.`CLOSE` / coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`))) - 1) when (`p`.`CHG_PCT` is not null) then (case when (abs(`p`.`CHG_PCT`) > 1) then (`p`.`CHG_PCT` / 100) else `p`.`CHG_PCT` end) else NULL end)) AS `avg_ret`,avg((case when ((coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) is not null) and (coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) <> 0)) then ((`p`.`CLOSE` / coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`))) - 1) when (`p`.`CHG_PCT` is not null) then (case when (abs(`p`.`CHG_PCT`) > 1) then (`p`.`CHG_PCT` / 100) else `p`.`CHG_PCT` end) else NULL end)) AS `median_ret`,avg((case when ((coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) is not null) and (coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) <> 0)) then ((`p`.`CLOSE` / coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`))) - 1) when (`p`.`CHG_PCT` is not null) then (case when (abs(`p`.`CHG_PCT`) > 1) then (`p`.`CHG_PCT` / 100) else `p`.`CHG_PCT` end) else NULL end)) AS `eqw_ret`,avg((case when ((case when ((coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) is not null) and (coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) <> 0)) then ((`p`.`CLOSE` / coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`))) - 1) when (`p`.`CHG_PCT` is not null) then (case when (abs(`p`.`CHG_PCT`) > 1) then (`p`.`CHG_PCT` / 100) else `p`.`CHG_PCT` end) else NULL end) is null) then NULL when ((case when ((coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) is not null) and (coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`)) <> 0)) then ((`p`.`CLOSE` / coalesce(`p`.`PRE_CLOSE`,(`p`.`CLOSE` - `p`.`CHANGE`))) - 1) when (`p`.`CHG_PCT` is not null) then (case when (abs(`p`.`CHG_PCT`) > 1) then (`p`.`CHG_PCT` / 100) else `p`.`CHG_PCT` end) else NULL end) > 0) then 1 else 0 end)) AS `up_ratio` from (`cn_board_member_map_d` `m` join `cn_stock_daily_price_active_v` `p` on(((`p`.`TRADE_DATE` = `m`.`trade_date`) and (`p`.`SYMBOL` = (`m`.`symbol` collate utf8mb4_unicode_ci))))) where (`m`.`sector_type` = 'INDUSTRY') group by `m`.`sector_id`,`p`.`TRADE_DATE` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `cn_sector_energy_v`
--

/*!50001 DROP VIEW IF EXISTS `cn_sector_energy_v`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `cn_sector_energy_v` AS with `x` as (select `cn_sector_rotation_state_v`.`trade_date` AS `trade_date`,`cn_sector_rotation_state_v`.`sector_type` AS `sector_type`,`cn_sector_rotation_state_v`.`sector_id` AS `sector_id`,`cn_sector_rotation_state_v`.`amount_sum` AS `amount_sum`,`cn_sector_rotation_state_v`.`amt_impulse` AS `amt_impulse`,`cn_sector_rotation_state_v`.`up_ratio` AS `up_ratio`,`cn_sector_rotation_state_v`.`mom_10` AS `mom_10`,`cn_sector_rotation_state_v`.`mom_20` AS `mom_20`,`cn_sector_rotation_state_v`.`up_slope` AS `up_slope`,`cn_sector_rotation_state_v`.`cov_10` AS `cov_10` from `cn_sector_rotation_state_v`), `r` as (select `x`.`trade_date` AS `trade_date`,`x`.`sector_type` AS `sector_type`,`x`.`sector_id` AS `sector_id`,`x`.`amount_sum` AS `amount_sum`,`x`.`amt_impulse` AS `amt_impulse`,`x`.`up_ratio` AS `up_ratio`,`x`.`mom_10` AS `mom_10`,`x`.`mom_20` AS `mom_20`,`x`.`up_slope` AS `up_slope`,`x`.`cov_10` AS `cov_10`,percent_rank() OVER (PARTITION BY `x`.`trade_date` ORDER BY `x`.`amt_impulse` )  AS `pr_amt_imp`,percent_rank() OVER (PARTITION BY `x`.`trade_date` ORDER BY `x`.`amount_sum` )  AS `pr_amt_sum`,percent_rank() OVER (PARTITION BY `x`.`trade_date` ORDER BY `x`.`up_ratio` )  AS `pr_up`,percent_rank() OVER (PARTITION BY `x`.`trade_date` ORDER BY `x`.`up_slope` )  AS `pr_up_slope`,percent_rank() OVER (PARTITION BY `x`.`trade_date` ORDER BY `x`.`mom_10` )  AS `pr_m10`,percent_rank() OVER (PARTITION BY `x`.`trade_date` ORDER BY `x`.`mom_20` )  AS `pr_m20`,percent_rank() OVER (PARTITION BY `x`.`trade_date` ORDER BY -(`x`.`cov_10`) )  AS `pr_cov_good` from `x`), `s` as (select `r`.`trade_date` AS `trade_date`,`r`.`sector_type` AS `sector_type`,`r`.`sector_id` AS `sector_id`,`r`.`amount_sum` AS `amount_sum`,`r`.`amt_impulse` AS `amt_impulse`,`r`.`up_ratio` AS `up_ratio`,`r`.`mom_10` AS `mom_10`,`r`.`mom_20` AS `mom_20`,`r`.`up_slope` AS `up_slope`,`r`.`cov_10` AS `cov_10`,`r`.`pr_amt_imp` AS `pr_amt_imp`,`r`.`pr_amt_sum` AS `pr_amt_sum`,`r`.`pr_up` AS `pr_up`,`r`.`pr_up_slope` AS `pr_up_slope`,`r`.`pr_m10` AS `pr_m10`,`r`.`pr_m20` AS `pr_m20`,`r`.`pr_cov_good` AS `pr_cov_good`,(((((((0.35 * `r`.`pr_amt_imp`) + (0.10 * `r`.`pr_amt_sum`)) + (0.20 * `r`.`pr_up`)) + (0.10 * `r`.`pr_up_slope`)) + (0.15 * `r`.`pr_m10`)) + (0.05 * `r`.`pr_m20`)) + (0.05 * `r`.`pr_cov_good`)) AS `energy_raw` from `r`) select `s`.`trade_date` AS `trade_date`,`s`.`sector_type` AS `sector_type`,`s`.`sector_id` AS `sector_id`,(100 * `s`.`energy_raw`) AS `energy_score`,percent_rank() OVER (PARTITION BY `s`.`trade_date` ORDER BY `s`.`energy_raw` )  AS `energy_pct` from `s` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `cn_sector_eod_agg_v`
--

/*!50001 DROP VIEW IF EXISTS `cn_sector_eod_agg_v`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `cn_sector_eod_agg_v` AS select `cn_board_industry_eod_agg_v`.`sector_type` AS `SECTOR_TYPE`,`cn_board_industry_eod_agg_v`.`sector_id` AS `SECTOR_ID`,`cn_board_industry_eod_agg_v`.`trade_date` AS `TRADE_DATE`,`cn_board_industry_eod_agg_v`.`members` AS `MEMBERS`,`cn_board_industry_eod_agg_v`.`amount_sum` AS `AMOUNT_SUM`,`cn_board_industry_eod_agg_v`.`avg_ret` AS `AVG_RET`,`cn_board_industry_eod_agg_v`.`median_ret` AS `MEDIAN_RET`,`cn_board_industry_eod_agg_v`.`eqw_ret` AS `EQW_RET`,`cn_board_industry_eod_agg_v`.`up_ratio` AS `UP_RATIO` from `cn_board_industry_eod_agg_v` union all select `cn_board_concept_eod_agg_v`.`sector_type` AS `SECTOR_TYPE`,`cn_board_concept_eod_agg_v`.`sector_id` AS `SECTOR_ID`,`cn_board_concept_eod_agg_v`.`trade_date` AS `TRADE_DATE`,`cn_board_concept_eod_agg_v`.`members` AS `MEMBERS`,`cn_board_concept_eod_agg_v`.`amount_sum` AS `AMOUNT_SUM`,`cn_board_concept_eod_agg_v`.`avg_ret` AS `AVG_RET`,`cn_board_concept_eod_agg_v`.`median_ret` AS `MEDIAN_RET`,`cn_board_concept_eod_agg_v`.`eqw_ret` AS `EQW_RET`,`cn_board_concept_eod_agg_v`.`up_ratio` AS `UP_RATIO` from `cn_board_concept_eod_agg_v` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `cn_sector_eod_feature_v`
--

/*!50001 DROP VIEW IF EXISTS `cn_sector_eod_feature_v`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `cn_sector_eod_feature_v` AS select `cn_sector_eod_agg_v`.`SECTOR_TYPE` AS `sector_type`,`cn_sector_eod_agg_v`.`SECTOR_ID` AS `sector_id`,`cn_sector_eod_agg_v`.`TRADE_DATE` AS `trade_date`,`cn_sector_eod_agg_v`.`MEMBERS` AS `members`,`cn_sector_eod_agg_v`.`AMOUNT_SUM` AS `amount_sum`,`cn_sector_eod_agg_v`.`AVG_RET` AS `avg_ret`,`cn_sector_eod_agg_v`.`MEDIAN_RET` AS `median_ret`,`cn_sector_eod_agg_v`.`EQW_RET` AS `eqw_ret`,`cn_sector_eod_agg_v`.`UP_RATIO` AS `up_ratio`,count(`cn_sector_eod_agg_v`.`EQW_RET`) OVER (PARTITION BY `cn_sector_eod_agg_v`.`SECTOR_TYPE`,`cn_sector_eod_agg_v`.`SECTOR_ID` ORDER BY `cn_sector_eod_agg_v`.`TRADE_DATE` ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)  AS `cov_5`,count(`cn_sector_eod_agg_v`.`EQW_RET`) OVER (PARTITION BY `cn_sector_eod_agg_v`.`SECTOR_TYPE`,`cn_sector_eod_agg_v`.`SECTOR_ID` ORDER BY `cn_sector_eod_agg_v`.`TRADE_DATE` ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)  AS `cov_10`,count(`cn_sector_eod_agg_v`.`EQW_RET`) OVER (PARTITION BY `cn_sector_eod_agg_v`.`SECTOR_TYPE`,`cn_sector_eod_agg_v`.`SECTOR_ID` ORDER BY `cn_sector_eod_agg_v`.`TRADE_DATE` ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)  AS `cov_20`,avg(`cn_sector_eod_agg_v`.`AMOUNT_SUM`) OVER (PARTITION BY `cn_sector_eod_agg_v`.`SECTOR_TYPE`,`cn_sector_eod_agg_v`.`SECTOR_ID` ORDER BY `cn_sector_eod_agg_v`.`TRADE_DATE` ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)  AS `amt_ma20`,(case when (avg(`cn_sector_eod_agg_v`.`AMOUNT_SUM`) OVER (PARTITION BY `cn_sector_eod_agg_v`.`SECTOR_TYPE`,`cn_sector_eod_agg_v`.`SECTOR_ID` ORDER BY `cn_sector_eod_agg_v`.`TRADE_DATE` ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)  = 0) then NULL else (`cn_sector_eod_agg_v`.`AMOUNT_SUM` / avg(`cn_sector_eod_agg_v`.`AMOUNT_SUM`) OVER (PARTITION BY `cn_sector_eod_agg_v`.`SECTOR_TYPE`,`cn_sector_eod_agg_v`.`SECTOR_ID` ORDER BY `cn_sector_eod_agg_v`.`TRADE_DATE` ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) ) end) AS `amt_impulse`,avg(`cn_sector_eod_agg_v`.`UP_RATIO`) OVER (PARTITION BY `cn_sector_eod_agg_v`.`SECTOR_TYPE`,`cn_sector_eod_agg_v`.`SECTOR_ID` ORDER BY `cn_sector_eod_agg_v`.`TRADE_DATE` ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)  AS `up_ma5`,(`cn_sector_eod_agg_v`.`UP_RATIO` - avg(`cn_sector_eod_agg_v`.`UP_RATIO`) OVER (PARTITION BY `cn_sector_eod_agg_v`.`SECTOR_TYPE`,`cn_sector_eod_agg_v`.`SECTOR_ID` ORDER BY `cn_sector_eod_agg_v`.`TRADE_DATE` ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) ) AS `up_slope`,(exp(sum(ln((1 + ifnull(`cn_sector_eod_agg_v`.`EQW_RET`,0)))) OVER (PARTITION BY `cn_sector_eod_agg_v`.`SECTOR_TYPE`,`cn_sector_eod_agg_v`.`SECTOR_ID` ORDER BY `cn_sector_eod_agg_v`.`TRADE_DATE` ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) ) - 1) AS `mom_5`,(exp(sum(ln((1 + ifnull(`cn_sector_eod_agg_v`.`EQW_RET`,0)))) OVER (PARTITION BY `cn_sector_eod_agg_v`.`SECTOR_TYPE`,`cn_sector_eod_agg_v`.`SECTOR_ID` ORDER BY `cn_sector_eod_agg_v`.`TRADE_DATE` ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) ) - 1) AS `mom_10`,(exp(sum(ln((1 + ifnull(`cn_sector_eod_agg_v`.`EQW_RET`,0)))) OVER (PARTITION BY `cn_sector_eod_agg_v`.`SECTOR_TYPE`,`cn_sector_eod_agg_v`.`SECTOR_ID` ORDER BY `cn_sector_eod_agg_v`.`TRADE_DATE` ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) ) - 1) AS `mom_20` from `cn_sector_eod_agg_v` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `cn_sector_rotation_named_v`
--

/*!50001 DROP VIEW IF EXISTS `cn_sector_rotation_named_v`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `cn_sector_rotation_named_v` AS select `s`.`trade_date` AS `trade_date`,date_format(cast(`s`.`trade_date` as date),'%Y-%m-%d') AS `trading_date`,`s`.`sector_type` AS `sector_type`,`s`.`sector_id` AS `sector_id`,(case when (`s`.`sector_type` = 'INDUSTRY') then `im`.`board_name` when (`s`.`sector_type` = 'CONCEPT') then `cm`.`concept_name` else NULL end) AS `sector_name`,`s`.`state` AS `state`,`s`.`score` AS `score`,`s`.`members` AS `members`,`s`.`cov_5` AS `cov_5`,`s`.`cov_10` AS `cov_10`,`s`.`cov_20` AS `cov_20`,`s`.`mom_5` AS `mom_5`,`s`.`mom_10` AS `mom_10`,`s`.`mom_20` AS `mom_20`,`s`.`amt_impulse` AS `amt_impulse`,`s`.`up_ratio` AS `up_ratio`,`s`.`up_ma5` AS `up_ma5`,`s`.`up_slope` AS `up_slope` from ((`cn_sector_rotation_state_v` `s` left join (select `t`.`board_id` AS `board_id`,`t`.`board_name` AS `board_name` from (select `m`.`BOARD_ID` AS `board_id`,`m`.`BOARD_NAME` AS `board_name`,row_number() OVER (PARTITION BY `m`.`BOARD_ID` ORDER BY `m`.`ASOF_DATE` desc )  AS `rn` from `cn_board_industry_master` `m`) `t` where (`t`.`rn` = 1)) `im` on(((`s`.`sector_type` = 'INDUSTRY') and (`s`.`sector_id` = `im`.`board_id`)))) left join (select `t`.`concept_id` AS `concept_id`,`t`.`concept_name` AS `concept_name` from (select `m`.`CONCEPT_ID` AS `concept_id`,`m`.`CONCEPT_NAME` AS `concept_name`,row_number() OVER (PARTITION BY `m`.`CONCEPT_ID` ORDER BY `m`.`ASOF_DATE` desc )  AS `rn` from `cn_board_concept_master` `m`) `t` where (`t`.`rn` = 1)) `cm` on(((`s`.`sector_type` = 'CONCEPT') and (`s`.`sector_id` = `cm`.`concept_id`)))) */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `cn_sector_rotation_ranked_v_too_slow`
--

/*!50001 DROP VIEW IF EXISTS `cn_sector_rotation_ranked_v_too_slow`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `cn_sector_rotation_ranked_v_too_slow` AS select `r`.`trade_date` AS `trade_date`,`r`.`trading_date` AS `trading_date`,`r`.`sector_type` AS `sector_type`,`r`.`sector_id` AS `sector_id`,`r`.`sector_name` AS `sector_name`,`r`.`theme_group` AS `theme_group`,`r`.`state` AS `state`,`r`.`confirm_streak` AS `confirm_streak`,`r`.`tier` AS `tier`,`r`.`tier_pri` AS `tier_pri`,`r`.`score` AS `score`,`r`.`members` AS `members`,`r`.`cov_5` AS `cov_5`,`r`.`cov_10` AS `cov_10`,`r`.`cov_20` AS `cov_20`,`r`.`mom_5` AS `mom_5`,`r`.`mom_10` AS `mom_10`,`r`.`mom_20` AS `mom_20`,`r`.`amt_impulse` AS `amt_impulse`,`r`.`up_ratio` AS `up_ratio`,`r`.`up_ma5` AS `up_ma5`,`r`.`up_slope` AS `up_slope`,`r`.`theme_rank` AS `theme_rank`,(case when (`r`.`theme_rank` = 1) then 'KEEP' else 'DUP_THEME' end) AS `theme_flag` from (select `t`.`trade_date` AS `trade_date`,`t`.`trading_date` AS `trading_date`,`t`.`sector_type` AS `sector_type`,`t`.`sector_id` AS `sector_id`,`t`.`sector_name` AS `sector_name`,`t`.`state` AS `state`,`t`.`score` AS `score`,`t`.`members` AS `members`,`t`.`cov_5` AS `cov_5`,`t`.`cov_10` AS `cov_10`,`t`.`cov_20` AS `cov_20`,`t`.`mom_5` AS `mom_5`,`t`.`mom_10` AS `mom_10`,`t`.`mom_20` AS `mom_20`,`t`.`amt_impulse` AS `amt_impulse`,`t`.`up_ratio` AS `up_ratio`,`t`.`up_ma5` AS `up_ma5`,`t`.`up_slope` AS `up_slope`,`t`.`rn_all` AS `rn_all`,`t`.`rn_conf` AS `rn_conf`,`t`.`grp_conf` AS `grp_conf`,`t`.`confirm_streak` AS `confirm_streak`,`t`.`theme_group` AS `theme_group`,`t`.`tier` AS `tier`,`t`.`tier_pri` AS `tier_pri`,row_number() OVER (PARTITION BY `t`.`trade_date`,`t`.`theme_group` ORDER BY `t`.`tier_pri`,`t`.`score` desc )  AS `theme_rank` from (select `a`.`trade_date` AS `trade_date`,`a`.`trading_date` AS `trading_date`,`a`.`sector_type` AS `sector_type`,`a`.`sector_id` AS `sector_id`,`a`.`sector_name` AS `sector_name`,`a`.`state` AS `state`,`a`.`score` AS `score`,`a`.`members` AS `members`,`a`.`cov_5` AS `cov_5`,`a`.`cov_10` AS `cov_10`,`a`.`cov_20` AS `cov_20`,`a`.`mom_5` AS `mom_5`,`a`.`mom_10` AS `mom_10`,`a`.`mom_20` AS `mom_20`,`a`.`amt_impulse` AS `amt_impulse`,`a`.`up_ratio` AS `up_ratio`,`a`.`up_ma5` AS `up_ma5`,`a`.`up_slope` AS `up_slope`,`a`.`rn_all` AS `rn_all`,`a`.`rn_conf` AS `rn_conf`,`a`.`grp_conf` AS `grp_conf`,`a`.`confirm_streak` AS `confirm_streak`,(case when ((`a`.`sector_name` like '%流感%') or (`a`.`sector_name` like '%肝炎%') or (`a`.`sector_name` like '%病毒%') or (`a`.`sector_name` like '%防治%') or (`a`.`sector_name` like '%疫苗%')) then 'MEDICAL_EVENT' when ((`a`.`sector_name` like '%中药%') or (`a`.`sector_name` like '%中医%')) then 'TCM' when ((`a`.`sector_name` like '%汽车%') or (`a`.`sector_name` like '%零部件%') or (`a`.`sector_name` like '%智能驾驶%')) then 'AUTO_CHAIN' when ((`a`.`sector_name` like '%有机硅%') or (`a`.`sector_name` like '%碳纤维%') or (`a`.`sector_name` like '%小金属%') or (`a`.`sector_name` like '%稀土%') or (`a`.`sector_name` like '%新材料%')) then 'MATERIALS' when (`a`.`sector_name` like '%核聚变%') then 'NUCLEAR_FUSION' when (`a`.`sector_name` like '%合成生物%') then 'SYNBIO' else 'OTHER' end) AS `theme_group`,(case when ((`a`.`state` = 'CONFIRM') and ((ifnull(`a`.`confirm_streak`,0) >= 2) or (ifnull(`a`.`mom_20`,0) > 0))) then 'T1' when (`a`.`state` = 'CONFIRM') then 'T2' when (`a`.`state` = 'HOLD') then 'T2' when (`a`.`state` = 'IGNITE') then 'T3' when (`a`.`state` = 'FADE') then 'T4' else 'T9' end) AS `tier`,(case when ((`a`.`state` = 'CONFIRM') and ((ifnull(`a`.`confirm_streak`,0) >= 2) or (ifnull(`a`.`mom_20`,0) > 0))) then 0 when (`a`.`state` = 'CONFIRM') then 1 when (`a`.`state` = 'HOLD') then 2 when (`a`.`state` = 'IGNITE') then 3 when (`a`.`state` = 'FADE') then 4 else 9 end) AS `tier_pri` from (select `n`.`trade_date` AS `trade_date`,`n`.`trading_date` AS `trading_date`,`n`.`sector_type` AS `sector_type`,`n`.`sector_id` AS `sector_id`,`n`.`sector_name` AS `sector_name`,`n`.`state` AS `state`,`n`.`score` AS `score`,`n`.`members` AS `members`,`n`.`cov_5` AS `cov_5`,`n`.`cov_10` AS `cov_10`,`n`.`cov_20` AS `cov_20`,`n`.`mom_5` AS `mom_5`,`n`.`mom_10` AS `mom_10`,`n`.`mom_20` AS `mom_20`,`n`.`amt_impulse` AS `amt_impulse`,`n`.`up_ratio` AS `up_ratio`,`n`.`up_ma5` AS `up_ma5`,`n`.`up_slope` AS `up_slope`,`n`.`rn_all` AS `rn_all`,`n`.`rn_conf` AS `rn_conf`,`n`.`grp_conf` AS `grp_conf`,(case when (`n`.`state` = 'CONFIRM') then row_number() OVER (PARTITION BY `n`.`sector_type`,`n`.`sector_id`,`n`.`grp_conf` ORDER BY `n`.`trade_date` )  else 0 end) AS `confirm_streak` from (select `m`.`trade_date` AS `trade_date`,`m`.`trading_date` AS `trading_date`,`m`.`sector_type` AS `sector_type`,`m`.`sector_id` AS `sector_id`,`m`.`sector_name` AS `sector_name`,`m`.`state` AS `state`,`m`.`score` AS `score`,`m`.`members` AS `members`,`m`.`cov_5` AS `cov_5`,`m`.`cov_10` AS `cov_10`,`m`.`cov_20` AS `cov_20`,`m`.`mom_5` AS `mom_5`,`m`.`mom_10` AS `mom_10`,`m`.`mom_20` AS `mom_20`,`m`.`amt_impulse` AS `amt_impulse`,`m`.`up_ratio` AS `up_ratio`,`m`.`up_ma5` AS `up_ma5`,`m`.`up_slope` AS `up_slope`,`m`.`rn_all` AS `rn_all`,`m`.`rn_conf` AS `rn_conf`,(case when (`m`.`state` = 'CONFIRM') then (`m`.`rn_all` - `m`.`rn_conf`) else NULL end) AS `grp_conf` from (select `v`.`trade_date` AS `trade_date`,`v`.`trading_date` AS `trading_date`,`v`.`sector_type` AS `sector_type`,`v`.`sector_id` AS `sector_id`,`v`.`sector_name` AS `sector_name`,`v`.`state` AS `state`,`v`.`score` AS `score`,`v`.`members` AS `members`,`v`.`cov_5` AS `cov_5`,`v`.`cov_10` AS `cov_10`,`v`.`cov_20` AS `cov_20`,`v`.`mom_5` AS `mom_5`,`v`.`mom_10` AS `mom_10`,`v`.`mom_20` AS `mom_20`,`v`.`amt_impulse` AS `amt_impulse`,`v`.`up_ratio` AS `up_ratio`,`v`.`up_ma5` AS `up_ma5`,`v`.`up_slope` AS `up_slope`,row_number() OVER (PARTITION BY `v`.`sector_type`,`v`.`sector_id` ORDER BY `v`.`trade_date` )  AS `rn_all`,(case when (`v`.`state` = 'CONFIRM') then row_number() OVER (PARTITION BY `v`.`sector_type`,`v`.`sector_id`,`v`.`state` ORDER BY `v`.`trade_date` )  else NULL end) AS `rn_conf` from `cn_sector_rotation_named_v` `v`) `m`) `n`) `a`) `t`) `r` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `cn_sector_rotation_state_v`
--

/*!50001 DROP VIEW IF EXISTS `cn_sector_rotation_state_v`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `cn_sector_rotation_state_v` AS select `t`.`sector_type` AS `sector_type`,`t`.`sector_id` AS `sector_id`,`t`.`trade_date` AS `trade_date`,`t`.`members` AS `members`,`t`.`amount_sum` AS `amount_sum`,`t`.`eqw_ret` AS `eqw_ret`,`t`.`up_ratio` AS `up_ratio`,`t`.`mom_5` AS `mom_5`,`t`.`mom_10` AS `mom_10`,`t`.`mom_20` AS `mom_20`,`t`.`amt_impulse` AS `amt_impulse`,`t`.`up_ma5` AS `up_ma5`,`t`.`up_slope` AS `up_slope`,`t`.`cov_5` AS `cov_5`,`t`.`cov_10` AS `cov_10`,`t`.`cov_20` AS `cov_20`,(case when (`t`.`base_state` = 'FADE') then 'FADE' when (`t`.`base_state` = 'CONFIRM') then 'CONFIRM' when ((`t`.`recent_confirm` = 1) and (`t`.`members` >= 30) and (`t`.`cov_10` >= 10) and (`t`.`mom_10` > 0) and (`t`.`up_ma5` >= 0.52) and (`t`.`up_ratio` >= 0.50) and (`t`.`amt_impulse` >= 0.95)) then 'HOLD' when (`t`.`base_state` = 'IGNITE') then 'IGNITE' else 'NEUTRAL' end) AS `state`,(((ifnull(`t`.`mom_10`,0) * 100) + ((ifnull(`t`.`amt_impulse`,1) - 1) * 50)) + ((ifnull(`t`.`up_ratio`,0) - 0.5) * 100)) AS `score` from (select `f`.`sector_type` AS `sector_type`,`f`.`sector_id` AS `sector_id`,`f`.`trade_date` AS `trade_date`,`f`.`members` AS `members`,`f`.`amount_sum` AS `amount_sum`,`f`.`avg_ret` AS `avg_ret`,`f`.`median_ret` AS `median_ret`,`f`.`eqw_ret` AS `eqw_ret`,`f`.`up_ratio` AS `up_ratio`,`f`.`cov_5` AS `cov_5`,`f`.`cov_10` AS `cov_10`,`f`.`cov_20` AS `cov_20`,`f`.`amt_ma20` AS `amt_ma20`,`f`.`amt_impulse` AS `amt_impulse`,`f`.`up_ma5` AS `up_ma5`,`f`.`up_slope` AS `up_slope`,`f`.`mom_5` AS `mom_5`,`f`.`mom_10` AS `mom_10`,`f`.`mom_20` AS `mom_20`,(case when ((`f`.`members` >= 30) and (`f`.`cov_10` >= 10) and (`f`.`amt_impulse` >= 1.20) and (`f`.`up_ratio` <= 0.45) and (`f`.`mom_5` < 0)) then 'FADE' when ((`f`.`members` >= 30) and (`f`.`cov_20` >= 20) and (`f`.`mom_5` > 0) and (`f`.`mom_10` > 0) and (`f`.`amt_impulse` >= 1.30) and (`f`.`up_ratio` >= 0.55) and (`f`.`up_ma5` >= 0.52)) then 'CONFIRM' when ((`f`.`members` >= 30) and (`f`.`cov_10` >= 10) and (`f`.`mom_5` > 0) and (`f`.`amt_impulse` >= 1.30) and (`f`.`up_ratio` >= 0.55)) then 'IGNITE' else 'NEUTRAL' end) AS `base_state`,max((case when ((`f`.`members` >= 30) and (`f`.`cov_20` >= 20) and (`f`.`mom_5` > 0) and (`f`.`mom_10` > 0) and (`f`.`amt_impulse` >= 1.30) and (`f`.`up_ratio` >= 0.55) and (`f`.`up_ma5` >= 0.52)) then 1 else 0 end)) OVER (PARTITION BY `f`.`sector_type`,`f`.`sector_id` ORDER BY `f`.`trade_date` ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)  AS `recent_confirm` from `cn_sector_eod_feature_v` `f`) `t` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `cn_stock_active_universe_v`
--

/*!50001 DROP VIEW IF EXISTS `cn_stock_active_universe_v`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `cn_stock_active_universe_v` AS select `s`.`symbol` AS `symbol`,`s`.`is_active` AS `is_active`,`s`.`inactive_reason` AS `inactive_reason`,`s`.`first_trade_date` AS `first_trade_date`,`s`.`last_trade_date` AS `last_trade_date`,`s`.`recent_trade_days` AS `recent_trade_days`,`s`.`updated_at` AS `updated_at` from `cn_stock_universe_status_t` `s` where (ifnull(`s`.`is_active`,1) = 1) */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `cn_stock_daily_price_active_v`
--

/*!50001 DROP VIEW IF EXISTS `cn_stock_daily_price_active_v`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `cn_stock_daily_price_active_v` AS select `p`.`SYMBOL` AS `SYMBOL`,`p`.`TRADE_DATE` AS `TRADE_DATE`,`p`.`OPEN` AS `OPEN`,`p`.`CLOSE` AS `CLOSE`,`p`.`PRE_CLOSE` AS `PRE_CLOSE`,`p`.`HIGH` AS `HIGH`,`p`.`LOW` AS `LOW`,`p`.`VOLUME` AS `VOLUME`,`p`.`AMOUNT` AS `AMOUNT`,`p`.`AMPLITUDE` AS `AMPLITUDE`,`p`.`CHG_PCT` AS `CHG_PCT`,`p`.`CHANGE` AS `CHANGE`,`p`.`TURNOVER_RATE` AS `TURNOVER_RATE`,`p`.`SOURCE` AS `SOURCE`,`p`.`WINDOW_START` AS `WINDOW_START`,`p`.`CREATED_AT` AS `CREATED_AT`,`p`.`EXCHANGE` AS `EXCHANGE`,`p`.`NAME` AS `NAME` from (`cn_stock_daily_price` `p` join `cn_stock_universe_status_t` `s` on((`s`.`symbol` = `p`.`SYMBOL`))) where (ifnull(`s`.`is_active`,1) = 1) */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `v_rotation_entry_exec_v1`
--

/*!50001 DROP VIEW IF EXISTS `v_rotation_entry_exec_v1`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `v_rotation_entry_exec_v1` AS with `cal` as (select distinct `b`.`RUN_ID` AS `p_run_id`,`b`.`TRADE_DATE` AS `trade_date` from `cn_sector_rot_bt_daily_t` `b`), `next_td` as (select `c1`.`p_run_id` AS `p_run_id`,`c1`.`trade_date` AS `signal_date`,min(`c2`.`trade_date`) AS `exec_date` from (`cal` `c1` join `cal` `c2` on(((`c2`.`p_run_id` = `c1`.`p_run_id`) and (`c2`.`trade_date` > `c1`.`trade_date`)))) group by `c1`.`p_run_id`,`c1`.`trade_date`), `enter_sig` as (select `s`.`SIGNAL_DATE` AS `signal_date`,`s`.`SECTOR_TYPE` AS `sector_type`,`s`.`SECTOR_ID` AS `sector_id`,`s`.`SECTOR_NAME` AS `sector_name`,`s`.`ENTRY_RANK` AS `entry_rank`,`s`.`ENTRY_CNT` AS `entry_cnt`,`s`.`WEIGHT_SUGGESTED` AS `weight_suggested`,`s`.`SCORE` AS `signal_score`,`s`.`STATE` AS `state`,`s`.`TRANSITION` AS `transition` from `cn_sector_rotation_signal_t` `s` where (upper(`s`.`ACTION`) = 'ENTER')), `energy` as (select `e`.`TRADE_DATE` AS `signal_date`,`e`.`SECTOR_TYPE` AS `sector_type`,`e`.`SECTOR_ID` AS `sector_id`,`e`.`ENERGY_SCORE` AS `energy_score`,`e`.`ENERGY_PCT` AS `energy_pct` from `cn_sector_energy_snap_t` `e`) select `nt`.`p_run_id` AS `p_run_id`,`es`.`signal_date` AS `signal_date`,`nt`.`exec_date` AS `exec_enter_date`,`es`.`sector_type` AS `sector_type`,`es`.`sector_id` AS `sector_id`,`es`.`sector_name` AS `sector_name`,`es`.`entry_rank` AS `entry_rank`,`es`.`entry_cnt` AS `entry_cnt`,`es`.`weight_suggested` AS `weight_suggested`,`es`.`signal_score` AS `signal_score`,`en`.`energy_score` AS `energy_score`,`en`.`energy_pct` AS `energy_pct`,(case when (`en`.`energy_pct` is null) then 'NA' when (`en`.`energy_pct` >= 0.80) then 'HIGH' when (`en`.`energy_pct` >= 0.60) then 'MID' else 'LOW' end) AS `energy_tier`,`es`.`state` AS `state`,`es`.`transition` AS `transition` from ((`enter_sig` `es` join `next_td` `nt` on((`nt`.`signal_date` = `es`.`signal_date`))) left join `energy` `en` on(((`en`.`signal_date` = `es`.`signal_date`) and (`en`.`sector_type` = `es`.`sector_type`) and (`en`.`sector_id` = `es`.`sector_id`)))) */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `v_rotation_exit_exec_v1`
--

/*!50001 DROP VIEW IF EXISTS `v_rotation_exit_exec_v1`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`cn_opr`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `v_rotation_exit_exec_v1` AS with `params` as (select `p`.`RUN_ID` AS `p_run_id`,`p`.`TRADE_DATE` AS `signal_date` from `cn_sector_rot_pos_daily_t` `p` group by `p`.`RUN_ID`,`p`.`TRADE_DATE`), `cal` as (select distinct `b`.`RUN_ID` AS `p_run_id`,`b`.`TRADE_DATE` AS `trade_date` from `cn_sector_rot_bt_daily_t` `b`), `next_td` as (select `c1`.`p_run_id` AS `p_run_id`,`c1`.`trade_date` AS `signal_date`,min(`c2`.`trade_date`) AS `exec_date` from (`cal` `c1` join `cal` `c2` on(((`c2`.`p_run_id` = `c1`.`p_run_id`) and (`c2`.`trade_date` > `c1`.`trade_date`)))) group by `c1`.`p_run_id`,`c1`.`trade_date`), `holding` as (select distinct `p`.`RUN_ID` AS `p_run_id`,`p`.`TRADE_DATE` AS `signal_date`,`p`.`SECTOR_TYPE` AS `sector_type`,`p`.`SECTOR_ID` AS `sector_id` from `cn_sector_rot_pos_daily_t` `p` where (`p`.`EXIT_FLAG` = 0)), `exit_events` as (select `pr`.`p_run_id` AS `p_run_id`,`pr`.`signal_date` AS `signal_date`,`nt`.`exec_date` AS `exec_exit_date`,`s`.`SECTOR_TYPE` AS `sector_type`,`s`.`SECTOR_ID` AS `sector_id`,`s`.`SECTOR_NAME` AS `sector_name`,`s`.`STATE` AS `state`,`s`.`TRANSITION` AS `transition`,`s`.`ENTRY_RANK` AS `entry_rank`,`s`.`SCORE` AS `signal_score` from (((`params` `pr` join `holding` `h` on(((`h`.`p_run_id` = `pr`.`p_run_id`) and (`h`.`signal_date` = `pr`.`signal_date`)))) join `cn_sector_rotation_signal_t` `s` on(((`s`.`SIGNAL_DATE` = `pr`.`signal_date`) and (`s`.`SECTOR_TYPE` = `h`.`sector_type`) and (`s`.`SECTOR_ID` = `h`.`sector_id`) and (upper(`s`.`ACTION`) = 'EXIT')))) join `next_td` `nt` on(((`nt`.`p_run_id` = `pr`.`p_run_id`) and (`nt`.`signal_date` = `pr`.`signal_date`))))), `last_enter` as (select `x`.`p_run_id` AS `p_run_id`,`x`.`signal_date` AS `exit_signal_date`,`x`.`sector_type` AS `sector_type`,`x`.`sector_id` AS `sector_id`,max(`s2`.`SIGNAL_DATE`) AS `enter_signal_date` from (`exit_events` `x` left join `cn_sector_rotation_signal_t` `s2` on(((`s2`.`SECTOR_TYPE` = `x`.`sector_type`) and (`s2`.`SECTOR_ID` = `x`.`sector_id`) and (upper(`s2`.`ACTION`) = 'ENTER') and (`s2`.`SIGNAL_DATE` < `x`.`signal_date`)))) group by `x`.`p_run_id`,`x`.`signal_date`,`x`.`sector_type`,`x`.`sector_id`), `enter_exec` as (select `le`.`p_run_id` AS `p_run_id`,`le`.`exit_signal_date` AS `exit_signal_date`,`le`.`sector_type` AS `sector_type`,`le`.`sector_id` AS `sector_id`,`le`.`enter_signal_date` AS `enter_signal_date`,`nt`.`exec_date` AS `exec_enter_date` from (`last_enter` `le` left join `next_td` `nt` on(((`nt`.`p_run_id` = `le`.`p_run_id`) and (`nt`.`signal_date` = `le`.`enter_signal_date`))))), `min_exit_exec` as (select `e`.`p_run_id` AS `p_run_id`,`e`.`exit_signal_date` AS `exit_signal_date`,`e`.`sector_type` AS `sector_type`,`e`.`sector_id` AS `sector_id`,(select `t`.`trade_date` from (select `c`.`trade_date` AS `trade_date`,row_number() OVER (ORDER BY `c`.`trade_date` )  AS `rn` from `cal` `c` where ((`c`.`p_run_id` = `e`.`p_run_id`) and (`c`.`trade_date` >= `e`.`exec_enter_date`))) `t` where (`t`.`rn` = 3)) AS `min_exit_exec_date` from `enter_exec` `e`) select `x`.`p_run_id` AS `p_run_id`,`x`.`signal_date` AS `signal_date`,`x`.`exec_exit_date` AS `exec_exit_date`,`x`.`sector_type` AS `sector_type`,`x`.`sector_id` AS `sector_id`,`x`.`sector_name` AS `sector_name`,`x`.`state` AS `state`,`x`.`transition` AS `transition`,`x`.`entry_rank` AS `entry_rank`,`x`.`signal_score` AS `signal_score`,`e`.`enter_signal_date` AS `enter_signal_date`,`e`.`exec_enter_date` AS `exec_enter_date`,(case when (`e`.`exec_enter_date` is null) then 'EXIT_NO_PRIOR_ENTER' when (`x`.`exec_exit_date` >= `m`.`min_exit_exec_date`) then 'EXIT_ALLOWED' else 'EXIT_PENDING' end) AS `exit_exec_status` from ((`exit_events` `x` left join `enter_exec` `e` on(((`e`.`p_run_id` = `x`.`p_run_id`) and (`e`.`exit_signal_date` = `x`.`signal_date`) and (`e`.`sector_type` = `x`.`sector_type`) and (`e`.`sector_id` = `x`.`sector_id`)))) left join `min_exit_exec` `m` on(((`m`.`p_run_id` = `x`.`p_run_id`) and (`m`.`exit_signal_date` = `x`.`signal_date`) and (`m`.`sector_type` = `x`.`sector_type`) and (`m`.`sector_id` = `x`.`sector_id`)))) */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-03-05 14:39:31
