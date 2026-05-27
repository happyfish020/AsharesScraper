# V8 Crosswalk Scope Guide

Date: 2026-05-25

## Purpose

This guide explains the two scopes supported by
`data_pipeline/builders/sw_v7_v8_crosswalk.py`.

## Background

The V8 production path now treats:

- `cn_local_industry_map_hist.industry_level = 'L3'` as the `LOCAL_FINE`
  fine-grained V8 industry universe
- `SW_L1` as the official Shenwan level-1 comparison layer

The historical crosswalk builder was originally built as a compatibility layer,
not as the definition of the full V8 local-industry universe.

## Supported scopes

### 1. `compat`

Use:

```bash
python -m data_pipeline.builders.sw_v7_v8_crosswalk --v8-local-scope compat
```

Meaning:

- compares only the legacy compatibility subset (`85%.SI`) from the V8
  `LOCAL_FINE` layer
- keeps backward-compatible behavior
- can write to `cn_v7_v8_industry_crosswalk`

This is the default mode.

### 2. `full`

Use:

```bash
python -m data_pipeline.builders.sw_v7_v8_crosswalk --v8-local-scope full
```

Meaning:

- compares the full V8 `LOCAL_FINE` universe from
  `cn_local_industry_map_hist.industry_level = 'L3'`
- intended for analysis and migration planning first
- writes CSV/report outputs by default
- does **not** write to `cn_v7_v8_industry_crosswalk` unless explicitly allowed

To allow DB writes:

```bash
python -m data_pipeline.builders.sw_v7_v8_crosswalk \
  --v8-local-scope full \
  --allow-full-db-write \
  --replace
```

## Recommended usage

### For current compatibility maintenance

Use `compat`.

### For V8 semantic expansion or migration research

Use `full` first, inspect the report outputs, then decide whether the full
universe should be persisted into the historical crosswalk table or into a new
table/view.

## Practical interpretation

- `compat` = legacy bridge
- `full` = V8 universe study / migration mode
