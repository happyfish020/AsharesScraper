from __future__ import annotations


def create_table_if_not_exists(*args, **kwargs):
    """Deprecated shim: project is MySQL-only; Oracle utilities are disabled."""
    return None


def create_tables_if_not_exists(*args, **kwargs):
    """Deprecated shim: project is MySQL-only; Oracle utilities are disabled."""
    return None


def get_engine(*args, **kwargs):
    raise RuntimeError('oracle_utils.get_engine is disabled (MySQL-only mode).')
