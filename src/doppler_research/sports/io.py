"""Read canonical files from a directory or ZIP and record their provenance."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import zipfile

import pandas as pd

from .schema import SCHEMA_VERSION, SchemaError

TABLES = ("markets", "quotes", "news_events", "fixtures_results")
SUFFIXES = (".csv", ".parquet")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(name: str, data: bytes) -> pd.DataFrame:
    if name.endswith(".parquet"):
        try:
            return pd.read_parquet(io.BytesIO(data))
        except ImportError as exc:
            raise SchemaError("Parquet input needs pyarrow: pip install '.[parquet]'") from exc
    return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8")


def _file_record(table: str, name: str, data: bytes, declaration: dict) -> dict:
    return {"table": table, "original_filename": name, "sha256": sha256_bytes(data), "bytes": len(data),
            "schema_version": SCHEMA_VERSION,
            "import_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **{k: declaration.get(k, "UNDECLARED") for k in
               ("provider", "licence_owner", "permitted_use", "capture_method")}}


def load_blobs(blobs: dict[str, tuple[str, bytes]], declaration: dict | None = None
               ) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """``blobs`` maps a table name to (original filename, bytes)."""
    declaration = declaration or {}
    missing = [t for t in TABLES if t not in blobs]
    if missing:
        raise SchemaError(f"missing canonical input tables {missing}")
    frames, manifest = {}, []
    for table in TABLES:
        name, data = blobs[table]
        frames[table] = _read(name, data)
        record = _file_record(table, name, data, declaration)
        record["rows"] = len(frames[table])
        manifest.append(record)
    return frames, manifest


def _table_for(name: str) -> str | None:
    stem = Path(name).name
    for table in TABLES:
        for suffix in SUFFIXES:
            if stem == f"{table}{suffix}":
                return table
    return None


def load_input(path: Path, declaration: dict | None = None) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Accept a directory or ZIP containing markets, quotes, news_events and fixtures_results."""
    path = Path(path)
    blobs: dict[str, tuple[str, bytes]] = {}
    if path.is_dir():
        for child in sorted(path.iterdir()):
            table = _table_for(child.name)
            if table:
                if table in blobs:
                    raise SchemaError(f"two files supplied for {table}")
                blobs[table] = (child.name, child.read_bytes())
    elif path.suffix == ".zip":
        blobs = blobs_from_zip(path.read_bytes())
    else:
        raise SchemaError("input must be a directory or a .zip")
    return load_blobs(blobs, declaration)


def blobs_from_zip(data: bytes) -> dict[str, tuple[str, bytes]]:
    blobs = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in sorted(z.namelist()):
            table = _table_for(member)
            if table:
                if table in blobs:
                    raise SchemaError(f"two files supplied for {table}")
                blobs[table] = (Path(member).name, z.read(member))
    return blobs


def code_fingerprint() -> dict:
    """SHA-256 of every source file in the package, so an output names exact code."""
    root = Path(__file__).resolve().parents[1]
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    h = hashlib.sha256()
    per_file = {}
    for p in files:
        data = p.read_bytes()
        per_file[str(p.relative_to(root.parent))] = sha256_bytes(data)
        h.update(str(p.relative_to(root)).encode() + b"\0" + data)
    from .. import __version__
    return {"package_version": __version__, "source_sha256": h.hexdigest(), "files": per_file}
