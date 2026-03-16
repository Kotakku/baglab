"""Rosbag loading with lazy evaluation, automatic field expansion, and disk cache."""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml
from rosbags.dataframe import get_dataframe
from rosbags.highlevel import AnyReader
from rosbags.interfaces import Nodetype
from rosbags.typesys import Stores, get_typestore

from baglab.io.typesys import register_msg_types

_log = logging.getLogger(__name__)
_CACHE_DIR = ".baglab_cache"
_CACHE_VERSION = 1


def _topic_to_filename(topic: str) -> str:
    """Convert topic name to a safe filename stem."""
    return topic.lstrip("/").replace("/", "__")


def _bag_data_files(bag_path: Path) -> list[Path]:
    """Return the data files in the bag directory."""
    return sorted(
        p for p in bag_path.iterdir()
        if p.is_file() and p.name not in ("metadata.yaml", "info.yaml")
    )


def _compute_fingerprint(bag_path: Path) -> dict[str, dict]:
    """Compute fingerprint of bag data files for cache invalidation."""
    result = {}
    for f in _bag_data_files(bag_path):
        stat = f.stat()
        result[f.name] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    return result


def _fix_metadata(path: Path) -> None:
    """Fix empty storage_identifier in metadata.yaml by detecting from file extensions."""
    meta_path = path / "metadata.yaml"
    if not meta_path.exists():
        return

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    info = meta.get("rosbag2_bagfile_information", {})
    if info.get("storage_identifier", ""):
        return

    # Auto-detect from file extensions
    rel_paths = info.get("relative_file_paths", [])
    extensions = {Path(p).suffix for p in rel_paths}
    if extensions == {".db3"}:
        info["storage_identifier"] = "sqlite3"
    elif extensions == {".mcap"}:
        info["storage_identifier"] = "mcap"
    else:
        return

    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)


def _parse_topics(path: Path) -> dict[str, str]:
    """Parse ``{topic_name: message_type}`` from metadata.yaml (no reader needed)."""
    meta_path = path / "metadata.yaml"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    topics_info = (
        meta.get("rosbag2_bagfile_information", {})
        .get("topics_with_message_count", [])
    )
    return {
        entry["topic_metadata"]["name"]: entry["topic_metadata"]["type"]
        for entry in topics_info
    }


def _needs_typestore(path: Path) -> bool:
    """Check if the bag uses sqlite3 storage (which has no embedded type definitions)."""
    meta_path = path / "metadata.yaml"
    if not meta_path.exists():
        return False
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    return meta.get("rosbag2_bagfile_information", {}).get("storage_identifier") == "sqlite3"


class Bag:
    """Rosbag handle with lazy per-topic DataFrame loading.

    Parameters
    ----------
    path : Path
        Path to the rosbag directory.

    """

    def __init__(self, path: Path, msg_paths: Sequence[Path] = ()) -> None:
        if not path.is_dir():
            raise FileNotFoundError(f"Bag directory not found: {path}")
        self._path = path
        self._msg_paths = list(msg_paths)
        _fix_metadata(path)
        self._topics = _parse_topics(path)
        self._reader: AnyReader | None = None
        self._closed = False
        self._cache: dict[str, pd.DataFrame] = {}

    def _ensure_reader(self) -> AnyReader:
        """Open the AnyReader lazily on first use."""
        if self._reader is None:
            typestore = get_typestore(Stores.LATEST) if _needs_typestore(self._path) else None
            if typestore is not None:
                for msg_path in self._msg_paths:
                    register_msg_types(typestore, msg_path)
            self._reader = AnyReader([self._path], default_typestore=typestore)
            self._reader.open()
        return self._reader

    @property
    def topics(self) -> dict[str, str]:
        """Return ``{topic_name: message_type}`` mapping."""
        return dict(self._topics)

    def _get_field_paths(self, typename: str, prefix: str = "") -> list[str]:
        """Recursively list leaf field paths in dot notation."""
        reader = self._ensure_reader()
        paths: list[str] = []
        _consts, fields = reader.typestore.fielddefs[typename]
        for name, desc in fields:
            path = f"{prefix}.{name}" if prefix else name
            if desc[0] == Nodetype.BASE:
                paths.append(path)
            elif desc[0] == Nodetype.NAME:
                paths.extend(self._get_field_paths(desc[1], path))
            else:
                # ARRAY / SEQUENCE — keep as single leaf
                paths.append(path)
        return paths

    # ---- Parquet / pickle disk cache ----

    def _cache_meta_path(self, topic: str) -> Path:
        return self._path / _CACHE_DIR / f"{_topic_to_filename(topic)}.meta.json"

    def _cache_data_path(self, topic: str, fmt: str = "parquet") -> Path:
        return self._path / _CACHE_DIR / f"{_topic_to_filename(topic)}.{fmt}"

    def _is_cache_valid(self, topic: str) -> bool:
        """Check whether a valid disk cache exists for *topic*."""
        meta_path = self._cache_meta_path(topic)
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("baglab_cache_version") != _CACHE_VERSION:
                return False
            return meta.get("bag_files") == _compute_fingerprint(self._path)
        except Exception:
            return False

    def _read_cache(self, topic: str) -> pd.DataFrame:
        """Read a cached DataFrame from disk."""
        meta = json.loads(self._cache_meta_path(topic).read_text())
        fmt = meta.get("format", "parquet")
        data_path = self._cache_data_path(topic, fmt)
        if fmt == "parquet":
            df = pd.read_parquet(data_path)
            for col in meta.get("pickle_columns", []):
                df[col] = df[col].apply(pickle.loads)
            return df
        return pd.read_pickle(data_path)

    def _write_cache(self, topic: str, df: pd.DataFrame) -> None:
        """Write a DataFrame to the disk cache."""
        cache_dir = self._path / _CACHE_DIR
        cache_dir.mkdir(exist_ok=True)

        pickle_cols: list[str] = []
        df_to_save = df.copy()

        # Detect columns with non-scalar values and pickle-serialize them
        if len(df_to_save) > 0:
            for col in df_to_save.columns:
                sample = df_to_save[col].iloc[0]
                if hasattr(sample, "__len__") and not isinstance(sample, (str, bytes)):
                    df_to_save[col] = df_to_save[col].apply(pickle.dumps)
                    pickle_cols.append(col)

        try:
            data_path = self._cache_data_path(topic, "parquet")
            df_to_save.to_parquet(data_path)
            fmt = "parquet"
        except Exception:
            data_path = self._cache_data_path(topic, "pkl")
            df.to_pickle(data_path)
            fmt = "pkl"
            pickle_cols = []

        meta = {
            "baglab_cache_version": _CACHE_VERSION,
            "bag_files": _compute_fingerprint(self._path),
            "format": fmt,
            "pickle_columns": pickle_cols,
        }
        self._cache_meta_path(topic).write_text(json.dumps(meta, indent=2))

    def _load_topic(
        self,
        topic: str,
        keys: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load a single topic into a DataFrame."""
        if topic not in self._topics:
            available = ", ".join(sorted(self._topics.keys()))
            raise KeyError(
                f"Topic '{topic}' not found. Available topics: {available}"
            )

        # Specific field selection — needs the reader, not disk-cached
        if keys is not None:
            reader = self._ensure_reader()
            return get_dataframe(reader, topic, keys)

        # Full-topic load: try disk cache first (no reader needed)
        if self._is_cache_valid(topic):
            try:
                df = self._read_cache(topic)
                _log.debug("Loaded %s from cache", topic)
                return df
            except Exception:
                _log.debug("Cache read failed for %s, falling back", topic)

        # Cache miss — open the reader and load from bag
        reader = self._ensure_reader()
        msgtype = self._topics[topic]
        all_keys = self._get_field_paths(msgtype)
        df = get_dataframe(reader, topic, all_keys)

        try:
            self._write_cache(topic, df)
        except Exception:
            _log.debug("Failed to write cache for %s", topic, exc_info=True)

        return df

    def __getitem__(
        self,
        key: str | tuple[str, Sequence[str]],
    ) -> pd.DataFrame:
        """Access topic data with optional field selection.

        Usage::

            bag["/motor/angle"]              # all fields, cached
            bag["/motor/angle", ["actual"]]   # specific fields, not cached

        """
        if isinstance(key, tuple):
            topic, fields = key
            return self._load_topic(topic, list(fields))

        if key not in self._cache:
            self._cache[key] = self._load_topic(key)
        return self._cache[key]

    def clear_cache(self) -> None:
        """Remove the Parquet disk cache directory."""
        import shutil
        cache_dir = self._path / _CACHE_DIR
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        self._cache.clear()

    def close(self) -> None:
        """Close the underlying reader."""
        if not getattr(self, "_closed", True):
            self._closed = True
            if self._reader is not None:
                self._reader.close()
            self._cache.clear()

    def __enter__(self) -> Bag:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        n = len(self._topics)
        return f"Bag('{self._path}', {n} topics)"

    def _repr_html_(self) -> str:
        rows = "".join(
            f"<tr><td><code>{topic}</code></td><td>{msgtype}</td></tr>"
            for topic, msgtype in sorted(self._topics.items())
        )
        return (
            f"<strong>Bag</strong>: {self._path.name} "
            f"({len(self._topics)} topics)"
            f"<table><thead><tr><th>Topic</th><th>Type</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )


def load(
    path: str | Path,
    topics: dict[str, list[str]] | None = None,
    msg_paths: Sequence[str | Path] = (),
) -> Bag:
    """Load a rosbag.

    Parameters
    ----------
    path : str | Path
        Path to the rosbag directory. Shell notation (``~``, ``$VAR``) is expanded.
    topics : dict[str, list[str]] | None
        If given, eagerly load specified topics with their fields.
        If ``None``, return a lazy :class:`Bag` handle.
    msg_paths : Sequence[str | Path]
        Directories to scan for ``.msg`` files.  Required for db3 bags
        that use custom message types not included in the standard ROS2
        type definitions.  Shell notation is expanded.

    Returns
    -------
    Bag
        A bag handle. When *topics* is provided, the requested
        DataFrames are pre-cached and accessible via ``bag[topic]``.

    """
    path = Path(os.path.expandvars(os.path.expanduser(path)))
    resolved_msg_paths = [
        Path(os.path.expandvars(os.path.expanduser(p))) for p in msg_paths
    ]
    bag = Bag(path, msg_paths=resolved_msg_paths)
    if topics is not None:
        for topic, keys in topics.items():
            bag._cache[topic] = bag._load_topic(topic, keys)
    return bag
