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

try:
    _YamlLoader = yaml.CSafeLoader  # type: ignore[attr-defined]
except AttributeError:
    _YamlLoader = yaml.SafeLoader  # type: ignore[assignment]
from rosbags.dataframe import get_dataframe
from rosbags.highlevel import AnyReader
from rosbags.interfaces import Nodetype
from rosbags.typesys import Stores, get_typestore

from baglab.io.typesys import register_msg_types

try:
    import baglab_mcap_backend as _mcap_backend
except ImportError:
    _mcap_backend = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)
_CACHE_DIR = ".baglab_cache"
_CACHE_VERSION = 1
_mcap_patched = False


_SUPPORTED_ENCODINGS = {"ros2msg"}


class _MsgProxy:
    """Lightweight proxy providing attribute access to flat dot-keyed dicts.

    The mcap backend deserialises message sequences into dicts with flat
    dot-notation keys (e.g. ``{"pose.position.x": 1.0, ...}``).  This
    wrapper lets user code access them with the same ``pt.pose.position.x``
    syntax used by rosbags message objects.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        prefix = name + "."
        sub = {k[len(prefix):]: v for k, v in data.items() if k.startswith(prefix)}
        if sub:
            return _MsgProxy(sub)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __repr__(self) -> str:
        data = object.__getattribute__(self, "_data")
        return f"MsgProxy({data})"


def _wrap_msg_dicts(raw: dict) -> None:
    """Wrap dicts inside message-sequence columns with :class:`_MsgProxy`."""
    for key, values in raw.items():
        if not isinstance(values, list) or len(values) == 0:
            continue
        sample = values[0]
        if isinstance(sample, list) and len(sample) > 0 and isinstance(sample[0], dict):
            raw[key] = [[_MsgProxy(d) for d in inner] for inner in values]


def has_mcap_backend() -> bool:
    """Return True if the baglab-mcap acceleration package is installed."""
    return _mcap_backend is not None


def _mcap_raw_to_dataframe(
    raw: dict,
    field_paths: list[str] | None = None,
) -> pd.DataFrame:
    """Convert raw columnar dict from mcap backend to DataFrame."""
    timestamps = raw.pop("__timestamps__")
    _wrap_msg_dicts(raw)
    df = pd.DataFrame(raw, index=pd.to_datetime(timestamps, unit="ns"))
    if field_paths is not None:
        df = df[field_paths]
    return df


class _FilteredSchemas(dict):
    """Dict subclass that hides schemas with unsupported encoding.

    rosbags' McapReader.open() iterates ``self.schemas.values()`` to build
    message definitions.  Schemas with empty or unsupported encodings (e.g.
    ``ros2idl``) cause errors.  By filtering them out of ``values()``,
    the affected channels fall back to ``MessageDefinitionFormat.NONE``.
    """

    def values(self):  # type: ignore[override]
        return [v for v in super().values() if v.encoding in _SUPPORTED_ENCODINGS]


def _patch_rosbags() -> None:
    """Patch rosbags to filter unsupported schemas and speed up QoS parsing."""
    global _mcap_patched  # noqa: PLW0603
    if _mcap_patched:
        return
    _mcap_patched = True

    # --- Filter unsupported schema encodings ---
    try:
        from rosbags.rosbag2 import storage_mcap  # noqa: PLC0415

        _orig_init = storage_mcap.McapReader.__init__

        def _patched_init(self: object, *args: object, **kwargs: object) -> None:
            _orig_init(self, *args, **kwargs)
            self.schemas = _FilteredSchemas()  # type: ignore[attr-defined]

        storage_mcap.McapReader.__init__ = _patched_init  # type: ignore[assignment]
    except Exception:
        _log.debug("Failed to patch McapReader schema filter", exc_info=True)

    # --- Speed up QoS parsing (ruamel.yaml → yaml.CSafeLoader) ---
    try:
        from rosbags.rosbag2 import metadata as _meta  # noqa: PLC0415

        _orig_parse_qos = _meta.parse_qos

        def _fast_parse_qos(dcts: list | str) -> list:
            if not dcts:
                return []
            if isinstance(dcts, str):
                dcts = yaml.load(dcts, _YamlLoader)
            return _orig_parse_qos(dcts)

        # Patch both module-level attribute and already-imported references
        _meta.parse_qos = _fast_parse_qos  # type: ignore[assignment]
        from rosbags.rosbag2 import reader as _reader  # noqa: PLC0415
        _reader.parse_qos = _fast_parse_qos  # type: ignore[attr-defined]
        storage_mcap.parse_qos = _fast_parse_qos  # type: ignore[attr-defined]
    except Exception:
        _log.debug("Failed to patch parse_qos", exc_info=True)


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


def _load_metadata(path: Path) -> dict:
    """Load and return the parsed metadata.yaml content, fixing it if needed."""
    meta_path = path / "metadata.yaml"
    if not meta_path.exists():
        return {}

    with open(meta_path) as f:
        meta = yaml.load(f, _YamlLoader)

    # Fix empty storage_identifier by detecting from file extensions
    info = meta.get("rosbag2_bagfile_information", {})
    if not info.get("storage_identifier", ""):
        rel_paths = info.get("relative_file_paths", [])
        extensions = {Path(p).suffix for p in rel_paths}
        if extensions == {".db3"}:
            info["storage_identifier"] = "sqlite3"
        elif extensions == {".mcap"}:
            info["storage_identifier"] = "mcap"

        if info.get("storage_identifier", ""):
            with open(meta_path, "w") as f:
                yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    return meta


def _parse_topics(meta: dict) -> dict[str, str]:
    """Extract ``{topic_name: message_type}`` from parsed metadata."""
    topics_info = (
        meta.get("rosbag2_bagfile_information", {})
        .get("topics_with_message_count", [])
    )
    return {
        entry["topic_metadata"]["name"]: entry["topic_metadata"]["type"]
        for entry in topics_info
    }


def _needs_typestore(meta: dict) -> bool:
    """Check if the bag uses sqlite3 storage (which has no embedded type definitions)."""
    return meta.get("rosbag2_bagfile_information", {}).get("storage_identifier") == "sqlite3"


class Bag:
    """Rosbag handle with lazy per-topic DataFrame loading.

    Use :func:`baglab.load` to create instances instead of calling this
    constructor directly — ``load`` expands shell paths, preloads topics,
    and selects the backend automatically.

    Access topic data via subscript::

        bag = baglab.load("path/to/rosbag")
        df = bag["/cmd_vel"]                       # all fields, cached
        df = bag["/cmd_vel", ["twist.linear.x"]]   # field selection, not cached

    Parameters
    ----------
    path : Path
        Path to the rosbag directory.

    """

    def __init__(
        self,
        path: Path,
        msg_paths: Sequence[Path] = (),
        *,
        use_cache: bool = True,
        backend: str = "auto",
    ) -> None:
        if not path.is_dir():
            raise FileNotFoundError(f"Bag directory not found: {path}")
        self._path = path
        self._msg_paths = list(msg_paths)
        self._use_cache = use_cache
        self._backend = self._resolve_backend(backend)
        self._metadata = _load_metadata(path)
        self._topics = _parse_topics(self._metadata)
        self._reader: AnyReader | None = None
        self._mcap_reader = None  # BagReader instance (lazy)
        self._closed = False
        self._cache: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        """Resolve the backend to use."""
        if backend == "auto":
            if _mcap_backend is not None:
                return "mcap"
            return "rosbags"
        if backend == "mcap":
            if _mcap_backend is None:
                raise ImportError(
                    "baglab-mcap-backend is not installed. "
                    "Install it with: pip install baglab-mcap-backend"
                )
            return "mcap"
        if backend == "rosbags":
            return "rosbags"
        raise ValueError(f"Unknown backend: {backend!r}")

    def _ensure_reader(self) -> AnyReader:
        """Open the AnyReader lazily on first use."""
        if self._reader is None:
            typestore = get_typestore(Stores.LATEST) if _needs_typestore(self._metadata) else None
            if typestore is not None:
                for msg_path in self._msg_paths:
                    register_msg_types(typestore, msg_path)
            _patch_rosbags()
            self._reader = AnyReader([self._path], default_typestore=typestore)
            self._reader.open()
            self._warn_skipped_topics()
        return self._reader

    def _ensure_mcap_reader(self):
        """Open the MCAP BagReader lazily on first use."""
        if self._mcap_reader is None:
            self._mcap_reader = _mcap_backend.BagReader(str(self._path))
        return self._mcap_reader

    def _warn_skipped_topics(self) -> None:
        """Emit a warning listing topics skipped due to unsupported schema encoding."""
        if self._reader is None:
            return
        import warnings  # noqa: PLC0415

        from rosbags.interfaces import MessageDefinitionFormat  # noqa: PLC0415

        skipped = sorted(
            c.topic
            for c in self._reader.connections
            if c.msgdef.format == MessageDefinitionFormat.NONE
        )
        if skipped:
            topics_str = "\n  ".join(skipped)
            warnings.warn_explicit(
                f"{len(skipped)} topic(s) skipped (unsupported schema encoding):\n  {topics_str}",
                category=UserWarning,
                filename="baglab",
                lineno=0,
            )

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

    def preload(self, topic_names: list[str]) -> None:
        """Batch-preload topics into the in-memory cache.

        On the mcap backend this uses a single-pass read through the file,
        which is much faster than reading topics one-by-one.
        Other backends fall back to sequential loading.

        Preloaded topics are returned instantly by ``bag[topic]``.

        Parameters
        ----------
        topic_names : list[str]
            Topic names to preload.

        """
        if self._backend == "mcap":
            mcap_r = self._ensure_mcap_reader()
            batch_raw = mcap_r.read_topics(topic_names, {})
            for topic, raw in batch_raw.items():
                try:
                    df = _mcap_raw_to_dataframe(raw)
                    self._cache[topic] = df
                    if self._use_cache:
                        try:
                            self._write_cache(topic, df)
                        except Exception:
                            _log.debug("Failed to write cache for %s", topic, exc_info=True)
                except Exception:
                    _log.debug("Failed to build DataFrame for %s", topic, exc_info=True)
        else:
            # Fallback: sequential load
            for topic in topic_names:
                try:
                    self._cache[topic] = self._load_topic(topic)
                except Exception:
                    _log.debug("Failed to load %s", topic, exc_info=True)

    def _set_topic_attrs(self, df: pd.DataFrame, topic: str) -> pd.DataFrame:
        """Attach topic metadata to the DataFrame."""
        df.attrs["topic"] = topic
        df.attrs["msg_type"] = self._topics.get(topic, "")
        return df

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
            if self._backend == "mcap":
                mcap_r = self._ensure_mcap_reader()
                raw = mcap_r.read_topic(topic, keys)
                return self._set_topic_attrs(_mcap_raw_to_dataframe(raw, keys), topic)
            reader = self._ensure_reader()
            return self._set_topic_attrs(get_dataframe(reader, topic, keys), topic)

        # Full-topic load: try disk cache first (no reader needed)
        if self._use_cache and self._is_cache_valid(topic):
            try:
                df = self._read_cache(topic)
                _log.debug("Loaded %s from cache", topic)
                return self._set_topic_attrs(df, topic)
            except Exception:
                _log.debug("Cache read failed for %s, falling back", topic)

        # Cache miss — load from bag
        if self._backend == "mcap":
            mcap_r = self._ensure_mcap_reader()
            raw = mcap_r.read_topic(topic, [])
            df = _mcap_raw_to_dataframe(raw)
        else:
            reader = self._ensure_reader()
            msgtype = self._topics[topic]
            all_keys = self._get_field_paths(msgtype)
            df = get_dataframe(reader, topic, all_keys)

        if self._use_cache:
            try:
                self._write_cache(topic, df)
            except Exception:
                _log.debug("Failed to write cache for %s", topic, exc_info=True)

        return self._set_topic_attrs(df, topic)

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

        # Return preloaded data if available (regardless of use_cache setting)
        if key in self._cache:
            return self._cache[key]

        if not self._use_cache:
            return self._load_topic(key)

        self._cache[key] = self._load_topic(key)
        return self._cache[key]

    def close(self) -> None:
        """Close the underlying reader."""
        if not getattr(self, "_closed", True):
            self._closed = True
            if self._reader is not None:
                self._reader.close()
            if self._mcap_reader is not None:
                self._mcap_reader.close()
                self._mcap_reader = None
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
    topics: Sequence[str] | dict[str, list[str]] | None = None,
    msg_paths: Sequence[str | Path] = (),
    *,
    use_cache: bool = True,
    backend: str = "auto",
) -> Bag:
    """Load a rosbag.

    Parameters
    ----------
    path : str | Path
        Path to the rosbag directory. Shell notation (``~``, ``$VAR``) is expanded.
    topics : list[str] | dict[str, list[str]] | None
        If a list of topic names, eagerly preload all fields for those topics
        in a single pass (fast batch read on mcap backend).
        If a dict ``{topic: [fields]}``, eagerly load specified fields per topic.
        If ``None``, return a lazy :class:`Bag` handle.
    msg_paths : Sequence[str | Path]
        Directories to scan for ``.msg`` files.  Required for db3 bags
        that use custom message types not included in the standard ROS2
        type definitions.  Shell notation is expanded.
    use_cache : bool
        If ``True`` (default), use disk cache for loaded DataFrames.
        Set to ``False`` to always read from the bag file directly.
    backend : str
        Backend to use for reading: ``"auto"`` (default), ``"mcap"``,
        or ``"rosbags"``.

    Returns
    -------
    Bag
        A bag handle. When *topics* is provided, the requested
        DataFrames are pre-cached and accessible via ``bag[topic]``.

    Examples
    --------
    Lazy loading (topics loaded on first access)::

        bag = baglab.load("~/rosbags/experiment1")
        df = bag["/cmd_vel"]

    Eager batch preloading::

        bag = baglab.load("~/rosbags/experiment1", topics=["/cmd_vel", "/odom"])

    Field-selective loading::

        bag = baglab.load("path", topics={"/imu": ["angular_velocity.x"]})

    """
    path = Path(os.path.expandvars(os.path.expanduser(path)))
    resolved_msg_paths = [
        Path(os.path.expandvars(os.path.expanduser(p))) for p in msg_paths
    ]
    bag = Bag(path, msg_paths=resolved_msg_paths, use_cache=use_cache, backend=backend)
    if topics is not None:
        if isinstance(topics, dict):
            # dict: per-topic field selection (sequential)
            for topic, keys in topics.items():
                bag._cache[topic] = bag._load_topic(topic, keys)
        else:
            # list: batch preload all fields
            bag.preload(list(topics))
    return bag


def clear_cache(path: str | Path) -> None:
    """Remove the disk cache for a rosbag.

    Parameters
    ----------
    path : str | Path
        Path to the rosbag directory. Shell notation (``~``, ``$VAR``) is expanded.

    """
    import shutil  # noqa: PLC0415

    bag_path = Path(os.path.expandvars(os.path.expanduser(path)))
    cache_dir = bag_path / _CACHE_DIR
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
