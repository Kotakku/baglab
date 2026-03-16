"""Custom ROS2 message type registration for rosbags.

db3 bags do not embed message definitions, so custom types must be
registered from ``.msg`` files on disk.  This module
provides :func:`register_msg_types` to scan a directory tree for ``.msg``
files and register them with a rosbags typestore.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rosbags.typesys.msg import get_types_from_msg
from rosbags.typesys.store import Typestore

_log = logging.getLogger(__name__)


def register_msg_types(typestore: Typestore, path: str | Path) -> int:
    """Scan *path* for ``.msg`` files and register them in *typestore*.

    The directory structure must follow the ROS2 convention::

        <pkg_name>/msg/<MsgName>.msg

    Types already present in *typestore* are silently skipped.

    Parameters
    ----------
    typestore : Typestore
        The rosbags typestore to register types into.
    path : str | Path
        Root directory to search for ``.msg`` files.

    Returns
    -------
    int
        Number of newly registered types.

    """
    path = Path(path)
    all_types: dict = {}

    for msg_file in path.rglob("*.msg"):
        parts = msg_file.parts
        # Find the nearest 'msg' directory in the path
        try:
            msg_idx = len(parts) - 1 - list(reversed(parts)).index("msg")
        except ValueError:
            continue
        if msg_idx < 1:
            continue

        pkg_name = parts[msg_idx - 1]
        msg_name = msg_file.stem
        full_name = f"{pkg_name}/msg/{msg_name}"

        try:
            typs = get_types_from_msg(msg_file.read_text(), full_name)
            all_types.update(typs)
        except Exception:
            _log.debug("Failed to parse %s", msg_file, exc_info=True)

    # Skip types already registered to avoid conflicts
    new_types = {k: v for k, v in all_types.items() if k not in typestore.fielddefs}
    if new_types:
        typestore.register(new_types)
        _log.debug("Registered %d new types from %s", len(new_types), path)

    return len(new_types)
