"""Shared types and helpers for the geometry subpackage."""

from __future__ import annotations

from typing import Any, Union

import pandas as pd

from baglab.io import FieldGroup

# Type alias for inputs that carry named fields (x, y, z, w, …).
FieldInput = Union[FieldGroup, pd.DataFrame, Any]

# Attribute names probed when detecting scalar-like objects.
_XYZW = ("x", "y", "z", "w")


def _is_scalar(src: FieldInput) -> bool:
    """Return True if *src* is a scalar-like object (not a FieldGroup/DataFrame)."""
    return not isinstance(src, (FieldGroup, pd.DataFrame))


def _to_df(src: FieldInput) -> pd.DataFrame:
    """Convert FieldGroup, DataFrame, or attribute-based object to a DataFrame.

    Supports:
    - ``FieldGroup`` — uses ``.df``
    - ``pd.DataFrame`` — returned as-is
    - Objects with ``.x``, ``.y``, etc. attributes (e.g. ROS msg) — wrapped
      in a single-row DataFrame
    """
    if isinstance(src, FieldGroup):
        return src.df
    if isinstance(src, pd.DataFrame):
        return src
    # Attribute-based object (ROS msg, namedtuple, etc.)
    d = {}
    for attr in _XYZW:
        if hasattr(src, attr):
            d[attr] = [getattr(src, attr)]
    if d:
        return pd.DataFrame(d)
    raise TypeError(
        f"Cannot convert {type(src).__name__} to DataFrame. "
        "Expected FieldGroup, DataFrame, or an object with x/y/z/w attributes."
    )
