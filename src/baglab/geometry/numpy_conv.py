"""Numpy array conversions."""

from __future__ import annotations

import numpy as np

from baglab.geometry._common import FieldInput, _is_scalar, _to_df


def to_numpy_2d(fg: FieldInput) -> np.ndarray:
    """Convert FieldGroup with x, y fields to ndarray of shape (N, 2).

    Parameters
    ----------
    fg : FieldGroup | DataFrame | object
        Must contain fields/attributes ``x``, ``y``.
        Scalar input returns shape (1, 2).

    Returns
    -------
    np.ndarray
        Shape (N, 2) or (1, 2).

    """
    d = _to_df(fg)
    return np.column_stack([d["x"].values, d["y"].values])


def to_numpy_3d(fg: FieldInput) -> np.ndarray:
    """Convert FieldGroup with x, y, z fields to ndarray of shape (N, 3).

    Parameters
    ----------
    fg : FieldGroup | DataFrame | object
        Must contain fields/attributes ``x``, ``y``, ``z``.
        Scalar input returns shape (1, 3).

    Returns
    -------
    np.ndarray
        Shape (N, 3) or (1, 3).

    """
    d = _to_df(fg)
    return np.column_stack([d["x"].values, d["y"].values, d["z"].values])
