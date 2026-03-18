"""ROS2-style dot access for DataFrame columns."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def explode_array(
    series: pd.Series,
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Expand an array-valued Series into a DataFrame with one column per element.

    Useful for ``sensor_msgs/JointState`` fields like ``position``,
    ``velocity``, and ``effort`` where each cell contains a list.

    Parameters
    ----------
    series : pd.Series
        Series whose cells are lists or arrays of equal length.
    names : sequence of str, optional
        Column names for the resulting DataFrame.  If *None*, columns are
        named ``0, 1, 2, ...``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same index as *series*.

    """
    result = pd.DataFrame(series.to_list(), index=series.index)
    if names is not None:
        if len(names) != result.shape[1]:
            raise ValueError(
                f"names has {len(names)} elements but array has {result.shape[1]}"
            )
        result.columns = list(names)
    return result


class FieldGroup:
    """Intermediate node for chained ROS2-style field access.

    Returned by :class:`MsgAccessor` attribute access.  Supports further
    chaining until a leaf (single column) is reached, at which point a
    :class:`~pandas.Series` is returned.

    Use the :attr:`df` property to obtain a :class:`~pandas.DataFrame`
    containing all columns under the current prefix, with the prefix
    stripped from column names.
    """

    def __init__(self, df: pd.DataFrame, prefix: str) -> None:
        self._df = df
        self._prefix = prefix
        prefix_dot = prefix + "."
        self._matching = [
            c for c in df.columns if c == prefix or c.startswith(prefix_dot)
        ]
        if not self._matching:
            raise AttributeError(f"No fields matching '{prefix}'")

    def __getattr__(self, name: str) -> pd.Series | FieldGroup:
        if name.startswith("_"):
            raise AttributeError(name)
        path = f"{self._prefix}.{name}"
        cols = [c for c in self._df.columns if c == path or c.startswith(path + ".")]
        if not cols:
            raise AttributeError(f"No fields matching '{path}'")
        if cols == [path]:
            return self._df[path]
        return FieldGroup(self._df, path)

    def __getitem__(self, name: str) -> pd.Series | FieldGroup:
        return self.__getattr__(name)

    @property
    def df(self) -> pd.DataFrame:
        """Return matching columns as a DataFrame with prefix stripped."""
        prefix_dot = self._prefix + "."
        cols = [c for c in self._df.columns if c.startswith(prefix_dot)]
        if not cols:
            # prefix itself is a leaf — return as single-column DataFrame
            return self._df[[self._prefix]]
        result = self._df[cols].copy()
        result.columns = [c[len(prefix_dot):] for c in cols]
        return result

    def __repr__(self) -> str:
        return f"FieldGroup('{self._prefix}', fields={self._matching})"

    def _repr_html_(self) -> str:
        return self.df._repr_html_()


@pd.api.extensions.register_dataframe_accessor("msg")
class MsgAccessor:
    """ROS2-style dot access for DataFrame columns.

    Registered as ``df.msg``.  Supports chained attribute access
    mirroring ROS2 message field hierarchy::

        df.msg.twist.linear.x   # → Series
        df.msg.twist.linear     # → FieldGroup
        df.msg.twist.linear.df  # → DataFrame with columns [x, y, z]

    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def __getattr__(self, name: str) -> pd.Series | FieldGroup:
        if name.startswith("_"):
            raise AttributeError(name)
        cols = [c for c in self._df.columns if c == name or c.startswith(name + ".")]
        if not cols:
            raise AttributeError(f"No fields matching '{name}'")
        if cols == [name]:
            return self._df[name]
        return FieldGroup(self._df, name)

    def __getitem__(self, name: str) -> pd.Series | FieldGroup:
        return self.__getattr__(name)
