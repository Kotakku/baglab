"""Terminal UI utilities for interactive rosbag selection."""

from __future__ import annotations

import curses
import datetime
from pathlib import Path

from baglab.io.bag import _bag_data_mtime, find_bags


def select_bag(pattern: str) -> Path:
    """Show a curses TUI to interactively select a rosbag directory.

    Bags matching *pattern* are listed with their data-file modification
    timestamp, sorted oldest-first (newest at the bottom).  Use arrow keys
    or ``j``/``k`` to move, Enter to confirm, ``q`` to cancel.

    Parameters
    ----------
    pattern : str
        Glob pattern forwarded to :func:`baglab.find_bags`.

    Returns
    -------
    Path
        The selected bag directory path.

    Raises
    ------
    FileNotFoundError
        If no bags match *pattern*.
    KeyboardInterrupt
        If the user cancels with ``q`` or Ctrl-C.
    """
    bags = find_bags(pattern)
    if not bags:
        raise FileNotFoundError(f"No bags found matching: {pattern}")

    # Pre-build display lines: "  YYYY-MM-DD HH:MM:SS  bag_name"
    lines: list[tuple[str, Path]] = []
    for bag in bags:
        mtime = _bag_data_mtime(bag)
        dt = datetime.datetime.fromtimestamp(mtime)
        ts = dt.strftime("%Y-%m-%d %H:%M:%S")
        lines.append((ts, bag))

    selected = len(lines) - 1  # default to newest (last)

    def _run(stdscr: curses.window) -> Path:
        nonlocal selected
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)

        while True:
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()

            title = f"Select a rosbag ({len(lines)} found)  [↑↓/jk: move, Enter: select, q: cancel]"
            stdscr.addnstr(0, 0, title, max_x - 1, curses.A_BOLD)

            # Scrollable area
            list_start = 2
            visible = max_y - list_start
            if visible <= 0:
                visible = 1

            # Keep selected item visible
            if selected < 0:
                selected = 0
            if selected >= len(lines):
                selected = len(lines) - 1

            # Calculate scroll offset
            offset = 0
            if selected >= visible:
                offset = selected - visible + 1

            for i in range(offset, min(offset + visible, len(lines))):
                ts, bag = lines[i]
                row = list_start + (i - offset)
                label = f"  {ts}  {bag.name}"
                if i == selected:
                    label = f"> {ts}  {bag.name}"
                    attr = curses.color_pair(1) | curses.A_BOLD
                else:
                    attr = curses.A_NORMAL
                stdscr.addnstr(row, 0, label, max_x - 1, attr)

            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = max(0, selected - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = min(len(lines) - 1, selected + 1)
            elif key in (curses.KEY_HOME, ord("g")):
                selected = 0
            elif key in (curses.KEY_END, ord("G")):
                selected = len(lines) - 1
            elif key in (curses.KEY_ENTER, 10, 13):
                return lines[selected][1]
            elif key in (ord("q"), 27):  # q or Escape
                raise KeyboardInterrupt("Selection cancelled")

    return curses.wrapper(_run)
